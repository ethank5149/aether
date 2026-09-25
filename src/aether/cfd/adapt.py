"""Solution-adaptive mesh refinement for shock-dominated flows.

After a solve converges, the volume solution drives a new mesh: cell sizes
are set inversely proportional to sensor field gradients, so shocks, shear
layers, expansion fans, and reaction zones get the resolution the physics
demands rather than whatever the a-priori estimate guessed.

Multi-sensor adaptation
-----------------------

A single gradient field (say, density) captures the bow shock but misses
the thermal nonequilibrium zone, the chemical reaction zone, and the wake.
The ``"auto"`` sensor computes target sizes from every available field —
density, pressure, temperature, vibrational temperature, species mass
fractions, and Mach number — and takes the element-wise minimum across all
of them, so every resolved feature drives refinement independently and the
mesh resolves whatever the physics is doing.

Complexity control
------------------

On a coarse initial mesh, the gradient sensor requests refinement almost
everywhere because the entire domain lies within the flow structure.
Granting all of it at once would produce an impractical mesh.  The standard
Lp normalization (Alauzet & Loseille, 2010) scales the target size field
uniformly until the predicted element count matches a target complexity,
preserving the relative priorities from the gradient — the strongest
features still get the smallest cells — while keeping the mesh tractable.
Subsequent adaptation cycles converge on the right mesh iteratively.

Mesh gradation control
----------------------

A raw target-size field jumps from millimetres at the shock to metres in the
farfield.  gmsh produces poor-quality tets at the transition, and those
degrade the solver's convergence rate and accuracy.  A Lipschitz gradation
constraint (Borouchaki et al., 1997) smooths the transition: adjacent nodes
may differ by at most ``growth_rate - 1`` per unit distance, which is the
standard guarantee that element quality degrades gracefully rather than in a
single layer.

The workflow is:

1. Read the converged mesh and volume file.
2. Compute one or more sensor gradients on every node.
3. Convert each gradient to a target cell size via ``C / |∇u|``, where
   ``C = Δu / cells_per_feature`` and ``Δu`` is the sensor's range.
4. Take the element-wise minimum across all sensors.
5. Apply Lp complexity control to cap the predicted mesh growth.
6. Enforce mesh gradation so adjacent sizes grow smoothly.
7. Write a gmsh ``.pos`` file carrying the final sizes.
8. Re-mesh with :func:`~aether.cfd.geometries.build_domain`, which passes
   the ``.pos`` file into the ``Min`` composition so cells are no larger
   than the solution demands **or** the a-priori estimate, whichever is
   smaller.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from aether.aerodynamics.cfd.fields import Mesh, VolumeField

log = logging.getLogger(__name__)

__all__ = [
    "adaptation_sizes",
    "write_pos",
]


# ---------------------------------------------------------------------------
# Mesh topology helpers
# ---------------------------------------------------------------------------


def _tet_volumes(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Signed volumes of tetrahedra."""
    a = points[tets[:, 1]] - points[tets[:, 0]]
    b = points[tets[:, 2]] - points[tets[:, 0]]
    c = points[tets[:, 3]] - points[tets[:, 0]]
    return np.abs(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0


def _node_sizes(mesh: Mesh) -> np.ndarray:
    """Characteristic cell size at each node (cube-root of mean tet volume)."""
    vols = _tet_volumes(mesh.points, mesh.tetrahedra)
    h = np.cbrt(vols)
    node_h = np.zeros(mesh.points.shape[0], dtype=np.float64)
    counts = np.zeros(mesh.points.shape[0], dtype=np.int32)
    for col in range(4):
        np.add.at(node_h, mesh.tetrahedra[:, col], h)
        np.add.at(counts, mesh.tetrahedra[:, col], 1)
    counts = np.maximum(counts, 1)
    return node_h / counts


def _unique_edges(tets: np.ndarray) -> np.ndarray:
    """Unique undirected edges from a tet connectivity, shape ``(E, 2)``."""
    local = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    parts = []
    for a, b in local:
        e = np.column_stack([tets[:, a], tets[:, b]])
        e = np.sort(e, axis=1)
        parts.append(e)
    all_e = np.vstack(parts)
    return np.unique(all_e, axis=0)


# ---------------------------------------------------------------------------
# Green-Gauss gradient
# ---------------------------------------------------------------------------


def _gradient_magnitude(mesh: Mesh, scalar: np.ndarray) -> np.ndarray:
    """Per-node gradient magnitude via Green-Gauss over surrounding tets.

    The gradient within each tetrahedron is exact for a linear field (the
    shape-function gradient).  Node values are the volume-weighted average
    of the surrounding tets, the standard finite-volume dual-cell approach.
    """
    tets = mesh.tetrahedra
    pts = mesh.points
    vols = _tet_volumes(pts, tets)
    vols = np.maximum(vols, 1e-30)

    tet_grad = np.zeros((tets.shape[0], 3), dtype=np.float64)
    for i in range(4):
        j, k, l = (i + 1) % 4, (i + 2) % 4, (i + 3) % 4  # noqa: E741
        face_normal = np.cross(
            pts[tets[:, k]] - pts[tets[:, j]],
            pts[tets[:, l]] - pts[tets[:, j]],
        )
        tet_grad += scalar[tets[:, i], np.newaxis] * face_normal

    tet_grad /= (6.0 * vols[:, np.newaxis])
    tet_mag = np.linalg.norm(tet_grad, axis=1)

    node_mag = np.zeros(pts.shape[0], dtype=np.float64)
    counts = np.zeros(pts.shape[0], dtype=np.int32)
    for col in range(4):
        np.add.at(node_mag, tets[:, col], tet_mag)
        np.add.at(counts, tets[:, col], 1)
    counts = np.maximum(counts, 1)
    return node_mag / counts


# ---------------------------------------------------------------------------
# Mesh gradation enforcement
# ---------------------------------------------------------------------------


def _enforce_gradation(
    h: np.ndarray,
    points: np.ndarray,
    edges: np.ndarray,
    growth_rate: float = 1.3,
    max_sweeps: int = 40,
) -> np.ndarray:
    r"""Enforce a Lipschitz gradation constraint on the size field.

    For every mesh edge :math:`(i,j)`:

    .. math::
        |h_i - h_j| \leq (\beta - 1) \cdot \|x_i - x_j\|

    where :math:`\beta` is the ``growth_rate``.  This ensures that adjacent
    cells differ by at most a factor of :math:`\beta`, producing smooth
    transitions from refined to coarse regions and preventing the slivered
    tets that gmsh would otherwise generate at an abrupt size jump.

    Uses the Jacobi-style algorithm of Borouchaki, George, Hecht, Laug, and
    Saltel (1997).  Each sweep propagates the constraint one edge further;
    convergence typically requires O(mesh_diameter) sweeps.
    """
    h = h.copy()
    alpha = growth_rate - 1.0

    edge_len = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    lim = alpha * edge_len

    max_change = 0.0
    for sweep in range(max_sweeps):
        h_i = h[edges[:, 0]]
        h_j = h[edges[:, 1]]

        cap_j = h_i + lim
        cap_i = h_j + lim

        h_old = h.copy()
        np.minimum.at(h, edges[:, 1], cap_j)
        np.minimum.at(h, edges[:, 0], cap_i)

        max_change = float(np.abs(h - h_old).max())
        if max_change < 1e-6 * h.max():
            log.info(
                "  gradation converged in %d sweeps (max Δ=%.2e)",
                sweep + 1, max_change,
            )
            break
    else:
        log.info(
            "  gradation: %d sweeps (max Δ=%.2e, α=%.2f)",
            max_sweeps, max_change, alpha,
        )

    return h


# ---------------------------------------------------------------------------
# Per-sensor target sizing
# ---------------------------------------------------------------------------


def _sensor_target(
    mesh: Mesh,
    scalar: np.ndarray,
    current_h: np.ndarray,
    cells_per_feature: float,
    *,
    log_transform: bool = False,
    name: str = "unnamed",
) -> np.ndarray:
    r"""Raw target size from one sensor field (no h_min clamping).

    For a sensor field *u*, the target at each node is:

    .. math::
        h_{\text{target}} = \min\!\bigl(h_{\text{current}},\;
        C \,/\, |\nabla u|\bigr)

    where :math:`C = \Delta u \,/\, N` and :math:`\Delta u` is the field's
    total range.  This puts *N* cells across whichever feature has the
    steepest gradient — typically the bow shock for density, or the
    vibrational relaxation zone for :math:`T_{ve}`.

    The result is NOT clamped to h_min here; that happens after complexity
    control in the caller, so the raw priorities are preserved for Lp
    normalization.
    """
    if log_transform:
        s = np.log(np.clip(scalar, 1e-30, None))
    else:
        s = scalar.copy()

    jump = float(s.max() - s.min())
    if jump < 1e-30:
        log.info("  sensor %-14s: no variation, skipped", name)
        return current_h.copy()

    grad = _gradient_magnitude(mesh, s)
    C = jump / cells_per_feature
    grad_safe = np.maximum(grad, 1e-30)
    target = np.minimum(current_h, C / grad_safe)

    n_ref = int((target < 0.8 * current_h).sum())
    n_strong = int((target < 0.5 * current_h).sum())
    log.info(
        "  sensor %-14s: C=%.3e, |∇|=[%.2e, %.2e], "
        "%d refined (%d strongly)",
        name, C, float(grad.min()), float(grad.max()), n_ref, n_strong,
    )
    return target


# ---------------------------------------------------------------------------
# Complexity control (Lp normalization)
# ---------------------------------------------------------------------------


def _apply_complexity_control(
    target: np.ndarray,
    current_h: np.ndarray,
    n_tets: int,
    max_growth: float,
    h_min: float,
) -> np.ndarray:
    r"""Scale the target size field to cap predicted element growth.

    In 3D, refining a node from :math:`h` to :math:`h'` increases the local
    element count by :math:`(h/h')^3`.  Summing over all nodes predicts the
    total mesh growth.  If that exceeds ``max_growth × n_tets``, the target
    sizes are uniformly scaled upward by the cube-root ratio, which is the
    :math:`L^\infty` normalization of Alauzet & Loseille (2010).

    The uniform scaling preserves relative priorities: nodes at the strongest
    gradients still get the smallest cells; only the absolute sizes shift.
    Subsequent adaptation cycles converge on the ideal mesh iteratively.
    """
    target_safe = np.maximum(target, 1e-12)
    growth_per_node = (current_h / target_safe) ** 3
    n_nodes = len(target)
    tets_per_node = max(n_tets / n_nodes, 1.0)
    predicted = float(growth_per_node.sum()) * tets_per_node
    target_complexity = max_growth * n_tets

    if predicted <= target_complexity:
        log.info(
            "  complexity: %.1f× growth (within %.1f× budget), no scaling",
            predicted / n_tets, max_growth,
        )
        return np.maximum(target, h_min)

    scaling = (predicted / target_complexity) ** (1.0 / 3.0)
    scaled = target * scaling
    scaled = np.minimum(scaled, current_h)
    scaled = np.maximum(scaled, h_min)

    actual_growth = float(np.sum((current_h / np.maximum(scaled, 1e-12)) ** 3) * tets_per_node)
    log.info(
        "  complexity: raw %.1f× → scaled by %.2f → %.1f× (target %.1f×)",
        predicted / n_tets, scaling, actual_growth / n_tets, max_growth,
    )
    return scaled


# ---------------------------------------------------------------------------
# Sensor collection
# ---------------------------------------------------------------------------


def _vehicle_length(mesh: Mesh) -> float:
    """Body length from the wall markers, or 10 % of domain as a fallback."""
    walls = [
        tag for tag in mesh.markers
        if "vehicle" in tag.lower() or "wall" in tag.lower()
    ]
    if walls:
        return float(np.ptp(mesh.marker_points(*walls)[:, 0]))
    return float(np.ptp(mesh.points[:, 0])) * 0.1


def _collect_sensors(
    field: VolumeField, sensor: str,
) -> list[tuple[np.ndarray, bool, str]]:
    """Build the list of ``(field, log_transform, name)`` tuples."""
    sensors: list[tuple[np.ndarray, bool, str]] = []

    if sensor == "density":
        sensors.append((field.density, True, "log(ρ)"))
        return sensors

    if sensor == "mach":
        if not field.has_primitives:
            raise ValueError("Mach adaptation requires primitive variables")
        sensors.append((field.mach(), False, "Mach"))
        return sensors

    # --- "auto": every field the solution provides ---

    sensors.append((field.density, True, "log(ρ)"))

    if field.has_primitives:
        sensors.append((field.pressure(), True, "log(p)"))
        sensors.append((field.temperature(), False, "T_tr"))
        sensors.append((field.mach(), False, "Mach"))

    if field.two_temperature:
        t_ve = field.vibrational_temperature()
        if t_ve is not None:
            sensors.append((t_ve, False, "T_ve"))

    rho_mix = np.maximum(field.density, 1e-30)
    for sp_name in field.species:
        rho_s = field.fields[sp_name]
        Y_s = rho_s / rho_mix
        if float(np.ptp(Y_s)) > 0.005:
            label = sp_name.replace("Density_", "Y_")
            sensors.append((Y_s, False, label))

    return sensors


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def adaptation_sizes(
    mesh: Mesh,
    field: VolumeField,
    sensor: str = "auto",
    *,
    cells_per_feature: float = 10.0,
    h_min_fraction: float = 0.001,
    growth_rate: float = 1.3,
    max_growth: float = 5.0,
    gradation_sweeps: int = 40,
) -> np.ndarray:
    r"""Target cell sizes from a converged solution, ready for gmsh.

    This is the main entry point for solution-adaptive mesh refinement.
    It implements multi-sensor metric-based adaptation with Lp complexity
    control and Lipschitz gradation, the standard pipeline used in
    production AMR codes (FUN3D, INRIA's Feflo.a, mmg).

    1. Compute :math:`|\nabla u|` for each sensor field *u*.
    2. Convert to a target size: :math:`h = C / |\nabla u|`, where
       :math:`C = \Delta u / N` puts *N* cells across the steepest feature.
    3. Take the element-wise minimum across all sensors.
    4. Apply Lp complexity control to cap mesh growth per cycle.
    5. Enforce mesh gradation so the size field is Lipschitz-continuous.

    Parameters
    ----------
    mesh, field
        The mesh and its volume solution.
    sensor
        ``"auto"`` (default) uses every available field — density, pressure,
        temperature, T_ve, Mach, and species mass fractions.  ``"density"``
        and ``"mach"`` use a single sensor.
    cells_per_feature
        How many cells to place across the steepest gradient feature.
        10 is a good default for second-order schemes.
    h_min_fraction
        Absolute floor on cell size as a fraction of the **vehicle** body
        length.  ``0.001`` on a 3.3 m capsule gives h_min = 3.3 mm, enough
        to resolve the stagnation-point shock structure.
    growth_rate
        Maximum ratio between adjacent cell sizes after gradation.  1.3
        (30 % growth per cell) is the standard for compressible flows.
    max_growth
        Maximum element-count growth factor per adaptation cycle.
        ``5.0`` on a 500K-tet mesh allows up to ~2.5M tets per cycle.
        Subsequent cycles further refine where needed; three cycles at 5×
        converge on the right mesh without producing an impractical
        intermediate.
    gradation_sweeps
        Jacobi iterations for the gradation limiter.

    Returns
    -------
    sizes : ndarray, shape (n_nodes,)
        Target cell size at each mesh node.  The caller writes this into
        a ``.pos`` file and passes it to ``build_domain(background_sizes=...)``.
    """
    current_h = _node_sizes(mesh)
    veh_len = _vehicle_length(mesh)
    h_min = h_min_fraction * veh_len
    n_tets = mesh.tetrahedra.shape[0]

    sensors = _collect_sensors(field, sensor)

    log.info(
        "adaptation: %d sensor(s), vehicle=%.2f m, h_min=%.4f m, "
        "growth_rate=%.2f, cells_per_feature=%.0f, max_growth=%.1f×",
        len(sensors), veh_len, h_min, growth_rate, cells_per_feature,
        max_growth,
    )

    # Element-wise minimum across all sensors (no h_min yet).
    target = current_h.copy()
    for scalar, log_xform, name in sensors:
        s_target = _sensor_target(
            mesh, scalar, current_h, cells_per_feature,
            log_transform=log_xform, name=name,
        )
        target = np.minimum(target, s_target)

    # Complexity control: scale target sizes if predicted mesh is too large.
    target = _apply_complexity_control(
        target, current_h, n_tets, max_growth, h_min,
    )

    # Enforce Lipschitz gradation.
    edges = _unique_edges(mesh.tetrahedra)
    target = _enforce_gradation(
        target, mesh.points, edges,
        growth_rate=growth_rate,
        max_sweeps=gradation_sweeps,
    )

    n_refined = int((target < 0.8 * current_h).sum())
    n_strong = int((target < 0.5 * current_h).sum())
    log.info(
        "adaptation: %d nodes refined (%d strongly), "
        "target h ∈ [%.4f, %.4f] m",
        n_refined, n_strong, float(target.min()), float(target.max()),
    )
    return target


# ---------------------------------------------------------------------------
# gmsh .pos writer
# ---------------------------------------------------------------------------


def write_pos(
    points: np.ndarray,
    sizes: np.ndarray,
    path: Path | str,
    name: str = "target_size",
) -> Path:
    """Write a gmsh Parsed Post-Processing ``.pos`` file.

    Each node becomes one ``SP(x, y, z){h};`` line.  gmsh reads this as a
    ``PostView`` field and the mesher interpolates it as a background size
    at arbitrary query points, so the ``.pos`` file need not match the new
    mesh's topology.

    For large meshes (500K+ nodes), this file can reach 50-80 MB of text.
    Writing it in chunks keeps memory bounded.
    """
    path = Path(path)
    chunk = 50_000
    with path.open("w") as fh:
        fh.write(f'View "{name}" {{\n')
        for start in range(0, len(points), chunk):
            end = min(start + chunk, len(points))
            lines = []
            for i in range(start, end):
                x, y, z = points[i]
                lines.append(f"SP({x:.6g},{y:.6g},{z:.6g}){{{sizes[i]:.6g}}};")
            fh.write("\n".join(lines))
            fh.write("\n")
        fh.write("};\n")
    log.info("wrote %s (%.1f MB, %d nodes)", path.name, path.stat().st_size / 1e6, len(points))
    return path
