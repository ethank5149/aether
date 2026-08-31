"""Reading a finished SU2 case, and cutting it.

Everything here has an answer that can be worked out independently of the
code: a prism's volume, the area a plane cuts from a unit cube, the gradient
of a linear field. That is the point -- a plane cut is geometry, so it is
checked against geometry rather than against a stored picture, and a
regression baseline would only record that the cut has not changed, not that
it was ever right.

The cube is subdivided by the Kuhn (Freudenthal) construction: one tetrahedron
per permutation of the axes, six of them, each of volume 1/6. It is used
because its cross-sections are known exactly.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from aether.aerodynamics.cfd.fields import (
    Mesh,
    node_gradient,
    plane_cut,
    read_case,
    read_restart,
    read_su2_mesh,
)

# Cube corners indexed by x + 2y + 4z, so corner 7 is (1, 1, 1).
_CORNERS = np.array(
    [[x, y, z] for z in (0.0, 1.0) for y in (0.0, 1.0) for x in (0.0, 1.0)], dtype=np.float64
)[[0, 1, 2, 3, 4, 5, 6, 7]]
_KUHN = np.array(
    [[0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7]],
    dtype=np.int64,
)


def _cube() -> Mesh:
    """The unit cube as six tetrahedra."""
    return Mesh(points=_CORNERS.copy(), tetrahedra=_KUHN.copy(), markers={})


def _tet_volumes(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    corners = points[cells]
    edges = corners[:, 1:, :] - corners[:, :1, :]
    return np.abs(np.linalg.det(edges)) / 6.0


def _cut_area(cut, points: np.ndarray) -> float:
    planar = cut.coordinates(points)[cut.triangles]
    spans = planar[:, 1:, :] - planar[:, :1, :]
    cross = spans[:, 0, 0] * spans[:, 1, 1] - spans[:, 0, 1] * spans[:, 1, 0]
    return float(0.5 * np.abs(cross).sum())


# --------------------------------------------------------------- geometry


def test_kuhn_subdivision_fills_the_cube() -> None:
    """The fixture itself must be right before anything is checked against it."""
    assert _tet_volumes(_CORNERS, _KUHN).sum() == pytest.approx(1.0, rel=1e-14)


def test_prisms_are_split_without_losing_or_double_counting_volume() -> None:
    """Three tetrahedra must fill a prism exactly -- no sliver, no overlap.

    A decomposition that overlaps still looks plausible in a picture; it shows
    up here as a volume that exceeds the prism's own.
    """
    from aether.aerodynamics.cfd.fields import _prisms_to_tetrahedra

    rng = np.random.default_rng(20260830)
    for _ in range(20):
        triangle = rng.normal(size=(3, 3))
        triangle[:, 2] = 0.0
        offset = np.array([0.0, 0.0, 1.0]) + 0.3 * rng.normal(size=3)
        points = np.vstack([triangle, triangle + offset])
        base = 0.5 * abs(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])[2])
        expected = base * abs(offset[2])

        cells = _prisms_to_tetrahedra(np.arange(6, dtype=np.int64)[None, :])
        assert cells.shape == (3, 4)
        assert _tet_volumes(points, cells).sum() == pytest.approx(expected, rel=1e-12)


def test_axis_aligned_cut_has_the_area_of_the_face_it_is_parallel_to() -> None:
    cut = plane_cut(_cube(), (0.5, 0.5, 0.5), (1.0, 0.0, 0.0))
    assert _cut_area(cut, _CORNERS) == pytest.approx(1.0, rel=1e-12)


def test_diagonal_cut_reproduces_the_regular_hexagon() -> None:
    """A cube cut through its centre normal to a body diagonal is a hexagon.

    Side :math:`\\sqrt{1/2}`, so area :math:`3\\sqrt{3}/4`. This is the check
    that exercises every entry of the marching-tetrahedra table -- an
    axis-aligned cut only ever produces the easy cases.
    """
    cut = plane_cut(_cube(), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
    assert _cut_area(cut, _CORNERS) == pytest.approx(3.0 * np.sqrt(3.0) / 4.0, rel=1e-12)


def test_a_cut_that_misses_the_mesh_is_empty_rather_than_an_error() -> None:
    cut = plane_cut(_cube(), (5.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert cut.size == 0
    assert cut.triangles.shape == (0, 3)


def test_a_plane_through_the_nodes_leaves_no_degenerate_triangles() -> None:
    """Cutting exactly along a face is the case that produces zero-area cells.

    They are filtered rather than perturbed, so what must hold is that every
    surviving triangle has area and nothing is NaN -- not that the plane
    quietly moved somewhere more convenient.
    """
    cut = plane_cut(_cube(), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    planar = cut.coordinates(_CORNERS)
    assert np.isfinite(planar).all()
    corners = planar[cut.triangles]
    spans = corners[:, 1:, :] - corners[:, :1, :]
    area = 0.5 * np.abs(spans[:, 0, 0] * spans[:, 1, 1] - spans[:, 0, 1] * spans[:, 1, 0])
    assert (area > 0.0).all()


def test_the_meridian_basis_puts_x_right_and_z_up() -> None:
    """The convention the flow figures depend on for their axis labels."""
    cut = plane_cut(_cube(), (0.5, 0.5, 0.5), (0.0, 1.0, 0.0))
    assert cut.basis[0] == pytest.approx([1.0, 0.0, 0.0])
    assert cut.basis[1] == pytest.approx([0.0, 0.0, 1.0])


# ------------------------------------------------------------ exactness


def test_cutting_a_linear_field_is_exact() -> None:
    """Linear interpolation on a simplex reproduces a linear field exactly.

    This is what makes the cut a restriction of the solution rather than a
    resampling of it, and it is the property the whole module is built on.
    """
    slope = np.array([2.0, -3.0, 0.5])
    values = _CORNERS @ slope + 7.0
    cut = plane_cut(_cube(), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
    positions = cut.interpolate(_CORNERS)
    assert cut.interpolate(values) == pytest.approx(positions @ slope + 7.0, rel=1e-12)


def test_cut_vertices_lie_on_the_cutting_plane() -> None:
    cut = plane_cut(_cube(), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
    offset = (cut.interpolate(_CORNERS) - cut.origin) @ cut.normal
    assert np.abs(offset).max() < 1e-12


def test_vector_fields_cut_with_the_same_topology_as_scalars() -> None:
    cut = plane_cut(_cube(), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
    vectors = cut.interpolate(_CORNERS)
    assert vectors.shape == (cut.size, 3)
    for axis in range(3):
        assert cut.interpolate(_CORNERS[:, axis]) == pytest.approx(vectors[:, axis])


def test_the_nodal_gradient_of_a_linear_field_is_its_slope() -> None:
    """Exact by construction on a simplex; anything else is a bug, not error."""
    slope = np.array([1.5, -0.25, 4.0])
    gradient = node_gradient(_cube(), _CORNERS @ slope - 2.0)
    assert gradient == pytest.approx(np.broadcast_to(slope, gradient.shape), rel=1e-12)


def test_the_nodal_gradient_rejects_a_field_of_the_wrong_length() -> None:
    with pytest.raises(ValueError, match="values for"):
        node_gradient(_cube(), np.zeros(3))


# ------------------------------------------------------------- file formats


def _write_restart(path: Path, names: list[str], values: np.ndarray) -> None:
    """Write a file in SU2's binary restart layout."""
    header = struct.pack("<5i", 535532, len(names), values.shape[0], 0, 0)
    padded = b"".join(name.encode().ljust(33, b"\x00") for name in names)
    path.write_bytes(header + padded + values.astype(np.float64).tobytes())


def test_a_restart_round_trips(tmp_path: Path) -> None:
    values = np.arange(24, dtype=np.float64).reshape(4, 6)
    names = ["x", "y", "z", "Density", "Momentum_x", "Energy"]
    _write_restart(tmp_path / "r.dat", names, values)
    field = read_restart(tmp_path / "r.dat")
    assert field.size == 4
    assert field.points == pytest.approx(values[:, :3])
    assert field.fields["Density"] == pytest.approx(values[:, 3])
    assert set(field.fields) == {"Density", "Momentum_x", "Energy"}


def test_a_file_without_the_magic_number_is_refused(tmp_path: Path) -> None:
    (tmp_path / "ascii.dat").write_bytes(b"PointID, x, y, z\n" + b"0" * 64)
    with pytest.raises(ValueError, match="magic number"):
        read_restart(tmp_path / "ascii.dat")


def test_a_truncated_restart_is_refused(tmp_path: Path) -> None:
    values = np.zeros((4, 6))
    _write_restart(tmp_path / "r.dat", ["x", "y", "z", "a", "b", "c"], values)
    raw = (tmp_path / "r.dat").read_bytes()
    (tmp_path / "short.dat").write_bytes(raw[:-64])
    with pytest.raises(ValueError, match="bytes of values"):
        read_restart(tmp_path / "short.dat")


def _write_mesh(path: Path, elements: list[str], points: np.ndarray, markers: dict) -> None:
    lines = ["NDIME= 3", f"NELEM= {len(elements)}", *elements, f"NPOIN= {len(points)}"]
    lines += [f"{x} {y} {z} {i}" for i, (x, y, z) in enumerate(points)]
    lines.append(f"NMARK= {len(markers)}")
    for tag, faces in markers.items():
        lines += [f"MARKER_TAG= {tag}", f"MARKER_ELEMS= {len(faces)}"]
        lines += [f"5 {a} {b} {c}" for a, b, c in faces]
    path.write_text("\n".join(lines) + "\n")


def test_a_tetrahedral_mesh_reads_with_and_without_trailing_indices(tmp_path: Path) -> None:
    """The trailing element index is optional and unannounced; both must work."""
    indexed = [f"10 {' '.join(map(str, cell))} {i}" for i, cell in enumerate(_KUHN)]
    bare = [f"10 {' '.join(map(str, cell))}" for cell in _KUHN]
    for name, elements in (("indexed", indexed), ("bare", bare)):
        _write_mesh(tmp_path / f"{name}.su2", elements, _CORNERS, {"wall": [(0, 1, 3)]})
        mesh = read_su2_mesh(tmp_path / f"{name}.su2")
        assert mesh.tetrahedra == pytest.approx(_KUHN)
        assert mesh.points == pytest.approx(_CORNERS)
        assert mesh.markers["wall"].tolist() == [[0, 1, 3]]


def test_a_mixed_tetrahedron_and_prism_mesh_reads(tmp_path: Path) -> None:
    """The boundary-layer meshes are prisms next to tetrahedra in one block."""
    points = np.vstack([_CORNERS, _CORNERS + np.array([0.0, 0.0, 2.0])])
    elements = ["13 8 9 10 12 13 14 0", "10 0 1 3 7 1"]
    _write_mesh(tmp_path / "m.su2", elements, points, {"wall": [(0, 1, 3)]})
    mesh = read_su2_mesh(tmp_path / "m.su2")
    # One tetrahedron of its own, plus the three the prism becomes.
    assert mesh.tetrahedra.shape == (4, 4)


def test_a_repeated_marker_tag_is_merged_not_replaced(tmp_path: Path) -> None:
    """The extruded meshes write ``vehicle_base`` twice, the first time empty.

    Taking the last entry would silently discard the real faces, and a base
    that vanishes from a figure looks like geometry, not like a parser bug.
    """
    elements = [f"10 {' '.join(map(str, cell))} {i}" for i, cell in enumerate(_KUHN)]
    _write_mesh(
        tmp_path / "m.su2",
        elements,
        _CORNERS,
        {"base": [], "base ": [(0, 1, 3), (0, 3, 2)]},
    )
    text = (tmp_path / "m.su2").read_text().replace("base ", "base")
    (tmp_path / "m.su2").write_text(text)
    mesh = read_su2_mesh(tmp_path / "m.su2")
    assert mesh.markers["base"].shape == (2, 3)


def test_an_unsupported_element_type_is_refused_rather_than_skipped(tmp_path: Path) -> None:
    _write_mesh(tmp_path / "m.su2", ["12 0 1 2 3 4 5 6 7 0"], _CORNERS, {})
    with pytest.raises(ValueError, match="neither a tetrahedron"):
        read_su2_mesh(tmp_path / "m.su2")


def test_a_mesh_and_restart_that_disagree_on_node_count_are_refused(tmp_path: Path) -> None:
    """The one cross-check the formats permit, and the failure it catches."""
    elements = [f"10 {' '.join(map(str, cell))} {i}" for i, cell in enumerate(_KUHN)]
    _write_mesh(tmp_path / "m.su2", elements, _CORNERS, {})
    _write_restart(tmp_path / "r.dat", ["x", "y", "z", "Density"], np.zeros((5, 4)))
    with pytest.raises(ValueError, match="not from the same case"):
        read_case(tmp_path / "m.su2", tmp_path / "r.dat")


# --------------------------------------------------------------- primitives


def _field(density, velocity, pressure, gamma):
    from aether.aerodynamics.cfd.fields import VolumeField

    momentum = density[:, None] * velocity
    energy = pressure / (gamma - 1.0) + 0.5 * density * (velocity**2).sum(axis=1)
    return VolumeField(
        points=np.zeros((density.size, 3)),
        fields={
            "Density": density,
            "Momentum_x": momentum[:, 0],
            "Momentum_y": momentum[:, 1],
            "Momentum_z": momentum[:, 2],
            "Energy": energy,
        },
    )


@pytest.mark.parametrize("mach", [3.0, 8.0, 15.0, 20.0, 27.0])
def test_primitives_recover_the_state_they_were_built_from(mach: float) -> None:
    """Density, velocity and pressure in; the same three back out.

    Run across the whole envelope rather than at one speed. The algebra is
    Mach-independent, which is the point of checking it at Mach 27 as well as
    Mach 3 -- what fails up there is the *physics* of a constant gamma, not
    this arithmetic, and keeping the two separate is what stops a
    perfect-gas-only test from being read as a perfect-gas-everywhere licence.
    """
    from aether.aerodynamics.cfd.fields import CaloricallyPerfect, gas_constant_air

    gas = CaloricallyPerfect(gamma=1.4)
    temperature = 226.5090836
    speed = mach * np.sqrt(gas.gamma * gas.gas_constant * temperature)
    density = np.array([0.05, 1.225])
    velocity = np.array([[speed, 0.0, 0.03 * speed], [10.0, 0.0, 0.0]])
    pressure = np.array([1197.0, 101325.0])

    field = _field(density, velocity, pressure, gas.gamma)
    assert field.pressure(gas) == pytest.approx(pressure)
    assert field.velocity == pytest.approx(velocity)
    expected_temperature = pressure / (density * gas_constant_air)
    assert field.temperature(gas) == pytest.approx(expected_temperature)
    sound = np.sqrt(gas.gamma * gas_constant_air * expected_temperature)
    assert field.mach(gas) == pytest.approx(np.linalg.norm(velocity, axis=1) / sound)


def test_the_gas_model_is_used_rather_than_assumed() -> None:
    """A case on a different gas must not be read as though it were air.

    ``GAS_CONSTANT`` and ``GAMMA_VALUE`` are both case settings in SU2, and
    for a while nothing here read the first of them.
    """
    from aether.aerodynamics.cfd.fields import CaloricallyPerfect

    density = np.array([0.05])
    velocity = np.array([[2400.0, 0.0, 0.0]])
    pressure = np.array([1197.0])
    other = CaloricallyPerfect(gamma=1.29, gas_constant=188.9)  # CO2, roughly

    field = _field(density, velocity, pressure, other.gamma)
    assert field.pressure(other) == pytest.approx(pressure)
    assert field.temperature(other) == pytest.approx(pressure / (density * other.gas_constant))
    # Reading it as air is wrong on both counts, not one: gamma sets the
    # pressure recovered from the energy, and R turns that into a temperature.
    from aether.aerodynamics.cfd.fields import PERFECT_AIR

    misread = field.temperature(PERFECT_AIR) / field.temperature(other)
    expected = ((PERFECT_AIR.gamma - 1.0) / (other.gamma - 1.0)) * (
        other.gas_constant / PERFECT_AIR.gas_constant
    )
    assert misread == pytest.approx(expected)
    assert misread != pytest.approx(1.0, abs=0.05)


def test_an_unphysical_gas_model_is_refused() -> None:
    from aether.aerodynamics.cfd.fields import CaloricallyPerfect

    with pytest.raises(ValueError, match="gamma must exceed 1"):
        CaloricallyPerfect(gamma=1.0)
    with pytest.raises(ValueError, match="gas constant must be positive"):
        CaloricallyPerfect(gas_constant=0.0)


def test_a_nonequilibrium_restart_is_refused_by_name_not_by_key_error() -> None:
    """The regime the suite exists for, and the one perfect gas cannot describe.

    A NEMO restart carries per-species densities and a second, vibrational
    energy. Asking it for a perfect-gas temperature has no answer, and the
    failure should say that rather than surfacing as ``KeyError: 'Density'``
    from somewhere three layers down.
    """
    from aether.aerodynamics.cfd.fields import VolumeField

    field = VolumeField(
        points=np.zeros((2, 3)),
        fields={
            **{f"Density_{i}": np.full(2, 0.01) for i in range(5)},
            "Momentum_x": np.zeros(2),
            "Momentum_y": np.zeros(2),
            "Momentum_z": np.zeros(2),
            "Energy": np.ones(2),
            "Energy_ve": np.ones(2),
        },
    )
    assert not field.is_perfect_gas
    with pytest.raises(ValueError, match="NEMO"):
        _ = field.density
    with pytest.raises(ValueError, match="nonequilibrium"):
        _ = field.temperature()


def test_a_single_species_restart_is_recognised_as_perfect_gas() -> None:
    field = _field(np.array([0.05]), np.array([[2400.0, 0.0, 0.0]]), np.array([1197.0]), 1.4)
    assert field.is_perfect_gas


# ------------------------------------------------ the solver's own primitives


def _tecplot(path: Path, names: list[str], rows: np.ndarray, packing: str = "POINT") -> None:
    """Write a file in SU2's Tecplot-ASCII volume layout."""
    variables = ",".join(f'"{name}"' for name in names)
    path.write_text(
        'TITLE = "Visualization of the solution"\n'
        f"VARIABLES = {variables}\n"
        f"ZONE NODES= {rows.shape[0]}, ELEMENTS= 1, DATAPACKING={packing}, ZONETYPE=FEBRICK\n"
        + "".join("\t".join(f"{v:.6e}" for v in row) + "\t\n" for row in rows)
    )


_VOLUME_NAMES = ["x", "y", "z", "Density", "Pressure", "Temperature", "Mach"]


def test_a_volume_file_round_trips(tmp_path: Path) -> None:
    from aether.aerodynamics.cfd.fields import read_volume

    rows = np.arange(21, dtype=np.float64).reshape(3, 7)
    _tecplot(tmp_path / "v.dat", _VOLUME_NAMES, rows)
    field = read_volume(tmp_path / "v.dat")
    assert field.size == 3
    assert field.points == pytest.approx(rows[:, :3])
    assert field.has_primitives
    assert field.pressure() == pytest.approx(rows[:, 4])
    assert field.temperature() == pytest.approx(rows[:, 5])
    assert field.mach() == pytest.approx(rows[:, 6])


def test_the_solver_primitives_win_over_the_gas_model(tmp_path: Path) -> None:
    """The whole reason the volume path exists.

    The stored values are deliberately inconsistent with what a perfect gas
    would give, so that a reader which quietly re-derived them would produce a
    different number and fail here. Above Mach 10 that difference is the
    difference between right and wrong.
    """
    from aether.aerodynamics.cfd.fields import CaloricallyPerfect, read_volume

    names = [*_VOLUME_NAMES, "Momentum_x", "Momentum_y", "Momentum_z", "Energy"]
    rows = np.array([[0.0, 0.0, 0.0, 0.05, 4321.0, 1234.0, 17.5, 120.0, 0.0, 0.0, 9.9e5]])
    _tecplot(tmp_path / "v.dat", names, rows)
    field = read_volume(tmp_path / "v.dat")

    assert field.pressure() == pytest.approx(4321.0)
    assert field.temperature() == pytest.approx(1234.0)
    assert field.mach() == pytest.approx(17.5)
    # And they do not move when a different gas model is handed in, because
    # the gas model is not consulted at all when the solver said so.
    exotic = CaloricallyPerfect(gamma=1.15, gas_constant=190.0)
    assert field.pressure(exotic) == pytest.approx(4321.0)
    assert field.temperature(exotic) == pytest.approx(1234.0)
    assert field.mach(exotic) == pytest.approx(17.5)


def test_a_restart_without_primitives_still_derives_them(tmp_path: Path) -> None:
    """The old path has to keep working; it is what the archive is in."""
    from aether.aerodynamics.cfd.fields import PERFECT_AIR

    field = _field(np.array([0.05]), np.array([[2400.0, 0.0, 0.0]]), np.array([1197.0]), 1.4)
    assert not field.has_primitives
    assert field.pressure(PERFECT_AIR) == pytest.approx(1197.0)


def test_velocity_is_read_rather_than_divided_when_the_solver_wrote_it(tmp_path: Path) -> None:
    from aether.aerodynamics.cfd.fields import read_volume

    names = [*_VOLUME_NAMES, "Velocity_x", "Velocity_y", "Velocity_z"]
    rows = np.array([[0.0, 0.0, 0.0, 0.05, 1197.0, 226.5, 8.0, 2400.0, -3.0, 7.0]])
    _tecplot(tmp_path / "v.dat", names, rows)
    velocity = read_volume(tmp_path / "v.dat").velocity
    assert velocity == pytest.approx(np.array([[2400.0, -3.0, 7.0]]))


def test_block_packed_and_headerless_files_are_refused(tmp_path: Path) -> None:
    """Refused by name: silently misreading a block file gives a plausible field."""
    from aether.aerodynamics.cfd.fields import read_volume

    rows = np.zeros((2, 7))
    _tecplot(tmp_path / "block.dat", _VOLUME_NAMES, rows, packing="BLOCK")
    with pytest.raises(ValueError, match="DATAPACKING=POINT"):
        read_volume(tmp_path / "block.dat")

    (tmp_path / "none.dat").write_text('TITLE = "x"\nVARIABLES = "x","y","z"\n')
    with pytest.raises(ValueError, match="no ZONE record"):
        read_volume(tmp_path / "none.dat")


def test_a_short_row_is_refused_rather_than_padded(tmp_path: Path) -> None:
    from aether.aerodynamics.cfd.fields import read_volume

    path = tmp_path / "v.dat"
    _tecplot(path, _VOLUME_NAMES, np.zeros((2, 7)))
    lines = path.read_text().splitlines()
    lines[3] = "1.0\t2.0\t3.0"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="values for"):
        read_volume(path)
