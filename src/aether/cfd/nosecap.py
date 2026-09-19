r"""A blunt nose without a pole.

A surface of revolution has a **singularity at its axis**: every meridian
converges to one vertex, so the panels there are slivers whose circumferential
extent goes to zero while their meridional extent does not. Measured on the
Mach 8 sphere-cone, the first ring off the pole sits at 4.44 degrees where the
circumferential spacing is 0.61 mm against 4.65 mm meridional -- an aspect ratio
of 7.6 that worsens all the way in, and the pole vertex itself carries **48**
triangles.

That is exactly what LAURA's Figure 11.5 calls a *"singularity free surface grid
over blunt body"* :cite:`cheatwood1996laura`, and the standard construction is a
**cubed-sphere** (equivalently, butterfly or O-H) patch: a square-topology grid
gnomonically projected onto the cap, so the centre vertex has **valence four**
and no cell is a sliver.

The stagnation point is the worst place in a hypersonic mesh to have the worst
cells. It carries the peak pressure, the peak heating, and the largest axial
force per unit area on the whole vehicle.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["CapReplacement", "nose_sphere_radius", "replace_nose_pole"]

#: Where the square core's corners sit, as a fraction of the seam angle. The
#: rest of the cap is the transition annulus.
CORE_FRACTION = 0.55

#: Tolerance on a ring's polar angle, radians. The rings are identified by
#: ``np.round(theta, 6)``, so two vertices of one ring can differ by up to 5e-7
#: and still be the same ring; every test against a ring angle has to use the
#: same figure or the ring is one set of vertices in one place and another set
#: somewhere else.
RING_TOLERANCE = 1.0e-6

#: Cell-to-cell growth the transition annulus is graded to. The ring *count*
#: follows from it and from how far the core boundary is from the seam, rather
#: than being fixed: a fixed count spaces the rings evenly and leaves a jump
#: wherever the core's own step does not happen to match.
GROWTH = 1.3


@dataclass(frozen=True)
class CapReplacement:
    """What the surgery did, so the log can say it rather than imply it."""

    vertices: np.ndarray
    faces: np.ndarray
    seam_angle: float
    """Polar angle of the ring the patch was stitched to, degrees."""

    removed: int
    """Faces discarded from the original cap."""

    added: int
    """Faces in the patch plus its transition band."""

    peak_valence_before: int
    peak_valence_after: int
    """Peak valence **inside the cap region only**.

    Taken over the vertices the patch replaces rather than over the whole mesh.
    A closed body of revolution has a second fan pole at the centre of its base
    disc with the same valence as the nose had, so a global maximum is
    unchanged by a successful cap and the log line reads "80 -> 80" on a
    surgery that worked. The base pole sits in the wake and is not what this
    function is for.
    """


def _polar(vertices: np.ndarray, centre: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Polar angle from the -x axis, azimuth, and radius about ``centre``."""
    offset = vertices - centre
    radius = np.linalg.norm(offset, axis=1)
    safe = np.maximum(radius, 1e-30)
    theta = np.arccos(np.clip(-offset[:, 0] / safe, -1.0, 1.0))
    azimuth = np.arctan2(offset[:, 2], offset[:, 1])
    return theta, azimuth, radius


def _valence(faces: np.ndarray, count: int) -> np.ndarray:
    incident = np.zeros(count, dtype=np.int64)
    np.add.at(incident, faces.ravel(), 1)
    return incident


def nose_sphere_radius(surface: Any, rings: int = 6) -> float | None:
    r"""Radius of the spherical cap at the apex, measured from the mesh.

    A caller that passes its own ``nose_radius`` has to know which parameter
    name this body spells it with, and that is a rule with an exception in it:
    the hemisphere's is ``radius``, so the worker's ``geometry.get("nose_radius")``
    gate silently skipped the cap on the one body whose stagnation point is the
    entire point of the run. Measuring removes the rule.

    For a vertex at axial offset :math:`u` from the apex and radius :math:`r`
    about the axis, lying on a sphere of radius :math:`R` tangent to the apex,

    .. math:: r^2 + u^2 = 2Ru \quad\Longrightarrow\quad R = \frac{r^2 + u^2}{2u}

    Taken over the first few rings and returned only if they agree, so a
    conical or ogival nose -- where they will not -- returns ``None`` rather
    than a number that happens to come out of the algebra.
    """
    vertices = np.asarray(surface.vertices, dtype=np.float64)
    axis = vertices[:, 0]
    apex = float(axis.min())
    offset = axis - apex
    radial = np.hypot(vertices[:, 1], vertices[:, 2])
    # Ring = distinct axial station. Skip the apex itself, where u = 0.
    levels = np.unique(np.round(offset, 9))
    levels = levels[levels > 0.0][:rings]
    if len(levels) < 3:
        return None
    estimates = []
    for level in levels:
        on = np.abs(offset - level) < 1e-9
        if not on.any():
            continue
        estimates.append(float(np.median((radial[on] ** 2 + level**2) / (2.0 * level))))
    if len(estimates) < 3:
        return None
    values = np.asarray(estimates)
    median = float(np.median(values))
    if median <= 0.0 or float(np.max(np.abs(values - median))) > 1e-3 * median:
        return None
    return median


def _peak_valence_within(
    vertices: np.ndarray, faces: np.ndarray, centre: np.ndarray, seam_theta: float
) -> int:
    """Highest valence inside the seam, recomputed on the finished mesh.

    Recomputed rather than indexed through the survivor map because the patch
    contributes vertices of its own -- including the one that stands where the
    pole did -- and those have no entry in a map built from the original set.

    The radial test is not redundant with the angular one. On a hemisphere the
    cap centre *is* the base plane, so the base disc's fan pole sits within
    floating-point noise of the centre; :func:`_polar` divides by a radius of
    1e-8 there and reports it at **theta = 0**, whichever side of the centre it
    falls on. Without the radius test the base pole is read as the nose pole
    and a successful cap reports "80 -> 80".
    """
    radius = float(np.linalg.norm(centre)) or 1.0
    theta, _, distance = _polar(vertices, centre)
    inside = (theta < seam_theta + RING_TOLERANCE) & (
        np.abs(distance - radius) < 1e-6 * radius
    )
    if not inside.any():
        return 0
    return int(_valence(faces, len(vertices))[inside].max())


def replace_nose_pole(
    surface: Any,
    nose_radius: float | None = None,
    seam_ring: int = 6,
) -> CapReplacement | None:
    r"""Swap the polar cap for a cubed-sphere patch. ``None`` if it does not apply.

    ``seam_ring`` counts rings of the original cap outward from the pole; the
    patch is stitched to that ring, so everything inside it is rebuilt. The
    square grid's per-side cell count and the gnomonic half-angle are both
    **derived** from that ring rather than passed in, so the patch boundary
    carries exactly one vertex per circumferential station and the stitch adds
    none of its own.

    Returns ``None`` rather than raising when the surface has no recognisable
    polar cap -- a waverider has an edge, not a point, and asking is cheaper
    than the caller knowing.
    """
    if nose_radius is None:
        nose_radius = nose_sphere_radius(surface)
        if nose_radius is None:
            return None
    vertices = np.asarray(surface.vertices, dtype=np.float64).copy()
    faces = np.asarray(surface.faces, dtype=np.int64).copy()
    centre = np.array([nose_radius, 0.0, 0.0])

    theta, azimuth, radius = _polar(vertices, centre)
    on_sphere = np.abs(radius - nose_radius) < 1e-6 * max(nose_radius, 1.0)
    if int(on_sphere.sum()) < 32:
        return None

    # Rings of the cap, by distinct polar angle.
    levels = np.unique(np.round(theta[on_sphere], 6))
    if len(levels) <= seam_ring:
        return None
    seam_theta = float(levels[seam_ring])

    seam = np.nonzero(on_sphere & (np.abs(theta - seam_theta) < RING_TOLERANCE))[0]
    if len(seam) < 8:
        return None
    seam = seam[np.argsort(azimuth[seam])]

    # Derived, not assumed: the patch boundary carries exactly one vertex per
    # circumferential station, so the stitch adds no vertices and leaves no
    # crack. A default that happened to match one resolution left the pole in
    # place at every other one -- silently, by declining to apply.
    divisions = len(seam) // 4
    if divisions < 2 or divisions % 2 or 4 * divisions != len(seam):
        return None
    # The square core stops well short of the seam: the rings between them are
    # what reconciles a spherical square with a circle, and they need somewhere
    # to be. At 0.92 -- the old value, chosen so the corners merely stayed
    # inside the seam -- there is no room and the join is one row of quads.
    half_angle = math.degrees(
        math.atan(math.tan(CORE_FRACTION * seam_theta) / math.sqrt(2.0))
    )

    # The same tolerance the seam was selected with. At 1e-9 a vertex could be
    # **both** the seam and inside it: ``seam_theta`` is a rounded value, so a
    # ring's own vertices sit within 5e-7 of it either side, and the ones that
    # fell short doomed the band of faces immediately *outside* the seam. The
    # result was a mesh with a 160-edge crack around the join that still
    # reported the right area and the right valence, and it only appeared once
    # `nose_sphere_radius` began measuring the centre from the mesh -- shifting
    # every angle by about 1e-8 and pushing the hemisphere's seam across the
    # rounding. A tolerance that two tests disagree about is not a tolerance.
    inside = on_sphere & (theta < seam_theta - RING_TOLERANCE)
    doomed = np.any(inside[faces], axis=1)
    kept = faces[~doomed]
    # Restricted to the cap, for the reason on ``peak_valence_after``.
    region = on_sphere & (theta < seam_theta + RING_TOLERANCE)
    before = int(_valence(faces, len(vertices))[region].max()) if region.any() else 0

    # ---- the patch: a square core, then an O-grid transition to the seam
    #
    # Square *topology* in the middle is the whole point -- the centre vertex
    # ends with valence four instead of forty-eight.
    #
    # The hard part is the join, and the previous four attempts all failed at
    # it in the same place. A gnomonic square's boundary is a spherical square:
    # its corners reach 0.92 of the seam angle while its edge midpoints sit at
    # 0.66 of it, and its azimuths bunch toward the corners. Snapping that
    # straight onto the circular seam ring puts the whole reconciliation into
    # **one row of quads**, and the corners take all of it.
    #
    # That was recorded as "a thin band at the seam -- small, bounded". It is
    # bounded and it is not harmless: on the hemisphere all 102 cells below the
    # quality floor sat at exactly phi = 22.46 deg, azimuth 45 deg mod 90 --
    # the four diagonal corners -- and the Mach 8 solve diverged from exactly
    # there, x = 0.0033, y = z = 0.0122, by iteration 200.
    #
    # So the reconciliation is spread over several rings instead of one, which
    # is what an O-grid does and what makes it the standard construction. Ring
    # zero is the square core's boundary, ring ``TRANSITION_RINGS`` is the seam,
    # and each ring between interpolates both the polar angle and the azimuth.
    # Both sequences are monotonic in perimeter position, so interpolating them
    # cannot let neighbours swap -- which is what sank the attempt that blended
    # the two maps as 3-vectors, at 3512 inverted prisms.
    n = divisions
    half = n // 2
    index = np.arange(n + 1) - half
    di, dj = np.meshgrid(index, index, indexing="ij")

    tangent = math.tan(math.radians(half_angle))
    xi, eta = di / half, dj / half
    directions = np.stack(
        [-np.ones_like(xi, dtype=np.float64), xi * tangent, eta * tangent], axis=-1
    ).reshape(-1, 3)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    patch = centre + nose_radius * directions

    def node(i: np.ndarray | int, j: np.ndarray | int) -> np.ndarray:
        return np.asarray(i) * (n + 1) + np.asarray(j)

    # Boundary of the square core, walked once anticlockwise.
    walk = np.array(
        [node(i, 0) for i in range(n)]
        + [node(n, j) for j in range(n)]
        + [node(n - i, n) for i in range(n)]
        + [node(0, n - j) for j in range(n)],
        dtype=np.int64,
    )

    # Every core vertex is new, boundary included: the core boundary is now
    # ring zero of the transition rather than something snapped onto the seam.
    offset = len(vertices)
    core_index = offset + np.arange(len(patch))
    vertices = np.vstack([vertices, patch])

    # Polar angle and azimuth of the core boundary, in walk order. The azimuth
    # is unwrapped so it increases monotonically all the way round instead of
    # jumping at pi.
    core_theta, core_azimuth, _ = _polar(patch[walk], centre)
    core_azimuth = np.unwrap(core_azimuth)

    # Which seam vertex each perimeter position joins. By azimuth rank, so the
    # ring is not given a twist.
    order = np.argsort(np.argsort(core_azimuth))
    seam_for = seam[order]
    seam_azimuth = np.unwrap(azimuth[seam_for])
    # Unwrapping the two independently can leave them a turn apart.
    seam_azimuth += 2.0 * np.pi * np.round(
        (core_azimuth.mean() - seam_azimuth.mean()) / (2.0 * np.pi)
    )

    # The annulus is **graded**, starting at the core's own radial step and
    # growing outward. Spacing it evenly is what an even blend does and it
    # leaves a jump at the join: the core reaches its boundary in `half` steps
    # of about 0.9 degrees while four even rings cross the remaining 13.7 in
    # steps of 2.1, and a 2.4x jump in cell size is a bad cell however smooth
    # the shape is. Measured on the hemisphere, evenly spaced: the corner
    # singularity at the seam was gone and 1050 cells below the quality floor
    # had appeared at 10.8 to 17.3 degrees -- the join -- instead.
    #
    # ``span`` and the first step both vary with perimeter position, because
    # the core boundary is a spherical square: its corners stand at
    # CORE_FRACTION of the seam angle and its edge midpoints much closer in. So
    # each ray gets its own exponent and its own growth, which is what a graded
    # O-grid is.
    span = seam_theta - core_theta
    first = core_theta / max(half, 1)
    ratio = np.clip(first / np.maximum(span, 1e-12), 1e-4, 0.5)
    rings_needed = np.log1p((1.0 / ratio) * (GROWTH - 1.0)) / math.log(GROWTH)
    count = int(np.clip(round(float(rings_needed.max())), 3, 12))
    # The exponent that puts the first ring exactly one core step out.
    exponent = np.log(ratio) / math.log(1.0 / count)

    rings: list[np.ndarray] = [core_index[walk]]
    for step in range(1, count):
        weight = (step / count) ** exponent
        ring_theta = core_theta + weight * span
        ring_azimuth = core_azimuth + weight * (seam_azimuth - core_azimuth)
        ring = centre + nose_radius * np.stack(
            [
                -np.cos(ring_theta),
                np.sin(ring_theta) * np.cos(ring_azimuth),
                np.sin(ring_theta) * np.sin(ring_azimuth),
            ],
            axis=-1,
        )
        rings.append(len(vertices) + np.arange(len(ring)))
        vertices = np.vstack([vertices, ring])
    rings.append(seam_for)

    quads = [
        (node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1))
        for i in range(n)
        for j in range(n)
    ]
    grid = core_index[np.array(quads, dtype=np.int64)]

    # The annulus: one quad per perimeter position per ring gap, closed round.
    perimeter = len(walk)
    for inner, outer in itertools.pairwise(rings):
        following = np.roll(np.arange(perimeter), -1)
        grid = np.vstack([
            grid,
            np.stack([inner, inner[following], outer[following], outer], axis=1),
        ])

    patch_faces = np.vstack([grid[:, [0, 1, 2]], grid[:, [0, 2, 3]]])

    # Wound outward, **checked rather than assumed** -- the same rule
    # `build_surface` applies to a whole body, for the same reason. An
    # inward-wound patch is closed, has the right area, and integrates to the
    # right panel loads; it only fails once it is extruded, as 7138 inverted
    # prisms with a scaled Jacobian of -0.9997.
    # Per face, not summed over all of them. The sum only decides correctly
    # when every face already agrees, which was true while the patch was one
    # square grid and stopped being true the moment it became a core plus an
    # annulus: the two are built by different loops and came out with opposite
    # senses, so the sum flipped whichever group happened to be outvoted and
    # left 6452 inverted prisms after extrusion. Every face here lies on a
    # sphere about `centre`, so outward is defined for each one individually
    # and there is no reason to take a vote.
    corners = vertices[patch_faces]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    outward = corners.mean(axis=1) - centre
    inverted = np.einsum("ij,ij->i", normals, outward) < 0.0
    patch_faces[inverted] = patch_faces[inverted][:, ::-1]

    added = np.vstack([kept, patch_faces])

    # Drop vertices nothing references any more -- chiefly the old pole, which
    # sits at exactly the position the new patch centre now occupies. Two
    # coincident vertices with one orphaned is how a mesh reads as watertight
    # and behaves as though it has a crack, and an unreferenced node in a .su2
    # is a node SU2 has to decide what to do with.
    used = np.unique(added)
    survivor = np.full(len(vertices), -1, dtype=np.int64)
    survivor[used] = np.arange(len(used))
    added = survivor[added]
    vertices = vertices[used]

    result = CapReplacement(
        vertices=vertices,
        faces=added,
        seam_angle=math.degrees(seam_theta),
        removed=int(doomed.sum()),
        added=len(patch_faces),
        peak_valence_before=before,
        peak_valence_after=_peak_valence_within(vertices, added, centre, seam_theta),
    )
    return result  # noqa: RET504
