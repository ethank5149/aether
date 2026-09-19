"""Vehicle outer mould line from a triangle mesh."""

import itertools
import struct
from pathlib import Path

import numpy as np
import pytest

from aether.geometry import load_stl

# The sanitized default vehicle mesh (formerly reference/model.stl).
_STL = Path(__file__).resolve().parents[1] / "models" / "default.stl"
_HAVE_STL = _STL.is_file()


def _write_stl(path, triangles):
    with open(path, "wb") as handle:
        handle.write(b"\0" * 80)
        handle.write(struct.pack("<I", len(triangles)))
        for tri in triangles:
            a, b, c = np.asarray(tri, dtype=np.float64)
            normal = np.cross(b - a, c - a)
            normal = normal / max(float(np.linalg.norm(normal)), 1e-30)
            handle.write(struct.pack("<3f", *normal))
            for point in (a, b, c):
                handle.write(struct.pack("<3f", *point))
            handle.write(b"\0\0")


def _unit_cube():
    """A closed axis-aligned unit cube, outward-wound."""
    v = np.array([[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)])
    quads = [
        (0, 1, 3, 2, [-1, 0, 0]), (4, 6, 7, 5, [1, 0, 0]),
        (0, 4, 5, 1, [0, -1, 0]), (2, 3, 7, 6, [0, 1, 0]),
        (0, 2, 6, 4, [0, 0, -1]), (1, 5, 7, 3, [0, 0, 1]),
    ]
    tris = []
    for i, j, k, m, outward in quads:
        for triple in ((i, j, k), (i, k, m)):
            p = v[list(triple)]
            if np.dot(np.cross(p[1] - p[0], p[2] - p[0]), outward) < 0:
                p = p[::-1]
            tris.append(p)
    return tris


class TestLoading:
    def test_a_cube_round_trips_through_a_binary_stl(self, tmp_path):
        path = tmp_path / "cube.stl"
        _write_stl(path, _unit_cube())
        mesh = load_stl(path)
        assert mesh.n_faces == 12
        assert len(mesh.vertices) == 8, "coincident vertices must be welded"
        assert mesh.wetted_area == pytest.approx(6.0, rel=1e-6)
        assert mesh.is_closed

    def test_welding_is_what_makes_topology_possible(self, tmp_path):
        """STL is a bag of triangles with no shared vertices at all, so
        without welding there is no edge adjacency and no closure test."""
        path = tmp_path / "cube.stl"
        _write_stl(path, _unit_cube())
        mesh = load_stl(path)
        _, counts = mesh.edge_counts()
        assert set(np.unique(counts).tolist()) == {2}

    def test_degenerate_triangles_are_dropped_and_counted(self, tmp_path):
        path = tmp_path / "spike.stl"
        tris = _unit_cube()
        tris.append(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
        _write_stl(path, tris)
        mesh = load_stl(path)
        assert mesh.degenerate_dropped == 1
        assert mesh.n_faces == 12
        assert np.all(np.isfinite(mesh.normals)), "a zero-area facet has no normal"

    def test_a_short_file_is_refused(self, tmp_path):
        path = tmp_path / "bad.stl"
        path.write_bytes(b"\0" * 10)
        with pytest.raises(ValueError, match="too short"):
            load_stl(path)

    def test_a_file_that_is_neither_format_says_so(self, tmp_path):
        path = tmp_path / "bad.stl"
        path.write_bytes(b"\0" * 80 + struct.pack("<I", 9999) + b"junk")
        with pytest.raises(ValueError, match="neither a valid binary STL"):
            load_stl(path)


class TestMassProperties:
    def test_a_unit_cube_has_unit_volume_and_a_centred_centroid(self, tmp_path):
        path = tmp_path / "cube.stl"
        _write_stl(path, _unit_cube())
        mesh = load_stl(path)
        properties = mesh.mass_properties(density=1000.0)
        assert properties["volume"] == pytest.approx(1.0, rel=1e-9)
        assert properties["mass"] == pytest.approx(1000.0, rel=1e-9)
        assert np.allclose(properties["centroid"], 0.5, atol=1e-9)

    def test_the_cube_inertia_matches_the_closed_form(self, tmp_path):
        """A solid cube of side a about its centroid has I = m a^2 / 6 on
        every diagonal and zero off it. Checked against arithmetic rather
        than against the code that produced it."""
        path = tmp_path / "cube.stl"
        _write_stl(path, _unit_cube())
        mesh = load_stl(path)
        inertia = mesh.mass_properties(density=1000.0)["inertia_about_centroid"]
        assert np.allclose(np.diag(inertia), 1000.0 / 6.0, rtol=1e-9, atol=1e-9)
        off = inertia - np.diag(np.diag(inertia))
        assert np.max(np.abs(off)) < 1e-6

    def test_an_open_mesh_refuses_to_report_mass(self, tmp_path):
        """The divergence-theorem sum still returns a number on an open
        surface, and that number can put the centroid outside the bounding
        box — which is exactly what the reference vehicle's does."""
        path = tmp_path / "open.stl"
        _write_stl(path, _unit_cube()[:-2])
        mesh = load_stl(path)
        assert not mesh.is_closed
        with pytest.raises(ValueError, match="not closed"):
            mesh.mass_properties(density=1000.0)

    def test_a_non_positive_density_is_refused(self, tmp_path):
        path = tmp_path / "cube.stl"
        _write_stl(path, _unit_cube())
        with pytest.raises(ValueError, match="density must be"):
            load_stl(path).mass_properties(density=0.0)


class TestFrameConventions:
    def test_to_body_axes_puts_the_nose_on_plus_x(self, tmp_path):
        """`PanelModel` and the vehicle glyph both take the nose along +x;
        meshes are authored along +z. Doing that rotation once, explicitly,
        is what stops the two conventions meeting silently."""
        path = tmp_path / "cone.stl"
        # A crude cone along +z with the tip at the top.
        angles = np.linspace(0.0, 2 * np.pi, 17)[:-1]
        rim = np.stack(
            [np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1
        )
        tip = np.array([0.0, 0.0, 4.0])
        base = np.array([0.0, 0.0, 0.0])
        tris = []
        for i in range(len(rim)):
            j = (i + 1) % len(rim)
            tris.append([rim[i], rim[j], tip])
            tris.append([rim[j], rim[i], base])
        _write_stl(path, tris)
        mesh = load_stl(path)
        assert mesh.axis == 2
        body = mesh.to_body_axes()
        assert body.axis == 0
        assert body.length == pytest.approx(mesh.length, rel=1e-9)
        # The tip sits at the origin and the body extends to -x.
        assert body.vertices[:, 0].max() == pytest.approx(0.0, abs=1e-9)
        assert body.vertices[:, 0].min() == pytest.approx(-4.0, rel=1e-6)
        nose = body.vertices[np.argmax(body.vertices[:, 0])]
        assert np.allclose(nose[1:], 0.0, atol=1e-6)

    def test_rotation_preserves_area(self, tmp_path):
        path = tmp_path / "cube.stl"
        _write_stl(path, _unit_cube())
        mesh = load_stl(path)
        assert mesh.to_body_axes("keep").wetted_area == pytest.approx(
            mesh.wetted_area, rel=1e-12
        )

    def test_an_unknown_origin_is_refused(self, tmp_path):
        path = tmp_path / "cube.stl"
        _write_stl(path, _unit_cube())
        with pytest.raises(ValueError, match="origin must be"):
            load_stl(path).to_body_axes("middle")


class TestFrontalArea:
    def test_a_cube_projects_to_its_face(self, tmp_path):
        path = tmp_path / "cube.stl"
        _write_stl(path, _unit_cube())
        assert load_stl(path).frontal_area(256) == pytest.approx(1.0, rel=2e-2)

    def test_internal_faces_do_not_inflate_it(self, tmp_path):
        """The reason for rasterising. The analytic form — half the sum of
        |n.a|A — counts an internal bulkhead in full, and the reference
        vehicle has two: it reported 26.29 m² for a body whose silhouette
        is 10.93."""
        path = tmp_path / "walled.stl"
        tris = _unit_cube()
        # An internal wall at x = 0.5, both windings so it is closed-ish.
        wall = [
            [np.array([0.5, 0.0, 0.0]), np.array([0.5, 1.0, 0.0]), np.array([0.5, 1.0, 1.0])],
            [np.array([0.5, 0.0, 0.0]), np.array([0.5, 1.0, 1.0]), np.array([0.5, 0.0, 1.0])],
        ]
        _write_stl(path, tris + wall)
        mesh = load_stl(path)
        analytic = 0.5 * np.sum(np.abs(mesh.normals @ np.array([1.0, 0.0, 0.0])) * mesh.areas)
        assert analytic > 1.4, "the analytic form must be fooled for this to matter"
        assert mesh.frontal_area(256) == pytest.approx(1.0, rel=2e-2)


@pytest.mark.skipif(not _HAVE_STL, reason="reference/model.stl not present")
class TestReferenceVehicle:
    @pytest.fixture(scope="class")
    @classmethod
    def mesh(cls):
        return load_stl(_STL, name="default vehicle")

    def test_it_is_a_35_m_body_of_revolution(self, mesh):
        assert mesh.length == pytest.approx(35.0, abs=1e-3)
        assert mesh.extent[0] == pytest.approx(mesh.extent[1], rel=1e-3)
        assert mesh.axis == 2

    def test_the_frontal_area_matches_the_maximum_radius(self, mesh):
        """An independent check on the rasteriser: the mesh's circular
        sections are 40-gons, and a regular 40-gon inscribed in a circle has
        0.41 % less area. The measured deficit is 0.48 %."""
        _, peak = mesh.station_profile()
        circle = np.pi * float(peak.max()) ** 2
        measured = mesh.frontal_area(512)
        assert measured < circle
        assert measured == pytest.approx(circle, rel=0.01)

    def test_the_nose_is_an_ogive_not_a_sphere(self, mesh):
        """Which means `nose_radius` is a window-dependent bound rather than
        a property, and Sutton-Graves — which assumes a hemispherical
        stagnation region — is being applied approximately."""
        assert 0.55 < mesh.nose_exponent() < 0.65
        near = mesh.nose_radius(0.05)
        far = mesh.nose_radius(1.0)
        assert far > 1.5 * near, "a spherical nose would give the same answer"

    def test_it_is_not_watertight_and_says_where(self, mesh):
        assert not mesh.is_closed
        holes = mesh.boundary_stations()
        assert holes.size >= 1
        # Both open loops are at the aft end, around the nozzle.
        assert np.all(holes < -18.0)

    def test_the_separation_bands_are_found(self, mesh):
        """Four rings stand 0.13 m proud of the 1.734 m body. They are
        where the separation hardware sits, and they are the natural first
        guess at the stage divisions — a guess, not a staging sequence."""
        bands = mesh.raised_bands()
        assert bands.size == 4
        assert np.allclose(np.sort(bands), [-17.90, -8.05, 1.63, 8.23], atol=0.05)

    def test_sections_partition_the_wetted_area(self, mesh):
        bands = np.sort(mesh.raised_bands())
        lo, hi = mesh.bounds
        cuts = [float(lo[2]) - 1.0, *bands.tolist(), float(hi[2]) + 1.0]
        total = sum(
            mesh.section(low, high).wetted_area
            for low, high in itertools.pairwise(cuts)
        )
        assert total == pytest.approx(mesh.wetted_area, rel=1e-9)

    def test_an_empty_section_is_refused(self, mesh):
        with pytest.raises(ValueError, match="no faces between"):
            mesh.section(100.0, 200.0)
        with pytest.raises(ValueError, match="need high > low"):
            mesh.section(5.0, 5.0)

    def test_it_feeds_the_panel_model_losslessly(self, mesh):
        """A triangle *is* a panel: centroid, outward unit normal, area.
        This step introduces no approximation, which is the reason the
        panel model was worth having a mesh for."""
        body = mesh.to_body_axes()
        model = body.panel_model()
        assert model.n_panels == body.n_faces
        assert model.total_area == pytest.approx(body.wetted_area, rel=1e-12)
        # And the model's own validation accepted the normals as unit.
        assert np.allclose(np.linalg.norm(model.normals, axis=1), 1.0, atol=1e-9)


class TestOrientationRepair:
    """Winding is not to be trusted, and this mesh proves why.

    Its barrel is wound outward and all 2,000 nose facets inward, and the
    STL's own stored normals agree with the bad winding — so the file
    carries no second opinion. Integrating pressure over it makes the nose,
    which carries most of a slender body's axial force, push the wrong way.
    """

    @staticmethod
    def _flipped_cube(tmp_path):
        path = tmp_path / "flipped.stl"
        tris = _unit_cube()
        tris = [t if i % 3 else t[::-1] for i, t in enumerate(tris)]
        _write_stl(path, tris)
        return load_stl(path)

    def test_orientation_is_repaired_on_a_cube_with_flipped_faces(self, tmp_path):
        mesh = self._flipped_cube(tmp_path)
        volume_before = mesh.mass_properties(1.0)["volume"]
        fixed = mesh.oriented()
        assert fixed.mass_properties(1.0)["volume"] == pytest.approx(1.0, rel=1e-9)
        assert volume_before != pytest.approx(1.0, rel=1e-6), (
            "the un-repaired mesh must actually be wrong for this to prove anything"
        )

    def test_repair_leaves_a_correct_mesh_alone(self, tmp_path):
        path = tmp_path / "cube.stl"
        _write_stl(path, _unit_cube())
        mesh = load_stl(path)
        assert np.allclose(mesh.oriented().normals, mesh.normals)

    def test_areas_and_topology_survive_the_repair(self, tmp_path):
        mesh = self._flipped_cube(tmp_path)
        fixed = mesh.oriented()
        assert fixed.wetted_area == pytest.approx(mesh.wetted_area, rel=1e-12)
        assert fixed.n_faces == mesh.n_faces

    def test_an_even_probe_count_is_refused(self, tmp_path):
        path = tmp_path / "cube.stl"
        _write_stl(path, _unit_cube())
        with pytest.raises(ValueError, match="cannot tie"):
            load_stl(path).oriented(probes=6)

    def test_the_visibility_test_keeps_a_convex_body_whole(self, tmp_path):
        """Nothing on a convex closed surface is enclosed by it."""
        path = tmp_path / "cube.stl"
        _write_stl(path, _unit_cube())
        assert load_stl(path).exterior_faces().all()

    def test_an_enclosed_shell_is_removed(self, tmp_path):
        """The reason the test exists: a mesh can carry geometry that is not
        outer surface, and a panel method integrates whatever it is handed."""
        path = tmp_path / "nested.stl"
        outer = _unit_cube()
        inner = [np.asarray(t) * 0.3 + 0.35 for t in _unit_cube()]
        _write_stl(path, outer + inner)
        mesh = load_stl(path).oriented()
        keep = mesh.exterior_faces()
        assert keep.sum() == 12, "only the outer cube should survive"
        assert mesh.areas[~keep].sum() == pytest.approx(6 * 0.3 * 0.3, rel=1e-6)


@pytest.mark.skipif(not _HAVE_STL, reason="reference/model.stl not present")
class TestReferenceVehicleOrientation:
    @pytest.fixture(scope="class")
    @classmethod
    def mesh(cls):
        return load_stl(_STL, name="default vehicle").to_body_axes()

    def test_the_raw_winding_is_inconsistent(self, mesh):
        """0.594 outward on a body of revolution is not a mesh you can
        integrate pressure over."""
        assert mesh.outward_fraction() < 0.7

    def test_repair_fixes_it(self, mesh):
        assert mesh.oriented().outward_fraction() > 0.85

    def test_the_nose_is_wound_inward_before_repair_and_outward_after(self, mesh):
        nose = mesh.centroids[:, 0] > -3.0
        assert nose.sum() > 500
        assert np.all(mesh.normals[nose, 0] < 0.0), "raw nose points aft"
        assert np.all(mesh.oriented().normals[nose, 0] > 0.0), "repaired nose points forward"

    def test_the_visibility_test_keeps_the_nose_and_drops_the_nozzle(self, mesh):
        fixed = mesh.oriented()
        keep = fixed.exterior_faces()
        nose = fixed.centroids[:, 0] > -3.0
        assert keep[nose].all(), "the nose is exterior by any definition"
        removed = fixed.centroids[~keep]
        assert removed[:, 0].max() < -3.0, "nothing near the nose may be removed"

    def test_removing_interior_geometry_cannot_change_the_silhouette(self, mesh):
        fixed = mesh.oriented()
        assert fixed.exterior().frontal_area(256) == pytest.approx(
            fixed.frontal_area(256), rel=1e-6
        )
