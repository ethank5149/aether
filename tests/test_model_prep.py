"""Conditioning and export of corrected master vehicle models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aether.geometry.mesh import VehicleMesh, load_stl, write_stl
from aether.geometry.prep import condition, export_master, measure, sha256_of


def _source_stl(directory: Path) -> Path:
    """A real file on disk to hash, made from the synthetic wedge.

    The provenance record is a SHA-256 of the *source file*, so these tests
    need an actual file rather than a mesh object. They used to reach for a
    named vehicle under ``data/vehicles/``, which is why they could not follow
    this module into the public kernel: the code is generic geometry and the
    mesh is not. Writing the wedge out gives the same coverage of the hashing
    and manifest path with nothing in it that identifies a system.
    """
    path = directory / "source.stl"
    write_stl(_wedge(1000.0), path)
    return path


def _wedge(scale: float = 1.0, span_on_z: bool = False) -> VehicleMesh:
    """A tiny asymmetric body: long on x, wide on one cross axis, thin on the other."""
    wide, thin = 2.0 * scale, 0.5 * scale
    y, z = (thin, wide) if span_on_z else (wide, thin)
    v = np.array([
        [0.0, 0.0, 0.0], [-4.0 * scale, y, z], [-4.0 * scale, -y, z],
        [-4.0 * scale, y, -z], [-4.0 * scale, -y, -z],
    ])
    f = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 4], [0, 4, 3], [1, 3, 4], [1, 4, 2]])
    return VehicleMesh(vertices=v, faces=f, name="wedge")


class TestCondition:
    def test_scales_to_the_published_length(self):
        mesh, report = condition(_wedge(1000.0), length_m=4.0,
                                 repair_winding=False, strip_interior=False)
        assert float(np.ptp(np.asarray(mesh.vertices)[:, 0])) == pytest.approx(4.0)
        assert report.scale_factor == pytest.approx(4.0 / 4000.0)
        assert any("scaled" in t for t in report.transforms)

    def test_rolls_a_lifting_body_span_onto_y(self):
        """A mesh authored with its span on z is pitched edge-on if left alone."""
        mesh, report = condition(_wedge(span_on_z=True), lifting_body=True,
                                 repair_winding=False, strip_interior=False)
        v = np.asarray(mesh.vertices)
        assert np.ptp(v[:, 1]) > np.ptp(v[:, 2])       # span now on y
        assert any("rolled" in t for t in report.transforms)

    def test_leaves_an_already_correct_body_unrolled(self):
        mesh, report = condition(_wedge(span_on_z=False), lifting_body=True,
                                 repair_winding=False, strip_interior=False)
        v = np.asarray(mesh.vertices)
        assert np.ptp(v[:, 1]) > np.ptp(v[:, 2])
        assert not any("rolled" in t for t in report.transforms)

    def test_centres_the_cross_section(self):
        offset = _wedge()
        shifted = VehicleMesh(
            vertices=np.asarray(offset.vertices) + np.array([0.0, 5.0, 3.0]),
            faces=np.asarray(offset.faces), name="shifted")
        mesh, _ = condition(shifted, repair_winding=False, strip_interior=False)
        v = np.asarray(mesh.vertices)
        for axis in (1, 2):
            assert 0.5 * (v[:, axis].max() + v[:, axis].min()) == pytest.approx(0.0, abs=1e-9)

    def test_rejects_a_degenerate_body(self):
        flat = VehicleMesh(
            vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 2]]), name="flat")
        with pytest.raises(ValueError, match="body axis"):
            condition(flat, length_m=1.0, repair_winding=False, strip_interior=False)


class TestMeasure:
    def test_omits_nose_descriptors_for_a_non_axisymmetric_body(self):
        """An 859 m 'nose radius' on a 3.6 m waverider is worse than no number."""
        mesh, _ = condition(_wedge(span_on_z=True), lifting_body=True,
                            repair_winding=False, strip_interior=False)
        recorded = measure(mesh)
        assert "nose_radius_m" not in recorded
        assert "not axisymmetric" in recorded["nose_descriptors"]

    def test_records_the_geometry(self):
        recorded = measure(_wedge())
        for key in ("length_m", "span_y_m", "height_z_m", "wetted_area_m2",
                    "frontal_area_m2", "faces", "vertices"):
            assert key in recorded and np.isfinite(float(recorded[key]))


class TestExportMaster:
    def test_writes_stl_and_manifest_with_provenance(self, tmp_path):
        source = _source_stl(tmp_path)
        mesh, report = condition(_wedge(1000.0), length_m=4.0,
                                 repair_winding=False, strip_interior=False)
        out = export_master(mesh, tmp_path / "veh", name="demo", report=report,
                            source=source, description="demo master",
                            published={"length_m": 4.0})
        assert out.exists() and out.name == "demo.stl"

        manifest = json.loads((tmp_path / "veh" / "manifest.json").read_text())
        assert manifest["schema"] == "aether-gambit.vehicle/1"
        assert manifest["source_mesh"]["sha256"] == sha256_of(source)
        assert manifest["conditioning"]["scale_factor"] == pytest.approx(4.0 / 4000.0)
        assert manifest["published"]["length_m"] == 4.0
        assert manifest["components"][0]["name"] == "demo"

    def test_a_second_body_accumulates_in_the_manifest(self, tmp_path):
        first, _ = condition(_wedge(), repair_winding=False, strip_interior=False)
        export_master(first, tmp_path / "v", name="stack")
        export_master(first, tmp_path / "v", name="payload")
        manifest = json.loads((tmp_path / "v" / "manifest.json").read_text())
        assert {c["name"] for c in manifest["components"]} == {"stack", "payload"}

    def test_reexport_replaces_rather_than_duplicates(self, tmp_path):
        mesh, _ = condition(_wedge(), repair_winding=False, strip_interior=False)
        export_master(mesh, tmp_path / "v", name="stack")
        export_master(mesh, tmp_path / "v", name="stack")
        manifest = json.loads((tmp_path / "v" / "manifest.json").read_text())
        assert len(manifest["components"]) == 1

    def test_the_exported_master_reloads_unchanged(self, tmp_path):
        """The whole point: a table built from the master matches the source."""
        source = _source_stl(tmp_path)
        mesh, report = condition(load_stl(source), length_m=3.6, lifting_body=True)
        export_master(mesh, tmp_path / "veh", name="body", report=report, source=source)
        reloaded = load_stl(tmp_path / "veh" / "body.stl").to_body_axes()
        before, after = np.asarray(mesh.vertices), np.asarray(reloaded.vertices)
        assert len(reloaded.faces) == len(mesh.faces)
        for axis in range(3):
            assert np.ptp(after[:, axis]) == pytest.approx(np.ptp(before[:, axis]), rel=1e-6)
