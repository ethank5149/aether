"""Turn an authored STL into a corrected master model the codebase can trust.

An STL is a bag of triangles. It carries no units, no orientation convention, no
guarantee that its facets wind consistently, and no record of where it came from —
and every one of those silently changes an aerodynamic answer rather than the
picture. A mesh authored in millimetres loads as a 3.6 km vehicle whose frontal
area is out by :math:`10^6`; a lifting body authored with its span on the wrong
axis is pitched edge-on by the panel solver and reports the lift of a knife.

This module is the conditioning step that closes that gap, and the export that
makes it auditable. :func:`condition` repairs and normalises a mesh —
consistent winding, degenerate facets dropped, interior geometry removed, nose
along +x, span on :math:`y`, scaled to a stated length. :func:`measure` records
what the corrected body actually is. :func:`export_master` writes the STL beside a
manifest carrying the source hash, every defect repaired, every factor applied and
every quantity measured, so a table built from it can be traced back to the
geometry it came from.

The manifest schema matches the one already carried by the bundled reference
vehicle, so a model prepared here is indistinguishable from one prepared before.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "ConditionReport",
    "condition",
    "export_master",
    "measure",
    "sha256_of",
]

#: Manifest schema identifier. Deliberately unchanged by the move into the
#: public kernel: it names a *format*, not a package, the format has not
#: changed, and three vehicle manifests on disk already carry this value.
#: Renaming it would invalidate them to gain nothing but tidiness.
_SCHEMA = "aether-gambit.vehicle/1"


def sha256_of(path: str | Path) -> str:
    """Hex digest of a file, recorded so a master can be traced to its source."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class ConditionReport:
    """What conditioning changed, in the words a manifest records."""

    faces_raw: int = 0
    faces_degenerate_dropped: int = 0
    defects_repaired: list[str] = field(default_factory=list)
    transforms: list[str] = field(default_factory=list)
    scale_factor: float = 1.0
    interior_faces_removed: int = 0


def condition(
    mesh: Any,
    *,
    length_m: float | None = None,
    lifting_body: bool = False,
    repair_winding: bool = True,
    strip_interior: bool = True,
    centre_cross_section: bool = True,
    backend: str = "numpy",
) -> tuple[Any, ConditionReport]:
    """Repair and normalise a mesh, returning it with a record of what changed.

    The order matters and is not arbitrary. Winding is repaired **first**, because
    outward normals are what every later step and the panel solver depend on;
    interior geometry is removed next, because a launch vehicle's modelled internals
    contribute wetted area and shadowed panels that no external flow ever sees;
    the body is then put nose-along-+x, rolled so a lifting body's span lies on
    :math:`y`, and only then scaled — so the scale factor is reported against a
    body already in its final attitude.

    Parameters
    ----------
    length_m:
        Published length to scale to. ``None`` leaves the mesh in its authored
        units, which is only right when those units are already metres.
    lifting_body:
        Roll the wider cross-dimension onto :math:`y`. A body of revolution has no
        meaningful roll and is left alone.
    """
    report = ConditionReport()
    current = mesh
    report.faces_raw = len(current.faces)
    report.faces_degenerate_dropped = int(getattr(current, "degenerate_dropped", 0) or 0)
    if report.faces_degenerate_dropped:
        report.defects_repaired.append(
            f"{report.faces_degenerate_dropped} zero-area facet(s) dropped on load"
        )

    if repair_winding:
        before = np.asarray(current.normals)
        oriented = current.oriented(backend=backend)
        after = np.asarray(oriented.normals)
        flipped = int(np.sum(np.einsum("ij,ij->i", before, after) < 0.0))
        if flipped:
            report.defects_repaired.append(
                f"inconsistent winding: {flipped} facet(s) wound inward, repaired by "
                "per-face parity with a ray majority vote"
            )
        current = oriented

    if strip_interior:
        kept = current.exterior(backend=backend)
        removed = int(len(current.faces) - len(kept.faces))
        if removed:
            report.interior_faces_removed = removed
            report.defects_repaired.append(
                f"{removed} interior facet(s) removed; they carry wetted area no "
                "external flow ever sees"
            )
            current = kept

    current = current.to_body_axes()
    report.transforms.append("nose along +x (to_body_axes)")

    vertices = np.asarray(current.vertices, dtype=np.float64).copy()
    if lifting_body and np.ptp(vertices[:, 2]) > np.ptp(vertices[:, 1]):
        y = vertices[:, 1].copy()
        vertices[:, 1] = vertices[:, 2]
        vertices[:, 2] = -y
        report.transforms.append(
            "rolled -90 deg about x so the span lies on y (the panel solver "
            "pitches in the x-z plane)"
        )
    if centre_cross_section:
        for axis in (1, 2):
            offset = 0.5 * (vertices[:, axis].max() + vertices[:, axis].min())
            if abs(offset) > 1e-12:
                vertices[:, axis] -= offset
        report.transforms.append("cross-section centred on the body axis")

    from aether.geometry.mesh import VehicleMesh

    current = VehicleMesh(
        vertices=vertices, faces=np.asarray(current.faces).copy(), name=current.name
    )

    if length_m is not None:
        span = float(np.ptp(np.asarray(current.vertices)[:, 0]))
        if span <= 0.0:
            raise ValueError("mesh has no extent along its body axis")
        factor = float(length_m) / span
        current = current.scaled(axial=factor, radial=factor)
        report.scale_factor = factor
        report.transforms.append(
            f"scaled uniformly by {factor:.6g} to a published length of {length_m:g} m"
        )
    return current, report


def measure(mesh: Any) -> dict[str, Any]:
    """The geometric quantities a manifest records about a corrected body."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    radius = np.hypot(vertices[:, 1], vertices[:, 2])
    out: dict[str, Any] = {
        "length_m": float(np.ptp(vertices[:, 0])),
        "span_y_m": float(np.ptp(vertices[:, 1])),
        "height_z_m": float(np.ptp(vertices[:, 2])),
        "max_diameter_m": float(2.0 * radius.max()),
        "wetted_area_m2": float(mesh.wetted_area),
        "frontal_area_m2": float(mesh.frontal_area()),
        "faces": len(mesh.faces),
        "vertices": len(mesh.vertices),
    }
    # Nose descriptors are defined for a body of revolution: both fit a radius
    # against station, which presumes the cross-section is a circle. A waverider's
    # is not, and forcing the fit returns a number that is finite, meaningless and
    # indistinguishable from a real one -- an 859 m "nose radius" on a 3.6 m
    # vehicle. Reported only where the body is close to axisymmetric, and the
    # reason recorded where it is not.
    span, height = out["span_y_m"], out["height_z_m"]
    axisymmetric = min(span, height) > 0.0 and max(span, height) / min(span, height) < 1.1
    if axisymmetric:
        # `nose_radius` fits a curve over a window near the tip and so returns an
        # upper bound rather than the true tip radius; named for what it is, as the
        # bundled reference vehicle's manifest already does.
        for name, call in (
            ("nose_exponent", mesh.nose_exponent),
            ("nose_radius_bound_m", mesh.nose_radius),
        ):
            try:
                value = float(call())
            except Exception:  # pragma: no cover - shape-dependent
                continue
            if np.isfinite(value):
                out[name] = value
    else:
        out["nose_descriptors"] = (
            f"omitted: cross-section is not axisymmetric (span/height "
            f"{max(span, height) / max(min(span, height), 1e-12):.2f}), so a "
            "radius-against-station fit has no meaning"
        )
    return out


def export_master(
    mesh: Any,
    directory: str | Path,
    *,
    name: str,
    report: ConditionReport | None = None,
    source: str | Path | None = None,
    published: dict[str, Any] | None = None,
    description: str = "",
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a corrected master model and its manifest.

    The STL goes to ``<directory>/<name>.stl`` and the manifest to
    ``<directory>/manifest.json``, carrying the source file's hash, the defects
    repaired, the transforms applied and the measured geometry. Re-exporting a
    second body into the same directory adds it to the manifest's component list
    rather than replacing it, so a multi-configuration vehicle accumulates.
    """
    from aether.geometry.mesh import write_stl

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    stl_path = out / f"{name}.stl"
    write_stl(mesh, stl_path, name=name)

    manifest_path = out / "manifest.json"
    manifest: dict[str, Any] = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {"schema": _SCHEMA, "name": description or name, "components": []}
    )
    manifest.setdefault("components", [])
    manifest["schema"] = _SCHEMA
    if description:
        manifest["name"] = description
    manifest["generated_utc"] = datetime.now(UTC).isoformat(timespec="seconds")

    if source is not None:
        manifest["source_mesh"] = {
            "file": str(source),
            "sha256": sha256_of(source),
            "faces_raw": report.faces_raw if report else None,
            "faces_degenerate_dropped": report.faces_degenerate_dropped if report else None,
            "defects_repaired": list(report.defects_repaired) if report else [],
        }
    if report is not None:
        manifest["conditioning"] = {
            "transforms": list(report.transforms),
            "scale_factor": float(report.scale_factor),
            "interior_faces_removed": int(report.interior_faces_removed),
        }
    if published:
        manifest["published"] = published
    if extra:
        manifest.update(extra)

    component = {"kind": "configuration", "name": name, "file": stl_path.name, **measure(mesh)}
    manifest["components"] = [c for c in manifest["components"] if c.get("name") != name] + [
        component
    ]
    manifest["measured"] = measure(mesh)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return stl_path
