"""``python -m aether.cfd`` -- mesh a reference body, verbosely.

Meshing is the step most likely to look like a hang: a 400k-element domain is
half a minute of a silent process, and the 3D pass is nearly all of it. From a
shell that is indistinguishable from a wedged job, so this reports by default
and has to be asked for silence.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from aether.aerodynamics.cfd.diagnostics import surface_quality
from aether.aerodynamics.cfd.meshing import console_progress
from aether.cfd.geometries import (
    REFERENCE_BODIES,
    build_domain,
    build_surface,
    reference_frame,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--body", choices=sorted(REFERENCE_BODIES), default="sphere-cone")
    # Not required at parse time so that --list works without one, but
    # required to mesh: the cell sizing is a function of Mach, so meshing
    # one speed and solving another is an error rather than a shortcut.
    parser.add_argument("--mach", type=float, default=None, help="sizing follows it")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true", help="no progress bar")
    parser.add_argument("--no-optimize", action="store_true", help="skip mesh optimisation")
    parser.add_argument(
        "--quality-threshold", type=float, default=0.1, help="count cells below this"
    )
    parser.add_argument("--list", action="store_true", help="name the bodies and stop")
    args = parser.parse_args(argv)

    if args.list:
        for name, body in sorted(REFERENCE_BODIES.items()):
            kind = "axisymmetric" if body.axisymmetric else "lifting"
            print(f"{name:22s} {kind:13s} {body.note}")
        return 0

    if args.mach is None:
        parser.error("--mach is required to mesh: the cell sizing depends on it")

    output = args.output or Path(f"{args.body}-m{args.mach:g}.su2")
    print(f"{args.body} at Mach {args.mach:g} -> {output}", file=sys.stderr)

    # The wall triangulation, before any volume is built around it. A sliver
    # here becomes a bad cell there, and this costs a second against the
    # minutes the domain takes -- so it is worth knowing first.
    wall = surface_quality(build_surface(args.body))
    print(f"surface: {wall.summary()}", file=sys.stderr)

    started = time.monotonic()
    result = build_domain(
        args.body,
        output,
        mach=args.mach,
        progress=None if args.quiet else console_progress(args.body),
        optimize=not args.no_optimize,
        quality_threshold=args.quality_threshold,
    )
    frame = reference_frame(args.body)
    print(
        f"{result.n_nodes} nodes, {result.n_elements} elements in "
        f"{time.monotonic() - started:.1f}s\n"
        f"volume:  {result.quality.summary() if result.quality else 'not measured'}\n"
        f"S_ref {frame.area:.5f} m2   L_ref {frame.length:.4f} m   "
        f"moment x {frame.moment_reference[0]:.4f} m",
        file=sys.stderr,
    )
    if result.quality is not None and not result.quality.usable:
        print(
            "WARNING: inverted cells. SU2 reports these as non-physical points "
            "before its first iteration, not as a meshing problem.",
            file=sys.stderr,
        )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
