"""``python -m aether.cfd.database`` -- build an aero table for a reference body.

The envelope sweep as a job you can start, stop, look at, and resume, because
at CFD cost per point that is the only way it is actually run. Everything
durable lives in the checkpoint beside the cases, so the process holding it
open is not where the results are.

Two knobs decide the wall time and they multiply:

``--ranks``
    MPI ranks per point. Measured on the 417k-cell biconic, one point scales
    about 5x out to 8-12 ranks and gives it back beyond that.
``--workers``
    Points solved at once. Bounded above by the number of distinct meshes the
    Mach grid needs -- points sharing a mesh must not run concurrently, and
    :func:`~aether.cfd.sweep.sweep_cfd_table` groups them so they don't.

``ranks * workers`` should sit at or under the core count.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from aether.cfd.config import Numerics
from aether.cfd.geometries import REFERENCE_BODIES, build_surface, reference_frame
from aether.cfd.solver import mesh_key
from aether.cfd.sweep import ENVELOPE_ALPHA_DEG, ENVELOPE_MACH, cfd_grid, sweep_cfd_table
from aether.paths import cfd_results


def _report(state: dict[str, Any]) -> None:
    """One line per progress callback, on stderr so a redirect keeps the table."""
    eta = state["eta"]
    stamp = (
        "--:--" if not np.isfinite(eta) else f"{int(eta) // 3600:d}h{int(eta) % 3600 // 60:02d}m"
    )
    print(
        f"  {state['completed']:4d}/{state['total']:<4d} points"
        f"  {state['elapsed'] / 60:7.1f} min elapsed"
        f"  {state['rate'] * 3600:5.1f} pts/h"
        f"  eta {stamp}",
        file=sys.stderr,
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--body", choices=sorted(REFERENCE_BODIES), default="sphere-cone")
    parser.add_argument("--store", type=Path, default=None,
                        help="default: $AETHER_RESULTS/cfd/database")
    parser.add_argument("--altitude", type=float, default=45.0, help="km")
    parser.add_argument("--ranks", type=int, default=8, help="MPI ranks per point")
    parser.add_argument("--workers", type=int, default=1, help="points solved at once")
    parser.add_argument("--iterations", type=int, default=6000)
    parser.add_argument(
        "--mach", type=float, nargs="+", default=None, help=f"default {ENVELOPE_MACH}"
    )
    parser.add_argument(
        "--alpha", type=float, nargs="+", default=None, help=f"deg, default {ENVELOPE_ALPHA_DEG}"
    )
    parser.add_argument("--max-points", type=int, default=None, help="stop after N new points")
    parser.add_argument("--hours", type=float, default=None, help="stop after this long")
    parser.add_argument(
        "--lenient",
        action="store_true",
        help="record a non-converged point as nan instead of stopping",
    )
    parser.add_argument("--plan", action="store_true", help="print the plan and exit")
    args = parser.parse_args(argv)

    grid = cfd_grid(
        mach=tuple(args.mach) if args.mach else ENVELOPE_MACH,
        alpha_deg=tuple(args.alpha) if args.alpha else ENVELOPE_ALPHA_DEG,
    )
    spec = REFERENCE_BODIES[args.body]
    geometry = spec.defaults()
    surface = build_surface(args.body, **geometry)
    frame = reference_frame(args.body, **geometry)

    # Both reference RV shapes have a blunt base, and an Euler base pressure is
    # not a physical quantity: folded into one coefficient it inverted the sign
    # of the reported drag on a Mach 8 sphere-cone. Split unconditionally here
    # rather than left to a flag a sweep would be run once without.
    store = (Path(args.store) if args.store else cfd_results() / "database") / args.body

    meshes = sorted({mesh_key(m) for m in np.asarray(grid.mach)})
    print(
        f"{args.body}: {grid.size} points "
        f"({np.asarray(grid.mach).size} Mach x {np.asarray(grid.alpha).size} alpha)\n"
        f"  {len(meshes)} distinct meshes -> at most {len(meshes)} concurrent workers\n"
        f"  requested {args.workers} workers x {args.ranks} ranks = "
        f"{args.workers * args.ranks} cores\n"
        f"  S_ref {frame.area:.5f} m2  L_ref {frame.length:.4f} m  "
        f"alt {args.altitude:g} km\n"
        f"  checkpoint {store / (args.body + '.jsonl')}",
        file=sys.stderr,
    )
    if args.plan:
        return 0

    started = time.perf_counter()
    table = sweep_cfd_table(
        surface,
        store_dir=store,
        reference=frame,
        altitude=1.0e3 * args.altitude,
        grid=grid,
        name=args.body,
        numerics=Numerics(iterations=args.iterations),
        processes=args.ranks,
        workers=args.workers,
        strict=not args.lenient,
        split_base=True,
        progress=_report,
        max_points=args.max_points,
        time_budget=None if args.hours is None else 3600.0 * args.hours,
        report_every=1,
    )
    print(
        f"\n{args.body}: {'complete' if table.complete else 'PARTIAL'} "
        f"after {(time.perf_counter() - started) / 60:.1f} min",
        file=sys.stderr,
    )
    # A partial table is a normal outcome of a sliced run, not a failure, but
    # it must not be reported as success to a script that would then read it.
    return 0 if table.complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
