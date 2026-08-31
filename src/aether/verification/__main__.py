"""Run all public verification tasks: ``python -m aether.verification``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aether.verification.p2v1_ultraspherical import run_p2v1
from aether.verification.p2v123_plates import run_p2v123
from aether.verification.r1v1_timescales import run_r1v1
from aether.verification.r1v2_attitude_gap import run_r1v2
from aether.verification.r1v3_impulse_rank import run_r1v3
from aether.verification.r1v4_residual_constant import run_r1v4
from aether.verification.r1v5_augmented_field import run_r1v5
from aether.verification.v1_structural import run_v1
from aether.verification.v2_slosh import run_v2
from aether.verification.v3_integrators import run_v3
from aether.verification.v4_thermal import run_v4
from aether.verification.v8_throughput import run_v8


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the public numerics verification tasks")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    all_passed = True
    for runner, stem in (
        (run_v1, "v1-structural"),
        (run_v2, "v2-slosh"),
        (run_v3, "v3-integrators"),
        (run_v4, "v4-thermal"),
        (run_v8, "v8-throughput"),
        (run_p2v1, "p2v1-ultraspherical"),
        (run_p2v123, "p2v123-plates"),
        (run_r1v1, "r1v1-timescales"),
        (run_r1v2, "r1v2-attitude-gap"),
        (run_r1v3, "r1v3-impulse-rank"),
        (run_r1v4, "r1v4-residual-constant"),
        (run_r1v5, "r1v5-augmented-field"),
    ):
        report = runner(args.output)
        path = report.write(args.output, stem)
        print(f"{report.task_id} {'PASS' if report.passed else 'FAIL'} -> {path}")
        all_passed = all_passed and report.passed
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
