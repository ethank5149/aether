"""Comprehensive dense-measurement tracking benchmark.

Explores how close the framework can get to sub-100m CEP by combining:
  - Dense terminal measurements (transponder at ~1 Hz)
  - Higher relaxation orders (d=4,5,6,7)
  - SCS vs Clarabel backends
  - Agnostic vs guided control

Each result is persisted to disk immediately on completion so that
partial results survive process interruption.

Usage:
    python benchmarks/dense_tracking.py
"""
import sys
import os
import time
import json
import traceback
from pathlib import Path
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from aether.certification.occupation import (
    CorridorParams,
    GuidanceLaw,
    SensorMeasurement,
    track_footprint,
    corridor_moment_bound,
)
from aether.certification.sensors import (
    RadarObservation,
    SatelliteIRObservation,
    TransponderObservation,
    fuse_observations,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "dense_tracking"

H = 8500.0

CORRIDOR = CorridorParams(
    speed_high_ms=7387.0,
    speed_low_ms=304.8,
    altitude_ceiling_m=50e3,
    altitude_floor_m=20e3,
    ballistic_coefficient=281.0,
    lift_to_drag=0.30,
    q_max_pa=200e3,
    g_load_max=50.0,
)

V0, h0, gamma0 = 7000.0, 48e3, np.radians(-5.0)
y0 = np.exp(-h0 / (2 * H))
s0, c0 = np.sin(gamma0), np.cos(gamma0)
ENTRY_STATE = (V0, y0, s0, c0)
ENTRY_HW = (100.0, y0 * 2000.0 / (2.0 * H), 0.0, 0.0)

K_V = -0.3 / (CORRIDOR.speed_high_ms - CORRIDOR.speed_low_ms)
K_s = 0.5
V_ref = (CORRIDOR.speed_high_ms + CORRIDOR.speed_low_ms) / 2
GUIDANCE_LAW = GuidanceLaw(control_poly={
    (0, 0, 0, 0): -K_V * V_ref,
    (1, 0, 0, 0): K_V,
    (0, 0, 1, 0): K_s,
})


def _build_measurements(density="sparse"):
    measurements = []

    # Phase 1: Pre-blackout radar + IR (V~5000, h~45km)
    measurements.append(fuse_observations([
        RadarObservation(
            range_m=320e3, elevation_rad=np.radians(8.1),
            range_unc_m=100.0, elevation_unc_rad=np.radians(0.1),
            range_rate_ms=-4800.0, range_rate_unc_ms=3.0,
            look_angle_from_velocity_rad=np.radians(10),
        ),
        SatelliteIRObservation(altitude_m=45000, altitude_unc_m=5000),
    ]))

    # Phase 2: RF blackout — IR only (V~3500, h~40km)
    measurements.append(fuse_observations([
        SatelliteIRObservation(altitude_m=40000, altitude_unc_m=5000),
    ]))

    # Phase 3: Post-blackout radar + IR (V~2500, h~30km)
    measurements.append(fuse_observations([
        RadarObservation(
            range_m=180e3, elevation_rad=np.radians(9.6),
            range_unc_m=80.0, elevation_unc_rad=np.radians(0.08),
            range_rate_ms=-2400.0, range_rate_unc_ms=2.0,
            look_angle_from_velocity_rad=np.radians(12),
        ),
        SatelliteIRObservation(altitude_m=30000, altitude_unc_m=5000),
    ]))

    # Phase 4: Terminal acquisition — radar + transponder (V~800, h~22km)
    measurements.append(fuse_observations([
        RadarObservation(
            range_m=80e3, elevation_rad=np.radians(16),
            range_unc_m=30.0, elevation_unc_rad=np.radians(0.05),
            range_rate_ms=-780.0, range_rate_unc_ms=1.0,
            look_angle_from_velocity_rad=np.radians(8),
        ),
        TransponderObservation(
            speed_ms=800.0, altitude_m=22000.0,
            speed_unc_ms=1.0, altitude_unc_m=50.0,
            fpa_rad=np.radians(-10),
        ),
    ]))

    if density == "sparse":
        return measurements

    speed_points = {
        "medium": np.arange(750, 304, -50),
        "dense":  np.arange(780, 304, -25),
        "ultra":  np.arange(790, 304, -15),
    }[density]

    for V_meas in speed_points:
        frac = (800 - V_meas) / (800 - 305)
        h_meas = 22000 - frac * 2000
        fpa_meas = np.radians(-10 - frac * 5)

        measurements.append(TransponderObservation(
            speed_ms=float(V_meas),
            altitude_m=float(h_meas),
            speed_unc_ms=1.0,
            altitude_unc_m=50.0,
            fpa_rad=float(fpa_meas),
        ).to_measurement())

    return measurements


def _run_config(args):
    config_name, density, order, backend, use_guidance = args
    out_file = RESULTS_DIR / f"{config_name}.json"

    if out_file.exists():
        print(f"  SKIP {config_name} (already completed)", flush=True)
        with open(out_file) as f:
            return json.load(f)

    print(f"  START {config_name}", flush=True)
    t0 = time.time()
    try:
        measurements = _build_measurements(density)
        guidance = GUIDANCE_LAW if use_guidance else None

        result = track_footprint(
            corridor=CORRIDOR,
            entry_state=ENTRY_STATE,
            entry_half_widths=ENTRY_HW,
            measurements=measurements,
            order=order,
            backend=backend,
            tau_multiplier=5.0,
            guidance_law=guidance,
        )
        elapsed = time.time() - t0

        seg_bounds = [
            {"idx": i, "bound_m": seg.bound.max_downrange_m,
             "status": seg.bound.sdp_status,
             "v_hi": seg.segment_params.speed_high_ms}
            for i, seg in enumerate(result.segments)
        ]

        out = {
            "config": config_name,
            "density": density,
            "order": order,
            "backend": backend,
            "guided": use_guidance,
            "n_measurements": len(measurements),
            "n_segments": len(result.segments),
            "terminal_m": result.terminal_footprint_m,
            "elapsed_s": elapsed,
            "segments": seg_bounds,
            "error": None,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    except Exception as e:
        elapsed = time.time() - t0
        out = {
            "config": config_name,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "elapsed_s": elapsed,
            "terminal_m": None,
            "density": density,
            "order": order,
            "backend": backend,
            "guided": use_guidance,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    with open(out_file, "w") as f:
        json.dump(out, f, indent=2, default=str)
    term = out.get("terminal_m")
    tag = "ERROR" if out["error"] else f"{term:.0f} m"
    print(f"  DONE  {config_name}  ({tag}  {elapsed:.0f}s)", flush=True)
    return out


def _collect_results():
    results = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        if p.name == "summary.json":
            continue
        with open(p) as f:
            results.append(json.load(f))
    return results


def _print_report(results):
    results_ok = [r for r in results if r.get("error") is None]
    results_err = [r for r in results if r.get("error") is not None]
    results_ok.sort(key=lambda r: r["terminal_m"])

    print(f"\n{'Config':<35s}  {'Terminal':>10s}  {'Segs':>5s}  {'Time':>8s}  {'Status'}")
    print("-" * 80)
    for r in results_ok:
        t = r["terminal_m"]
        t_str = f"{t:.0f} m" if t < 1000 else f"{t/1e3:.1f} km"
        last = r["segments"][-1] if r.get("segments") else {}
        status = last.get("status", "?")
        print(f"{r['config']:<35s}  {t_str:>10s}  {r.get('n_segments','?'):>5}  "
              f"{r['elapsed_s']:>7.0f}s  {status}")

    if results_err:
        print(f"\n--- ERRORS ({len(results_err)}) ---")
        for r in results_err:
            print(f"  {r['config']}: {r['error']}")

    print("\n" + "=" * 90)
    print("SEGMENT BREAKDOWN (top 5)")
    print("=" * 90)
    for r in results_ok[:5]:
        t = r["terminal_m"]
        t_str = f"{t:.0f} m" if t < 1000 else f"{t/1e3:.1f} km"
        print(f"\n{r['config']}  (terminal: {t_str}, {r['elapsed_s']:.0f}s)")
        for seg in (r.get("segments") or []):
            label = "entry" if seg["idx"] == 0 else f"meas{seg['idx']:02d}"
            b = seg["bound_m"]
            b_str = f"{b:.0f} m" if b < 1000 else f"{b/1e3:.1f} km"
            print(f"    {label}: {b_str:>10s}  V_hi={seg['v_hi']:.0f}  [{seg['status']}]")

    print("\n" + "=" * 90)
    print("SUMMARY: Terminal footprint by configuration")
    print("=" * 90)
    for guided in [False, True]:
        g_label = "GUIDED" if guided else "AGNOSTIC"
        print(f"\n--- {g_label} ---")
        print(f"{'':>16s}", end="")
        for density in ["sparse", "medium", "dense", "ultra"]:
            print(f"  {density:>10s}", end="")
        print()
        for order in [4, 5, 6, 7]:
            for backend in ["scs", "clarabel"]:
                label = f"d={order} {backend}"
                print(f"{label:>16s}", end="")
                for density in ["sparse", "medium", "dense", "ultra"]:
                    match = [r for r in results_ok
                             if r["order"] == order and r["backend"] == backend
                             and r["density"] == density and r["guided"] == guided]
                    if match:
                        t = match[0]["terminal_m"]
                        if t < 1000:
                            print(f"  {t:>8.0f} m", end="")
                        else:
                            print(f"  {t/1e3:>7.1f} km", end="")
                    else:
                        print(f"  {'—':>10s}", end="")
                print()


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("Dense-Measurement Tracking Benchmark")
    print(f"Hardware: {cpu_count()} cores, "
          f"{os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1e9:.0f} GB RAM")
    print(f"Results: {RESULTS_DIR}")
    print("=" * 90)

    configs = []

    for density in ["sparse", "medium", "dense"]:
        for use_guidance in [False, True]:
            g = "guided" if use_guidance else "agnostic"
            configs.append((f"d4_scs_{density}_{g}", density, 4, "scs", use_guidance))

    for density in ["sparse", "medium", "dense"]:
        for use_guidance in [False, True]:
            g = "guided" if use_guidance else "agnostic"
            configs.append((f"d4_clarabel_{density}_{g}", density, 4, "clarabel", use_guidance))

    for order in [5, 6]:
        for backend in ["scs", "clarabel"]:
            for use_guidance in [False, True]:
                g = "guided" if use_guidance else "agnostic"
                configs.append((f"d{order}_{backend}_dense_{g}", "dense", order, backend, use_guidance))

    configs.append(("d5_scs_ultra_guided", "ultra", 5, "scs", True))
    configs.append(("d5_clarabel_ultra_guided", "ultra", 5, "clarabel", True))
    configs.append(("d7_scs_dense_guided", "dense", 7, "scs", True))

    already = {p.stem for p in RESULTS_DIR.glob("*.json") if p.stem != "summary"}
    remaining = [c for c in configs if c[0] not in already]

    print(f"\nTotal configurations: {len(configs)}")
    print(f"Already completed: {len(already)}")
    print(f"Remaining: {len(remaining)}")

    if remaining:
        print(f"\n{'Config':<35s}  {'Density':<8s}  {'d':>2s}  {'Backend':<9s}  {'Guided':<7s}")
        print("-" * 70)
        for name, density, order, backend, guided in remaining:
            print(f"{name:<35s}  {density:<8s}  {order:>2d}  {backend:<9s}  "
                  f"{'yes' if guided else 'no':<7s}")

        n_workers = max(1, min(len(remaining), cpu_count() // 2))
        print(f"\nUsing {n_workers} parallel workers")
        print(f"Starting at {time.strftime('%H:%M:%S')}...")
        sys.stdout.flush()

        t_total = time.time()
        with Pool(n_workers) as pool:
            pool.map(_run_config, remaining)
        t_total = time.time() - t_total
        print(f"\nBatch wall time: {t_total/60:.1f} min")
    else:
        print("\nAll configs already completed.")

    results = _collect_results()

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 90)
    print("RESULTS")
    print("=" * 90)
    _print_report(results)
    print(f"\nResults saved to {RESULTS_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
