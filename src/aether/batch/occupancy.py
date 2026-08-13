"""CUDA occupancy model for the batch layer (Paper I, §5.2 / V8).

Paper I's V8 asks for *achieved* occupancy alongside throughput. Achieved
occupancy is a hardware counter and needs a profiler (Nsight Compute),
which is not available in this environment. What *is* computable, exactly
and without a profiler, is **theoretical occupancy**: the standard CUDA
occupancy model, evaluated from the compiled kernel's register and
shared-memory footprint together with the device's per-SM limits.

That is a real number, not an estimate — it is the same arithmetic the
CUDA Occupancy Calculator performs — and it bounds achieved occupancy
from above. Reporting it, together with the throughput-saturation curve
that is achieved occupancy's externally observable consequence, closes
most of what V8 asks for; the residual gap is stated rather than papered
over.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "AchievedOccupancy",
    "OccupancyReport",
    "achieved_occupancy",
    "device_limits",
    "theoretical_occupancy",
]


@dataclass(frozen=True)
class OccupancyReport:
    """Theoretical occupancy of one kernel at one block size."""

    threads_per_block: int
    registers_per_thread: int
    shared_bytes_per_block: int
    active_blocks_per_sm: int
    active_warps_per_sm: int
    max_warps_per_sm: int
    limiter: str
    """Which resource binds: ``"registers"``, ``"shared_memory"``,
    ``"warps"`` or ``"blocks"``."""

    @property
    def occupancy(self) -> float:
        """Active warps per SM divided by the architectural maximum."""
        return self.active_warps_per_sm / self.max_warps_per_sm


def device_limits(device: int = 0) -> dict[str, int]:
    """Per-SM architectural limits of a CUDA device."""
    import cupy.cuda.runtime as runtime

    props = runtime.getDeviceProperties(device)
    return {
        "multiprocessors": int(props["multiProcessorCount"]),
        "max_threads_per_sm": int(props["maxThreadsPerMultiProcessor"]),
        "registers_per_sm": int(props["regsPerMultiprocessor"]),
        "shared_bytes_per_sm": int(props["sharedMemPerMultiprocessor"]),
        "warp_size": int(props["warpSize"]),
        "max_threads_per_block": int(props["maxThreadsPerBlock"]),
        "compute_capability_major": int(props["major"]),
        "compute_capability_minor": int(props["minor"]),
    }


def theoretical_occupancy(
    kernel: Any,
    threads_per_block: int,
    device: int = 0,
    max_blocks_per_sm: int = 16,
    register_allocation_unit: int = 256,
) -> OccupancyReport:
    """Theoretical occupancy from the compiled kernel's resource use.

    Parameters
    ----------
    kernel:
        A compiled ``cupy.RawKernel`` (its ``attributes`` supply the
        register and shared-memory footprint).
    threads_per_block:
        Launch block size.
    max_blocks_per_sm:
        Architectural resident-block limit per SM (16 on Ampere).
    register_allocation_unit:
        Register file allocation granularity, in registers per warp.
    """
    if threads_per_block < 1:
        raise ValueError(f"threads_per_block must be >= 1, got {threads_per_block}")
    limits = device_limits(device)
    warp = limits["warp_size"]
    if threads_per_block > limits["max_threads_per_block"]:
        raise ValueError(
            f"threads_per_block {threads_per_block} exceeds the device limit "
            f"{limits['max_threads_per_block']}"
        )

    attributes = kernel.attributes
    regs = int(attributes["num_regs"])
    shared = int(attributes["shared_size_bytes"])

    warps_per_block = -(-threads_per_block // warp)  # ceil
    max_warps = limits["max_threads_per_sm"] // warp

    # registers are allocated per warp, rounded to the allocation unit
    regs_per_warp = regs * warp
    rounded = -(-regs_per_warp // register_allocation_unit) * register_allocation_unit
    by_registers = (
        limits["registers_per_sm"] // (rounded * warps_per_block)
        if rounded > 0
        else max_blocks_per_sm
    )
    by_shared = (
        limits["shared_bytes_per_sm"] // shared if shared > 0 else max_blocks_per_sm
    )
    by_warps = max_warps // warps_per_block
    candidates = {
        "registers": by_registers,
        "shared_memory": by_shared,
        "warps": by_warps,
        "blocks": max_blocks_per_sm,
    }
    active_blocks = min(candidates.values())
    limiter = min(candidates, key=lambda k: candidates[k])
    return OccupancyReport(
        threads_per_block=threads_per_block,
        registers_per_thread=regs,
        shared_bytes_per_block=shared,
        active_blocks_per_sm=int(active_blocks),
        active_warps_per_sm=int(active_blocks * warps_per_block),
        max_warps_per_sm=int(max_warps),
        limiter=limiter,
    )


@dataclass(frozen=True)
class AchievedOccupancy:
    """Profiler-measured occupancy, or the reason it is unavailable."""

    available: bool
    achieved: float | None = None
    """``sm__warps_active.avg.pct_of_peak_sustained_active`` as a fraction."""
    kernel: str | None = None
    launches: int = 0
    """Number of matching launches averaged."""
    reason: str | None = None
    """Why the measurement is unavailable, when it is."""


def achieved_occupancy(
    script: str | Path,
    python_executable: str | None = None,
    ncu_executable: str = "ncu",
    timeout: float = 600.0,
    kernel_name: str | None = None,
) -> AchievedOccupancy:
    """Measure achieved occupancy by running ``script`` under Nsight Compute.

    Achieved occupancy is a hardware counter — the time-averaged ratio of
    resident warps to the architectural maximum — and unlike
    :func:`theoretical_occupancy` it reflects what actually happened,
    including tail effects and load imbalance.

    Counter access is gated by the NVIDIA driver: unless the module is
    loaded with ``NVreg_RestrictProfilingToAdminUsers=0``, a non-root
    user gets ``ERR_NVGPUCTRPERM`` and no counters at all. That is a
    host-level security setting, so this function reports the condition
    rather than attempting to change it.

    Parameters
    ----------
    kernel_name:
        Select this kernel from the profile. Required in practice: a
        CuPy process launches its own fill and copy kernels alongside
        the one under study, and averaging over them would report the
        occupancy of the setup rather than of the workload. When several
        launches match, the mean is returned.

    Returns
    -------
    AchievedOccupancy
        ``available=False`` with a ``reason`` when profiling is blocked,
        the profiler is missing, or no kernel was captured.
    """
    import shutil
    import subprocess
    import sys

    path = Path(script)
    if not path.is_file():
        raise FileNotFoundError(f"profiling script not found: {path}")
    resolved = shutil.which(ncu_executable)
    if resolved is None:
        # Nsight ships alongside the interpreter in a conda environment, and
        # invoking that interpreter by absolute path leaves its bin/ off PATH.
        sibling = Path(sys.executable).parent / ncu_executable
        if sibling.is_file():
            resolved = str(sibling)
    if resolved is None:
        return AchievedOccupancy(
            available=False, reason=f"{ncu_executable} not found on PATH"
        )
    metric = "sm__warps_active.avg.pct_of_peak_sustained_active"
    command = [
        resolved, "--metrics", metric, "--csv", "--target-processes", "all",
        python_executable or sys.executable, str(path),
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        return AchievedOccupancy(available=False, reason="profiler timed out")
    output = completed.stdout + completed.stderr
    if "ERR_NVGPUCTRPERM" in output:
        return AchievedOccupancy(
            available=False,
            reason=(
                "ERR_NVGPUCTRPERM: the NVIDIA driver restricts performance "
                "counters to administrators. Load the module with "
                "NVreg_RestrictProfilingToAdminUsers=0 (a host-level change) "
                "to enable it"
            ),
        )
    rows = [
        [field.strip('"') for field in line.split('","')]
        for line in output.splitlines()
        if metric in line and '","' in line
    ]
    if not rows:
        return AchievedOccupancy(
            available=False, reason="no occupancy metric captured from the profile"
        )

    def _value(fields: list[str]) -> float | None:
        for field in reversed(fields):
            try:
                return float(field.replace(",", ""))
            except ValueError:
                continue
        return None

    if kernel_name is not None:
        rows = [r for r in rows if any(kernel_name in field for field in r)]
        if not rows:
            return AchievedOccupancy(
                available=False,
                reason=f"kernel {kernel_name!r} not present in the profile",
            )
    values = [v for v in (_value(r) for r in rows) if v is not None]
    if not values:
        return AchievedOccupancy(available=False, reason="could not parse the profile")
    mean = sum(values) / len(values)
    matched = kernel_name
    if matched is None:
        for field in rows[-1]:
            if field and not field.replace(".", "").replace(",", "").isdigit():
                matched = field
                break
    return AchievedOccupancy(
        available=True, achieved=mean / 100.0, kernel=matched, launches=len(values)
    )
