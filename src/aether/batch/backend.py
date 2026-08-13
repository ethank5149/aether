"""Array-backend selection: NumPy (CPU) or CuPy (CUDA).

The batched integrator is written against the NumPy API surface shared
by CuPy, so the same code runs on either device; the backend is an
explicit argument, never an ambient global. Requesting ``"cupy"``
without a working CUDA runtime raises immediately rather than silently
falling back — a silent fallback would invalidate any V8 throughput
number recorded against it.
"""

from __future__ import annotations

from types import ModuleType
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

__all__ = ["Backend", "cuda_available", "get_array_module", "to_numpy"]

Backend = Literal["numpy", "cupy"]


def cuda_available() -> bool:
    """True if CuPy imports, a device answers, **and a kernel actually runs**.

    The last clause is not paranoia. A machine can have a driver and a
    visible GPU while CuPy still cannot compile anything, because its JIT
    needs CUDA toolkit *headers* that ship separately from the driver:
    ``getDeviceCount()`` returns 1 and the first elementwise operation then
    raises ``Failed to find CUDA headers``. Reporting availability from the
    device count alone therefore hands callers a backend that fails on
    first use, which is worse than reporting it unavailable.

    A one-element reduction is enough to force a kernel launch, and it
    costs a few milliseconds once.
    """
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() < 1:
            return False
        return bool(float(cupy.asarray([1.0, 2.0]).sum()) == 3.0)
    except Exception:
        return False


def get_array_module(backend: Backend) -> ModuleType:
    """The array module implementing the requested backend."""
    if backend == "numpy":
        return np
    if backend == "cupy":
        try:
            import cupy
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "backend 'cupy' requested but cupy is not installed"
            ) from exc
        if cupy.cuda.runtime.getDeviceCount() < 1:  # pragma: no cover
            raise RuntimeError("backend 'cupy' requested but no CUDA device is present")
        return cast(ModuleType, cupy)
    raise ValueError(f"backend must be 'numpy' or 'cupy', got {backend!r}")


def to_numpy(array: object) -> NDArray[np.float64]:
    """Bring a backend array to host memory as float64."""
    if hasattr(array, "get"):  # cupy.ndarray
        return np.asarray(array.get(), dtype=np.float64)
    return np.asarray(array, dtype=np.float64)
