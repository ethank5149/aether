"""Standalone kernel launch profiled by Nsight Compute for I-V8.

Kept as a separate script because ``ncu`` profiles a *process*: the
batched stage kernel has to be the only significant launch in it.
"""

from __future__ import annotations


def main() -> None:
    import cupy

    kernel = cupy.RawKernel(
        r"""
extern "C" __global__ void rk4_stage(const double* y, const double* beta,
                                     double* out, int n_state, int n_rep) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_state * n_rep) {
        int rep = i / n_state;
        out[i] = y[i] * 2.0 - y[i] * y[i] / beta[rep];
    }
}""",
        "rk4_stage",
    )
    n_state, n_rep = 6, 65536
    y = cupy.ones(n_state * n_rep)
    beta = cupy.full(n_rep, 8000.0)
    out = cupy.empty_like(y)
    threads = 256
    blocks = (y.size + threads - 1) // threads
    for _ in range(3):
        kernel((blocks,), (threads,), (y, beta, out, n_state, n_rep))
    cupy.cuda.Stream.null.synchronize()


if __name__ == "__main__":
    main()
