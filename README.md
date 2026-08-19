# AETHER: Aero-thermo-Elastic Trajectory & Hypersonic Estimation Research

## Manuscript: Certified Analytical Bounds for Multi-Phase Hypersonic Reachability
### Goals:
- Develop a concentration-compactness framework for the reachable sets of multi-phase atmospheric entry trajectories, spanning equilibrium glides, Sänger-type skip profiles, and fractional orbits with synergetic transfers.
- Embed the hybrid non-linear dynamics into an infinite-dimensional Banach space of occupation measures to obtain a single linear Liouville partial differential equation.
- Execute a Galerkin projection to collapse this formulation into a finite sequence of Riccati ordinary differential equations for baseline covariance, Deploy Sum-of-Squares (SOS) optimization to construct polynomial Barrier Certificates for tight, non-ellipsoidal bounding.

This deterministic profile decomposition should reduce the high-dimensional reachability problem to a finite-dimensional skeleton of concentrating trajectories plus a compact residual.

## Package layout
| Module | Contents |
|---|---|
| `spectral/` | Chebyshev–Gauss–Lobatto operators by direct recurrence with the negative-sum trick; Clenshaw–Curtis quadrature; barycentric interpolation |
| `ultraspherical/` | Olver–Townsend banded operators and assembly |
| `thermal/` | Landau immobilization frame; semi-discrete charring/Stefan solver on the fixed grid |
| `structures/` | Variable-rigidity Euler–Bernoulli operator; free-free BCs by null-space projection; modal reduction; integrators |
| `plates/` | Mindlin plates, laminate stiffness |
| `coupling/` | Quadrature-normalized kernel force transfer |
| `batch/` | NumPy/CuPy backend abstraction; batched common-outer-grid integrator; sampling; occupancy model |
| `atmosphere/` | US Standard 1976, MSIS thermosphere, ERA5 reanalysis winds |
| `dynamics/` | Quaternion attitude kinematics, deformed-surface incidence |
| `verification/` | Executable verification tasks |

## Setup

```bash
pip install -e .[dev]
pip install -e .[atmosphere]   # optional: MSIS thermosphere
pip install -e .[cuda]         # optional: GPU batch backend
```

## Running

```bash
pytest tests -q                 # 331 tests
python -m aether.verification   # verification tasks -> results/
```

## License

MIT. See [`LICENSE`](LICENSE).
