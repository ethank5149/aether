# AETHER

**Aero-thermo-Elastic Trajectory & Hypersonic Estimation Research**

A fixed-grid spectral method for coupled moving-boundary problems, with
batched uncertainty quantification.

---

## The idea

Moving-boundary problems are usually solved on a mesh that moves with the
front. That costs twice: interpolation error wherever coupled fields
exchange data across a deforming interface, and a remesh per sample, which
is what makes Monte Carlo over such problems expensive.

Immobilize the boundary instead. Under a Landau coordinate transformation
the front becomes stationary, the spatial operator is assembled **once**,
and the same operator then serves every field, every timestep, and every
UQ sample. The third of those is the interesting one: because the operator
no longer depends on the sample, an *M*-sample ensemble collapses to a
single batched tensor solve.

Three contributions, stated narrowly:

1. Boundary immobilization extended from the scalar heat equation to
   **coupled** moving-boundary multiphysics, as a single autonomous ODE
   system.
2. A well-conditioned ultraspherical formulation with an explicit
   conditioning and stiffness bound, and the IMEX splitting it implies.
3. The observation that immobilization is precisely what collapses
   moving-boundary UQ to one batched operator — with a measured speedup
   curve.

Immobilization is Landau (1950) and the ultraspherical basis is
Olver–Townsend; what is new here is the synthesis and its consequences.

## Manuscript

The manuscript is in [`manuscript/`](manuscript/) and is the primary
deliverable of this repository.

```bash
cd manuscript && make        # or ./build.sh
```

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
python tools/check_boundary.py  # scope check, see below
```

## Scope

This package is deliberately limited to numerical-methods and
standard-published-model content. Applied flight-systems capability —
guidance, trajectory optimization, sensing, tracking, and engagement
modelling — is developed separately and is not distributed here.

That boundary is enforced mechanically rather than by convention:
`tools/check_boundary.py` fails the build on any import reaching outside
the public kernel. Run it before publishing, and after moving any module
in either direction.

## License

MIT. See [`LICENSE`](LICENSE).
