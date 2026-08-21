# AETHER: Aero-thermo-Elastic Trajectory & Hypersonic Estimation Research

## Manuscripts

The reachable set of a hypersonic entry vehicle — every state it can be driven to
under admissible control — is what safety-critical and threat analysis actually
need bounded. Computing it exactly means solving a Hamilton–Jacobi–Isaacs equation
on a grid, which is exponential in the state dimension and caps out in practice
around four or five states. A rigid-body entry vehicle has thirteen before any
augmentation.

The approach taken here embeds the hybrid vehicle dynamics into a Banach space of
occupation measures, where the problem becomes *linear* despite the nonlinearity of
the flight mechanics. In the thin-atmosphere limit the resulting measure family is
not compact: mass concentrates onto a low-dimensional set of trajectories. The aim
is to turn that failure of compactness into a profile decomposition — a
finite-dimensional skeleton plus a controlled residual — and to use its structure
to *derive* a comparison certificate rather than search for one numerically, so the
resulting bound is analytical and verifiable in exact arithmetic.

Three papers, in series:

| # | Title | Adds |
|---|---|---|
| 1 | Certified Analytical Bounds for Multi-Phase Hypersonic Reachable Sets | the embedding, the decomposition, the certificate |
| 2 | Certified Contraction of Reachable Sets under Sequential Telemetry | successive measurements tighten the bound |
| 3 | Probabilistic Reachability Bounds for Multi-Phase Entry under Process and Measurement Noise | unbounded noise, probabilistic guarantees |

### Scope

- **Six degrees of freedom.** Attitude and body rates are states; angle of attack
  and sideslip are outputs of the attitude solution, not commanded quantities.
  Controls are flap deflections and RCS moments.
- **Multi-phase.** Vacuum and atmospheric flight, with and without thrust, are
  modes of a hybrid system. Equilibrium glide, skip-glide and fractional-orbit
  profiles are then admissible words over that alphabet rather than separate models.
- **Hypersonic constraints are load-bearing.** Heating rate, dynamic pressure, load
  factor and integrated heat load define the entry corridor, and the corridor is
  what makes the state set compact — a hypothesis every result depends on.
  Real-gas, rarefaction, ablation and aerothermoelastic effects enter as certified
  envelopes on the vector field.

### Status

**Early.** The manuscripts are section structure and a working bibliography.
No derivations are written and no results are computed; nothing in them should
yet be read as a claim.

[`manuscript/shared.bib`](manuscript/shared.bib) collects ~165 references in
twenty-one groups with a reading order in its header. Bibliographic details there
are provisional and flagged for verification against the published record.

### Building

```bash
cd manuscript && ./build.sh          # all three papers
./build.sh paper1                    # one paper
./build.sh watch paper1              # continuous rebuild
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
```

## License

MIT. See [`LICENSE`](LICENSE).
