# AETHER: Aero-thermo-Elastic Trajectory & Hypersonic Estimation Research

## Manuscript program

Three papers. Each consumes its predecessors as a black box.

| # | Title | Depends on | Venue target |
|---|---|---|---|
| **1** | Certified Analytical Bounds for Multi-Phase Hypersonic Reachable Sets | — | SICON / ESAIM:COCV / Automatica |
| 2 | Contraction of Certified Reachable Sets under Sequential Set-Membership Observations | Paper 1 | IEEE TAC / Automatica |
| 3 | Probabilistic Reachable Sets under Process and Measurement Noise | Papers 1 and 2 | SICON / IEEE TAC |

`manuscript/legacy/` (fixed-grid spectral moving-boundary) is **not** a fourth
paper. It is preliminary machinery, summarized in Paper 1 §1.2 and lifted into
its spectral appendix.

### Paper 1 — contributions

- **C1 — Embedding.** Hybrid multi-phase occupation-measure formulation: per-mode
  measures plus guard measures coupled by linear equality constraints. Everything
  stays linear in the measures. Replaces the Heaviside/Dirac-delta route, which is
  formally undefined.
- **C2 — Decomposition (headline).** In the thin-atmosphere limit the
  occupation-measure family *loses compactness in a structured way*: a
  finite-dimensional skeleton carried by the equilibrium-glide manifold, one
  rescaled boundary-layer profile per atmospheric skip, and a residual admitting a
  quantitative bound.
- **C3 — Analytical certification.** A comparison certificate *derived* from C2
  rather than searched for numerically, discharged in exact rational arithmetic
  over the entry corridor. No floating point enters the guarantee.

### Two commitments that shape everything

**The bound is analytical, in the strong sense.** Derived by hand, then verified
in exact arithmetic — the computer *checks*, it never *chooses*. A polynomial
certificate whose coefficients exist only because an interior-point solver found
them has analytical form but inherits the solver's accuracy. Moment-SOS appears
only as an independent numerical cross-check (Paper 1 §6.3), never in the
guarantee.

**The hypersonic constraints are load-bearing mathematically, not just
physically.** Two mechanisms:

- corridor constraints (heating rate, dynamic pressure, load factor, integrated
  heat load) bound the **state set** — this is where the compactness every result
  assumes comes from, and it is the Archimedean hypothesis supplied physically
  rather than assumed;
- constitutive uncertainty (real-gas, rarefaction, coefficient-map fit error)
  bounds the **vector field**, as certified envelopes — a differential inclusion
  whose bound holds for every selection.

**The vehicle is a rigid body, not a point mass.** Attitude quaternion and body
rates are states; angle of attack and sideslip are *outputs* of the attitude
solution; controls are flap deflections and RCS moments. Thirteen states before
augmentation, more once recession depth and modal amplitudes are carried — which
they are, because the aerothermoelastic loop (deformation → incidence → moments →
attitude → incidence) only closes in 6-DOF. A 3-DOF model would assume that loop
open, assume trim is always solvable, and hard-code two fast subsystems onto their
slow manifolds.

That choice also gives the decomposition more to do: the skeleton becomes the
intersection of **three** slow manifolds — quasi-equilibrium glide, aerodynamic
trim, and quasi-static deflection — rather than one. Each contributes a
normal-hyperbolicity condition that must be checked rather than assumed.

### Status

Nothing below is proved or computed yet. This table is the gate: no claim enters a
manuscript until its row moves.

| Claim | Proved? | Computed? | Blocker |
|---|---|---|---|
| C1 hybrid embedding is a valid relaxation | no | n/a | measure identity not written; check prior art (`claeys2016modal`, `shia2014convex`) |
| C2 measure family loses compactness as ε → 0 | **no** | no | critical path; mechanism identified, unproved |
| C2 residual bound is *quantitative* at physical ε | **no** | no | §5 cannot be written without it |
| C2 reduces the dimension the certificate is posed on | **no** | no | load-bearing — without it C2 is decoration |
| C3 certificate derived from the decomposition | **no** | n/a | depends on the row above |
| C3 discharged in exact arithmetic | no | no | verification script does not exist |
| Corridor closes a compact set (Archimedean) | no | n/a | what bounds altitude from *above* on a word with a coast? |
| Envelopes are certified, not fitted | no | no | each needs a stated reason the truth cannot leave it |
| Bound encloses HJ ground truth, planar case | n/a | no | no HJ reference solver in repo |
| Distinguished limit: ordering of ε_atm, ε_att, ε_ela | no | n/a | reduced model depends on it; needs measured modal frequencies |
| Trim manifold is normally hyperbolic (attracting) | no | n/a | not automatic — a statically unstable configuration has none |
| Moment-coefficient envelopes are sourceable | no | no | to be generated from vehicle mesh in `aether-gambit`; needs V&V-based error bounding, not the raw CFD number |
| Worked example uses a publishable geometry | n/a | no | otherwise the reproducibility rule is a fiction |

**Cheapest thing that can invalidate the plan:** check whether matched-asymptotic
entry analysis (Vinh–Busemann–Culp; Loh's second-order theory) already contains the
decomposition in non-measure-theoretic language. Do this before anything else.

**Open question worth resolving early:** Lions-type concentration-compactness treats
loss of compactness in critical Sobolev embeddings, where the invariance is a
dilation group on function space. What Paper 1 has is a singular limit on a compact
phase space, mass concentrating onto an attracting slow manifold — that is Fenichel
theory, and unlike Lions it gives quantitative ε-explicit estimates, which is
exactly what C3 needs.

### Reading

[`manuscript/shared.bib`](manuscript/shared.bib) holds ~160 entries in 21 groups,
with tier tags and a **reading order in the file header** (Rounds 0–5, each with a
gate). Bibliographic details there are recorded to the best of available knowledge
and **must be verified against the published record before submission** — 74
entries carry an explicit `VERIFY` note.

Per-section planning, open gaps and decisions live as `%` comments in the section
files themselves — grep for `[BODY]`, `\gap`, `\decide`, `\risk`, `\idea`,
`[CHECK]`.

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
