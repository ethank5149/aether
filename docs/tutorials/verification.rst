Verification Guide
==================

What Verification Is
--------------------

AETHER's verification suite consists of **executable verification tasks**, each
stating its **failure criterion in advance** (before any results existed) and
writing a timestamped Markdown report plus machine-readable CSV into a results
directory.

The tasks cover the full numerics kernel — from the spectral operators
through the coupled thermal-structural-aerodynamic system — and the certification
layer's rigorous arithmetic.

Task Categories
---------------

.. list-table::
   :header-rows: 1
   :widths: 15 30 55

   * - Category
     - Tasks
     - Description
   * - **V1–V8** (Core)
     - V1, V2, V3, V4, V5, V7, V8
     - Core numerics: structural conditioning, exact force transfer,
       integrator performance, thermal MMS convergence, filter
       diagnostics, dispersion containment, batch throughput.
   * - **P2V\*** (Phase 2)
     - P2V1, P2V4, P2V5, P2V8, P2V67, P2V123
     - Phase 2 verification: ultraspherical conditioning,
       blending functions, coast phase, aerothermal, GNC,
       plate benchmarks.
   * - **R1V\*** (Rigorous)
     - R1V1, R1V2, R1V3, R1V4, R1V5
     - Rigorous verification in exact arithmetic: timescale
       separation, attitude gap, impulse rank, residual constant,
       augmented field.
   * - **Integration**
     - int-coupled
     - Coupled system verification.

The canonical list of all 22 task runners is in
:mod:`aether.verification.__main__`.

Running All Tasks
-----------------

.. code-block:: bash

   make verify

or equivalently:

.. code-block:: bash

   python -m aether.verification [--output RESULTS_DIR]

The default output directory is ``results/`` (created if it doesn't exist).

Output Format
-------------

Each task writes two files into the output directory:

1. **Markdown report** (``{stem}.md``): Human-readable summary stating:
   - Task ID and title
   - Acceptance criterion (stated before the run)
   - Verdict: **PASS** or **FAIL**
   - Environment: NumPy version, SciPy version, Python version, platform
   - Key numerical results

2. **CSV data** (``{stem}.csv``): Machine-readable results for post-processing
   or plotting.

Example report structure:

.. code-block:: markdown

   # V1: Structural Conditioning

   **Criterion**: The reduced stiffness operator condition number κ(K̃)
   must scale as O(N²) vs. O(N⁴) for the unreduced operator, and
   free-free frequencies must match the analytic uniform-beam solution
   within 1×10⁻⁶ relative error.

   **Verdict**: PASS

   **Environment**: numpy=1.26.4, scipy=1.11.4, python=3.12.3, linux

   **Results**:
   - N=16: κ=2.3e2, freq error=3.1e-10
   - N=32: κ=9.1e2, freq error=1.2e-10
   - ...

Task Descriptions
-----------------

The following is adapted from :mod:`aether.verification.__init__` and
:mod:`aether.verification.__main__`:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Task
     - Description
   * - **V1** (:mod:`aether.verification.v1_structural`)
     - Conditioning of the reduced stiffness operator versus :math:`N`;
       free-free frequencies against the analytic uniform-beam solution.
   * - **V2** (:mod:`aether.verification.v2_slosh`)
     - Exact force transfer under the collocation quadrature rule.
   * - **V3** (:mod:`aether.verification.v3_integrators`)
     - Achieved step size and wall clock for explicit, modally truncated,
       and IMEX strategies.
   * - **V4** (:mod:`aether.verification.v4_thermal`)
     - Manufactured-solution convergence for the charring/Stefan system on
       the immobilized grid, using :mod:`aether.verification.mms_ablation`.
   * - **V4-FIAT** (:mod:`aether.verification.v4_fiat`)
     - FIAT cross-verification: independent implementation of the Chen–Milos
       formulation against the thermal solver.
   * - **V5** (:mod:`aether.verification.v5_filter`)
     - Filter diagnostics: χ² anomaly gating and IAE convergence.
   * - **V7** (:mod:`aether.verification.v7_dispersion`)
     - Terminal dispersion from measured ERA5 winds, with exact elliptical
       containment radii checked against Siouris Table 5.2.
   * - **V8** (:mod:`aether.verification.v8_throughput`)
     - Batched Monte Carlo throughput and occupancy against replicate count.
   * - **P2V1** (:mod:`aether.verification.p2v1_ultraspherical`)
     - Conditioning of the ultraspherical formulation versus collocation.
   * - **P2V4** (:mod:`aether.verification.p2v4_blending`)
     - Blending function verification for multi-regime transitions.
   * - **P2V5** (:mod:`aether.verification.p2v5_coast`)
     - Coast phase verification in the hybrid dynamics.
   * - **P2V8** (:mod:`aether.verification.p2v8_aerothermal`)
     - Aerothermal model verification against stagnation heating data.
   * - **P2V67** (:mod:`aether.verification.p2v67_gnc`)
     - GNC (guidance, navigation, control) integration verification.
   * - **P2V123** (:mod:`aether.verification.p2v123_plates`)
     - Mindlin plate and laminate stiffness benchmarks.
   * - **R1V1** (:mod:`aether.verification.r1v1_timescales`)
     - Timescale separation verification in exact arithmetic.
   * - **R1V2** (:mod:`aether.verification.r1v2_attitude_gap`)
     - Attitude gap verification in exact arithmetic.
   * - **R1V3** (:mod:`aether.verification.r1v3_impulse_rank`)
     - Impulse rank verification in exact arithmetic.
   * - **R1V4** (:mod:`aether.verification.r1v4_residual_constant`)
     - Residual constant verification in exact arithmetic.
   * - **R1V5** (:mod:`aether.verification.r1v5_augmented_field`)
     - Augmented field verification in exact arithmetic.
   * - **Integration** (:mod:`aether.verification.integration`)
     - Coupled system verification across all physics.

The :mod:`aether.verification.common.VerificationReport` class is the
standardised output type; all runners return an instance of it, and its
:meth:`~aether.verification.common.VerificationReport.write` method produces
the Markdown and CSV files.

Interpreting Results
--------------------

- **PASS**: The acceptance criterion was met. The numerical result is within
  the stated tolerance.
- **FAIL**: The acceptance criterion was not met. This indicates either a
  regression in the implementation or that the stated criterion was too
  aggressive for the current numerical fidelity.

Reports are timestamped and include the full environment so that results are
reproducible and auditable. The CSV files enable automated tracking of
verification metrics over time.