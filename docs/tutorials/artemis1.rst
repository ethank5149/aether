Artemis I Tutorial
==================

What the Example Does
---------------------

The ``examples/artemis1/`` demonstration flies Orion's Artemis I lunar-return
skip entry from its published entry interface to terminal speed. Every input
is either taken from a NASA source cited where it is used or computed by this
package from those numbers:

- The Orion shape from the CEV wind-tunnel proportions, and its hypersonic
  drag from the Newtonian panel method at the design L/D — which lands the
  trim angle inside the band the Orion aerodynamic database gives, without
  being asked to.
- The flight through a predictor-corrector skip guidance in the PredGuid
  lineage (:mod:`aether.guidance.skip`), against an atmosphere and an L/D that
  differ from the guidance's model by what the flight's own estimators
  reported.
- The flight flown with the body rolling at a finite rate, which is not a
  detail: a reversal sweeps the lift vector through wings-level, and a loop
  that assumes the commanded bank is reached instantly plans a trajectory the
  body cannot fly.
- The reachable landing footprint from entry interface, skip apogee, and the
  start of the Final phase, with the target inside it as it shrinks.
- A **certified** upper bound on the downrange still available, holding for
  every admissible bank history and for atmospheres 0.85 to 1.15 times
  standard, discharged in exact ball arithmetic and conditional on a stated
  corridor. Inner sweep and outer certificate bracket the flown trajectory
  between them.

Against the flight: Final phase at 538 s (flown 551 s), Terminal at 880 s
(882 s), skip apogee 284 kft (287 kft), and the Terminal phase starting
2.0 nmi from the target (2.7 nmi).

Inputs
------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Input
     - Source / Computation
   * - Orion outer mould line
     - CEV wind-tunnel proportions (geometry module)
   * - Hypersonic drag / L/D
     - Newtonian panel method at design L/D (:mod:`aether.aerodynamics.panels`)
   * - Skip guidance
     - PredGuid predictor-corrector bank-angle modulation (:mod:`aether.guidance.skip`)
   * - Atmosphere
     - ERA5 reanalysis winds + MSIS thermosphere (:mod:`aether.atmosphere`)
   * - Finite roll rate
     - Explicitly modelled in the coupled simulator (:mod:`aether.flight`)
   * - Certified downrange bound
     - Exact ball arithmetic over corridor (:mod:`aether.certification.range_bound`)

Running the Example
-------------------

From the repository root:

.. code-block:: bash

   make example

or directly:

.. code-block:: bash

   python -m examples.artemis1.entry

The script requires the ``viz``, ``atmosphere``, and ``reanalysis`` extras
for the full demonstration (ERA5 winds, rendering). Install them with:

.. code-block:: bash

   pip install -e ".[viz,atmosphere,reanalysis]"

Key Outputs
-----------

- **Landing footprint**: The reachable set of landing points from entry
  interface, skip apogee, and the start of the Final phase.
- **Certified downrange bound**: A mathematically rigorous upper bound on
  downrange, valid for all admissible bank histories and atmospheres within
  the stated corridor (0.85–1.15× standard).
- **Verification against flight data**: Skip apogee, phase times, and terminal
  conditions compared to the flown Artemis I trajectory.

Code Walkthrough
----------------

The demonstration script is at ``examples/artemis1/entry.py`` (outside
``src/aether``, so not part of the autodoc API reference). Key modules used:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Role in the demonstration
   * - :mod:`aether.guidance.skip`
     - PredGuid skip guidance (predictor-corrector bank-angle modulation)
   * - :mod:`aether.atmosphere`
     - ERA5 reanalysis winds, MSIS thermosphere, standard atmosphere
   * - :mod:`aether.certification.range_bound`
     - Certified downrange bound in exact ball arithmetic
   * - :mod:`aether.batch`
     - Batched Monte Carlo propagation for the landing footprint
   * - :mod:`aether.geometry`
     - Orion outer mould line from CEV wind-tunnel proportions
   * - :mod:`aether.aerodynamics`
     - Newtonian panel method for hypersonic drag and L/D

The script is structured as a sequence of phases:

1. **Geometry & aerodynamics setup** — build the Orion shape, compute the
   panel-method aerodynamic database.
2. **Atmosphere & winds** — load ERA5 reanalysis for the entry epoch, construct
   the wind field.
3. **Guidance & simulator** — configure the PredGuid skip guidance and the
   coupled 13+ state simulator with finite roll rate.
4. **Reference trajectory** — fly the nominal trajectory to terminal speed.
5. **Batched dispersion** — run the Monte Carlo batch over wind and model
   uncertainty.
6. **Certified bound** — compute the exact-arithmetic downrange certificate.
7. **Visualization** — render the globe, footprint, and trajectory.

The source file ``examples/artemis1/entry.py`` is the canonical reference —
read it alongside this tutorial to see how the modules compose.