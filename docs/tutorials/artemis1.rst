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

Against the flight: Final phase at 498 s (flown 551 s), Terminal at 866 s
(882 s), skip apogee 273 kft (287 kft), six bank reversals (six), and the
Terminal phase starting 0.3 nmi from the target (2.7 nmi).

What the example leaves out, and says so where it runs: wind; the dependence
of the aerodynamic coefficients on Mach number; the attitude dynamics; the
vehicle's mass change; radiative heating; and the heat shield's thermal
response.

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
   * - Equations of motion
     - A point mass at trim over a rotating, oblate Earth, with altitude above
       the WGS 84 ellipsoid: the routine compiled from the symbolic field
       (:mod:`aether.flight.entry`, :mod:`aether.symbolic`)
   * - Entry interface
     - The published geodetic, inertial state, carried into the field's
       geocentric, planet-relative one (:func:`aether.flight.entry.interface_state`)
   * - Atmosphere
     - The 1976 standard to 80 km, joined to a fit of NRLMSISE-00 above, as
       one expression (:mod:`aether.symbolic.atmosphere`); no wind
   * - Finite roll rate
     - In the loop that flies the plant, and as a state of the guidance's own
       predictions (:mod:`aether.guidance.skip`)
   * - Certified downrange bound
     - Ball arithmetic over a corridor (:mod:`aether.certification.range_bound`)

Running the Example
-------------------

From the repository root:

.. code-block:: bash

   make example

or directly:

.. code-block:: bash

   python -m examples.artemis1.entry

It needs SymPy for the symbolic field, ``python-flint`` for the certified
bound, and the upper-atmosphere model the standard atmosphere is fitted to
above 86 km. A C compiler is used when one is found and is not required:
without one the generated code runs through NumPy, more slowly.

Key Outputs
-----------

- **Landing footprint**: The reachable set of landing points from entry
  interface, skip apogee, and the start of the Final phase.
- **Certified downrange bound**: A mathematically rigorous upper bound on
  downrange, valid for all admissible bank histories and atmospheres within
  the stated corridor (0.85–1.15× standard).
- **Verification against flight data**: Skip apogee, phase times, and terminal
   conditions compared to the flown Artemis I trajectory.

.. figure:: /_static/hero_trajectory_footprint.png
   :align: center
   :width: 100%
   :alt: Artemis I trajectory and reachable footprint

   Rendered from an earlier version of the example, before it flew the
   compiled symbolic field; the areas quoted are that version's.
   Top: altitude vs. downrange for the simulated skip entry, colored by speed,
   with the four PredGuid guidance phases shaded. Bottom-left: the bank-angle
   command history with bank reversals marked. Bottom-right: the reachable
   landing footprint from entry interface (4.7 M km²), skip apogee
   (345 K km²), and the start of the Final phase (255 K km²), with individual
   simulated landings as scatter points. The footprint shrinks as the capsule
   spends energy, with the target inside every set. Computed by
   :func:`aether.guidance.footprint.entry_footprint`.

.. figure:: /_static/hero_globe_trajectory.png
   :align: center
   :width: 100%
   :alt: Artemis I trajectory on a 3D globe

   The same simulated trajectory rendered on a ray-traced WGS84 globe with
   Blue Marble Next Generation imagery, using the :mod:`aether.viz.scene`
   rendering pipeline. The trajectory arc from the south Pacific to Baja
   California is coloured by speed (red = 11 km/s at entry interface,
   blue = terminal). Reachable landing footprints from skip apogee and the
   Final phase are drawn on the surface near the splashdown point.

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
   * - :mod:`aether.flight.entry`
     - The simulator: the compiled symbolic field, with event-accurate predictions
   * - :mod:`aether.symbolic`
     - The field, the atmosphere and the reference figure on symbols, and the
       C generated from them
   * - :mod:`aether.guidance.footprint`
     - The landing footprint, by sweeping bank schedules
   * - :mod:`aether.certification.range_bound`
     - Certified downrange bound in ball arithmetic
   * - :mod:`aether.geometry`
     - Orion outer mould line from CEV wind-tunnel proportions
   * - :mod:`aether.aerodynamics`
     - Newtonian panel method for hypersonic drag and L/D

The script is structured as a sequence of steps:

1. **Geometry and aerodynamics** — build the Orion shape and find the trim
   angle at the design L/D by the panel method.
2. **Entry interface** — carry the published state into the field's.
3. **Guidance and plant** — configure the skip guidance on its own model, and
   the plant on the atmosphere and L/D the flight met.
4. **The flight** — fly to terminal speed and compare with the flight's
   published phases and reversals.
5. **Footprints** — sweep bank schedules from three instants.
6. **Certified bound** — compute the downrange certificate in ball arithmetic.

The source file ``examples/artemis1/entry.py`` is the canonical reference —
read it alongside this tutorial to see how the modules compose.