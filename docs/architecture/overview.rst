Architecture Overview
=====================

What is AETHER
--------------

AETHER — Aero-thermo-Elastic Trajectory & Hypersonic Estimation Research — is
a Python package implementing a fixed-grid spectral formulation for coupled
moving-boundary problems, with batched uncertainty quantification. It is the
numerics kernel behind the thesis proposal in ``manuscript/``.

The motivating idea: solve the coupled, highly nonlinear dynamics of a
maneuvering hypersonic body **weakly**, and extract a finite-dimensional
skeleton in the form of a trajectory outer-bound via Galerkin projection. This
evolved into a research program on *certified* reachable sets for hypersonic
entry vehicles.

The Problem
-----------

The reachable set of a hypersonic entry vehicle — every state it can be driven
to under admissible control — is what safety-critical and threat analysis
actually need bounded. Computing it exactly means solving a Hamilton–Jacobi–Isaacs
equation on a grid: exponential in state dimension, capped in practice around
four or five states. A rigid-body entry vehicle has thirteen before any
augmentation, and twenty-five once ablation, heat load and structural
deformation are carried.

Worse, a grid gives a numerical approximation, not a certificate. And a
polynomial barrier whose coefficients came out of an interior-point solver has
analytical *form* but inherits the solver's accuracy. There are two things to
beat here, not one.

The Approach
------------

Embed the hybrid vehicle dynamics into a Banach space of occupation measures,
where the problem becomes **linear** despite the nonlinearity of the flight
mechanics. In the thin-atmosphere limit the resulting measure family is *not
compact*: mass concentrates onto a low-dimensional set of trajectories. Turn
that failure of compactness into a profile decomposition — a finite-dimensional
skeleton, one boundary-layer profile per event, and a residual with a
quantitative bound — and use its structure to **derive** a comparison
certificate rather than search for one numerically.

The bound is then analytical in the strong sense: derived by hand, discharged
in exact rational arithmetic over the entry corridor. Sum-of-squares
programming appears only as an independent numerical cross-check, never as the
source of the guarantee.

What the Framework Does and Does Not Own
----------------------------------------

It owns **propagation fidelity**: given the input models and their uncertainty,
that uncertainty is carried into reachable-set uncertainty soundly and without
leakage. It does **not** own **flow-physics fidelity** — the coefficient
envelopes, the temperature field, the softening law, transition and
shock/boundary-layer interaction are empirical inputs with their own error
bars, and no amount of analysis manufactures fidelity that is not in them.

The honest headline is therefore: *the tightest certified reachable set
consistent with the fidelity of the aerothermodynamic inputs, with that
fidelity faithfully propagated and never silently exceeded.*

Package Layout
--------------

The package is organised into four layers, each building on the one below.

Foundations
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Module
     - Role
   * - :mod:`aether.spectral`
     - Chebyshev–Gauss–Lobatto collocation operators by direct recurrence with
       the negative-sum trick; Clenshaw–Curtis quadrature; barycentric
       interpolation.
   * - :mod:`aether.ultraspherical`
     - Olver–Townsend banded operators and assembly — the bandedness the
       sparsity argument rests on.
   * - :mod:`aether.ellipsoid`
     - WGS84 ellipsoid geometry and geodesic computations.
   * - :mod:`aether.geodesy`
     - Geodetic conversions, great-ellipse navigation, and footprint
       computation.
   * - :mod:`aether.blending`
     - Smooth blending functions for multi-regime transitions.
   * - :mod:`aether.paths`
     - Environment-variable-backed paths for results and reference data.

Numerics Kernel
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Module
     - Role
   * - :mod:`aether.thermal`
     - Landau boundary-immobilization frame; semi-discrete charring/Stefan
       solver on the fixed grid.
   * - :mod:`aether.structures`
     - Variable-rigidity Euler–Bernoulli operator with the full product-rule
       expansion; free-free BCs by null-space projection; modal reduction;
       integrators.
   * - :mod:`aether.plates`
     - Mindlin plates, laminate stiffness.
   * - :mod:`aether.coupling`
     - Quadrature-normalized kernel force transfer.
   * - :mod:`aether.atmosphere`
     - US Standard 1976, MSIS thermosphere, ERA5 reanalysis winds.
   * - :mod:`aether.dynamics`
     - Quaternion attitude kinematics with norm-error diagnostics; incidence
       on a deformed surface.

Flight
~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Module
     - Role
   * - :mod:`aether.flight`
     - The coupled simulator: thirteen rigid-body states augmented by mass,
       recession and retained structural modes, as one system of ODEs.
   * - :mod:`aether.orbital`
     - Two-body astrodynamics over an arbitrary central body; atmospheric
       coast.
   * - :mod:`aether.guidance`
     - Entry guidance: drag tracking and bank-angle modulation; the entry
       trajectory as an optimal control problem; landing footprints by
       bank-schedule sweep, bracketed by the optimised outer edge.
   * - :mod:`aether.trajectory`
     - Reference trajectories as pseudospectral polynomials; problem
       specification, objectives, constraints and phase triggers; an offline
       library handed to the guidance loop as its nominal.
   * - :mod:`aether.optimal_control`
     - Legendre–Gauss–Lobatto direct transcription and Pontryagin refinement
       over an arbitrary problem.

Aerothermodynamics and CFD
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Module
     - Role
   * - :mod:`aether.aerodynamics`
     - Modified-Newtonian and Prandtl–Meyer impact, panels, skin friction,
       real-gas and free-molecular closures; axisymmetric Euler CFD for the
       sub- and transonic band.
   * - :mod:`aether.aerodynamics.cfd`
     - Axisymmetric Euler CFD subpackage (shock-fitted grids, solver).
   * - :mod:`aether.aerothermal`
     - Fay–Riddell stagnation heating with the Lewis exponent stated;
       modified-Newtonian velocity gradient.
   * - :mod:`aether.fiat`
     - Fully implicit ablation and thermal response of a multilayer TPS stack;
       independent implementation of the Chen–Milos formulation.
   * - :mod:`aether.cfd`
     - Three-dimensional SU2 across the flight envelope: shock-fitted grids,
       preflight checks, a run registry and worker that survive a dropped
       connection, and the sweep that turns solved runs into aerodynamic
       tables; plasma sheath and RF blackout from the shock layer.

Geometry and Sensing
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Module
     - Role
   * - :mod:`aether.geometry`
     - Outer mould line from a mesh; measured shape scalars rather than
       stipulated ones. ``prep`` conditions an authored STL into a master —
       consistent winding, degenerate facets dropped, units and orientation
       normalised — and writes a manifest recording every repair and factor
       applied.
   * - :mod:`aether.estimation`
     - Adaptive state estimation (χ² anomaly gating and IAE), navigation
       through plasma blackout, and heterogeneous measurement models with
       their covariance.

Certification
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Module
     - Role
   * - :mod:`aether.certification`
     - Machinery answering "is this inequality *provably* true", as distinct
       from "did a float suggest so": rigorous enclosures and verified
       reachable tubes in Arb ball arithmetic, the entry field written for
       every arithmetic at once, monotone density enclosures, and a certified
       downrange bound that carries the corridor hypothesis it rests on.

Infrastructure
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Module
     - Role
   * - :mod:`aether.batch`
     - NumPy/CuPy backend abstraction; batched common-outer-grid integrator
       — a Monte Carlo batch as a rank-3 tensor operation; terminal
       dispersion from measured ERA5 winds, with exact elliptical containment
       radii checked against Siouris Table 5.2.
   * - :mod:`aether.viz`
     - Textured WGS84 ellipsoid, terrain and Blue Marble Next Generation
       imagery tiles, vehicle glyphs, flow-field rendering, and ``scene`` — a
       chase-camera rig with polyline, mesh and glyph projection over the
       globe.
   * - :mod:`aether.verification`
     - Executable verification tasks with failure criteria stated in advance.

Scope Note
----------

This package is deliberately limited to numerical-methods and standard-model
content. Applied flight-systems capability — guidance, trajectory
optimization, sensing, tracking, and engagement modelling — is maintained
separately and is not distributed with this package. See :doc:`boundary` for
the formal boundary definition.

Repository Boundary
-------------------

This repository is public and holds a one-way dependency boundary: the public
kernel must not import controlled code, which lives in a separate repository
rather than in an ignored subdirectory here. An ignored directory is one
``git add -f`` away from being published; a separate repository is not. The
boundary is checked by :mod:`tools.check_boundary`.

See :doc:`boundary` for the formal boundary definition and enforcement.