Repository Boundary
===================

The One-Way Dependency Boundary
-------------------------------

This repository is public and holds a **one-way dependency boundary**: the
public kernel (``aether``) must not import controlled code, which lives in a
separate repository (``aether-gambit``) rather than in an ignored subdirectory
here.

An ignored directory is one ``git add -f`` away from being published; a
separate repository is not.

Enforcement
-----------

The boundary is checked by :mod:`tools.check_boundary`. This script scans the
source tree for any import of ``aether_gambit`` or its submodules and fails
the build if found. It is part of the ``make check`` target, so the boundary
is a build-breaking condition, not a lint.

.. code-block:: bash

   make boundary

Controlled Code
---------------

Controlled code lives in a separate repository (``aether-gambit``). It imports
``aether`` as a dependency and provides applied flight-systems capability:

- Guidance and trajectory optimization beyond the standard models
- Sensing, tracking, and engagement modelling
- Mission-specific vehicle configurations and operational workflows

Relationship
------------

The public kernel (``aether``) must not import controlled code (``aether-gambit``).
Controlled code imports the public kernel.

.. code-block:: text

   aether (public kernel) <-- aether-gambit (controlled)

The public kernel is deliberately limited to numerical-methods and standard-model
content. Applied flight-systems capability — guidance, trajectory
optimization, sensing, tracking, and engagement modelling — is maintained
separately in ``aether-gambit`` and is not distributed with this package.

The demonstration vehicle in this repository is NASA's Orion capsule, and
every parameter it uses is taken from the open literature and cited where it
is used. No controlled data or models are included.