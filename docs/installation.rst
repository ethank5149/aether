Installation
============

Prerequisites
-------------

- Python 3.10 or later
- ``pip`` (bundled with Python)

Core Install
------------

The package is not yet published to PyPI. For development, install in editable
mode:

.. code-block:: bash

   pip install -e .[dev]

This installs the core dependencies (``numpy``, ``scipy``, ``pyproj``) plus the
development tooling (``pytest``, ``ruff``, ``mypy``).

Optional Extras
---------------

AETHER uses optional dependency groups (extras) for features that require
additional libraries. Install any combination by listing them in square
brackets:

.. code-block:: bash

   pip install -e ".[viz,atmosphere]"

Available extras:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Extra
     - Purpose / Dependencies
   * - ``dev``
     - Test and lint tooling: ``pytest``, ``ruff``, ``mypy``
   * - ``atmosphere``
     - MSIS thermosphere model (above 86 km): ``pymsis``
   * - ``viz``
     - Rendering and imagery: ``matplotlib``, ``pillow``, ``rasterio``, ``tqdm``
   * - ``certification``
     - Rigorous interval arithmetic: ``python-flint``
   * - ``cuda``
     - GPU batch backend: ``cupy-cuda13x`` (match your CUDA major version)
   * - ``reanalysis``
     - ERA5 reanalysis winds: ``cfgrib``, ``xarray``
   * - ``cfd``
     - 3D CFD meshing and fields: ``gmsh``, ``h5py``

Full Development Install
------------------------

To install all extras at once:

.. code-block:: bash

   pip install -e ".[dev,viz,atmosphere,certification,cuda,reanalysis,cfd]"

.. note::

   The ``cfd`` extra installs ``gmsh`` and ``h5py`` from PyPI, but CFD
   additionally requires **SU2** and a working ``gmsh`` import. These are
   not pip-installable:

   - **SU2**: A compiled solver. Install via conda:
     ``conda create -n su2 -c conda-forge su2``. The function
     :func:`aether.cfd.su2.find_su2` locates it on ``PATH``.
   - **gmsh**: The PyPI wheel links against X11 libraries (``libXft``,
     ``libXcursor``, ``libXinerama``). On headless machines, ``pip install
     gmsh`` succeeds but ``import gmsh`` fails. Use the conda-forge package
     instead: ``conda install -c conda-forge gmsh python-gmsh``, which pulls
     in the required X libraries. See :func:`aether.geometry.backend.require_gmsh`
     for the runtime check and error message.

Development Container
---------------------

The repository includes a full devcontainer definition in ``.devcontainer/``
with CUDA 12.9, conda environment, and all system dependencies (including
``gmsh`` from conda-forge with X libraries). Open the repository in VS Code
with the Dev Containers extension, or build the image manually:

.. code-block:: bash

   devcontainer build --workspace-folder . --image-name aether-dev:latest

The ``postCreateCommand`` installs both ``aether`` and the sibling
``aether-gambit`` repository in editable mode with their respective extras.