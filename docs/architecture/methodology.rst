Methodology
===========

This guide summarises the mathematical methods implemented in the numerics
kernel. For the full derivation, see the manuscript appendices and the
derivation PDF at the repository root.

Spectral Methods
----------------

The foundation is the Chebyshev–Gauss–Lobatto (CGL) collocation method. The
CGL points are the extrema of the Chebyshev polynomial :math:`T_N(x)` on
:math:`[-1, 1]`:

.. math::

   x_k = \cos\left(\frac{k\pi}{N}\right), \quad k = 0, \dots, N.

The differentiation matrix is constructed by the **direct recurrence with the
negative-sum trick** (Weideman & Reddy, 2000), which is numerically stable and
avoids the :math:`O(N^2)` cancellation errors of the naive formula. The
negative-sum trick computes the diagonal entries as the negative sum of the
off-diagonals in each row, preserving the exact nullspace of the derivative
operator.

**Clenshaw–Curtis quadrature** uses the same nodes with weights computed by a
fast cosine transform, giving exact integrals for polynomials up to degree
:math:`N` (or :math:`2N-1` for the modified rule).

**Barycentric interpolation** evaluates the polynomial interpolant at arbitrary
points in :math:`O(N)` using the second barycentric form with precomputed
weights. This is the evaluation engine used throughout the package.

Key modules: :mod:`aether.spectral.chebyshev`, :mod:`aether.spectral.chebyshev`.

Ultraspherical Formulation
--------------------------

The ultraspherical (Gegenbauer) spectral method (Olver & Townsend, 2013)
represents functions in the ultraspherical polynomial basis
:math:`C_k^{(\lambda)}(x)` with :math:`\lambda > -1/2`. The key property is
that the derivative of an ultraspherical polynomial is a *banded* combination
of lower-degree polynomials:

.. math::

   \frac{d}{dx} C_k^{(\lambda)}(x) = 2\lambda \sum_{j=0}^{k-1} c_{kj}
   C_j^{(\lambda+1)}(x).

This bandedness — the differentiation matrix has bandwidth :math:`O(1)` rather
than :math:`O(N)` — is what the sparsity argument rests on. The
:mod:`aether.ultraspherical` module provides:

- :mod:`aether.ultraspherical.operators` — banded differentiation,
  integration, and conversion operators between :math:`\lambda` and
  :math:`\lambda+1` spaces.
- :mod:`aether.ultraspherical.assembly` — the block-banded assembly of
  multi-physics operators (thermal, structural, coupling) that preserves
  sparsity across the coupled system.

Why ultraspherical vs. collocation conditioning: the collocation
differentiation matrix has condition number :math:`O(N^2)`; the ultraspherical
formulation with banded operators reduces this to :math:`O(1)` for the
stiffest modes, which is essential for the high-order accuracy needed in the
certification layer.

Polynomial Embedding
--------------------

The **Savageau–Voit / Varma–Voit recast** makes transcendental terms
polynomial by adjoining auxiliary states. The core examples:

- **Square-root density lift**: :math:`y = \sqrt{\rho/\rho_0}` with
  :math:`\dot{y} \propto y`. This converts the exponential atmosphere into a
  polynomial rate law.
- **Reciprocal radius**: :math:`z = 1/r` for the gravitational term.
- **Temperature-dependent elastic modulus**: the same exponential lift,
  :math:`\dot{w} \propto w \cdot \dot{T}`.

No fit residual is smuggled in as a disturbance — the closure is *exact* by
construction. This is the keystone every downstream result routes through.

Reference: the manuscript appendix
``manuscript/proposal/appendices/entry_dynamics_and_polynomial_embedding.tex``
and the derivation PDF ``embedding_derivation.pdf`` at the repository root.

Thermal Kernel
--------------

The **Landau boundary-immobilization frame** maps the moving-boundary Stefan
problem to a fixed grid. Let :math:`s(t)` be the ablation front position.
The transformation :math:`\xi = x / s(t)` immobilizes the boundary at
:math:`\xi = 1`, introducing a convection term :math:`-(\dot{s}/s) \xi
\partial_\xi T` in the heat equation.

The semi-discrete charring/Stefan solver on the resulting fixed grid is
implemented in :mod:`aether.thermal.solver`. The governing equations are
Eqs. (3.14)--(3.20) in the manuscript (referenced extensively in the module
docstrings). The key variables:

- :math:`T(\xi, t)` — temperature in the immobilized frame
- :math:`s(t)` — recession depth, with rate :math:`\dot{s} = f(q_{\text{wall}})`
- :math:`q_{\text{wall}}` — wall heat flux from the aerothermal model

The coupling to the structural solver is through the temperature-dependent
elastic modulus :math:`E(T)`, which enters the banded stiffness operator in
:mod:`aether.structures`.

Verification Philosophy
-----------------------

Each V-task states its **acceptance criterion before running** and writes a
timestamped Markdown report plus machine-readable CSV into a results directory.
The verification tasks are executable — they are not unit tests but
independent numerical experiments with a pass/fail verdict.

The verification suite is organised as:

- **V1–V8**: Core numerics (structural, slosh, integrators, thermal, FIAT,
  filter, dispersion, throughput)
- **P2V1–P2V123**: Phase 2 verification (ultraspherical, blending, coast,
  aerothermal, GNC, plates)
- **R1V1–R1V5**: Rigorous verification (timescales, attitude gap, impulse
  rank, residual constant, augmented field)
- **Integration**: Coupled system verification

Run all tasks with:

.. code-block:: bash

   make verify

or equivalently:

.. code-block:: bash

   python -m aether.verification

Each task's report states: task ID, title, criterion, verdict, and environment
(NumPy/SciPy/Python versions, platform). See :doc:`../tutorials/verification`
for the full task list and output format.