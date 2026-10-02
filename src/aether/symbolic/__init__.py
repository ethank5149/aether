"""The symbolic arithmetic: derivations as identities, and numerics printed from them.

:mod:`aether.certification` writes a field once against an operation set and
evaluates it in floats to simulate and in Arb balls to prove. This package is
the third arithmetic. Evaluated on SymPy symbols the field returns the
displayed equations themselves, and two things follow that neither of the
other arithmetics can supply.

**Derivations become checks.** Every claim the manuscript derives -- that the
equations of motion are Newton's law in the rotating frame, that the energy
dissipates at the drag power, what damps the phugoid and at what rate -- is
stated in :mod:`aether.symbolic.derivations` as a residual built from the
field of record and reduced to zero by exact rewriting. ``python -m
aether.symbolic`` runs them all and exits non-zero if any fails.

The translational problem is the field of record on symbols
(:mod:`~aether.symbolic.entry`, :mod:`~aether.symbolic.glide`). The rest of
the dynamics appendix -- the attitude and its passage to the aerodynamic
force, Euler's equations, the heating and the structural oscillator, and the
full state they make up -- is stated once in :mod:`~aether.symbolic.attitude`
and :mod:`~aether.symbolic.coupled`, as the appendix displays it. Two blocks
of it are derived and not only stated: the heat shield, from conservation of
mass and energy in depth to the cells the state carries
(:mod:`~aether.symbolic.thermal`), and the force of the air on the structural
modes, from one-dimensional gas dynamics to the generalized forces
(:mod:`~aether.symbolic.piston`). What the field is evaluated *on* is here
too: the reference ellipsoid the altitude is measured from, and the passage
from a published state to the field's (:mod:`~aether.symbolic.figure`), and
the atmosphere as an expression (:mod:`~aether.symbolic.atmosphere`).

**Simulations run what was checked.** A field with its closures bound is a
:class:`~aether.symbolic.system.SymbolicSystem`, and compiling one gives the
routine a simulator integrates: :mod:`~aether.symbolic.trim` states the
entry of a body at trim that way, and :mod:`aether.flight.entry` flies it.

**Fast numerics become auditable.** :mod:`aether.symbolic.codegen` prints the
same expressions to C, compiles them, and can measure the compiled routine
against the expression it was printed from in extended precision. The
performance comes from the compiler; the correctness is inherited from the
derivation, and the link between them is a code generator rather than a
second author.

It sits second in the order of evidence the proposal sets out: closed-form
constants evaluated in validated interval arithmetic first, symbolic checks of
the derivations behind them second, and a moment-SOS computation as an
independent cross-check third.

SymPy is an optional dependency (``pip install aether[symbolic]``), and a C
compiler an optional one beyond that; without a compiler the generated code
runs through NumPy instead.
"""

from aether.symbolic.atmosphere import StandardAtmosphere, standard_atmosphere
from aether.symbolic.attitude import AttitudeSymbols
from aether.symbolic.codegen import (
    CompiledField,
    CSource,
    LambdifiedField,
    compile_field,
    find_c_compiler,
    generate_c,
)
from aether.symbolic.coupled import CoupledSymbols
from aether.symbolic.derivations import (
    DERIVATIONS,
    STATED_NOT_DERIVED,
    Derivation,
    DerivationResult,
    check,
    check_all,
)
from aether.symbolic.entry import EntrySymbols
from aether.symbolic.figure import FigureSymbols
from aether.symbolic.glide import GlideSymbols
from aether.symbolic.ops import SYMPY_OPS, is_identically_zero, lie_derivative
from aether.symbolic.piston import (
    IsentropicFlow,
    PistonSymbols,
    StripSymbols,
    WavyWall,
    downwash,
)
from aether.symbolic.system import CompiledSystem, SymbolicSystem
from aether.symbolic.thermal import CharringCells, CharringSlab, SurfaceBalance
from aether.symbolic.trim import entry_system, prediction_system

__all__ = [
    "DERIVATIONS",
    "STATED_NOT_DERIVED",
    "SYMPY_OPS",
    "AttitudeSymbols",
    "CSource",
    "CharringCells",
    "CharringSlab",
    "CompiledField",
    "CompiledSystem",
    "CoupledSymbols",
    "Derivation",
    "DerivationResult",
    "EntrySymbols",
    "FigureSymbols",
    "GlideSymbols",
    "IsentropicFlow",
    "LambdifiedField",
    "PistonSymbols",
    "StandardAtmosphere",
    "StripSymbols",
    "SurfaceBalance",
    "SymbolicSystem",
    "WavyWall",
    "check",
    "check_all",
    "compile_field",
    "downwash",
    "entry_system",
    "find_c_compiler",
    "generate_c",
    "is_identically_zero",
    "lie_derivative",
    "prediction_system",
    "standard_atmosphere",
]
