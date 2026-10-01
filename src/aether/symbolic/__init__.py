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

from aether.symbolic.codegen import (
    CompiledField,
    CSource,
    LambdifiedField,
    compile_field,
    find_c_compiler,
    generate_c,
)
from aether.symbolic.derivations import (
    DERIVATIONS,
    Derivation,
    DerivationResult,
    check,
    check_all,
)
from aether.symbolic.entry import EntrySymbols
from aether.symbolic.glide import GlideSymbols
from aether.symbolic.ops import SYMPY_OPS, is_identically_zero, lie_derivative

__all__ = [
    "DERIVATIONS",
    "SYMPY_OPS",
    "CSource",
    "CompiledField",
    "Derivation",
    "DerivationResult",
    "EntrySymbols",
    "GlideSymbols",
    "LambdifiedField",
    "check",
    "check_all",
    "compile_field",
    "find_c_compiler",
    "generate_c",
    "is_identically_zero",
    "lie_derivative",
]
