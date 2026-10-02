r"""Compiled numerics printed from the symbolic field, so the fast code can be audited.

A hand-written numerical routine and the derivation it implements are two
objects, and nothing connects them except the care of whoever wrote the second
one. Here there is one object. The expressions the derivations reduce are
printed to C99 by SymPy's code generator, with common subexpressions lifted,
and compiled; the routine that runs is a rendering of the expression that was
checked, and :meth:`CompiledField.audit` measures the one against the other in
extended precision rather than trusting that the rendering was faithful.

What is generated and what is not is kept explicit in the emitted source. The
field and its Jacobian are generated. The fixed-step Runge--Kutta driver that
calls them is not -- it is twenty lines of scheme, the same for every field --
and it is tested against an independent integrator instead. Nor is the loop
that evaluates a routine at many states in one call, which is three lines.

A C compiler is an optional dependency. Without one, :func:`compile_field`
returns the same interface backed by :func:`sympy.lambdify`, slower but
generated from the same expressions, so nothing downstream has to know which
it was handed. The compiler is looked for in ``$CC``, then on ``PATH``, then
beside the interpreter, where a conda toolchain puts it under its target
triple.
"""

from __future__ import annotations

import ctypes
import hashlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import sympy as sp
from numpy.typing import ArrayLike, NDArray
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.printing.c import C99CodePrinter
from sympy.utilities.codegen import C99CodeGen

__all__ = [
    "CSource",
    "CompiledField",
    "Field",
    "LambdifiedField",
    "compile_field",
    "find_c_compiler",
    "generate_c",
]

_FloatArray = NDArray[np.float64]

#: IEEE arithmetic, operation by operation. ``-ffp-contract=off`` forbids fusing
#: a multiply and an add into one rounding, which a compiler is otherwise free
#: to do and which makes the result depend on the machine it was built for.
_CFLAGS = ("-O2", "-std=c99", "-fPIC", "-shared", "-ffp-contract=off")

#: Integer powers up to this magnitude are printed as repeated multiplication.
_MAX_EXPANDED_POWER = 8


class _ProductPowerPrinter(C99CodePrinter):  # type: ignore[misc]
    """Prints a small integer power of a named quantity as a product.

    ``pow`` is a library call, and for a fourth power it is slower and no more
    accurate than three multiplications. The rewrite is done here, at print
    time, on purpose. Doing it to the *expression* first -- SymPy's
    ``create_expand_pow_optimization`` -- and eliminating common subexpressions
    afterwards produced, for this field, a routine in which the Coriolis factor
    :math:`2\\omega_E V` had become :math:`\\omega_E^2 V`: it compiled, ran, and
    was wrong by 1.7 %. :meth:`CompiledField.audit` is what caught it, which is
    the argument for auditing a generated routine rather than trusting the
    generator.

    Only a symbol, an array entry or a lifted temporary is repeated. A compound
    base is left to ``pow``, since repeating it would evaluate it twice.
    """

    def _print_Pow(self, expr: Any) -> str:  # noqa: N802 - SymPy's dispatch name
        base, exponent = expr.as_base_exp()
        atomic = base.is_Symbol or isinstance(base, MatrixElement)
        if atomic and exponent.is_Integer and 2 <= abs(int(exponent)) <= _MAX_EXPANDED_POWER:
            product = "*".join([self._print(base)] * abs(int(exponent)))
            return f"({product})" if exponent > 0 else f"(1.0/({product}))"
        return str(super()._print_Pow(expr))

_DRIVER = """
/* ---- fixed driver: classical fourth-order Runge-Kutta. NOT generated. ----
 * Writes the state after every `stride` steps into `out`, row by row;
 * `out` must hold (steps / stride) * {n} doubles.
 */
void {name}_rk4(const double *x0, const double *p, double h, long steps,
{pad}long stride, double *out) {{
   double x[{n}], y[{n}], k1[{n}], k2[{n}], k3[{n}], k4[{n}];
   long row = 0;
   for (int i = 0; i < {n}; ++i) x[i] = x0[i];
   for (long s = 1; s <= steps; ++s) {{
      {name}(x, (double *)p, k1);
      for (int i = 0; i < {n}; ++i) y[i] = x[i] + 0.5 * h * k1[i];
      {name}(y, (double *)p, k2);
      for (int i = 0; i < {n}; ++i) y[i] = x[i] + 0.5 * h * k2[i];
      {name}(y, (double *)p, k3);
      for (int i = 0; i < {n}; ++i) y[i] = x[i] + h * k3[i];
      {name}(y, (double *)p, k4);
      for (int i = 0; i < {n}; ++i)
         x[i] += h * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
      if (s % stride == 0) {{
         for (int i = 0; i < {n}; ++i) out[row * {n} + i] = x[i];
         ++row;
      }}
   }}
}}
"""

_BATCH = """
/* ---- fixed driver: the routine above at each of `rows` states. NOT generated. ----
 * `xs` holds the states row by row; `out` must hold rows * {outputs} doubles.
 */
void {name}_batch(const double *xs, const double *p, long rows, double *out) {{
   for (long r = 0; r < rows; ++r)
      {name}((double *)(xs + r * {n}), (double *)p, out + r * {outputs});
}}
"""


@dataclass(frozen=True)
class CSource:
    """Generated C for one field: the text, and the expressions it renders."""

    name: str
    code: str
    header: str
    expressions: tuple[Any, ...]
    state: tuple[Any, ...]
    parameters: tuple[Any, ...]
    has_jacobian: bool

    @property
    def digest(self) -> str:
        """A short content hash, naming the build so identical sources share it."""
        return hashlib.sha256(self.code.encode()).hexdigest()[:16]


class Field(Protocol):
    """What a generated field offers, whichever backend produced it."""

    backend: str

    def __call__(self, state: ArrayLike, parameters: ArrayLike) -> _FloatArray: ...

    def batch(self, states: ArrayLike, parameters: ArrayLike) -> _FloatArray: ...

    def jacobian(self, state: ArrayLike, parameters: ArrayLike) -> _FloatArray: ...

    def rk4(
        self, state: ArrayLike, parameters: ArrayLike, step: float, steps: int, stride: int = 1
    ) -> _FloatArray: ...


def find_c_compiler() -> str | None:
    """A C compiler, or ``None``.

    ``$CC`` first, since that is how a build is told which one to use; then the
    usual names on ``PATH``; then the interpreter's own ``bin``, where a conda
    toolchain installs ``<triple>-gcc`` and only exports it as ``$CC`` once the
    environment is activated.
    """
    named = os.environ.get("CC", "").split()
    for candidate in (*named[:1], "cc", "gcc", "clang"):
        found = shutil.which(candidate)
        if found:
            return found
    beside = Path(sys.prefix) / "bin"
    for pattern in ("*-gcc", "*-clang", "*-cc"):
        hits = sorted(beside.glob(pattern))
        if hits:
            return str(hits[0])
    return None


def generate_c(
    name: str,
    expressions: Sequence[Any],
    state: Sequence[Any],
    parameters: Sequence[Any],
    *,
    jacobian: bool = True,
) -> CSource:
    r"""C99 source for ``expressions`` as a function of ``state`` and ``parameters``.

    Emits ``void name(double *x, double *p, double *out)`` and, with
    ``jacobian``, ``void name_jacobian(double *x, double *p, double *out)``
    writing :math:`\partial f_i/\partial x_j` row-major. Both are printed by
    :class:`sympy.utilities.codegen.C99CodeGen` with common subexpressions
    eliminated; the symbols are mapped onto array entries in the order given,
    which is the only convention the caller has to remember.
    """
    if not name.isidentifier():
        raise ValueError(f"{name!r} is not usable as a C function name")
    outputs = sp.Matrix([sp.sympify(e) for e in expressions])
    free = outputs.free_symbols - set(state) - set(parameters)
    if free:
        raise ValueError(
            f"the expressions depend on {sorted(map(str, free))}, which are neither "
            f"state nor parameters; every symbol must be bound to an argument"
        )
    n, m = len(state), len(parameters)
    x = sp.MatrixSymbol("x", n, 1)
    p = sp.MatrixSymbol("p", max(m, 1), 1)
    bind = {symbol: x[i, 0] for i, symbol in enumerate(state)}
    bind.update({symbol: p[j, 0] for j, symbol in enumerate(parameters)})

    generator = C99CodeGen(project="aether", printer=_ProductPowerPrinter(), cse=True)

    def routine(label: str, values: Any) -> Any:
        out = sp.MatrixSymbol("out", len(values), 1)
        return generator.routine(label, sp.Eq(out, values), argument_sequence=(x, p, out))

    routines = [routine(name, outputs.xreplace(bind))]
    if jacobian:
        partials = outputs.jacobian(sp.Matrix(list(state))).xreplace(bind)
        routines.append(
            routine(f"{name}_jacobian", sp.Matrix(n * len(outputs), 1, list(partials)))
        )
    code_text, header_text = io.StringIO(), io.StringIO()
    generator.dump_c(routines, code_text, name, header=False, empty=False)
    generator.dump_h(routines, header_text, name, header=False, empty=False)
    code, header = code_text.getvalue(), header_text.getvalue()
    banner = (
        f"/* {name}: generated by SymPy {sp.__version__} from the symbolic field.\n"
        f" * state      x[0..{n - 1}] = {', '.join(map(str, state))}\n"
        f" * parameters p[0..{max(m, 1) - 1}] = {', '.join(map(str, parameters)) or '(none)'}\n"
        f" * Do not edit: regenerate from the expressions instead. */\n"
    )
    driver = _BATCH.format(name=name, n=n, outputs=len(outputs))
    if len(outputs) == n:
        driver += _DRIVER.format(name=name, n=n, pad=" " * (len(name) + 10))
    return CSource(
        name=name,
        code=banner + code + driver,
        header=header,
        expressions=tuple(outputs),
        state=tuple(state),
        parameters=tuple(parameters),
        has_jacobian=jacobian,
    )


def _as_vector(values: ArrayLike, length: int, what: str) -> _FloatArray:
    vector = np.ascontiguousarray(values, dtype=np.float64)
    if vector.shape != (length,):
        raise ValueError(f"{what} has {length} components, got shape {vector.shape}")
    return vector


def _pointer(array: _FloatArray) -> Any:
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


class CompiledField:
    """A field compiled from generated C and called through :mod:`ctypes`."""

    backend = "c"

    def __init__(self, source: CSource, library: Path) -> None:
        self.source = source
        self._n = len(source.state)
        self._m = max(len(source.parameters), 1)
        self._outputs = len(source.expressions)
        self._library = ctypes.CDLL(str(library))
        pointer = ctypes.POINTER(ctypes.c_double)
        self._field = getattr(self._library, source.name)
        self._field.argtypes = [pointer, pointer, pointer]
        self._field.restype = None
        if source.has_jacobian:
            self._jacobian = getattr(self._library, f"{source.name}_jacobian")
            self._jacobian.argtypes = [pointer, pointer, pointer]
            self._jacobian.restype = None
        self._batch = getattr(self._library, f"{source.name}_batch")
        self._batch.argtypes = [pointer, pointer, ctypes.c_long, pointer]
        self._batch.restype = None
        self._rk4 = None
        if self._outputs == self._n:
            self._rk4 = getattr(self._library, f"{source.name}_rk4")
            self._rk4.argtypes = [
                pointer, pointer, ctypes.c_double, ctypes.c_long, ctypes.c_long, pointer,
            ]
            self._rk4.restype = None

    def _arguments(
        self, state: ArrayLike, parameters: ArrayLike
    ) -> tuple[_FloatArray, _FloatArray]:
        # A field with no parameters still takes the pointer; give it one slot.
        bound = parameters if len(self.source.parameters) else [0.0]
        return (
            _as_vector(state, self._n, "the state"),
            _as_vector(bound, self._m, "the parameters"),
        )

    def __call__(self, state: ArrayLike, parameters: ArrayLike = ()) -> _FloatArray:
        x, p = self._arguments(state, parameters)
        out = np.empty(self._outputs, dtype=np.float64)
        self._field(_pointer(x), _pointer(p), _pointer(out))
        return out

    def batch(self, states: ArrayLike, parameters: ArrayLike = ()) -> _FloatArray:
        """The field at each row of ``states``, evaluated in one call."""
        rows = np.ascontiguousarray(states, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != self._n:
            raise ValueError(f"states must have shape (rows, {self._n}), got {rows.shape}")
        _, p = self._arguments(rows[0] if rows.shape[0] else np.zeros(self._n), parameters)
        out = np.empty((rows.shape[0], self._outputs), dtype=np.float64)
        if rows.shape[0]:
            self._batch(_pointer(rows), _pointer(p), rows.shape[0], _pointer(out))
        return out

    def jacobian(self, state: ArrayLike, parameters: ArrayLike = ()) -> _FloatArray:
        if not self.source.has_jacobian:
            raise ValueError("this field was generated without its Jacobian")
        x, p = self._arguments(state, parameters)
        out = np.empty(self._outputs * self._n, dtype=np.float64)
        self._jacobian(_pointer(x), _pointer(p), _pointer(out))
        return out.reshape(self._outputs, self._n)

    def rk4(
        self,
        state: ArrayLike,
        parameters: ArrayLike = (),
        step: float = 1.0,
        steps: int = 1,
        stride: int = 1,
    ) -> _FloatArray:
        """States after every ``stride`` fixed steps, integrated entirely in C."""
        if self._rk4 is None:
            raise ValueError("only a vector field -- one rate per state -- can be integrated")
        if steps < 1 or stride < 1:
            raise ValueError(f"steps and stride must be positive, got {steps} and {stride}")
        x, p = self._arguments(state, parameters)
        out = np.empty((steps // stride, self._n), dtype=np.float64)
        self._rk4(_pointer(x), _pointer(p), float(step), int(steps), int(stride), _pointer(out))
        return out

    def audit(self, state: ArrayLike, parameters: ArrayLike = (), digits: int = 40) -> float:
        """Largest relative gap between the compiled field and the expressions it renders.

        The expressions are evaluated in ``digits`` significant digits and the
        compiled doubles compared against them, each scaled by the magnitude of
        the reference (or by one, for an entry that is zero). A few units in
        the sixteenth place is what a faithful rendering costs; anything larger
        means the printed code and the checked expression are not the same
        function.
        """
        return _audit(self, state, parameters, digits)


class LambdifiedField:
    """The same interface without a compiler: NumPy code generated by SymPy."""

    backend = "numpy"

    def __init__(self, source: CSource) -> None:
        self.source = source
        self._n = len(source.state)
        self._m = len(source.parameters)
        arguments = [list(source.state), list(source.parameters)]
        outputs = sp.Matrix(list(source.expressions))
        self._field = sp.lambdify(arguments, list(outputs), modules="numpy", cse=True)
        self._jacobian = None
        if source.has_jacobian:
            partials = outputs.jacobian(sp.Matrix(list(source.state)))
            self._jacobian = sp.lambdify(arguments, partials, modules="numpy", cse=True)

    def _arguments(
        self, state: ArrayLike, parameters: ArrayLike
    ) -> tuple[_FloatArray, _FloatArray]:
        return (
            _as_vector(state, self._n, "the state"),
            _as_vector(parameters, self._m, "the parameters"),
        )

    def __call__(self, state: ArrayLike, parameters: ArrayLike = ()) -> _FloatArray:
        x, p = self._arguments(state, parameters)
        # A piecewise expression is rendered as a selection among all of its
        # branches, each evaluated everywhere; the ones not selected may be
        # outside their domain, and say so.
        with np.errstate(all="ignore"):
            return np.asarray(self._field(x, p), dtype=np.float64)

    def batch(self, states: ArrayLike, parameters: ArrayLike = ()) -> _FloatArray:
        """The field at each row of ``states``."""
        rows = np.asarray(states, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != self._n:
            raise ValueError(f"states must have shape (rows, {self._n}), got {rows.shape}")
        out = np.empty((rows.shape[0], len(self.source.expressions)), dtype=np.float64)
        for index, row in enumerate(rows):
            out[index] = self(row, parameters)
        return out

    def jacobian(self, state: ArrayLike, parameters: ArrayLike = ()) -> _FloatArray:
        if self._jacobian is None:
            raise ValueError("this field was generated without its Jacobian")
        x, p = self._arguments(state, parameters)
        with np.errstate(all="ignore"):
            return np.asarray(self._jacobian(x, p), dtype=np.float64)

    def rk4(
        self,
        state: ArrayLike,
        parameters: ArrayLike = (),
        step: float = 1.0,
        steps: int = 1,
        stride: int = 1,
    ) -> _FloatArray:
        """The same scheme as the compiled driver, in Python."""
        if len(self.source.expressions) != self._n:
            raise ValueError("only a vector field -- one rate per state -- can be integrated")
        if steps < 1 or stride < 1:
            raise ValueError(f"steps and stride must be positive, got {steps} and {stride}")
        x, p = self._arguments(state, parameters)
        x = x.copy()
        out = np.empty((steps // stride, self._n), dtype=np.float64)
        for s in range(1, steps + 1):
            k1 = self(x, p)
            k2 = self(x + 0.5 * step * k1, p)
            k3 = self(x + 0.5 * step * k2, p)
            k4 = self(x + step * k3, p)
            x = x + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            if s % stride == 0:
                out[s // stride - 1] = x
        return out

    def audit(self, state: ArrayLike, parameters: ArrayLike = (), digits: int = 40) -> float:
        """As :meth:`CompiledField.audit`, for the NumPy rendering."""
        return _audit(self, state, parameters, digits)


def _audit(field: Any, state: ArrayLike, parameters: ArrayLike, digits: int) -> float:
    source: CSource = field.source
    x = np.asarray(state, dtype=np.float64)
    p = np.asarray(parameters, dtype=np.float64)
    # Bind the *exact* value of each double, not its shortest decimal repr, so
    # the reference is evaluated at the point the compiled code was given.
    symbols = (*source.state, *source.parameters)
    values = (*x, *p)
    bind = {
        symbol: sp.Rational(float(value)) for symbol, value in zip(symbols, values, strict=True)
    }
    computed = field(x, p)
    worst = 0.0
    for expression, value in zip(source.expressions, computed, strict=True):
        reference = sp.sympify(expression).xreplace(bind).evalf(digits)
        scale = max(abs(float(reference)), 1.0)
        worst = max(worst, abs(float(reference - sp.Float(float(value), digits))) / scale)
    return worst


def compile_field(
    expressions: Sequence[Any],
    state: Sequence[Any],
    parameters: Sequence[Any] = (),
    *,
    name: str = "field",
    jacobian: bool = True,
    compiler: str | None = None,
    build_dir: Path | None = None,
) -> CompiledField | LambdifiedField:
    """Generate, compile and load a field; fall back to NumPy when nothing can compile it.

    ``build_dir`` keeps the emitted ``.c`` beside the shared object, which is
    the artefact to read when the question is what exactly was compiled. Left
    unset, the build goes to a temporary directory that is removed when the
    interpreter exits, and a source already built in this process is not built
    again.
    """
    source = generate_c(name, expressions, state, parameters, jacobian=jacobian)
    chosen = compiler or find_c_compiler()
    if chosen is None:
        return LambdifiedField(source)
    key = (source.digest, chosen)
    if build_dir is None and key in _BUILT:
        return CompiledField(source, _BUILT[key])
    directory = Path(build_dir) if build_dir else _temporary_build_dir()
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{name}-{source.digest}"
    c_path = directory / f"{stem}.c"
    library = directory / f"{stem}.so"
    c_path.write_text(source.code.replace(f'#include "{name}.h"\n', ""), encoding="utf-8")
    command = [chosen, *_CFLAGS, "-o", str(library), str(c_path), "-lm"]
    built = subprocess.run(command, capture_output=True, text=True, check=False)
    if built.returncode != 0:
        raise RuntimeError(
            f"{chosen} could not compile the generated source {c_path}:\n{built.stderr.strip()}"
        )
    if build_dir is None:
        _BUILT[key] = library
    return CompiledField(source, library)


#: Libraries built into temporary directories by this process, by source digest
#: and compiler, and the directories themselves, held so they outlive the call
#: and are removed at exit.
_BUILT: dict[tuple[str, str], Path] = {}
_TEMPORARY: list[tempfile.TemporaryDirectory[str]] = []


def _temporary_build_dir() -> Path:
    holder = tempfile.TemporaryDirectory(prefix="aether-codegen-")
    _TEMPORARY.append(holder)
    return Path(holder.name)
