r"""A symbolic system: a field with named state, parameters and outputs, and its compiled form.

:mod:`aether.symbolic.codegen` turns a list of expressions into a routine.
A simulation needs slightly more than a routine: which entry of the state is
the speed, which parameter is the bank angle, and the quantities one wants to
read along a trajectory -- the altitude, the drag, the load factor -- that
are functions of the state and are not part of it. :class:`SymbolicSystem`
holds those together on symbols, and :class:`CompiledSystem` is what
:meth:`SymbolicSystem.compile` returns: the same thing with the expressions
printed to C and compiled, so that everything a simulator evaluates has an
expression behind it that can be read, differentiated, and audited against
the routine in extended precision.

What is generated and what is not stays explicit, as in the code generator.
The rates, their Jacobian and the outputs are generated. The integration
scheme is not: :meth:`CompiledSystem.advance` drives the fixed-step
Runge--Kutta routine the generator emits beside every field, and controls
its error by repeating the interval at half the step.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp
from numpy.typing import ArrayLike, NDArray

from aether.symbolic.codegen import CompiledField, LambdifiedField, compile_field

__all__ = ["CompiledSystem", "SymbolicSystem"]

_FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SymbolicSystem:
    """A vector field on symbols, with the names a simulator needs.

    Attributes
    ----------
    name:
        Used for the generated routines; a C identifier.
    state, parameters:
        The symbols the field is a function of, in the order the compiled
        routine takes them.
    rates:
        One expression per state component.
    outputs:
        Named functions of the state and the parameters that are read along
        a trajectory but not integrated.
    """

    name: str
    state: tuple[Any, ...]
    parameters: tuple[Any, ...]
    rates: tuple[Any, ...]
    outputs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.rates) != len(self.state):
            raise ValueError(
                f"one rate per state component: {len(self.state)} components, "
                f"{len(self.rates)} rates"
            )
        names = [str(symbol) for symbol in (*self.state, *self.parameters)]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f"{repeated} name more than one state component or parameter")
        bound = set(self.state) | set(self.parameters)
        for label, expressions in (("rates", self.rates), ("outputs", self.outputs.values())):
            free = set().union(*(sp.sympify(e).free_symbols for e in expressions)) - bound
            if free:
                raise ValueError(
                    f"the {label} of {self.name} depend on {sorted(map(str, free))}, which "
                    f"are neither state nor parameters"
                )

    @property
    def state_names(self) -> tuple[str, ...]:
        return tuple(str(symbol) for symbol in self.state)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(str(symbol) for symbol in self.parameters)

    def compile(self, *, jacobian: bool = True, build_dir: Path | None = None) -> CompiledSystem:
        """Print the rates and the outputs to C, compile them, and load the result."""
        rates = compile_field(
            self.rates, self.state, self.parameters,
            name=self.name, jacobian=jacobian, build_dir=build_dir,
        )
        observer = None
        if self.outputs:
            observer = compile_field(
                tuple(self.outputs.values()), self.state, self.parameters,
                name=f"{self.name}_outputs", jacobian=False, build_dir=build_dir,
            )
        return CompiledSystem(self, rates, observer)


@dataclass(frozen=True)
class CompiledSystem:
    """A :class:`SymbolicSystem` whose expressions have been compiled."""

    system: SymbolicSystem
    field: CompiledField | LambdifiedField
    observer: CompiledField | LambdifiedField | None = None

    @property
    def backend(self) -> str:
        """``"c"`` when a compiler was found, ``"numpy"`` otherwise."""
        return self.field.backend

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(self.system.outputs)

    # -- packing ---------------------------------------------------------------------

    def parameter_vector(
        self, values: Mapping[str, float] | None = None, **named: float
    ) -> _FloatArray:
        """The parameters as the array the routines take, from their names.

        Every parameter has to be given: a default would be a number nobody
        chose.
        """
        given = {**(values or {}), **named}
        names = self.system.parameter_names
        unknown = sorted(set(given) - set(names))
        if unknown:
            raise ValueError(f"{unknown} are not parameters of {self.system.name}: {names}")
        missing = [name for name in names if name not in given]
        if missing:
            raise ValueError(f"{self.system.name} needs a value for {missing}")
        return np.array([float(given[name]) for name in names], dtype=np.float64)

    def state_index(self, name: str) -> int:
        return self.system.state_names.index(name)

    def parameter_index(self, name: str) -> int:
        return self.system.parameter_names.index(name)

    # -- evaluation ------------------------------------------------------------------

    def rates(self, state: ArrayLike, parameters: ArrayLike) -> _FloatArray:
        return self.field(state, parameters)

    def jacobian(self, state: ArrayLike, parameters: ArrayLike) -> _FloatArray:
        return self.field.jacobian(state, parameters)

    def output_vector(self, state: ArrayLike, parameters: ArrayLike) -> _FloatArray:
        if self.observer is None:
            raise ValueError(f"{self.system.name} declares no outputs")
        return self.observer(state, parameters)

    def outputs(self, state: ArrayLike, parameters: ArrayLike) -> dict[str, float]:
        """The named outputs at one state."""
        values = self.output_vector(state, parameters)
        return {name: float(value) for name, value in zip(self.output_names, values, strict=True)}

    def output_table(self, states: ArrayLike, parameters: ArrayLike) -> dict[str, _FloatArray]:
        """The named outputs along a trajectory: one array per name, one entry per row."""
        if self.observer is None:
            raise ValueError(f"{self.system.name} declares no outputs")
        table = self.observer.batch(states, parameters)
        return {name: table[:, column] for column, name in enumerate(self.output_names)}

    def audit(self, state: ArrayLike, parameters: ArrayLike, digits: int = 40) -> float:
        """Largest relative gap between the compiled routines and their expressions."""
        worst = self.field.audit(state, parameters, digits)
        if self.observer is not None:
            worst = max(worst, self.observer.audit(state, parameters, digits))
        return worst

    # -- integration -----------------------------------------------------------------

    def trajectory(
        self,
        state: ArrayLike,
        parameters: ArrayLike,
        duration: float,
        *,
        step: float,
        stride: int = 1,
    ) -> tuple[_FloatArray, _FloatArray]:
        """Times and states every ``stride`` fixed steps over ``duration``.

        The step is the largest that divides the duration into a whole number
        of steps and does not exceed ``step``. The times are measured from
        the start of the interval, which is not included.
        """
        if not (duration > 0.0 and step > 0.0):
            raise ValueError(f"duration and step must be positive, got {duration} and {step}")
        strides = max(1, math.ceil(duration / (step * stride) - 1e-12))
        steps = strides * stride
        actual = duration / steps
        rows = self.field.rk4(state, parameters, actual, steps, stride)
        times = actual * stride * np.arange(1, strides + 1, dtype=np.float64)
        return times, rows

    def advance(
        self,
        state: ArrayLike,
        parameters: ArrayLike,
        duration: float,
        *,
        step: float,
        rtol: float = 1e-10,
        atol: ArrayLike = 0.0,
        max_halvings: int = 12,
    ) -> _FloatArray:
        """The state after ``duration``, with the step halved until the answer holds.

        The interval is integrated at ``step`` and again at half of it. The
        two fourth-order answers differ by fifteen times the error of the
        finer one, so their difference is the estimate, and the step is
        halved again while it exceeds ``atol + rtol * |state|``. The finer
        answer is returned. An interval that will not converge raises: a
        silent loss of accuracy in a plant would be read as physics.
        """
        if not (duration > 0.0 and step > 0.0):
            raise ValueError(f"duration and step must be positive, got {duration} and {step}")
        steps = max(1, math.ceil(duration / step - 1e-12))
        coarse = self.field.rk4(state, parameters, duration / steps, steps, steps)[-1]
        for _ in range(max_halvings):
            steps *= 2
            fine = self.field.rk4(state, parameters, duration / steps, steps, steps)[-1]
            error = np.abs(fine - coarse) / 15.0
            if np.all(error <= np.asarray(atol) + rtol * np.abs(fine)):
                return np.asarray(fine, dtype=np.float64)
            coarse = fine
        raise RuntimeError(
            f"{self.system.name} did not converge over {duration} s at a step of "
            f"{duration / steps:.3e} s; the largest error estimate was {float(error.max()):.3e}"
        )


def named(values: Sequence[float], names: Sequence[str]) -> dict[str, float]:
    """Pair values with names; a small convenience for reading a state."""
    return {name: float(value) for name, value in zip(names, values, strict=True)}
