"""Reduce every registered derivation: ``python -m aether.symbolic``.

Exits non-zero if any residual fails to reach zero, so the manuscript's
derivations can gate a build the way a failing test does. ``--emit-c DIR``
additionally writes the generated C for the rotating point-mass field, which
is the file to read when the question is what the compiled numerics compute.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import sympy as sp

from aether.certification.rotating_field import exponential_density, point_mass_field
from aether.symbolic.codegen import CSource, generate_c
from aether.symbolic.derivations import check_all
from aether.symbolic.entry import EntrySymbols
from aether.symbolic.ops import SYMPY_OPS

__all__ = ["point_mass_source"]


def point_mass_source(name: str = "entry_point_mass") -> CSource:
    r"""Generated C for the rotating, oblate point-mass field.

    State :math:`(r, \lambda, \phi, V, \gamma, \psi)`; parameters, in order,
    the bank angle, ballistic coefficient, lift-to-drag ratio, reference
    density, scale height, and the planet's :math:`\mu`, :math:`R_e`,
    :math:`J_2`, :math:`\omega_E` and flattening. Everything a certificate
    would range over is a parameter, so one compiled routine serves a whole
    class of vehicles and atmospheres.
    """
    s = EntrySymbols()
    bank = sp.Symbol("sigma", real=True)
    beta, lift_to_drag, density, height = sp.symbols("beta E rho_0 H_s", positive=True)
    field = point_mass_field(
        beta, lift_to_drag, density=exponential_density(density, height), planet=s.planet
    )
    planet = s.planet
    parameters = (
        bank, beta, lift_to_drag, density, height,
        planet.mu, planet.radius, planet.j2, planet.rotation_rate, planet.flattening,
    )
    return generate_c(name, field(s.state, bank, SYMPY_OPS), s.state, parameters)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the manuscript's derivations symbolically")
    parser.add_argument(
        "--emit-c",
        type=Path,
        metavar="DIR",
        help="also write the generated C for the rotating point-mass field into DIR",
    )
    args = parser.parse_args()

    results = check_all()
    width = max(len(r.derivation.key) for r in results)
    for result in results:
        d = result.derivation
        verdict = "PASS" if result.passed else "FAIL"
        print(f"{verdict}  {d.key:<{width}}  {d.kind:<8}  {result.seconds:6.2f}s  {d.claim}")
        if not result.passed:
            print(f"      used in {d.where}; residual left: {result.remainder}")
    failed = [r for r in results if not r.passed]
    print(f"{len(results) - len(failed)} of {len(results)} derivations reduce to zero")

    if args.emit_c is not None:
        source = point_mass_source()
        args.emit_c.mkdir(parents=True, exist_ok=True)
        (args.emit_c / f"{source.name}.c").write_text(source.code, encoding="utf-8")
        (args.emit_c / f"{source.name}.h").write_text(source.header, encoding="utf-8")
        print(f"wrote {args.emit_c / source.name}.c and .h")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
