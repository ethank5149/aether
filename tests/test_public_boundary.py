"""The kernel must not depend on the applied layer.

`aether` is public; `aether-gambit` is not. Machinery may be promoted downward
into the kernel and applications may import upward from it, but never the
reverse -- a single import of `aether_gambit` from inside `aether` would carry
the applied layer into the public one, and it would do so silently. Nothing
about such an import looks wrong at the call site, and the test suite would go on
passing because both packages are installed here.

This is written as a source scan rather than an import check on purpose. An
import check only sees the modules a test happens to load; a scan sees every
line that could ever execute, including branches nothing currently reaches.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_KERNEL = pathlib.Path(__file__).resolve().parents[1] / "src" / "aether"
_FORBIDDEN = "aether_gambit"


def _kernel_sources() -> list[pathlib.Path]:
    return sorted(p for p in _KERNEL.rglob("*.py") if "__pycache__" not in p.parts)


def test_the_scan_is_actually_looking_at_something():
    """A guard that silently matched nothing would be worse than no guard."""
    sources = _kernel_sources()
    assert len(sources) > 20
    assert any(p.name == "geodesy.py" for p in sources)


@pytest.mark.parametrize("path", _kernel_sources(), ids=lambda p: p.name)
def test_no_kernel_module_imports_the_applied_layer(path: pathlib.Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offending: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            offending += [
                a.name for a in node.names
                if a.name == _FORBIDDEN or a.name.startswith(_FORBIDDEN + ".")
            ]
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and (
                node.module == _FORBIDDEN
                or node.module.startswith(_FORBIDDEN + ".")
            )
        ):
            offending.append(node.module)
    assert not offending, (
        f"{path.relative_to(_KERNEL.parent.parent)} imports {offending} -- the "
        f"public kernel must not depend on the applied layer. Promote what is "
        f"generic instead of importing what is not."
    )


def test_the_certification_package_names_no_vehicle():
    """The package was promoted on the argument that it is application-free.

    Checked rather than asserted, because that argument is the reason it is
    public at all. Terms are matched as whole words: a package about reachable
    sets may legitimately mention a 'target' set.
    """
    import re

    banned = (
        "glide", "hypersonic", "reentry", "warhead", "interceptor",
        "missile", "threat", "decoy", "ballistic", "kill",
    )
    package = _KERNEL / "certification"
    for path in sorted(package.rglob("*.py")):
        text = path.read_text(encoding="utf-8").lower()
        for term in banned:
            if re.search(rf"\b{term}\b", text):
                # An entry glide is named once, in prose, as the system the
                # measured figures came from. Anything more is a leak.
                occurrences = len(re.findall(rf"\b{term}\b", text))
                assert term == "glide" and occurrences <= 2, (
                    f"{path.name} mentions '{term}' {occurrences} time(s); the "
                    f"kernel's certification package is meant to be free of any "
                    f"particular application"
                )
