"""Where the project's data lives, resolved once rather than assumed.

The layout is a property of the deployment, not of the code: in the
devcontainer the results share is bind-mounted at ``/workspace/results``, on a
bare checkout it is a directory beside the repository. Neither is more correct,
so nothing here hard-codes one -- ``AETHER_RESULTS`` names it, and the relative
fallback keeps a plain ``git clone && pytest`` working with no environment set
up at all.

Subsystems take a *subdirectory* of the root (``results/cfd``, ``results/e2e``)
rather than each claiming their own top-level path, so the share stays legible
as one thing when several of them are writing into it.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["cfd_results", "results_root"]

#: Names the results share. Set by the devcontainer; unset on a bare checkout.
RESULTS_ENV = "AETHER_RESULTS"

#: Used when :data:`RESULTS_ENV` is unset. Relative on purpose -- an absolute
#: default would be this machine's layout leaking into every other one.
DEFAULT_RESULTS = Path("results")


def results_root() -> Path:
    """The root every result directory hangs off."""
    named = os.environ.get(RESULTS_ENV)
    return Path(named) if named else DEFAULT_RESULTS


def cfd_results() -> Path:
    """Where solved CFD cases are written and read back from."""
    return results_root() / "cfd"
