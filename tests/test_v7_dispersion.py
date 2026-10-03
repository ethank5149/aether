"""V7 sample-export CSVs: the hand-analysis files track the verdict's draw.

The run flies the full 32k batch, so these tests carry the ``slow`` marker
and are deselected by default; run them with ``pytest -m slow``.
"""

import csv
import re

import pytest

from aether.verification.v7_dispersion import run_v7

pytestmark = pytest.mark.slow


def _column(path, name):
    with path.open(newline="", encoding="utf-8") as fh:
        return [row[name] for row in csv.DictReader(fh)]


@pytest.fixture(scope="module")
def v7_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("v7")
    report = run_v7(out)
    return report, out


def test_sample_files_share_ids(v7_run):
    _, out = v7_run
    input_ids = _column(out / "v7-samples-inputs.csv", "sample_id")
    impact_ids = _column(out / "v7-samples-impacts.csv", "sample_id")
    assert input_ids == impact_ids
    assert input_ids == [str(i) for i in range(1, 32_001)]


def test_inside_r95_mean_matches_reported_containment(v7_run):
    report, out = v7_run
    flags = [int(v) for v in _column(out / "v7-samples-impacts.csv", "inside_r95")]
    mean = sum(flags) / len(flags)
    match = re.search(r"empirical containment of R95 ellipse \| ([\d.]+)", report.to_markdown())
    assert match is not None, "containment row missing from report"
    assert f"{mean:.4f}" == match.group(1)
