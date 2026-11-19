"""``make test-report``: counts from pytest's JUnit XML and coverage per module."""

from __future__ import annotations

import json
import subprocess

import pytest

from leakwatch.eval import test_summary

JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" time="12.5">
<testcase classname="tests.unit.test_a" name="ok1" time="0.1"/>
<testcase classname="tests.unit.test_a" name="ok2" time="0.1"/>
<testcase classname="tests.unit.test_a" name="bad"><failure message="x"/></testcase>
<testcase classname="tests.contract.test_b" name="err"><error message="y"/></testcase>
<testcase classname="tests.integration.test_c" name="skip"><skipped message="z"/></testcase>
</testsuite></testsuites>
"""

COVERAGE = {
    "totals": {"percent_covered": 71.234},
    "files": {
        "src/leakwatch/bdt/alerts.py": {
            "summary": {"num_statements": 100, "percent_covered": 99.0}
        },
        "src\\leakwatch\\common\\lake.py": {
            "summary": {"num_statements": 50, "percent_covered": 40.04}
        },
    },
}


@pytest.fixture
def reports(tmp_path):
    junit, cov = tmp_path / "junit.xml", tmp_path / "coverage.json"
    junit.write_text(JUNIT, encoding="utf-8")
    cov.write_text(json.dumps(COVERAGE), encoding="utf-8")
    return junit, cov


def test_counts_folders_and_coverage(reports):
    s = test_summary.summarise(*reports)
    assert (s["passed"], s["failed"], s["skipped"]) == (2, 2, 1)
    assert s["duration_s"] == 12.5
    assert s["by_folder"] == {"unit": 3, "contract": 1, "integration": 1}
    assert s["coverage_percent"] == 71.2
    assert s["coverage_by_module"] == {
        "bdt/alerts.py": {"statements": 100, "percent": 99.0},
        "common/lake.py": {"statements": 50, "percent": 40.0},
    }


def test_missing_coverage_file_gives_no_number(reports, tmp_path):
    s = test_summary.summarise(reports[0], tmp_path / "absent.json")
    assert s["coverage_percent"] is None and s["coverage_by_module"] == {}


@pytest.mark.parametrize(
    ("argv", "marker"),
    [
        ([], "not slow and not integration"),
        (["--all"], "not integration"),
        (["--all", "--integration"], "slow or not slow"),
    ],
)
def test_main_selects_the_marker_and_writes_the_summary(monkeypatch, argv, marker):
    calls, written = [], {}

    def fake_run(cmd, cwd, check):
        calls.append(cmd)
        out = {a.split("=", 1)[0]: a.split("=", 1)[1] for a in cmd if "=" in a}
        with open(out["--junitxml"], "w", encoding="utf-8") as f:
            f.write(JUNIT)
        with open(out["--cov-report"].removeprefix("json:"), "w", encoding="utf-8") as f:
            json.dump(COVERAGE, f)
        return subprocess.CompletedProcess(cmd, 1)

    monkeypatch.setattr(test_summary.subprocess, "run", fake_run)
    monkeypatch.setattr(
        test_summary, "write_metrics", lambda name, payload: written.update({name: payload})
    )
    assert test_summary.main(argv) == 1  # pytest's exit code is passed on
    assert calls[0][calls[0].index("-m", 3) + 1] == marker
    summary = written["test_summary"]
    assert summary["marker"] == marker and summary["pytest_exit_code"] == 1
    assert summary["failed"] == 2
