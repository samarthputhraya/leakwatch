"""``make test-report``: run the test suite and record counts and coverage (ST module, PRD §11).

Runs pytest (unit + contract + metamorphic; slow and integration tests excluded unless
``--all``) with JUnit XML and JSON coverage, and writes ``reports/metrics/test_summary.json``:
passed / failed / skipped, duration, tests per folder, and line coverage of ``src/leakwatch``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile

# Parses pytest's own JUnit output from a private temp dir, never untrusted XML.
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from leakwatch.common.metrics import write_metrics
from leakwatch.common.paths import REPO_ROOT


def summarise(junit: Path, coverage: Path | None) -> dict:
    """Counts from a JUnit XML file and the total from a coverage JSON file."""
    root = ET.parse(junit).getroot()
    suites = [root] if root.tag == "testsuite" else list(root)
    passed = failed = skipped = 0
    duration = 0.0
    folders: Counter[str] = Counter()
    for suite in suites:
        duration += float(suite.get("time", 0))
        for case in suite.iter("testcase"):
            cls = case.get("classname", "")
            parts = cls.split(".")
            folders[parts[1] if len(parts) > 1 and parts[0] == "tests" else parts[0]] += 1
            if case.find("failure") is not None or case.find("error") is not None:
                failed += 1
            elif case.find("skipped") is not None:
                skipped += 1
            else:
                passed += 1
    cov = None
    if coverage and coverage.exists():
        cov = json.loads(coverage.read_text(encoding="utf-8"))["totals"]["percent_covered"]
    return {
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "duration_s": round(duration, 1),
        "by_folder": dict(folders),
        "coverage_percent": None if cov is None else round(float(cov), 1),
    }


def main(argv: list[str] | None = None) -> int:
    """Run pytest and write the summary."""
    parser = argparse.ArgumentParser(description="Run tests, record counts and coverage.")
    parser.add_argument("--all", action="store_true", help="include slow tests (not integration)")
    args = parser.parse_args(argv)
    marker = "not integration" if args.all else "not slow and not integration"
    with tempfile.TemporaryDirectory() as tmp:
        junit, cov = Path(tmp) / "junit.xml", Path(tmp) / "coverage.json"
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-m",
            marker,
            "-q",
            "-p",
            "no:warnings",
            f"--junitxml={junit}",
            "--cov",
            f"--cov-report=json:{cov}",
        ]
        code = subprocess.run(cmd, cwd=REPO_ROOT, check=False).returncode
        summary = summarise(junit, cov)
    summary["marker"] = marker
    summary["pytest_exit_code"] = code
    path = write_metrics("test_summary", summary)
    print(f"{summary} -> {path}")  # noqa: T201 - CLI output
    return 0 if code == 0 else code


if __name__ == "__main__":
    sys.exit(main())
