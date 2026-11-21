"""``make test-report``: run the test suite and record counts and coverage (ST module, PRD §11).

Runs pytest (unit + contract by default; ``--all`` adds the slow metamorphic/oracle tests,
``--all --integration`` also the Kafka tests) with JUnit XML and JSON coverage, and writes
``reports/metrics/test_summary.json``: passed / failed / skipped, duration, tests per folder,
line coverage of ``src/leakwatch`` (dashboard UI excluded) overall and per module.
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
    cov, per_module = None, {}
    if coverage and coverage.exists():
        doc = json.loads(coverage.read_text(encoding="utf-8"))
        cov = doc["totals"]["percent_covered"]
        for name, f in doc.get("files", {}).items():
            module = name.replace("\\", "/").split("src/leakwatch/")[-1]
            per_module[module] = {
                "statements": f["summary"]["num_statements"],
                "covered": f["summary"].get("covered_lines", 0),
                "percent": round(float(f["summary"]["percent_covered"]), 1),
            }
    return {
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "duration_s": round(duration, 1),
        "by_folder": dict(folders),
        "coverage_percent": None if cov is None else round(float(cov), 1),
        "coverage_by_module": dict(sorted(per_module.items())),
    }


def core_coverage(per_module: dict, core: list[str]) -> dict:
    """Statement-weighted line coverage of the core modules (``st.core_modules``, D-049).

    Core = the code behind a reported number or a safety check (detection, alerts, scoring,
    re-baselining, localization, SQL and number guardrails), not job drivers or the UI.
    """
    rows = {m: per_module[m] for m in core if m in per_module}
    stmts = sum(int(r["statements"]) for r in rows.values())
    covered = sum(int(r.get("covered", 0)) for r in rows.values())
    return {
        "modules": len(rows),
        "missing": sorted(set(core) - set(rows)),
        "statements": stmts,
        "percent": round(100.0 * covered / stmts, 1) if stmts else None,
        "below_80": sorted(m for m, r in rows.items() if float(r["percent"]) < 80.0),
    }


def main(argv: list[str] | None = None) -> int:
    """Run pytest and write the summary."""
    parser = argparse.ArgumentParser(description="Run tests, record counts and coverage.")
    parser.add_argument("--all", action="store_true", help="include slow tests (not integration)")
    parser.add_argument(
        "--integration", action="store_true", help="with --all: also the Kafka tests (`make up`)"
    )
    args = parser.parse_args(argv)
    if args.all and args.integration:
        marker = "slow or not slow"  # every test
    elif args.all:
        marker = "not integration"
    else:
        marker = "not slow and not integration"
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
    from leakwatch.common.config import load_config

    core = list(load_config().get("st", {}).get("core_modules", []))
    summary["core_coverage"] = core_coverage(summary["coverage_by_module"], core)
    summary["marker"] = marker
    summary["pytest_exit_code"] = code
    path = write_metrics("test_summary", summary)
    print(f"{summary} -> {path}")  # noqa: T201 - CLI output
    return 0 if code == 0 else code


if __name__ == "__main__":
    sys.exit(main())
