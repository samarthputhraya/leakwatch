"""``make mutation``: summarize a mutmut run into ``reports/metrics/st_mutation.json`` (D-040).

Mutation testing changes the code on purpose (a ``<`` becomes ``<=``, a constant changes, a
branch flips) and reruns the unit tests that cover the changed function; a test suite that is
worth its coverage number fails ("kills" the mutant). mutmut 3 keeps one result file per mutated
source file (``mutants/<path>.meta``, ``exit_code_by_key``); this module counts the verdicts with
mutmut's own exit-code table and reports the mutation score = killed / (killed + survived), per
file and overall. Mutants that no test executes ("no tests") and timeouts are counted apart.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from leakwatch.common.logging import get_logger
from leakwatch.common.paths import REPO_ROOT

log = get_logger("leakwatch.eval.mutation")

#: mutmut 3.8 ``stats.status_by_exit_code`` (anything else is "suspicious").
STATUS = {
    1: "killed",
    3: "killed",
    0: "survived",
    5: "no tests",
    33: "no tests",
    34: "skipped",
    35: "suspicious",
    36: "timeout",
    37: "caught by type check",
    -24: "timeout",
    24: "timeout",
    152: "timeout",
    255: "timeout",
    -11: "segfault",
    -9: "segfault",
    2: "interrupted",
    None: "not checked",
}


def status_of(code: int | None) -> str:
    """Mutmut's verdict for a test run's exit code."""
    return STATUS.get(code, "suspicious")


def summarize(mutants_dir: Path) -> dict[str, Any]:
    """Verdict counts and mutation score per mutated file and overall."""
    files: dict[str, Any] = {}
    total: Counter[str] = Counter()
    for meta in sorted(mutants_dir.rglob("*.py.meta")):
        doc = json.loads(meta.read_text(encoding="utf-8"))
        counts = Counter(status_of(c) for c in doc.get("exit_code_by_key", {}).values())
        if not counts:
            continue
        rel = meta.relative_to(mutants_dir).as_posix().removesuffix(".meta")
        files[rel] = {**dict(sorted(counts.items())), "mutation_score": score(counts)}
        total.update(counts)
    survivors = sorted(
        key
        for meta in mutants_dir.rglob("*.py.meta")
        for key, code in json.loads(meta.read_text(encoding="utf-8"))
        .get("exit_code_by_key", {})
        .items()
        if status_of(code) == "survived"
    )
    return {
        "tool": "mutmut 3.8.1",
        "mutants": sum(total.values()),
        "by_status": dict(sorted(total.items())),
        "mutation_score": score(total),
        "files": files,
        "survivors": survivors,
    }


def score(counts: Counter[str]) -> float | None:
    """Killed / (killed + survived); None when neither happened."""
    k, s = counts.get("killed", 0), counts.get("survived", 0)
    return round(k / (k + s), 4) if k + s else None


def main(argv: list[str] | None = None) -> int:
    """Write ``st_mutation.json`` from ``mutants/`` (run ``mutmut run`` first)."""
    parser = argparse.ArgumentParser(description="Summarize a mutmut run.")
    parser.add_argument("--mutants", type=Path, default=REPO_ROOT / "mutants")
    args = parser.parse_args(argv)
    from leakwatch.common.metrics import write_metrics

    if not args.mutants.exists():
        log.error("%s missing: run `mutmut run` (make mutation)", args.mutants)
        return 1
    payload = summarize(args.mutants)
    path = write_metrics("st_mutation", payload)
    log.info(
        "%d mutants, score %s, %s -> %s",
        payload["mutants"],
        payload["mutation_score"],
        payload["by_status"],
        path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
