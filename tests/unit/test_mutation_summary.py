"""Mutation-testing summary (D-040): mutmut exit codes -> verdicts and score."""

from __future__ import annotations

import json

from leakwatch.eval.mutation import status_of, summarize


def test_summary_counts_verdicts_and_score(tmp_path):
    meta = tmp_path / "src" / "leakwatch" / "x.py.meta"
    meta.parent.mkdir(parents=True)
    codes = {"x.f__mutmut_1": 1, "x.f__mutmut_2": 0, "x.f__mutmut_3": 33, "x.g__mutmut_1": 3}
    meta.write_text(json.dumps({"exit_code_by_key": codes}), encoding="utf-8")
    s = summarize(tmp_path)
    assert s["mutants"] == 4
    assert s["by_status"] == {"killed": 2, "no tests": 1, "survived": 1}
    assert s["mutation_score"] == round(2 / 3, 4)
    assert s["files"]["src/leakwatch/x.py"]["mutation_score"] == round(2 / 3, 4)
    assert s["survivors"] == ["x.f__mutmut_2"]
    assert status_of(999) == "suspicious" and status_of(None) == "not checked"
