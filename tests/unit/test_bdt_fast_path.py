"""Choosing the live model's fast-path factor on 2018 (D-043)."""

from __future__ import annotations

from leakwatch.bdt.fast_path import choose_factor


def row(factor, score, normal):
    return {"factor": factor, "score_2018_eur": score, "normal_window_alerts": normal}


def test_smallest_factor_that_is_no_worse_on_2018():
    rows = [row(None, 100.0, 0), row(20.0, 100.0, 0), row(8.0, 100.0, 0), row(3.0, 90.0, 0)]
    assert choose_factor(rows) == 8.0


def test_a_factor_that_adds_normal_window_alerts_is_not_chosen():
    rows = [row(None, 100.0, 1), row(8.0, 120.0, 2), row(12.0, 100.0, 1)]
    assert choose_factor(rows) == 12.0


def test_no_factor_qualifies_and_the_fast_path_stays_off():
    rows = [row(None, 100.0, 0), row(20.0, 99.0, 0), row(8.0, 50.0, 3)]
    assert choose_factor(rows) is None
