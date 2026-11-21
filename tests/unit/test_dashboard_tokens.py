"""Colour contrast of the control-room screen (D-052, D-055), read from ``static/tokens.css``.

WCAG 2.1: text needs 4.5:1 against what it sits on, marks (lines, rings, bars, focus outlines)
3:1. Every pair the stylesheets and the map actually use is listed here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

TOKENS = Path(__file__).parents[2] / "src/leakwatch/dashboard/static/tokens.css"


def tokens() -> dict[str, str]:
    text = TOKENS.read_text(encoding="utf-8")
    return dict(re.findall(r"--([a-z0-9-]+):\s*(#[0-9A-Fa-f]{6})\s*;", text))


def luminance(hex_colour: str) -> float:
    def channel(c: int) -> float:
        s = c / 255
        return s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4

    r, g, b = (int(hex_colour[i : i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def contrast(a: str, b: str) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


SURFACES = ("sheet", "sheet-deep", "panel")
TEXT = ("ink", "ink-2", "ink-3", "amber", "coral")  # amber and coral are also map labels
MARKS = ("pipe", "amber", "coral", "series-1", "series-2", "series-3", "ink-2")


def test_the_token_file_has_every_colour_the_checks_need():
    found = tokens()
    for name in (*SURFACES, *TEXT, *MARKS, "rule"):
        assert name in found, name


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("ink", TEXT)
def test_text_reaches_4_5_to_1_on_every_surface(ink, surface):
    t = tokens()
    assert contrast(t[ink], t[surface]) >= 4.5, (
        ink,
        surface,
        round(contrast(t[ink], t[surface]), 2),
    )


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("mark", MARKS)
def test_marks_reach_3_to_1_on_every_surface(mark, surface):
    t = tokens()
    assert contrast(t[mark], t[surface]) >= 3.0, (
        mark,
        surface,
        round(contrast(t[mark], t[surface]), 2),
    )


@pytest.mark.parametrize("fill", ("ink", "amber"))
def test_text_on_filled_chips_and_buttons(fill):
    """Buttons, priority chip and candidate numbers: sheet-coloured text on ink or amber."""
    t = tokens()
    assert contrast(t["sheet"], t[fill]) >= 4.5


def test_the_documented_ratios_still_hold():
    """The plan quotes these ratios on the sheet; a token edit that changes them shows up here."""
    t = tokens()
    quoted = {"ink": 13.1, "ink-2": 8.6, "ink-3": 6.2, "pipe": 3.9, "amber": 8.2, "coral": 5.3}
    for name, ratio in quoted.items():
        assert contrast(t[name], t["sheet"]) == pytest.approx(ratio, abs=0.1), name
