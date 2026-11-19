from __future__ import annotations

import json
from collections import Counter

from leakwatch.copilot.rag import ocr
from leakwatch.copilot.rag.chunking import (
    DocumentInfo,
    chunk_pdf_pages,
    follows,
    ocr_heading,
    ocr_heading_readings,
    split_caps,
    word_counts,
)

INFO = DocumentInfo("ws", "Water Supply Manual", "WS Manual 1999", "pdf", "x.pdf")


def box(x: float, y: float, text: str, score: float = 0.95, h: float = 30.0):
    return ([[x, y], [x + 100, y], [x + 100, y + h], [x, y + h]], text, score)


def test_boxes_are_grouped_into_lines_and_low_scores_dropped():
    boxes = [
        box(400, 102, "of the line"),
        box(10, 100, "Start"),
        box(10, 200, "Second line"),
        box(300, 205, "noise", score=0.2),
        box(10, 300, "382"),
    ]
    assert ocr.group_lines(boxes, min_score=0.5) == ["Start of the line", "Second line", "382"]
    # the page number at the foot moves to the first line, where the chunker reads it
    assert ocr.page_text(boxes, 0.5).splitlines() == ["382", "Start of the line", "Second line"]
    assert ocr.folio_first(["12", "text", "13"]) == ["12", "text", "13"]
    assert ocr.folio_first(["only"]) == ["only"]


def test_scans_are_detected_by_their_missing_text_layer():
    scan = ["", " ", "x" * 10]
    digital = ["x" * 80, "y" * 80, ""]
    assert ocr.needs_ocr(scan, 0.5) and not ocr.needs_ocr(digital, 0.5)
    assert ocr.text_share(digital) == 2 / 3 and not ocr.needs_ocr([], 0.5)


def test_settings_from_config():
    assert ocr.OcrSettings.from_config(None) == ocr.OcrSettings()
    s = ocr.OcrSettings.from_config({"dpi": 300, "workers": 0, "min_score": 0.7})
    assert (s.dpi, s.workers, s.min_score) == (300, 1, 0.7)


def test_cache_is_reused_only_for_the_same_file_and_settings(tmp_path, monkeypatch):
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    settings = ocr.OcrSettings()
    meta = {"engine": ocr.engine_name(), "dpi": settings.dpi, "min_score": settings.min_score}
    meta["sha256"] = ocr.sha256_of(pdf)
    cache = ocr.cache_path(pdf)
    rows = [{"meta": meta}, {"page": 2, "text": "two"}, {"page": 1, "text": "one"}]
    cache.write_text("\n".join(json.dumps(r) for r in rows) + "\n{cut off", encoding="utf-8")
    assert ocr.read_cache(cache, meta) == {1: "one", 2: "two"}
    assert ocr.read_cache(cache, {**meta, "dpi": 300}) == {}
    assert ocr.read_cache(tmp_path / "missing.jsonl", meta) == {}

    def no_ocr(*args, **kwargs):  # every page is cached: no worker pool may start
        raise AssertionError("OCR should not run")

    monkeypatch.setattr(ocr.mp, "get_context", no_ocr)
    assert ocr.ocr_pdf(pdf, settings, n_pages=2) == ["one", "two"]


def test_garbled_heading_numbers_have_several_readings():
    readings = {(n, t) for n, t, _ in ocr_heading_readings("10.1OPREVENTIVE MAINTENANCE")}
    assert ((10, 10), "PREVENTIVE MAINTENANCE") in readings
    assert ((10, 1), "OPREVENTIVE MAINTENANCE") in readings
    assert ((10, 10, 3), "CLEANING OFPIPES") in {
        (n, t) for n, t, _ in ocr_heading_readings("1010.3CLEANING OFPIPES")
    }
    assert ((10, 10, 2), "WASTE") in {(n, t) for n, t, _ in ocr_heading_readings("10.J0.2WASTE")}
    assert ocr_heading_readings("Wastage is due to leakage") == []
    assert all(len(n) <= 3 for n, _, _ in ocr_heading_readings("7.6.8.1 Constant Rate"))


def test_a_heading_must_continue_the_section_sequence():
    assert follows((10, 1), 10, None)  # first heading after "CHAPTER 10"
    assert follows((10, 10), 10, (10, 9, 4))
    assert follows((10, 10, 2), 10, (10, 10))  # 10.10.1 missed
    assert not follows((10, 1), 10, (10, 9, 4))  # backwards
    assert not follows((10, 14), 10, (10, 10, 2))  # too far ahead
    assert follows((11, 1), 10, (10, 12))  # next chapter without a CHAPTER line
    assert not follows((2, 1), 10, (10, 12))  # e.g. a table row "2.1 ..."


def test_run_together_capitals_are_split_into_known_words():
    vocab = Counter({"waste": 5, "assessment": 3, "and": 40, "detection": 4, "of": 30, "pipes": 6})
    assert split_caps("WASTEASSESSMENTANDDETECTION", vocab) == "WASTE ASSESSMENT AND DETECTION"
    assert split_caps("CLEANING OFPIPES", vocab) == "CLEANING OF PIPES"
    assert split_caps("ZQXWVUTSR", vocab) == "ZQXWVUTSR"  # unknown: kept
    assert word_counts(["Waste and waste\nTHE CAPS LINE"]) == Counter({"waste": 2, "and": 1})


def test_the_reading_that_continues_the_sequence_wins():
    vocab = Counter({"preventive": 3, "maintenance": 4})
    assert ocr_heading("10.1OPREVENTIVEMAINTENANCE", 10, (10, 9, 4)) == (
        (10, 10),
        "PREVENTIVEMAINTENANCE",
    )
    assert split_caps("PREVENTIVEMAINTENANCE", vocab) == "PREVENTIVE MAINTENANCE"
    assert ocr_heading("10.2 The pipes are laid", 10, (10, 1)) is None  # running text
    assert ocr_heading("10.2.1 Zoning", 10, (10, 2)) == ((10, 2, 1), "Zoning")


SCAN = [
    "MANUAL ON WATER SUPPLY\nCONTENTS\n10.2.2 Waste Assessment and Detection 382",
    "CHAPTER10\nDISTRIBUTION SYSTEM\n10.1GENERAL\nThe distribution system conveys water.\n"
    "Assessment of waste and detection of leaks and cleaning of pipes matter.",
    "381\nService pipes of the house.\n10.2PREVENTIVEMAINTENANCE\n10.2.1 GENERAL\n"
    "Preventive maintenance of pipes.",
    "382\n1O.2.2WASTEASSESSMENTANDDETECTION\nWastage is due to leakage in mains.\n"
    "15 to 25m between zones.\nWaste assessment and detection by night flow; cleaning of pipes.",
    "386\nSounding rods.\n1023CLEANING OFPIPES\nFlushing and swabbing; preventive maintenance.",
]


def test_ocr_pages_are_chunked_by_repaired_headings():
    chunks = chunk_pdf_pages(SCAN, INFO, max_chars=1000, overlap=100, ocr=True)
    got = [(c.chapter, c.section, c.page, c.printed_page) for c in chunks]
    ch = "Ch. 10 Distribution System"
    assert got == [
        (ch, "§10.1 General", 2, None),
        (ch, "§10.2 Preventive Maintenance", 3, "381"),
        (ch, "§10.2.1 General", 3, "381"),
        (ch, "§10.2.2 Waste Assessment and Detection", 4, "382"),
        (ch, "§10.2.3 Cleaning of Pipes", 5, "386"),
    ]
    assert "15 to 25m" in chunks[3].text  # a number at a line start is not a heading
    assert "CONTENTS" not in " ".join(c.text for c in chunks)  # front matter skipped
    assert chunks[3].citation == "[WS Manual 1999, §10.2.2 Waste Assessment and Detection, p. 4]"


def test_misread_heading_words_are_corrected_from_the_running_text():
    vocab = Counter({"preventive": 4, "maintenance": 9, "water": 50, "works": 6, "hdpe": 1})
    assert split_caps("PREVENTTVEMAINTENANCE", vocab) == "PREVENTIVE MAINTENANCE"
    assert split_caps("WATER WORRS", vocab) == "WATER WORKS"
    assert split_caps("HDPE PIPES", vocab) == "HDPE PIPES"  # unknown: unchanged


def test_appendices_are_sections_of_their_own():
    pages = [
        "560\nCHAPTER18\nLEGAL ASPECTS\n18.1GENERAL\nLaws on water. Laws on pollution.",
        "573\nAPPENDIX10.1\nCALCULATIONOFCAPACITY\n1.1 Population 2,16,479\nCapacity of the tank.",
        "574\n2.1 DEMAND\nMore capacity of water.\nAPPENDIX B\nCONVERSION FACTORS\n1 inch 25.4 mm",
    ]
    vocab_lines = "capacity of the tank calculation calculation of capacity"
    chunks = chunk_pdf_pages([*pages, vocab_lines], INFO, ocr=True)
    assert [(c.chapter, c.section, c.page) for c in chunks] == [
        ("Ch. 18 Legal Aspects", "§18.1 General", 1),
        (None, "Appendix 10.1 Calculation of Capacity", 2),
        (None, "Appendix B Conversion Factors", 3),
    ]
    assert "2.1 DEMAND" in chunks[1].text  # numbered lines in an appendix are not headings


def test_ocr_chapter_lines_and_a_five_read_as_s():
    pages = [
        "49\nCHAPTER'5\nSOURCES OF SUPPLY\nS.1KINDS OF SOURCES\nWater sources of supply.\n"
        "5.1.1 Rain Water\nRain water of sources.",
    ]
    chunks = chunk_pdf_pages(pages, INFO, ocr=True)
    assert [(c.chapter, c.section) for c in chunks] == [
        ("Ch. 5 Sources of Supply", "§5.1 Kinds of Sources"),
        ("Ch. 5 Sources of Supply", "§5.1.1 Rain Water"),
    ]
