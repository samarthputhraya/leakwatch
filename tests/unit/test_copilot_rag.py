from __future__ import annotations

import json

import numpy as np
import pytest

from leakwatch.common import paths
from leakwatch.common.config import load_config
from leakwatch.copilot.rag import build
from leakwatch.copilot.rag.chunking import (
    DocumentInfo,
    chunk_markdown,
    chunk_pdf_pages,
    decode_glyphs,
    markdown_info,
    read_front_matter,
    title_case,
)
from leakwatch.copilot.rag.embed import HashingEmbedder, get_embedder
from leakwatch.copilot.rag.index import RagIndexError, Retriever, VectorIndex

CFG = load_config(environ={})["copilot"]
MD_INFO = DocumentInfo("kb_test", "Test SOP", "Test SOP", "markdown", "x.md")
PDF_INFO = DocumentInfo("manual", "A Manual", "Manual 2005", "pdf", "x.pdf")


def test_front_matter_and_markdown_sections():
    meta, body = read_front_matter(
        "---\nshort: Test SOP\n---\n# Title\n\nIntro text.\n\n## 1. First\nAlpha.\n\n"
        "## 2. Second\nBeta.\n### Detail\nGamma.\n## 3. Empty\n"
    )
    assert meta == {"short": "Test SOP"}
    chunks = chunk_markdown(body, MD_INFO)
    assert [c.section for c in chunks] == [
        "Test SOP",
        "1. First",
        "2. Second",
        "2. Second › Detail",
    ]
    assert chunks[1].citation == "[Test SOP, 1. First]" and chunks[1].page is None
    assert chunks[3].chapter == "2. Second"


def test_long_sections_are_windowed_with_overlap():
    body = "## Long\n" + "\n".join(f"line {i:03d} " + "x" * 40 for i in range(60))
    chunks = chunk_markdown(body, MD_INFO, max_chars=500, overlap=100)
    assert len(chunks) > 3 and all(len(c.text) <= 500 for c in chunks)
    assert chunks[0].text.splitlines()[-1] == chunks[1].text.splitlines()[1]  # carried line


def test_title_case_keeps_acronyms_and_small_words():
    assert title_case("LEAKAGE  DETECTION AND MONITORING") == "Leakage Detection and Monitoring"
    assert (
        title_case("PREVENTION OF UFW IN CONSUMER CONNECTION")
        == "Prevention of UFW in Consumer Connection"
    )
    assert title_case("PHYSICAL LOSSES (TECHNICAL LOSSES)") == "Physical Losses (Technical Losses)"
    assert title_case("Preliminary data collection") == "Preliminary data collection"


def test_glyph_decoding():
    assert decode_glyphs("/G31/G35/G2E/G31/G30/G20/G4C/G45/G41/G4B") == "15.10 LEAK"


PAGES = [
    "FOREWORD\nSome front matter.",
    "13\nCHAPTER 14\nSYSTEM MANAGEMENT\n14.1 GENERAL\nWater supply basics.",
    "CONTENTS\n15.1 Introduction ........................ 343\n15.2 Objective ........... 344",
    "342\n35.1 Number of leaks reported per day Number of leaks/km\nAnnex table row.",
    "343\n/G31/G35/G2E/G31/G20/G49/G4E/G54/G52/G4F\nWater audit text.\n"
    "CHAPTER 15\nWATER AUDIT AND\nLEAKAGE CONTROL\nMore text.",
    "344\n15.2 OBJECTIVE OF WATER AUDIT\nObjective text.\n15.2.1 Mixed case subheading\nSub text.",
]


def test_pdf_pages_are_chunked_by_chapter_and_section():
    chunks = chunk_pdf_pages(PAGES, PDF_INFO, max_chars=1000, overlap=100)
    sections = [(c.chapter, c.section, c.page, c.printed_page) for c in chunks]
    assert sections == [
        ("Ch. 14 System Management", "§14.1 General", 2, "13"),
        ("Ch. 15 Water Audit and Leakage Control", "§15.1 Intro", 5, "343"),
        ("Ch. 15 Water Audit and Leakage Control", "§15.2 Objective of Water Audit", 6, "344"),
        ("Ch. 15 Water Audit and Leakage Control", "§15.2.1 Mixed case subheading", 6, "344"),
    ]
    text = " ".join(c.text for c in chunks)
    assert (
        "front matter" not in text and "........" not in text
    )  # front matter and contents skipped
    assert "35.1 Number of leaks" in chunks[0].text and chunks[0].page_end == 4  # not a heading
    assert "WATER AUDIT AND" not in chunks[1].text  # chapter title lines are not body text
    assert chunks[2].citation == "[Manual 2005, §15.2 Objective of Water Audit, p. 6]"


def test_hashing_embedder_is_deterministic_and_normalised():
    emb = HashingEmbedder(64)
    a = emb.encode(["leak detection by sounding", "water audit"])
    b = emb.encode(["leak detection by sounding", "water audit"])
    assert np.array_equal(a, b) and np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert get_embedder("hashing-64").name == "hashing-64"


def test_index_round_trip_and_search(tmp_path):
    emb = HashingEmbedder()
    texts = [
        "sounding with a listening stick locates leaks",
        "billing and collection of water charges",
        "chlorine dose",
    ]
    chunks = [
        {"doc_id": "d", "citation": f"[d, s{i}]", "text": t, "section": f"s{i}"}
        for i, t in enumerate(texts)
    ]
    index = VectorIndex(emb.encode(texts), chunks, {"embedder": emb.name})
    index.save(tmp_path / "idx")
    loaded = VectorIndex.load(tmp_path / "idx")
    retriever = Retriever(loaded, emb)
    hits = retriever.search("listening stick to locate leaks", k=2)
    assert hits[0]["citation"] == "[d, s0]" and hits[0]["score"] >= hits[1]["score"]
    assert retriever.search("listening stick", k=3, min_score=0.99) == []
    with pytest.raises(RagIndexError, match="make index"):
        Retriever(loaded, HashingEmbedder(32))
    with pytest.raises(RagIndexError):
        VectorIndex.load(tmp_path / "missing")


def test_build_index_over_the_knowledge_base(tmp_path, monkeypatch):
    monkeypatch.setenv("LEAKWATCH_DATA_DIR", str(tmp_path))
    paths.ensure_dir(paths.docs_data_dir())
    (paths.docs_data_dir() / "extra_notes.md").write_text(
        "# Extra\n\n## Valves\nClose slowly.\n", encoding="utf-8"
    )
    docs, status = build.collect_documents(CFG, download=False)
    ids = [d.doc_id for d in docs]
    assert {
        "kb_leak_response_sop",
        "kb_detection_methodology",
        "kb_l_town_network",
        "doc_extra_notes",
    } <= set(ids)
    assert status["cpheeo_om_2005"].startswith("missing")
    index = build.build_index(docs, HashingEmbedder(), CFG)
    assert index.meta["embedder"] == "hashing-256" and index.meta["n_chunks"] == len(index.chunks)
    sections = {(c["doc_short"], c["section"]) for c in index.chunks}
    assert ("LeakWatch SOP", "6. Confirm the leak before digging") in sections
    json.dumps(index.meta)


def test_eval_expected_kb_sections_exist():
    """Every knowledge-base section an eval item expects to be cited exists in docs/kb."""
    from leakwatch.copilot.evaluate import load_items

    kb = {}
    for path in sorted((paths.REPO_ROOT / CFG["rag"]["kb_dir"]).glob("*.md")):
        info, body = markdown_info(path)
        kb[info.short] = [c.section for c in chunk_markdown(body, info)]
    for item in load_items():
        for src in item.get("expected_sources", []):
            if src["doc"] in kb:
                assert any(src["section"] in s for s in kb[src["doc"]]), (item["id"], src)


@pytest.mark.slow
def test_real_embedding_model_finds_the_sop_section(tmp_path, monkeypatch):
    pytest.importorskip("sentence_transformers")
    monkeypatch.setenv("LEAKWATCH_DATA_DIR", str(tmp_path))
    docs, _ = build.collect_documents(CFG, download=False)
    emb = get_embedder(CFG["rag"]["embedding_model"])
    retriever = Retriever(build.build_index(docs, emb, CFG), emb)
    hits = retriever.search("How do we confirm a leak before digging?", k=3)
    assert any(h["section"] == "6. Confirm the leak before digging" for h in hits)
