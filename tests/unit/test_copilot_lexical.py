"""Hybrid retrieval (D-039): BM25 over OCR-folded words, fused with embeddings."""

from __future__ import annotations

import numpy as np

from leakwatch.copilot.rag.embed import HashingEmbedder
from leakwatch.copilot.rag.index import Retriever, VectorIndex
from leakwatch.copilot.rag.lexical import Bm25, fold, rrf_fuse, tokens

TEXTS = [
    "Levcls of wastage upto 10% may be considered as low, 10 to 20% as average, 20 to 50% as "
    "excessivc and over 50% as alarming.",
    "Valves should be exercised once a year to keep them operable.",
    "Chlorine residual at the tail end of the network should be at least 0.2 mg/l.",
    "Night flow measurement helps find waste in a district metered area.",
]


def test_fold_unifies_ocr_confusions():
    assert fold("excessivc") == fold("Excessive")
    assert fold("Levcls") == fold("levels")
    assert tokens("What is the level of wastage?") == [fold("level"), fold("wastage")]


def test_bm25_finds_misspelt_passage_and_counts_terms():
    bm = Bm25(TEXTS)
    s = bm.scores("Which wastage levels are excessive or alarming?")
    assert int(np.argmax(s)) == 0 and s[1] == 0.0
    assert bm.matched_terms("wastage levels alarming")[0] == 3
    assert bm.scores("").sum() == 0.0


def test_rrf_fuse_rewards_agreement():
    f = rrf_fuse([np.array([2, 0, 1]), np.array([0, 2])], rrf_k=60, n=3)
    assert f[0] > f[1] and f[2] > f[1]


def test_hybrid_retriever_returns_lexical_hit_below_min_score():
    emb = HashingEmbedder(64)
    chunks = [
        {"doc_id": "d", "section": f"s{i}", "text": t, "citation": f"c{i}"}
        for i, t in enumerate(TEXTS)
    ]
    index = VectorIndex(emb.encode(TEXTS), chunks, {"embedder": emb.name})
    q = "low average excessive alarming wastage"
    dense = Retriever(index, emb).search(q, k=2, min_score=0.99)
    assert dense == []  # nothing reaches the cosine gate
    hybrid = Retriever(index, emb, {"enabled": True}).search(q, k=2, min_score=0.99)
    assert hybrid and hybrid[0]["section"] == "s0" and hybrid[0]["lexical"] is True
    one_word = Retriever(index, emb, {"enabled": True, "min_lexical_terms": 2}).search(
        "valves of rain", k=2, min_score=0.99
    )
    assert all(h["section"] == "s1" for h in one_word)  # the only word overlap
