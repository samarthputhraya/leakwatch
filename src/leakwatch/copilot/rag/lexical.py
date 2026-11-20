"""Lexical (BM25) search over the passages, tolerant of OCR letter confusions (D-039).

Embeddings miss passages whose key words are misspelt by OCR (``excessivc``, ``Levcls``) or are
rare terms the embedding model does not weigh ("alarming", "MNF"). BM25 matches exact words, and
:func:`fold` maps the letters OCR confuses to one class on both sides (query and passage):
``c``/``e``, ``l``/``i``/``1``, ``o``/``0``, ``s``/``5``. "excessivc" and "excessive" both fold to
"exeessive", so they match; legitimate words fold consistently, so nothing is lost.

The retriever fuses the BM25 ranking with the embedding ranking by reciprocal rank fusion
(``1 / (rrf_k + rank)`` summed; Cormack et al. 2009).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

import numpy as np
from scipy.sparse import csr_matrix

_TOKEN = re.compile(r"[a-z0-9]+")
_FOLD = str.maketrans({"c": "e", "l": "i", "1": "i", "0": "o", "5": "s"})
#: Very common words that carry no topic (stored in folded spelling).
_WORDS = (
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "which",
    "with",
    "what",
    "how",
    "does",
    "do",
    "should",
    "we",
    "our",
    "can",
    "not",
    "no",
    "if",
    "when",
    "than",
    "then",
    "there",
    "these",
    "they",
    "will",
    "shall",
    "may",
    "per",
)
STOPWORDS = frozenset(w.translate(_FOLD) for w in _WORDS)


def fold(text: str) -> str:
    """Lower-case and map OCR-confused characters to one class."""
    return text.lower().translate(_FOLD)


def tokens(text: str) -> list[str]:
    """Folded word tokens of at least 2 characters, without stopwords."""
    return [t for t in _TOKEN.findall(fold(text)) if len(t) > 1 and t not in STOPWORDS]


class Bm25:
    """Okapi BM25 over a fixed list of passages (sparse term weights, computed once)."""

    def __init__(self, texts: Sequence[str], k1: float = 1.2, b: float = 0.75) -> None:
        """Index ``texts`` (one passage each)."""
        docs = [Counter(tokens(t)) for t in texts]
        vocab: dict[str, int] = {}
        rows, cols, tfs = [], [], []
        for i, doc in enumerate(docs):
            for term, tf in doc.items():
                rows.append(i)
                cols.append(vocab.setdefault(term, len(vocab)))
                tfs.append(tf)
        n = len(docs)
        lengths = np.array([sum(d.values()) for d in docs], dtype="float64")
        avg = lengths.mean() if n and lengths.mean() > 0 else 1.0
        df = np.bincount(np.asarray(cols, dtype="int64"), minlength=len(vocab)).astype("float64")
        idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5))
        tf = np.asarray(tfs, dtype="float64")
        r = np.asarray(rows, dtype="int64")
        c = np.asarray(cols, dtype="int64")
        weight = idf[c] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lengths[r] / avg))
        self.vocab = vocab
        self.n = n
        self.matrix = csr_matrix((weight, (r, c)), shape=(n, max(len(vocab), 1)))

    def _terms(self, query: str) -> list[int]:
        return [self.vocab[t] for t in set(tokens(query)) if t in self.vocab]

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every passage for ``query`` (0 where no query word occurs)."""
        q = self._terms(query)
        if not q or self.n == 0:
            return np.zeros(self.n)
        return np.asarray(self.matrix[:, q].sum(axis=1)).ravel()

    def matched_terms(self, query: str) -> np.ndarray:
        """How many distinct query words each passage contains."""
        q = self._terms(query)
        if not q or self.n == 0:
            return np.zeros(self.n, dtype="int64")
        return np.asarray((self.matrix[:, q] > 0).sum(axis=1)).ravel().astype("int64")


def rrf_fuse(rankings: Sequence[np.ndarray], rrf_k: float, n: int) -> np.ndarray:
    """Reciprocal-rank-fusion score per item from several rankings (arrays of item indices)."""
    fused = np.zeros(n)
    for ranking in rankings:
        fused[ranking] += 1.0 / (rrf_k + np.arange(1, len(ranking) + 1))
    return fused
