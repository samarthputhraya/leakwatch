"""A small numpy vector index on disk and the retriever the ``search_manual`` tool uses.

Layout (``paths.index_dir() / <index_name>``): ``embeddings.npy`` (float32, unit rows),
``chunks.jsonl`` (one passage per line, same order) and ``meta.json`` (embedder, sources, counts).
A few thousand passages need no FAISS/Chroma: a matrix-vector product is instant.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from leakwatch.common import paths
from leakwatch.copilot.rag.embed import Embedder, get_embedder

EMBEDDINGS_FILE = "embeddings.npy"
CHUNKS_FILE = "chunks.jsonl"
META_FILE = "meta.json"


class RagIndexError(RuntimeError):
    """The index is missing, incomplete or built with another embedder."""


@dataclass
class VectorIndex:
    """Passages and their embeddings."""

    embeddings: np.ndarray
    chunks: list[dict[str, Any]]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2 or len(self.embeddings) != len(self.chunks):
            raise RagIndexError("embeddings and chunks do not line up")

    def save(self, directory: Path) -> Path:
        """Write the three files, replacing an existing index only once all are written."""
        tmp = directory.with_name(directory.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        np.save(tmp / EMBEDDINGS_FILE, self.embeddings.astype(np.float32))
        with (tmp / CHUNKS_FILE).open("w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        (tmp / META_FILE).write_text(json.dumps(self.meta, indent=2) + "\n", encoding="utf-8")
        if directory.exists():
            shutil.rmtree(directory)
        tmp.rename(directory)
        return directory

    @classmethod
    def load(cls, directory: Path) -> VectorIndex:
        """Read an index written by :meth:`save`."""
        files = [directory / n for n in (EMBEDDINGS_FILE, CHUNKS_FILE, META_FILE)]
        if not all(f.exists() for f in files):
            raise RagIndexError(f"no index at {directory}: run `make index`")
        embeddings = np.load(files[0])
        with files[1].open(encoding="utf-8") as fh:
            chunks = [json.loads(line) for line in fh if line.strip()]
        meta = json.loads(files[2].read_text(encoding="utf-8"))
        return cls(embeddings=embeddings, chunks=chunks, meta=meta)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[float, dict[str, Any]]]:
        """The ``k`` passages most similar to ``query_vector`` (cosine), best first."""
        if not self.chunks:
            return []
        scores = self.embeddings @ query_vector.astype(np.float32).ravel()
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(float(scores[i]), self.chunks[i]) for i in top]


def index_path(cfg: Mapping[str, Any]) -> Path:
    """Directory of the configured index (``paths.index_dir() / rag.index_name``)."""
    return paths.index_dir() / str(cfg["rag"]["index_name"])


class Retriever:
    """Embeds a question and returns the best passages with their citations."""

    def __init__(
        self, index: VectorIndex, embedder: Embedder, hybrid: Mapping[str, Any] | None = None
    ) -> None:
        """Pair an index with the embedder it was built with.

        ``hybrid`` (``rag.hybrid``, D-039): when enabled, BM25 over OCR-folded words is fused
        with the embedding ranking.
        """
        built_with = index.meta.get("embedder")
        if built_with and built_with != embedder.name:
            raise RagIndexError(
                f"index built with {built_with!r} but the embedder is {embedder.name!r}: "
                "run `make index`"
            )
        self.index = index
        self.embedder = embedder
        self.hybrid = dict(hybrid or {})
        self.bm25 = None
        if self.hybrid.get("enabled") and index.chunks:
            from leakwatch.copilot.rag.lexical import Bm25

            self.bm25 = Bm25(
                [f"{c.get('section', '')} {c.get('text', '')}" for c in index.chunks],
                float(self.hybrid.get("k1", 1.2)),
                float(self.hybrid.get("b", 0.75)),
            )

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], directory: Path | None = None) -> Retriever:
        """Load the index under ``paths.index_dir()`` with the configured embedder."""
        index = VectorIndex.load(directory or index_path(cfg))
        name = str(index.meta.get("embedder") or cfg["rag"]["embedding_model"])
        return cls(
            index,
            get_embedder(name, int(cfg["rag"].get("embed_batch_size", 32))),
            cfg["rag"].get("hybrid"),
        )

    def search(self, query: str, k: int = 4, min_score: float = 0.0) -> list[dict[str, Any]]:
        """The best ``k`` passages, each with its cosine ``score``.

        Dense only: passages with cosine ≥ ``min_score``. Hybrid: the reciprocal-rank fusion of
        the embedding and BM25 rankings; a passage is kept if its cosine reaches ``min_score``
        or it is among the top ``k`` lexical matches that share ``min_lexical_terms`` of the
        question's words (``lexical: true``).
        """
        vector = self.embedder.encode([query])[0]
        if self.bm25 is None:
            return [
                {**chunk, "score": round(score, 4)}
                for score, chunk in self.index.search(vector, k)
                if score >= min_score
            ]
        from leakwatch.copilot.rag.lexical import rrf_fuse

        cos = self.index.embeddings @ vector.astype(np.float32).ravel()
        lex = self.bm25.scores(query)
        pool = min(int(self.hybrid.get("pool", 50)), len(cos))
        dense_rank = np.argsort(-cos, kind="stable")[:pool]
        lex_rank = np.argsort(-lex, kind="stable")[:pool]
        lex_rank = lex_rank[lex[lex_rank] > 0]
        fused = rrf_fuse([dense_rank, lex_rank], float(self.hybrid.get("rrf_k", 60)), len(cos))
        # a passage kept on words alone must share several of the question's words
        matched = self.bm25.matched_terms(query)
        need = min(int(self.hybrid.get("min_lexical_terms", 2)), int(matched.max(initial=0)))
        lexical_top = {int(i) for i in lex_rank if matched[i] >= max(need, 1)}
        lexical_top = set(sorted(lexical_top, key=lambda i: -lex[i])[:k])
        order = np.argsort(-fused, kind="stable")
        out = []
        for i in order:
            if fused[i] <= 0 or len(out) >= k:
                break
            if cos[i] >= min_score or int(i) in lexical_top:
                out.append(
                    {
                        **self.index.chunks[i],
                        "score": round(float(cos[i]), 4),
                        "lexical": int(i) in lexical_top,
                    }
                )
        return out

    @property
    def documents(self) -> list[str]:
        """Ids of the documents in the index."""
        return sorted({str(c["doc_id"]) for c in self.index.chunks})
