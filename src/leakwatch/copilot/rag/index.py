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

    def __init__(self, index: VectorIndex, embedder: Embedder) -> None:
        """Pair an index with the embedder it was built with."""
        built_with = index.meta.get("embedder")
        if built_with and built_with != embedder.name:
            raise RagIndexError(
                f"index built with {built_with!r} but the embedder is {embedder.name!r}: "
                "run `make index`"
            )
        self.index = index
        self.embedder = embedder

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], directory: Path | None = None) -> Retriever:
        """Load the index under ``paths.index_dir()`` with the configured embedder."""
        index = VectorIndex.load(directory or index_path(cfg))
        name = str(index.meta.get("embedder") or cfg["rag"]["embedding_model"])
        return cls(index, get_embedder(name, int(cfg["rag"].get("embed_batch_size", 32))))

    def search(self, query: str, k: int = 4, min_score: float = 0.0) -> list[dict[str, Any]]:
        """Passages scoring at least ``min_score``, best first, each with ``score``."""
        vector = self.embedder.encode([query])[0]
        return [
            {**chunk, "score": round(score, 4)}
            for score, chunk in self.index.search(vector, k)
            if score >= min_score
        ]

    @property
    def documents(self) -> list[str]:
        """Ids of the documents in the index."""
        return sorted({str(c["doc_id"]) for c in self.index.chunks})
