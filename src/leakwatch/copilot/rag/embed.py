"""Text embedders for the RAG index.

- :class:`SentenceTransformerEmbedder`: ``all-MiniLM-L6-v2`` on CPU (downloaded from the
  Hugging Face hub on first use, ~90 MB, cached in ``~/.cache/huggingface``).
- :class:`HashingEmbedder`: a deterministic bag-of-words embedder with no model and no network,
  for unit tests. Its name is stored in the index, so an index built with it is never searched
  with the real model by mistake.

Both return float32 rows with unit length, so a dot product is the cosine similarity.
"""

from __future__ import annotations

import hashlib
import itertools
import re
from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

import numpy as np

HASHING_PREFIX = "hashing"
_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    """Turns texts into unit-length vectors."""

    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """``(len(texts), dim)`` float32 array, rows L2-normalised."""
        ...


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


class HashingEmbedder:
    """Signed feature hashing of words and word pairs (tests only)."""

    def __init__(self, dim: int = 256) -> None:
        """Create the embedder with ``dim`` hash buckets."""
        self.dim = dim
        self.name = f"{HASHING_PREFIX}-{dim}"

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        h = int.from_bytes(digest, "little")
        return h % self.dim, 1.0 if (h >> 63) & 1 else -1.0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Hash each text's words (length >= 3) and adjacent word pairs into ``dim`` buckets."""
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        for i, text in enumerate(texts):
            words = [w for w in _TOKEN.findall(text.lower()) if len(w) >= 3]
            for token in words + [f"{a}_{b}" for a, b in itertools.pairwise(words)]:
                j, sign = self._bucket(token)
                out[i, j] += sign
        return _normalise(out)


class SentenceTransformerEmbedder:
    """A sentence-transformers model on CPU, loaded on first use."""

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        """Remember the model name; nothing is downloaded until :meth:`encode` runs."""
        self.name = model_name
        self.batch_size = batch_size
        self._model = None

    def _load(self):  # -> SentenceTransformer
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.name, device="cpu")
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Embed ``texts`` (normalised by the model)."""
        model = self._load()
        vectors = model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return _normalise(np.asarray(vectors, dtype=np.float32))


@lru_cache(maxsize=4)
def get_embedder(name: str, batch_size: int = 32) -> Embedder:
    """The embedder called ``name`` (``hashing-<dim>`` or a sentence-transformers model id)."""
    if name.startswith(HASHING_PREFIX):
        _, _, dim = name.partition("-")
        return HashingEmbedder(int(dim) if dim.isdigit() else 256)
    return SentenceTransformerEmbedder(name, batch_size)
