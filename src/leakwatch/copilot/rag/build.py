"""``make index``: build the copilot's manual index (PRD §9.5).

Corpus:

1. our own knowledge base, ``docs/kb/*.md`` (leak-response SOP, detection methodology, L-Town
   sensors and areas);
2. the PDFs listed in ``copilot.rag.documents`` that are present in ``paths.docs_data_dir()``.
   A listed PDF that is missing is downloaded from its ``url`` when it has one (and its SHA-256 is
   checked when configured); otherwise it is skipped with a warning (the owner places it by hand);
3. any other ``*.pdf`` / ``*.md`` dropped into ``paths.docs_data_dir()`` (title = file name).

Passages are embedded with ``copilot.rag.embedding_model`` and written to
``paths.index_dir() / copilot.rag.index_name``. A summary goes to
``reports/metrics/rag_index.json`` (documents, passages per document, skipped files and why).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from leakwatch.common import paths
from leakwatch.common.logging import get_logger
from leakwatch.common.metrics import write_metrics
from leakwatch.copilot.rag.chunking import (
    Chunk,
    DocumentInfo,
    chunk_markdown,
    chunk_pdf,
    markdown_info,
)
from leakwatch.copilot.rag.embed import Embedder, get_embedder
from leakwatch.copilot.rag.index import VectorIndex, index_path
from leakwatch.copilot.settings import copilot_config

log = get_logger("leakwatch.copilot.rag.build")


def sha256_of(path: Path) -> str:
    """Hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_document(spec: Mapping[str, Any], target: Path, timeout_s: float) -> str:
    """Download one configured PDF to ``target``; returns a status string (never raises)."""
    import requests

    url = spec.get("url")
    if not url:
        return "missing: no download URL (place the PDF in data/docs by hand)"
    part = target.with_name(target.name + ".part")
    try:
        with requests.get(str(url), stream=True, timeout=timeout_s) as resp:
            resp.raise_for_status()
            with part.open("wb") as fh:
                for block in resp.iter_content(chunk_size=1 << 20):
                    fh.write(block)
    except requests.RequestException as exc:
        part.unlink(missing_ok=True)
        return f"download failed: {type(exc).__name__}: {str(exc)[:160]}"
    with part.open("rb") as fh:
        if fh.read(5) != b"%PDF-":
            part.unlink(missing_ok=True)
            return "download failed: the server did not return a PDF"
    expected = spec.get("sha256")
    if expected and sha256_of(part) != expected:
        part.unlink(missing_ok=True)
        return "download failed: SHA-256 differs from the configured value"
    part.rename(target)
    return "downloaded"


def collect_documents(
    cfg: Mapping[str, Any], download: bool = True
) -> tuple[list[DocumentInfo], dict[str, str]]:
    """Documents to index and a status line for every configured or found file."""
    rag = cfg["rag"]
    status: dict[str, str] = {}
    docs: list[DocumentInfo] = []
    kb_dir = paths.REPO_ROOT / str(rag["kb_dir"])
    for path in sorted(kb_dir.glob("*.md")):
        info, _ = markdown_info(path)
        docs.append(info)
        status[info.doc_id] = "indexed (knowledge base)"
    docs_dir = paths.docs_data_dir()
    configured_files: set[str] = set()
    for spec in rag.get("documents") or []:
        target = docs_dir / str(spec["file"])
        configured_files.add(target.name)
        if not target.exists() and download:
            paths.ensure_dir(docs_dir)
            result = download_document(spec, target, float(rag.get("download_timeout_s", 300)))
            log.info("%s: %s", spec["id"], result)
            if not target.exists():
                status[str(spec["id"])] = result
                continue
        elif not target.exists():
            status[str(spec["id"])] = "missing (download disabled)"
            continue
        expected = spec.get("sha256")
        if expected and sha256_of(target) != expected:
            log.warning("%s: SHA-256 differs from the configured value; indexing anyway", target)
        docs.append(
            DocumentInfo(
                doc_id=str(spec["id"]),
                title=str(spec["title"]),
                short=str(spec.get("short") or spec["title"]),
                source_type="pdf",
                path=str(target),
                url=spec.get("url"),
            )
        )
        status[str(spec["id"])] = "indexed"
    if docs_dir.exists():
        for path in sorted(docs_dir.iterdir()):
            if path.name in configured_files or path.suffix.lower() not in (".pdf", ".md"):
                continue
            if path.suffix.lower() == ".md":
                info, _ = markdown_info(path, doc_id=f"doc_{path.stem}")
            else:
                info = DocumentInfo(
                    doc_id=f"doc_{path.stem}",
                    title=path.stem.replace("_", " "),
                    short=path.stem.replace("_", " "),
                    source_type="pdf",
                    path=str(path),
                )
            docs.append(info)
            status[info.doc_id] = "indexed (found in data/docs)"
    return docs, status


def chunk_documents(docs: Sequence[DocumentInfo], cfg: Mapping[str, Any]) -> list[Chunk]:
    """Passages of every document."""
    rag = cfg["rag"]
    max_chars, overlap = int(rag["chunk_max_chars"]), int(rag["chunk_overlap_chars"])
    chunks: list[Chunk] = []
    for info in docs:
        path = Path(info.path)
        if info.source_type == "markdown":
            _, body = markdown_info(path, doc_id=info.doc_id)
            found = chunk_markdown(body, info, max_chars, overlap)
        else:
            found = chunk_pdf(path, info, max_chars, overlap)
        log.info("%s: %d passages", info.doc_id, len(found))
        chunks.extend(found)
    return chunks


def build_index(
    docs: Sequence[DocumentInfo], embedder: Embedder, cfg: Mapping[str, Any]
) -> VectorIndex:
    """Chunk and embed ``docs``."""
    chunks = chunk_documents(docs, cfg)
    texts = [f"{c.section}\n{c.text}" for c in chunks]
    vectors = embedder.encode(texts) if texts else np.zeros((0, 1), dtype=np.float32)
    per_doc: dict[str, int] = {}
    for c in chunks:
        per_doc[c.doc_id] = per_doc.get(c.doc_id, 0) + 1
    meta = {
        "embedder": embedder.name,
        "dim": int(vectors.shape[1]) if len(vectors) else None,
        "n_chunks": len(chunks),
        "documents": [
            {
                "doc_id": d.doc_id,
                "title": d.title,
                "short": d.short,
                "source_type": d.source_type,
                "file": Path(d.path).name,
                "url": d.url,
                "chunks": per_doc.get(d.doc_id, 0),
            }
            for d in docs
        ],
        "chunk_max_chars": int(cfg["rag"]["chunk_max_chars"]),
        "chunk_overlap_chars": int(cfg["rag"]["chunk_overlap_chars"]),
    }
    return VectorIndex(embeddings=vectors, chunks=[c.to_dict() for c in chunks], meta=meta)


def main(argv: list[str] | None = None) -> int:
    """Build the index; ``--no-download`` skips fetching missing PDFs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-download", action="store_true", help="do not fetch missing PDFs")
    parser.add_argument("--embedder", help="override copilot.rag.embedding_model")
    args = parser.parse_args(argv)
    cfg = copilot_config()
    t0 = time.perf_counter()
    docs, status = collect_documents(cfg, download=not args.no_download)
    if not docs:
        log.error("nothing to index: no docs/kb/*.md and no PDFs in %s", paths.docs_data_dir())
        return 1
    name = args.embedder or str(cfg["rag"]["embedding_model"])
    embedder = get_embedder(name, int(cfg["rag"].get("embed_batch_size", 32)))
    index = build_index(docs, embedder, cfg)
    target = index.save(index_path(cfg))
    seconds = round(time.perf_counter() - t0, 1)
    log.info(
        "index: %d passages from %d documents -> %s (%.1f s)",
        len(index.chunks),
        len(docs),
        target,
        seconds,
    )
    for doc_id, line in status.items():
        if not line.startswith("indexed"):
            log.warning("not indexed: %s (%s)", doc_id, line)
    write_metrics(
        "rag_index",
        {
            "embedder": index.meta["embedder"],
            "dim": index.meta["dim"],
            "passages": index.meta["n_chunks"],
            "documents": index.meta["documents"],
            "status": status,
            "build_seconds": seconds,
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
