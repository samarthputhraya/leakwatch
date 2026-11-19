"""Text for scanned PDFs (no text layer): OCR on rendered pages, cached next to the PDF.

The CPHEEO Manual on Water Supply and Treatment (1999) is online only as a 777-page scan
(bilevel images, ~190 dpi), so ``pypdf`` finds no text in it. For such a PDF ``make index``
renders every page with ``pypdfium2``, runs RapidOCR (PP-OCR detection and recognition models
on ONNX Runtime, CPU only) and groups the recognised text boxes into lines, top to bottom.

Results are cached page by page in ``<file>.ocr.jsonl`` next to the PDF, so re-indexing (or
resuming an interrupted run) does not OCR again. The first line of the cache records the
engine, the settings and the PDF's SHA-256; a cache made from another file or with other
settings is rebuilt.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from leakwatch.common.logging import get_logger

log = get_logger("leakwatch.copilot.rag.ocr")

# One recognised text box: four corner points, the text, the recognition score.
Box = tuple[Sequence[Sequence[float]], str, float]

_FOLIO = re.compile(r"^\d{1,4}\.?$")


@dataclass(frozen=True)
class OcrSettings:
    """How pages are rendered and recognised (``copilot.rag.ocr``)."""

    dpi: int = 200
    min_score: float = 0.5  # recognition score below which a box is dropped
    workers: int = 4  # OCR processes
    threads_per_worker: int = 4  # ONNX Runtime threads per process

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> OcrSettings:
        """Settings from the ``copilot.rag.ocr`` mapping (missing keys keep the defaults)."""
        cfg = cfg or {}
        return cls(
            dpi=int(cfg.get("dpi", cls.dpi)),
            min_score=float(cfg.get("min_score", cls.min_score)),
            workers=max(1, int(cfg.get("workers", cls.workers))),
            threads_per_worker=int(cfg.get("threads_per_worker", cls.threads_per_worker)),
        )


def text_share(pages: Sequence[str], min_chars: int = 50) -> float:
    """Share of pages whose text layer has at least ``min_chars`` non-blank characters."""
    if not pages:
        return 0.0
    return sum(len(p.strip()) >= min_chars for p in pages) / len(pages)


def needs_ocr(pages: Sequence[str], min_text_share: float) -> bool:
    """True when too few pages have a text layer (a scanned document)."""
    return bool(pages) and text_share(pages) < min_text_share


def group_lines(boxes: Iterable[Box], min_score: float = 0.0) -> list[str]:
    """Text boxes -> lines of text, top to bottom and left to right within a line.

    A box joins the current line when its vertical centre is within 60% of a line height of
    the line's first box; boxes below ``min_score`` are dropped.
    """
    items = []
    for points, text, score in boxes:
        if float(score) < min_score or not str(text).strip():
            continue
        ys = [float(p[1]) for p in points]
        x0 = min(float(p[0]) for p in points)
        items.append(((min(ys) + max(ys)) / 2, x0, max(ys) - min(ys), str(text).strip()))
    items.sort()
    lines: list[str] = []
    current: list[tuple[float, str]] = []
    ref: float | None = None
    for y, x, height, text in items:
        if ref is not None and abs(y - ref) > max(height, 12.0) * 0.6:
            lines.append(" ".join(t for _, t in sorted(current)))
            current, ref = [], None
        current.append((x, text))
        ref = y if ref is None else ref
    if current:
        lines.append(" ".join(t for _, t in sorted(current)))
    return lines


def folio_first(lines: list[str]) -> list[str]:
    """Move a page number printed at the foot of the page to the first line.

    The chunker reads the printed page number from the first line (as in PDFs with a text
    layer, where the number comes first); scanned pages carry it at the bottom.
    """
    if len(lines) > 1 and _FOLIO.match(lines[-1]) and not _FOLIO.match(lines[0]):
        return [lines[-1].rstrip("."), *lines[:-1]]
    return lines


def page_text(boxes: Iterable[Box], min_score: float) -> str:
    """Text of one page from its OCR boxes."""
    return "\n".join(folio_first(group_lines(boxes, min_score)))


def sha256_of(path: Path) -> str:
    """Hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def engine_name() -> str:
    """OCR engine and version recorded in the cache."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return f"rapidocr-onnxruntime {version('rapidocr-onnxruntime')}"
    except PackageNotFoundError:
        return "rapidocr-onnxruntime (not installed)"


def cache_path(pdf: Path) -> Path:
    """``<file>.ocr.jsonl`` next to the PDF."""
    return pdf.with_name(pdf.name + ".ocr.jsonl")


def read_cache(path: Path, meta: Mapping[str, Any]) -> dict[int, str]:
    """Cached page texts (1-based page -> text) when the cache matches ``meta``, else ``{}``."""
    if not path.exists():
        return {}
    pages: dict[int, str] = {}
    with path.open(encoding="utf-8") as fh:
        first = fh.readline()
        try:
            head = json.loads(first) if first.strip() else {}
        except json.JSONDecodeError:
            return {}
        if head.get("meta") != dict(meta):
            log.info("%s: OCR cache made with other settings or another file; rebuilding", path)
            return {}
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break  # a line cut off by an interrupted run: OCR that page again
            pages[int(row["page"])] = str(row["text"])
    return pages


# ------------------------------------------------------------------------------------------
# OCR engine (RapidOCR + pypdfium2), one per worker process
# ------------------------------------------------------------------------------------------

_WORKER: dict[str, Any] = {}


def _rapidocr_engine(threads: int) -> Callable[[Any], list[Box]]:
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR(intra_op_num_threads=threads)

    def run(image: Any) -> list[Box]:
        result, _ = engine(image, use_cls=False)  # scanned pages are upright
        return list(result or [])

    return run


def _init_worker(pdf: str, settings: dict[str, Any]) -> None:
    import pypdfium2 as pdfium

    _WORKER["pdf"] = pdfium.PdfDocument(pdf)
    _WORKER["settings"] = OcrSettings(**settings)
    _WORKER["engine"] = _rapidocr_engine(_WORKER["settings"].threads_per_worker)


def _ocr_page(page_no: int) -> tuple[int, str]:
    settings: OcrSettings = _WORKER["settings"]
    page = _WORKER["pdf"][page_no - 1]
    try:
        bitmap = page.render(scale=settings.dpi / 72, grayscale=True)
        image = bitmap.to_pil().convert("RGB")
    finally:
        page.close()
    return page_no, page_text(_WORKER["engine"](image), settings.min_score)


def page_count(pdf: Path) -> int:
    """Number of pages in a PDF."""
    from pypdf import PdfReader

    return len(PdfReader(str(pdf)).pages)


def ocr_pdf(
    pdf: Path,
    settings: OcrSettings,
    cache: Path | None = None,
    n_pages: int | None = None,
) -> list[str]:
    """Text of every page of a scanned PDF (cached; only missing pages are recognised)."""
    cache = cache or cache_path(pdf)
    meta = {"engine": engine_name(), "dpi": settings.dpi, "min_score": settings.min_score}
    meta["sha256"] = sha256_of(pdf)
    n_pages = n_pages or page_count(pdf)
    done = read_cache(cache, meta)
    todo = [p for p in range(1, n_pages + 1) if p not in done]
    if todo:
        log.info(
            "OCR %s: %d of %d pages to recognise (%d workers x %d threads, %d dpi)",
            pdf.name,
            len(todo),
            n_pages,
            settings.workers,
            settings.threads_per_worker,
            settings.dpi,
        )
        fresh = not done
        with cache.open("w" if fresh else "a", encoding="utf-8") as fh:
            if fresh:
                fh.write(json.dumps({"meta": meta}) + "\n")
            t0 = time.perf_counter()
            # spawn: no fork after ONNX Runtime / torch threads exist in the parent
            ctx = mp.get_context("spawn")
            init = (str(pdf), asdict(settings))
            with ctx.Pool(settings.workers, initializer=_init_worker, initargs=init) as pool:
                for i, (page_no, text) in enumerate(pool.imap_unordered(_ocr_page, todo), 1):
                    done[page_no] = text
                    fh.write(json.dumps({"page": page_no, "text": text}) + "\n")
                    fh.flush()
                    if i % 25 == 0 or i == len(todo):
                        elapsed = time.perf_counter() - t0
                        log.info("OCR %s: %d/%d pages (%.0f s)", pdf.name, i, len(todo), elapsed)
    return [done.get(p, "") for p in range(1, n_pages + 1)]


def main(argv: list[str] | None = None) -> int:
    """OCR one PDF into its cache (``python -m leakwatch.copilot.rag.ocr FILE``)."""
    import argparse

    from leakwatch.copilot.settings import copilot_config

    parser = argparse.ArgumentParser(description="OCR a scanned PDF into <file>.ocr.jsonl")
    parser.add_argument("pdf", type=Path)
    args = parser.parse_args(argv)
    settings = OcrSettings.from_config(copilot_config()["rag"].get("ocr"))
    pages = ocr_pdf(args.pdf, settings)
    log.info("%s: %d pages, %d with text", args.pdf.name, len(pages), sum(bool(p) for p in pages))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
