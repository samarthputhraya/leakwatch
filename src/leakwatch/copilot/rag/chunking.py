"""Split documents into passages by section, keeping page metadata (PRD §9.5).

- **Markdown** (our knowledge base, ``docs/kb``): one passage per ``##`` section (``###`` sections
  are named ``"<h2> › <h3>"``); long sections are split into overlapping windows. No page numbers.
- **PDF** (CPHEEO manuals): text per page with ``pypdf``. Headings are found in the text, not
  guessed: ``CHAPTER 15`` lines give the chapter (its title is the upper-case lines that follow),
  numbered headings such as ``15.10 LEAKAGE DETECTION AND MONITORING`` give the section. A numbered
  line only counts as a heading when its chapter number matches the current chapter (tables in
  annexures also start with numbers). Front matter before the first chapter and table-of-contents
  pages (dot leaders) are skipped. ``page`` is the 1-based PDF page; ``printed_page`` is the number
  printed on the page when the first line is one.

Some PDFs store heading glyphs as ``/G31/G35`` (hex character codes); :func:`decode_glyphs` turns
them back into text.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

_GLYPH = re.compile(r"/G([0-9A-Fa-f]{2})")
_CHAPTER = re.compile(r"^CHAPTER\s+(\d{1,2})$", re.I)
_HEADING = re.compile(r"^(\d{1,2})\.(\d{1,2})(?:\.(\d{1,2}))?\.?\s+([A-Za-z(].{2,90})$")
_DOT_LEADER = re.compile(r"\.{5,}|(?:\. ){4,}")
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SMALL_WORDS = {"and", "or", "of", "the", "in", "on", "for", "to", "a", "an", "by", "with", "at"}
_ACRONYMS = {"UFW", "NRW", "SCADA", "O&M", "GIS", "DMA", "PVC", "HDPE", "WHO", "PPP", "CPHEEO"}


@dataclass(frozen=True)
class DocumentInfo:
    """A document in the corpus."""

    doc_id: str
    title: str
    short: str  # name used in citations, e.g. "CPHEEO O&M Manual 2005"
    source_type: str  # "pdf" | "markdown"
    path: str
    url: str | None = None


@dataclass(frozen=True)
class Chunk:
    """One passage with where it came from."""

    chunk_id: str
    doc_id: str
    doc_title: str
    doc_short: str
    source_type: str
    chapter: str | None
    section: str
    page: int | None  # 1-based PDF page where the passage starts (None for Markdown)
    page_end: int | None
    printed_page: str | None
    text: str

    @property
    def citation(self) -> str:
        """``[doc, section, p. N]`` (no page for Markdown documents)."""
        parts = [self.doc_short, self.section]
        if self.page is not None:
            parts.append(f"p. {self.page}")
        return "[" + ", ".join(parts) + "]"

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict including the citation string."""
        return {**asdict(self), "citation": self.citation}


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def title_case(text: str) -> str:
    """``LEAKAGE  DETECTION AND MONITORING`` -> ``Leakage Detection and Monitoring``."""
    words = _squash(text).split(" ")
    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) < 0.8:
        return " ".join(words)  # already mixed case: keep as written
    out = []
    for i, word in enumerate(words):
        low = word.lower()
        bare = re.sub(r"[^A-Za-z&]", "", word).upper()
        if bare in _ACRONYMS:
            out.append(word)
        elif i and low in _SMALL_WORDS:
            out.append(low)
        else:
            out.append(re.sub(r"[a-z]", lambda m: m.group(0).upper(), low, count=1))
    return " ".join(out)


def _window(lines: list[str], max_chars: int, overlap: int) -> list[tuple[list[str], int, int]]:
    """Split lines into windows of <= max_chars, carrying ~overlap chars; (lines, first, last)."""
    out: list[tuple[list[str], int, int]] = []
    start = 0
    while start < len(lines):
        size, end = 0, start
        while end < len(lines) and (size + len(lines[end]) + 1 <= max_chars or end == start):
            size += len(lines[end]) + 1
            end += 1
        out.append((lines[start:end], start, end - 1))
        if end >= len(lines):
            break
        back, carried = end, 0
        while back - 1 > start and carried + len(lines[back - 1]) + 1 <= overlap:
            back -= 1
            carried += len(lines[back]) + 1
        start = back
    return out


# ------------------------------------------------------------------------------------------
# Markdown
# ------------------------------------------------------------------------------------------


def read_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Split an optional ``---`` YAML front-matter block from a Markdown text."""
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            meta = yaml.safe_load(text[4:end]) or {}
            body = text[end + 4 :].lstrip("\n")
            return (meta if isinstance(meta, dict) else {}), body
    return {}, text


def markdown_info(path: Path, doc_id: str | None = None) -> tuple[DocumentInfo, str]:
    """Document metadata (front matter ``short``/``title`` or the H1) and the Markdown body."""
    meta, body = read_front_matter(path.read_text(encoding="utf-8"))
    h1 = next(
        (m.group(2) for m in map(_MD_HEADING.match, body.splitlines()) if m and m.group(1) == "#"),
        None,
    )
    title = str(meta.get("title") or h1 or path.stem)
    info = DocumentInfo(
        doc_id=doc_id or f"kb_{path.stem}",
        title=title,
        short=str(meta.get("short") or title),
        source_type="markdown",
        path=str(path),
        url=meta.get("url"),
    )
    return info, body


def chunk_markdown(
    body: str, info: DocumentInfo, max_chars: int = 1000, overlap: int = 150
) -> list[Chunk]:
    """Passages of a Markdown document, one or more per ``##``/``###`` section."""
    # (chapter, section name, has heading line, lines)
    sections: list[tuple[str | None, str, bool, list[str]]] = []
    h2: str | None = None
    current: tuple[str | None, str, bool, list[str]] | None = None
    for line in body.splitlines():
        m = _MD_HEADING.match(line)
        if m and len(m.group(1)) in (2, 3):
            level, text = len(m.group(1)), _squash(m.group(2))
            if level == 2:
                h2 = text
                name = text
            else:
                name = f"{h2} › {text}" if h2 else text
            current = (h2, name, True, [text])
            sections.append(current)
            continue
        if m and len(m.group(1)) == 1:
            continue  # document title
        if current is None:
            current = (None, info.title, False, [])
            sections.append(current)
        if line.strip():
            current[3].append(line.rstrip())
    chunks: list[Chunk] = []
    for chapter, name, has_heading, lines in sections:
        if len(lines) <= (1 if has_heading else 0):
            continue  # heading without text
        for part, _, _ in _window(lines, max_chars, overlap):
            chunks.append(
                Chunk(
                    chunk_id=f"{info.doc_id}:{len(chunks):05d}",
                    doc_id=info.doc_id,
                    doc_title=info.title,
                    doc_short=info.short,
                    source_type="markdown",
                    chapter=chapter,
                    section=name,
                    page=None,
                    page_end=None,
                    printed_page=None,
                    text="\n".join(part),
                )
            )
    return chunks


# ------------------------------------------------------------------------------------------
# PDF
# ------------------------------------------------------------------------------------------


def decode_glyphs(text: str) -> str:
    """Replace ``/GXX`` glyph codes with their characters (``/G31/G35`` -> ``15``)."""
    return _GLYPH.sub(lambda m: chr(int(m.group(1), 16)), text).replace("\x7f", "-")


def read_pdf_pages(path: Path) -> list[str]:
    """Text of every page (empty string where extraction fails)."""
    import logging

    from pypdf import PdfReader

    logging.getLogger("pypdf").setLevel(logging.ERROR)  # malformed-xref chatter
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return pages


def _upper_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(c.isupper() for c in letters) / len(letters) if letters else 0.0


def chapter_titles(pages: Sequence[str]) -> tuple[dict[int, list[str]], int | None]:
    """Chapter number -> its title lines (as printed), and the PDF page of the first chapter."""
    titles: dict[int, list[str]] = {}
    first_page: int | None = None
    for page_no, raw in enumerate(pages, start=1):
        lines = [ln.strip() for ln in decode_glyphs(raw).splitlines() if ln.strip()]
        for i, line in enumerate(lines):
            m = _CHAPTER.match(line)
            if not m:
                continue
            number = int(m.group(1))
            words: list[str] = []
            for nxt in lines[i + 1 : i + 4]:
                if _HEADING.match(nxt) or _upper_share(nxt) < 0.9 or len(nxt) > 60:
                    break
                words.append(nxt)
            if number not in titles and words:
                titles[number] = words
            if first_page is None:
                first_page = page_no
    return titles, first_page


def _is_toc(lines: Sequence[str]) -> bool:
    return bool(lines) and sum(bool(_DOT_LEADER.search(x)) for x in lines) / len(lines) >= 0.25


def chunk_pdf_pages(
    pages: Sequence[str],
    info: DocumentInfo,
    max_chars: int = 1000,
    overlap: int = 150,
    skip_front_matter: bool = True,
) -> list[Chunk]:
    """Passages of a PDF given the text of its pages (see module docstring)."""
    title_lines_of, first_chapter_page = chapter_titles(pages)
    titles = {n: title_case(" ".join(words)) for n, words in title_lines_of.items()}
    chapter: int | None = None
    section: str | None = None
    # (chapter, section, [(page, printed, line)])
    blocks: list[tuple[int | None, str | None, list[tuple[int, str | None, str]]]] = []
    current: list[tuple[int, str | None, str]] = []

    def flush() -> None:
        nonlocal current
        if current:
            blocks.append((chapter, section, current))
        current = []

    for page_no, raw in enumerate(pages, start=1):
        if skip_front_matter and first_chapter_page is not None and page_no < first_chapter_page:
            continue
        lines = [ln.strip() for ln in decode_glyphs(raw).splitlines() if ln.strip()]
        if _is_toc(lines):
            continue
        printed = None
        if lines and lines[0].isdigit() and len(lines[0]) <= 4:
            printed = lines.pop(0)
        expect_title: list[str] = []  # chapter title lines still to skip, in order
        for line in lines:
            if expect_title and line == expect_title[0]:
                expect_title.pop(0)
                continue
            expect_title = []
            m_ch = _CHAPTER.match(line)
            if m_ch:
                number = int(m_ch.group(1))
                if number != chapter:
                    flush()
                    chapter, section = number, None
                expect_title = list(title_lines_of.get(number, []))
                continue
            m_h = _HEADING.match(line)
            if m_h:
                ch, minor = int(m_h.group(1)), int(m_h.group(2))
                level3 = m_h.group(3) is not None
                text = m_h.group(4).strip()
                new_chapter = (
                    chapter is not None and ch == chapter + 1 and minor == 1 and not level3
                )
                same_chapter = chapter is not None and ch == chapter
                if chapter is None and minor == 1 and not level3:
                    new_chapter = True
                looks_like = (_upper_share(text) >= 0.8) or (level3 and len(text) <= 70)
                if (same_chapter or new_chapter) and looks_like and not text.endswith(","):
                    flush()
                    if new_chapter:
                        chapter = ch
                    number = f"{ch}.{minor}" + (f".{m_h.group(3)}" if level3 else "")
                    section = f"§{number} {title_case(text)}"
            current.append((page_no, printed, line))
    flush()

    chunks: list[Chunk] = []
    for ch, sec, entries in blocks:
        chapter_label = None
        if ch is not None:
            chapter_label = f"Ch. {ch} {titles[ch]}" if ch in titles else f"Ch. {ch}"
        name = sec or chapter_label or info.title
        texts = [e[2] for e in entries]
        for part, first, last in _window(texts, max_chars, overlap):
            chunks.append(
                Chunk(
                    chunk_id=f"{info.doc_id}:{len(chunks):05d}",
                    doc_id=info.doc_id,
                    doc_title=info.title,
                    doc_short=info.short,
                    source_type="pdf",
                    chapter=chapter_label,
                    section=name,
                    page=entries[first][0],
                    page_end=entries[last][0],
                    printed_page=entries[first][1],
                    text="\n".join(part),
                )
            )
    return chunks


def chunk_pdf(
    path: Path, info: DocumentInfo, max_chars: int = 1000, overlap: int = 150
) -> list[Chunk]:
    """Read a PDF and split it into passages."""
    return chunk_pdf_pages(read_pdf_pages(path), info, max_chars, overlap)
