"""PDF/HTML -> text, preserving page and section metadata for every block.

`pymupdf` parses PDFs (it exposes per-page font sizes for header detection and native
table extraction); a direct `lxml` DOM walk parses HTML, since financial filings use
HTML `<table>` tags almost as often for pure visual layout as for real data, and a
generic HTML-to-text library doesn't distinguish the two (see DECISIONS.md). A PDF with
no extractable text layer (e.g. a scanned filing) raises `UnparseablePdfError` rather
than falling back to an OCR/layout-model pipeline — this project's actual corpus is
100% HTML, so that case hasn't come up in practice.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import lxml.html
import pymupdf

MIN_EXTRACTED_CHARS_PER_PAGE = 20  # below this, assume pymupdf got no real text layer
HEADER_MAX_CHARS = 120
HEADER_FONT_RATIO = 1.15  # header font size must exceed this multiple of body median
MIN_TABLE_ROWS = 2
MIN_TABLE_COLS = 2
MIN_TABLE_NUMERIC_FRACTION = 0.25  # below this, a <table> is layout, not financial data


@dataclass(frozen=True)
class TextBlock:
    text: str
    page: int | None
    section: str | None
    is_table: bool
    char_start: int
    char_end: int


@dataclass(frozen=True)
class ParsedDocument:
    doc_id: str
    source_path: str
    full_text: str
    blocks: tuple[TextBlock, ...]


def _assemble(doc_id: str, source_path: str, raw_blocks: list[tuple]) -> ParsedDocument:
    """Join (text, page, section, is_table) tuples into a ParsedDocument with offsets."""
    texts: list[str] = []
    blocks: list[TextBlock] = []
    cursor = 0
    for text, page, section, is_table in raw_blocks:
        text = text.strip()
        if not text:
            continue
        start = cursor
        end = start + len(text)
        blocks.append(TextBlock(text, page, section, is_table, start, end))
        texts.append(text)
        cursor = end + 2  # for the "\n\n" separator joined below
    full_text = "\n\n".join(texts)
    return ParsedDocument(doc_id, source_path, full_text, tuple(blocks))


def _serialize_table(rows: list[list[str | None]]) -> str:
    """Render a table's rows as pipe-delimited text so numbers stay row/column-attached."""
    lines = []
    for row in rows:
        cells = [(cell or "").strip().replace("\n", " ") for cell in row]
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def parse_pdf(path: Path, doc_id: str) -> ParsedDocument:
    document = pymupdf.open(path)
    raw_blocks: list[tuple] = []
    current_section: str | None = None
    extracted_chars = 0
    page_count = document.page_count

    for page_index in range(page_count):
        page = document[page_index]
        page_number = page_index + 1

        tables = page.find_tables()
        table_bboxes = [pymupdf.Rect(t.bbox) for t in tables]
        for table in tables:
            serialized = _serialize_table(table.extract())
            raw_blocks.append((serialized, page_number, current_section, True))

        text_dict = page.get_text("dict")
        body_sizes = [
            span["size"]
            for block in text_dict["blocks"]
            for line in block.get("lines", [])
            for span in line.get("spans", [])
        ]
        median_size = sorted(body_sizes)[len(body_sizes) // 2] if body_sizes else 0

        for block in text_dict["blocks"]:
            if "lines" not in block:
                continue
            block_rect = pymupdf.Rect(block["bbox"])
            if any(block_rect.intersects(bbox) for bbox in table_bboxes):
                continue  # already emitted as part of a table above

            block_text = "\n".join(
                span["text"] for line in block["lines"] for span in line["spans"]
            ).strip()
            if not block_text:
                continue
            extracted_chars += len(block_text)

            max_size = max(
                (span["size"] for line in block["lines"] for span in line["spans"]),
                default=0,
            )
            is_header = (
                median_size > 0
                and max_size >= median_size * HEADER_FONT_RATIO
                and len(block_text) <= HEADER_MAX_CHARS
                and "\n" not in block_text
            )
            if is_header:
                current_section = block_text
            raw_blocks.append((block_text, page_number, current_section, False))

    document.close()

    if extracted_chars < MIN_EXTRACTED_CHARS_PER_PAGE * page_count:
        # No text layer pymupdf can read (e.g. a scanned filing with no embedded text)
        # — this project's actual corpus is 100% HTML (EDGAR serves HTML natively for
        # modern 10-K/10-Q filings), so a PDF landing here is unexpected. Flag it as
        # unparseable rather than pull in an OCR/layout-model stack for a case that
        # has not occurred once in practice; see DECISIONS.md.
        raise UnparseablePdfError(
            f"{path}: no extractable text layer (got {extracted_chars} chars across "
            f"{page_count} pages) — likely a scanned PDF with no OCR support"
        )

    return _assemble(doc_id, str(path), raw_blocks)


class UnparseablePdfError(RuntimeError):
    """Raised when pymupdf finds no usable text layer in a PDF."""


_HTML_TAG_RE = re.compile(rb"<html", re.IGNORECASE)
_HTML_CLOSE_TAG_RE = re.compile(rb"</html\s*>", re.IGNORECASE)
_NUMERIC_CELL_RE = re.compile(r"^[\$\(\-]?[\d,]+\.?\d*\)?%?$")
LEAF_BLOCK_TAGS = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "td", "th"}


def _strip_sgml_wrapper(raw: bytes) -> bytes:
    """Older EDGAR filings (pre-~2020, pre filing-agent HTML) wrap the real document in
    SGML submission tags (`<DOCUMENT><TYPE>...<TEXT>...html...</TEXT></DOCUMENT>`).
    A standard HTML parser can choke on that prefix, so trim to the `<html>...</html>`
    span before parsing."""
    start_match = _HTML_TAG_RE.search(raw)
    if start_match is None:
        return raw
    end_match = _HTML_CLOSE_TAG_RE.search(raw, start_match.start())
    end = end_match.end() if end_match else len(raw)
    return raw[start_match.start() : end]


def _has_page_break(el, before: bool) -> bool:
    style = (el.get("style") or "").lower()
    prop = "page-break-before" if before else "page-break-after"
    return prop in style and "always" in style


def _leaf_block_elements(root) -> set:
    """Elements whose tag is block-level and which contain no block-level descendant —
    the units we extract narrative text from. One bottom-up pass, no recursion, so it's
    safe on the deeply-nested table-for-layout structures these filings use.

    Returns the element objects themselves, not `id()` values: lxml recycles the
    Python-level proxy objects it hands out for underlying libxml2 nodes, so a bare
    `id()` captured here can be reassigned to an unrelated element by the time a later
    traversal checks membership, causing silent false-positive matches (this bit once —
    a paragraph got emitted twice because its proxy's old address had been reused).
    Keeping the objects themselves in the returned set holds a live reference, which
    keeps lxml's proxy cache from recycling them, so identity comparisons against a
    later `.iter()` stay valid.
    """
    order: list = []
    stack = [(root, False)]
    while stack:
        el, expanded = stack.pop()
        if expanded:
            order.append(el)
            continue
        stack.append((el, True))
        for child in el:
            if isinstance(child.tag, str):
                stack.append((child, False))

    contains_block: dict = {}
    leaves = set()
    for el in order:
        child_has_block = any(
            isinstance(c.tag, str) and (c.tag in LEAF_BLOCK_TAGS or contains_block.get(c, False))
            for c in el
        )
        contains_block[el] = child_has_block
        if el.tag in LEAF_BLOCK_TAGS and not child_has_block:
            leaves.add(el)
    return leaves


def _extract_table_rows(table_el) -> list[list[str]]:
    rows = []
    for tr in table_el.iter("tr"):
        cells = [c.text_content().strip() for c in tr if c.tag in ("td", "th")]
        if cells:
            rows.append(cells)
    return rows


def _looks_like_financial_table(rows: list[list[str]]) -> bool:
    """Distinguish a real data table from the layout-only <table> tags these filings
    also use for positioning prose (a near-universal pattern in older EDGAR HTML)."""
    if len(rows) < MIN_TABLE_ROWS or max((len(r) for r in rows), default=0) < MIN_TABLE_COLS:
        return False
    cells = [c for row in rows for c in row if c]
    if not cells:
        return False
    numeric = sum(1 for c in cells if _NUMERIC_CELL_RE.match(c.replace(" ", "")))
    return (numeric / len(cells)) >= MIN_TABLE_NUMERIC_FRACTION


def _looks_like_header(el, text: str) -> bool:
    style = (el.get("style") or "").lower()
    is_bold = "font-weight:700" in style.replace(" ", "") or "font-weight:bold" in style.replace(
        " ", ""
    )
    return el.tag in {"h1", "h2", "h3", "h4", "h5", "h6"} or (
        is_bold and len(text) <= HEADER_MAX_CHARS and "\n" not in text
    )


def parse_html(path: Path, doc_id: str) -> ParsedDocument:
    """Parse EDGAR HTML filings with a direct DOM walk (lxml), not a generic HTML-to-
    text framework: financial filings use HTML `<table>` tags almost as often for pure
    visual layout as for real data, so table/non-table classification needs to inspect
    each table's actual cell content (row/column count, numeric density), not just its
    tag. Page numbers come from the `page-break-before/after:always` CSS these filings
    carry over from their paginated (Word/PDF) source — a real, reproducible signal.
    """
    raw_html = _strip_sgml_wrapper(path.read_bytes())
    tree = lxml.html.fromstring(raw_html)
    body = tree.find("body")
    if body is None:
        body = tree

    leaves = _leaf_block_elements(body)
    raw_blocks: list[tuple] = []
    current_section: str | None = None
    current_page = 1
    skip_elements: set = set()

    for el in body.iter():
        if not isinstance(el.tag, str) or el in skip_elements:
            continue
        if _has_page_break(el, before=True):
            current_page += 1

        if el.tag == "table":
            rows = _extract_table_rows(el)
            if _looks_like_financial_table(rows):
                serialized = _serialize_table(rows)
                raw_blocks.append((serialized, current_page, current_section, True))
                skip_elements.update(el.iterdescendants())
        elif el in leaves:
            text = el.text_content().strip()
            if text:
                if _looks_like_header(el, text):
                    current_section = text
                raw_blocks.append((text, current_page, current_section, False))

        if _has_page_break(el, before=False):
            current_page += 1

    return _assemble(doc_id, str(path), raw_blocks)


_HTML_SNIFF_RE = re.compile(rb"<html|<!doctype html", re.IGNORECASE)


def parse_document(path: Path, doc_id: str) -> ParsedDocument:
    """Dispatch on file extension, sniffing content for the ambiguous .htm/.txt cases."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path, doc_id)
    if suffix in (".htm", ".html"):
        return parse_html(path, doc_id)
    with open(path, "rb") as f:
        head = f.read(2048)
    if _HTML_SNIFF_RE.search(head):
        return parse_html(path, doc_id)
    raise ValueError(f"Unrecognized document type for {path} (suffix={suffix!r})")
