"""Unit tests for src/ingestion/parser.py. The HTML path gets the most coverage since
every document EDGAR actually served for this corpus turned out to be HTML, not PDF —
the PDF path is exercised here only against a synthetic fixture, not real EDGAR data."""

import pymupdf

from src.ingestion.parser import (
    UnparseablePdfError,
    _extract_table_rows,
    _looks_like_financial_table,
    _strip_sgml_wrapper,
    parse_html,
    parse_pdf,
)


def write_html(tmp_path, body: str, name: str = "doc.htm"):
    path = tmp_path / name
    path.write_text(f"<html><body>{body}</body></html>", encoding="utf-8")
    return path


def write_pdf_with_table(tmp_path, name: str = "doc.pdf"):
    """A minimal synthetic PDF: a heading, a narrative line, and a real ruled 2x3
    table — drawn with actual vector lines, not just aligned text, since that's what
    pymupdf's own `find_tables()` detection keys on."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Quarterly Report", fontsize=16)
    page.insert_text((72, 100), "Some narrative text about the business.")

    x0, y0, w, h = 72, 150, 200, 60
    rows, cols = 3, 2
    for r in range(rows + 1):
        page.draw_line((x0, y0 + r * h / rows), (x0 + w, y0 + r * h / rows))
    for c in range(cols + 1):
        page.draw_line((x0 + c * w / cols, y0), (x0 + c * w / cols, y0 + h))
    cells = [["Revenue", "100"], ["Expenses", "50"], ["Net", "50"]]
    for r in range(rows):
        for c in range(cols):
            page.insert_text(
                (x0 + c * w / cols + 5, y0 + r * h / rows + 15), cells[r][c], fontsize=9
            )

    path = tmp_path / name
    doc.save(str(path))
    doc.close()
    return path


def write_blank_pdf(tmp_path, name: str = "blank.pdf"):
    doc = pymupdf.open()
    doc.new_page()  # no text inserted -> no extractable text layer
    path = tmp_path / name
    doc.save(str(path))
    doc.close()
    return path


class TestParsePdf:
    def test_extracts_narrative_text(self, tmp_path):
        doc = parse_pdf(write_pdf_with_table(tmp_path), "doc1")
        assert any("narrative text about the business" in b.text for b in doc.blocks)

    def test_detects_a_real_ruled_table_with_numbers(self, tmp_path):
        doc = parse_pdf(write_pdf_with_table(tmp_path), "doc1")
        table_blocks = [b for b in doc.blocks if b.is_table]
        assert len(table_blocks) == 1
        assert "100" in table_blocks[0].text
        assert "Revenue" in table_blocks[0].text

    def test_all_blocks_are_on_page_one(self, tmp_path):
        doc = parse_pdf(write_pdf_with_table(tmp_path), "doc1")
        assert all(b.page == 1 for b in doc.blocks)

    def test_round_trips_char_spans(self, tmp_path):
        doc = parse_pdf(write_pdf_with_table(tmp_path), "doc1")
        for block in doc.blocks:
            assert doc.full_text[block.char_start : block.char_end] == block.text

    def test_raises_on_a_pdf_with_no_text_layer(self, tmp_path):
        try:
            parse_pdf(write_blank_pdf(tmp_path), "doc1")
            raise AssertionError("expected UnparseablePdfError")
        except UnparseablePdfError:
            pass


class TestStripSgmlWrapper:
    def test_strips_legacy_sgml_document_wrapper(self):
        raw = (
            b"<DOCUMENT>\n<TYPE>10-K\n<SEQUENCE>1\n<TEXT>\n"
            b"<html><body><p>hello</p></body></html>\n</TEXT>\n</DOCUMENT>\n"
        )
        cleaned = _strip_sgml_wrapper(raw)
        assert cleaned.startswith(b"<html")
        assert cleaned.endswith(b"</html>")

    def test_leaves_plain_html_untouched(self):
        raw = b"<html><body><p>hello</p></body></html>"
        assert _strip_sgml_wrapper(raw) == raw


class TestFinancialTableDetection:
    def test_numeric_grid_is_a_financial_table(self):
        rows = [["Revenue", "2023", "2022"], ["Net sales", "1,234", "1,100"]]
        assert _looks_like_financial_table(rows)

    def test_single_cell_layout_wrapper_is_not_a_table(self):
        rows = [["This is just prose used for layout positioning, not data."]]
        assert not _looks_like_financial_table(rows)

    def test_narrow_all_text_grid_is_not_a_table(self):
        rows = [["Name", "Title"], ["Alice", "Director"], ["Bob", "Director"]]
        assert not _looks_like_financial_table(rows)


class TestParseHtml:
    def test_extracts_a_real_financial_table_with_numbers(self, tmp_path):
        body = """
            <table>
                <tr><td>Revenue</td><td>2023</td><td>2022</td></tr>
                <tr><td>Net sales</td><td>1,234</td><td>1,100</td></tr>
            </table>
        """
        doc = parse_html(write_html(tmp_path, body), "doc1")
        table_blocks = [b for b in doc.blocks if b.is_table]
        assert len(table_blocks) == 1
        assert "1,234" in table_blocks[0].text

    def test_layout_table_decomposes_into_narrative_blocks(self, tmp_path):
        body = """
            <table><tr><td><p>Just a positioned paragraph of prose.</p></td></tr></table>
        """
        doc = parse_html(write_html(tmp_path, body), "doc1")
        assert not any(b.is_table for b in doc.blocks)
        assert any("positioned paragraph" in b.text for b in doc.blocks)

    def test_page_break_increments_page_number(self, tmp_path):
        body = """
            <p>First page content.</p>
            <p style="page-break-after:always">&nbsp;</p>
            <p>Second page content.</p>
        """
        doc = parse_html(write_html(tmp_path, body), "doc1")
        first = next(b for b in doc.blocks if "First page" in b.text)
        second = next(b for b in doc.blocks if "Second page" in b.text)
        assert second.page > first.page

    def test_bold_short_text_becomes_a_section_header(self, tmp_path):
        body = """
            <p style="font-weight:700;">Risk Factors</p>
            <p>Our business faces various risks.</p>
        """
        doc = parse_html(write_html(tmp_path, body), "doc1")
        narrative = next(b for b in doc.blocks if "various risks" in b.text)
        assert narrative.section == "Risk Factors"

    def test_repeated_leaf_content_is_not_double_emitted(self, tmp_path):
        # Regression test: lxml recycles its Python-level element proxies, so an
        # earlier version of this parser could emit the same paragraph twice when a
        # freed proxy's memory address got reused by an unrelated element. Build many
        # nested div/p/font wrappers so the recycling condition actually reproduces.
        rows = "".join(
            f"<tr><td><p><font>row {i} unique text</font></p></td></tr>" for i in range(200)
        )
        body = f"<table>{rows}</table>"
        doc = parse_html(write_html(tmp_path, body), "doc1")
        texts = [b.text for b in doc.blocks]
        assert len(texts) == len(set(texts))

    def test_round_trips_char_spans(self, tmp_path):
        body = """
            <p>Alpha paragraph.</p>
            <table><tr><td>Rev</td><td>100</td></tr><tr><td>Exp</td><td>50</td></tr></table>
            <p>Beta paragraph.</p>
        """
        doc = parse_html(write_html(tmp_path, body), "doc1")
        for block in doc.blocks:
            assert doc.full_text[block.char_start : block.char_end] == block.text


class TestExtractTableRows:
    def test_reads_rows_in_order(self, tmp_path):
        import lxml.html

        tree = lxml.html.fromstring(
            "<table><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr></table>"
        )
        rows = _extract_table_rows(tree)
        assert rows == [["a", "b"], ["c", "d"]]
