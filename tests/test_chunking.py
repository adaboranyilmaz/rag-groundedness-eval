"""Unit tests for the three chunking strategies in src/ingestion/chunking.py."""

from itertools import pairwise

from src.ingestion.chunking import (
    FixedSizeChunker,
    RecursiveStructuralChunker,
    TableAwareChunker,
)
from src.ingestion.parser import ParsedDocument, TextBlock


def make_document(specs: list[tuple[str, int | None, str | None, bool]]) -> ParsedDocument:
    """Build a ParsedDocument the same way parser._assemble does, so tests exercise the
    exact offset/joining convention real parsing produces."""
    texts = []
    blocks = []
    cursor = 0
    for text, page, section, is_table in specs:
        start = cursor
        end = start + len(text)
        blocks.append(TextBlock(text, page, section, is_table, start, end))
        texts.append(text)
        cursor = end + 2
    full_text = "\n\n".join(texts)
    return ParsedDocument("doc1", "test", full_text, tuple(blocks))


def assert_round_trips(document: ParsedDocument, chunks) -> None:
    for chunk in chunks:
        assert document.full_text[chunk.char_start : chunk.char_end] == chunk.text
        assert chunk.doc_id == document.doc_id


class TestFixedSizeChunker:
    def test_round_trips(self):
        document = make_document([("word " * 200, 1, "Intro", False)])
        chunks = FixedSizeChunker(chunk_size=100, overlap=20).chunk(document)
        assert len(chunks) > 1
        assert_round_trips(document, chunks)

    def test_respects_overlap_between_consecutive_chunks(self):
        document = make_document([("word " * 200, 1, "Intro", False)])
        chunks = FixedSizeChunker(chunk_size=100, overlap=20).chunk(document)
        for prev, nxt in pairwise(chunks):
            assert nxt.char_start < prev.char_end  # overlap actually occurs
            assert nxt.char_start >= prev.char_start  # but chunks still advance

    def test_rejects_overlap_not_smaller_than_chunk_size(self):
        document = make_document([("hello world", 1, None, False)])
        try:
            FixedSizeChunker(chunk_size=50, overlap=50).chunk(document)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass

    def test_can_slice_through_a_table(self):
        # A blind character window with no table awareness at all is the point of this
        # strategy — it's expected to cut a table in half if one falls at a boundary.
        table_text = "Revenue | 100\nExpenses | 50\n" * 20
        document = make_document([(table_text, 1, None, True)])
        chunks = FixedSizeChunker(chunk_size=50, overlap=10).chunk(document)
        assert len(chunks) > 1
        assert_round_trips(document, chunks)


class TestRecursiveStructuralChunker:
    def test_round_trips(self):
        document = make_document(
            [
                ("Intro paragraph one.", 1, "Overview", False),
                ("Intro paragraph two.", 1, "Overview", False),
                ("Risk factor text.", 2, "Risk Factors", False),
            ]
        )
        chunks = RecursiveStructuralChunker(max_chunk_size=1000).chunk(document)
        assert_round_trips(document, chunks)

    def test_flushes_on_section_change_even_within_budget(self):
        document = make_document(
            [
                ("short a", 1, "Section A", False),
                ("short b", 1, "Section B", False),
            ]
        )
        chunks = RecursiveStructuralChunker(max_chunk_size=1000).chunk(document)
        assert len(chunks) == 2
        assert chunks[0].section == "Section A"
        assert chunks[1].section == "Section B"

    def test_never_splits_a_single_oversized_block(self):
        huge_table = "row " * 500
        document = make_document([(huge_table, 1, "Financials", True)])
        chunks = RecursiveStructuralChunker(max_chunk_size=100).chunk(document)
        assert len(chunks) == 1
        assert chunks[0].text == huge_table
        assert chunks[0].is_table

    def test_can_merge_a_table_with_adjacent_narrative(self):
        document = make_document(
            [
                ("Some short narrative.", 1, "Financials", False),
                ("Rev | 10", 1, "Financials", True),
            ]
        )
        chunks = RecursiveStructuralChunker(max_chunk_size=1000).chunk(document)
        assert len(chunks) == 1
        assert chunks[0].is_table  # any() over the merged blocks


class TestTableAwareChunker:
    def test_round_trips(self):
        document = make_document(
            [
                ("Some narrative before.", 1, "Financials", False),
                ("Rev | 10\nExp | 5", 1, "Financials", True),
                ("Some narrative after.", 1, "Financials", False),
            ]
        )
        chunks = TableAwareChunker(max_chunk_size=1000).chunk(document)
        assert_round_trips(document, chunks)

    def test_table_is_never_merged_with_narrative(self):
        document = make_document(
            [
                ("Some narrative before.", 1, "Financials", False),
                ("Rev | 10\nExp | 5", 1, "Financials", True),
                ("Some narrative after.", 1, "Financials", False),
            ]
        )
        chunks = TableAwareChunker(max_chunk_size=1000).chunk(document)
        table_chunks = [c for c in chunks if c.is_table]
        assert len(table_chunks) == 1
        assert table_chunks[0].text == "Rev | 10\nExp | 5"

    def test_consecutive_tables_each_get_their_own_chunk(self):
        document = make_document(
            [
                ("Rev | 10", 1, "Financials", True),
                ("Exp | 5", 1, "Financials", True),
            ]
        )
        chunks = TableAwareChunker(max_chunk_size=1000).chunk(document)
        assert len(chunks) == 2
        assert all(c.is_table for c in chunks)

    def test_never_splits_a_table_regardless_of_size(self):
        huge_table = "row | value\n" * 500
        document = make_document([(huge_table, 1, "Financials", True)])
        chunks = TableAwareChunker(max_chunk_size=50).chunk(document)
        assert len(chunks) == 1
        assert chunks[0].text == huge_table
