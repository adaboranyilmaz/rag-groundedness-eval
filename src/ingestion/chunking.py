"""Three chunking strategies behind one interface.

Every chunk's `char_start`/`char_end` index into the source `ParsedDocument.full_text`,
so `document.full_text[chunk.char_start:chunk.char_end] == chunk.text` always holds —
that equality is the round-trip guarantee scripts/02_chunk.py checks for every chunk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.ingestion.parser import ParsedDocument, TextBlock


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: str
    strategy: str
    page: int | None
    section: str | None
    char_start: int
    char_end: int
    text: str
    is_table: bool


class Chunker(Protocol):
    name: str

    def chunk(self, document: ParsedDocument) -> list[Chunk]: ...


def _block_at(blocks: tuple[TextBlock, ...], offset: int) -> TextBlock | None:
    """Return the block containing `offset`, or the nearest preceding one."""
    best = None
    for block in blocks:
        if block.char_start <= offset < block.char_end:
            return block
        if block.char_start <= offset:
            best = block
    return best


@dataclass
class FixedSizeChunker:
    """Blind character window. Ignores block/table boundaries by design — this is the
    strategy the qualitative report expects to mangle tables it slices through."""

    chunk_size: int = 512
    overlap: int = 64
    name: str = "fixed_size"

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        text = document.full_text
        n = len(text)
        if n == 0:
            return []
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        chunks: list[Chunk] = []
        index = 0
        start = 0
        while start < n:
            end = min(start + self.chunk_size, n)
            if end < n:
                snapped = text.rfind(" ", start, end)
                if snapped > start:
                    end = snapped
            block = _block_at(document.blocks, start)
            fully_within_table = block is not None and block.is_table and end <= block.char_end
            chunks.append(
                Chunk(
                    doc_id=document.doc_id,
                    chunk_id=f"{document.doc_id}:{self.name}:{index}",
                    strategy=self.name,
                    page=block.page if block else None,
                    section=block.section if block else None,
                    char_start=start,
                    char_end=end,
                    text=text[start:end],
                    is_table=fully_within_table,
                )
            )
            index += 1
            if end >= n:
                break
            start = end - self.overlap if end - self.overlap > start else end
        return chunks


@dataclass
class RecursiveStructuralChunker:
    """Packs whole blocks into chunks up to `max_chunk_size`, flushing on a section
    change or size overflow. Blocks are atomic — a table may end up sharing a chunk
    with adjacent narrative text if both fit the budget, or standing alone if not."""

    max_chunk_size: int = 1024
    name: str = "recursive_structural"

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        index = 0
        buffer_blocks: list[TextBlock] = []

        def flush() -> None:
            nonlocal index
            if not buffer_blocks:
                return
            start = buffer_blocks[0].char_start
            end = buffer_blocks[-1].char_end
            chunks.append(
                Chunk(
                    doc_id=document.doc_id,
                    chunk_id=f"{document.doc_id}:{self.name}:{index}",
                    strategy=self.name,
                    page=buffer_blocks[0].page,
                    section=buffer_blocks[0].section,
                    char_start=start,
                    char_end=end,
                    text=document.full_text[start:end],
                    is_table=any(b.is_table for b in buffer_blocks),
                )
            )
            index += 1
            buffer_blocks.clear()

        for block in document.blocks:
            if buffer_blocks and block.section != buffer_blocks[-1].section:
                flush()
            if buffer_blocks:
                prospective_len = block.char_end - buffer_blocks[0].char_start
                if prospective_len > self.max_chunk_size:
                    flush()
            buffer_blocks.append(block)
        flush()
        return chunks


@dataclass
class TableAwareChunker:
    """Same packing logic as recursive-structural for narrative text, except a table
    block always gets its own chunk — never merged with neighbors, never split."""

    max_chunk_size: int = 1024
    name: str = "table_aware"

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        index = 0
        buffer_blocks: list[TextBlock] = []

        def flush(is_table: bool = False) -> None:
            nonlocal index
            if not buffer_blocks:
                return
            start = buffer_blocks[0].char_start
            end = buffer_blocks[-1].char_end
            chunks.append(
                Chunk(
                    doc_id=document.doc_id,
                    chunk_id=f"{document.doc_id}:{self.name}:{index}",
                    strategy=self.name,
                    page=buffer_blocks[0].page,
                    section=buffer_blocks[0].section,
                    char_start=start,
                    char_end=end,
                    text=document.full_text[start:end],
                    is_table=is_table,
                )
            )
            index += 1
            buffer_blocks.clear()

        for block in document.blocks:
            if block.is_table:
                flush()
                buffer_blocks.append(block)
                flush(is_table=True)
                continue
            if buffer_blocks:
                same_section = block.section == buffer_blocks[-1].section
                prospective_len = block.char_end - buffer_blocks[0].char_start
                if not same_section or prospective_len > self.max_chunk_size:
                    flush()
            buffer_blocks.append(block)
        flush()
        return chunks


ALL_CHUNKERS: dict[str, type] = {
    "fixed_size": FixedSizeChunker,
    "recursive_structural": RecursiveStructuralChunker,
    "table_aware": TableAwareChunker,
}
