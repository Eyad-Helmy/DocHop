
from dataclasses import dataclass


@dataclass
class Document:
    id: str
    owner_symbol_id: str
    source_type: str
    source_file: str
    start_line: int
    end_line: int
    text: str



@dataclass
class Chunk:
    id: str
    parent_document_id: str
    owner_symbol_id: str
    source_type: str
    source_file: str
    start_line: int
    end_line: int
    chunk_index: int
    text: str
    start_char: int
    end_char: int
    context: str
    embedding: list[float] | None = None