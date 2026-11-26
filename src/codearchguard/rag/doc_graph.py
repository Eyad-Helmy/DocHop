"""Represent retrieval documentation chunks inside the structural graph."""
from __future__ import annotations

from codearchguard.models import CodeGraph, Edge, EdgeKind, Node, NodeKind
from codearchguard.rag.document import Chunk


def doc_chunk_node_id(chunk: Chunk) -> str:
    """Namespace chunk identifiers so they cannot collide with code symbols."""
    return f"docchunk:{chunk.id}"


def attach_doc_chunks(graph: CodeGraph, chunks: list[Chunk]) -> None:
    """Add documentation chunks and links to known symbol owners.

    Repository-level Markdown uses the explicit ``__repo__`` owner sentinel.
    It remains an unlinked DocChunk because no code symbol truthfully owns it;
    this avoids manufacturing a dangling relationship.
    """
    for chunk in chunks:
        node_id = doc_chunk_node_id(chunk)
        graph.add_node(Node(
            id=node_id,
            kind=NodeKind.DOC_CHUNK,
            name=chunk.id,
            file=chunk.source_file,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            content=chunk.text,
            owner_symbol_id=chunk.owner_symbol_id,
            source_type=chunk.source_type,
        ))
        if chunk.owner_symbol_id in graph.nodes:
            graph.add_edge(Edge(
                src=chunk.owner_symbol_id,
                dst=node_id,
                kind=EdgeKind.DOCUMENTED_BY,
                line=chunk.start_line,
            ))
