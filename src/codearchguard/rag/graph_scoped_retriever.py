from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import re

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, TextNode

from codearchguard.models import CodeGraph, EdgeKind
from codearchguard.rag.document import Chunk
from codearchguard.rag.llama_bm25_retriever import build_bm25_retriever
from codearchguard.rag.llama_embedder import LlamaBGEEmbedder
from codearchguard.rag.llama_hybrid_retriever import build_hybrid_retriever


@dataclass
class GraphScopedRetrievalResult:
    """Result returned by Graph-Scoped Hybrid RAG (FR-07 + FR-09)."""

    target_symbol: str
    query: str
    blast_radius_symbols: set[str]
    scoped_chunk_count: int
    total_chunk_count: int
    results: list[NodeWithScore]


def compute_graph_blast_radius(
    graph: CodeGraph,
    start_symbol: str,
    max_hops: int | None = None,
    include_containers: bool = True,
) -> set[str]:
    """
    Cycle-safe BFS over CodeGraph matching Milestone 1.5 blast-radius rules:
      - CALLS (reversed: dst -> src): upstream callers affected if dst changes
      - INHERITS (reversed: dst -> src): subclasses affected if base dst changes
      - DEFINES (forward: src -> dst): members defined inside affected container src
    After BFS, optionally adds the enclosing Class/Module IDs of affected nodes
    so module/class docstrings are included without pulling in unrelated siblings.
    """
    if start_symbol not in graph.nodes:
        raise ValueError(f"Symbol '{start_symbol}' not found in CodeGraph.")

    adjacency: dict[str, set[str]] = {nid: set() for nid in graph.nodes}
    parent_container: dict[str, str] = {}

    for edge in graph.edges:
        if edge.kind == EdgeKind.CALLS:
            adjacency.setdefault(edge.dst, set()).add(edge.src)
        elif edge.kind == EdgeKind.INHERITS:
            adjacency.setdefault(edge.dst, set()).add(edge.src)
        elif edge.kind == EdgeKind.DEFINES:
            adjacency.setdefault(edge.src, set()).add(edge.dst)
            parent_container[edge.dst] = edge.src

    visited: set[str] = {start_symbol}
    queue: deque[tuple[str, int]] = deque([(start_symbol, 0)])

    while queue:
        current, hops = queue.popleft()
        if max_hops is not None and hops >= max_hops:
            continue
        for neighbor in adjacency.get(current, ()):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, hops + 1))

    if include_containers:
        for sym in list(visited):
            curr = parent_container.get(sym)
            while curr:
                visited.add(curr)
                curr = parent_container.get(curr)

    return visited


def _chunk_mentions_blast_symbol(text: str, blast_symbols: set[str]) -> bool:
    """Check if a Markdown chunk explicitly mentions any dotted symbol in the blast radius."""
    for sym in blast_symbols:
        if "." not in sym:
            continue
        if re.search(rf"(?<![\w.]){re.escape(sym)}(?![\w.])", text):
            return True
    return False


def filter_chunks_by_blast_radius(
    chunks: list[Chunk],
    graph: CodeGraph,
    start_symbol: str,
    max_hops: int | None = None,
) -> tuple[set[str], list[Chunk]]:
    """
    Keep only chunks whose owner_symbol_id is in `start_symbol`'s blast radius,
    or Markdown chunks whose text explicitly references a symbol in the blast radius.
    """
    blast_symbols = compute_graph_blast_radius(
        graph=graph,
        start_symbol=start_symbol,
        max_hops=max_hops,
        include_containers=True,
    )

    scoped_chunks: list[Chunk] = []
    for chunk in chunks:
        if chunk.owner_symbol_id in blast_symbols:
            scoped_chunks.append(chunk)
        elif chunk.source_type == "markdown" and _chunk_mentions_blast_symbol(
            chunk.text, blast_symbols
        ):
            scoped_chunks.append(chunk)

    return blast_symbols, scoped_chunks


def _hydrate_embeddings_from_index(
    chunks: list[Chunk],
    vector_index: VectorStoreIndex | None,
) -> None:
    """Copy cached vectors from an existing VectorStoreIndex onto Chunk.embedding."""
    if vector_index is None:
        return

    store_data = getattr(vector_index.vector_store, "_data", None)
    embedding_map = getattr(store_data, "embedding_dict", {}) if store_data else {}

    for chunk in chunks:
        if chunk.embedding is None and chunk.id in embedding_map:
            chunk.embedding = embedding_map[chunk.id]


def retrieve_graph_scoped(
    query: str,
    start_symbol: str,
    graph: CodeGraph,
    chunks: list[Chunk],
    embedder: LlamaBGEEmbedder,
    similarity_top_k: int = 5,
    max_hops: int | None = None,
    base_vector_index: VectorStoreIndex | None = None,
) -> GraphScopedRetrievalResult:
    """
    Run Hybrid (BGE Vector + BM25 Reciprocal Rank Fusion) retrieval restricted
    strictly to chunks in the upstream blast radius of `start_symbol` (FR-09).
    """
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    _hydrate_embeddings_from_index(chunks, base_vector_index)

    blast_symbols, scoped_chunks = filter_chunks_by_blast_radius(
        chunks=chunks,
        graph=graph,
        start_symbol=start_symbol,
        max_hops=max_hops,
    )

    if not scoped_chunks:
        return GraphScopedRetrievalResult(
            target_symbol=start_symbol,
            query=query,
            blast_radius_symbols=blast_symbols,
            scoped_chunk_count=0,
            total_chunk_count=len(chunks),
            results=[],
        )

    effective_top_k = min(similarity_top_k, len(scoped_chunks))

    nodes = [
        TextNode(
            id_=chunk.id,
            text=chunk.text,
            embedding=chunk.embedding,
            metadata={
                "owner_symbol_id": chunk.owner_symbol_id,
                "source_type": chunk.source_type,
                "source_file": chunk.source_file,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "parent_document_id": chunk.parent_document_id,
            },
        )
        for chunk in scoped_chunks
    ]

    scoped_vector_index = VectorStoreIndex(
        nodes=nodes,
        embed_model=embedder.model,
        show_progress=False,
    )
    scoped_vector_retriever = scoped_vector_index.as_retriever(
        similarity_top_k=effective_top_k
    )
    scoped_bm25_retriever = build_bm25_retriever(
        chunks=scoped_chunks,
        similarity_top_k=effective_top_k,
    )
    scoped_hybrid_retriever = build_hybrid_retriever(
        vector_retriever=scoped_vector_retriever,
        bm25_retriever=scoped_bm25_retriever,
        similarity_top_k=effective_top_k,
    )

    results = scoped_hybrid_retriever.retrieve(query)

    return GraphScopedRetrievalResult(
        target_symbol=start_symbol,
        query=query,
        blast_radius_symbols=blast_symbols,
        scoped_chunk_count=len(scoped_chunks),
        total_chunk_count=len(chunks),
        results=results,
    )