
from dataclasses import dataclass
from pathlib import Path

from codearchguard.analysis.extractor import extract_graph
from codearchguard.models import CodeGraph

from codearchguard.rag.document import Document, Chunk
from codearchguard.rag.ingestion import extract_documents
from codearchguard.rag.code_ingestion import extract_code_documents
from codearchguard.rag.chunking import chunk_documents
from codearchguard.rag.llama_embedder import LlamaBGEEmbedder


@dataclass
class RAGIngestionResult:
    graph: CodeGraph
    documents: list[Document]
    chunks: list[Chunk]


def run_ingestion(
    repo_root: str | Path,
    max_chars: int = 1000,
    embedder: LlamaBGEEmbedder | None = None
) -> RAGIngestionResult:

    repo_root = Path(repo_root).resolve()

    # 1. Extract code graph
    graph = extract_graph(repo_root)

    # 2. Extract code documents
    code_documents = extract_code_documents(graph, repo_root)

    # 3. Extract documentation documents
    documentation_documents = extract_documents(graph)

    # 4. Combine all documents
    documents = code_documents + documentation_documents

    # 5. Convert documents into chunks
    chunks = chunk_documents(documents, max_chars=max_chars)

    # 6. Generate embeddings if an embedder is provided
    if embedder is not None and chunks:

        embeddings = embedder.embed_documents(
            [chunk.text for chunk in chunks]
        )

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

    return RAGIngestionResult(
        graph=graph,
        documents=documents,
        chunks=chunks
    )
