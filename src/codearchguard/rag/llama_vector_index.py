
import hashlib
import json
from pathlib import Path

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.schema import TextNode

from codearchguard.rag.document import Chunk
from codearchguard.rag.llama_embedder import LlamaBGEEmbedder


def calculate_chunks_fingerprint(chunks: list[Chunk]) -> str:
    """Calculate a stable fingerprint for the indexed chunks."""

    chunks_data = []

    for chunk in sorted(chunks, key=lambda item: item.id):
        chunks_data.append({
            "id": chunk.id,
            "parent_document_id": chunk.parent_document_id,
            "owner_symbol_id": chunk.owner_symbol_id,
            "source_type": chunk.source_type,
            "source_file": chunk.source_file,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
        })

    serialized_data = json.dumps(
        chunks_data,
        sort_keys=True,
        ensure_ascii=False,
    )

    return hashlib.sha256(
        serialized_data.encode("utf-8")
    ).hexdigest()


def build_vector_index(
    chunks: list[Chunk],
    embedder: LlamaBGEEmbedder,
    persist_dir: str = "storage/vector_index",
) -> VectorStoreIndex:

    if not chunks:
        raise ValueError("No chunks available to index.")

    nodes = []

    for chunk in chunks:
        node = TextNode(
            id_=chunk.id,
            text=chunk.text,
            metadata={
                "owner_symbol_id": chunk.owner_symbol_id,
                "source_type": chunk.source_type,
                "source_file": chunk.source_file,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "parent_document_id": chunk.parent_document_id,
            },
        )

        nodes.append(node)

    index = VectorStoreIndex(
        nodes=nodes,
        embed_model=embedder.model,
        show_progress=True,
    )

    storage_path = Path(persist_dir)
    storage_path.mkdir(parents=True, exist_ok=True)

    index.storage_context.persist(
        persist_dir=str(storage_path)
    )

    fingerprint = calculate_chunks_fingerprint(chunks)

    fingerprint_file = storage_path / "fingerprint.json"

    fingerprint_file.write_text(
        json.dumps(
            {"fingerprint": fingerprint},
            indent=4,
        ),
        encoding="utf-8",
    )

    return index


def load_vector_index(
    chunks: list[Chunk],
    embedder: LlamaBGEEmbedder,
    persist_dir: str = "storage/vector_index",
) -> VectorStoreIndex:

    storage_path = Path(persist_dir)

    fingerprint_file = storage_path / "fingerprint.json"

    if not fingerprint_file.exists():
        raise FileNotFoundError(
            "Index fingerprint not found."
        )

    saved_data = json.loads(
        fingerprint_file.read_text(encoding="utf-8")
    )

    saved_fingerprint = saved_data["fingerprint"]

    current_fingerprint = calculate_chunks_fingerprint(chunks)

    if saved_fingerprint != current_fingerprint:
        raise ValueError(
            "Stored vector index is outdated. Rebuild required."
        )

    storage_context = StorageContext.from_defaults(
        persist_dir=str(storage_path)
    )

    index = load_index_from_storage(
        storage_context,
        embed_model=embedder.model,
    )

    return index


def load_or_build_vector_index(
    chunks: list[Chunk],
    embedder: LlamaBGEEmbedder,
    persist_dir: str = "storage/vector_index",
) -> VectorStoreIndex:

    storage_path = Path(persist_dir)

    fingerprint_file = storage_path / "fingerprint.json"
    index_file = storage_path / "index_store.json"

    if fingerprint_file.exists() and index_file.exists():

        try:
            print("\nChecking saved vector index...")

            index = load_vector_index(
                chunks=chunks,
                embedder=embedder,
                persist_dir=persist_dir,
            )

            print("Existing vector index is up to date.")

            return index

        except (ValueError, FileNotFoundError, KeyError):
            print("Stored index is outdated. Rebuilding...")

    else:
        print("No valid stored index found. Building a new one...")

    return build_vector_index(
        chunks=chunks,
        embedder=embedder,
        persist_dir=persist_dir,
    )
