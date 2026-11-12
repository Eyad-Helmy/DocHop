
from pathlib import Path

from codearchguard.rag.pipeline import run_ingestion
from codearchguard.rag.llama_embedder import LlamaBGEEmbedder
from codearchguard.rag.llama_vector_index import (
    load_or_build_vector_index,
)
from codearchguard.rag.llama_bm25_retriever import (
    build_bm25_retriever,
)
from codearchguard.rag.llama_hybrid_retriever import (
    build_hybrid_retriever,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPO_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "flask"

PERSIST_DIR = PROJECT_ROOT / "storage" / "vector_index"


def test_llama_hybrid_search_on_flask(flask_ingestion_result, flask_vector_index):
    # 1 - 3. Chunk, load embedder, load embeddings or create them
    chunks = flask_ingestion_result.chunks
    print("\nTotal chunks:", len(chunks))

    # 4. Build vector retriever
    vector_retriever = flask_vector_index.as_retriever(
        similarity_top_k=10
    )

    # 5. Build BM25 retriever
    bm25_retriever = build_bm25_retriever(
        chunks=chunks,
        similarity_top_k=10,
    )

    # 6. Build hybrid retriever
    hybrid_retriever = build_hybrid_retriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        similarity_top_k=10,
    )

    # 7. Test queries
    queries = [
        "How does Flask create and initialize an application?",
        "Where is the request context implemented?",
        "How does Flask register routes?",
    ]

    for query in queries:

        print(f"\n========== QUERY: {query} ==========")

        results = hybrid_retriever.retrieve(query)

        assert results, f"No results found for: {query}"

        for rank, result in enumerate(results, start=1):

            metadata = result.node.metadata

            print(f"\nResult {rank}")
            print("Score:", round(result.score, 4))
            print("File:", metadata["source_file"])
            print("Symbol:", metadata["owner_symbol_id"])
            print("Type:", metadata["source_type"])
            print("Text:", result.node.get_content()[:500])
