
from pathlib import Path

from codearchguard.rag.pipeline import run_ingestion
from codearchguard.rag.llama_embedder import LlamaBGEEmbedder
from codearchguard.rag.llama_vector_index import load_or_build_vector_index


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPO_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "flask"

PERSIST_DIR = PROJECT_ROOT / "storage" / "vector_index"


def test_vector_retriever_on_flask():

    ingestion_result = run_ingestion(
        repo_root=REPO_ROOT
    )

    embedder = LlamaBGEEmbedder()

    vector_index = load_or_build_vector_index(
        chunks=ingestion_result.chunks,
        embedder=embedder,
        persist_dir=str(PERSIST_DIR),
    )

    retriever = vector_index.as_retriever(
        similarity_top_k=5
    )

    query = "How does Flask create and initialize an application?"

    results = retriever.retrieve(query)

    assert len(results) == 5

    print("\nUser Query:", query)
    print("\nTop 5 Retrieved Results:")

    for index, result in enumerate(results, start=1):

        metadata = result.node.metadata

        print(f"\nResult {index}")
        print("Score:", round(result.score, 4))
        print("File:", metadata["source_file"])
        print("Symbol:", metadata["owner_symbol_id"])
        print("Type:", metadata["source_type"])
        print("Text:", result.node.get_content()[:300])
