# fixture for components executed multiple times throughout multiple tests
# still needs to be used in the full_ingestion and test_retriever
from pathlib import Path
import pytest

from codearchguard.rag.llama_embedder import LlamaBGEEmbedder
from codearchguard.rag.llama_vector_index import load_or_build_vector_index
from codearchguard.rag.pipeline import run_ingestion

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FLASK_REPO_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "flask"
PERSIST_DIR = PROJECT_ROOT / "storage" / "vector_index"


@pytest.fixture(scope="session")
def bge_embedder():
    return LlamaBGEEmbedder()


@pytest.fixture(scope="session")
def flask_ingestion_result():
    return run_ingestion(repo_root=FLASK_REPO_ROOT)


@pytest.fixture(scope="session")
def flask_vector_index(flask_ingestion_result, bge_embedder):
    return load_or_build_vector_index(
        chunks=flask_ingestion_result.chunks,
        embedder=bge_embedder,
        persist_dir=str(PERSIST_DIR),
    )