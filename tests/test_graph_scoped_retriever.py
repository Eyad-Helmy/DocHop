from pathlib import Path

from codearchguard.analysis.extractor import extract_graph
from codearchguard.rag.chunking import chunk_documents
from codearchguard.rag.code_ingestion import extract_code_documents
from codearchguard.rag.graph_scoped_retriever import (
    compute_graph_blast_radius,
    filter_chunks_by_blast_radius,
    retrieve_graph_scoped,
)
from codearchguard.rag.ingestion import extract_documents
from codearchguard.rag.llama_embedder import LlamaBGEEmbedder
from codearchguard.rag.pipeline import run_ingestion

REPO = Path(__file__).parent / "fixtures" / "sample_repo"


def test_compute_graph_blast_radius_excludes_unrelated_symbols():
    graph = extract_graph(REPO)
    blast = compute_graph_blast_radius(graph, "shop.models.Item")

    # Upstream callers, subclasses, and defined methods ARE in the blast radius
    assert "shop.models.Item" in blast
    assert "shop.models.Item.__init__" in blast
    assert "shop.models.DiscountedItem" in blast
    assert "shop.models.DiscountedItem.total" in blast
    assert "shop.billing.make_item" in blast
    assert "shop.billing.checkout" in blast

    # Unrelated sibling functions in shop.billing are NOT in the blast radius
    assert "shop.billing.calculate_total" not in blast
    assert "shop.billing.helper" not in blast


def test_filter_chunks_by_blast_radius_and_markdown_adr(tmp_path):
    pkg = tmp_path / "shop"
    pkg.mkdir()
    (pkg / "__init__.py").write_text('"""Shop package docstring."""\n', encoding="utf-8")
    (pkg / "models.py").write_text(
        "class Item:\n"
        "    def __init__(self, price):\n"
        "        self.price = price\n",
        encoding="utf-8",
    )
    (pkg / "billing.py").write_text(
        "from shop.models import Item\n\n"
        "def checkout(p):\n"
        "    return Item(p)\n\n"
        "def unrelated_tax(p):\n"
        '    """Compute unrelated tax."""\n'
        "    return p * 0.15\n",
        encoding="utf-8",
    )

    # Real ADR referencing shop.billing.checkout
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "adr-001.md").write_text(
        "# ADR 001: Checkout SLA\n"
        "Calling shop.billing.checkout must complete within 200ms.\n",
        encoding="utf-8",
    )

    # Hidden dot-directory (.pytest_cache) that must be ignored
    cache_dir = tmp_path / ".pytest_cache"
    cache_dir.mkdir()
    (cache_dir / "README.md").write_text("# pytest cache dir\n", encoding="utf-8")

    result = run_ingestion(tmp_path, max_chars=200)

    # Verify .pytest_cache/README.md was skipped and docs/adr-001.md was ingested
    md_docs = [d for d in result.documents if d.source_type == "markdown"]
    assert len(md_docs) == 1
    assert md_docs[0].source_file == "docs/adr-001.md"
    assert md_docs[0].owner_symbol_id == "shop.billing.checkout"

    # Filter chunks to blast radius of shop.models.Item
    blast, scoped_chunks = filter_chunks_by_blast_radius(
        result.chunks, result.graph, "shop.models.Item"
    )
    scoped_owners = {c.owner_symbol_id for c in scoped_chunks}

    assert "shop.billing.checkout" in scoped_owners
    assert "shop.billing.unrelated_tax" not in scoped_owners
    assert any(c.source_type == "markdown" for c in scoped_chunks)


def test_retrieve_graph_scoped_on_sample_repo():
    result = run_ingestion(REPO, max_chars=200)
    embedder = LlamaBGEEmbedder()

    scoped_res = retrieve_graph_scoped(
        query="Calculate the total price of products",
        start_symbol="shop.models.Item",
        graph=result.graph,
        chunks=result.chunks,
        embedder=embedder,
        similarity_top_k=5,
    )

    retrieved_owners = {
        node_with_score.node.metadata["owner_symbol_id"]
        for node_with_score in scoped_res.results
    }

    assert len(scoped_res.results) > 0
    # Even though query matches calculate_total's docstring word-for-word,
    # graph-scoped retrieval strictly excludes it because it is outside Item's blast radius!
    assert "shop.billing.calculate_total" not in retrieved_owners