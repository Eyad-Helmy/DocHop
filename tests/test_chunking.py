
from pathlib import Path

from codearchguard.analysis.extractor import extract_graph
from codearchguard.rag.ingestion import extract_documents
from codearchguard.rag.chunking import chunk_documents


def test_chunking():

    project_root = Path(__file__).resolve().parents[1]
    sample_repo = project_root / "tests" / "fixtures" / "sample_repo"

    graph = extract_graph(sample_repo)

    documents = extract_documents(graph)

    chunks = chunk_documents(documents, max_chars=100)

    print("Total Documents:", len(documents))
    print("Total Chunks:", len(chunks))

    for chunk in chunks:
        print("\n-------------------------")
        print("ID:", chunk.id)
        print("Parent:", chunk.parent_document_id)
        print("Owner:", chunk.owner_symbol_id)
        print("Chunk Index:", chunk.chunk_index)
        print("Text:\n", chunk.text)

    assert len(chunks) > 0

    for chunk in chunks:
        assert chunk.text.strip()
        assert chunk.parent_document_id
        assert chunk.source_type == "docstring"

def test_long_document_is_split():

    from codearchguard.rag.document import Document
    from codearchguard.rag.chunking import chunk_document

    document = Document(
        id="test_document",
        owner_symbol_id="shop.billing.calculate_total",
        source_type="docstring",
        source_file="shop/billing.py",
        start_line=1,
        end_line=20,
        text="A" * 250
    )

    chunks = chunk_document(document, max_chars=100)

    assert len(chunks) == 3

    for chunk in chunks:
        assert len(chunk.text) <= 100


def test_chunk_context_and_offsets():

    from codearchguard.rag.document import Document
    from codearchguard.rag.chunking import chunk_document

    document = Document(
        id="test_context",
        owner_symbol_id="shop.billing.calculate_total",
        source_type="docstring",
        source_file="shop/billing.py",
        start_line=1,
        end_line=10,
        text="First paragraph.\n\nSecond paragraph."
    )

    chunks = chunk_document(document, max_chars=100)

    assert len(chunks) == 2

    first_chunk = chunks[0]

    assert first_chunk.context == "shop.billing.calculate_total"
    assert first_chunk.start_char == 0
    assert first_chunk.text == "First paragraph."

    second_chunk = chunks[1]

    assert second_chunk.text == "Second paragraph."
    assert second_chunk.start_char == 18
