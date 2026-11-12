
from codearchguard.rag.pipeline import run_ingestion


def test_run_ingestion(tmp_path):

    # Create a small sample repository
    sample_file = tmp_path / "sample.py"

    sample_file.write_text(
        '''
def calculate_total(price, quantity):
    """Calculate the total price."""
    return price * quantity


class Product:
    """Represent a product."""

    def __init__(self, name):
        self.name = name
''',
        encoding="utf-8"
    )

    # Run the complete ingestion pipeline
    result = run_ingestion(tmp_path, max_chars=100)

    # Check the returned result
    assert result.graph is not None
    assert len(result.documents) > 0
    assert len(result.chunks) > 0

    # Check that code and documentation were extracted
    source_types = {doc.source_type for doc in result.documents}

    assert "code" in source_types
    assert "docstring" in source_types

    # Check code chunks remain complete
    code_documents = [
        doc for doc in result.documents
        if doc.source_type == "code"
    ]

    code_chunks = [
        chunk for chunk in result.chunks
        if chunk.source_type == "code"
    ]

    assert len(code_chunks) == len(code_documents)

    for document in code_documents:
        document_chunks = [
            chunk for chunk in code_chunks
            if chunk.parent_document_id == document.id
        ]

        assert len(document_chunks) == 1
        assert document_chunks[0].text == document.text.strip()

    # Check every chunk has a valid parent document
    document_ids = {doc.id for doc in result.documents}

    for chunk in result.chunks:
        assert chunk.parent_document_id in document_ids
