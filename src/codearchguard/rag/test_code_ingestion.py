
from pathlib import Path

from codearchguard.analysis.extractor import extract_graph
from codearchguard.rag.code_ingestion import extract_code_documents


def test_extract_code_documents():

    # Locate project and sample repository
    project_root = Path(__file__).resolve().parents[3]
    sample_repo = project_root / "tests" / "fixtures" / "sample_repo"

    # Extract code graph
    graph = extract_graph(sample_repo)

    # Extract code documents
    documents = extract_code_documents(graph, sample_repo)

    # Display results
    print("Total Code Nodes:", len(graph.nodes))
    print("Total Code Documents:", len(documents))

    for doc in documents:
        print("\n-------------------------")
        print("ID:", doc.id)
        print("Owner:", doc.owner_symbol_id)
        print("Source:", doc.source_file)
        print("Lines:", doc.start_line, "-", doc.end_line)
        print("Text:\n", doc.text)

    # Basic validation
    assert len(documents) > 0

    # Test normal function extraction
    calculate_total = next(
        doc for doc in documents
        if doc.owner_symbol_id == "shop.billing.calculate_total"
    )

    assert calculate_total.source_type == "code"
    assert "def calculate_total" in calculate_total.text

    # Test Item class document
    item_class = next(
        doc for doc in documents
        if doc.owner_symbol_id == "shop.models.Item"
    )

    assert "class Item:" in item_class.text
    assert "def __init__" not in item_class.text

    # Test DiscountedItem class document
    discounted_class = next(
        doc for doc in documents
        if doc.owner_symbol_id == "shop.models.DiscountedItem"
    )

    assert "class DiscountedItem(Item):" in discounted_class.text
    assert "def total" not in discounted_class.text

    # Test independent method document
    total_method = next(
        doc for doc in documents
        if doc.owner_symbol_id == "shop.models.DiscountedItem.total"
    )

    assert "def total" in total_method.text
    assert "return self.price * 0.9" in total_method.text

    # Test constructor document
    item_constructor = next(
        doc for doc in documents
        if doc.owner_symbol_id == "shop.models.Item.__init__"
    )

    assert "def __init__" in item_constructor.text
    assert "self.price = price" in item_constructor.text
