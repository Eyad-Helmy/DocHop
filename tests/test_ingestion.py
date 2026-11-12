
from pathlib import Path

from codearchguard.analysis.extractor import extract_graph
from codearchguard.rag.ingestion import extract_documents


def test_extract_documents():
    project_root = Path(__file__).resolve().parents[1]

    sample_repo = project_root / "tests" / "fixtures" / "sample_repo"

    print("Sample repo path:", sample_repo)
    print("Path exists:", sample_repo.exists())

    graph = extract_graph(sample_repo)

    documents = extract_documents(graph)

    print("Total Code Nodes:", len(graph.nodes))
    print("Total Documents:", len(documents))

    for doc in documents:
        print("\n-------------------------")
        print("ID:", doc.id)
        print("Owner:", doc.owner_symbol_id)
        print("Source:", doc.source_file)
        print("Text:", doc.text)

    assert len(graph.nodes) > 0
    assert len(documents) == 1
    assert documents[0].owner_symbol_id == "shop.billing.calculate_total"
