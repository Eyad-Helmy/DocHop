
from codearchguard.rag.llama_embedder import LlamaBGEEmbedder


def test_llama_bge_embedder():
    embedder = LlamaBGEEmbedder()

    documents = [
        "Flask is a Python web framework.",
        "A request context stores request-specific information."
    ]

    embeddings = embedder.embed_documents(documents)
    query_embedding = embedder.embed_query(
        "How does Flask handle requests?"
    )

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    assert len(query_embedding) == 384

    print("Documents:", len(embeddings))
    print("Embedding dimensions:", len(embeddings[0]))
    print("Query dimensions:", len(query_embedding))
