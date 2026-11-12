
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.llms import MockLLM


def build_hybrid_retriever(
    vector_retriever,
    bm25_retriever,
    similarity_top_k: int = 5,
):
    hybrid_retriever = QueryFusionRetriever(
        retrievers=[
            vector_retriever,
            bm25_retriever,
        ],
        similarity_top_k=similarity_top_k,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
        llm=MockLLM(),
    )

    return hybrid_retriever
