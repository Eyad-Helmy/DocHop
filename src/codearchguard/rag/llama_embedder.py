
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class LlamaBGEEmbedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5"
    ):
        self.model = HuggingFaceEmbedding(
            model_name=model_name
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        return self.model.get_text_embedding_batch(texts)

    def embed_query(self, query: str) -> list[float]:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        return self.model.get_query_embedding(query)
