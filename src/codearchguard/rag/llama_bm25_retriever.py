
from llama_index.core import Document as LlamaDocument
from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever

from codearchguard.rag.document import Chunk


def build_bm25_retriever(
    chunks: list[Chunk],
    similarity_top_k: int = 5,
) -> BM25Retriever:

    if not chunks:
        raise ValueError("No chunks available for BM25 retrieval.")

    nodes = []

    for chunk in chunks:
        node = TextNode(
            id_=chunk.id,
            text=chunk.text,
            metadata={
                "owner_symbol_id": chunk.owner_symbol_id,
                "source_type": chunk.source_type,
                "source_file": chunk.source_file,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "parent_document_id": chunk.parent_document_id,
            },
        )

        nodes.append(node)

    retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=similarity_top_k,
    )

    return retriever
