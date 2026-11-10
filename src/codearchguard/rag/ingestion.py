
from codearchguard.models import CodeGraph, NodeKind
from codearchguard.rag.document import Document


def extract_documents(graph: CodeGraph) -> list[Document]:

    documents = []

    for node in graph.nodes.values():

        # Skip nodes without documentation
        if not node.docstring.strip():
            continue

        # Skip module nodes initially
        if node.kind == NodeKind.MODULE:
            continue

        document = Document(

            id=f"doc_{node.id}",

            owner_symbol_id=node.id,

            source_type="docstring",

            source_file=node.file,

            start_line=node.start_line,

            end_line=node.end_line,

            text=node.docstring.strip()
        )

        documents.append(document)

    return documents
