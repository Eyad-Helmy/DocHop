from codearchguard.models import CodeGraph, EdgeKind, Node, NodeKind
from codearchguard.rag.doc_graph import attach_doc_chunks, doc_chunk_node_id
from codearchguard.rag.document import Chunk


def make_chunk(chunk_id: str, owner: str, source_type: str = "docstring") -> Chunk:
    return Chunk(
        id=chunk_id,
        parent_document_id="document",
        owner_symbol_id=owner,
        source_type=source_type,
        source_file="docs/architecture.md",
        start_line=4,
        end_line=6,
        chunk_index=0,
        text="Documentation content.",
        start_char=0,
        end_char=22,
        context="",
    )


def test_attach_doc_chunks_creates_namespaced_nodes_and_symbol_edges():
    graph = CodeGraph()
    graph.add_node(Node("pkg.module.work", NodeKind.FUNCTION, "work", "module.py"))
    chunk = make_chunk("doc_pkg.module.work_0", "pkg.module.work")

    attach_doc_chunks(graph, [chunk])

    node_id = doc_chunk_node_id(chunk)
    node = graph.nodes[node_id]
    assert node.kind == NodeKind.DOC_CHUNK
    assert node.content == chunk.text
    assert node.owner_symbol_id == "pkg.module.work"
    assert node.source_type == "docstring"
    assert {(edge.src, edge.dst) for edge in graph.edges_of(EdgeKind.DOCUMENTED_BY)} == {
        ("pkg.module.work", node_id)
    }


def test_attach_doc_chunks_keeps_repository_markdown_explicit_and_unlinked():
    graph = CodeGraph()
    chunk = make_chunk("md_README_0", "__repo__", source_type="markdown")

    attach_doc_chunks(graph, [chunk])

    node = graph.nodes[doc_chunk_node_id(chunk)]
    assert node.owner_symbol_id == "__repo__"
    assert node.source_type == "markdown"
    assert graph.edges_of(EdgeKind.DOCUMENTED_BY) == []
