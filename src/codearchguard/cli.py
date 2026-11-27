from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from codearchguard.analysis.extractor import _git_snapshot, extract_graph
from codearchguard.models import CodeGraph, EdgeKind, Node, NodeRole


def _add_common_repo_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("repo", help="Path to the Python repository")
    parser.add_argument(
        "--source-root",
        action="append",
        help="Source directory relative to the repository; repeat for multiple roots",
    )
    parser.add_argument("--no-docs", action="store_true", help="Skip DocChunk and DOCUMENTED_BY extraction")
    parser.add_argument("--neo4j-uri", help="Neo4j Bolt URI (e.g. bolt://localhost:7687)")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="password")


def _build_graph(repo: str, source_roots: list[str] | None, include_docs: bool) -> CodeGraph:
    repo_root = Path(repo).resolve()
    graph = extract_graph(repo_root, source_roots=source_roots)
    if include_docs:
        from codearchguard.rag.chunking import chunk_documents
        from codearchguard.rag.code_ingestion import extract_code_documents
        from codearchguard.rag.doc_graph import attach_doc_chunks
        from codearchguard.rag.ingestion import extract_documents

        documents = extract_code_documents(graph, repo_root) + extract_documents(graph, repo_root)
        chunks = chunk_documents(documents)
        attach_doc_chunks(graph, chunks)
    return graph


def _snapshot_only_graph(repo: str) -> CodeGraph:
    repo_root = Path(repo).resolve()
    commit_hash, is_dirty = _git_snapshot(repo_root)
    return CodeGraph(repository_root=str(repo_root), commit_hash=commit_hash, is_dirty=is_dirty)


def _format_node_tags(node: Node) -> str:
    tags = [role.value for role in sorted(node.roles, key=lambda r: r.value)]
    if node.has_dynamic_warning:
        tags.append("DynamicWarning")
    return f" [{', '.join(tags)}]" if tags else ""


def _run_analyze(args: argparse.Namespace) -> None:
    graph = _build_graph(args.repo, args.source_root, include_docs=not args.no_docs)
    roles = Counter(role.value for node in graph.nodes.values() for role in node.roles)
    dynamic_count = sum(1 for node in graph.nodes.values() if node.has_dynamic_warning)

    print(f"Snapshot: commit={graph.commit_hash or 'none'} dirty={graph.is_dirty}")
    print("Nodes:", dict(Counter(n.kind.value for n in graph.nodes.values())))
    print("Roles:", dict(roles))
    print("Dynamic warnings:", dynamic_count)
    print("Edges:", dict(Counter(e.kind.value for e in graph.edges)))

    if args.neo4j_uri:
        from codearchguard.graph.neo4j_store import Neo4jStore

        store = Neo4jStore(args.neo4j_uri, args.user, args.password)
        try:
            store.init_schema()
            store.save(graph)
        finally:
            store.close()
        print("Loaded into Neo4j.")


def _impact_from_memory(
    graph: CodeGraph,
    symbol: str,
    max_hops: int | None,
    include_containers: bool,
) -> tuple[list[Node], list[Node], list[Node]]:
    from codearchguard.rag.graph_scoped_retriever import compute_graph_blast_radius

    impacted_ids = compute_graph_blast_radius(
        graph, symbol, max_hops=max_hops, include_containers=include_containers
    )
    impacted = sorted((graph.nodes[nid] for nid in impacted_ids), key=lambda n: n.id)
    tests = [node for node in impacted if NodeRole.TEST in node.roles]
    doc_ids = {
        edge.dst
        for edge in graph.edges_of(EdgeKind.DOCUMENTED_BY)
        if edge.src in impacted_ids and edge.dst in graph.nodes
    }
    docs = sorted((graph.nodes[did] for did in doc_ids), key=lambda n: n.id)
    return impacted, tests, docs


def _run_impact(args: argparse.Namespace) -> None:
    include_containers = not args.no_containers

    if args.neo4j_uri:
        from codearchguard.graph.neo4j_store import Neo4jStore

        graph = _snapshot_only_graph(args.repo)
        store = Neo4jStore(args.neo4j_uri, args.user, args.password)
        try:
            impacted = store.blast_radius(
                graph,
                args.symbol,
                max_hops=args.max_hops,
                include_containers=include_containers,
                max_nodes=args.max_nodes,
            )
            tests = store.impacted_tests(graph, args.symbol, max_hops=args.max_hops)
            docs = store.related_doc_chunks(graph, args.symbol, max_hops=args.max_hops)
        finally:
            store.close()
        backend = "Neo4j"
    else:
        graph = _build_graph(args.repo, args.source_root, include_docs=not args.no_docs)
        impacted, tests, docs = _impact_from_memory(
            graph, args.symbol, args.max_hops, include_containers
        )
        backend = "In-Memory"

    print(f"Impact for {args.symbol} ({backend}):")
    print(f"\nImpacted symbols ({len(impacted)}):")
    for node in impacted:
        print(f"  - [{node.kind.value}] {node.id}{_format_node_tags(node)}")

    print(f"\nImpacted tests ({len(tests)}):")
    for node in tests:
        print(f"  - {node.id} ({node.file}:{node.start_line})")

    print(f"\nRelated DocChunks ({len(docs)}):")
    for node in docs:
        print(f"  - {node.id} [{node.source_type}] ({node.file}:{node.start_line}-{node.end_line})")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="cag")
    sub = parser.add_subparsers(dest="cmd", required=True)

    analyze_parser = sub.add_parser(
        "analyze",
        help="Analyze a repo, print graph/role/doc stats, and optionally load Neo4j",
    )
    _add_common_repo_args(analyze_parser)

    impact_parser = sub.add_parser(
        "impact",
        help="Show blast radius, impacted tests, and related DocChunks for a symbol",
    )
    _add_common_repo_args(impact_parser)
    impact_parser.add_argument("symbol", help="Fully qualified symbol ID (e.g. shop.models.Item)")
    impact_parser.add_argument("--max-hops", type=int, default=None, help="Optional hop limit (default: exhaustive)")
    impact_parser.add_argument("--max-nodes", type=int, default=None, help="Optional Neo4j node safety limit")
    impact_parser.add_argument("--no-containers", action="store_true", help="Exclude enclosing classes/modules")

    ask_parser = sub.add_parser("ask", help="Ask an LLM about graph-scoped repository excerpts")
    ask_parser.add_argument("repo", type=Path)
    ask_parser.add_argument("symbol", help="Fully qualified symbol ID")
    ask_parser.add_argument("question")

    args = parser.parse_args(argv)
    if args.cmd == "analyze":
        _run_analyze(args)
    elif args.cmd == "impact":
        _run_impact(args)
    elif args.cmd == "ask":
        from codearchguard.rag.answer import ask_repository

        try:
            print(ask_repository(args.repo, args.symbol, args.question))
        except (ValueError, RuntimeError) as error:
            ask_parser.exit(1, f"cag ask: {error}\n")


if __name__ == "__main__":
    main()
