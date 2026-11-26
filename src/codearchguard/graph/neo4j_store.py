"""Persist snapshot-scoped CodeGraphs into Neo4j 5."""
from __future__ import annotations

import hashlib

from codearchguard.models import CodeGraph, EdgeKind, NodeKind, NodeRole


SCHEMA = [
    "CREATE CONSTRAINT code_snapshot_id IF NOT EXISTS FOR (n:CodeNode) REQUIRE n.snapshot_id IS UNIQUE",
    "CREATE INDEX code_kind IF NOT EXISTS FOR (n:CodeNode) ON (n.kind)",
    "CREATE INDEX code_owner_symbol IF NOT EXISTS FOR (n:CodeNode) ON (n.owner_symbol_id)",
]


def _snapshot_namespace(graph: CodeGraph) -> str:
    """Stable namespace for a repository checkout; dirty is metadata, not identity."""
    revision = graph.commit_hash or "unversioned"
    raw = f"{graph.repository_root}\0{revision}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _snapshot_node_id(graph: CodeGraph, node_id: str) -> str:
    return f"{_snapshot_namespace(graph)}:{node_id}"


class Neo4jStore:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        from neo4j import GraphDatabase  # lazy: tests don't need a driver
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def init_schema(self):
        """Migrate the legacy global-id schema without deleting saved graphs.

        Earlier releases made ``CodeNode.id`` globally unique. Existing nodes are
        retained under a ``legacy:`` snapshot namespace before that constraint is
        removed, allowing current commit-scoped snapshots to coexist with them.
        """
        with self._driver.session() as session:
            session.run("DROP CONSTRAINT code_id IF EXISTS")
            session.run(
                "MATCH (n:CodeNode) WHERE n.snapshot_id IS NULL "
                "SET n.snapshot_id = 'legacy:' + n.id"
            )
            for query in SCHEMA:
                session.run(query)

    def clear(self):
        with self._driver.session() as session:
            session.run("MATCH (n:CodeNode) DETACH DELETE n")

    def save(self, graph: CodeGraph, batch: int = 500):
        namespace = _snapshot_namespace(graph)
        nodes = [
            dict(
                snapshot_id=f"{namespace}:{node.id}",
                id=node.id,
                kind=node.kind.value,
                name=node.name,
                file=node.file,
                start_line=node.start_line,
                end_line=node.end_line,
                docstring=node.docstring,
                content=node.content,
                owner_symbol_id=node.owner_symbol_id,
                source_type=node.source_type,
                has_dynamic_warning=node.has_dynamic_warning,
                roles=sorted(role.value for role in node.roles),
                repository_root=graph.repository_root,
                commit_hash=graph.commit_hash,
                is_dirty=graph.is_dirty,
            )
            for node in graph.nodes.values()
        ]
        with self._driver.session() as session:
            for kind in NodeKind:
                rows = [row for row in nodes if row["kind"] == kind.value]
                self._save_node_rows(session, kind, rows, batch)
            self._save_role_labels(session, nodes, batch)

            for kind in EdgeKind:  # relationship types cannot be parameterised
                rows = [
                    dict(
                        src=_snapshot_node_id(graph, edge.src),
                        dst=_snapshot_node_id(graph, edge.dst),
                        line=edge.line,
                    )
                    for edge in graph.edges_of(kind)
                ]
                for index in range(0, len(rows), batch):
                    session.run(
                        f"UNWIND $rows AS r "
                        f"MATCH (a:CodeNode {{snapshot_id: r.src}}), "
                        f"(b:CodeNode {{snapshot_id: r.dst}}) "
                        f"MERGE (a)-[e:{kind.value}]->(b) SET e.line = r.line",
                        rows=rows[index:index + batch],
                    )

    @staticmethod
    def _save_node_rows(session, kind: NodeKind, rows: list[dict], batch: int) -> None:
        for index in range(0, len(rows), batch):
            session.run(
                f"UNWIND $rows AS r MERGE (n:CodeNode:{kind.value} {{snapshot_id: r.snapshot_id}}) "
                "SET n += r",
                rows=rows[index:index + batch],
            )

    @staticmethod
    def _save_role_labels(session, nodes: list[dict], batch: int) -> None:
        for index in range(0, len(nodes), batch):
            rows = nodes[index:index + batch]
            session.run(
                "UNWIND $rows AS r MATCH (n:CodeNode {snapshot_id: r.snapshot_id}) "
                "REMOVE n:Test:EntryPoint",
                rows=rows,
            )
        for role in NodeRole:
            rows = [
                {"snapshot_id": row["snapshot_id"]}
                for row in nodes if role.value in row["roles"]
            ]
            for index in range(0, len(rows), batch):
                session.run(
                    f"UNWIND $rows AS r MATCH (n:CodeNode {{snapshot_id: r.snapshot_id}}) SET n:{role.value}",
                    rows=rows[index:index + batch],
                )
