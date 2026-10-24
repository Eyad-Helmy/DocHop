"""Persist a CodeGraph into Neo4j 5 (idempotent: uses MERGE)."""
from __future__ import annotations
from codearchguard.models import CodeGraph, EdgeKind

SCHEMA = [
    "CREATE CONSTRAINT code_id IF NOT EXISTS FOR (n:CodeNode) REQUIRE n.id IS UNIQUE",
    "CREATE INDEX code_kind IF NOT EXISTS FOR (n:CodeNode) ON (n.kind)",
]


class Neo4jStore:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        from neo4j import GraphDatabase  # lazy: tests don't need a driver
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def init_schema(self):
        with self._driver.session() as s:
            for q in SCHEMA:
                s.run(q)

    def clear(self):
        with self._driver.session() as s:
            s.run("MATCH (n:CodeNode) DETACH DELETE n")

    def save(self, graph: CodeGraph, batch: int = 500):
        nodes = [dict(id=n.id, kind=n.kind.value, name=n.name, file=n.file,
                      start_line=n.start_line, end_line=n.end_line, docstring=n.docstring)
                 for n in graph.nodes.values()]
        with self._driver.session() as s:
            for i in range(0, len(nodes), batch):
                s.run("UNWIND $rows AS r MERGE (n:CodeNode {id: r.id}) SET n += r",
                      rows=nodes[i:i + batch])
            for kind in EdgeKind:  # relationship types can't be parameterised
                rows = [dict(src=e.src, dst=e.dst, line=e.line) for e in graph.edges_of(kind)]
                for i in range(0, len(rows), batch):
                    s.run(f"UNWIND $rows AS r MATCH (a:CodeNode {{id: r.src}}), (b:CodeNode {{id: r.dst}}) "
                          f"MERGE (a)-[e:{kind.value}]->(b) SET e.line = r.line",
                          rows=rows[i:i + batch])
