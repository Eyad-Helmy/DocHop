"""Persist snapshot-scoped CodeGraphs into Neo4j 5."""
from __future__ import annotations

import hashlib

from codearchguard.models import CodeGraph, EdgeKind, Node, NodeKind, NodeRole


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


class SnapshotNotFoundError(LookupError):
    """The requested graph snapshot has not been persisted to Neo4j."""


def _validated_hops(max_hops: int | None) -> int | None:
    """Validate an optional BFS hop bound (None means exhaustive)."""
    if max_hops is None:
        return None
    if not isinstance(max_hops, int) or isinstance(max_hops, bool):
        raise TypeError("max_hops must be an integer or None")
    if max_hops < 0:
        raise ValueError("max_hops must be non-negative")
    return max_hops


def _validated_max_nodes(max_nodes: int | None) -> int | None:
    """Validate an optional node-count safety budget."""
    if max_nodes is None:
        return None
    if not isinstance(max_nodes, int) or isinstance(max_nodes, bool):
        raise TypeError("max_nodes must be an integer or None")
    if max_nodes < 1:
        raise ValueError("max_nodes must be at least 1")
    return max_nodes


def _node_from_record(record) -> Node:
    """Convert a Neo4j record (or a dict from a test double) into a Node."""
    data = record.data() if hasattr(record, "data") else dict(record)
    return Node(
        id=data["id"],
        kind=NodeKind(data["kind"]),
        name=data["name"],
        file=data["file"],
        start_line=data.get("start_line", 0),
        end_line=data.get("end_line", 0),
        docstring=data.get("docstring") or "",
        has_dynamic_warning=bool(data.get("has_dynamic_warning", False)),
        roles=frozenset(NodeRole(role) for role in (data.get("roles") or [])),
        content=data.get("content") or "",
        owner_symbol_id=data.get("owner_symbol_id") or "",
        source_type=data.get("source_type") or "",
    )


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

    def blast_radius(
        self,
        graph: CodeGraph,
        start_symbol: str,
        max_hops: int | None = None,
        include_containers: bool = True,
        max_nodes: int | None = None,
    ) -> list[Node]:
        """Return a snapshot-scoped impact set via cycle-safe frontier BFS.

        When ``max_hops`` is ``None`` (the default), traversal is exhaustive
        and matches the in-memory BFS: CALLS and INHERITS are traversed against
        their stored direction; DEFINES is traversed with its stored direction.
        Each BFS level runs a single 1-hop Cypher query over the current frontier
        to avoid unbounded variable-length Cypher path enumeration.
        """
        hops = _validated_hops(max_hops)
        node_limit = _validated_max_nodes(max_nodes)
        start_snapshot_id = _snapshot_node_id(graph, start_symbol)
        fields = self._node_fields("node")

        start_query = (
            "MATCH (node:CodeNode {snapshot_id: $start_snapshot_id}) "
            f"RETURN {fields}"
        )
        frontier_query = f"""
            MATCH (src:CodeNode)-[r:CALLS|INHERITS|DEFINES]-(node:CodeNode)
            WHERE src.snapshot_id IN $frontier_snapshot_ids
              AND NOT node.snapshot_id IN $visited_snapshot_ids
              AND (
                (type(r) IN ['CALLS', 'INHERITS'] AND endNode(r) = src)
                OR (type(r) = 'DEFINES' AND startNode(r) = src)
              )
            RETURN DISTINCT {fields} ORDER BY node.id
        """
        container_query = f"""
            MATCH (node:CodeNode)-[:DEFINES]->(child:CodeNode)
            WHERE child.snapshot_id IN $frontier_snapshot_ids
              AND NOT node.snapshot_id IN $visited_snapshot_ids
            RETURN DISTINCT {fields} ORDER BY node.id
        """

        with self._driver.session() as session:
            start_records = list(session.run(
                start_query, start_snapshot_id=start_snapshot_id,
            ))
            if not start_records:
                raise SnapshotNotFoundError(
                    f"Symbol '{start_symbol}' was not found in the requested Neo4j snapshot."
                )

            start_node = _node_from_record(start_records[0])
            visited_by_id: dict[str, Node] = {start_node.id: start_node}
            visited_snapshot_ids: list[str] = [start_snapshot_id]
            frontier_snapshot_ids: list[str] = [start_snapshot_id]

            hop = 0
            while frontier_snapshot_ids and (hops is None or hop < hops):
                if node_limit is not None and len(visited_by_id) >= node_limit:
                    break
                next_nodes = [
                    _node_from_record(record)
                    for record in session.run(
                        frontier_query,
                        frontier_snapshot_ids=frontier_snapshot_ids,
                        visited_snapshot_ids=visited_snapshot_ids,
                    )
                ]
                next_frontier: list[str] = []
                for node in next_nodes:
                    if node.id in visited_by_id:
                        continue
                    if node_limit is not None and len(visited_by_id) >= node_limit:
                        break
                    visited_by_id[node.id] = node
                    snap_id = _snapshot_node_id(graph, node.id)
                    visited_snapshot_ids.append(snap_id)
                    next_frontier.append(snap_id)
                frontier_snapshot_ids = next_frontier
                hop += 1

            if include_containers:
                container_frontier = list(visited_snapshot_ids)
                while container_frontier:
                    if node_limit is not None and len(visited_by_id) >= node_limit:
                        break
                    parents = [
                        _node_from_record(record)
                        for record in session.run(
                            container_query,
                            frontier_snapshot_ids=container_frontier,
                            visited_snapshot_ids=visited_snapshot_ids,
                        )
                    ]
                    next_containers: list[str] = []
                    for parent in parents:
                        if parent.id in visited_by_id:
                            continue
                        if node_limit is not None and len(visited_by_id) >= node_limit:
                            break
                        visited_by_id[parent.id] = parent
                        snap_id = _snapshot_node_id(graph, parent.id)
                        visited_snapshot_ids.append(snap_id)
                        next_containers.append(snap_id)
                    container_frontier = next_containers

        return [node for _, node in sorted(visited_by_id.items())]

    def impacted_tests(
        self,
        graph: CodeGraph,
        start_symbol: str,
        max_hops: int | None = None,
    ) -> list[Node]:
        """Return Test-role symbols inside a database-computed blast radius."""
        impacted = self.blast_radius(graph, start_symbol, max_hops=max_hops)
        return self._nodes_by_snapshot_ids(
            [_snapshot_node_id(graph, node.id) for node in impacted], label="Test"
        )

    def related_doc_chunks(
        self,
        graph: CodeGraph,
        start_symbol: str,
        max_hops: int | None = None,
    ) -> list[Node]:
        """Return DocChunks attached to symbols in a database-computed impact set."""
        impacted = self.blast_radius(graph, start_symbol, max_hops=max_hops)
        query = f"""
            MATCH (symbol:CodeNode)-[:DOCUMENTED_BY]->(doc:CodeNode:DocChunk)
            WHERE symbol.snapshot_id IN $snapshot_ids
            RETURN DISTINCT {self._node_fields('doc')} ORDER BY doc.id
        """
        with self._driver.session() as session:
            return [_node_from_record(record) for record in session.run(
                query,
                snapshot_ids=[_snapshot_node_id(graph, node.id) for node in impacted],
            )]

    def _nodes_by_snapshot_ids(self, snapshot_ids: list[str], label: str) -> list[Node]:
        query = (
            f"MATCH (node:CodeNode:{label}) WHERE node.snapshot_id IN $snapshot_ids "
            f"RETURN DISTINCT {self._node_fields('node')} ORDER BY node.id"
        )
        with self._driver.session() as session:
            return [_node_from_record(record) for record in session.run(
                query, snapshot_ids=snapshot_ids,
            )]

    @staticmethod
    def _node_fields(variable: str) -> str:
        return (
            f"{variable}.id AS id, {variable}.kind AS kind, {variable}.name AS name, "
            f"{variable}.file AS file, {variable}.start_line AS start_line, "
            f"{variable}.end_line AS end_line, {variable}.docstring AS docstring, "
            f"{variable}.has_dynamic_warning AS has_dynamic_warning, "
            f"{variable}.roles AS roles, {variable}.content AS content, "
            f"{variable}.owner_symbol_id AS owner_symbol_id, "
            f"{variable}.source_type AS source_type"
        )

    @staticmethod
    def _deduplicate_nodes(nodes: list[Node]) -> list[Node]:
        return [node for _, node in sorted({node.id: node for node in nodes}.items())]

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
