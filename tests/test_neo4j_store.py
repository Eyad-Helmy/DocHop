from codearchguard.graph.neo4j_store import Neo4jStore, _snapshot_node_id
from codearchguard.models import CodeGraph, Edge, EdgeKind, Node, NodeKind, NodeRole


class FakeSession:
    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def run(self, query, **parameters):
        self.calls.append((query, parameters))


class FakeDriver:
    def __init__(self):
        self.session_instance = FakeSession()

    def session(self):
        return self.session_instance


class QuerySession(FakeSession):
    def __init__(self, responder):
        super().__init__()
        self.responder = responder

    def run(self, query, **parameters):
        super().run(query, **parameters)
        return self.responder(query, parameters)


def store_with_fake_driver():
    store = Neo4jStore.__new__(Neo4jStore)
    store._driver = FakeDriver()
    return store


def test_init_schema_migrates_legacy_global_ids_without_deleting_nodes():
    store = store_with_fake_driver()

    store.init_schema()

    queries = [query for query, _ in store._driver.session_instance.calls]
    assert "DROP CONSTRAINT code_id IF EXISTS" in queries
    assert any("SET n.snapshot_id = 'legacy:' + n.id" in query for query in queries)
    assert any("code_snapshot_id" in query for query in queries)
    assert not any("DETACH DELETE" in query for query in queries)


def test_save_persists_snapshot_scoped_docchunks_roles_and_edges():
    graph = CodeGraph(repository_root="C:/repo", commit_hash="abc123", is_dirty=False)
    graph.add_node(Node(
        "pkg.work", NodeKind.FUNCTION, "work", "pkg.py",
        has_dynamic_warning=True, roles=frozenset({NodeRole.TEST}),
    ))
    graph.add_node(Node(
        "docchunk:doc_pkg.work_0", NodeKind.DOC_CHUNK, "doc_pkg.work_0", "docs.md",
        content="Explains work.", owner_symbol_id="pkg.work", source_type="markdown",
    ))
    graph.add_edge(Edge("pkg.work", "docchunk:doc_pkg.work_0", EdgeKind.DOCUMENTED_BY, 4))
    store = store_with_fake_driver()

    store.save(graph)

    calls = store._driver.session_instance.calls
    queries = [query for query, _ in calls]
    assert any(":DocChunk" in query for query in queries)
    assert any("SET n:Test" in query for query in queries)
    assert any("[e:DOCUMENTED_BY]" in query for query in queries)
    node_rows = next(params["rows"] for query, params in calls if ":Function" in query)
    assert node_rows[0]["has_dynamic_warning"] is True
    assert node_rows[0]["commit_hash"] == "abc123"
    assert node_rows[0]["snapshot_id"] == _snapshot_node_id(graph, "pkg.work")


def test_blast_radius_expands_frontier_exhaustively_beyond_three_hops():
    graph = CodeGraph(repository_root="C:/repo", commit_hash="abc123")
    store = store_with_fake_driver()

    def make_row(node_id):
        return {
            "id": node_id, "kind": "Function", "name": node_id.split(".")[-1], "file": "pkg.py",
            "start_line": 1, "end_line": 2, "docstring": None, "has_dynamic_warning": False,
            "roles": [], "content": None, "owner_symbol_id": None, "source_type": None,
        }

    chain = {
        _snapshot_node_id(graph, "pkg.h0"): [make_row("pkg.h1")],
        _snapshot_node_id(graph, "pkg.h1"): [make_row("pkg.h2")],
        _snapshot_node_id(graph, "pkg.h2"): [make_row("pkg.h3")],
        _snapshot_node_id(graph, "pkg.h3"): [make_row("pkg.h4")],
        _snapshot_node_id(graph, "pkg.h4"): [],
    }

    def responder(query, parameters):
        if "start_snapshot_id" in parameters:
            return [make_row("pkg.h0")]
        if "CALLS|INHERITS|DEFINES" in query:
            frontier = parameters["frontier_snapshot_ids"]
            return [row for src in frontier for row in chain.get(src, [])]
        return []

    store._driver.session_instance = QuerySession(responder)

    nodes = store.blast_radius(graph, "pkg.h0", include_containers=False)

    assert [node.id for node in nodes] == ["pkg.h0", "pkg.h1", "pkg.h2", "pkg.h3", "pkg.h4"]


def test_blast_radius_respects_explicit_hop_and_node_limits():
    graph = CodeGraph(repository_root="C:/repo", commit_hash="abc123")
    store = store_with_fake_driver()

    def make_row(node_id):
        return {
            "id": node_id, "kind": "Function", "name": node_id.split(".")[-1], "file": "pkg.py",
            "start_line": 1, "end_line": 2, "docstring": None, "has_dynamic_warning": False,
            "roles": [], "content": None, "owner_symbol_id": None, "source_type": None,
        }

    def responder(query, parameters):
        if "start_snapshot_id" in parameters:
            return [make_row("pkg.h0")]
        if "CALLS|INHERITS|DEFINES" in query:
            frontier = parameters["frontier_snapshot_ids"]
            if frontier == [_snapshot_node_id(graph, "pkg.h0")]:
                return [make_row("pkg.h1"), make_row("pkg.h2")]
            if _snapshot_node_id(graph, "pkg.h1") in frontier:
                return [make_row("pkg.h3")]
        return []

    store._driver.session_instance = QuerySession(responder)

    hop_limited = store.blast_radius(graph, "pkg.h0", max_hops=1, include_containers=False)
    node_limited = store.blast_radius(graph, "pkg.h0", max_nodes=2, include_containers=False)

    assert [node.id for node in hop_limited] == ["pkg.h0", "pkg.h1", "pkg.h2"]
    assert [node.id for node in node_limited] == ["pkg.h0", "pkg.h1"]


def test_blast_radius_rejects_invalid_hop_and_node_limits():
    store = store_with_fake_driver()
    graph = CodeGraph(repository_root="C:/repo", commit_hash="abc123")

    for invalid in (-1, "3", True):
        try:
            store.blast_radius(graph, "pkg.changed", max_hops=invalid)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f"Expected invalid hop count {invalid!r} to fail")

    for invalid in (0, -2, "5", False):
        try:
            store.blast_radius(graph, "pkg.changed", max_nodes=invalid)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f"Expected invalid max_nodes {invalid!r} to fail")


def test_impacted_test_and_doc_queries_use_the_blast_radius_snapshot_only():
    graph = CodeGraph(repository_root="C:/repo", commit_hash="abc123")
    store = store_with_fake_driver()
    impacted = {
        "id": "pkg.changed", "kind": "Function", "name": "changed", "file": "pkg.py",
        "start_line": 1, "end_line": 2, "docstring": "", "has_dynamic_warning": False,
        "roles": [], "content": "", "owner_symbol_id": "", "source_type": "",
    }
    test = {**impacted, "id": "tests.test_pkg.test_changed", "name": "test_changed", "roles": ["Test"]}
    doc = {**impacted, "id": "docchunk:doc_pkg.changed_0", "kind": "DocChunk", "content": "Explains change."}

    def responder(query, parameters):
        if "start_snapshot_id" in parameters:
            return [impacted]
        if ":Test)" in query:
            return [test]
        if "[:DOCUMENTED_BY]" in query:
            return [doc]
        return []

    store._driver.session_instance = QuerySession(responder)

    assert [node.id for node in store.impacted_tests(graph, "pkg.changed")] == ["tests.test_pkg.test_changed"]
    assert [node.id for node in store.related_doc_chunks(graph, "pkg.changed")] == ["docchunk:doc_pkg.changed_0"]
    query_text = "\n".join(query for query, _ in store._driver.session_instance.calls)
    assert "MATCH (node:CodeNode:Test)" in query_text
    assert "[:DOCUMENTED_BY]->(doc:CodeNode:DocChunk)" in query_text
