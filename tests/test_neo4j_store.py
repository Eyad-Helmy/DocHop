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
