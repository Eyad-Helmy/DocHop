import sys
import types

from codearchguard import cli


def test_cli_analyze_prints_snapshot_roles_warnings_and_docs(tmp_path, capsys):
    (tmp_path / "app.py").write_text(
        '"""App module."""\n\n'
        "def run():\n"
        '    """Run app."""\n'
        "    return eval('1')\n\n"
        "if __name__ == '__main__':\n"
        "    run()\n",
        encoding="utf-8",
    )
    (tmp_path / "test_app.py").write_text("from app import run\n\ndef test_run():\n    run()\n", encoding="utf-8")

    cli.main(["analyze", str(tmp_path)])
    out = capsys.readouterr().out

    assert "Snapshot:" in out
    assert "'DocChunk':" in out
    assert "'Test':" in out
    assert "'EntryPoint':" in out
    assert "Dynamic warnings: 1" in out
    assert "'DOCUMENTED_BY':" in out


def test_cli_impact_in_memory_prints_impacted_symbols_tests_and_docs(tmp_path, capsys):
    (tmp_path / "app.py").write_text(
        "def helper():\n"
        '    """Help."""\n'
        "    return 1\n",
        encoding="utf-8",
    )
    (tmp_path / "test_app.py").write_text(
        "from app import helper\n\n"
        "def test_helper():\n"
        "    assert helper() == 1\n",
        encoding="utf-8",
    )

    cli.main(["impact", str(tmp_path), "app.helper"])
    out = capsys.readouterr().out

    assert "Impact for app.helper (In-Memory):" in out
    assert "test_app [Test]" in out
    assert "test_app.test_helper [Test]" in out
    assert "Impacted tests (2):" in out
    assert "docchunk:doc_app.helper_chunk_0 [docstring]" in out


def test_cli_impact_can_query_neo4j(tmp_path, monkeypatch, capsys):
    (tmp_path / "app.py").write_text("def helper(): return 1\n", encoding="utf-8")
    calls = []

    class FakeStore:
        def __init__(self, uri, user, password):
            calls.append(("init", uri, user, password))

        def blast_radius(self, _graph, symbol, max_hops=None, include_containers=True, max_nodes=None):
            calls.append(("blast", symbol, max_hops, include_containers, max_nodes))
            return []

        def impacted_tests(self, _graph, symbol, max_hops=None):
            calls.append(("tests", symbol, max_hops))
            return []

        def related_doc_chunks(self, _graph, symbol, max_hops=None):
            calls.append(("docs", symbol, max_hops))
            return []

        def close(self):
            calls.append(("close",))

    fake_mod = types.ModuleType("codearchguard.graph.neo4j_store")
    fake_mod.Neo4jStore = FakeStore
    monkeypatch.setitem(sys.modules, "codearchguard.graph.neo4j_store", fake_mod)
    monkeypatch.setattr(cli, "extract_graph", lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("should not re-parse repo")))

    cli.main([
        "impact", str(tmp_path), "app.helper",
        "--neo4j-uri", "bolt://localhost:7687",
        "--max-hops", "4",
        "--max-nodes", "50",
    ])
    out = capsys.readouterr().out

    assert "Impact for app.helper (Neo4j):" in out
    assert ("blast", "app.helper", 4, True, 50) in calls
    assert ("tests", "app.helper", 4) in calls
    assert ("docs", "app.helper", 4) in calls
    assert ("close",) in calls
