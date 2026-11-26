
from pathlib import Path

from codearchguard import analysis
from codearchguard.analysis.extractor import extract_graph
from codearchguard.models import EdgeKind, NodeRole


REPO = Path(__file__).parent / "fixtures" / "sample_repo"


def pairs(g, kind):
    return {(e.src, e.dst) for e in g.edges_of(kind)}


def test_nodes_found():
    g = extract_graph(REPO)

    for nid in [
        "shop.billing",
        "shop.models.Item",
        "shop.billing.checkout",
        "shop.models.DiscountedItem.total",
    ]:
        assert nid in g.nodes


def test_defines_and_inherits():
    g = extract_graph(REPO)

    assert ("shop.models", "shop.models.Item") in pairs(g, EdgeKind.DEFINES)
    assert ("shop.models.DiscountedItem", "shop.models.Item") in pairs(
        g, EdgeKind.INHERITS
    )


def test_imports():
    g = extract_graph(REPO)

    assert ("shop.billing", "shop.models") in pairs(g, EdgeKind.IMPORTS)


def test_calls():
    g = extract_graph(REPO)
    c = pairs(g, EdgeKind.CALLS)

    assert ("shop.billing.checkout", "shop.billing.make_item") in c
    assert ("shop.billing.checkout", "shop.billing.helper") in c

    # Constructor calls should point to __init__, not the class node.
    assert ("shop.billing.make_item", "shop.models.Item.__init__") in c
    assert ("shop.billing.make_item", "shop.models.DiscountedItem") not in c


def test_repo_root_src_layout_uses_importable_module_names(tmp_path):
    package = tmp_path / "src" / "app"
    package.mkdir(parents=True)

    (package / "__init__.py").write_text("")
    (package / "models.py").write_text("class Item:\n    pass\n")
    (package / "billing.py").write_text(
        "from app.models import Item\n\n"
        "def make_item():\n"
        "    return Item()\n"
    )

    graph = extract_graph(tmp_path)

    assert "app.models" in graph.nodes
    assert "src.app.models" not in graph.nodes
    assert ("app.billing", "app.models") in pairs(graph, EdgeKind.IMPORTS)
    assert ("app.billing.make_item", "app.models.Item") in pairs(
        graph, EdgeKind.CALLS
    )


def test_explicit_source_root_supports_nonstandard_container(tmp_path):
    package = tmp_path / "backend" / "app"
    package.mkdir(parents=True)

    (package / "__init__.py").write_text("")
    (package / "models.py").write_text("class Item:\n    pass\n")
    (package / "billing.py").write_text(
        "from app.models import Item\n\n"
        "def make_item():\n"
        "    return Item()\n"
    )

    graph = extract_graph(tmp_path, source_roots=["backend"])

    assert "app.models" in graph.nodes
    assert "backend.app.models" not in graph.nodes
    assert ("app.billing", "app.models") in pairs(graph, EdgeKind.IMPORTS)


def test_pyproject_source_root_supports_nonstandard_container(tmp_path):
    package = tmp_path / "backend" / "app"
    package.mkdir(parents=True)

    (tmp_path / "pyproject.toml").write_text(
        "[tool.setuptools.packages.find]\nwhere = [\"backend\"]\n"
    )
    (package / "__init__.py").write_text("")

    graph = extract_graph(tmp_path)

    assert "app" in graph.nodes
    assert "backend.app" not in graph.nodes


def test_setup_cfg_source_root_supports_nonstandard_container(tmp_path):
    package = tmp_path / "backend" / "app"
    package.mkdir(parents=True)

    (tmp_path / "setup.cfg").write_text(
        "[options]\npackage_dir =\n    = backend\n"
    )
    (package / "__init__.py").write_text("")

    graph = extract_graph(tmp_path)

    assert "app" in graph.nodes
    assert "backend.app" not in graph.nodes


def test_setup_py_source_root_supports_nonstandard_container(tmp_path):
    package = tmp_path / "backend" / "app"
    package.mkdir(parents=True)

    (tmp_path / "setup.py").write_text(
        'setup(package_dir={"": "backend"})\n'
    )
    (package / "__init__.py").write_text("")

    graph = extract_graph(tmp_path)

    assert "app" in graph.nodes
    assert "backend.app" not in graph.nodes


def test_unknown_container_is_not_stripped(tmp_path):
    package = tmp_path / "backend" / "app"
    package.mkdir(parents=True)

    (package / "__init__.py").write_text("")

    graph = extract_graph(tmp_path)

    assert "backend.app" in graph.nodes


def test_local_instance_method_calls_are_resolved(tmp_path):
    (tmp_path / "store.py").write_text(
        "class Neo4jStore:\n"
        "    def __init__(self): pass\n"
        "    def init_schema(self): pass\n"
        "    def save(self, graph): pass\n"
        "    def close(self): pass\n"
    )

    (tmp_path / "cli.py").write_text(
        "from store import Neo4jStore\n\n"
        "def main():\n"
        "    store = Neo4jStore()\n"
        "    store.init_schema()\n"
        "    store.save(None)\n"
        "    store.close()\n"
    )

    graph = extract_graph(tmp_path)
    calls = pairs(graph, EdgeKind.CALLS)

    assert ("cli.main", "store.Neo4jStore.init_schema") in calls
    assert ("cli.main", "store.Neo4jStore.save") in calls
    assert ("cli.main", "store.Neo4jStore.close") in calls


def test_module_level_calls_are_traced(tmp_path):
    (tmp_path / "helpers.py").write_text("def work(): pass\n")
    (tmp_path / "entry.py").write_text(
        "from helpers import work\n\n"
        "work()\n"
    )

    graph = extract_graph(tmp_path)

    assert ("entry", "helpers.work") in pairs(graph, EdgeKind.CALLS)


def test_snapshot_metadata_is_attached_without_changing_graph_extraction(tmp_path, monkeypatch):
    (tmp_path / "module.py").write_text("def work(): pass\n")
    monkeypatch.setattr(
        analysis.extractor,
        "_git_snapshot",
        lambda root: ("a" * 40, True),
    )

    graph = extract_graph(tmp_path)

    assert graph.repository_root == str(tmp_path.resolve())
    assert graph.commit_hash == "a" * 40
    assert graph.is_dirty is True
    assert "module.work" in graph.nodes


def test_non_git_snapshot_is_explicitly_unknown(tmp_path, monkeypatch):
    (tmp_path / "module.py").write_text("")
    monkeypatch.setattr(analysis.extractor, "_git_snapshot", lambda root: (None, None))

    graph = extract_graph(tmp_path)

    assert graph.commit_hash is None
    assert graph.is_dirty is None


def test_dynamic_construct_warnings_are_scoped_to_the_enclosing_symbol(tmp_path):
    (tmp_path / "dynamic.py").write_text(
        "import importlib as il\n"
        "from importlib import import_module as load_module\n\n"
        "def uses_eval():\n    return eval('1 + 1')\n\n"
        "def uses_exec():\n    exec('value = 1')\n\n"
        "def uses_runtime_import():\n    return il.import_module('pkg')\n\n"
        "def uses_import_alias():\n    return load_module('pkg')\n\n"
        "def uses_dynamic_getattr(obj, name):\n    return getattr(obj, name)\n\n"
        "def uses_literal_getattr(obj):\n    return getattr(obj, 'name')\n\n"
        "class Holder:\n"
        "    generated = __import__('generated')\n"
        "    def ordinary(self):\n        return 1\n"
    )

    graph = extract_graph(tmp_path)

    for symbol in (
        "dynamic.uses_eval",
        "dynamic.uses_exec",
        "dynamic.uses_runtime_import",
        "dynamic.uses_import_alias",
        "dynamic.uses_dynamic_getattr",
        "dynamic.Holder",
    ):
        assert graph.nodes[symbol].has_dynamic_warning
    assert not graph.nodes["dynamic.uses_literal_getattr"].has_dynamic_warning
    assert not graph.nodes["dynamic.Holder.ordinary"].has_dynamic_warning
    assert not graph.nodes["dynamic"].has_dynamic_warning


def test_pytest_and_main_guard_roles_preserve_structural_node_kinds(tmp_path):
    (tmp_path / "helpers.py").write_text("def work(): pass\n")
    (tmp_path / "test_helpers.py").write_text(
        "def test_function(): pass\n\n"
        "class TestFeature:\n"
        "    def test_case(self): pass\n"
    )
    (tmp_path / "entry.py").write_text(
        "from helpers import work\n\n"
        "def main():\n    work()\n\n"
        "if __name__ == '__main__':\n    main()\n"
    )

    graph = extract_graph(tmp_path)

    assert NodeRole.TEST in graph.nodes["test_helpers"].roles
    assert NodeRole.TEST in graph.nodes["test_helpers.test_function"].roles
    assert NodeRole.TEST in graph.nodes["test_helpers.TestFeature"].roles
    assert NodeRole.TEST in graph.nodes["test_helpers.TestFeature.test_case"].roles
    assert NodeRole.ENTRY_POINT in graph.nodes["entry"].roles
    assert NodeRole.ENTRY_POINT in graph.nodes["entry.main"].roles
    assert graph.nodes["test_helpers.test_function"].kind.value == "Function"
    assert graph.nodes["entry.main"].kind.value == "Function"
