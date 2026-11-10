from pathlib import Path
from codearchguard.analysis.extractor import extract_graph
from codearchguard.models import EdgeKind

REPO = Path(__file__).parent / "fixtures" / "sample_repo"


def pairs(g, kind):
    return {(e.src, e.dst) for e in g.edges_of(kind)}


def test_nodes_found():
    g = extract_graph(REPO)
    for nid in ["shop.billing", "shop.models.Item", "shop.billing.checkout", "shop.models.DiscountedItem.total"]:
        assert nid in g.nodes


def test_defines_and_inherits():
    g = extract_graph(REPO)
    assert ("shop.models", "shop.models.Item") in pairs(g, EdgeKind.DEFINES)
    assert ("shop.models.DiscountedItem", "shop.models.Item") in pairs(g, EdgeKind.INHERITS)


def test_imports():
    g = extract_graph(REPO)
    assert ("shop.billing", "shop.models") in pairs(g, EdgeKind.IMPORTS)


def test_calls():
    g = extract_graph(REPO)
    c = pairs(g, EdgeKind.CALLS)
    assert ("shop.billing.checkout", "shop.billing.make_item") in c
    assert ("shop.billing.checkout", "shop.billing.helper") in c
    assert ("shop.billing.make_item", "shop.models.DiscountedItem") in c


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
    assert ("app.billing.make_item", "app.models.Item") in pairs(graph, EdgeKind.CALLS)


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
    (tmp_path / "setup.cfg").write_text("[options]\npackage_dir =\n    = backend\n")
    (package / "__init__.py").write_text("")

    graph = extract_graph(tmp_path)

    assert "app" in graph.nodes
    assert "backend.app" not in graph.nodes


def test_setup_py_source_root_supports_nonstandard_container(tmp_path):
    package = tmp_path / "backend" / "app"
    package.mkdir(parents=True)
    (tmp_path / "setup.py").write_text('setup(package_dir={"": "backend"})\n')
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
