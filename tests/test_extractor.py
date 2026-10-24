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
