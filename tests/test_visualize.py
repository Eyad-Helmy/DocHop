import importlib.util
import sys
from pathlib import Path

import pytest


def test_blast_radius_follows_calls_inheritance_and_definitions(tmp_path, monkeypatch):
    nx = pytest.importorskip("networkx")
    pytest.importorskip("matplotlib")

    visualize_path = Path(__file__).parents[1] / "visualize.py"
    monkeypatch.setattr(sys, "argv", [str(visualize_path), str(tmp_path), "--no-image"])
    spec = importlib.util.spec_from_file_location("codearchguard_test_visualize", visualize_path)
    visualize = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(visualize)

    graph = nx.DiGraph()
    graph.add_edge("shop.models.DiscountedItem", "shop.models.Item",
                   kind=visualize.EdgeKind.INHERITS)
    graph.add_edge("shop.models.Item", "shop.models.Item.total",
                   kind=visualize.EdgeKind.DEFINES)
    graph.add_edge("shop.billing.make_item", "shop.models.DiscountedItem",
                   kind=visualize.EdgeKind.CALLS)
    graph.add_edge("shop.billing.checkout", "shop.billing.make_item",
                   kind=visualize.EdgeKind.CALLS)

    assert visualize.compute_blast_radius(graph, "shop.models.Item") == {
        "shop.models.Item",
        "shop.models.Item.total",
        "shop.models.DiscountedItem",
        "shop.billing.make_item",
        "shop.billing.checkout",
    }
