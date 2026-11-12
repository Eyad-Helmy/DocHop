"""
CodeArchGuard -- Graph Visualizer (Milestone 1)
-----------------------------------------------
Usage:
    python visualize.py path/to/repo
    python visualize.py path/to/repo --out my_graph.png
    python visualize.py path/to/repo --blast-from shop.billing.checkout
    python visualize.py path/to/repo --no-image

Outputs an annotated PNG and prints a text call tree.
No Neo4j or Docker required.
"""
from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import sys

# Make sure the project src is importable
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "src"))

from codearchguard.analysis.extractor import extract_graph
from codearchguard.models import EdgeKind, NodeKind
from codearchguard.visualize import (
    NODE_COLORS,
    NODE_SIZES,
    EDGE_STYLES,
    short_label,
    compute_blast_radius,
    build_nx_graph,
    draw_graph,
    print_text_tree,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="visualize",
        description="CodeArchGuard graph visualizer — no Neo4j needed"
    )
    parser.add_argument("repo", help="Path to the Python repository to analyse")
    parser.add_argument("--out", default="codegraph.png",
                        help="Output PNG filename (default: codegraph.png)")
    parser.add_argument("--blast-from", metavar="SYMBOL_ID",
                        help="Highlight blast radius from this symbol "
                             "(e.g. shop.billing.checkout)")
    parser.add_argument("--no-image", action="store_true",
                        help="Skip image generation, print text tree only")
    args = parser.parse_args(argv)

    print(f"\n[*] Analysing repo: {args.repo}")
    cg = extract_graph(args.repo)

    nc = Counter(n.kind.value for n in cg.nodes.values())
    ec = Counter(e.kind.value for e in cg.edges)
    print(f"    Nodes: {dict(nc)}")
    print(f"    Edges: {dict(ec)}")

    print_text_tree(cg)

    if not args.no_image:
        G = build_nx_graph(cg)
        blast = compute_blast_radius(G, args.blast_from) if args.blast_from else set()
        out = Path(args.out)
        print(f"[*] Rendering graph image -> {out.resolve()}")
        draw_graph(G, blast, out, args.blast_from or "")
        print(f"[OK] Saved: {out.resolve()}")

        try:
            os.startfile(str(out.resolve()))
        except Exception:
            pass


if __name__ == "__main__":
    main()
