from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from codearchguard.models import EdgeKind, NodeKind

# ── visual config ─────────────────────────────────────────────────────────────
NODE_COLORS = {
    NodeKind.MODULE:   "#4A90D9",   # blue
    NodeKind.CLASS:    "#E8A838",   # amber
    NodeKind.FUNCTION: "#5BB85D",   # green
}
NODE_SIZES = {
    NodeKind.MODULE:   3200,
    NodeKind.CLASS:    2200,
    NodeKind.FUNCTION: 1400,
}
EDGE_STYLES = {
    EdgeKind.DEFINES:  dict(edge_color="#888888", style="dotted",  width=1.2, arrows=False),
    EdgeKind.IMPORTS:  dict(edge_color="#4A90D9", style="dashed",  width=1.8, arrows=True),
    EdgeKind.CALLS:    dict(edge_color="#E05252", style="solid",   width=2.2, arrows=True),
    EdgeKind.INHERITS: dict(edge_color="#9B59B6", style="solid",   width=2.2, arrows=True),
}


def short_label(node_id: str) -> str:
    parts = node_id.split(".")
    return ".".join(parts[-2:]) if len(parts) > 2 else node_id


def compute_blast_radius(G: nx.DiGraph, start: str) -> set[str]:
    """Find callers, subclasses, and members affected by a changed symbol."""
    affected = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        if data.get("kind") == EdgeKind.CALLS:
            affected.add_edge(v, u)  # callers
        elif data.get("kind") == EdgeKind.INHERITS:
            affected.add_edge(v, u)  # subclasses
        elif data.get("kind") == EdgeKind.DEFINES:
            affected.add_edge(u, v)  # contained classes and functions
    if start not in affected:
        return {start}
    return set(nx.descendants(affected, start)) | {start}


def build_nx_graph(cg):
    G = nx.DiGraph()
    for node in cg.nodes.values():
        G.add_node(node.id, kind=node.kind, file=node.file,
                   start=node.start_line, end=node.end_line,
                   docstring=node.docstring)
    for edge in cg.edges:
        G.add_edge(edge.src, edge.dst, kind=edge.kind, line=edge.line)
    return G


def draw_graph(G: nx.DiGraph, blast: set[str], out: Path, blast_symbol: str):
    fig, ax = plt.subplots(figsize=(20, 13))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#1A1A2E")
    ax.set_title("CodeArchGuard -- Code Graph  (Milestone 1)",
                 color="white", fontsize=16, fontweight="bold", pad=20)

    # Layout
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G, k=3.5, seed=42, iterations=100)

    node_ids = list(G.nodes())

    # Draw nodes
    for nid in node_ids:
        kind   = G.nodes[nid]["kind"]
        color  = NODE_COLORS[kind]
        size   = NODE_SIZES[kind]
        alpha  = 1.0 if (not blast or nid in blast) else 0.2
        nx.draw_networkx_nodes(G, pos, nodelist=[nid],
                               node_color=[color], node_size=[size],
                               alpha=alpha, ax=ax)

    # Draw labels
    for nid in node_ids:
        alpha = 1.0 if (not blast or nid in blast) else 0.2
        nx.draw_networkx_labels(G, pos, labels={nid: short_label(nid)},
                                font_size=7.5, font_color="white",
                                font_weight="bold", alpha=alpha, ax=ax)

    # Draw edges per kind
    def _draw_edges(edgelist, style_dict, alpha):
        if not edgelist:
            return
        # Always force FancyArrowPatches so arrowstyle/arrowsize/connectionstyle work
        nx.draw_networkx_edges(
            G, pos, edgelist=edgelist,
            edge_color=style_dict["edge_color"],
            style=style_dict["style"],
            width=style_dict["width"],
            arrows=True,
            arrowsize=18,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.1",
            alpha=alpha,
            ax=ax,
            min_source_margin=20,
            min_target_margin=20,
        )

    for kind, style in EDGE_STYLES.items():
        edges = [(u, v) for u, v, d in G.edges(data=True) if d["kind"] == kind]
        if not edges:
            continue
        if blast:
            in_blast  = [(u, v) for u, v in edges if u in blast or v in blast]
            out_blast = [(u, v) for u, v in edges if u not in blast and v not in blast]
            _draw_edges(in_blast,  style, 0.9)
            _draw_edges(out_blast, style, 0.08)
        else:
            _draw_edges(edges, style, 0.85)

    # Legend
    legend_nodes = [mpatches.Patch(color=NODE_COLORS[k], label=k.value) for k in NodeKind]
    legend_edges = [
        mpatches.Patch(color=EDGE_STYLES[k]["edge_color"],
                       label=f"{k.value} ({'directed' if EDGE_STYLES[k]['arrows'] else 'undirected'})")
        for k in EdgeKind
    ]
    legend = ax.legend(handles=legend_nodes + legend_edges, loc="upper left",
                       fontsize=9, facecolor="#0F3460", edgecolor="white",
                       labelcolor="white", title="Legend", title_fontsize=10)
    plt.setp(legend.get_title(), color="white")

    # Stats box
    nc = Counter(G.nodes[n]["kind"].value for n in G.nodes())
    ec = Counter(d["kind"].value for _, _, d in G.edges(data=True))
    stats = "Nodes: " + "  ".join(f"{k}={v}" for k, v in nc.items())
    stats += "\nEdges: " + "  ".join(f"{k}={v}" for k, v in ec.items())
    if blast and blast_symbol:
        stats += f"\n\nBlast radius from:\n{blast_symbol}\n({len(blast)} nodes highlighted)"
    ax.text(0.01, 0.01, stats, transform=ax.transAxes, fontsize=8,
            color="#CCCCCC", verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#0F3460", alpha=0.8))

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


def print_text_tree(cg):
    callers  = defaultdict(list)
    callees  = defaultdict(list)
    for e in cg.edges:
        if e.kind == EdgeKind.CALLS:
            callers[e.dst].append(e.src)
            callees[e.src].append(e.dst)

    print("\n" + "=" * 60)
    print("  CALL GRAPH")
    print("=" * 60)
    for nid, node in sorted(cg.nodes.items()):
        if node.kind != NodeKind.FUNCTION:
            continue
        print(f"\n  [FUNC] {nid}")
        print(f"         file : {node.file}  (lines {node.start_line}-{node.end_line})")
        if node.docstring:
            print(f"         doc  : {node.docstring[:100]}")
        if callers.get(nid):
            print(f"         <- called by : {', '.join(callers[nid])}")
        if callees.get(nid):
            print(f"         -> calls     : {', '.join(callees[nid])}")

    print("\n" + "=" * 60)
    print("  IMPORTS")
    print("=" * 60)
    for e in sorted(cg.edges, key=lambda x: x.src):
        if e.kind == EdgeKind.IMPORTS:
            print(f"  {e.src}  ->  {e.dst}  (line {e.line})")

    print("\n" + "=" * 60)
    print("  INHERITANCE")
    print("=" * 60)
    for e in sorted(cg.edges, key=lambda x: x.src):
        if e.kind == EdgeKind.INHERITS:
            print(f"  {e.src}  ->  {e.dst}  (line {e.line})")
    print()
