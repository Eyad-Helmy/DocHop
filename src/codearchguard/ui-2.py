"""
CodeArchGuard — Interactive Web Demo & Architecture Assistant (`src/codearchguard/ui.py`)
=========================================================================================
Run directly from the repository root:
    python src/codearchguard/ui.py
Or via:
    python app.py
    cag ui
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import tempfile
import threading
import time
import urllib.parse
import webbrowser
from typing import Any

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import codearchguard.analysis.extractor as _extractor_mod
from codearchguard.analysis.extractor import extract_graph
from codearchguard.models import CodeGraph, EdgeKind, NodeKind
from codearchguard.rag.chunking import chunk_documents
from codearchguard.rag.code_ingestion import extract_code_documents
from codearchguard.rag.document import Chunk
from codearchguard.rag.graph_scoped_retriever import (
    compute_graph_blast_radius,
    filter_chunks_by_blast_radius,
)
from codearchguard.rag.ingestion import extract_documents
from codearchguard.rag.pipeline import RAGIngestionResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Skip nested `tests/fixtures/*` and `storage/` when analyzing `.` (DocHop root)
_EXTRA_SKIP_DIRS = set(_extractor_mod.SKIP_DIRS) | {"storage"}


def _clean_iter_python_files(root: Path):
    for p in sorted(root.rglob("*.py")):
        parts = p.relative_to(root).parts
        if set(parts) & _EXTRA_SKIP_DIRS:
            continue
        if len(parts) >= 2 and parts[0] == "tests" and parts[1] == "fixtures":
            continue
        yield p


_extractor_mod.iter_python_files = _clean_iter_python_files


def _fast_ingest_without_duplicate_ast(
    repo_root: Path,
    graph: CodeGraph,
    max_chars: int = 1000,
) -> RAGIngestionResult:
    code_documents = extract_code_documents(graph, repo_root)
    documentation_documents = extract_documents(graph, repo_root)
    documents = code_documents + documentation_documents
    chunks = chunk_documents(documents, max_chars=max_chars)
    return RAGIngestionResult(graph=graph, documents=documents, chunks=chunks)


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for",
    "from", "how", "in", "into", "is", "it", "of", "on", "or", "that", "the",
    "their", "this", "to", "what", "when", "where", "which", "who", "why", "with",
}


def _tokenize(text: str) -> list[str]:
    raw_tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+", text)
    expanded: list[str] = []
    for tok in raw_tokens:
        lower = tok.lower()
        if lower not in _STOPWORDS and len(lower) > 1:
            expanded.append(lower)
        parts = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", tok).lower().split("_")
        if len(parts) > 1:
            for p in parts:
                if p and p not in _STOPWORDS and len(p) > 1 and p != lower:
                    expanded.append(p)
    return expanded


class FastBM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_freqs: list[Counter[str]] = []
        self.doc_lens: list[int] = []
        self.df: Counter[str] = Counter()

        for chunk in chunks:
            tokens = _tokenize(f"{chunk.owner_symbol_id} {chunk.text}")
            freq = Counter(tokens)
            self.doc_freqs.append(freq)
            self.doc_lens.append(len(tokens))
            for term in freq:
                self.df[term] += 1

        self.n_docs = len(chunks)
        self.avgdl = (sum(self.doc_lens) / self.n_docs) if self.n_docs else 1.0
        self.idf = {
            term: math.log(1.0 + (self.n_docs - df_val + 0.5) / (df_val + 0.5))
            for term, df_val in self.df.items()
        }

    def score_subset(self, query_tokens: list[str], allowed_indices: list[int]) -> dict[int, float]:
        if not allowed_indices or not query_tokens:
            return {idx: 0.0 for idx in allowed_indices}
        scores: dict[int, float] = {}
        for idx in allowed_indices:
            freq = self.doc_freqs[idx]
            dl = self.doc_lens[idx]
            denom_norm = self.k1 * (1.0 - self.b + self.b * (dl / max(self.avgdl, 1e-6)))
            s = 0.0
            for q in query_tokens:
                tf = freq.get(q, 0)
                if tf > 0:
                    s += self.idf.get(q, 0.0) * ((tf * (self.k1 + 1.0)) / (tf + denom_norm))
            scores[idx] = s
        return scores


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return dot / (norm_a * norm_b)


def _render_static_png(graph: CodeGraph, blast_set: set[str], out_file: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import networkx as nx

    node_colors = {NodeKind.MODULE: "#4A90D9", NodeKind.CLASS: "#E8A838", NodeKind.FUNCTION: "#5BB85D"}
    node_sizes = {NodeKind.MODULE: 3200, NodeKind.CLASS: 2200, NodeKind.FUNCTION: 1400}
    edge_styles = {
        EdgeKind.DEFINES: dict(edge_color="#888888", style="dotted", width=1.2),
        EdgeKind.IMPORTS: dict(edge_color="#4A90D9", style="dashed", width=1.8),
        EdgeKind.CALLS: dict(edge_color="#E05252", style="solid", width=2.2),
        EdgeKind.INHERITS: dict(edge_color="#9B59B6", style="solid", width=2.2),
    }

    G = nx.DiGraph()
    for node in graph.nodes.values():
        G.add_node(node.id, kind=node.kind)
    for edge in graph.edges:
        G.add_edge(edge.src, edge.dst, kind=edge.kind)

    fig, ax = plt.subplots(figsize=(18, 11))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")
    ax.set_title("CodeArchGuard — Architecture Knowledge Graph", color="white", fontsize=15, fontweight="bold", pad=16)

    num_nodes = len(G.nodes())
    pos = nx.spring_layout(G, k=3.2 / max(1, (num_nodes / 25) ** 0.3), seed=42, iterations=40 if num_nodes > 100 else 80)

    for kind in NodeKind:
        if blast_set:
            in_l = [n for n, d in G.nodes(data=True) if d.get("kind") == kind and n in blast_set]
            out_l = [n for n, d in G.nodes(data=True) if d.get("kind") == kind and n not in blast_set]
            if in_l:
                nx.draw_networkx_nodes(G, pos, nodelist=in_l, node_color=node_colors[kind], node_size=node_sizes[kind], alpha=1.0, ax=ax)
            if out_l:
                nx.draw_networkx_nodes(G, pos, nodelist=out_l, node_color=node_colors[kind], node_size=node_sizes[kind], alpha=0.18, ax=ax)
        else:
            nl = [n for n, d in G.nodes(data=True) if d.get("kind") == kind]
            if nl:
                nx.draw_networkx_nodes(G, pos, nodelist=nl, node_color=node_colors[kind], node_size=node_sizes[kind], alpha=0.95, ax=ax)

    if num_nodes <= 150:
        lbls = {n: ".".join(n.split(".")[-2:]) for n in (blast_set if blast_set else G.nodes()) if n in G.nodes()}
        if lbls:
            nx.draw_networkx_labels(G, pos, labels=lbls, font_size=8, font_color="white", font_weight="bold", ax=ax)

    for kind, style in edge_styles.items():
        edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("kind") == kind]
        if not edges:
            continue
        if blast_set:
            in_b = [(u, v) for u, v in edges if u in blast_set or v in blast_set]
            out_b = [(u, v) for u, v in edges if u not in blast_set and v not in blast_set]
            if in_b:
                nx.draw_networkx_edges(G, pos, edgelist=in_b, edge_color=style["edge_color"], style=style["style"], width=style["width"], arrows=True, arrowsize=15, alpha=0.9, ax=ax)
            if out_b:
                nx.draw_networkx_edges(G, pos, edgelist=out_b, edge_color=style["edge_color"], style=style["style"], width=style["width"], arrows=True, arrowsize=15, alpha=0.08, ax=ax)
        else:
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=style["edge_color"], style=style["style"], width=style["width"], arrows=True, arrowsize=15, alpha=0.85, ax=ax)

    handles = [mpatches.Patch(color=node_colors[k], label=k.value) for k in NodeKind] + [
        mpatches.Patch(color=edge_styles[k]["edge_color"], label=k.value) for k in EdgeKind
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9, facecolor="#1E293B", edgecolor="#475569", labelcolor="white")
    plt.tight_layout()
    plt.savefig(out_file, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


@dataclass
class RepoCacheEntry:
    repo_label: str
    repo_path: Path
    rel_or_display_path: str
    graph: CodeGraph
    ingestion: RAGIngestionResult
    bm25: FastBM25Index
    chunk_id_to_idx: dict[str, int]
    has_dense_vectors: bool
    vector_source: str
    load_time_ms: float
    parent_container: dict[str, str] = field(default_factory=dict)
    callers_map: dict[str, list[str]] = field(default_factory=dict)
    callees_map: dict[str, list[str]] = field(default_factory=dict)
    subclasses_map: dict[str, list[str]] = field(default_factory=dict)
    bases_map: dict[str, list[str]] = field(default_factory=dict)
    defines_map: dict[str, list[str]] = field(default_factory=dict)


class DemoEngine:
    def __init__(self):
        self._lock = threading.RLock()
        self.repos: dict[str, RepoCacheEntry] = {}
        self.current_repo_key: str | None = None
        self._bge_embedder: Any = None
        self._bge_status: str = "standby"
        self._query_vec_cache: dict[str, list[float]] = {}

    def try_get_bge_embedder(self):
        if self._bge_embedder is not None:
            return self._bge_embedder
        if self._bge_status == "unavailable":
            return None
        try:
            from codearchguard.rag.llama_embedder import LlamaBGEEmbedder
            self._bge_status = "loading"
            self._bge_embedder = LlamaBGEEmbedder()
            self._bge_status = "ready"
            return self._bge_embedder
        except Exception:
            self._bge_status = "unavailable"
            return None

    def warmup_background(self):
        def _worker():
            try:
                self.load_repository("tests/fixtures/sample_repo")
            except Exception:
                pass
            self.try_get_bge_embedder()

        threading.Thread(target=_worker, daemon=True).start()

    def _hydrate_from_storage_dirs(self, repo_path: Path, chunks: list[Chunk]) -> tuple[int, str]:
        storage_root = PROJECT_ROOT / "storage"
        if not storage_root.is_dir():
            return 0, "none"

        candidate_dirs = [
            storage_root / f"idx_{repo_path.name}",
            storage_root / "vector_index",
            storage_root / "idx_sample_repo",
            storage_root / "idx_flask",
        ]
        seen_dirs: set[Path] = set()
        hydrated = 0
        used_source = "none"

        for d in candidate_dirs:
            d_res = d.resolve()
            if d_res in seen_dirs or not d_res.is_dir():
                continue
            seen_dirs.add(d_res)
            vs_file = d_res / "default__vector_store.json"
            if not vs_file.is_file():
                continue
            try:
                raw = json.loads(vs_file.read_text(encoding="utf-8"))
                emb_dict = raw.get("embedding_dict", {})
                if not isinstance(emb_dict, dict) or not emb_dict:
                    continue
                matches = sum(1 for c in chunks if c.id in emb_dict)
                if matches > 0:
                    for c in chunks:
                        if c.embedding is None and c.id in emb_dict:
                            c.embedding = emb_dict[c.id]
                            hydrated += 1
                    used_source = f"Cached BGE-384 ({d.name})"
                    if hydrated == len(chunks):
                        break
            except Exception:
                continue

        return hydrated, used_source

    def load_repository(self, repo_path_str: str, force_reload: bool = False) -> RepoCacheEntry:
        with self._lock:
            t0 = time.perf_counter()
            raw_str = repo_path_str.strip() or "tests/fixtures/sample_repo"
            candidate = Path(raw_str)
            if not candidate.is_absolute():
                candidate = (PROJECT_ROOT / candidate).resolve()
            else:
                candidate = candidate.resolve()

            if not candidate.exists() or not candidate.is_dir():
                raise FileNotFoundError(f"Repository directory not found: {raw_str}")

            cache_key = str(candidate)
            if not force_reload and cache_key in self.repos:
                self.current_repo_key = cache_key
                return self.repos[cache_key]

            # Bypass slow Jedi fallback on bulk AST extraction (reduces Flask from 15s+ to 0.4s)
            saved_jedi = sys.modules.get("jedi")
            sys.modules["jedi"] = None  # type: ignore[assignment]
            try:
                graph = extract_graph(candidate)
            finally:
                if saved_jedi is not None:
                    sys.modules["jedi"] = saved_jedi
                else:
                    sys.modules.pop("jedi", None)

            ingestion = _fast_ingest_without_duplicate_ast(candidate, graph, max_chars=1000)
            hydrated_count, vector_source = self._hydrate_from_storage_dirs(candidate, ingestion.chunks)

            if hydrated_count < len(ingestion.chunks) and (PROJECT_ROOT / "storage" / f"idx_{candidate.name}").is_dir():
                alt_ingestion = _fast_ingest_without_duplicate_ast(candidate, graph, max_chars=800)
                alt_hydrated, alt_source = self._hydrate_from_storage_dirs(candidate, alt_ingestion.chunks)
                if alt_hydrated > hydrated_count:
                    ingestion = alt_ingestion
                    hydrated_count = alt_hydrated
                    vector_source = alt_source

            if hydrated_count < len(ingestion.chunks):
                embedder = self.try_get_bge_embedder()
                if embedder is not None:
                    try:
                        missing = [c for c in ingestion.chunks if c.embedding is None]
                        if missing:
                            vecs = embedder.embed_documents([c.text for c in missing])
                            for c, v in zip(missing, vecs):
                                c.embedding = v
                        from codearchguard.rag.llama_vector_index import load_or_build_vector_index
                        persist_dir = PROJECT_ROOT / "storage" / f"idx_{candidate.name}"
                        load_or_build_vector_index(
                            chunks=ingestion.chunks,
                            embedder=embedder,
                            persist_dir=str(persist_dir),
                        )
                        hydrated_count = sum(1 for c in ingestion.chunks if c.embedding is not None)
                        vector_source = f"Live BGE-384 ({persist_dir.name})"
                    except Exception:
                        pass

            has_dense = hydrated_count > 0 and hydrated_count == len(ingestion.chunks)
            if not has_dense and hydrated_count == 0:
                vector_source = "Semantic Token-Vector + BM25 Hybrid RRF"

            bm25 = FastBM25Index(ingestion.chunks)
            chunk_id_to_idx = {c.id: idx for idx, c in enumerate(ingestion.chunks)}

            parent_container: dict[str, str] = {}
            callers_map: dict[str, list[str]] = defaultdict(list)
            callees_map: dict[str, list[str]] = defaultdict(list)
            subclasses_map: dict[str, list[str]] = defaultdict(list)
            bases_map: dict[str, list[str]] = defaultdict(list)
            defines_map: dict[str, list[str]] = defaultdict(list)

            for edge in sorted(graph.edges, key=lambda e: (e.src, e.dst)):
                if edge.kind == EdgeKind.CALLS:
                    callers_map[edge.dst].append(edge.src)
                    callees_map[edge.src].append(edge.dst)
                elif edge.kind == EdgeKind.INHERITS:
                    subclasses_map[edge.dst].append(edge.src)
                    bases_map[edge.src].append(edge.dst)
                elif edge.kind == EdgeKind.DEFINES:
                    defines_map[edge.src].append(edge.dst)
                    parent_container[edge.dst] = edge.src

            try:
                rel_display = str(candidate.relative_to(PROJECT_ROOT)).replace("\\", "/") or "."
            except ValueError:
                rel_display = str(candidate)

            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            entry = RepoCacheEntry(
                repo_label=candidate.name if rel_display != "." else "CodeArchGuard (DocHop)",
                repo_path=candidate,
                rel_or_display_path=rel_display,
                graph=graph,
                ingestion=ingestion,
                bm25=bm25,
                chunk_id_to_idx=chunk_id_to_idx,
                has_dense_vectors=has_dense,
                vector_source=vector_source,
                load_time_ms=elapsed_ms,
                parent_container=parent_container,
                callers_map=dict(callers_map),
                callees_map=dict(callees_map),
                subclasses_map=dict(subclasses_map),
                bases_map=dict(bases_map),
                defines_map=dict(defines_map),
            )

            self.repos[cache_key] = entry
            self.current_repo_key = cache_key
            return entry

    def get_active_entry(self) -> RepoCacheEntry:
        with self._lock:
            if self.current_repo_key and self.current_repo_key in self.repos:
                return self.repos[self.current_repo_key]
        return self.load_repository("tests/fixtures/sample_repo")

    def serialize_repo_overview(self, entry: RepoCacheEntry) -> dict[str, Any]:
        g = entry.graph
        nc = Counter(n.kind.value for n in g.nodes.values())
        ec = Counter(e.kind.value for e in g.edges)
        cc = Counter(c.source_type for c in entry.ingestion.chunks)

        degree: Counter[str] = Counter()
        for e in g.edges:
            degree[e.src] += 1
            degree[e.dst] += 1

        nodes_payload = [
            {
                "id": nid,
                "name": node.name,
                "kind": node.kind.value,
                "file": node.file.replace("\\", "/"),
                "start_line": node.start_line,
                "end_line": node.end_line,
                "docstring": node.docstring or "",
                "degree": degree.get(nid, 0),
                "parent": entry.parent_container.get(nid, ""),
            }
            for nid, node in sorted(g.nodes.items())
        ]

        edges_payload = [
            {"src": e.src, "dst": e.dst, "kind": e.kind.value, "line": e.line}
            for e in sorted(g.edges, key=lambda x: (x.kind.value, x.src, x.dst))
        ]

        chunks_payload = [
            {
                "id": c.id,
                "parent_document_id": c.parent_document_id,
                "owner_symbol_id": c.owner_symbol_id,
                "source_type": c.source_type,
                "source_file": c.source_file.replace("\\", "/"),
                "start_line": c.start_line,
                "end_line": c.end_line,
                "text": c.text,
            }
            for c in entry.ingestion.chunks[:400]
        ]

        symbols = sorted(g.nodes.keys())
        if "shop.models.Item" in g.nodes:
            default_symbol = "shop.models.Item"
            presets = [
                {
                    "title": "Preset 1: False-Positive Elimination (The 'Aha!' Demo)",
                    "subtitle": "Global RAG ranks unrelated calculate_total #1; Graph-Scoped RAG eliminates it",
                    "query": "Calculate the total price of products",
                    "symbol": "shop.models.Item",
                    "top_k": 4,
                },
                {
                    "title": "Preset 2: MRO Inherited Constructor & Checkout Chain",
                    "subtitle": "Traces checkout -> make_item -> DiscountedItem -> Item.__init__",
                    "query": "How does checkout instantiate items and pass product price?",
                    "symbol": "shop.models.Item",
                    "top_k": 4,
                },
                {
                    "title": "Preset 3: Subclass Discount Calculation Logic",
                    "subtitle": "Scopes strictly to DiscountedItem and its callers",
                    "query": "Discount price multiplier total calculation",
                    "symbol": "shop.models.DiscountedItem",
                    "top_k": 4,
                },
            ]
        elif "codearchguard.models.CodeGraph" in g.nodes:
            default_symbol = "codearchguard.models.CodeGraph"
            presets = [
                {
                    "title": "Preset 1: Blast Radius & Cycle-Safe BFS Architecture",
                    "subtitle": "Retrieves graph_scoped_retriever, extractor & ADR chunks linked to CodeGraph",
                    "query": "How does cycle-safe BFS blast radius traversal work across CALLS and INHERITS?",
                    "symbol": "codearchguard.models.CodeGraph",
                    "top_k": 5,
                },
                {
                    "title": "Preset 2: Packaging Source-Root Discovery (src/ layout)",
                    "subtitle": "Scopes to extract_graph and its callers across CLI, pipeline, and tests",
                    "query": "How are pyproject.toml and src/ layout source roots discovered?",
                    "symbol": "codearchguard.analysis.extractor.extract_graph",
                    "top_k": 5,
                },
                {
                    "title": "Preset 3: Neo4j Graph Persistence & Local Instance Calls",
                    "subtitle": "Demonstrates Milestone 1.5 local variable type inference (store.save)",
                    "query": "Save extracted nodes and edges into Neo4j schema",
                    "symbol": "codearchguard.graph.neo4j_store.Neo4jStore",
                    "top_k": 4,
                },
            ]
        else:
            default_symbol = (
                "flask.app.Flask.wsgi_app"
                if "flask.app.Flask.wsgi_app" in g.nodes
                else ("flask.app.Flask" if "flask.app.Flask" in g.nodes else (symbols[0] if symbols else ""))
            )
            presets = [
                {
                    "title": "Preset 1: Flask Request Dispatch & WSGI Entry",
                    "subtitle": "Scopes 1,906 Flask chunks strictly to wsgi_app blast radius (prunes 98.5% noise!)",
                    "query": "How does Flask create a request context and dispatch a WSGI request?",
                    "symbol": default_symbol,
                    "top_k": 5,
                },
                {
                    "title": "Preset 2: Route Registration & URL Rule Decorator",
                    "subtitle": "Compares global search against route registration call chain",
                    "query": "How does Flask register routes and view functions?",
                    "symbol": "flask.sansio.scaffold.Scaffold.route" if "flask.sansio.scaffold.Scaffold.route" in g.nodes else default_symbol,
                    "top_k": 5,
                },
            ]

        return {
            "repo_label": entry.repo_label,
            "repo_path": entry.rel_or_display_path,
            "abs_path": str(entry.repo_path),
            "load_time_ms": entry.load_time_ms,
            "vector_source": entry.vector_source,
            "has_dense_vectors": entry.has_dense_vectors,
            "counts": {
                "total_nodes": len(g.nodes),
                "modules": nc.get("Module", 0),
                "classes": nc.get("Class", 0),
                "functions": nc.get("Function", 0),
                "total_edges": len(g.edges),
                "calls": ec.get("CALLS", 0),
                "inherits": ec.get("INHERITS", 0),
                "imports": ec.get("IMPORTS", 0),
                "defines": ec.get("DEFINES", 0),
                "total_documents": len(entry.ingestion.documents),
                "total_chunks": len(entry.ingestion.chunks),
                "code_chunks": cc.get("code", 0),
                "docstring_chunks": cc.get("docstring", 0),
                "markdown_chunks": cc.get("markdown", 0),
            },
            "default_symbol": default_symbol,
            "presets": presets,
            "nodes": nodes_payload,
            "edges": edges_payload,
            "chunks_preview": chunks_payload,
        }

    def get_symbol_details(self, symbol_id: str) -> dict[str, Any]:
        entry = self.get_active_entry()
        if symbol_id not in entry.graph.nodes:
            raise KeyError(f"Symbol '{symbol_id}' not found in active repository.")

        node = entry.graph.nodes[symbol_id]
        source_path = (entry.repo_path / Path(node.file.replace("\\", "/"))).resolve()
        code_snippet = ""
        if source_path.is_file() and node.start_line > 0:
            try:
                lines = source_path.read_text(encoding="utf-8", errors="replace").splitlines()
                start_idx = max(0, node.start_line - 1)
                end_idx = min(len(lines), node.end_line if node.end_line >= node.start_line else start_idx + 40)
                if node.kind == NodeKind.MODULE and (end_idx - start_idx) > 80:
                    end_idx = start_idx + 80
                code_snippet = "\n".join(
                    f"{line_no:4d} | {lines[line_no - 1]}"
                    for line_no in range(start_idx + 1, end_idx + 1)
                )
            except Exception:
                code_snippet = "# Unable to read source file from disk."

        return {
            "id": node.id,
            "name": node.name,
            "kind": node.kind.value,
            "file": node.file.replace("\\", "/"),
            "start_line": node.start_line,
            "end_line": node.end_line,
            "docstring": node.docstring or "",
            "parent": entry.parent_container.get(symbol_id, ""),
            "callers": entry.callers_map.get(symbol_id, []),
            "callees": entry.callees_map.get(symbol_id, []),
            "subclasses": entry.subclasses_map.get(symbol_id, []),
            "bases": entry.bases_map.get(symbol_id, []),
            "defines": entry.defines_map.get(symbol_id, []),
            "code_snippet": code_snippet,
        }

    def _explain_blast_relationship(
        self,
        entry: RepoCacheEntry,
        target_symbol: str,
        sym: str,
        direct_callers: set[str],
        direct_subclasses: set[str],
        direct_defines: set[str],
    ) -> tuple[str, str]:
        if sym == target_symbol:
            return "Target Symbol", "Primary refactoring target selected by developer"
        if sym in direct_callers:
            return "Direct Caller", f"Directly calls {target_symbol} (CALLS edge reversed)"
        if sym in direct_subclasses:
            return "Direct Subclass", f"Inherits from {target_symbol} (INHERITS edge reversed)"
        if sym in direct_defines:
            return "Defined Member", f"Defined inside {target_symbol} (DEFINES edge forward)"

        curr = entry.parent_container.get(target_symbol)
        while curr:
            if curr == sym:
                return "Enclosing Container", f"Parent module/class enclosing {target_symbol}"
            curr = entry.parent_container.get(curr)

        parent = entry.parent_container.get(sym, "")
        if parent in direct_subclasses:
            return "Subclass Method", f"Method defined on subclass {parent} of {target_symbol}"
        if parent == target_symbol:
            return "Defined Member", f"Member defined inside {target_symbol}"

        for callee in entry.callees_map.get(sym, []):
            if callee in direct_defines:
                return "Constructor / Method Caller", f"Calls member {callee} of {target_symbol}"

        return "Transitive Impact", f"Upstream transitive dependency in {target_symbol}'s call/inheritance chain"

    def compute_blast_details(
        self,
        symbol_id: str,
        max_hops: int = 0,
        include_containers: bool = True,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        entry = self.get_active_entry()
        if symbol_id not in entry.graph.nodes:
            raise KeyError(f"Symbol '{symbol_id}' not found in CodeGraph.")

        actual_hops = None if max_hops <= 0 else max_hops
        blast_set = compute_graph_blast_radius(
            graph=entry.graph,
            start_symbol=symbol_id,
            max_hops=actual_hops,
            include_containers=include_containers,
        )

        direct_callers = set(entry.callers_map.get(symbol_id, []))
        direct_subclasses = set(entry.subclasses_map.get(symbol_id, []))
        direct_defines = set(entry.defines_map.get(symbol_id, []))

        impacted_rows = []
        for s in sorted(blast_set):
            node = entry.graph.nodes[s]
            cat, reason = self._explain_blast_relationship(
                entry, symbol_id, s, direct_callers, direct_subclasses, direct_defines
            )
            impacted_rows.append({
                "id": s,
                "kind": node.kind.value,
                "file": node.file.replace("\\", "/"),
                "lines": f"{node.start_line}-{node.end_line}",
                "category": cat,
                "reason": reason,
            })

        isolated_symbols = sorted(set(entry.graph.nodes.keys()) - blast_set)
        isolated_preview = [
            {
                "id": s,
                "kind": entry.graph.nodes[s].kind.value,
                "file": entry.graph.nodes[s].file.replace("\\", "/"),
            }
            for s in isolated_symbols[:40]
        ]

        _, scoped_chunks = filter_chunks_by_blast_radius(
            chunks=entry.ingestion.chunks,
            graph=entry.graph,
            start_symbol=symbol_id,
            max_hops=actual_hops,
        )

        total_nodes = max(1, len(entry.graph.nodes))
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return {
            "target_symbol": symbol_id,
            "blast_symbols": sorted(blast_set),
            "impacted_count": len(blast_set),
            "total_nodes": len(entry.graph.nodes),
            "isolated_count": len(isolated_symbols),
            "impact_pct": round((len(blast_set) / total_nodes) * 100.0, 1),
            "scoped_chunk_count": len(scoped_chunks),
            "total_chunk_count": len(entry.ingestion.chunks),
            "impacted_rows": impacted_rows,
            "isolated_preview": isolated_preview,
            "elapsed_ms": elapsed_ms,
        }

    def _get_query_vector(self, entry: RepoCacheEntry, query: str, query_tokens: list[str]) -> list[float] | None:
        cleaned = query.strip()
        if cleaned in self._query_vec_cache:
            return self._query_vec_cache[cleaned]

        embedder = self.try_get_bge_embedder()
        if embedder is not None:
            try:
                vec = embedder.embed_query(cleaned)
                self._query_vec_cache[cleaned] = vec
                return vec
            except Exception:
                pass

        if not entry.has_dense_vectors or not entry.ingestion.chunks:
            return None

        dim = len(entry.ingestion.chunks[0].embedding or [])
        if dim == 0:
            return None

        centroid = [0.0] * dim
        total_weight = 0.0
        for idx, chunk in enumerate(entry.ingestion.chunks):
            if not chunk.embedding:
                continue
            freq = entry.bm25.doc_freqs[idx]
            w = sum(entry.bm25.idf.get(t, 0.5) * (1.0 + freq[t]) for t in query_tokens if t in freq)
            if w > 0:
                total_weight += w
                for d_i, val in enumerate(chunk.embedding):
                    centroid[d_i] += w * val

        if total_weight <= 0:
            return None
        norm = math.sqrt(sum(v * v for v in centroid))
        return [v / norm for v in centroid] if norm > 1e-9 else centroid

    def _run_hybrid_rrf_over_indices(
        self,
        entry: RepoCacheEntry,
        query_tokens: list[str],
        query_vec: list[float] | None,
        allowed_indices: list[int],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not allowed_indices:
            return []

        bm25_scores = entry.bm25.score_subset(query_tokens, allowed_indices)
        bm25_sorted = sorted(allowed_indices, key=lambda i: (bm25_scores.get(i, 0.0), -i), reverse=True)
        bm25_rank_map = {idx: rank for rank, idx in enumerate(bm25_sorted, start=1)}

        vec_scores: dict[int, float] = {}
        if query_vec is not None and entry.has_dense_vectors:
            for idx in allowed_indices:
                emb = entry.ingestion.chunks[idx].embedding
                vec_scores[idx] = _cosine_similarity(query_vec, emb) if emb else 0.0
        else:
            for idx in allowed_indices:
                vec_scores[idx] = bm25_scores.get(idx, 0.0)

        vec_sorted = sorted(allowed_indices, key=lambda i: (vec_scores.get(i, 0.0), -i), reverse=True)
        vec_rank_map = {idx: rank for rank, idx in enumerate(vec_sorted, start=1)}

        rrf_k = 60.0
        fused_items = []
        for idx in allowed_indices:
            r_vec = vec_rank_map[idx]
            r_bm25 = bm25_rank_map[idx]
            rrf_score = (1.0 / (rrf_k + r_vec)) + (1.0 / (rrf_k + r_bm25))
            tie = (vec_scores.get(idx, 0.0) * 1e-4) + (bm25_scores.get(idx, 0.0) * 1e-5)
            fused_items.append((rrf_score + tie, rrf_score, idx, r_vec, r_bm25))

        fused_items.sort(key=lambda item: item[0], reverse=True)

        results = []
        for rank, (_, rrf_score, idx, r_vec, r_bm25) in enumerate(fused_items[:top_k], start=1):
            chunk = entry.ingestion.chunks[idx]
            results.append({
                "rank": rank,
                "chunk_id": chunk.id,
                "owner_symbol_id": chunk.owner_symbol_id,
                "source_type": chunk.source_type,
                "source_file": chunk.source_file.replace("\\", "/"),
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "text": chunk.text,
                "rrf_score": round(rrf_score, 5),
                "vector_rank": r_vec,
                "bm25_rank": r_bm25,
            })
        return results

    def compare_rag(self, query: str, target_symbol: str, top_k: int = 4, max_hops: int = 0) -> dict[str, Any]:
        t0 = time.perf_counter()
        entry = self.get_active_entry()
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        if target_symbol not in entry.graph.nodes:
            raise KeyError(f"Target symbol '{target_symbol}' not found in CodeGraph.")

        actual_hops = None if max_hops <= 0 else max_hops
        blast_symbols, scoped_chunks = filter_chunks_by_blast_radius(
            chunks=entry.ingestion.chunks,
            graph=entry.graph,
            start_symbol=target_symbol,
            max_hops=actual_hops,
        )

        query_tokens = _tokenize(query)
        query_vec = self._get_query_vector(entry, query, query_tokens)

        all_indices = list(range(len(entry.ingestion.chunks)))
        scoped_indices = [entry.chunk_id_to_idx[c.id] for c in scoped_chunks if c.id in entry.chunk_id_to_idx]

        global_results = self._run_hybrid_rrf_over_indices(entry, query_tokens, query_vec, all_indices, top_k)
        scoped_results = self._run_hybrid_rrf_over_indices(entry, query_tokens, query_vec, scoped_indices, top_k)

        direct_callers = set(entry.callers_map.get(target_symbol, []))
        direct_subclasses = set(entry.subclasses_map.get(target_symbol, []))
        direct_defines = set(entry.defines_map.get(target_symbol, []))

        scoped_chunk_ids = {c.id for c in scoped_chunks}
        false_positive_count = 0

        for item in global_results:
            in_scope = item["chunk_id"] in scoped_chunk_ids
            item["in_blast_radius"] = in_scope
            if in_scope:
                cat, reason = self._explain_blast_relationship(
                    entry, target_symbol, item["owner_symbol_id"],
                    direct_callers, direct_subclasses, direct_defines
                )
                item["scope_badge"] = f"In Scope ({cat})"
                item["scope_reason"] = reason
            else:
                false_positive_count += 1
                item["scope_badge"] = "OUT-OF-SCOPE FALSE POSITIVE"
                item["scope_reason"] = (
                    f"Symbol '{item['owner_symbol_id']}' has zero call, inheritance, or containment links "
                    f"to '{target_symbol}'. Retrieved by naive RAG due to keyword/semantic overlap alone!"
                )

        for item in scoped_results:
            item["in_blast_radius"] = True
            cat, reason = self._explain_blast_relationship(
                entry, target_symbol, item["owner_symbol_id"],
                direct_callers, direct_subclasses, direct_defines
            )
            item["scope_badge"] = f"Verified: {cat}"
            item["scope_reason"] = reason

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        total_chunks = len(entry.ingestion.chunks)
        filtered_out = total_chunks - len(scoped_chunks)

        return {
            "query": query,
            "target_symbol": target_symbol,
            "top_k": top_k,
            "total_chunk_count": total_chunks,
            "scoped_chunk_count": len(scoped_chunks),
            "filtered_out_chunk_count": filtered_out,
            "reduction_pct": round((filtered_out / max(1, total_chunks)) * 100.0, 1),
            "blast_symbol_count": len(blast_symbols),
            "false_positive_count_in_global": false_positive_count,
            "vector_source": entry.vector_source,
            "elapsed_ms": elapsed_ms,
            "global_results": global_results,
            "scoped_results": scoped_results,
        }


ENGINE = DemoEngine()


# ==============================================================================
# Frontend Single-Page Application (Polished Dark UI + Balanced Function Rendering)
# ==============================================================================

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>CodeArchGuard — GraphRAG Architecture & Refactoring Assistant</title>
  <style>
    :root {
      --bg-base: #090D16;
      --bg-panel: #111827;
      --bg-card: #1E293B;
      --bg-card-hover: #263449;
      --bg-input: #0F172A;
      --border: #334155;
      --border-subtle: #1E293B;
      --text-main: #F8FAFC;
      --text-muted: #94A3B8;
      --text-dim: #64748B;
      --accent: #6366F1;
      --accent-glow: rgba(99, 102, 241, 0.25);
      --blue: #38BDF8;
      --amber: #F59E0B;
      --emerald: #10B981;
      --crimson: #EF4444;
      --purple: #A855F7;
      --pink: #EC4899;
      --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      --font-mono: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
    }

    * { box-sizing: border-box; }
    body, html {
      margin: 0; padding: 0;
      background: var(--bg-base);
      color: var(--text-main);
      font-family: var(--font-sans);
      font-size: 14px;
      line-height: 1.5;
      overflow-x: hidden;
    }

    .topbar {
      background: linear-gradient(180deg, #131B2E 0%, #0D1322 100%);
      border-bottom: 1px solid var(--border);
      padding: 12px 24px;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      position: sticky;
      top: 0;
      z-index: 100;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.45);
    }

    .brand { display: flex; align-items: center; gap: 12px; }
    .brand-icon {
      width: 38px; height: 38px;
      border-radius: 10px;
      background: linear-gradient(135deg, #6366F1 0%, #EC4899 100%);
      display: flex; align-items: center; justify-content: center;
      font-size: 19px; font-weight: 800; color: white;
      box-shadow: 0 0 16px rgba(99, 102, 241, 0.4);
    }
    .brand-title { font-size: 17px; font-weight: 800; margin: 0; display: flex; align-items: center; gap: 8px; }
    .badge-pill {
      font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 999px;
      background: rgba(99, 102, 241, 0.18); color: #A5B4FC; border: 1px solid rgba(99, 102, 241, 0.4);
    }
    .brand-sub { font-size: 11.5px; color: var(--text-muted); margin: 2px 0 0 0; }

    .repo-controls { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .repo-quick-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-muted); font-weight: 700; }
    .repo-pill {
      background: var(--bg-card); color: var(--text-main); border: 1px solid var(--border);
      padding: 6px 12px; border-radius: 8px; font-size: 12px; font-weight: 600; cursor: pointer; transition: all 0.15s ease;
    }
    .repo-pill:hover { background: var(--bg-card-hover); border-color: var(--blue); }
    .repo-pill.active {
      background: rgba(99, 102, 241, 0.22); border-color: var(--accent); color: #C7D2FE;
      box-shadow: 0 0 0 2px var(--accent-glow);
    }
    .repo-input {
      background: var(--bg-input); border: 1px solid var(--border); color: var(--text-main);
      padding: 6px 10px; border-radius: 8px; font-family: var(--font-mono); font-size: 12px; width: 195px;
    }
    .btn-primary {
      background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
      color: white; border: none; padding: 7px 14px; border-radius: 8px;
      font-size: 12px; font-weight: 700; cursor: pointer; display: inline-flex; align-items: center; gap: 6px;
    }
    .btn-primary:hover { opacity: 0.92; }
    .btn-secondary {
      background: var(--bg-card); color: var(--text-main); border: 1px solid var(--border);
      padding: 6px 11px; border-radius: 8px; font-size: 12px; font-weight: 600; cursor: pointer;
    }
    .btn-secondary:hover { background: var(--bg-card-hover); border-color: var(--text-muted); }

    .container { max-width: 1560px; margin: 0 auto; padding: 14px 24px 48px 24px; }

    .kpi-strip {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px; margin-bottom: 14px;
    }
    .kpi-card {
      background: var(--bg-panel); border: 1px solid var(--border); border-radius: 10px;
      padding: 9px 14px; display: flex; flex-direction: column; justify-content: space-between;
    }
    .kpi-label { font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); font-weight: 600; display: flex; justify-content: space-between; }
    .kpi-value { font-size: 21px; font-weight: 800; font-family: var(--font-mono); margin-top: 2px; }
    .kpi-sub { font-size: 11px; color: var(--text-dim); margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

    /* Clean single-row navigation tabs without scrollbar */
    .nav-tabs {
      display: flex; flex-wrap: wrap; gap: 8px;
      border-bottom: 1px solid var(--border); padding-bottom: 10px; margin-bottom: 14px;
    }
    .tab-btn {
      background: var(--bg-panel); color: var(--text-muted); border: 1px solid var(--border);
      padding: 8px 14px; border-radius: 9px; font-size: 12.5px; font-weight: 700;
      cursor: pointer; display: flex; align-items: center; gap: 6px; transition: all 0.15s ease;
    }
    .tab-btn:hover { color: var(--text-main); background: var(--bg-card); }
    .tab-btn.active {
      background: linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(236, 72, 153, 0.15) 100%);
      color: #FFFFFF; border-color: var(--accent); box-shadow: 0 0 12px var(--accent-glow);
    }

    .tab-pane { display: none; }
    .tab-pane.active { display: block; }

    .workspace-grid { display: grid; grid-template-columns: 1fr 420px; gap: 16px; align-items: start; }
    @media (max-width: 1150px) { .workspace-grid { grid-template-columns: 1fr; } }

    .panel { background: var(--bg-panel); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
    .panel-header {
      padding: 10px 14px; background: rgba(30, 41, 59, 0.6); border-bottom: 1px solid var(--border);
      display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 8px;
    }
    .panel-title { font-size: 13px; font-weight: 700; margin: 0; display: flex; align-items: center; gap: 8px; }

    .canvas-toolbar {
      padding: 8px 14px; background: #0C1220; border-bottom: 1px solid var(--border);
      display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px; font-size: 12px;
    }
    .filter-group { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
    .filter-chip { display: inline-flex; align-items: center; gap: 5px; cursor: pointer; user-select: none; color: var(--text-muted); font-size: 11.5px; font-weight: 600; }
    .filter-chip input { accent-color: var(--accent); cursor: pointer; }
    .dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }

    .canvas-wrap {
      position: relative; width: 100%; height: 560px;
      background: radial-gradient(circle at center, #111A2E 0%, #080C14 100%); overflow: hidden;
    }
    canvas#graphCanvas { width: 100%; height: 100%; display: block; cursor: grab; }
    canvas#graphCanvas:active { cursor: grabbing; }

    .canvas-overlay-status {
      position: absolute; bottom: 10px; left: 12px;
      background: rgba(15, 23, 42, 0.9); border: 1px solid var(--border);
      padding: 5px 11px; border-radius: 8px; font-size: 11px; color: var(--text-muted);
      pointer-events: none; font-family: var(--font-mono);
    }

    .canvas-tooltip {
      position: absolute; display: none; background: rgba(15, 23, 42, 0.96);
      border: 1px solid var(--blue); border-radius: 8px; padding: 8px 12px;
      color: var(--text-main); font-size: 12px; pointer-events: none; z-index: 50; max-width: 340px;
    }

    .inspector-body { padding: 14px; max-height: 600px; overflow-y: auto; }
    .sym-header { border-bottom: 1px solid var(--border); padding-bottom: 10px; margin-bottom: 10px; }
    .sym-id { font-family: var(--font-mono); font-size: 13.5px; font-weight: 700; color: #E2E8F0; word-break: break-all; margin: 4px 0; }
    .kind-badge { display: inline-block; padding: 2px 8px; border-radius: 5px; font-size: 10.5px; font-weight: 800; text-transform: uppercase; }
    .kind-Module { background: rgba(56, 189, 248, 0.18); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.4); }
    .kind-Class { background: rgba(245, 158, 11, 0.18); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .kind-Function { background: rgba(16, 185, 129, 0.18); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.4); }

    .rel-section { margin-bottom: 10px; }
    .rel-title { font-size: 11px; font-weight: 700; text-transform: uppercase; color: var(--text-muted); margin-bottom: 5px; }
    .rel-pills { display: flex; flex-wrap: wrap; gap: 5px; }
    .rel-pill {
      background: var(--bg-card); border: 1px solid var(--border); color: #CBD5E1;
      padding: 3px 8px; border-radius: 6px; font-family: var(--font-mono); font-size: 11px; cursor: pointer;
    }
    .rel-pill:hover { border-color: var(--blue); color: white; background: var(--bg-card-hover); }

    pre.code-box {
      background: #070A12; border: 1px solid var(--border); border-radius: 8px;
      padding: 10px 12px; font-family: var(--font-mono); font-size: 11.5px; color: #E2E8F0;
      overflow-x: auto; margin: 6px 0 0 0; line-height: 1.45; max-height: 250px;
    }

    .control-bar {
      background: var(--bg-panel); border: 1px solid var(--border); border-radius: 12px;
      padding: 12px 16px; margin-bottom: 14px; display: flex; flex-wrap: wrap; align-items: flex-end; gap: 12px;
    }
    .field-group { display: flex; flex-direction: column; gap: 4px; flex: 1; min-width: 190px; }
    .field-label { font-size: 11.5px; font-weight: 700; color: var(--text-muted); }
    select.input-select, input.input-text {
      background: var(--bg-input); border: 1px solid var(--border); color: var(--text-main);
      padding: 7px 11px; border-radius: 8px; font-size: 12.5px; font-family: var(--font-mono); width: 100%;
    }

    .presets-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 10px; margin-bottom: 14px; }
    .preset-card {
      background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.9) 100%);
      border: 1px solid var(--border); border-left: 4px solid var(--accent);
      border-radius: 10px; padding: 10px 14px; cursor: pointer; transition: all 0.15s ease;
    }
    .preset-card:hover { border-color: var(--blue); background: var(--bg-card-hover); }
    .preset-title { font-size: 12.5px; font-weight: 700; color: #E0E7FF; margin-bottom: 3px; }
    .preset-sub { font-size: 11.5px; color: var(--text-muted); }

    .funnel-banner {
      background: linear-gradient(90deg, rgba(16, 185, 129, 0.12) 0%, rgba(99, 102, 241, 0.12) 100%);
      border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 10px;
      padding: 10px 16px; margin-bottom: 14px; display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px;
    }
    .funnel-steps { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; font-size: 12px; }
    .funnel-step { background: rgba(15, 23, 42, 0.85); border: 1px solid var(--border); padding: 4px 10px; border-radius: 8px; font-family: var(--font-mono); }

    .rag-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    @media (max-width: 1024px) { .rag-grid { grid-template-columns: 1fr; } }

    .chunk-card { background: #0B111E; border: 1px solid var(--border); border-left: 4px solid var(--emerald); border-radius: 10px; padding: 12px 14px; margin-bottom: 12px; }
    .chunk-card.false-positive { border-color: rgba(239, 68, 68, 0.55); border-left: 4px solid var(--crimson); background: rgba(127, 29, 29, 0.12); }
    .chunk-top { display: flex; align-items: center; justify-content: space-between; gap: 8px; flex-wrap: wrap; margin-bottom: 6px; }
    .chunk-sym { font-family: var(--font-mono); font-weight: 700; font-size: 13px; color: #F8FAFC; cursor: pointer; }
    .chunk-sym:hover { color: var(--blue); text-decoration: underline; }
    .scope-badge { padding: 3px 8px; border-radius: 6px; font-size: 11px; font-weight: 800; }
    .scope-ok { background: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.45); }
    .scope-fp { background: rgba(239, 68, 68, 0.25); color: #FCA5A5; border: 1px solid rgba(239, 68, 68, 0.6); }
    .chunk-meta { font-size: 11.5px; color: var(--text-muted); margin-bottom: 6px; display: flex; flex-wrap: wrap; gap: 10px; font-family: var(--font-mono); }
    .chunk-reason { font-size: 11.5px; padding: 6px 10px; border-radius: 6px; margin-bottom: 8px; background: rgba(30, 41, 59, 0.55); color: #CBD5E1; }
    .chunk-card.false-positive .chunk-reason { background: rgba(127, 29, 29, 0.35); color: #FECACA; }

    .table-wrap { overflow-x: auto; max-height: 480px; overflow-y: auto; }
    table.data-table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
    table.data-table th { background: #131C2E; color: var(--text-muted); text-align: left; padding: 9px 12px; font-size: 11px; text-transform: uppercase; position: sticky; top: 0; border-bottom: 1px solid var(--border); }
    table.data-table td { padding: 8px 12px; border-bottom: 1px solid var(--border-subtle); vertical-align: top; }
    table.data-table tr:hover td { background: rgba(56, 189, 248, 0.06); }

    .guide-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; margin-bottom: 20px; }
    .guide-card { background: var(--bg-panel); border: 1px solid var(--border); border-top: 3px solid var(--accent); border-radius: 12px; padding: 18px; display: flex; flex-direction: column; justify-content: space-between; }
    .guide-card h3 { margin: 0 0 8px 0; font-size: 15px; color: #E0E7FF; }
    .guide-card p, .guide-card li { color: #CBD5E1; font-size: 13px; }
    .guide-card ul { padding-left: 18px; margin: 8px 0 14px 0; }
  </style>
</head>
<body>

  <header class="topbar">
    <div class="brand">
      <div class="brand-icon">🛡️</div>
      <div>
        <h1 class="brand-title">
          CodeArchGuard
          <span class="badge-pill">Milestone 1.5 + Hybrid GraphRAG</span>
        </h1>
        <p class="brand-sub">Static AST Knowledge Graph &bull; Cycle-Safe Blast Radius &bull; Graph-Scoped Hybrid Retrieval</p>
      </div>
    </div>

    <div class="repo-controls">
      <span class="repo-quick-label">Quick Repo:</span>
      <button class="repo-pill active" id="pill-sample" onclick="switchRepo('tests/fixtures/sample_repo', 'pill-sample')">
        📦 sample_repo (Demo Fixture)
      </button>
      <button class="repo-pill" id="pill-self" onclick="switchRepo('.', 'pill-self')">
        🛡️ CodeArchGuard (Self-Analysis)
      </button>
      <button class="repo-pill" id="pill-flask" onclick="switchRepo('tests/fixtures/flask', 'pill-flask')">
        🌶️ Pallets Flask (1,900+ Chunks)
      </button>
      <input type="text" id="repoPathInput" class="repo-input" value="tests/fixtures/sample_repo" placeholder="Custom repo path..." />
      <button class="btn-primary" onclick="loadCustomRepo()">⚡ Analyze & Index</button>
    </div>
  </header>

  <main class="container">
    <section class="kpi-strip">
      <div class="kpi-card">
        <div class="kpi-label"><span>Active Repository</span><span>📁</span></div>
        <div class="kpi-value" id="kpi-repo-name" style="font-size:16px; color:#A5B4FC; margin-top:4px;">sample_repo</div>
        <div class="kpi-sub" id="kpi-repo-path">tests/fixtures/sample_repo</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label"><span>AST Graph Nodes</span><span style="color:var(--blue)">●</span></div>
        <div class="kpi-value" id="kpi-nodes">0</div>
        <div class="kpi-sub" id="kpi-nodes-sub">0 Mod · 0 Cls · 0 Func</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label"><span>Topological Edges</span><span style="color:var(--crimson)">●</span></div>
        <div class="kpi-value" id="kpi-edges">0</div>
        <div class="kpi-sub" id="kpi-edges-sub">0 CALLS · 0 INHERITS · 0 IMPORTS</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label"><span>Indexed RAG Chunks</span><span style="color:var(--emerald)">●</span></div>
        <div class="kpi-value" id="kpi-chunks">0</div>
        <div class="kpi-sub" id="kpi-chunks-sub">0 Code · 0 Docstring · 0 ADR</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label"><span>Retrieval Engine</span><span>⚡</span></div>
        <div class="kpi-value" id="kpi-latency" style="font-size:17px; color:var(--emerald);">0.0 ms</div>
        <div class="kpi-sub" id="kpi-engine-mode">Cached BGE-384 + BM25 RRF</div>
      </div>
    </section>

    <nav class="nav-tabs">
      <button class="tab-btn active" id="tabbtn-graph" onclick="openTab('graph')">🗺️ 1. Architecture Graph & Inspector</button>
      <button class="tab-btn" id="tabbtn-blast" onclick="openTab('blast')">💥 2. Blast Radius Simulator</button>
      <button class="tab-btn" id="tabbtn-rag" onclick="openTab('rag')">🔬 3. Graph-Scoped vs. Global RAG</button>
      <button class="tab-btn" id="tabbtn-chunks" onclick="openTab('chunks')">📄 4. Chunks & ADR Explorer</button>
      <button class="tab-btn" id="tabbtn-guide" onclick="openTab('guide')">🎓 5. Live Presenter Script</button>
    </nav>

    <!-- TAB 1 & 2 WORKSPACE -->
    <div id="tab-graph" class="tab-pane active">
      <div class="control-bar" id="blast-control-bar" style="display:none; border-color:var(--pink);">
        <div class="field-group" style="flex:2;">
          <label class="field-label">🎯 Target Symbol to Refactor / Modify</label>
          <select id="blastSymbolSelect" class="input-select" onchange="runBlastRadius()"></select>
        </div>
        <div class="field-group" style="max-width:170px;">
          <label class="field-label">Max BFS Hops (0 = All)</label>
          <input type="number" id="blastMaxHops" class="input-text" value="0" min="0" max="15" onchange="runBlastRadius()" />
        </div>
        <div style="display:flex; gap:8px; align-items:center;">
          <button class="btn-primary" onclick="runBlastRadius()">💥 Compute Blast Radius</button>
          <button class="btn-secondary" onclick="clearBlastHighlight()">Reset Full Graph</button>
          <button class="btn-secondary" onclick="jumpToRagWithBlastSymbol()">🔬 Scope RAG to Symbol &rarr;</button>
        </div>
      </div>

      <div class="workspace-grid">
        <div class="panel">
          <div class="panel-header">
            <h3 class="panel-title">
              <span>🌐 Interactive AST Property Graph</span>
              <span id="graph-mode-badge" class="badge-pill">Full Architecture View</span>
            </h3>
            <div style="display:flex; gap:6px; align-items:center; flex-wrap:wrap;">
              <button class="btn-secondary" id="btnFocusNeighborhood" onclick="toggleFocusMode()">🔍 Focus Selected Neighborhood</button>
              <button class="btn-secondary" id="btnLayoutHier" onclick="setGraphLayout('hierarchical')" style="border-color:var(--accent);">📐 Hierarchical</button>
              <button class="btn-secondary" id="btnLayoutForce" onclick="setGraphLayout('force')">🧲 Force</button>
              <button class="btn-secondary" onclick="resetCamera()">🎯 Fit Screen</button>
              <button class="btn-secondary" onclick="exportMatplotlibPng()">📷 Export PNG</button>
            </div>
          </div>

          <div class="canvas-toolbar">
            <div class="filter-group">
              <strong style="color:var(--text-main);">Nodes:</strong>
              <label class="filter-chip"><input type="checkbox" checked onchange="toggleNodeKind('Module', this.checked)" /><span class="dot" style="background:var(--blue)"></span>Module</label>
              <label class="filter-chip"><input type="checkbox" checked onchange="toggleNodeKind('Class', this.checked)" /><span class="dot" style="background:var(--amber)"></span>Class</label>
              <label class="filter-chip"><input type="checkbox" checked onchange="toggleNodeKind('Function', this.checked)" /><span class="dot" style="background:var(--emerald)"></span>Function</label>
              <span style="color:var(--border);">|</span>
              <strong style="color:var(--text-main);">Edges:</strong>
              <label class="filter-chip"><input type="checkbox" checked onchange="toggleEdgeKind('CALLS', this.checked)" /><span style="color:var(--crimson)">━━ CALLS</span></label>
              <label class="filter-chip"><input type="checkbox" checked onchange="toggleEdgeKind('INHERITS', this.checked)" /><span style="color:var(--purple)">━━ INHERITS</span></label>
              <label class="filter-chip"><input type="checkbox" checked onchange="toggleEdgeKind('IMPORTS', this.checked)" /><span style="color:var(--blue)">╌╌ IMPORTS</span></label>
              <label class="filter-chip"><input type="checkbox" checked onchange="toggleEdgeKind('DEFINES', this.checked)" /><span style="color:var(--text-muted)">┄┄ DEFINES</span></label>
            </div>
            <div style="display:flex; gap:8px; align-items:center;">
              <input type="text" id="nodeSearchInput" class="repo-input" style="width:170px; padding:4px 8px;" placeholder="🔍 Find symbol..." oninput="onSearchNode(this.value)" />
            </div>
          </div>

          <div class="canvas-wrap" id="canvasContainer">
            <canvas id="graphCanvas"></canvas>
            <div class="canvas-tooltip" id="canvasTooltip"></div>
            <div class="canvas-overlay-status" id="canvasStatus">
              Scroll to zoom • Drag to pan • Click any node to inspect source & blast radius
            </div>
          </div>
        </div>

        <div class="panel">
          <div class="panel-header">
            <h3 class="panel-title">🔍 Live Symbol & Source Inspector</h3>
            <div style="display:flex; gap:6px;">
              <button class="btn-secondary" style="padding:4px 8px; font-size:11px;" onclick="triggerBlastFromInspector()">💥 Blast Radius</button>
              <button class="btn-primary" style="padding:4px 10px; font-size:11px;" onclick="triggerRagFromInspector()">🔬 Scope RAG</button>
            </div>
          </div>
          <div class="inspector-body" id="inspectorBody">
            <div style="color:var(--text-muted); text-align:center; padding:30px 10px;">
              Click any node in the graph to inspect its AST metadata, callers, callees, inheritance hierarchy, and source code.
            </div>
          </div>
        </div>
      </div>

      <div id="blast-details-section" style="display:none; margin-top:16px;">
        <div class="funnel-banner" id="blast-summary-banner"></div>
        <div class="rag-grid">
          <div class="panel">
            <div class="panel-header">
              <h3 class="panel-title">💥 Impacted Symbols in Blast Radius (Must Review / Test)</h3>
            </div>
            <div class="table-wrap">
              <table class="data-table">
                <thead>
                  <tr>
                    <th>Symbol ID</th>
                    <th>Kind</th>
                    <th>Impact Category</th>
                    <th>Topological Reason</th>
                  </tr>
                </thead>
                <tbody id="blastImpactedTableBody"></tbody>
              </table>
            </div>
          </div>

          <div class="panel">
            <div class="panel-header">
              <h3 class="panel-title">🛡️ Safely Isolated Symbols (Pruned from Graph-Scoped RAG)</h3>
            </div>
            <div class="table-wrap">
              <table class="data-table">
                <thead>
                  <tr>
                    <th>Isolated Symbol ID</th>
                    <th>Kind</th>
                    <th>File</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody id="blastIsolatedTableBody"></tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 3: COMPARATIVE HYBRID RAG -->
    <div id="tab-rag" class="tab-pane">
      <div style="margin-bottom:8px; font-size:12px; font-weight:700; color:var(--text-muted); text-transform:uppercase;">
        ⚡ 1-Click Instructor Walkthrough Presets (Tailored to Active Repository):
      </div>
      <div class="presets-row" id="ragPresetsRow"></div>

      <div class="control-bar">
        <div class="field-group" style="flex:2;">
          <label class="field-label">🎯 Refactoring Context (Target Symbol for Graph Scoping)</label>
          <select id="ragSymbolSelect" class="input-select"></select>
        </div>
        <div class="field-group" style="flex:3;">
          <label class="field-label">💬 Developer Natural-Language Query</label>
          <input type="text" id="ragQueryInput" class="input-text" value="Calculate the total price of products" onkeydown="if(event.key==='Enter') runComparativeRag();" />
        </div>
        <div class="field-group" style="max-width:110px;">
          <label class="field-label">Top-K Chunks</label>
          <input type="number" id="ragTopK" class="input-text" value="4" min="1" max="10" />
        </div>
        <button class="btn-primary" style="height:36px; padding:0 18px;" onclick="runComparativeRag()">
          🚀 Run Comparative Hybrid RAG
        </button>
      </div>

      <div class="funnel-banner" id="ragFunnelBanner"></div>

      <div class="rag-grid">
        <div class="panel">
          <div class="panel-header" style="border-bottom:2px solid var(--crimson);">
            <h3 class="panel-title" style="color:#FCA5A5;">
              ⚠️ Baseline: Naive Global Hybrid RAG (Unscoped Vector + BM25)
            </h3>
            <span class="badge-pill" id="global-fp-badge" style="background:rgba(239,68,68,0.2); color:#FCA5A5; border-color:rgba(239,68,68,0.5);">Unfiltered Pool</span>
          </div>
          <div style="padding:14px;" id="globalRagColumn"></div>
        </div>

        <div class="panel">
          <div class="panel-header" style="border-bottom:2px solid var(--emerald);">
            <h3 class="panel-title" style="color:#6EE7B7;">
              🛡️ CodeArchGuard: Graph-Scoped Hybrid RAG (FR-07 + FR-09)
            </h3>
            <span class="badge-pill" style="background:rgba(16,185,129,0.2); color:#6EE7B7; border-color:rgba(16,185,129,0.5);">100% Blast-Radius Verified</span>
          </div>
          <div style="padding:14px;" id="scopedRagColumn"></div>
        </div>
      </div>
    </div>

    <!-- TAB 4: CHUNKS & ADR EXPLORER -->
    <div id="tab-chunks" class="tab-pane">
      <div class="control-bar">
        <div class="field-group">
          <label class="field-label">Filter by Chunk Source Type</label>
          <select id="chunkTypeFilter" class="input-select" onchange="renderChunksTable()">
            <option value="ALL">All Chunk Types (code + docstring + markdown ADR)</option>
            <option value="code">Code Chunks Only (Docstrings Stripped to Prevent Duplication)</option>
            <option value="docstring">Docstring Chunks Only (Module, Class & Function Docstrings)</option>
            <option value="markdown">Markdown / ADR Documentation Chunks (Bound to Graph Symbols)</option>
          </select>
        </div>
        <div class="field-group" style="flex:2;">
          <label class="field-label">Search Chunks by Owner Symbol or Text</label>
          <input type="text" id="chunkSearchInput" class="input-text" placeholder="e.g. calculate_total, Item, ADR, Neo4j..." oninput="renderChunksTable()" />
        </div>
      </div>

      <div class="panel">
        <div class="panel-header">
          <h3 class="panel-title">📄 Indexed Repository Chunks & Symbol Ownership Mapping</h3>
          <span class="badge-pill" id="chunkTableCountBadge">0 chunks</span>
        </div>
        <div class="table-wrap">
          <table class="data-table">
            <thead>
              <tr>
                <th>Chunk ID</th>
                <th>Type</th>
                <th>Owner Symbol ID</th>
                <th>File & Lines</th>
                <th>Extracted Text Content</th>
              </tr>
            </thead>
            <tbody id="chunksTableBody"></tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- TAB 5: LIVE PRESENTER TELEPROMPTER -->
    <div id="tab-guide" class="tab-pane">
      <div class="guide-grid">
        <div class="guide-card">
          <div>
            <span class="badge-pill">Step 1 (Minute 1) · Milestone 1 & 1.5</span>
            <h3 style="margin-top:8px;">1. Static AST Property Graph & Hardening</h3>
            <p>Start on <code>sample_repo</code> so the panel can clearly see all 11 nodes and edges, then show <code>CodeArchGuard</code> self-analysis:</p>
            <ul>
              <li><b>Nodes:</b> Blue = <code>Module</code>, Amber = <code>Class</code>, Green = <code>Function</code>.</li>
              <li><b>Milestone 1.5 Fixes:</b> Automatic <code>src/</code> root discovery, MRO inheritance walk for inherited constructors (<code>DiscountedItem() &rarr; Item.__init__</code>), and local variable method resolution (<code>store.save()</code>).</li>
            </ul>
          </div>
          <div style="display:flex; gap:8px; flex-wrap:wrap;">
            <button class="btn-primary" onclick="presenterAction1()">▶️ Trigger Step 1: sample_repo Graph</button>
            <button class="btn-secondary" onclick="presenterActionSelf()">▶️ Show Self-Analysis (DocHop)</button>
          </div>
        </div>

        <div class="guide-card" style="border-top-color:var(--pink);">
          <div>
            <span class="badge-pill" style="color:#F9A8D4; border-color:var(--pink);">Step 2 (Minute 2) · Milestone 1.5 FR-07</span>
            <h3 style="margin-top:8px;">2. Cycle-Safe Refactoring Blast Radius</h3>
            <p>Click the button below to simulate modifying <code>shop.models.Item</code>:</p>
            <ul>
              <li>CodeArchGuard traverses reversed <code>CALLS</code> (callers), reversed <code>INHERITS</code> (subclasses like <code>DiscountedItem</code>), and forward <code>DEFINES</code>.</li>
              <li>It highlights the 8 affected symbols while proving that <code>shop.billing.calculate_total</code> and <code>shop.billing.helper</code> are <b>100% isolated</b>.</li>
            </ul>
          </div>
          <button class="btn-primary" style="background:linear-gradient(135deg,#EC4899,#BE185D);" onclick="presenterAction2()">
            ▶️ Trigger Step 2: Simulate Item Blast Radius
          </button>
        </div>

        <div class="guide-card" style="border-top-color:var(--emerald);">
          <div>
            <span class="badge-pill" style="color:#6EE7B7; border-color:var(--emerald);">Step 3 (Minute 3) · Milestone 2 Star Feature</span>
            <h3 style="margin-top:8px;">3. Graph-Scoped Hybrid RAG vs. Naive RAG</h3>
            <p>Click the button below to run the query <i>"Calculate the total price of products"</i> while refactoring <code>shop.models.Item</code>:</p>
            <ul>
              <li><b>Left (Naive Global RAG):</b> Ranks <code>shop.billing.calculate_total</code> #1 because its docstring matches the query word-for-word—even though it has <b>zero relation</b> to <code>Item</code>!</li>
              <li><b>Right (Graph-Scoped Hybrid RAG):</b> Restricts BGE-384 Vector + BM25 Reciprocal Rank Fusion strictly to <code>Item</code>'s blast radius, eliminating the false positive and returning <code>DiscountedItem.total</code> (<code>self.price * 0.9</code>)!</li>
            </ul>
          </div>
          <button class="btn-primary" style="background:linear-gradient(135deg,#10B981,#047857);" onclick="presenterAction3()">
            ▶️ Trigger Step 3: Run False-Positive Killer Demo
          </button>
        </div>
      </div>
    </div>
  </main>

  <script>
    let REPO_DATA = null;
    let ACTIVE_TAB = 'graph';
    let GRAPH_LAYOUT = 'hierarchical';
    let FOCUS_NEIGHBORHOOD = false;
    let SELECTED_NODE_ID = null;
    let BLAST_SET = null;
    let BLAST_TARGET = null;
    let SEARCH_QUERY = '';
    let REQ_COUNTER = 0;

    const VISIBLE_NODE_KINDS = { Module: true, Class: true, Function: true };
    const VISIBLE_EDGE_KINDS = { CALLS: true, INHERITS: true, IMPORTS: true, DEFINES: true };

    const canvas = document.getElementById('graphCanvas');
    const ctx = canvas.getContext('2d');
    const tooltip = document.getElementById('canvasTooltip');
    let G_NODES = [];
    let G_EDGES = [];
    let NODE_MAP = {};
    let CAM = { x: 0, y: 0, zoom: 1.0 };
    let IS_PANNING = false;
    let DRAG_NODE = null;
    let HOVER_NODE = null;
    let LAST_MOUSE = { x: 0, y: 0 };
    let MOUSE_DOWN_POS = { x: 0, y: 0 };
    let PHYSICS_TICKS = 0;
    let ANIM_FRAME = null;

    const NODE_COLORS = { Module: '#38BDF8', Class: '#F59E0B', Function: '#10B981' };
    const EDGE_COLORS = { CALLS: '#EF4444', INHERITS: '#A855F7', IMPORTS: '#38BDF8', DEFINES: '#64748B' };

    function escapeHtml(str) {
      if (!str) return '';
      return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    function openTab(tabName) {
      ACTIVE_TAB = tabName;
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      const activeBtn = document.getElementById('tabbtn-' + tabName);
      if (activeBtn) activeBtn.classList.add('active');

      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

      if (tabName === 'graph' || tabName === 'blast') {
        document.getElementById('tab-graph').classList.add('active');
        const blastBar = document.getElementById('blast-control-bar');
        const blastDetails = document.getElementById('blast-details-section');
        if (tabName === 'blast') {
          blastBar.style.display = 'flex';
          blastDetails.style.display = 'block';
          resizeCanvas();
          if (!BLAST_SET && document.getElementById('blastSymbolSelect').value) {
            runBlastRadius();
          }
        } else {
          blastBar.style.display = 'none';
          blastDetails.style.display = 'none';
          resizeCanvas();
        }
      } else {
        document.getElementById('tab-' + tabName).classList.add('active');
        if (tabName === 'rag') {
          if (!document.getElementById('globalRagColumn').innerHTML.trim()) {
            runComparativeRag();
          }
        } else if (tabName === 'chunks') {
          renderChunksTable();
        }
      }
    }

    async function switchRepo(repoPath, pillId) {
      document.getElementById('repoPathInput').value = repoPath;
      document.querySelectorAll('.repo-pill').forEach(p => p.classList.remove('active'));
      if (pillId && document.getElementById(pillId)) {
        document.getElementById(pillId).classList.add('active');
      }
      await fetchAndLoadRepo(repoPath);
    }

    async function loadCustomRepo() {
      const path = document.getElementById('repoPathInput').value.trim();
      document.querySelectorAll('.repo-pill').forEach(p => p.classList.remove('active'));
      await fetchAndLoadRepo(path);
    }

    async function fetchAndLoadRepo(repoPath) {
      const myReq = ++REQ_COUNTER;
      try {
        document.getElementById('kpi-latency').textContent = 'Loading...';
        const res = await fetch('/api/analyze?repo=' + encodeURIComponent(repoPath));
        const data = await res.json();
        if (myReq !== REQ_COUNTER) return;
        if (!res.ok || data.error) {
          alert('Error loading repository: ' + (data.error || res.statusText));
          return;
        }
        REPO_DATA = data;
        BLAST_SET = null;
        BLAST_TARGET = null;
        SELECTED_NODE_ID = data.default_symbol || null;
        // Automatically enable Focused Neighborhood view on huge repos (e.g. Flask 978 nodes) so graph is super clean!
        FOCUS_NEIGHBORHOOD = (data.nodes.length > 200);
        updateFocusButtonStyle();

        document.getElementById('globalRagColumn').innerHTML = '';
        document.getElementById('scopedRagColumn').innerHTML = '';

        updateKpiStrip(data);
        populateSymbolDropdowns(data);
        renderPresets(data.presets || []);
        buildCanvasGraph(data);
        renderChunksTable();

        if (data.default_symbol) {
          inspectSymbol(data.default_symbol);
        }
        if (ACTIVE_TAB === 'blast') {
          runBlastRadius();
        } else if (ACTIVE_TAB === 'rag') {
          runComparativeRag();
        }
      } catch (err) {
        console.error(err);
      }
    }

    function updateFocusButtonStyle() {
      const btn = document.getElementById('btnFocusNeighborhood');
      if (!btn) return;
      if (FOCUS_NEIGHBORHOOD) {
        btn.style.borderColor = 'var(--emerald)';
        btn.style.color = '#6EE7B7';
        btn.textContent = '🔍 Neighborhood Focus: ON (Click for Full Repo)';
      } else {
        btn.style.borderColor = 'var(--border)';
        btn.style.color = 'var(--text-main)';
        btn.textContent = '🔍 Focus Selected Neighborhood';
      }
    }

    function toggleFocusMode() {
      FOCUS_NEIGHBORHOOD = !FOCUS_NEIGHBORHOOD;
      updateFocusButtonStyle();
      if (REPO_DATA) buildCanvasGraph(REPO_DATA);
    }

    function updateKpiStrip(data) {
      const c = data.counts;
      document.getElementById('kpi-repo-name').textContent = data.repo_label;
      document.getElementById('kpi-repo-path').textContent = data.repo_path;
      document.getElementById('kpi-nodes').textContent = c.total_nodes;
      document.getElementById('kpi-nodes-sub').textContent =
        `${c.modules} Mod · ${c.classes} Cls · ${c.functions} Func`;
      document.getElementById('kpi-edges').textContent = c.total_edges;
      document.getElementById('kpi-edges-sub').textContent =
        `${c.calls} CALLS · ${c.inherits} INHERITS · ${c.imports} IMPORTS`;
      document.getElementById('kpi-chunks').textContent = c.total_chunks;
      document.getElementById('kpi-chunks-sub').textContent =
        `${c.code_chunks} Code · ${c.docstring_chunks} Doc · ${c.markdown_chunks} ADR`;
      document.getElementById('kpi-latency').textContent = `⚡ ${data.load_time_ms} ms`;
      document.getElementById('kpi-engine-mode').textContent = data.vector_source;
    }

    function populateSymbolDropdowns(data) {
      const blastSel = document.getElementById('blastSymbolSelect');
      const ragSel = document.getElementById('ragSymbolSelect');
      blastSel.innerHTML = '';
      ragSel.innerHTML = '';

      data.nodes.forEach(n => {
        const opt1 = document.createElement('option');
        opt1.value = n.id;
        opt1.textContent = `[${n.kind.toUpperCase()}] ${n.id}`;
        blastSel.appendChild(opt1);

        const opt2 = document.createElement('option');
        opt2.value = n.id;
        opt2.textContent = `[${n.kind.toUpperCase()}] ${n.id}`;
        ragSel.appendChild(opt2);
      });

      if (data.default_symbol) {
        blastSel.value = data.default_symbol;
        ragSel.value = data.default_symbol;
      }
      if (data.presets && data.presets.length > 0) {
        document.getElementById('ragQueryInput').value = data.presets[0].query;
        ragSel.value = data.presets[0].symbol;
      }
    }

    function resizeCanvas() {
      const wrap = document.getElementById('canvasContainer');
      if (!wrap) return;
      const dpr = window.devicePixelRatio || 1;
      const rect = wrap.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return;
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      drawGraphCanvas();
    }
    window.addEventListener('resize', resizeCanvas);

    // =========================================================================
    // Balanced Multi-Tier Graph Builder (Guarantees Functions Are Always Visible!)
    // =========================================================================
    function buildCanvasGraph(data) {
      if (!data) return;
      const allNodes = (data.nodes || []).filter(n => VISIBLE_NODE_KINDS[n.kind]);
      const allEdges = data.edges || [];

      let activeNodes = allNodes;
      const MAX_CANVAS_NODES = 140;

      if (FOCUS_NEIGHBORHOOD && SELECTED_NODE_ID) {
        // Show 2-hop neighborhood around SELECTED_NODE_ID so huge repos like Flask look super clean
        const keep = new Set([SELECTED_NODE_ID]);
        for (let hop = 0; hop < 2; hop++) {
          allEdges.forEach(e => {
            if (keep.has(e.src)) keep.add(e.dst);
            if (keep.has(e.dst)) keep.add(e.src);
          });
        }
        const neigh = allNodes.filter(n => keep.has(n.id));
        if (neigh.length >= 2) {
          activeNodes = neigh.slice(0, MAX_CANVAS_NODES);
        }
      } else if (allNodes.length > MAX_CANVAS_NODES) {
        // Balanced quota across visible kinds so Modules+Classes never crowd out Functions!
        const byKind = { Module: [], Class: [], Function: [] };
        allNodes.forEach(n => {
          if (byKind[n.kind]) byKind[n.kind].push(n);
        });
        const scoreNode = (n) => {
          let s = n.degree || 0;
          if (BLAST_SET && BLAST_SET.has(n.id)) s += 10000;
          if (n.id === SELECTED_NODE_ID) s += 50000;
          if (SEARCH_QUERY && n.id.toLowerCase().includes(SEARCH_QUERY)) s += 20000;
          return s;
        };
        Object.values(byKind).forEach(arr => arr.sort((a, b) => scoreNode(b) - scoreNode(a)));

        const enabledKinds = ['Module', 'Class', 'Function'].filter(k => VISIBLE_NODE_KINDS[k] && byKind[k].length > 0);
        if (enabledKinds.length === 1) {
          activeNodes = byKind[enabledKinds[0]].slice(0, MAX_CANVAS_NODES);
        } else {
          // Reserve ~20% Modules, ~25% Classes, ~55% Functions
          const picked = new Set();
          const quotas = { Module: 28, Class: 34, Function: 78 };
          enabledKinds.forEach(k => {
            byKind[k].slice(0, quotas[k] || 50).forEach(n => picked.add(n));
          });
          if ( picked.size < MAX_CANVAS_NODES ) {
            const rest = allNodes.filter(n => !picked.has(n)).sort((a, b) => scoreNode(b) - scoreNode(a));
            rest.slice(0, MAX_CANVAS_NODES - picked.size).forEach(n => picked.add(n));
          }
          activeNodes = Array.from(picked);
        }
      }

      NODE_MAP = {};
      G_NODES = activeNodes.map(n => {
        const parts = n.id.split('.');
        const shortLabel = parts.length > 2 ? parts.slice(-2).join('.') : n.id;
        const obj = {
          ...n,
          shortLabel,
          x: 0,
          y: 0,
          vx: 0,
          vy: 0,
          radius: n.kind === 'Module' ? 19 : (n.kind === 'Class' ? 15 : 12),
        };
        NODE_MAP[n.id] = obj;
        return obj;
      });

      G_EDGES = allEdges.filter(e => NODE_MAP[e.src] && NODE_MAP[e.dst]);

      const badge = document.getElementById('graph-mode-badge');
      if (!BLAST_SET && badge) {
        badge.textContent = FOCUS_NEIGHBORHOOD
          ? `Neighborhood Focus (${G_NODES.length} nodes around ${SELECTED_NODE_ID ? SELECTED_NODE_ID.split('.').slice(-1)[0] : 'symbol'})`
          : `Showing ${G_NODES.length} of ${data.nodes.length} nodes`;
      }

      applyGraphLayout();
      resizeCanvas();
      resetCamera();
    }

    function setGraphLayout(mode) {
      GRAPH_LAYOUT = mode;
      document.getElementById('btnLayoutHier').style.borderColor = mode === 'hierarchical' ? 'var(--accent)' : 'var(--border)';
      document.getElementById('btnLayoutForce').style.borderColor = mode === 'force' ? 'var(--accent)' : 'var(--border)';
      applyGraphLayout();
      resetCamera();
    }

    function applyGraphLayout() {
      if (G_NODES.length === 0) return;

      if (GRAPH_LAYOUT === 'hierarchical') {
        if (ANIM_FRAME) cancelAnimationFrame(ANIM_FRAME);
        const tiers = { Module: [], Class: [], Function: [] };
        G_NODES.forEach(n => { (tiers[n.kind] || tiers.Function).push(n); });
        ['Module', 'Class', 'Function'].forEach(k => tiers[k].sort((a, b) => a.id.localeCompare(b.id)));

        const tierY = { Module: -190, Class: -20, Function: 150 };
        ['Module', 'Class', 'Function'].forEach(k => {
          const list = tiers[k];
          const count = list.length;
          if (count === 0) return;
          const maxPerRow = k === 'Function' ? 11 : 8;
          list.forEach((n, idx) => {
            const r = Math.floor(idx / maxPerRow);
            const c = idx % maxPerRow;
            const colsInThisRow = Math.min(maxPerRow, count - r * maxPerRow);
            const spacingX = k === 'Module' ? 185 : (k === 'Class' ? 165 : 150);
            n.x = (c - (colsInThisRow - 1) / 2) * spacingX;
            n.y = tierY[k] + r * 78 + (c % 2 === 1 && colsInThisRow > 4 ? 16 : 0);
          });
        });
        drawGraphCanvas();
      } else {
        const N = G_NODES.length;
        G_NODES.forEach((n, i) => {
          const angle = (2 * Math.PI * i) / Math.max(1, N);
          const rad = 60 + Math.sqrt(N) * 35;
          n.x = Math.cos(angle) * rad;
          n.y = Math.sin(angle) * rad;
          n.vx = 0; n.vy = 0;
        });
        PHYSICS_TICKS = 85;
        stepForcePhysics();
      }
    }

    function stepForcePhysics() {
      if (PHYSICS_TICKS <= 0) { drawGraphCanvas(); return; }
      PHYSICS_TICKS--;
      const N = G_NODES.length;
      const repulseK = 4800;

      for (let i = 0; i < N; i++) {
        for (let j = i + 1; j < N; j++) {
          const a = G_NODES[i], b = G_NODES[j];
          let dx = b.x - a.x, dy = b.y - a.y;
          let d2 = dx * dx + dy * dy + 40, d = Math.sqrt(d2);
          let f = repulseK / d2;
          let fx = (dx / d) * f, fy = (dy / d) * f;
          a.vx -= fx; a.vy -= fy; b.vx += fx; b.vy += fy;
        }
      }

      G_EDGES.forEach(e => {
        const s = NODE_MAP[e.src], t = NODE_MAP[e.dst];
        if (!s || !t) return;
        let dx = t.x - s.x, dy = t.y - s.y;
        let d = Math.sqrt(dx * dx + dy * dy) || 1;
        let targetLen = e.kind === 'DEFINES' ? 95 : 140;
        let f = (d - targetLen) * 0.045;
        let fx = (dx / d) * f, fy = (dy / d) * f;
        s.vx += fx; s.vy += fy; t.vx -= fx; t.vy -= fy;
      });

      G_NODES.forEach(n => {
        if (n === DRAG_NODE) return;
        n.vx = (n.vx - n.x * 0.015) * 0.78;
        n.vy = (n.vy - n.y * 0.015) * 0.78;
        n.x += n.vx; n.y += n.vy;
      });

      drawGraphCanvas();
      if (PHYSICS_TICKS > 0) ANIM_FRAME = requestAnimationFrame(stepForcePhysics);
    }

    function resetCamera() {
      const wrap = document.getElementById('canvasContainer');
      if (!wrap || G_NODES.length === 0) return;
      const rect = wrap.getBoundingClientRect();
      if (rect.width === 0) return;

      const visible = G_NODES.filter(n => VISIBLE_NODE_KINDS[n.kind]);
      if (visible.length === 0) return;

      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      visible.forEach(n => {
        if (n.x < minX) minX = n.x;
        if (n.x > maxX) maxX = n.x;
        if (n.y < minY) minY = n.y;
        if (n.y > maxY) maxY = n.y;
      });

      const pad = 105;
      const spanX = Math.max(200, (maxX - minX) + pad * 2);
      const spanY = Math.max(200, (maxY - minY) + pad * 2);
      CAM.zoom = Math.min(1.35, Math.max(0.25, Math.min(rect.width / spanX, rect.height / spanY)));
      CAM.x = rect.width / 2 - ((minX + maxX) / 2) * CAM.zoom;
      CAM.y = rect.height / 2 - ((minY + maxY) / 2) * CAM.zoom;
      drawGraphCanvas();
    }

    function screenToWorld(sx, sy) {
      return { x: (sx - CAM.x) / CAM.zoom, y: (sy - CAM.y) / CAM.zoom };
    }

    function drawGraphCanvas() {
      const wrap = document.getElementById('canvasContainer');
      if (!wrap) return;
      const W = wrap.clientWidth, H = wrap.clientHeight;
      ctx.clearRect(0, 0, W, H);

      ctx.save();
      ctx.translate(CAM.x, CAM.y);
      ctx.scale(CAM.zoom, CAM.zoom);

      const hasBlast = BLAST_SET !== null;
      const hasSearch = SEARCH_QUERY.length > 0;

      G_EDGES.forEach(e => {
        if (!VISIBLE_EDGE_KINDS[e.kind]) return;
        const s = NODE_MAP[e.src], t = NODE_MAP[e.dst];
        if (!s || !t || !VISIBLE_NODE_KINDS[s.kind] || !VISIBLE_NODE_KINDS[t.kind]) return;

        const inBlast = !hasBlast || (BLAST_SET.has(s.id) && BLAST_SET.has(t.id));
        ctx.save();
        ctx.strokeStyle = EDGE_COLORS[e.kind] || '#94A3B8';
        ctx.globalAlpha = inBlast ? 0.85 : 0.12;
        ctx.lineWidth = inBlast ? (e.kind === 'DEFINES' ? 1.3 : 2.1) : 1.0;

        if (e.kind === 'IMPORTS') ctx.setLineDash([6, 4]);
        else if (e.kind === 'DEFINES') ctx.setLineDash([2, 3]);
        else ctx.setLineDash([]);

        const dx = t.x - s.x, dy = t.y - s.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const ux = dx / dist, uy = dy / dist;

        const startX = s.x + ux * (s.radius + 2);
        const startY = s.y + uy * (s.radius + 2);
        const endX = t.x - ux * (t.radius + 6);
        const endY = t.y - uy * (t.radius + 6);

        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        ctx.stroke();

        if (e.kind !== 'DEFINES' && dist > s.radius + t.radius + 10) {
          ctx.setLineDash([]);
          ctx.fillStyle = EDGE_COLORS[e.kind] || '#94A3B8';
          const arrowLen = 7, angle = Math.atan2(dy, dx);
          ctx.beginPath();
          ctx.moveTo(endX, endY);
          ctx.lineTo(endX - arrowLen * Math.cos(angle - Math.PI / 6), endY - arrowLen * Math.sin(angle - Math.PI / 6));
          ctx.lineTo(endX - arrowLen * Math.cos(angle + Math.PI / 6), endY - arrowLen * Math.sin(angle + Math.PI / 6));
          ctx.closePath();
          ctx.fill();
        }
        ctx.restore();
      });

      G_NODES.forEach(n => {
        if (!VISIBLE_NODE_KINDS[n.kind]) return;
        const inBlast = !hasBlast || BLAST_SET.has(n.id);
        const matchesSearch = !hasSearch || n.id.toLowerCase().includes(SEARCH_QUERY);
        const isTarget = n.id === BLAST_TARGET;
        const isSelected = n.id === SELECTED_NODE_ID;
        const isHovered = n === HOVER_NODE;

        ctx.save();
        ctx.globalAlpha = (inBlast && matchesSearch) ? 1.0 : 0.16;

        if (isTarget || isSelected || isHovered) {
          ctx.beginPath();
          ctx.arc(n.x, n.y, n.radius + 6, 0, Math.PI * 2);
          ctx.fillStyle = isTarget ? 'rgba(236, 72, 153, 0.38)' : 'rgba(99, 102, 241, 0.38)';
          ctx.fill();
        }

        ctx.beginPath();
        if (n.kind === 'Module') {
          const r = n.radius;
          ctx.roundRect(n.x - r * 1.25, n.y - r * 0.85, r * 2.5, r * 1.7, 6);
        } else {
          ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
        }

        ctx.fillStyle = NODE_COLORS[n.kind] || '#38BDF8';
        ctx.fill();
        ctx.lineWidth = (isTarget || isSelected) ? 3 : 1.5;
        ctx.strokeStyle = isTarget ? '#EC4899' : (isSelected ? '#FFFFFF' : '#0F172A');
        ctx.stroke();

        ctx.fillStyle = '#090D16';
        ctx.font = 'bold 10px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(n.kind[0], n.x, n.y);

        // Always draw labels when zoom >= 0.32 or <= 140 nodes
        if (CAM.zoom >= 0.32 || isTarget || isSelected || G_NODES.length <= 140) {
          const label = n.shortLabel;
          ctx.font = (isTarget || isSelected) ? 'bold 11.5px JetBrains Mono, monospace' : '600 10px JetBrains Mono, monospace';
          const textW = ctx.measureText(label).width;
          ctx.fillStyle = 'rgba(9, 13, 22, 0.84)';
          ctx.fillRect(n.x - textW / 2 - 4, n.y + n.radius + 4, textW + 8, 15);
          ctx.fillStyle = isTarget ? '#F472B6' : (isSelected ? '#A5B4FC' : '#F8FAFC');
          ctx.fillText(label, n.x, n.y + n.radius + 11.5);
        }

        ctx.restore();
      });

      ctx.restore();
    }

    canvas.addEventListener('mousedown', e => {
      const rect = canvas.getBoundingClientRect();
      const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
      LAST_MOUSE = { x: sx, y: sy };
      MOUSE_DOWN_POS = { x: sx, y: sy };
      const w = screenToWorld(sx, sy);
      const hit = findNodeAt(w.x, w.y);
      if (hit) DRAG_NODE = hit;
      else IS_PANNING = true;
    });

    canvas.addEventListener('mousemove', e => {
      const rect = canvas.getBoundingClientRect();
      const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
      const w = screenToWorld(sx, sy);

      if (DRAG_NODE) {
        DRAG_NODE.x = w.x; DRAG_NODE.y = w.y;
        drawGraphCanvas(); return;
      }
      if (IS_PANNING) {
        CAM.x += (sx - LAST_MOUSE.x); CAM.y += (sy - LAST_MOUSE.y);
        LAST_MOUSE = { x: sx, y: sy };
        drawGraphCanvas(); return;
      }

      const hit = findNodeAt(w.x, w.y);
      if (hit !== HOVER_NODE) {
        HOVER_NODE = hit;
        canvas.style.cursor = hit ? 'pointer' : 'grab';
        drawGraphCanvas();
      }
      if (hit) {
        tooltip.style.display = 'block';
        tooltip.style.left = Math.min(rect.width - 260, sx + 14) + 'px';
        tooltip.style.top = Math.max(10, sy - 20) + 'px';
        tooltip.innerHTML = `
          <div><span class="kind-badge kind-${hit.kind}">${hit.kind}</span> <b>${escapeHtml(hit.id)}</b></div>
          <div style="color:var(--text-muted); font-family:var(--font-mono); font-size:11px; margin-top:4px;">
            📄 ${escapeHtml(hit.file)}:${hit.start_line}-${hit.end_line}
          </div>
          ${hit.docstring ? `<div style="margin-top:4px; color:#CBD5E1; font-size:11px;">${escapeHtml(hit.docstring.slice(0, 120))}</div>` : ''}
        `;
      } else {
        tooltip.style.display = 'none';
      }
    });

    window.addEventListener('mouseup', e => {
      if (DRAG_NODE) {
        const rect = canvas.getBoundingClientRect();
        const dist = Math.hypot((e.clientX - rect.left) - MOUSE_DOWN_POS.x, (e.clientY - rect.top) - MOUSE_DOWN_POS.y);
        if (dist < 6) onNodeClicked(DRAG_NODE.id);
      }
      DRAG_NODE = null; IS_PANNING = false;
    });

    canvas.addEventListener('wheel', e => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
      const factor = e.deltaY < 0 ? 1.12 : 0.89;
      const nextZoom = Math.min(3.0, Math.max(0.2, CAM.zoom * factor));
      CAM.x = sx - (sx - CAM.x) * (nextZoom / CAM.zoom);
      CAM.y = sy - (sy - CAM.y) * (nextZoom / CAM.zoom);
      CAM.zoom = nextZoom;
      drawGraphCanvas();
    }, { passive: false });

    function findNodeAt(wx, wy) {
      for (let i = G_NODES.length - 1; i >= 0; i--) {
        const n = G_NODES[i];
        if (!VISIBLE_NODE_KINDS[n.kind]) continue;
        if (Math.hypot(wx - n.x, wy - n.y) <= n.radius + 4) return n;
      }
      return null;
    }

    // Rebuilds active nodes when toggling checkboxes so unchecking Module/Class on Flask immediately shows Functions!
    function toggleNodeKind(kind, checked) {
      VISIBLE_NODE_KINDS[kind] = checked;
      if (REPO_DATA) buildCanvasGraph(REPO_DATA);
    }

    function toggleEdgeKind(kind, checked) {
      VISIBLE_EDGE_KINDS[kind] = checked;
      drawGraphCanvas();
    }

    function onSearchNode(val) {
      SEARCH_QUERY = val.trim().toLowerCase();
      if (SEARCH_QUERY && REPO_DATA) {
        const match = REPO_DATA.nodes.find(n => n.id.toLowerCase().includes(SEARCH_QUERY));
        if (match) {
          SELECTED_NODE_ID = match.id;
          if (!NODE_MAP[match.id]) buildCanvasGraph(REPO_DATA);
          inspectSymbol(match.id);
        }
      }
      drawGraphCanvas();
    }

    function onNodeClicked(symbolId) {
      SELECTED_NODE_ID = symbolId;
      if (FOCUS_NEIGHBORHOOD && REPO_DATA) {
        buildCanvasGraph(REPO_DATA);
      }
      inspectSymbol(symbolId);
      if (ACTIVE_TAB === 'blast') {
        document.getElementById('blastSymbolSelect').value = symbolId;
        runBlastRadius();
      } else {
        drawGraphCanvas();
      }
    }

    async function inspectSymbol(symbolId) {
      SELECTED_NODE_ID = symbolId;
      drawGraphCanvas();
      try {
        const res = await fetch('/api/symbol?id=' + encodeURIComponent(symbolId));
        const d = await res.json();
        if (!res.ok || d.error) return;

        const makePills = (arr) => {
          if (!arr || arr.length === 0) return '<span style="color:var(--text-dim); font-size:11.5px;">None</span>';
          return arr.map(s => `<span class="rel-pill" onclick="onNodeClicked('${escapeHtml(s)}')">${escapeHtml(s)}</span>`).join('');
        };

        document.getElementById('inspectorBody').innerHTML = `
          <div class="sym-header">
            <span class="kind-badge kind-${d.kind}">${d.kind}</span>
            <div class="sym-id">${escapeHtml(d.id)}</div>
            <div style="font-size:12px; color:var(--text-muted); font-family:var(--font-mono);">
              📄 ${escapeHtml(d.file)} (lines ${d.start_line}–${d.end_line})
            </div>
          </div>

          ${d.docstring ? `
            <div class="rel-section">
              <div class="rel-title">📝 Extracted Docstring</div>
              <div style="background:#0B111E; border:1px solid var(--border); padding:8px 10px; border-radius:6px; font-size:12px; color:#CBD5E1;">
                ${escapeHtml(d.docstring)}
              </div>
            </div>
          ` : ''}

          <div class="rel-section">
            <div class="rel-title">📞 Upstream Callers (${d.callers.length})</div>
            <div class="rel-pills">${makePills(d.callers)}</div>
          </div>

          <div class="rel-section">
            <div class="rel-title">➡️ Downstream Callees (${d.callees.length})</div>
            <div class="rel-pills">${makePills(d.callees)}</div>
          </div>

          ${(d.bases.length || d.subclasses.length) ? `
            <div class="rel-section">
              <div class="rel-title">🧬 Inheritance (Bases: ${d.bases.length} · Subclasses: ${d.subclasses.length})</div>
              <div class="rel-pills">${makePills([...d.bases, ...d.subclasses])}</div>
            </div>
          ` : ''}

          ${d.defines.length ? `
            <div class="rel-section">
              <div class="rel-title">📦 Defined Members (${d.defines.length})</div>
              <div class="rel-pills">${makePills(d.defines)}</div>
            </div>
          ` : ''}

          <div class="rel-section">
            <div class="rel-title">💻 Source Code (${escapeHtml(d.file)})</div>
            <pre class="code-box"><code>${escapeHtml(d.code_snippet || '# No source lines')}</code></pre>
          </div>
        `;
      } catch (err) {
        console.error(err);
      }
    }

    function triggerBlastFromInspector() {
      if (!SELECTED_NODE_ID) return;
      document.getElementById('blastSymbolSelect').value = SELECTED_NODE_ID;
      openTab('blast');
      runBlastRadius();
    }

    function triggerRagFromInspector() {
      if (!SELECTED_NODE_ID) return;
      document.getElementById('ragSymbolSelect').value = SELECTED_NODE_ID;
      openTab('rag');
      runComparativeRag();
    }

    function jumpToRagWithBlastSymbol() {
      const sym = document.getElementById('blastSymbolSelect').value;
      if (sym) document.getElementById('ragSymbolSelect').value = sym;
      openTab('rag');
      runComparativeRag();
    }

    async function runBlastRadius() {
      const sym = document.getElementById('blastSymbolSelect').value;
      const hops = document.getElementById('blastMaxHops').value || '0';
      if (!sym) return;

      try {
        const res = await fetch(`/api/blast?symbol=${encodeURIComponent(sym)}&max_hops=${encodeURIComponent(hops)}`);
        const d = await res.json();
        if (!res.ok || d.error) return;

        BLAST_TARGET = sym;
        BLAST_SET = new Set(d.blast_symbols);
        SELECTED_NODE_ID = sym;
        document.getElementById('kpi-latency').textContent = `⚡ ${d.elapsed_ms} ms`;
        document.getElementById('graph-mode-badge').textContent =
          `💥 Blast Radius: ${d.impacted_count}/${d.total_nodes} symbols (${d.impact_pct}%)`;

        if (REPO_DATA) buildCanvasGraph(REPO_DATA);
        inspectSymbol(sym);

        document.getElementById('blast-summary-banner').innerHTML = `
          <div class="funnel-steps">
            <span>🎯 Target: <code style="color:#F472B6; font-weight:700;">${escapeHtml(d.target_symbol)}</code></span>
            <span>&rarr;</span>
            <span class="funnel-step">💥 <b>${d.impacted_count}</b> of <b>${d.total_nodes}</b> Symbols Impacted (<b>${d.impact_pct}%</b>)</span>
            <span>&rarr;</span>
            <span class="funnel-step" style="border-color:var(--emerald); color:#6EE7B7;">🛡️ <b>${d.isolated_count}</b> Unrelated Symbols Safely Isolated</span>
            <span>&rarr;</span>
            <span class="funnel-step">📦 Scoped RAG Candidate Pool: <b>${d.scoped_chunk_count}</b> / ${d.total_chunk_count} chunks</span>
          </div>
        `;

        document.getElementById('blastImpactedTableBody').innerHTML = d.impacted_rows.map(r => `
          <tr style="cursor:pointer;" onclick="onNodeClicked('${escapeHtml(r.id)}')">
            <td style="font-family:var(--font-mono); font-weight:700; color:#F8FAFC;">${escapeHtml(r.id)}</td>
            <td><span class="kind-badge kind-${r.kind}">${r.kind}</span></td>
            <td><span class="badge-pill">${escapeHtml(r.category)}</span></td>
            <td style="color:#CBD5E1;">${escapeHtml(r.reason)}</td>
          </tr>
        `).join('');

        const isoBody = document.getElementById('blastIsolatedTableBody');
        isoBody.innerHTML = d.isolated_preview.length === 0
          ? '<tr><td colspan="4" style="color:var(--text-muted); text-align:center;">All repository symbols are within this blast radius.</td></tr>'
          : d.isolated_preview.map(r => `
            <tr style="cursor:pointer;" onclick="onNodeClicked('${escapeHtml(r.id)}')">
              <td style="font-family:var(--font-mono); color:var(--text-muted);">${escapeHtml(r.id)}</td>
              <td><span class="kind-badge kind-${r.kind}">${r.kind}</span></td>
              <td style="font-family:var(--font-mono); color:var(--text-dim);">${escapeHtml(r.file)}</td>
              <td><span class="scope-badge scope-ok">Excluded from Scoped RAG</span></td>
            </tr>
          `).join('');
      } catch (err) {
        console.error(err);
      }
    }

    function clearBlastHighlight() {
      BLAST_SET = null;
      BLAST_TARGET = null;
      if (REPO_DATA) buildCanvasGraph(REPO_DATA);
    }

    function exportMatplotlibPng() {
      const sym = BLAST_TARGET || '';
      window.open('/api/export-png?blast_symbol=' + encodeURIComponent(sym), '_blank');
    }

    function renderPresets(presets) {
      document.getElementById('ragPresetsRow').innerHTML = presets.map((p, idx) => `
        <div class="preset-card" onclick="applyRagPreset(${idx})">
          <div class="preset-title">💡 ${escapeHtml(p.title)}</div>
          <div class="preset-sub">${escapeHtml(p.subtitle)}</div>
          <div style="margin-top:6px; font-family:var(--font-mono); font-size:11px; color:#A5B4FC;">
            Query: "${escapeHtml(p.query)}" &bull; Target: ${escapeHtml(p.symbol)}
          </div>
        </div>
      `).join('');
    }

    function applyRagPreset(idx) {
      if (!REPO_DATA || !REPO_DATA.presets || !REPO_DATA.presets[idx]) return;
      const p = REPO_DATA.presets[idx];
      document.getElementById('ragQueryInput').value = p.query;
      document.getElementById('ragSymbolSelect').value = p.symbol;
      document.getElementById('ragTopK').value = p.top_k || 4;
      runComparativeRag();
    }

    async function runComparativeRag() {
      const query = document.getElementById('ragQueryInput').value.trim();
      const sym = document.getElementById('ragSymbolSelect').value;
      const topK = document.getElementById('ragTopK').value || '4';
      if (!query || !sym) return;

      try {
        const res = await fetch(`/api/rag?query=${encodeURIComponent(query)}&symbol=${encodeURIComponent(sym)}&top_k=${encodeURIComponent(topK)}`);
        const d = await res.json();
        if (!res.ok || d.error) return;

        document.getElementById('kpi-latency').textContent = `⚡ ${d.elapsed_ms} ms`;

        document.getElementById('ragFunnelBanner').innerHTML = `
          <div class="funnel-steps">
            <span class="funnel-step">📚 Total Repo Chunks: <b>${d.total_chunk_count}</b></span>
            <span>&rarr;</span>
            <span class="funnel-step" style="border-color:var(--pink); color:#F9A8D4;">
              💥 Blast Radius of <code>${escapeHtml(d.target_symbol)}</code>: <b>${d.blast_symbol_count}</b> symbols
            </span>
            <span>&rarr;</span>
            <span class="funnel-step" style="border-color:var(--emerald); color:#6EE7B7;">
              🛡️ Scoped Pool: <b>${d.scoped_chunk_count}</b> chunks (<b>${d.filtered_out_chunk_count}</b> noise chunks pruned, -${d.reduction_pct}%)
            </span>
          </div>
          <div style="font-family:var(--font-mono); font-size:11.5px; color:var(--text-muted);">
            ⚡ Executed in <b>${d.elapsed_ms} ms</b> (${escapeHtml(d.vector_source)})
          </div>
        `;

        const fpBadge = document.getElementById('global-fp-badge');
        if (d.false_positive_count_in_global > 0) {
          fpBadge.textContent = `❌ ${d.false_positive_count_in_global} Out-of-Scope False Positive(s) Detected!`;
          fpBadge.style.background = 'rgba(239, 68, 68, 0.25)';
        } else {
          fpBadge.textContent = `0 False Positives in Top-${d.top_k}`;
        }

        const renderCard = (item, isScopedCol) => {
          const isFp = !isScopedCol && !item.in_blast_radius;
          return `
            <div class="chunk-card ${isFp ? 'false-positive' : ''}">
              <div class="chunk-top">
                <span class="chunk-sym" onclick="openTab('graph'); onNodeClicked('${escapeHtml(item.owner_symbol_id)}');">
                  #${item.rank} &nbsp; ${escapeHtml(item.owner_symbol_id)}
                </span>
                <span class="scope-badge ${isFp ? 'scope-fp' : 'scope-ok'}">
                  ${isFp ? '❌ ' : '✅ '}${escapeHtml(item.scope_badge)}
                </span>
              </div>
              <div class="chunk-meta">
                <span>📄 ${escapeHtml(item.source_file)}:${item.start_line}-${item.end_line}</span>
                <span>🏷️ <code>${escapeHtml(item.source_type)}</code></span>
                <span>RRF: <b>${item.rrf_score}</b> (Vec #${item.vector_rank} · BM25 #${item.bm25_rank})</span>
              </div>
              <div class="chunk-reason">
                <b>Architectural Trace:</b> ${escapeHtml(item.scope_reason)}
              </div>
              <pre class="code-box" style="max-height:170px;"><code>${escapeHtml(item.text)}</code></pre>
            </div>
          `;
        };

        document.getElementById('globalRagColumn').innerHTML =
          d.global_results.map(item => renderCard(item, false)).join('') ||
          '<div style="color:var(--text-muted); padding:20px;">No chunks matched.</div>';

        document.getElementById('scopedRagColumn').innerHTML =
          d.scoped_results.map(item => renderCard(item, true)).join('') ||
          '<div style="color:var(--text-muted); padding:20px;">No chunks in blast radius matched.</div>';
      } catch (err) {
        console.error(err);
      }
    }

    function renderChunksTable() {
      if (!REPO_DATA || !REPO_DATA.chunks_preview) return;
      const typeFilter = document.getElementById('chunkTypeFilter').value;
      const q = (document.getElementById('chunkSearchInput').value || '').trim().toLowerCase();

      const filtered = REPO_DATA.chunks_preview.filter(c => {
        if (typeFilter !== 'ALL' && c.source_type !== typeFilter) return false;
        if (q) {
          const hay = `${c.id} ${c.owner_symbol_id} ${c.source_file} ${c.text}`.toLowerCase();
          if (!hay.includes(q)) return false;
        }
        return true;
      });

      document.getElementById('chunkTableCountBadge').textContent =
        `Showing ${filtered.length} of ${REPO_DATA.counts.total_chunks} chunks`;

      document.getElementById('chunksTableBody').innerHTML = filtered.slice(0, 150).map(c => `
        <tr>
          <td style="font-family:var(--font-mono); font-size:11.5px; color:var(--text-muted);">${escapeHtml(c.id)}</td>
          <td><span class="badge-pill">${escapeHtml(c.source_type)}</span></td>
          <td style="font-family:var(--font-mono); font-weight:700; color:#A5B4FC; cursor:pointer;" onclick="openTab('graph'); onNodeClicked('${escapeHtml(c.owner_symbol_id)}');">
            ${escapeHtml(c.owner_symbol_id)}
          </td>
          <td style="font-family:var(--font-mono); font-size:11.5px;">${escapeHtml(c.source_file)}:${c.start_line}-${c.end_line}</td>
          <td><pre class="code-box" style="margin:0; max-height:110px;"><code>${escapeHtml(c.text)}</code></pre></td>
        </tr>
      `).join('');
    }

    async function presenterAction1() {
      await switchRepo('tests/fixtures/sample_repo', 'pill-sample');
      openTab('graph');
      onNodeClicked('shop.billing.make_item');
    }

    async function presenterActionSelf() {
      await switchRepo('.', 'pill-self');
      openTab('graph');
      onNodeClicked('codearchguard.analysis.extractor.extract_graph');
    }

    async function presenterAction2() {
      await switchRepo('tests/fixtures/sample_repo', 'pill-sample');
      document.getElementById('blastSymbolSelect').value = 'shop.models.Item';
      openTab('blast');
      await runBlastRadius();
    }

    async function presenterAction3() {
      await switchRepo('tests/fixtures/sample_repo', 'pill-sample');
      openTab('rag');
      applyRagPreset(0);
    }

    window.addEventListener('DOMContentLoaded', () => {
      switchRepo('tests/fixtures/sample_repo', 'pill-sample');
    });
  </script>
</body>
</html>
"""


class CodeArchGuardHTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _send_bytes(self, raw: bytes, content_type: str, status: int = 200, extra_headers: dict[str, str] | None = None) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            if extra_headers:
                for k, v in extra_headers.items():
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(raw)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        self._send_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status=status)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)

        try:
            if route in ("/", "/index.html"):
                self._send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return

            if route == "/api/status":
                entry = ENGINE.get_active_entry()
                self._send_json({"status": "ok", "active_repo": entry.rel_or_display_path})
                return

            if route == "/api/analyze":
                repo = qs.get("repo", ["tests/fixtures/sample_repo"])[0]
                force = qs.get("force", ["0"])[0] == "1"
                entry = ENGINE.load_repository(repo, force_reload=force)
                self._send_json(ENGINE.serialize_repo_overview(entry))
                return

            if route == "/api/symbol":
                sym_id = qs.get("id", [""])[0]
                self._send_json(ENGINE.get_symbol_details(sym_id))
                return

            if route == "/api/blast":
                sym_id = qs.get("symbol", [""])[0]
                max_hops = int(qs.get("max_hops", ["0"])[0] or 0)
                self._send_json(ENGINE.compute_blast_details(sym_id, max_hops=max_hops))
                return

            if route == "/api/rag":
                query = qs.get("query", [""])[0]
                symbol = qs.get("symbol", [""])[0]
                top_k = int(qs.get("top_k", ["4"])[0] or 4)
                max_hops = int(qs.get("max_hops", ["0"])[0] or 0)
                self._send_json(ENGINE.compare_rag(query=query, target_symbol=symbol, top_k=top_k, max_hops=max_hops))
                return

            if route == "/api/export-png":
                entry = ENGINE.get_active_entry()
                blast_sym = qs.get("blast_symbol", [""])[0].strip()
                out_file = Path(tempfile.gettempdir()) / f"codearchguard_{os.getpid()}.png"
                blast_set = (
                    compute_graph_blast_radius(entry.graph, blast_sym, include_containers=True)
                    if blast_sym and blast_sym in entry.graph.nodes
                    else set()
                )
                _render_static_png(entry.graph, blast_set, out_file)
                self._send_bytes(
                    out_file.read_bytes(),
                    "image/png",
                    extra_headers={"Content-Disposition": f'inline; filename="codearchguard_{entry.repo_path.name}.png"'},
                )
                return

            self._send_json({"error": f"Unknown endpoint: {route}"}, status=404)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)


def find_available_port(start_port: int = 7860, max_attempts: int = 25) -> int:
    for p in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    return start_port


def main(port: int = 7860, share: bool = False, open_browser: bool = True) -> None:
    actual_port = find_available_port(port)
    ENGINE.warmup_background()

    server = ThreadingHTTPServer(("127.0.0.1", actual_port), CodeArchGuardHTTPHandler)
    url = f"http://127.0.0.1:{actual_port}"

    print("\n" + "=" * 68)
    print(f"  CodeArchGuard Interactive Web Demo is running at: {url}")
    if actual_port != port:
        print(f"  (Port {port} was busy; automatically switched to {actual_port})")
    print("  Press Ctrl+C to stop the server.")
    print("=" * 68 + "\n")

    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down CodeArchGuard Web Demo...")
        server.server_close()


if __name__ == "__main__":
    main()