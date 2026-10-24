"""Static analysis: walk a Python repo with `ast` and build a CodeGraph.

Pass 1 collects every definition (Module/Class/Function) and its imports.
Pass 2 resolves names to fully qualified ids to create IMPORTS / INHERITS /
CALLS edges. Unresolvable names are skipped (jedi refinement comes later).
"""
from __future__ import annotations
import ast
from pathlib import Path

from codearchguard.models import CodeGraph, Edge, EdgeKind, Node, NodeKind

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", "build", "dist", ".tox"}


def module_name(root: Path, file: Path) -> str:
    rel = file.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def iter_python_files(root: Path):
    for p in sorted(root.rglob("*.py")):
        if not (set(p.relative_to(root).parts) & SKIP_DIRS):
            yield p


class _FileInfo:
    def __init__(self, mod: str, rel: str, tree: ast.Module):
        self.mod, self.rel, self.tree = mod, rel, tree
        self.aliases: dict[str, str] = {}   # local name -> dotted target


def _resolve_relative(mod: str, is_pkg: bool, level: int, target: str | None) -> str:
    base = mod.split(".") if mod else []
    drop = level - 1 if is_pkg else level
    base = base[: len(base) - drop] if drop else base
    return ".".join(base + ([target] if target else []))


def _dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = _dotted(node.value)
        return f"{left}.{node.attr}" if left else None
    return None


def extract_graph(root: str | Path) -> CodeGraph:
    root = Path(root).resolve()
    graph = CodeGraph()
    files: list[_FileInfo] = []

    # ---- Pass 1: definitions -------------------------------------------
    for path in iter_python_files(root):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        mod, rel = module_name(root, path), str(path.relative_to(root))
        info = _FileInfo(mod, rel, tree)
        files.append(info)
        graph.add_node(Node(mod, NodeKind.MODULE, mod.split(".")[-1] or mod, rel,
                            1, len(path.read_text().splitlines()), ast.get_docstring(tree) or ""))
        _collect_defs(graph, tree.body, mod, mod, rel)

    modules = {f.mod for f in files}

    # ---- Pass 2: relationships -----------------------------------------
    for info in files:
        is_pkg = info.rel.endswith("__init__.py")
        _collect_imports(graph, info, modules, is_pkg)
        _collect_relations(graph, info)
    return graph


def _collect_defs(graph, body, parent_id, qual_prefix, rel):
    for stmt in body:
        if isinstance(stmt, ast.ClassDef):
            cid = f"{qual_prefix}.{stmt.name}"
            graph.add_node(Node(cid, NodeKind.CLASS, stmt.name, rel, stmt.lineno,
                                stmt.end_lineno or stmt.lineno, ast.get_docstring(stmt) or ""))
            graph.add_edge(Edge(parent_id, cid, EdgeKind.DEFINES, stmt.lineno))
            _collect_defs(graph, stmt.body, cid, cid, rel)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fid = f"{qual_prefix}.{stmt.name}"
            graph.add_node(Node(fid, NodeKind.FUNCTION, stmt.name, rel, stmt.lineno,
                                stmt.end_lineno or stmt.lineno, ast.get_docstring(stmt) or ""))
            graph.add_edge(Edge(parent_id, fid, EdgeKind.DEFINES, stmt.lineno))


def _collect_imports(graph, info, modules, is_pkg):
    for node in ast.walk(info.tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                info.aliases[(a.asname or a.name).split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
                _link_module(graph, info.mod, a.name, modules, node.lineno)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_relative(info.mod, is_pkg, node.level, node.module) if node.level else (node.module or "")
            for a in node.names:
                full = f"{base}.{a.name}" if base else a.name
                info.aliases[a.asname or a.name] = full
                _link_module(graph, info.mod, full if full in modules else base, modules, node.lineno)


def _link_module(graph, src, target, modules, line):
    # walk up dotted path to find the longest internal module match
    parts = target.split(".")
    while parts:
        cand = ".".join(parts)
        if cand in modules:
            if cand != src:
                graph.add_edge(Edge(src, cand, EdgeKind.IMPORTS, line))
            return
        parts.pop()


def _lookup(graph, info, dotted: str, scope_cls: str | None) -> str | None:
    head, _, rest = dotted.partition(".")
    candidates = []
    if head in ("self", "cls") and scope_cls and rest:
        candidates.append(f"{scope_cls}.{rest}")
    candidates.append(f"{info.mod}.{dotted}")
    if head in info.aliases:
        tgt = info.aliases[head]
        candidates.append(f"{tgt}.{rest}" if rest else tgt)
    return next((c for c in candidates if c in graph.nodes), None)


def _collect_relations(graph, info):
    def visit(body, prefix, scope_cls):
        for stmt in body:
            if isinstance(stmt, ast.ClassDef):
                cid = f"{prefix}.{stmt.name}"
                for base in stmt.bases:
                    d = _dotted(base)
                    t = _lookup(graph, info, d, None) if d else None
                    if t and graph.nodes[t].kind == NodeKind.CLASS:
                        graph.add_edge(Edge(cid, t, EdgeKind.INHERITS, stmt.lineno))
                visit(stmt.body, cid, cid)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fid = f"{prefix}.{stmt.name}"
                for call in (n for n in ast.walk(stmt) if isinstance(n, ast.Call)):
                    d = _dotted(call.func)
                    t = _lookup(graph, info, d, scope_cls) if d else None
                    if t and t != fid:
                        if graph.nodes[t].kind == NodeKind.CLASS:   # constructor call
                            t = f"{t}.__init__" if f"{t}.__init__" in graph.nodes else t
                        graph.add_edge(Edge(fid, t, EdgeKind.CALLS, call.lineno))
    visit(info.tree.body, info.mod, None)
