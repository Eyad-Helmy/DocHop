"""Static analysis: walk a Python repo with `ast` and build a CodeGraph.

Pass 1 collects every definition (Module/Class/Function) and its imports.
Pass 2 resolves names to fully qualified ids to create IMPORTS / INHERITS /
CALLS edges. Unresolvable names are skipped (jedi refinement comes later).
"""
from __future__ import annotations
import ast
import configparser
from pathlib import Path
from typing import Iterable

from codearchguard.models import CodeGraph, Edge, EdgeKind, Node, NodeKind

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", "build", "dist", ".tox"}


def _path_from_root(root: Path, value: str) -> Path:
    # Convert a configured relative path into an absolute repository path.
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _pyproject_source_roots(root: Path) -> list[str]:
    # Read source locations declared by modern Python packaging tools.
    path = root / "pyproject.toml"
    if not path.is_file():
        return []
    try:
        try:
            import tomllib
        except ModuleNotFoundError:  # Python versions < 3.11
            # Python 3.10 needs the compatible TOML reader dependency.
            import tomli as tomllib
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []

    tool = data.get("tool", {})
    setuptools = tool.get("setuptools", {})
    # Support the two common Setuptools declarations.
    roots = list(setuptools.get("packages", {}).get("find", {}).get("where", []))
    package_dir = setuptools.get("package-dir", {})
    if isinstance(package_dir, dict) and isinstance(package_dir.get(""), str):
        roots.append(package_dir[""])
    # Poetry stores each package's source location in its ``from`` field.
    for package in tool.get("poetry", {}).get("packages", []):
        if isinstance(package, dict) and isinstance(package.get("from"), str):
            roots.append(package["from"])
    return roots


def _setup_cfg_source_roots(root: Path) -> list[str]:
    # Read source locations from legacy declarative Setuptools configuration.
    path = root / "setup.cfg"
    if not path.is_file():
        return []
    parser = configparser.ConfigParser()
    try:
        parser.read(path, encoding="utf-8")
    except (OSError, configparser.Error):
        return []

    roots: list[str] = []
    if parser.has_option("options.packages.find", "where"):
        roots.extend(parser.get("options.packages.find", "where").split())
    if parser.has_option("options", "package_dir"):
        for line in parser.get("options", "package_dir").splitlines():
            key, separator, value = line.partition("=")
            if separator and not key.strip() and value.strip():
                roots.append(value.strip())
    return roots


def _setup_py_source_roots(root: Path) -> list[str]:
    """Read literal setuptools source-root declarations without executing setup.py."""
    path = root / "setup.py"
    if not path.is_file():
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return []

    roots: list[str] = []
    # Inspect only literal arguments, never execute project setup code.
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        for keyword in call.keywords:
            if keyword.arg == "package_dir" and isinstance(keyword.value, ast.Dict):
                for key, value in zip(keyword.value.keys, keyword.value.values):
                    if (isinstance(key, ast.Constant) and key.value == "" and
                            isinstance(value, ast.Constant) and isinstance(value.value, str)):
                        roots.append(value.value)
            elif (keyword.arg == "where" and isinstance(keyword.value, ast.Constant) and
                  isinstance(keyword.value.value, str)):
                roots.append(keyword.value.value)
    return roots


def discover_source_roots(root: Path, source_roots: Iterable[str | Path] | None = None) -> list[Path]:
    """Return source roots in precedence order: explicit, config, convention."""
    # An explicit caller choice takes priority over project configuration.
    declared = list(source_roots) if source_roots is not None else (
        _pyproject_source_roots(root) or _setup_cfg_source_roots(root) or _setup_py_source_roots(root)
    )
    # ``src`` is the only naming convention used when no configuration exists.
    if not declared and (root / "src").is_dir():
        declared = ["src"]

    resolved: list[Path] = []
    for value in declared:
        candidate = _path_from_root(root, str(value))
        # Reject invalid or external roots before scanning files.
        if not candidate.is_dir():
            raise ValueError(f"Source root does not exist or is not a directory: {value}")
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Source root must be inside the repository: {value}") from error
        if candidate not in resolved:
            resolved.append(candidate)
    return resolved


def module_name(root: Path, file: Path, source_roots: Iterable[Path] = ()) -> str:
    # Use the deepest matching root; otherwise retain the full repository path.
    source_root = next(
        (candidate for candidate in sorted(source_roots, key=lambda p: len(p.parts), reverse=True)
         if file.is_relative_to(candidate)),
        root,
    )
    # Convert ``source_root/pkg/file.py`` into ``pkg.file``.
    rel = file.relative_to(source_root).with_suffix("")
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


def extract_graph(root: str | Path, source_roots: Iterable[str | Path] | None = None) -> CodeGraph:
    root = Path(root).resolve()
    # Decide import roots once, then reuse them for every scanned file.
    resolved_source_roots = discover_source_roots(root, source_roots)
    graph = CodeGraph()
    files: list[_FileInfo] = []

    # ---- Pass 1: definitions -------------------------------------------
    for path in iter_python_files(root):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        mod, rel = module_name(root, path, resolved_source_roots), str(path.relative_to(root))
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
