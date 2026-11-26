"""Static analysis: walk a Python repo with `ast` and build a CodeGraph.

Pass 1 collects every definition (Module/Class/Function) and its imports.
Pass 2 resolves names to fully qualified ids to create IMPORTS / INHERITS /
CALLS edges. Unresolvable names are skipped (jedi refinement comes later).
"""
from __future__ import annotations
import ast
import configparser
from pathlib import Path
import subprocess
from typing import Iterable
import builtins

from codearchguard.models import CodeGraph, Edge, EdgeKind, Node, NodeKind, NodeRole

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", "build", "dist", ".tox"}
_BUILTIN_NAMES = set(dir(builtins))


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
    # Inspect only literal arguments to not execute project setup code.
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
    def __init__(self, mod: str, rel: str, path: Path, source: str, tree: ast.Module):
        self.mod, self.rel, self.path, self.source, self.tree = mod, rel, path, source, tree
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


def _git_snapshot(root: Path) -> tuple[str | None, bool | None]:
    """Return the checked-out commit and dirty state, or nulls outside Git."""
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    return head, bool(status.strip())


def _import_aliases(tree: ast.Module) -> dict[str, str]:
    """Return static import aliases needed by dynamic-call detection."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module == "importlib":
            for alias in node.names:
                if alias.name == "import_module":
                    aliases[alias.asname or alias.name] = "importlib.import_module"
    return aliases


def _has_dynamic_construct(body: list[ast.stmt], aliases: dict[str, str]) -> bool:
    """Detect calls whose target or member name cannot be known statically.

    Nested definitions are intentionally excluded: each collected symbol receives
    its own warning rather than warning every enclosing class or function.
    """
    dynamic_names = {"eval", "exec", "__import__", "importlib.import_module"}

    class Visitor(ast.NodeVisitor):
        found = False

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return

        def visit_Call(self, node: ast.Call) -> None:
            dotted = _dotted(node.func)
            if dotted:
                head, dot, rest = dotted.partition(".")
                resolved = f"{aliases[head]}.{rest}" if dot and head in aliases else aliases.get(head, dotted)
                if resolved in dynamic_names:
                    self.found = True
                elif resolved == "getattr":
                    # A literal member remains statically understandable.
                    if len(node.args) < 2 or not isinstance(node.args[1], ast.Constant) or not isinstance(node.args[1].value, str):
                        self.found = True
            self.generic_visit(node)

    visitor = Visitor()
    for statement in body:
        visitor.visit(statement)
    return visitor.found


def _is_main_guard(node: ast.If) -> bool:
    test = node.test
    return (
        isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.left, ast.Name) and test.left.id == "__name__"
        and isinstance(test.comparators[0], ast.Constant) and test.comparators[0].value == "__main__"
    )


def _is_test_file(rel: str) -> bool:
    name = Path(rel).name
    return name.startswith("test_") or name.endswith("_test.py")


def extract_graph(root: str | Path, source_roots: Iterable[str | Path] | None = None) -> CodeGraph:
    root = Path(root).resolve()
    # Decide import roots once, then reuse them for every scanned file.
    resolved_source_roots = discover_source_roots(root, source_roots)
    commit_hash, is_dirty = _git_snapshot(root)
    graph = CodeGraph(
        repository_root=str(root),
        commit_hash=commit_hash,
        is_dirty=is_dirty,
    )
    files: list[_FileInfo] = []

    # ---- Pass 1: definitions -------------------------------------------
    for path in iter_python_files(root):
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        mod, rel = module_name(root, path, resolved_source_roots), str(path.relative_to(root))
        info = _FileInfo(mod, rel, path, source, tree)
        dynamic_aliases = _import_aliases(tree)
        files.append(info)
        graph.add_node(Node(mod, NodeKind.MODULE, mod.split(".")[-1] or mod, rel,
                            1, len(source.splitlines()), ast.get_docstring(tree) or "",
                            _has_dynamic_construct(tree.body, dynamic_aliases)))
        if _is_test_file(rel):
            graph.add_role(mod, NodeRole.TEST)
        _collect_defs(graph, tree.body, mod, mod, rel, dynamic_aliases)

    modules = {f.mod for f in files}

    # ---- Pass 2: imports, then relationships ---------------------------
    # Resolve every import before collecting calls and inheritance so a
    # constructor lookup can follow bases declared in any source file.
    for info in files:
        is_pkg = info.rel.endswith("__init__.py")
        _collect_imports(graph, info, modules, is_pkg)
    for info in files:
        _collect_relations(graph, info, collect_inheritance=True, collect_calls=False)
    for info in files:
        _collect_relations(graph, info, collect_inheritance=False, collect_calls=True)
        _mark_entry_points(graph, info)
    return graph


def _collect_defs(graph, body, parent_id, qual_prefix, rel, dynamic_aliases):
    for stmt in body:
        if isinstance(stmt, ast.ClassDef):
            cid = f"{qual_prefix}.{stmt.name}"
            graph.add_node(Node(cid, NodeKind.CLASS, stmt.name, rel, stmt.lineno,
                                stmt.end_lineno or stmt.lineno, ast.get_docstring(stmt) or "",
                                _has_dynamic_construct(stmt.body, dynamic_aliases)))
            graph.add_edge(Edge(parent_id, cid, EdgeKind.DEFINES, stmt.lineno))
            if stmt.name.startswith("Test"):
                graph.add_role(cid, NodeRole.TEST)
            _collect_defs(graph, stmt.body, cid, cid, rel, dynamic_aliases)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fid = f"{qual_prefix}.{stmt.name}"
            graph.add_node(Node(fid, NodeKind.FUNCTION, stmt.name, rel, stmt.lineno,
                                stmt.end_lineno or stmt.lineno, ast.get_docstring(stmt) or "",
                                _has_dynamic_construct(stmt.body, dynamic_aliases)))
            graph.add_edge(Edge(parent_id, fid, EdgeKind.DEFINES, stmt.lineno))
            if stmt.name.startswith("test_"):
                graph.add_role(fid, NodeRole.TEST)


def _mark_entry_points(graph: CodeGraph, info: _FileInfo) -> None:
    """Mark main-guard modules and statically resolved functions they invoke."""
    for statement in info.tree.body:
        if not isinstance(statement, ast.If) or not _is_main_guard(statement):
            continue
        graph.add_role(info.mod, NodeRole.ENTRY_POINT)
        for call in (node for node in ast.walk(statement) if isinstance(node, ast.Call)):
            dotted = _dotted(call.func)
            target = _lookup(graph, info, dotted, None, call.func) if dotted else None
            if target:
                graph.add_role(target, NodeRole.ENTRY_POINT)


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


def _lookup(graph, info, dotted: str, scope_cls: str | None, node: ast.AST | None = None,
            local_types: dict[str, str] | None = None) -> str | None:
    head, _, rest = dotted.partition(".")
    candidates = []
    if head in ("self", "cls") and scope_cls and rest:
        candidates.append(f"{scope_cls}.{rest}")
    if local_types and head in local_types and rest:
        candidates.append(f"{local_types[head]}.{rest}")
    candidates.append(f"{info.mod}.{dotted}")
    if head in info.aliases:
        tgt = info.aliases[head]
        candidates.append(f"{tgt}.{rest}" if rest else tgt)
    resolved = next((c for c in candidates if c in graph.nodes), None)
    if resolved or node is None:
        return resolved

    # Skip Jedi if `head` is a Python builtin or neither a local/alias/same-module name
    if head in _BUILTIN_NAMES:
        return None
    if head not in info.aliases and (not local_types or head not in local_types) and head not in ("self", "cls"):
        if f"{info.mod}.{head}" not in graph.nodes:
            return None

    # Jedi can resolve calls that the lightweight AST/name rules cannot.
    try:
        import jedi
        if getattr(info, "_jedi_script", None) is None:
            info._jedi_script = jedi.Script(code=info.source, path=str(info.path))
        inferred = info._jedi_script.infer(
            line=node.end_lineno or node.lineno,
            column=node.end_col_offset if node.end_col_offset is not None else node.col_offset,
        )
    except Exception:
        return None

    for definition in inferred:
        full_name = definition.full_name
        if full_name in graph.nodes:
            return full_name
    return None


def _collect_relations(graph, info, collect_inheritance=True, collect_calls=True):
    def visit(body, prefix, scope_cls):
        for stmt in body:
            if isinstance(stmt, ast.ClassDef):
                cid = f"{prefix}.{stmt.name}"
                for base in stmt.bases:
                    d = _dotted(base)
                    t = _lookup(graph, info, d, None) if d else None
                    if collect_inheritance and t and graph.nodes[t].kind == NodeKind.CLASS:
                        graph.add_edge(Edge(cid, t, EdgeKind.INHERITS, stmt.lineno))
                visit(stmt.body, cid, cid)
            elif collect_calls and isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fid = f"{prefix}.{stmt.name}"
                local_types = {}
                for assigned in (n for n in ast.walk(stmt) if isinstance(n, (ast.Assign, ast.AnnAssign))):
                    value = assigned.value
                    if not isinstance(value, ast.Call):
                        continue
                    target = _lookup(graph, info, _dotted(value.func) or "", scope_cls)
                    if not target or graph.nodes[target].kind != NodeKind.CLASS:
                        continue
                    targets = assigned.targets if isinstance(assigned, ast.Assign) else [assigned.target]
                    for name in targets:
                        if isinstance(name, ast.Name):
                            local_types[name.id] = target
                for call in (n for n in ast.walk(stmt) if isinstance(n, ast.Call)):
                    d = _dotted(call.func)
                    t = _lookup(graph, info, d, scope_cls, call.func, local_types) if d else None
                    if t and t != fid:
                        if graph.nodes[t].kind == NodeKind.CLASS:   # constructor call
                            constructor = t
                            visited = set()
                            while constructor and constructor not in visited:
                                visited.add(constructor)
                                initializer = f"{constructor}.__init__"
                                if initializer in graph.nodes:
                                    t = initializer
                                    break
                                constructor = next(
                                    (edge.dst for edge in graph.edges_of(EdgeKind.INHERITS)
                                     if edge.src == constructor),
                                    None,
                                )
                        graph.add_edge(Edge(fid, t, EdgeKind.CALLS, call.lineno))
    visit(info.tree.body, info.mod, None)

    if collect_calls:
        for stmt in info.tree.body:
            if isinstance(stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for call in (n for n in ast.walk(stmt) if isinstance(n, ast.Call)):
                d = _dotted(call.func)
                target = _lookup(graph, info, d, None, call.func) if d else None
                if target and target != info.mod:
                    if graph.nodes[target].kind == NodeKind.CLASS:
                        constructor = target
                        visited = set()
                        while constructor and constructor not in visited:
                            visited.add(constructor)
                            initializer = f"{constructor}.__init__"
                            if initializer in graph.nodes:
                                target = initializer
                                break
                            constructor = next(
                                (edge.dst for edge in graph.edges_of(EdgeKind.INHERITS)
                                 if edge.src == constructor),
                                None,
                            )
                    graph.add_edge(Edge(info.mod, target, EdgeKind.CALLS, call.lineno))
