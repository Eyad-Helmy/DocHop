from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

from codearchguard.models import CodeGraph
from codearchguard.rag.document import Document

TOP_LEVEL_SKIP_DIRS = {"venv", "node_modules", "build", "dist", "storage"}


def _should_skip_markdown_path(rel_path: Path) -> bool:
    """Skip hidden dot-directories (.git, .venv, .pytest_cache), build/storage, and test fixtures."""
    parts = rel_path.parts
    if not parts:
        return True
    if any(part.startswith(".") or part == "__pycache__" for part in parts):
        return True
    if parts[0] in TOP_LEVEL_SKIP_DIRS:
        return True
    if len(parts) >= 2 and parts[0] == "tests" and parts[1] == "fixtures":
        return True
    return False


def _infer_markdown_owner_symbol(text: str, graph: CodeGraph | None) -> str:
    """Return the most-mentioned dotted CodeGraph symbol ID in text, or '__repo__'."""
    if graph is None or not graph.nodes:
        return "__repo__"

    hits: Counter[str] = Counter()
    for node in graph.nodes.values():
        if "." not in node.id:  # skip bare 1-word top-level module names
            continue
        pattern = rf"(?<![\w.]){re.escape(node.id)}(?![\w.])"
        count = len(re.findall(pattern, text))
        if count:
            hits[node.id] = count

    if not hits:
        return "__repo__"

    # Most-mentioned symbol wins; ties broken by longest (most specific) ID
    return max(hits, key=lambda sym_id: (hits[sym_id], len(sym_id)))


def extract_markdown_documents(
    repo_root: str | Path,
    graph: CodeGraph | None = None,
) -> list[Document]:
    """Read each repository Markdown (.md) file as a single Document (chunking.py splits it later)."""
    root = Path(repo_root).resolve()
    documents: list[Document] = []

    for md_path in sorted(root.rglob("*.md")):
        rel_path = md_path.relative_to(root)
        if _should_skip_markdown_path(rel_path):
            continue

        try:
            text = md_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            continue

        if not text:
            continue

        rel_str = str(rel_path).replace("\\", "/")
        documents.append(
            Document(
                id=f"md:{rel_str}",
                owner_symbol_id=_infer_markdown_owner_symbol(text, graph),
                source_type="markdown",
                source_file=rel_str,
                start_line=1,
                end_line=len(text.splitlines()),
                text=text,
            )
        )

    return documents


def extract_documents(
    graph: CodeGraph,
    repo_root: str | Path | None = None,
) -> list[Document]:
    """Extract docstring Documents (including modules) and optional Markdown Documents."""
    documents: list[Document] = []

    for node in graph.nodes.values():
        if not node.docstring.strip():
            continue

        documents.append(
            Document(
                id=f"doc_{node.id}",
                owner_symbol_id=node.id,
                source_type="docstring",
                source_file=node.file,
                start_line=node.start_line,
                end_line=node.end_line,
                text=node.docstring.strip(),
            )
        )

    if repo_root is not None:
        documents.extend(extract_markdown_documents(repo_root, graph))

    return documents