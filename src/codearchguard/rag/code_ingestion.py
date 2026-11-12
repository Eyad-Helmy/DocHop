
import ast
from pathlib import Path
import tokenize

from codearchguard.models import CodeGraph, NodeKind
from codearchguard.rag.document import Document



def _is_docstring(stmt):
    """Check whether a statement is a class docstring."""
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _extract_class_code(source_lines, class_node):
    """
    Extract class definition and attributes,
    excluding methods, nested classes, and class docstring.
    """

    excluded_lines = set()

    for stmt in class_node.body:

        # Exclude methods and nested classes
        should_exclude = isinstance(
            stmt,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        )

        # Exclude class docstring
        if should_exclude or _is_docstring(stmt):

            start_line = stmt.lineno

            # Include decorators in the excluded range
            if hasattr(stmt, "decorator_list") and stmt.decorator_list:
                start_line = min(
                    start_line,
                    *(decorator.lineno for decorator in stmt.decorator_list)
                )

            end_line = stmt.end_lineno or stmt.lineno

            excluded_lines.update(
                range(start_line, end_line + 1)
            )

    result = []

    for line_number in range(
        class_node.lineno,
        (class_node.end_lineno or class_node.lineno) + 1
    ):
        if line_number not in excluded_lines:
            result.append(source_lines[line_number - 1])

    return "".join(result).strip()



def _find_class_node(tree, start_line, end_line):

    for node in ast.walk(tree):

        if (
            isinstance(node, ast.ClassDef)
            and node.lineno == start_line
            and node.end_lineno == end_line
        ):
            return node

    return None




def extract_code_documents(
    graph: CodeGraph,
    repo_root: str | Path
) -> list[Document]:

    documents = []
    repo_root = Path(repo_root).resolve()
    file_cache: dict[Path, tuple[list[str], ast.Module]] = {}

    for node in graph.nodes.values():

        if node.kind not in (NodeKind.CLASS, NodeKind.FUNCTION):
            continue

        if node.start_line <= 0 or node.end_line < node.start_line:
            continue

        relative_path = Path(node.file.replace("\\", "/"))
        source_path = (repo_root / relative_path).resolve()

        if not source_path.is_file():
            raise FileNotFoundError(
                f"Source file not found for {node.id}: {source_path}"
            )

        if source_path not in file_cache:
            with tokenize.open(source_path) as file:
                source = file.read()
            file_cache[source_path] = (
                source.splitlines(keepends=True),
                ast.parse(source, filename=str(source_path)),
            )

        source_lines, tree = file_cache[source_path]

        if node.kind == NodeKind.CLASS:

            class_node = _find_class_node(
                tree,
                node.start_line,
                node.end_line
            )

            if class_node is None:
                continue

            code_text = _extract_class_code(
                source_lines,
                class_node
            )

        else:

            code_text = "".join(
                source_lines[node.start_line - 1:node.end_line]
            ).strip()

        if not code_text:
            continue

        document = Document(
            id=f"code_{node.id}",
            owner_symbol_id=node.id,
            source_type="code",
            source_file=node.file,
            start_line=node.start_line,
            end_line=node.end_line,
            text=code_text
        )

        documents.append(document)

    return documents
