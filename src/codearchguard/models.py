"""Core graph data model shared by analysis, storage and retrieval."""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from enum import Enum


class NodeKind(str, Enum):
    MODULE = "Module"
    CLASS = "Class"
    FUNCTION = "Function"
    DOC_CHUNK = "DocChunk"


class NodeRole(str, Enum):
    """Non-exclusive behavioural roles assigned by static analysis."""

    TEST = "Test"
    ENTRY_POINT = "EntryPoint"


class EdgeKind(str, Enum):
    DEFINES = "DEFINES"      # Module/Class -> Class/Function
    IMPORTS = "IMPORTS"      # Module -> Module
    CALLS = "CALLS"          # Function -> Function
    INHERITS = "INHERITS"    # Class -> Class
    DOCUMENTED_BY = "DOCUMENTED_BY"  # Symbol -> documentation chunk


@dataclass(frozen=True)
class Node:
    id: str                  # fully qualified name, e.g. pkg.mod.Class.method
    kind: NodeKind
    name: str
    file: str                # path relative to repo root
    start_line: int = 0
    end_line: int = 0
    docstring: str = ""
    has_dynamic_warning: bool = False
    roles: frozenset[NodeRole] = field(default_factory=frozenset)
    content: str = ""         # populated only for DocChunk nodes
    owner_symbol_id: str = ""
    source_type: str = ""


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    kind: EdgeKind
    line: int = 0


@dataclass
class CodeGraph:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: set[Edge] = field(default_factory=set)
    repository_root: str = ""
    commit_hash: str | None = None
    is_dirty: bool | None = None

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        self.edges.add(edge)

    def add_role(self, node_id: str, role: NodeRole) -> None:
        """Add a role while retaining Node's immutable value semantics."""
        node = self.nodes[node_id]
        if role not in node.roles:
            self.nodes[node_id] = replace(node, roles=node.roles | {role})

    def edges_of(self, kind: EdgeKind) -> list[Edge]:
        return sorted((e for e in self.edges if e.kind == kind),
                      key=lambda e: (e.src, e.dst))
