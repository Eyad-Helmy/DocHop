"""Core graph data model shared by analysis, storage and retrieval."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class NodeKind(str, Enum):
    MODULE = "Module"
    CLASS = "Class"
    FUNCTION = "Function"


class EdgeKind(str, Enum):
    DEFINES = "DEFINES"      # Module/Class -> Class/Function
    IMPORTS = "IMPORTS"      # Module -> Module
    CALLS = "CALLS"          # Function -> Function
    INHERITS = "INHERITS"    # Class -> Class


@dataclass(frozen=True)
class Node:
    id: str                  # fully qualified name, e.g. pkg.mod.Class.method
    kind: NodeKind
    name: str
    file: str                # path relative to repo root
    start_line: int = 0
    end_line: int = 0
    docstring: str = ""


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

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        self.edges.add(edge)

    def edges_of(self, kind: EdgeKind) -> list[Edge]:
        return sorted((e for e in self.edges if e.kind == kind),
                      key=lambda e: (e.src, e.dst))
