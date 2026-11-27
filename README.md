# CodeArchGuard (DocHop)

### Ask Gemini about a symbol

Install the optional generation dependency with `pip install -e ".[llm]"`.
Set `GEMINI_API_KEY` in your shell, then run:

```powershell
cag ask tests/fixtures/sample_repo shop.models.DiscountedItem.total "How is the discount calculated?"
```

This sends the question, target symbol and up to five retrieved excerpts to the configured provider.
The default LiteLLM model is `gemini/gemini-flash-latest`. To switch to OpenRouter,
set `CAG_LLM_MODEL=openrouter/<provider>/<model>` and supply `OPENROUTER_API_KEY`.
The printed file locations identify the retrieved excerpts. Model-generated citations are unverified.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Architecture](https://img.shields.io/badge/architecture-GraphRAG-success.svg)](#architecture)
[![Embeddings](https://img.shields.io/badge/embeddings-BGE--small--en--v1.5%20(Local)-brightgreen.svg)](#local-embeddings--caching)
[![Retrieval](https://img.shields.io/badge/retrieval-Hybrid%20Dense%20%2B%20BM25-orange.svg)](#graph-scoped-hybrid-rag)
[![Graph DB](https://img.shields.io/badge/graph-Neo4j%20%7C%20NetworkX-purple.svg)](#graph-visualization--neo4j)
[![Tests](https://img.shields.io/badge/tests-pytest-informational.svg)](#running-tests)

**CodeArchGuard** is a **GraphRAG-powered codebase architecture and refactoring impact assistant** for multi-file Python repositories. It bridges deterministic Abstract Syntax Tree (AST) static analysis with local, zero-token hybrid semantic retrieval (Dense Vector + BM25 Sparse Search via Reciprocal Rank Fusion).

---

## 🎯 The Problem & The CodeArchGuard Solution

### The Naive Vector RAG Problem
Standard Vector RAG computes cosine similarity globally across all chunks in a repository. When a developer investigates modifying a class like `shop.models.Item`, an unrelated helper such as `shop.billing.calculate_total("Calculate total price")` matches the query semantically. This floods the LLM context with irrelevant code distractors, wastes tokens, and triggers hallucinations.

### The CodeArchGuard Formula
CodeArchGuard enforces **architecturally bounded retrieval**:
```
Target Symbol ──► Static Blast Radius ──► Subgraph Chunk Filter ──► Hybrid Retrieval
 (shop.models.Item)   (CALLS + INHERITS + DEFINES)    (Prunes Unrelated Code)   (BGE Dense + BM25 Sparse)
                                   │
                                   ▼
                   Zero Distractors • Zero Hallucinations
```

---

## 🏗️ Architecture & Data Flow

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. Zero-Token Static AST Analysis (extractor.py)                                       │
│    • 5-Level Source Root Cascade: CLI arg ➔ pyproject.toml ➔ setup.cfg ➔ setup.py ➔ src/│
│    • Pass 1: Modules, Classes, Functions & DEFINES edges                                │
│    • Pass 2a: Dotted relative/absolute cross-file IMPORTS & symbol aliases             │
│    • Pass 2b: Multi-hop MRO constructor resolution via INHERITS edges                  │
│    • Local instance method call tracking (var.method()) + Cached Jedi fallback         │
│    • Top-level script & entry-point call tracking (Module ➔ Function CALLS)            │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ produces CodeGraph (Nodes & Edges)
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. AST-Aware Non-Redundant Document Ingestion (code_ingestion.py & ingestion.py)      │
│    • Code Extraction: Classes extracted with attributes only (methods/docstrings stripped)│
│      Functions extracted with body only (docstrings stripped)                           │
│    • Doc Extraction: Module, Class, and Function docstrings                             │
│    • ADR Ingestion: Markdown Architecture Decision Records with dotted symbol linkage  │
│    • Chunking: Source-attributed chunks (start/end lines, char offsets, owner_symbol_id)│
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. Local Embedding & Persistent Fingerprint Index (llama_embedder.py & vector_index.py)│
│    • Local BAAI/bge-small-en-v1.5 (384 dims) — 100% offline, 0 LLM token cost          │
│    • SHA-256 fingerprint cache: instant reload without re-embedding unchanged repos   │
│    • VectorStoreIndex with precomputed embedding hydration (<5ms subgraph index)       │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 4. Blast Radius Boundary & Scoped Hybrid Retrieval (graph_scoped_retriever.py)        │
│    • Cycle-safe BFS over reversed CALLS, reversed INHERITS, forward DEFINES, containers │
│    • Scoped Filter: Keep only chunks owned by blast symbols (or ADRs citing them)       │
│    • Reciprocal Rank Fusion (RRF): Dense Vector + BM25 Sparse via MockLLM (0 tokens)   │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 5. Interfaces & Tooling                                                                │
│    • Interactive Web Demo UI (src/codearchguard/ui-2.py) with Vis.js network graph     │
│    • Standalone Visualizer (visualize.py) with Matplotlib annotated PNG export         │
│    • CLI Analysis (cag analyze) & Neo4j Database Ingestion                             │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Key Features

* **Deterministic Code Graph Extraction**:
  * **Nodes**: `Module` | `Class` | `Function`
  * **Edges**: `DEFINES` | `IMPORTS` | `CALLS` | `INHERITS`
  * Supports complex Python package layouts (`src/`, flat, mono-repo containers) automatically.
* **Graph-Scoped Hybrid Retrieval (GraphRAG)**:
  * Strict blast-radius subgraph filtering prevents semantic leakage from unrelated code.
  * Eliminates false positives with 100% precision on distractor benchmarks.
* **Zero Cloud LLM Token Overhead**:
  * Local embedding with `BAAI/bge-small-en-v1.5` (384 dimensions).
  * LlamaIndex `QueryFusionRetriever` with `MockLLM()` for zero-token Reciprocal Rank Fusion.
* **Instant Subgraph Hydration**:
  * Precomputed vectors are reused directly (`_hydrate_embeddings_from_index`), building blast-radius subgraph indexes on-the-fly in `<5ms`.
* **Markdown ADR & Docstring Linking**:
  * Ingests architecture documentation and markdown ADRs, automatically resolving references to code symbols.
* **Interactive Web Demo UI & Standalone Visualization**:
  * Full browser UI with interactive physics graphs, blast-radius highlights, and side-by-side Global RAG vs. Graph-Scoped RAG comparisons.
  * Headless Matplotlib PNG visualizer with terminal text call trees.
* **Neo4j Graph Database Integration**:
  * One-command ingestion into Neo4j instances with cypher-ready schemas.

---

## 📦 Installation

### Prerequisites
* Python 3.10 or higher
* [Optional] Docker (if using Neo4j database storage)

### Setup

```bash
# Clone the repository
git clone https://github.com/Eyad-Helmy/DocHop.git
cd DocHop

# Create and activate a virtual environment
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Install package with all dependencies (RAG, dev, and visualizer)
pip install -e ".[all]"
```

### Running Tests
Execute the comprehensive test suite (unit tests, static analyzer regression, and RAG retrieval tests):
```bash
pytest
```

---

## 💻 Usage Guide

### 1. Interactive Web Demo UI

Launch the built-in interactive web UI:

```bash
python src/codearchguard/ui-2.py
```

* Opens automatically at `http://127.0.0.1:7860`.
* **Features**:
  * **Interactive Vis.js Graph**: Drag, zoom, and inspect code nodes (Modules, Classes, Functions).
  * **Dynamic Blast Radius**: Select any symbol to visualize upstream callers, subclasses, and member impact in real time.
  * **Side-by-Side RAG Comparison**: Directly contrast naive Global RAG against Graph-Scoped RAG to see distractors filtered out.
  * **Symbol Inspector**: View source files, exact line numbers, and docstrings.
  * **Static PNG Export**: Generate annotated architecture diagrams directly from the browser.

---

### 2. Standalone Graph Visualizer (`visualize.py`)

No Docker, Neo4j, or web browser required. Generates high-resolution annotated PNG diagrams and terminal call trees:

```bash
# Render full graph of a repository and open diagram automatically
python visualize.py path/to/repo

# Simulate refactoring blast radius (affected callers/subclasses highlighted; others dimmed)
python visualize.py path/to/repo --blast-from shop.models.Item

# Save diagram to a custom path
python visualize.py path/to/repo --out my_architecture.png

# Terminal-only inspection (prints callers, callees, imports, inheritance without GUI)
python visualize.py path/to/repo --no-image
```

---

### 3. CLI Analysis & Neo4j Ingestion (`cag`)

The `cag` command line interface analyzes codebases and optionally loads them into a Neo4j property graph:

#### A. Repository Graph Summary
```bash
cag analyze path/to/repo
```
Outputs node counts by kind (`Module`, `Class`, `Function`) and edge counts by kind (`DEFINES`, `IMPORTS`, `CALLS`, `INHERITS`).

For repositories with custom source directories:
```bash
cag analyze path/to/repo --source-root backend
```

#### B. Ingest into Neo4j
```bash
# Start Neo4j with Docker Compose
docker compose up -d

# Ingest the analyzed graph into Neo4j
cag analyze path/to/repo --neo4j-uri bolt://localhost:7687 --user neo4j --password password
```
*Compatible with local Neo4j Desktop and Neo4j AuraDB cloud instances.*

---

### 4. Python API: GraphRAG Retrieval

You can use CodeArchGuard as a library in your own Python applications:

```python
from pathlib import Path
from codearchguard.analysis.extractor import extract_graph
from codearchguard.rag.pipeline import run_ingestion
from codearchguard.rag.llama_embedder import LlamaBGEEmbedder
from codearchguard.rag.graph_scoped_retriever import (
    compute_graph_blast_radius,
    retrieve_graph_scoped,
)

repo_path = Path("path/to/repo")

# 1. Ingest repository into documents and chunks
embedder = LlamaBGEEmbedder()
ingestion_result = run_ingestion(repo_path, embedder=embedder)

# 2. Compute blast radius for a target symbol
blast_symbols = compute_graph_blast_radius(
    ingestion_result.graph,
    start_symbol="shop.models.Item",
)
print(f"Blast radius contains {len(blast_symbols)} symbols: {blast_symbols}")

# 3. Retrieve chunks strictly scoped within the blast radius
retrieval = retrieve_graph_scoped(
    query="Where is the price initialized?",
    start_symbol="shop.models.Item",
    graph=ingestion_result.graph,
    chunks=ingestion_result.chunks,
    embedder=embedder,
    similarity_top_k=3,
)

for node_with_score in retrieval.results:
    symbol = node_with_score.node.metadata.get("owner_symbol_id")
    file_path = node_with_score.node.metadata.get("source_file")
    print(f"[{node_with_score.score:.4f}] {symbol} ({file_path}):\n{node_with_score.node.text}\n")
```

---

## 📊 Milestone Comparison & Benchmarks

| Capability / Metric | Milestone 1 | Milestone 1.5 | Milestone 2 |
| :--- | :---: | :---: | :---: |
| **System Scope** | AST Static Analysis | Hardened AST & Call Resolution | End-to-End GraphRAG & Web UI |
| **Source Root Cascade** | Direct path only | 5-tier discovery (`src/`, config) | 5-tier discovery (`src/`, config) |
| **Constructor Inheritance** | Direct only | Full MRO traversal across `INHERITS` | Full MRO traversal across `INHERITS` |
| **Local Method Calls** | Skipped | AST assignment + Jedi inference | AST assignment + Jedi inference |
| **Retrieval Engine** | None | None | Dense Vector + BM25 Sparse (RRF) |
| **Token Cost** | 0 tokens | 0 tokens | **0 LLM Tokens** (`BGE-small-en-v1.5`) |
| **Distractor Precision** | N/A | N/A | **100%** (Noise eliminated via blast radius) |
| **Document Ingestion** | None | None | Non-redundant Code, Docstrings, ADRs |
| **User Interfaces** | Matplotlib PNG / CLI | Matplotlib PNG / CLI | **Interactive Vis.js Web UI** + PNG + CLI |

---

## 📁 Repository Structure

```
DocHop/
├── src/
│   └── codearchguard/
│       ├── analysis/
│       │   └── extractor.py              # 2-pass AST graph extractor & source-root cascade
│       ├── graph/
│       │   └── neo4j_store.py            # Neo4j property graph client & schema
│       ├── rag/
│       │   ├── chunking.py               # AST & paragraph chunker with symbol attribution
│       │   ├── code_ingestion.py         # Non-redundant class/function code extraction
│       │   ├── document.py               # Document & Chunk data models
│       │   ├── graph_scoped_retriever.py # Blast-radius BFS & scoped hybrid RAG
│       │   ├── ingestion.py              # Docstring & Markdown ADR document extractor
│       │   ├── llama_bm25_retriever.py   # BM25 sparse keyword retriever
│       │   ├── llama_embedder.py         # Local BGE embedding model (zero LLM token)
│       │   ├── llama_hybrid_retriever.py # LlamaIndex QueryFusionRetriever (RRF)
│       │   ├── llama_vector_index.py     # Persistent vector index with SHA-256 fingerprinting
│       │   └── pipeline.py               # End-to-end ingestion orchestrator
│       ├── cli.py                        # `cag` command line interface
│       ├── models.py                     # Domain models (CodeGraph, Node, Edge, etc.)
│       └── ui-2.py                       # Interactive Web Demo UI (Vis.js + HTTP server)
├── docs/
│   ├── milestone1-architecture.md        # Milestone 1 design specifications
│   ├── milestone1_5_architecture.md      # Milestone 1.5 architecture hardening deep dive
│   ├── milestone2_architecture.md        # Milestone 2 GraphRAG & UI technical report
│   └── CHANGELOG.md                      # Detailed micro changelog
├── tests/
│   ├── fixtures/                         # Fixtures (sample_repo & Flask benchmark)
│   ├── conftest.py                       # Pytest session fixtures
│   ├── test_extractor.py                 # Static extractor regression tests
│   ├── test_visualize.py                 # Graph visualizer tests
│   ├── test_chunking.py                  # Chunking tests
│   ├── test_code_ingestion.py            # Code slicing tests
│   ├── test_ingestion.py                 # Docstring & ADR ingestion tests
│   ├── test_llama_embedder.py            # Local BGE embedder tests
│   ├── test_retriever.py                 # Retriever unit tests
│   └── test_graph_scoped_retriever.py    # Graph-scoped retrieval & distractor tests
├── docker-compose.yml                    # Neo4j service configuration
├── pyproject.toml                        # Package configuration & dependencies
├── requirements.txt                      # Base dependencies
└── visualize.py                          # Standalone NetworkX / Matplotlib visualizer
```

---

## 📄 License

This project is licensed under the MIT License — see the repository for details.
