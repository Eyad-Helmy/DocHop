# CodeArchGuard

GraphRAG-powered codebase architecture and refactoring impact assistant for multi-file Python repos.

---

## Milestone 1: Code Graph Extraction

Extracts an AST-based property graph from any Python repository:
* **Nodes**: `Module | Class | Function`
* **Edges**: `DEFINES | IMPORTS | CALLS | INHERITS`

---

## 1. Installation

```bash
# Clone/enter the directory
cd DocHop

# Install package in editable mode with dev & visualization support
pip install -e ".[all]"
```

Run test suite:
```bash
pytest
```

---

## 2. Graph Visualization (No Docker / Neo4j Required)

The standalone visualizer (`visualize.py`) inspects the graph and renders visual graphs using `matplotlib` and `networkx`.

### A. Render Full Graph Image
Generates an annotated, color-coded PNG diagram and opens it automatically:
```bash
python visualize.py tests/fixtures/sample_repo
# Or point to any target Python project:
python visualize.py path/to/your/repo
```

Save to a custom filename:
```bash
python visualize.py tests/fixtures/sample_repo --out my_graph.png
```

### B. Blast Radius Simulation
Highlights all direct and transitive upstream callers affected if a given symbol is modified (unaffected nodes are dimmed):
```bash
python visualize.py tests/fixtures/sample_repo --blast-from shop.models.DiscountedItem
```

### C. Terminal Text Tree Inspection
Prints callers, callees, cross-file imports, and inheritance chains directly in the terminal without generating an image:
```bash
python visualize.py tests/fixtures/sample_repo --no-image
```

---

## 3. CLI Analysis & Neo4j Database Ingestion

### A. Quick Summary CLI
Prints high-level node and edge count statistics:
```bash
cag analyze path/to/repo
```

### B. Ingest into Neo4j
If you have a running Neo4j instance:

```bash
# Start Neo4j via Docker (optional)
docker compose up -d

# Ingest the repository into Neo4j
cag analyze path/to/repo --neo4j-uri bolt://localhost:7687 --user neo4j --password password
```
*You can also connect to a local Neo4j Desktop instance or a free cloud instance on Neo4j AuraDB by passing its Bolt URI.*
