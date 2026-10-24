# CodeArchGuard
GraphRAG-powered codebase architecture and refactoring impact assistant for multi-file Python repos.

## Milestone 1: code graph extraction
```bash
pip install -e ".[dev]"
pytest
cag analyze path/to/repo                       # prints graph stats
docker compose up -d
cag analyze path/to/repo --neo4j-uri bolt://localhost:7687
```
Graph: nodes `Module | Class | Function`; edges `DEFINES | IMPORTS | CALLS | INHERITS`.
