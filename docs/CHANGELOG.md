# Changelog

All micro changes to be noted for others to read should be noted here

## - 2026-11-12

- fixed an issue in extract_code_documents and added this changelog
- stripped function docstrings out of code documents during RAG extraction
- kept precomputed chunk embeddings when building the VectorStoreIndex
- moved the RAG tests into the tests/ folder

## - 2026-11-11

- pulled Ahmed Yasser's new branch to inspect features added locally before pull request acceptance
- ran pytest before any changes and 4 tests failed only due to uninstalled libraries and submodules
- all were successful after installing all stepups options and submodules