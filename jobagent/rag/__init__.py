"""Retrieval layer: JD chunks and embeddings in DuckDB, used for evidence-based scoring and search.

    chunking.py  split a JD (or the CV, in memory only) into sections, drop boilerplate
    embed.py     the Embedder protocol: fastembed (local, multilingual) in production, FakeEmbedder in tests
    index.py     incremental index of JD chunks and their embeddings in jobs.duckdb
"""
