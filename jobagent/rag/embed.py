"""Embedders: text -> L2-normalised float32 vectors of a fixed size.

The production embedder runs locally (fastembed, ONNX, no torch, no API key, no per-call cost) with a
multilingual model, because JDs are in English and Hebrew. Anything with `embed`, `model_id` and `dim`
can replace it, e.g. an API embedder, as long as `dim` matches the FLOAT[DIM] column in db.py.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)

DIM = 384  # size of chunk_embeddings.embedding (FLOAT[384]); every embedder must produce this
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class Embedder(Protocol):
    model_id: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """(len(texts), dim) float32 array, each row L2-normalised."""
        ...


def _normalise(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(norms == 0, 1, norms)


class FastEmbedder:
    """Local ONNX model via fastembed. The model files (~220 MB) are downloaded once into `cache_dir`
    (env RAG_MODEL_CACHE; the Docker image bakes them in so a run never needs the network to score)."""

    def __init__(self, model_name: str = DEFAULT_MODEL, cache_dir: str | None = None, batch_size: int = 64):
        from fastembed import TextEmbedding  # optional dependency: requirements-rag.txt
        self.model_id = model_name
        self.batch_size = batch_size
        self._model = TextEmbedding(model_name, cache_dir=cache_dir or os.environ.get("RAG_MODEL_CACHE"))
        self.dim = int(self._model.embedding_size)
        if self.dim != DIM:
            raise ValueError(f"{model_name} produces {self.dim}-dim vectors, the index stores {DIM}")

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalise(np.stack(list(self._model.embed(list(texts), batch_size=self.batch_size))))


TOKEN = re.compile(r"[\w+#]+", re.UNICODE)


class FakeEmbedder:
    """Deterministic hashed bag-of-words vectors, for tests and the demo DB: no model, no network.
    Texts that share words get similar vectors, which is enough to test retrieval end to end."""

    def __init__(self, dim: int = DIM, model_id: str = "fake-bow-v1"):
        self.dim = dim
        self.model_id = model_id

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in TOKEN.findall((text or "").lower()):
                h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
                out[i, h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        return _normalise(out)


def get_embedder() -> Embedder | None:
    """The configured embedder (env RAG_EMBEDDER: 'fastembed', the default, or 'fake'), or None when
    fastembed is not installed, so the RAG step is skipped instead of failing the run."""
    kind = os.environ.get("RAG_EMBEDDER", "fastembed").lower()
    if kind == "fake":
        return FakeEmbedder()
    try:
        return FastEmbedder(os.environ.get("RAG_MODEL") or DEFAULT_MODEL)
    except ImportError:
        log.info("fastembed is not installed (requirements-rag.txt); skipping the RAG index")
        return None
