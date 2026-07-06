import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL, INDEX_DIR

logger = logging.getLogger(__name__)

_model: Optional[SentenceTransformer] = None
_cache_dir = INDEX_DIR
_vectors_file = _cache_dir / "embeddings.npz"
_meta_file = _cache_dir / "meta.json"

_cache: dict[str, np.ndarray] = {}


def _load_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _load_cache() -> dict[str, np.ndarray]:
    global _cache
    if _cache:
        return _cache

    if _vectors_file.exists() and _meta_file.exists():
        try:
            data = np.load(_vectors_file)
            with open(_meta_file) as f:
                meta = json.load(f)
            for arxiv_id, idx in meta.items():
                _cache[arxiv_id] = data[f"vec_{idx}"]
            logger.info("Loaded %d cached embeddings", len(_cache))
        except Exception as e:
            logger.warning("Failed to load embedding cache: %s", e)
            _cache = {}

    return _cache


def _save_cache() -> None:
    if not _cache:
        return

    vectors = {}
    meta = {}
    for i, (arxiv_id, vec) in enumerate(_cache.items()):
        vectors[f"vec_{i}"] = vec
        meta[arxiv_id] = i

    np.savez(_vectors_file, **vectors)
    with open(_meta_file, "w") as f:
        json.dump(meta, f)


def embed_text(text: str) -> np.ndarray:
    model = _load_model()
    embedding = model.encode(text, convert_to_numpy=True)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


def get_or_compute_embedding(arxiv_id: str, text: str) -> np.ndarray:
    cache = _load_cache()

    if arxiv_id in cache:
        return cache[arxiv_id]

    vec = embed_text(text)
    cache[arxiv_id] = vec
    _save_cache()
    return vec


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    return float(np.dot(vec_a, vec_b))


def top_k_similar(query_vec: np.ndarray, ids: list[str], k: int = 5) -> list[tuple[str, float]]:
    cache = _load_cache()
    results = []

    for arxiv_id in ids:
        if arxiv_id in cache:
            sim = cosine_similarity(query_vec, cache[arxiv_id])
            results.append((arxiv_id, sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]
