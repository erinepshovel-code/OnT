# src/embed.py
import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer

_MODEL_CACHE = {}


def _get_model(model_name: str) -> SentenceTransformer:
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name, trust_remote_code=True)
    return _MODEL_CACHE[model_name]


def embed_texts(
    texts: List[str],
    model_name: str = "nomic-ai/nomic-embed-text-v1.5",
    prefix: Optional[str] = None,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Encode texts with an optional task-prefix (e.g. 'search_query: ').
    Returns L2-normalised float32 array of shape (N, D).
    """
    model = _get_model(model_name)
    inputs = texts if prefix is None else [f"{prefix}{t}" for t in texts]
    vecs = model.encode(
        inputs,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(vecs, dtype=np.float32)


def embed_corpus_and_queries(
    corpus_texts: List[str],
    query_texts: List[str],
    model_name: str = "nomic-ai/nomic-embed-text-v1.5",
    doc_prefix: Optional[str] = "search_document: ",
    query_prefix: Optional[str] = "search_query: ",
    batch_size: int = 64,
):
    """Convenience wrapper that respects Nomic-style task prefixes."""
    doc_vecs = embed_texts(corpus_texts, model_name, prefix=doc_prefix, batch_size=batch_size)
    q_vecs = embed_texts(query_texts, model_name, prefix=query_prefix, batch_size=batch_size)
    return doc_vecs, q_vecs
