#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${1:-.}"
TARGET_DIR="${TARGET_DIR%/}"

mkdir -p "$TARGET_DIR"

if [ "$(find "$TARGET_DIR" -mindepth 1 -maxdepth 1 | head -n 1)" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "Refusing to scaffold into non-empty directory: $TARGET_DIR"
  echo "Set FORCE=1 to continue intentionally."
  exit 1
fi

mkdir -p "$TARGET_DIR"/{src,data,outputs/{runs,metrics,plots},notebooks}
cd "$TARGET_DIR"

cat > README.md <<'MD'
# Unit Circle Embedding Benchmark

This repository tests whether a unit-circle / number-base coordinate remapping can improve the usefulness of existing embedding spaces.

## Principle

We do not train a new encoder first.
We begin with a proven embedding model, then test whether a derived coordinate system improves retrieval, clustering, and neighborhood structure.
MD

cat > requirements.txt <<'REQ'
numpy
pandas
scikit-learn
scipy
sentence-transformers
transformers
torch
tqdm
orjson
REQ

cat > pyproject.toml <<'PY'
[project]
name = "unit-circle-benchmark"
version = "0.1.0"
description = "Unit-circle transform benchmark over embedding spaces"
readme = "README.md"
requires-python = ">=3.10"
dependencies = []

[tool.black]
line-length = 100
PY

cat > .replit <<'REP'
run = "python -m src.main"
language = "python3"
REP

cat > src/__init__.py <<'PY'
"""Unit-circle benchmark package."""
PY

cat > src/config.py <<'PY'
from dataclasses import dataclass


@dataclass
class EmbedPreset:
    model_name: str
    doc_prefix: str = ""
    query_prefix: str = ""
    batch_size: int = 32


EMBED_PRESETS = {
    "nomic": EmbedPreset(
        model_name="nomic-ai/nomic-embed-text-v1.5",
        doc_prefix="search_document: ",
        query_prefix="search_query: ",
    ),
    "bge-m3": EmbedPreset(
        model_name="BAAI/bge-m3",
    ),
}


@dataclass
class BenchmarkConfig:
    corpus_path: str = "data/corpus.jsonl"
    queries_path: str = "data/queries.jsonl"
    qrels_path: str = "data/qrels.jsonl"
    output_dir: str = "outputs"
    k_values: tuple[int, ...] = (1, 5, 10)
    geometry_k: int = 10
PY

cat > src/data.py <<'PY'
import orjson


def _read_jsonl(path: str):
    rows = []
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(orjson.loads(line))
    return rows


def load_corpus(path: str):
    return _read_jsonl(path)


def load_queries(path: str):
    return _read_jsonl(path)


def load_qrels(path: str):
    return _read_jsonl(path)
PY

cat > src/embed.py <<'PY'
import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_CACHE = {}


def _get_model(model_name: str):
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name, trust_remote_code=True)
    return _MODEL_CACHE[model_name]


def embed_texts(texts, model_name="nomic-ai/nomic-embed-text-v1.5", prefix=None, batch_size=32):
    model = _get_model(model_name)
    inputs = texts if prefix is None else [f"{prefix}{t}" for t in texts]
    vecs = model.encode(inputs, normalize_embeddings=True, show_progress_bar=True, batch_size=batch_size)
    return np.asarray(vecs, dtype=np.float32)


def embed_corpus_and_queries(doc_texts, query_texts, model_name, doc_prefix, query_prefix, batch_size=32):
    docs = embed_texts(doc_texts, model_name=model_name, prefix=doc_prefix, batch_size=batch_size)
    queries = embed_texts(query_texts, model_name=model_name, prefix=query_prefix, batch_size=batch_size)
    return docs, queries
PY

cat > src/transforms.py <<'PY'
import numpy as np
from sklearn.decomposition import PCA


def l2_normalize(x, eps=1e-12):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def identity_transform(x):
    return l2_normalize(x)


def pca_transform(doc_vecs, query_vecs, out_dim=256):
    pca = PCA(n_components=out_dim, random_state=42)
    d = pca.fit_transform(doc_vecs)
    q = pca.transform(query_vecs)
    return l2_normalize(d), l2_normalize(q)


def unit_circle_transform(doc_vecs, query_vecs, out_dim=256, mode="direct"):
    d = _uc_map(doc_vecs, out_dim=out_dim, mode=mode)
    q = _uc_map(query_vecs, out_dim=out_dim, mode=mode)
    return l2_normalize(d), l2_normalize(q)


def _uc_map(x, out_dim=256, mode="direct"):
    x = x[:, :out_dim]
    if mode == "direct":
        theta = np.pi * np.clip(x, -1.0, 1.0)
        z = np.concatenate([np.cos(theta), np.sin(theta)], axis=1)
    elif mode == "rank":
        ranks = np.argsort(np.argsort(x, axis=1), axis=1)
        theta = 2 * np.pi * ranks / max(1, x.shape[1] - 1)
        mag = np.abs(x)
        z = np.concatenate([mag * np.cos(theta), mag * np.sin(theta)], axis=1)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return z.astype(np.float32)
PY

cat > src/benchmark.py <<'PY'
import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score


def run_retrieval(query_vecs, doc_vecs, queries, corpus, qrels, k_values=(1, 5, 10)):
    qrel_map = {}
    for row in qrels:
        qrel_map.setdefault(row["query_id"], set()).add(row["doc_id"])

    doc_ids = [d["id"] for d in corpus]
    sims = query_vecs @ doc_vecs.T

    hits = {k: [] for k in k_values}
    reciprocal_ranks = []

    for i, q in enumerate(queries):
        ranked = np.argsort(-sims[i])
        ranked_doc_ids = [doc_ids[j] for j in ranked]
        gold = qrel_map.get(q["id"], set())

        first_rr = 0.0
        for rank_idx, doc_id in enumerate(ranked_doc_ids, start=1):
            if doc_id in gold:
                first_rr = 1.0 / rank_idx
                break
        reciprocal_ranks.append(first_rr)

        for k in k_values:
            topk = set(ranked_doc_ids[:k])
            hits[k].append(1.0 if gold & topk else 0.0)

    out = {f"recall@{k}": float(np.mean(v)) for k, v in hits.items()}
    out["mrr"] = float(np.mean(reciprocal_ranks))
    return out


def run_clustering(doc_vecs, labels):
    if labels is None or any(x is None for x in labels):
        return {}
    uniq = sorted(set(labels))
    label_to_int = {x: i for i, x in enumerate(uniq)}
    y = np.array([label_to_int[x] for x in labels])
    km = KMeans(n_clusters=len(uniq), n_init=10, random_state=42)
    pred = km.fit_predict(doc_vecs)
    return {
        "silhouette": float(silhouette_score(doc_vecs, pred)),
        "nmi": float(normalized_mutual_info_score(y, pred)),
    }


def run_geometry_suite(native_vecs, transformed_vecs, k=10):
    native_sims = native_vecs @ native_vecs.T
    trans_sims = transformed_vecs @ transformed_vecs.T

    overlaps, rank_corrs = [], []
    for i in range(native_vecs.shape[0]):
        n_rank = np.argsort(-native_sims[i])[1 : k + 1]
        t_rank = np.argsort(-trans_sims[i])[1 : k + 1]
        overlaps.append(len(set(n_rank) & set(t_rank)) / k)

        n_vals = native_sims[i][n_rank]
        t_vals = trans_sims[i][n_rank]
        corr = spearmanr(n_vals, t_vals).statistic
        rank_corrs.append(0.0 if np.isnan(corr) else corr)

    return {
        "neighbor_overlap@k": float(np.mean(overlaps)),
        "spearman_on_native_neighbors": float(np.mean(rank_corrs)),
    }
PY

cat > src/metrics.py <<'PY'
def format_results_table(runs):
    return runs
PY

cat > src/utils.py <<'PY'
import os
import orjson


def ensure_dirs(*paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)


def save_json(path: str, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
PY

cat > src/main.py <<'PY'
from src.benchmark import run_clustering, run_geometry_suite, run_retrieval
from src.config import EMBED_PRESETS, BenchmarkConfig
from src.data import load_corpus, load_qrels, load_queries
from src.embed import embed_corpus_and_queries
from src.transforms import identity_transform, pca_transform, unit_circle_transform
from src.utils import ensure_dirs, save_json


def main():
    cfg = BenchmarkConfig()
    preset = EMBED_PRESETS["nomic"]

    corpus = load_corpus(cfg.corpus_path)
    queries = load_queries(cfg.queries_path)
    qrels = load_qrels(cfg.qrels_path)

    doc_texts = [x["text"] for x in corpus]
    query_texts = [x["text"] for x in queries]
    labels = [x.get("label") for x in corpus]

    docs_native, queries_native = embed_corpus_and_queries(
        doc_texts,
        query_texts,
        model_name=preset.model_name,
        doc_prefix=preset.doc_prefix,
        query_prefix=preset.query_prefix,
        batch_size=preset.batch_size,
    )

    transforms = {
        "native": lambda d, q: (identity_transform(d), identity_transform(q)),
        "pca_256": lambda d, q: pca_transform(d, q, out_dim=256),
        "uc_256": lambda d, q: unit_circle_transform(d, q, out_dim=256, mode="direct"),
    }

    runs = {}
    for name, fn in transforms.items():
        d_vecs, q_vecs = fn(docs_native, queries_native)
        runs[name] = {
            "retrieval": run_retrieval(q_vecs, d_vecs, queries, corpus, qrels, k_values=cfg.k_values),
            "clustering": run_clustering(d_vecs, labels),
            "geometry": run_geometry_suite(docs_native, d_vecs, k=cfg.geometry_k),
        }

    ensure_dirs("outputs/metrics")
    save_json("outputs/metrics/results.json", runs)
    print("Saved outputs/metrics/results.json")


if __name__ == "__main__":
    main()
PY

cat > data/corpus.jsonl <<'JSONL'
{"id":"d1","text":"Apple pie recipe with cinnamon and butter.","label":"cooking"}
{"id":"d2","text":"Neural embeddings map text into vectors for retrieval.","label":"ml"}
{"id":"d3","text":"Sourdough starter hydration and fermentation timing.","label":"cooking"}
{"id":"d4","text":"Approximate nearest neighbors speed up semantic search.","label":"ml"}
JSONL

cat > data/queries.jsonl <<'JSONL'
{"id":"q1","text":"How do I improve sourdough fermentation?"}
{"id":"q2","text":"What helps semantic search over documents?"}
JSONL

cat > data/qrels.jsonl <<'JSONL'
{"query_id":"q1","doc_id":"d3","relevance":1}
{"query_id":"q2","doc_id":"d4","relevance":1}
JSONL

cat > notebooks/sanity_checks.ipynb <<'NB'
{
 "cells": [],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}
NB

echo "Scaffold ready at $(pwd)"
