# src/main.py
"""
Unit-circle embedding benchmark — main runner.

Phase 1 ablation ladder on nomic-embed-text-v1.5:
  A0  native
  A1  pca_256 / pca_64
  A3  uc_256 / uc_64          (direct angle map)
  A3b uc_256_rank / uc_64_rank (rank angle map)

Set MODEL_PRESET env var to switch to "bge-m3":
  MODEL_PRESET=bge-m3 python -m src.main
"""

import os
import numpy as np

from src.config import EMBED_PRESETS, BenchmarkConfig
from src.data import load_corpus, load_queries, load_qrels, validate_corpus, validate_queries, validate_qrels
from src.embed import embed_corpus_and_queries
from src.transforms import (
    identity_transform,
    pca_transform,
    unit_circle_transform,
    random_rotation_transform,
    uc_broken_phase_transform,
)
from src.benchmark import run_retrieval, run_clustering, run_geometry_suite
from src.metrics import format_results_table, delta_vs_baseline, check_success_criteria
from src.utils import save_json, ensure_dirs, pretty_print_results


def main():
    preset_name = os.environ.get("MODEL_PRESET", "nomic")
    cfg = BenchmarkConfig()
    embed_cfg = EMBED_PRESETS.get(preset_name)
    if embed_cfg is None:
        raise ValueError(f"Unknown MODEL_PRESET '{preset_name}'. Choose from: {list(EMBED_PRESETS)}")

    print(f"[benchmark] model preset : {preset_name}")
    print(f"[benchmark] model        : {embed_cfg.model_name}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    corpus = load_corpus(cfg.corpus_path)
    queries = load_queries(cfg.queries_path)
    qrels = load_qrels(cfg.qrels_path)

    validate_corpus(corpus)
    validate_queries(queries)
    validate_qrels(qrels, corpus, queries)

    print(f"[benchmark] corpus  : {len(corpus)} docs")
    print(f"[benchmark] queries : {len(queries)}")
    print(f"[benchmark] qrels   : {len(qrels)}")

    doc_texts = [x["text"] for x in corpus]
    query_texts = [x["text"] for x in queries]
    labels = [x.get("label") for x in corpus]

    # ------------------------------------------------------------------
    # Embed (native, once)
    # ------------------------------------------------------------------
    print("[benchmark] embedding corpus and queries …")
    docs_native, queries_native = embed_corpus_and_queries(
        doc_texts,
        query_texts,
        model_name=embed_cfg.model_name,
        doc_prefix=embed_cfg.doc_prefix,
        query_prefix=embed_cfg.query_prefix,
        batch_size=embed_cfg.batch_size,
    )
    print(f"[benchmark] native shape : {docs_native.shape}")

    # ------------------------------------------------------------------
    # Transform registry
    # ------------------------------------------------------------------
    transforms = {
        # A0
        "native": lambda d, q: (identity_transform(d), identity_transform(q)),
        # A1
        "pca_256": lambda d, q: pca_transform(d, q, out_dim=256),
        "pca_64":  lambda d, q: pca_transform(d, q, out_dim=64),
        # A2
        "rot_768": lambda d, q: random_rotation_transform(d, q),
        # A3
        "uc_256":       lambda d, q: unit_circle_transform(d, q, out_dim=256, mode="direct"),
        "uc_64":        lambda d, q: unit_circle_transform(d, q, out_dim=64,  mode="direct"),
        "uc_256_rank":  lambda d, q: unit_circle_transform(d, q, out_dim=256, mode="rank"),
        "uc_64_rank":   lambda d, q: unit_circle_transform(d, q, out_dim=64,  mode="rank"),
        # A5 — negative control
        "uc_256_broken": lambda d, q: uc_broken_phase_transform(d, q, out_dim=256, mode="direct"),
    }

    # ------------------------------------------------------------------
    # Run ablations
    # ------------------------------------------------------------------
    runs = {}
    for name, fn in transforms.items():
        print(f"[benchmark] running transform: {name}")
        d_vecs, q_vecs = fn(docs_native, queries_native)

        retrieval = run_retrieval(q_vecs, d_vecs, queries, corpus, qrels, k_values=cfg.k_values)
        clustering = run_clustering(d_vecs, labels)
        geometry = run_geometry_suite(docs_native, d_vecs, k=cfg.geometry_k)

        runs[name] = {
            "retrieval": retrieval,
            "clustering": clustering,
            "geometry": geometry,
        }
        print(f"           mrr={retrieval['mrr']:.4f}  recall@10={retrieval.get('recall@10', 0):.4f}")

    # ------------------------------------------------------------------
    # Persist and report
    # ------------------------------------------------------------------
    ensure_dirs(cfg.output_dir, f"{cfg.output_dir}/metrics", f"{cfg.output_dir}/runs")
    out_path = f"{cfg.output_dir}/metrics/{preset_name}_results.json"
    save_json(out_path, runs)
    print(f"\n[benchmark] results saved → {out_path}\n")

    pretty_print_results(runs)

    deltas = delta_vs_baseline(runs, baseline="native")
    save_json(f"{cfg.output_dir}/metrics/{preset_name}_deltas.json", deltas)

    checks = check_success_criteria(runs, uc_key="uc_256", pca_key="pca_256")
    print("\n[benchmark] success criteria:")
    for criterion, passed in checks.items():
        mark = "PASS" if passed else "FAIL"
        print(f"  {mark}  {criterion}")

    save_json(f"{cfg.output_dir}/metrics/{preset_name}_success_checks.json", checks)


if __name__ == "__main__":
    main()
