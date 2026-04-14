# src/metrics.py
"""
Formatting and comparison helpers for benchmark results.
"""
import numpy as np
from typing import Dict, Any


def format_results_table(runs: Dict[str, Dict[str, Any]]) -> str:
    """Return a plain-text table of all run results for quick inspection."""
    all_keys = sorted({
        f"{task}/{metric}"
        for run in runs.values()
        for task, scores in run.items()
        for metric in (scores.keys() if isinstance(scores, dict) else [])
    })

    col_width = max(20, max((len(k) for k in all_keys), default=20))
    run_names = list(runs.keys())
    header = f"{'metric':<{col_width}}" + "".join(f"  {n:>12}" for n in run_names)
    sep = "-" * len(header)
    rows = [header, sep]

    for key in all_keys:
        task, metric = key.split("/", 1)
        row = f"{key:<{col_width}}"
        for name in run_names:
            val = runs[name].get(task, {}).get(metric)
            if val is None:
                row += f"  {'—':>12}"
            else:
                row += f"  {val:>12.4f}"
        rows.append(row)

    return "\n".join(rows)


def delta_vs_baseline(
    runs: Dict[str, Dict[str, Any]],
    baseline: str = "native",
) -> Dict[str, Dict[str, float]]:
    """
    Return delta of each run relative to the baseline run.
    Positive = better than baseline.
    """
    base = runs.get(baseline, {})
    deltas = {}
    for name, result in runs.items():
        if name == baseline:
            continue
        d = {}
        for task, scores in result.items():
            if not isinstance(scores, dict):
                continue
            for metric, val in scores.items():
                base_val = base.get(task, {}).get(metric)
                if base_val is not None:
                    d[f"{task}/{metric}"] = float(val) - float(base_val)
        deltas[name] = d
    return deltas


def check_success_criteria(
    runs: Dict[str, Dict[str, Any]],
    uc_key: str = "uc_256",
    pca_key: str = "pca_256",
) -> Dict[str, bool]:
    """
    Evaluate the spec's success criteria.
    Returns a dict of named checks and whether they pass.
    """
    def _get(run_name, task, metric):
        return runs.get(run_name, {}).get(task, {}).get(metric)

    checks = {}

    # 1. unit-circle beats PCA on retrieval
    uc_mrr = _get(uc_key, "retrieval", "mrr")
    pca_mrr = _get(pca_key, "retrieval", "mrr")
    checks["uc_beats_pca_retrieval_mrr"] = (
        uc_mrr is not None and pca_mrr is not None and uc_mrr > pca_mrr
    )

    # 2. unit-circle beats PCA on clustering
    uc_sil = _get(uc_key, "clustering", "silhouette")
    pca_sil = _get(pca_key, "clustering", "silhouette")
    checks["uc_beats_pca_clustering_silhouette"] = (
        uc_sil is not None and pca_sil is not None and uc_sil > pca_sil
    )

    # 3. neighbor overlap does not collapse while retrieval improves
    uc_overlap = _get(uc_key, "geometry", "neighbor_overlap@10")
    native_overlap = _get("native", "geometry", "neighbor_overlap@10")
    checks["neighbor_overlap_preserved"] = (
        uc_overlap is not None and native_overlap is not None and uc_overlap >= 0.5 * native_overlap
    )

    return checks
