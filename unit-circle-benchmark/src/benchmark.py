# src/benchmark.py
import numpy as np
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from scipy.stats import spearmanr
from typing import List, Dict, Any, Tuple, Optional


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def run_retrieval(
    query_vecs: np.ndarray,
    doc_vecs: np.ndarray,
    queries: List[Dict[str, Any]],
    corpus: List[Dict[str, Any]],
    qrels: List[Dict[str, Any]],
    k_values: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    qrel_map: Dict[str, set] = {}
    for row in qrels:
        qrel_map.setdefault(row["query_id"], set()).add(row["doc_id"])

    doc_ids = [d["id"] for d in corpus]
    # cosine similarity (vectors are L2-normalised going in)
    sims = query_vecs @ doc_vecs.T

    hits = {k: [] for k in k_values}
    reciprocal_ranks = []
    ndcg_scores = {k: [] for k in k_values}

    for i, q in enumerate(queries):
        ranked_indices = np.argsort(-sims[i])
        ranked_doc_ids = [doc_ids[j] for j in ranked_indices]
        gold = qrel_map.get(q["id"], set())

        # MRR
        first_rr = 0.0
        for rank_idx, doc_id in enumerate(ranked_doc_ids, start=1):
            if doc_id in gold:
                first_rr = 1.0 / rank_idx
                break
        reciprocal_ranks.append(first_rr)

        # Recall@k  +  nDCG@k
        for k in k_values:
            topk_ids = ranked_doc_ids[:k]
            hits[k].append(1.0 if gold & set(topk_ids) else 0.0)
            ndcg_scores[k].append(_ndcg(gold, topk_ids, k))

    result = {f"recall@{k}": float(np.mean(v)) for k, v in hits.items()}
    result["mrr"] = float(np.mean(reciprocal_ranks))
    result.update({f"ndcg@{k}": float(np.mean(v)) for k, v in ndcg_scores.items()})
    return result


def _ndcg(gold: set, ranked: List[str], k: int) -> float:
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, doc_id in enumerate(ranked[:k])
        if doc_id in gold
    )
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(gold), k)))
    return float(dcg / ideal) if ideal > 0 else 0.0


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def run_clustering(
    doc_vecs: np.ndarray,
    labels: Optional[List[Any]],
) -> Dict[str, float]:
    if labels is None or any(x is None for x in labels):
        return {}

    uniq = sorted(set(labels))
    if len(uniq) < 2:
        return {}

    label_to_int = {x: i for i, x in enumerate(uniq)}
    y = np.array([label_to_int[x] for x in labels])

    km = KMeans(n_clusters=len(uniq), n_init=10, random_state=42)
    pred = km.fit_predict(doc_vecs)

    return {
        "silhouette": float(silhouette_score(doc_vecs, pred, sample_size=min(len(doc_vecs), 5000))),
        "nmi": float(normalized_mutual_info_score(y, pred)),
        "ari": float(adjusted_rand_score(y, pred)),
    }


# ---------------------------------------------------------------------------
# Geometry / neighborhood preservation
# ---------------------------------------------------------------------------

def run_geometry_suite(
    native_vecs: np.ndarray,
    transformed_vecs: np.ndarray,
    k: int = 10,
) -> Dict[str, float]:
    n = native_vecs.shape[0]
    # Use cosine similarity (vectors should already be L2-normalised)
    native_sims = native_vecs @ native_vecs.T
    trans_sims = transformed_vecs @ transformed_vecs.T

    overlaps = []
    rank_corrs = []
    trustworthy = []
    continuous = []

    for i in range(n):
        native_order = np.argsort(-native_sims[i])
        trans_order = np.argsort(-trans_sims[i])

        # exclude self (index 0 after sort)
        n_nbrs = set(native_order[1: k + 1])
        t_nbrs = set(trans_order[1: k + 1])

        overlaps.append(len(n_nbrs & t_nbrs) / k)

        # Spearman on the transformed similarities for the native top-k neighbours
        n_nbr_list = list(n_nbrs)
        n_vals = native_sims[i][n_nbr_list]
        t_vals = trans_sims[i][n_nbr_list]
        corr = spearmanr(n_vals, t_vals).statistic
        rank_corrs.append(0.0 if (corr is None or np.isnan(corr)) else float(corr))

        # Trustworthiness contribution: penalise items in t_nbrs but not n_nbrs
        extra_in_t = t_nbrs - n_nbrs
        for j in extra_in_t:
            rank_in_native = int(np.where(native_order == j)[0][0])
            trustworthy.append(rank_in_native - k)

        # Continuity contribution: penalise items in n_nbrs but not t_nbrs
        missing_in_t = n_nbrs - t_nbrs
        for j in missing_in_t:
            rank_in_trans = int(np.where(trans_order == j)[0][0])
            continuous.append(rank_in_trans - k)

    # Normalise trustworthiness / continuity to [0,1] range (simplified)
    t_penalty = float(np.mean(trustworthy)) if trustworthy else 0.0
    c_penalty = float(np.mean(continuous)) if continuous else 0.0

    return {
        f"neighbor_overlap@{k}": float(np.mean(overlaps)),
        "spearman_on_native_neighbors": float(np.mean(rank_corrs)),
        "trustworthiness_penalty": t_penalty,
        "continuity_penalty": c_penalty,
    }
