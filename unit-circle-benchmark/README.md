# Unit Circle Embedding Benchmark

This repository tests whether a unit-circle / number-base coordinate remapping can improve the usefulness of existing embedding spaces.

## Principle

We do not train a new encoder first.
We begin with a proven embedding model, then test whether a derived coordinate system improves retrieval, clustering, and neighborhood structure.

## Initial substrates

- `nomic-ai/nomic-embed-text-v1.5` — Phase 1 (Matryoshka-style dim reduction, long context)
- `BAAI/bge-m3` — Phase 2 (dense + sparse + multi-vector, multilingual)
- `text-embedding-3-large` — optional Phase 3 hosted ceiling-check

## Core question

Does the transform preserve or improve semantic structure better than:
1. native vectors
2. matched-dimension PCA
3. negative-control transforms (random rotation, broken phase)

---

## Ablation ladder

| Step | Name            | Description                                              |
|------|-----------------|----------------------------------------------------------|
| A0   | `native`        | Plain L2-normalised vectors                              |
| A1   | `pca_256/64`    | PCA to same target dim — is gain just dimensional cleanup? |
| A2   | `rot_768`       | Random orthogonal rotation — is gain just reorientation? |
| A3   | `uc_256/64`     | Unit-circle direct angle map — main hypothesis           |
| A3b  | `uc_256_rank`   | Unit-circle rank angle map                               |
| A4   | *(in notebook)* | UC → inverse PCA back to Euclidean                       |
| A5   | `uc_256_broken` | Shuffled phase — negative control / theory destroyer     |

---

## Unit-circle transform variants

**T1 — direct angle map** (`mode="direct"`)
```
theta_i = pi * x_i
output_i = [cos(theta_i), sin(theta_i)]
```

**T2 — rank angle map** (`mode="rank"`)
```
theta_i = 2*pi * rank(x_i) / (D-1)
output_i = [|x_i|*cos(theta_i), |x_i|*sin(theta_i)]
```

---

## Metrics

| Category    | Metrics                                       |
|-------------|-----------------------------------------------|
| Retrieval   | Recall@1/5/10, MRR, nDCG@10                  |
| Clustering  | Silhouette, NMI, ARI                          |
| Geometry    | Neighbor overlap@k, Spearman rank correlation, trustworthiness/continuity penalties |

---

## Quick start

```bash
pip install -r requirements.txt
python -m src.main                     # Phase 1: nomic
MODEL_PRESET=bge-m3 python -m src.main # Phase 2: BGE-M3
```

Results are written to `outputs/metrics/`.

---

## Project layout

```
unit-circle-benchmark/
├── src/
│   ├── main.py        — runner (ablation loop)
│   ├── config.py      — model presets, transform configs
│   ├── data.py        — JSONL loaders + validators
│   ├── embed.py       — SentenceTransformer wrapper
│   ├── transforms.py  — all A0–A5 transforms
│   ├── benchmark.py   — retrieval, clustering, geometry suites
│   ├── metrics.py     — table formatting, delta, success checks
│   └── utils.py       — JSON I/O, dir helpers
├── data/
│   ├── corpus.jsonl   — {id, text, label}
│   ├── queries.jsonl  — {id, text}
│   └── qrels.jsonl    — {query_id, doc_id, relevance}
├── outputs/
│   ├── metrics/       — JSON result files
│   ├── runs/          — per-run artefacts
│   └── plots/         — visualisations from notebook
└── notebooks/
    └── sanity_checks.ipynb
```

---

## Success criteria (spec)

Call it promising only if:
- gain appears on **at least two corpora or two models**
- gain **survives against PCA control**
- gain is **not confined to a single metric**
- **negative control breaks** as expected (`uc_256_broken` < `native`)

A single bump on one dataset is noise until proven otherwise.
