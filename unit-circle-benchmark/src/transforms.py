# src/transforms.py
"""
Ablation ladder transforms.

A0  identity_transform        — native L2-normalised baseline
A1  pca_transform             — PCA control (same target dim)
A2  random_rotation_transform — orthogonal rotation control
A3  unit_circle_transform     — main hypothesis (modes: direct, rank)
A4  uc_then_inverse_transform — test whether gain survives back-projection
A5  uc_broken_phase_transform — negative control: destroy angular meaning
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


# ---------------------------------------------------------------------------
# A0 — native
# ---------------------------------------------------------------------------

def identity_transform(x: np.ndarray) -> np.ndarray:
    return l2_normalize(x)


# ---------------------------------------------------------------------------
# A1 — PCA control
# ---------------------------------------------------------------------------

def pca_transform(
    doc_vecs: np.ndarray,
    query_vecs: np.ndarray,
    out_dim: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=out_dim, random_state=42)
    d = pca.fit_transform(doc_vecs)
    q = pca.transform(query_vecs)
    return l2_normalize(d), l2_normalize(q)


# ---------------------------------------------------------------------------
# A2 — random orthogonal rotation
# ---------------------------------------------------------------------------

def random_rotation_transform(
    doc_vecs: np.ndarray,
    query_vecs: np.ndarray,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    dim = doc_vecs.shape[1]
    # Gram-Schmidt on a random matrix gives a uniformly random orthogonal matrix
    H = rng.standard_normal((dim, dim))
    Q, _ = np.linalg.qr(H)
    d = doc_vecs @ Q
    q = query_vecs @ Q
    return l2_normalize(d), l2_normalize(q)


# ---------------------------------------------------------------------------
# A3 — unit-circle transform (main hypothesis)
# ---------------------------------------------------------------------------

def unit_circle_transform(
    doc_vecs: np.ndarray,
    query_vecs: np.ndarray,
    out_dim: int = 256,
    mode: str = "direct",
) -> Tuple[np.ndarray, np.ndarray]:
    d = _uc_map(doc_vecs, out_dim=out_dim, mode=mode)
    q = _uc_map(query_vecs, out_dim=out_dim, mode=mode)
    return l2_normalize(d), l2_normalize(q)


def _uc_map(x: np.ndarray, out_dim: int = 256, mode: str = "direct") -> np.ndarray:
    # Truncate (or pad) to out_dim before mapping
    if x.shape[1] >= out_dim:
        x = x[:, :out_dim]
    else:
        pad = np.zeros((x.shape[0], out_dim - x.shape[1]), dtype=x.dtype)
        x = np.concatenate([x, pad], axis=1)

    if mode == "direct":
        # T1: theta_i = pi * x_i  (x already in [-1,1] due to L2-norm)
        theta = np.pi * np.clip(x, -1.0, 1.0)
        z = np.concatenate([np.cos(theta), np.sin(theta)], axis=1)

    elif mode == "rank":
        # T2: assign angles by coordinate rank, preserve magnitude separately
        ranks = np.argsort(np.argsort(x, axis=1), axis=1).astype(np.float32)
        theta = 2.0 * np.pi * ranks / max(1, x.shape[1] - 1)
        mag = np.abs(x)
        z = np.concatenate([mag * np.cos(theta), mag * np.sin(theta)], axis=1)

    else:
        raise ValueError(f"Unknown unit-circle mode '{mode}'. Choose 'direct' or 'rank'.")

    return z.astype(np.float32)


# ---------------------------------------------------------------------------
# A4 — unit-circle then inverse-project back to Euclidean
# ---------------------------------------------------------------------------

def uc_then_inverse_transform(
    doc_vecs: np.ndarray,
    query_vecs: np.ndarray,
    out_dim: int = 256,
    mode: str = "direct",
) -> Tuple[np.ndarray, np.ndarray]:
    """Map to unit-circle space then project back via pseudo-inverse PCA."""
    d_uc, q_uc = unit_circle_transform(doc_vecs, query_vecs, out_dim=out_dim, mode=mode)
    # Project back to original dimensionality via PCA fit on the UC-mapped docs
    pca = PCA(n_components=doc_vecs.shape[1], random_state=42)
    d = pca.fit_transform(d_uc)
    q = pca.transform(q_uc)
    return l2_normalize(d), l2_normalize(q)


# ---------------------------------------------------------------------------
# A5 — broken phase (negative control)
# ---------------------------------------------------------------------------

def uc_broken_phase_transform(
    doc_vecs: np.ndarray,
    query_vecs: np.ndarray,
    out_dim: int = 256,
    mode: str = "direct",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Unit-circle map with shuffled angular assignments — destroys the theory."""
    rng = np.random.default_rng(seed)

    def _broken(x: np.ndarray) -> np.ndarray:
        mapped = _uc_map(x, out_dim=out_dim, mode=mode)
        # Shuffle the column assignment independently per sample
        idx = rng.permutation(mapped.shape[1])
        return mapped[:, idx]

    d = _broken(doc_vecs)
    q = _broken(query_vecs)
    return l2_normalize(d), l2_normalize(q)
