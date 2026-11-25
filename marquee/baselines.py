"""Classic baselines: popularity, item-kNN (cosine) and implicit-feedback matrix factorisation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def interaction_matrix(df: pd.DataFrame, n_users: int, n_items: int) -> sp.csr_matrix:
    m = sp.csr_matrix((np.ones(len(df), dtype=np.float32), (df.user.to_numpy(), df.item.to_numpy())),
                      shape=(n_users, n_items))
    m.data[:] = 1.0  # binarise duplicates
    return m


def _topk(scores: np.ndarray, seen: np.ndarray | None, k: int) -> list[int]:
    s = scores.astype(np.float64, copy=True)
    if seen is not None and len(seen):
        s[seen] = -np.inf
    k = min(k, len(s))
    idx = np.argpartition(-s, k - 1)[:k]
    return idx[np.argsort(-s[idx])].tolist()


class Popularity:
    name = "Popularity"

    def fit(self, X: sp.csr_matrix, recent: sp.csr_matrix | None = None):
        # blend all-time with recent counts so the list is not frozen in the past
        self.scores = np.asarray(X.sum(0)).ravel().astype(np.float64)
        if recent is not None:
            r = np.asarray(recent.sum(0)).ravel()
            self.scores = self.scores / (self.scores.max() + 1e-9) + 2 * r / (r.max() + 1e-9)
        return self

    def score(self, users: np.ndarray, X: sp.csr_matrix) -> np.ndarray:
        return np.tile(self.scores, (len(users), 1))

    def recommend(self, u: int, X: sp.csr_matrix, k: int) -> list[int]:
        return _topk(self.scores, X[u].indices, k)


class ItemKNN:
    name = "Item-kNN"

    def __init__(self, neighbours: int = 100, shrink: float = 10.0):
        self.neighbours, self.shrink = neighbours, shrink

    def fit(self, X: sp.csr_matrix):
        Xc = X.tocsc().astype(np.float32)
        norms = np.sqrt(np.asarray(Xc.multiply(Xc).sum(0)).ravel())
        co = (Xc.T @ Xc).toarray()
        sim = co / (np.outer(norms, norms) + self.shrink + 1e-9)
        np.fill_diagonal(sim, 0.0)
        # keep top neighbours per item
        n = sim.shape[0]
        k = min(self.neighbours, n - 1)
        if k > 0:
            cut = np.partition(sim, n - k, axis=1)[:, n - k][:, None]
            sim[sim < cut] = 0.0
        self.sim = sim.astype(np.float32)
        return self

    def score(self, users: np.ndarray, X: sp.csr_matrix) -> np.ndarray:
        return np.asarray(X[users] @ self.sim)

    def score_items(self, liked: list[int]) -> np.ndarray:
        return self.sim[liked].sum(0)

    def recommend(self, u: int, X: sp.csr_matrix, k: int) -> list[int]:
        return _topk(self.score(np.array([u]), X)[0], X[u].indices, k)


class ImplicitMF:
    """Wraps the `implicit` library's ALS or BPR (CPU) with a common interface."""

    def __init__(self, kind: str = "als", factors: int = 64, iterations: int = 20,
                 regularization: float = 0.05, alpha: float = 10.0, seed: int = 0):
        self.kind, self.alpha = kind, alpha
        from implicit.als import AlternatingLeastSquares
        from implicit.bpr import BayesianPersonalizedRanking
        if kind == "als":
            self.name = "ALS (implicit)"
            self.model = AlternatingLeastSquares(factors=factors, iterations=iterations,
                                                 regularization=regularization, random_state=seed,
                                                 use_gpu=False)
        else:
            self.name = "BPR (implicit)"
            self.model = BayesianPersonalizedRanking(factors=factors, iterations=max(iterations, 100),
                                                     learning_rate=0.05, regularization=0.01,
                                                     random_state=seed, use_gpu=False)

    def fit(self, X: sp.csr_matrix):
        from threadpoolctl import threadpool_limits
        conf = X * self.alpha if self.kind == "als" else X
        with threadpool_limits(1, "blas"):
            self.model.fit(conf.tocsr().astype(np.float32), show_progress=False)
        self.U = np.asarray(self.model.user_factors)
        self.V = np.asarray(self.model.item_factors)
        return self

    def score(self, users: np.ndarray, X: sp.csr_matrix) -> np.ndarray:
        return self.U[users] @ self.V.T

    def recommend(self, u: int, X: sp.csr_matrix, k: int) -> list[int]:
        return _topk(self.U[u] @ self.V.T, X[u].indices, k)


def recommend_all(model, users: list[int], X: sp.csr_matrix, k: int, batch: int = 512) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for s in range(0, len(users), batch):
        ub = np.array(users[s:s + batch])
        S = model.score(ub, X).astype(np.float64)
        for row, u in enumerate(ub):
            out[int(u)] = _topk(S[row], X[u].indices, k)
    return out
