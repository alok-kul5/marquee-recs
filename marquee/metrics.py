"""Offline ranking metrics. All functions take ranked lists of item ids per user."""
from __future__ import annotations

import numpy as np


def recall_at_k(recs: list[int], truth: set[int], k: int) -> float:
    if not truth:
        return 0.0
    hits = sum(1 for i in recs[:k] if i in truth)
    return hits / min(len(truth), k)


def ndcg_at_k(recs: list[int], truth: set[int], k: int) -> float:
    if not truth:
        return 0.0
    dcg = sum(1.0 / np.log2(r + 2) for r, i in enumerate(recs[:k]) if i in truth)
    idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(truth), k)))
    return dcg / idcg


def ap_at_k(recs: list[int], truth: set[int], k: int) -> float:
    if not truth:
        return 0.0
    hits, score = 0, 0.0
    for r, i in enumerate(recs[:k]):
        if i in truth:
            hits += 1
            score += hits / (r + 1)
    return score / min(len(truth), k)


def evaluate(all_recs: dict[int, list[int]], truth: dict[int, set[int]], item_pop: np.ndarray,
             n_items: int, ks: tuple[int, ...] = (10, 50)) -> dict[str, float]:
    """Mean Recall/NDCG/MAP@K over users in `truth`, plus catalog coverage and novelty @10.

    item_pop: interaction counts per item in the training window (for novelty)."""
    users = [u for u in truth if u in all_recs]
    out: dict[str, float] = {"users": float(len(users))}
    for k in ks:
        out[f"recall@{k}"] = float(np.mean([recall_at_k(all_recs[u], truth[u], k) for u in users]))
        out[f"ndcg@{k}"] = float(np.mean([ndcg_at_k(all_recs[u], truth[u], k) for u in users]))
    out["map@10"] = float(np.mean([ap_at_k(all_recs[u], truth[u], 10) for u in users]))
    shown = {i for u in users for i in all_recs[u][:10]}
    out["coverage@10"] = len(shown) / n_items
    p = (item_pop + 1) / (item_pop.sum() + n_items)
    self_info = -np.log2(p)
    out["novelty@10"] = float(np.mean([self_info[all_recs[u][:10]].mean() for u in users]))
    return out
