"""End-to-end experiment: temporal split -> fit -> rank -> evaluate -> save artifacts.

Protocol (no leakage):
  stage A  fit every candidate model on TRAIN, build ranker examples on VAL, train the ranker
           (early-stopped on a held-out slice of VAL users)
  stage B  refit candidate models on TRAIN+VAL with the same settings, score TEST with the
           stage-A ranker, and evaluate every system on TEST against the same users.
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import data as D
from .baselines import recommend_all
from .metrics import evaluate
from .pipeline import CandidateStack, Ranker, label
from .two_tower import TowerConfig

ROOT = Path(__file__).resolve().parent.parent


def first_ts(df: pd.DataFrame) -> dict[int, int]:
    return df.groupby("user").ts.min().astype(int).to_dict()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="folder with ratings.dat/users.dat/movies.dat")
    ap.add_argument("--synthetic", action="store_true", help="use a tiny generated dataset (CI)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--no-extra-window", dest="extra_window", action="store_false")
    ap.add_argument("--out", default=str(ROOT / "artifacts"))
    a = ap.parse_args(argv)

    if a.synthetic:
        root = D.make_synthetic(ROOT / "data" / "synthetic")
    else:
        root = Path(a.data) if a.data else D.fetch_ml1m()
    ds = D.load(root)
    sp = D.temporal_split(ds.ratings)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fmt = lambda t: pd.Timestamp(t, unit="s").strftime("%Y-%m-%d")
    print(f"{len(ds.ratings):,} ratings, {ds.n_users:,} users, {ds.n_items:,} items")
    print(f"train < {fmt(sp.t_val)} <= val < {fmt(sp.t_test)} <= test "
          f"({len(sp.train):,} / {len(sp.val):,} / {len(sp.test):,})")
    cfg = TowerConfig(epochs=a.epochs)
    timings = {}

    # ---------------- stage A ----------------
    t0 = time.time()
    A = CandidateStack(ds, sp.train, sp.t_val, cfg, fit_bpr=False)
    timings["stage_a_fit_s"] = time.time() - t0
    val_truth = D.targets(sp.train, sp.val)
    val_users = sorted(u for u in val_truth if u in A.hist)
    rng = np.random.default_rng(0)
    rng.shuffle(val_users)
    cut = int(len(val_users) * 0.85)
    tr_u, es_u = sorted(val_users[:cut]), sorted(val_users[cut:])
    ctx = first_ts(sp.val)
    t0 = time.time()
    tr = A.candidates(tr_u, ctx)
    es = A.candidates(es_u, ctx)
    ytr, yes = label(tr, val_truth), label(es, val_truth)
    cand_recall = ytr.sum() / sum(len(val_truth[u]) for u in tr_u)
    print(f"ranker train rows {len(tr):,} (positives {ytr.sum():,}); candidate recall {cand_recall:.3f}")
    if a.extra_window:
        # a second, earlier labelled window (all inside TRAIN) doubles the ranker's examples
        del A
        gc.collect()
        t_a = int(np.quantile(sp.train.ts, 0.875))
        early, late = sp.train[sp.train.ts < t_a], sp.train[sp.train.ts >= t_a]
        A0 = CandidateStack(ds, early, t_a, cfg, fit_bpr=False, log=lambda *_: None)
        tr0_truth = D.targets(early, late)
        u0 = sorted(u for u in tr0_truth if u in A0.hist)
        tr0 = A0.candidates(u0, first_ts(late))
        y0 = label(tr0, tr0_truth)
        tr0["user"] += 1_000_000  # separate query groups from the val window
        print(f"extra window from {fmt(t_a)}: {len(tr0):,} rows (positives {y0.sum():,})")
        tr = pd.concat([tr, tr0], ignore_index=True)
        ytr = np.concatenate([ytr, y0])
        del A0
        gc.collect()
    # drop users without a single positive candidate: they carry no ranking signal
    pos_users = set(tr.user[ytr == 1])
    keep = tr.user.isin(pos_users).to_numpy()
    tr, ytr = tr[keep].reset_index(drop=True), ytr[keep]
    ranker = Ranker().fit(tr, ytr, valid=(es.reset_index(drop=True), yes))
    timings["ranker_fit_s"] = time.time() - t0
    imp = ranker.importance()
    print("top features:", ", ".join(f"{k}" for k in imp.index[:8]))

    # free stage A before refitting (keeps peak memory low on small machines)
    del tr, es
    A = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------------- stage B ----------------
    hist_b = pd.concat([sp.train, sp.val], ignore_index=True)
    t0 = time.time()
    B = CandidateStack(ds, hist_b, sp.t_test, cfg, fit_bpr=True)
    timings["stage_b_fit_s"] = time.time() - t0
    test_truth = D.targets(hist_b, sp.test)
    warm = sorted(u for u in test_truth if u in B.hist)
    cold = sorted(u for u in test_truth if u not in B.hist)
    print(f"test users: {len(warm):,} warm, {len(cold):,} cold-start")
    K = 50
    results = {}
    for m in [B.pop, B.knn, B.als, B.bpr]:
        recs = recommend_all(m, warm, B.X, K)
        results[m.name] = evaluate(recs, test_truth, B.item_pop, ds.n_items)
        print(f"  {m.name:<18} {results[m.name]['ndcg@10']:.4f}")
    tt = B.two_tower_recs(warm, K)
    results["Two-tower (retrieval only)"] = evaluate(tt, test_truth, B.item_pop, ds.n_items)
    t0 = time.time()
    te = B.candidates(warm, first_ts(sp.test))
    yte = label(te, test_truth)
    two_stage = ranker.rank(te, K)
    timings["test_rank_s"] = time.time() - t0
    results["Two-stage (two-tower + LightGBM)"] = evaluate(two_stage, test_truth, B.item_pop, ds.n_items)
    test_cand_recall = yte.sum() / sum(len(test_truth[u]) for u in warm)
    for k, v in results.items():
        print(f"  {k:<34} R@10 {v['recall@10']:.4f}  NDCG@10 {v['ndcg@10']:.4f}  MAP@10 {v['map@10']:.4f}")

    # ---------------- cold start ----------------
    cold_res = {}
    if cold:
        pop_cold = {u: B.pop_order[:K].tolist() for u in cold}
        cold_res["Popularity"] = evaluate(pop_cold, test_truth, B.item_pop, ds.n_items)
        q = B.user_vectors(cold, [[] for _ in cold])
        _, I = B.index.search(q, K)
        cold_res["Two-tower (demographics only)"] = evaluate(
            {u: I[r].tolist() for r, u in enumerate(cold)}, test_truth, B.item_pop, ds.n_items)
        cr = B.candidates(cold, first_ts(sp.test))
        cold_res["Two-stage (demographics only)"] = evaluate(ranker.rank(cr, K), test_truth, B.item_pop, ds.n_items)
    # cold items: test positives whose item had no interactions before t_test
    cold_items = np.flatnonzero(B.item_pop == 0)
    cold_item_hits = sum(1 for u in warm for i in two_stage[u][:50] if i in test_truth[u] and B.item_pop[i] == 0)
    cold_item_pos = sum(1 for u in warm for i in test_truth[u] if B.item_pop[i] == 0)

    summary = {
        "dataset": "synthetic" if a.synthetic else "MovieLens-1M",
        "split": {"t_val": fmt(sp.t_val), "t_test": fmt(sp.t_test), "train": len(sp.train),
                  "val": len(sp.val), "test": len(sp.test)},
        "warm_users": len(warm), "cold_users": len(cold),
        "candidate_recall_val": float(cand_recall), "candidate_recall_test": float(test_cand_recall),
        "results": results, "cold_start": cold_res,
        "cold_items": {"catalog_cold": int(len(cold_items)), "test_positives": int(cold_item_pos),
                       "hits@50": int(cold_item_hits)},
        "feature_importance": {k: float(v) for k, v in imp.items()},
        "timings": timings,
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    # serving bundle = stage-B candidate stack + ranker
    with open(out / "bundle.pkl", "wb") as f:
        pickle.dump({"stack": _strip(B), "ranker": ranker}, f)
    print(f"saved {out / 'metrics.json'} and bundle.pkl")
    return summary


def _strip(stack: CandidateStack) -> CandidateStack:
    stack.tower = stack.tower.cpu()
    stack.log = print
    return stack


if __name__ == "__main__":
    main()
