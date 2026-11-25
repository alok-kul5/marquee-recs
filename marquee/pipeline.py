"""The two-stage system: multi-source candidate generation followed by a LightGBM LambdaRank ranker."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .baselines import ImplicitMF, ItemKNN, Popularity, interaction_matrix
from .data import Dataset
from .two_tower import (AnnIndex, Featurizer, TowerConfig, item_embeddings, train_two_tower,
                        user_embeddings)

DAY = 86_400
N_TT, N_ALS, N_POP = 150, 50, 20

FEATURES = [
    "tt_score", "tt_rank", "als_score", "knn_score", "src_tt", "src_als", "src_pop",
    "item_pop_log", "item_recent_log", "item_avg_rating", "item_n_ratings_log", "item_age_years",
    "item_days_on_site", "user_n_log", "user_avg_rating", "user_days_since_last", "user_gender",
    "user_age", "user_occ", "genre_affinity", "year_gap", "ctx_hour", "ctx_weekday",
]


class CandidateStack:
    """Everything fit on one history window: baselines, two-tower, ANN index and feature stats."""

    def __init__(self, ds: Dataset, history: pd.DataFrame, cutoff: int,
                 tower_cfg: TowerConfig = TowerConfig(), log=print, fit_bpr: bool = True):
        self.ds, self.history, self.cutoff, self.log = ds, history, cutoff, log
        n_u, n_i = ds.n_users, ds.n_items
        self.X = interaction_matrix(history, n_u, n_i)
        recent = history[history.ts >= cutoff - 30 * DAY]
        Xr = interaction_matrix(recent, n_u, n_i)
        log("fitting popularity / item-kNN / ALS / BPR")
        self.pop = Popularity().fit(self.X, Xr)
        self.knn = ItemKNN().fit(self.X)
        self.als = ImplicitMF("als").fit(self.X)
        self.bpr = ImplicitMF("bpr").fit(self.X) if fit_bpr else None

        known = np.asarray(self.X.sum(0)).ravel() > 0
        self.feat = Featurizer(ds.items, ds.users, known_items=known)
        log("training two-tower")
        self.tower = train_two_tower(history, self.feat, tower_cfg, log=log)
        self.item_vecs = item_embeddings(self.tower, self.feat)
        self.index = AnnIndex(self.item_vecs)

        # per-user history lists in time order
        h = history.sort_values(["user", "ts"], kind="stable")
        self.hist = {int(u): g.item.tolist() for u, g in h.groupby("user")}
        self.user_last = h.groupby("user").ts.max()
        self.user_mean_rating = h.groupby("user").rating.mean()
        # item stats
        cnt = history.groupby("item").rating.agg(["count", "mean"])
        gm = history.rating.mean()
        item_n = np.zeros(n_i)
        item_n[cnt.index] = cnt["count"]
        item_mean = np.full(n_i, gm)
        item_mean[cnt.index] = cnt["mean"]
        self.item_avg = (item_n * item_mean + 20 * gm) / (item_n + 20)
        self.item_n = item_n
        self.item_pop = np.asarray(self.X.sum(0)).ravel()
        self.item_recent = np.asarray(Xr.sum(0)).ravel()
        first = history.groupby("item").ts.min()
        self.item_first = np.full(n_i, np.nan)
        self.item_first[first.index] = first.to_numpy()
        self.item_year = ds.items.year.to_numpy()
        self.genres = np.stack(ds.items.genre_vec.to_numpy())
        self.pop_order = np.argsort(-self.pop.scores)

    # FAISS indexes are not picklable: drop it and rebuild from the vectors on load
    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop("index", None)
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.index = AnnIndex(self.item_vecs)

    # ---- user representation ------------------------------------------------------------
    def user_vectors(self, users: list[int], hists: list[list[int]] | None = None) -> np.ndarray:
        hists = hists if hists is not None else [self.hist.get(u, []) for u in users]
        uu = np.array(users)
        demo = (self.feat.gender[uu], self.feat.age[uu], self.feat.occ[uu])
        return user_embeddings(self.tower, self.feat, hists, demo)

    def two_tower_recs(self, users: list[int], k: int) -> dict[int, list[int]]:
        q = self.user_vectors(users)
        _, I = self.index.search(q, k + 400)
        out = {}
        for r, u in enumerate(users):
            seen = set(self.hist.get(u, []))
            out[u] = [int(i) for i in I[r] if i >= 0 and i not in seen][:k]
        return out

    # ---- candidates + features ---------------------------------------------------------
    def candidates(self, users: list[int], ctx_ts: dict[int, int] | None = None,
                   hists: list[list[int]] | None = None, demo_users: list[int] | None = None) -> pd.DataFrame:
        """Return one row per (user, candidate) with all ranker features.

        `hists` lets the API pass an ad-hoc history (e.g. films picked in the demo); in that
        case `users` are synthetic keys and `demo_users` supplies demographics."""
        demo_users = demo_users if demo_users is not None else users
        hists = hists if hists is not None else [self.hist.get(u, []) for u in users]
        q = self.user_vectors(demo_users, hists)
        D, I = self.index.search(q, N_TT + 300)
        rows = []
        for r, u in enumerate(users):
            h = hists[r]
            seen = set(h)
            src: dict[int, list] = {}
            rank = 0
            for d, i in zip(D[r], I[r]):
                if i < 0 or i in seen:
                    continue
                src[int(i)] = [float(d), rank, 1, 0, 0]
                rank += 1
                if rank >= N_TT:
                    break
            if h:
                als_u = self._als_vector(h, u)
                als_s = self.als.V @ als_u
                order = np.argsort(-als_s)
                added = 0
                for i in order:
                    if i in seen:
                        continue
                    if int(i) in src:
                        src[int(i)][3] = 1
                    else:
                        src[int(i)] = [np.nan, N_TT, 0, 1, 0]
                    added += 1
                    if added >= N_ALS:
                        break
            else:
                als_s = np.zeros(self.ds.n_items)
            added = 0
            for i in self.pop_order:
                if i in seen:
                    continue
                if int(i) in src:
                    src[int(i)][4] = 1
                else:
                    src[int(i)] = [np.nan, N_TT, 0, 0, 1]
                added += 1
                if added >= N_POP:
                    break
            cand = np.fromiter(src.keys(), dtype=np.int64)
            meta = np.array(list(src.values()), dtype=np.float64)
            # fill missing two-tower scores exactly (cheap dot product)
            miss = np.isnan(meta[:, 0])
            if miss.any():
                meta[miss, 0] = self.item_vecs[cand[miss]] @ q[r]
            knn_s = self.knn.score_items(h)[cand] if h else np.zeros(len(cand))
            rows.append(self._features(u, demo_users[r], h, cand, meta, als_s[cand], knn_s,
                                       (ctx_ts or {}).get(u, self.cutoff)))
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["user", "item"] + FEATURES)

    def _als_vector(self, h: list[int], u: int) -> np.ndarray:
        if 0 <= u < self.als.U.shape[0] and self.X[u].nnz:
            return self.als.U[u]
        # fold-in: least-squares user vector from item factors (works for brand-new users)
        V = self.als.V[h]
        A = V.T @ V + 0.1 * np.eye(V.shape[1])
        return np.linalg.solve(A, V.sum(0))

    def _features(self, u, demo_u, h, cand, meta, als_s, knn_s, ts) -> pd.DataFrame:
        n = len(cand)
        prof = self.genres[h].mean(0) if h else np.zeros(self.genres.shape[1])
        g = self.genres[cand]
        aff = (g @ prof) / (np.linalg.norm(g, axis=1) * (np.linalg.norm(prof) + 1e-9) + 1e-9)
        hy = self.item_year[h].mean() if h else 1990.0
        t = pd.Timestamp(ts, unit="s")
        last = self.user_last.get(u, np.nan) if u in self.hist else np.nan
        first = self.item_first[cand]
        return pd.DataFrame({
            "user": u, "item": cand,
            "tt_score": meta[:, 0], "tt_rank": meta[:, 1], "als_score": als_s, "knn_score": knn_s,
            "src_tt": meta[:, 2], "src_als": meta[:, 3], "src_pop": meta[:, 4],
            "item_pop_log": np.log1p(self.item_pop[cand]),
            "item_recent_log": np.log1p(self.item_recent[cand]),
            "item_avg_rating": self.item_avg[cand],
            "item_n_ratings_log": np.log1p(self.item_n[cand]),
            "item_age_years": t.year - self.item_year[cand],
            "item_days_on_site": np.where(np.isnan(first), -1.0, (ts - first) / DAY),
            "user_n_log": np.full(n, np.log1p(len(h))),
            "user_avg_rating": np.full(n, self.user_mean_rating.get(u, np.nan)),
            "user_days_since_last": np.full(n, (ts - last) / DAY if not np.isnan(last) else -1.0),
            "user_gender": self.feat.gender[demo_u], "user_age": self.feat.age[demo_u],
            "user_occ": self.feat.occ[demo_u],
            "genre_affinity": aff, "year_gap": self.item_year[cand] - hy,
            "ctx_hour": t.hour, "ctx_weekday": t.day_of_week,
        })


class Ranker:
    def __init__(self, seed: int = 0):
        import lightgbm as lgb
        self.model = lgb.LGBMRanker(objective="lambdarank", n_estimators=400, learning_rate=0.05,
                                    num_leaves=63, min_child_samples=50, subsample=0.8,
                                    subsample_freq=1, colsample_bytree=0.8, random_state=seed,
                                    verbose=-1, lambdarank_truncation_level=50)

    def fit(self, df: pd.DataFrame, labels: np.ndarray, valid: tuple | None = None):
        df = df.sort_values("user", kind="stable")
        groups = df.groupby("user", sort=False).size().to_numpy()
        y = labels[df.index.to_numpy()]
        kw = {}
        if valid is not None:
            vd, vy = valid
            vd = vd.sort_values("user", kind="stable")
            kw = dict(eval_set=[(vd[FEATURES], vy[vd.index.to_numpy()])],
                      eval_group=[vd.groupby("user", sort=False).size().to_numpy()],
                      eval_at=[10], callbacks=[__import__("lightgbm").early_stopping(40, verbose=False)])
        self.model.fit(df[FEATURES], y, group=groups, **kw)
        return self

    def rank(self, df: pd.DataFrame, k: int) -> dict[int, list[int]]:
        s = self.model.predict(df[FEATURES])
        d = df[["user", "item"]].assign(s=s).sort_values(["user", "s"], ascending=[True, False])
        return {int(u): g.item.head(k).tolist() for u, g in d.groupby("user", sort=False)}

    def importance(self) -> pd.Series:
        return pd.Series(self.model.booster_.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)


def label(df: pd.DataFrame, truth: dict[int, set[int]]) -> np.ndarray:
    return np.array([1 if i in truth.get(u, ()) else 0 for u, i in zip(df.user, df.item)], dtype=np.int32)
