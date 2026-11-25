"""Two-tower retrieval model (PyTorch) trained with in-batch sampled softmax + logQ correction.

User tower: pooled embeddings of the user's last L items, their genre profile and demographics.
It has no user-id embedding, so a brand-new user who picks a few films gets a real vector.
Item tower: item-id embedding (with id dropout) + genres + release-year bucket, so items that
were never seen in training still get a content-based vector.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HIST_LEN = 50
N_YEAR_BUCKETS = 12


def year_bucket(years: np.ndarray) -> np.ndarray:
    edges = np.array([1940, 1960, 1970, 1980, 1985, 1990, 1993, 1995, 1997, 1998, 1999])
    return np.searchsorted(edges, years, side="right").astype(np.int64)


@dataclass
class TowerConfig:
    dim: int = 64
    hidden: int = 256
    temperature: float = 0.05
    id_dropout: float = 0.15
    epochs: int = 6
    batch_size: int = 2048
    lr: float = 2e-3
    weight_decay: float = 1e-6
    logq_weight: float = 2.0  # >1 keeps a popularity prior (tuned on the val window)
    seed: int = 0


class ItemTower(nn.Module):
    def __init__(self, n_items: int, n_genres: int, cfg: TowerConfig):
        super().__init__()
        self.id_emb = nn.Embedding(n_items + 1, cfg.dim)  # 0 = unknown item
        self.year_emb = nn.Embedding(N_YEAR_BUCKETS, 16)
        self.genre = nn.Linear(n_genres, 32)
        self.mlp = nn.Sequential(nn.Linear(cfg.dim + 16 + 32, cfg.hidden), nn.GELU(),
                                 nn.Linear(cfg.hidden, cfg.dim))

    def forward(self, ids, genres, years):
        x = torch.cat([self.id_emb(ids), self.year_emb(years), self.genre(genres)], -1)
        return F.normalize(self.mlp(x), dim=-1)


class UserTower(nn.Module):
    def __init__(self, n_items: int, n_genres: int, cfg: TowerConfig):
        super().__init__()
        self.hist_emb = nn.Embedding(n_items + 1, cfg.dim, padding_idx=0)
        self.gender = nn.Embedding(2, 8)
        self.age = nn.Embedding(8, 8)
        self.occ = nn.Embedding(21, 8)
        self.genre = nn.Linear(n_genres, 32)
        self.mlp = nn.Sequential(nn.Linear(cfg.dim * 2 + 24 + 32 + 1, cfg.hidden), nn.GELU(),
                                 nn.Linear(cfg.hidden, cfg.hidden), nn.GELU(),
                                 nn.Linear(cfg.hidden, cfg.dim))

    def forward(self, hist, genre_prof, gender, age, occ):
        mask = (hist > 0).float().unsqueeze(-1)
        e = self.hist_emb(hist)
        n = mask.sum(1).clamp(min=1.0)
        mean = (e * mask).sum(1) / n
        # recency-weighted pooling: the last slots of `hist` are the most recent items
        w = torch.linspace(0.2, 1.0, hist.shape[1], device=hist.device).view(1, -1, 1) * mask
        recent = (e * w).sum(1) / w.sum(1).clamp(min=1e-6)
        length = torch.log1p(mask.sum(1))
        x = torch.cat([mean, recent, self.gender(gender), self.age(age), self.occ(occ),
                       self.genre(genre_prof), length], -1)
        return F.normalize(self.mlp(x), dim=-1)


class TwoTower(nn.Module):
    def __init__(self, n_items: int, n_genres: int, cfg: TowerConfig):
        super().__init__()
        self.cfg = cfg
        self.user = UserTower(n_items, n_genres, cfg)
        self.item = ItemTower(n_items, n_genres, cfg)


class Featurizer:
    """Holds the static item/user side features as tensors."""

    def __init__(self, items: pd.DataFrame, users: pd.DataFrame, known_items: np.ndarray | None = None):
        self.genres = np.stack(items.genre_vec.to_numpy()).astype(np.float32)
        self.years = year_bucket(items.year.to_numpy())
        self.n_items = len(items)
        self.gender = users.gender.to_numpy().astype(np.int64)
        self.age = users.age_bucket.to_numpy().astype(np.int64)
        self.occ = users.occupation.to_numpy().astype(np.int64)
        # items never seen during training map to the "unknown" id 0
        self.item_ids = np.arange(1, self.n_items + 1)
        if known_items is not None:
            self.item_ids = np.where(known_items, self.item_ids, 0)

    def history_rows(self, hists: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
        H = np.zeros((len(hists), HIST_LEN), dtype=np.int64)
        G = np.zeros((len(hists), self.genres.shape[1]), dtype=np.float32)
        for r, h in enumerate(hists):
            h = h[-HIST_LEN:]
            if h:
                H[r, HIST_LEN - len(h):] = np.asarray(h) + 1
                G[r] = self.genres[h].mean(0)
        return H, G


def build_training_samples(train: pd.DataFrame, feat: Featurizer):
    """One sample per interaction: (history strictly before it, target item)."""
    df = train.sort_values(["user", "ts"], kind="stable")
    users = df.user.to_numpy()
    items = df.item.to_numpy()
    n = len(df)
    H = np.zeros((n, HIST_LEN), dtype=np.int32)
    starts = np.r_[0, np.flatnonzero(np.diff(users)) + 1]
    ends = np.r_[starts[1:], n]
    for s, e in zip(starts, ends):
        seq = items[s:e] + 1
        for p in range(1, e - s):
            h = seq[max(0, p - HIST_LEN):p]
            H[s + p, HIST_LEN - len(h):] = h
    return users, items, H


def _genre_profile(H: torch.Tensor, genres: torch.Tensor) -> torch.Tensor:
    g = torch.cat([torch.zeros(1, genres.shape[1], device=genres.device), genres], 0)[H]
    m = (H > 0).float().unsqueeze(-1)
    return (g * m).sum(1) / m.sum(1).clamp(min=1.0)


def train_two_tower(train: pd.DataFrame, feat: Featurizer, cfg: TowerConfig = TowerConfig(),
                    device: str | None = None, log=print) -> TwoTower:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    users, items, H = build_training_samples(train, feat)
    counts = np.bincount(items, minlength=feat.n_items).astype(np.float64)
    logq = torch.tensor(np.log((counts + 1) / (counts.sum() + feat.n_items)), dtype=torch.float32,
                        device=device)

    model = TwoTower(feat.n_items, feat.genres.shape[1], cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    genres = torch.tensor(feat.genres, device=device)
    years = torch.tensor(feat.years, device=device)
    gender = torch.tensor(feat.gender, device=device)
    age = torch.tensor(feat.age, device=device)
    occ = torch.tensor(feat.occ, device=device)
    H_t = torch.from_numpy(H).to(device)  # int32 on device; cast per batch
    del H
    users_t = torch.tensor(users, device=device)
    items_t = torch.tensor(items, device=device)

    n = len(items)
    steps = math.ceil(n / cfg.batch_size) * cfg.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps, pct_start=0.1)
    model.train()
    for ep in range(cfg.epochs):
        t0, tot = time.time(), 0.0
        perm = torch.randperm(n, device=device)
        for s in range(0, n, cfg.batch_size):
            b = perm[s:s + cfg.batch_size]
            u, it, h = users_t[b], items_t[b], H_t[b].long()
            # randomly truncate histories so the model also learns short / empty ones (cold start)
            keep = torch.randint(0, HIST_LEN + 1, (len(b), 1), device=device)
            pos = torch.arange(HIST_LEN, device=device).view(1, -1)
            h = torch.where(pos >= HIST_LEN - keep, h, torch.zeros_like(h)) if rng.random() < 0.3 else h
            ue = model.user(h, _genre_profile(h, genres), gender[u], age[u], occ[u])
            ids = it + 1
            drop = torch.rand(len(b), device=device) < cfg.id_dropout
            ids = torch.where(drop, torch.zeros_like(ids), ids)
            ie = model.item(ids, genres[it], years[it])
            logits = ue @ ie.T / cfg.temperature - cfg.logq_weight * logq[it].view(1, -1)
            same = it.view(-1, 1) == it.view(1, -1)
            eye = torch.eye(len(b), dtype=torch.bool, device=device)
            logits = logits.masked_fill(same & ~eye, float("-inf"))
            loss = F.cross_entropy(logits, torch.arange(len(b), device=device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        log(f"  two-tower epoch {ep + 1}/{cfg.epochs} loss {tot / n:.4f} ({time.time() - t0:.1f}s, {device})")
    model.eval()
    return model


@torch.no_grad()
def item_embeddings(model: TwoTower, feat: Featurizer) -> np.ndarray:
    dev = next(model.parameters()).device
    e = model.item(torch.tensor(feat.item_ids, device=dev), torch.tensor(feat.genres, device=dev),
                   torch.tensor(feat.years, device=dev))
    return e.cpu().numpy().astype(np.float32)


@torch.no_grad()
def user_embeddings(model: TwoTower, feat: Featurizer, hists: list[list[int]],
                    demo: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    dev = next(model.parameters()).device
    H, G = feat.history_rows(hists)
    out = []
    for s in range(0, len(hists), 4096):
        sl = slice(s, s + 4096)
        e = model.user(torch.tensor(H[sl], device=dev), torch.tensor(G[sl], device=dev),
                       torch.tensor(demo[0][sl], device=dev), torch.tensor(demo[1][sl], device=dev),
                       torch.tensor(demo[2][sl], device=dev))
        out.append(e.cpu().numpy())
    return np.concatenate(out).astype(np.float32) if out else np.zeros((0, model.cfg.dim), np.float32)


class AnnIndex:
    """FAISS HNSW index over item vectors (inner product on normalised vectors = cosine)."""

    def __init__(self, vecs: np.ndarray, m: int = 32, ef_search: int = 256):
        import faiss
        self.index = faiss.IndexHNSWFlat(vecs.shape[1], m, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = 200
        self.index.add(np.ascontiguousarray(vecs))
        self.index.hnsw.efSearch = ef_search
        self.n = vecs.shape[0]

    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        return self.index.search(np.ascontiguousarray(q, dtype=np.float32), min(k, self.n))
