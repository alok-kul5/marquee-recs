"""FastAPI service: retrieval (two-tower + FAISS HNSW, ALS, popularity) -> LightGBM ranking."""
from __future__ import annotations

import json
import os
import pickle
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
ART = Path(os.environ.get("MARQUEE_ARTIFACTS", ROOT / "artifacts"))
WEB = ROOT / "web"

app = FastAPI(title="Marquee", version="1.0")
_state: dict = {}
_lat = deque(maxlen=2000)
_lock = threading.Lock()


def bundle():
    if "stack" not in _state:
        with open(ART / "bundle.pkl", "rb") as f:
            b = pickle.load(f)
        _state.update(b)
        s = b["stack"]
        items = s.ds.items
        _state["catalog"] = [
            {"id": int(r.item), "title": r.clean_title, "year": int(r.year), "genres": r.genre_list,
             "n": int(s.item_pop[r.item])}
            for r in items.itertuples()
        ]
        mpath = ART / "metrics.json"
        _state["metrics"] = json.loads(mpath.read_text()) if mpath.exists() else {}
    return _state


def movie(i: int, extra: dict | None = None) -> dict:
    m = dict(bundle()["catalog"][i])
    if extra:
        m.update(extra)
    return m


class RecRequest(BaseModel):
    liked: list[int] = Field(default_factory=list, max_length=50)
    k: int = Field(12, ge=1, le=50)
    gender: int | None = None  # 0 = F, 1 = M
    age: int | None = None     # age bucket 0-6


def _demo_user(s, gender, age) -> int:
    f = s.feat
    ok = np.ones(len(f.gender), dtype=bool)
    if gender is not None:
        ok &= f.gender == gender
    if age is not None:
        ok &= f.age == age
    idx = np.flatnonzero(ok)
    return int(idx[0]) if len(idx) else 0


def _recommend(liked: list[int], k: int, demo_u: int, key: int):
    st = bundle()
    s, ranker = st["stack"], st["ranker"]
    t0 = time.perf_counter()
    cand = s.candidates([key], ctx_ts={key: s.cutoff}, hists=[liked], demo_users=[demo_u])
    t1 = time.perf_counter()
    top = ranker.rank(cand, k).get(key, [])
    t2 = time.perf_counter()
    row = cand.set_index("item")
    out = []
    for rank, i in enumerate(top):
        because = None
        if liked:
            sims = s.item_vecs[liked] @ s.item_vecs[i]
            j = int(np.argmax(sims))
            because = {"id": int(liked[j]), "title": st["catalog"][liked[j]]["title"],
                       "similarity": round(float(sims[j]), 3)}
        r = row.loc[i]
        srcs = [n for n, c in (("two-tower", "src_tt"), ("ALS", "src_als"), ("item-kNN", "src_knn"), ("trending", "src_pop")) if r[c] > 0]
        out.append(movie(int(i), {"rank": rank + 1, "because": because, "sources": srcs,
                                  "retrieval_score": round(float(r.tt_score), 3),
                                  "genre_affinity": round(float(r.genre_affinity), 3)}))
    t3 = time.perf_counter()
    timing = {"retrieve_ms": (t1 - t0) * 1e3, "rank_ms": (t2 - t1) * 1e3, "total_ms": (t3 - t0) * 1e3,
              "candidates": int(len(cand)), "catalog": int(s.ds.n_items)}
    with _lock:
        _lat.append(timing["total_ms"])
    return out, {k2: (round(v, 2) if isinstance(v, float) else v) for k2, v in timing.items()}


@app.get("/api/health")
def health():
    st = bundle()
    return {"ok": True, "items": st["stack"].ds.n_items, "users": st["stack"].ds.n_users}


@app.get("/api/movies")
def search(q: str = "", limit: int = 24):
    cat = bundle()["catalog"]
    ql = q.strip().lower()
    hits = [m for m in cat if ql in m["title"].lower()] if ql else cat
    hits = sorted(hits, key=lambda m: -m["n"])[: max(1, min(limit, 60))]
    return {"results": hits}


@app.get("/api/popular")
def popular(limit: int = 24):
    s = bundle()["stack"]
    return {"results": [movie(int(i)) for i in s.pop_order[: max(1, min(limit, 60))]]}


@app.post("/api/recommend")
def recommend(req: RecRequest):
    s = bundle()["stack"]
    liked = [i for i in dict.fromkeys(req.liked) if 0 <= i < s.ds.n_items]
    recs, timing = _recommend(liked, req.k, _demo_user(s, req.gender, req.age), key=-1)
    return {"liked": [movie(i) for i in liked], "recommendations": recs, "timing": timing,
            "cold_start": not liked}


@app.get("/api/users/{user}")
def user_recs(user: int, k: int = 12):
    s = bundle()["stack"]
    if not 0 <= user < s.ds.n_users:
        raise HTTPException(404, "unknown user")
    h = s.hist.get(user, [])
    recs, timing = _recommend(h, k, user, key=user)
    return {"user": user, "history": [movie(i) for i in h[-12:][::-1]], "history_size": len(h),
            "recommendations": recs, "timing": timing}


@app.get("/api/users")
def sample_users(n: int = 1, seed: int | None = None):
    s = bundle()["stack"]
    rng = np.random.default_rng(seed)
    active = [u for u, h in s.hist.items() if 20 <= len(h) <= 300]
    return {"users": [int(u) for u in rng.choice(active, size=min(n, len(active)), replace=False)]}


@app.get("/api/stats")
def stats():
    st = bundle()
    with _lock:
        lat = np.array(_lat)
    live = {"requests": int(len(lat))}
    if len(lat):
        live.update(p50_ms=round(float(np.percentile(lat, 50)), 2), p95_ms=round(float(np.percentile(lat, 95)), 2))
    bench = ART / "latency.json"
    return {"live": live, "benchmark": json.loads(bench.read_text()) if bench.exists() else None,
            "offline": st["metrics"]}


if WEB.exists():
    app.mount("/static", StaticFiles(directory=WEB), name="static")

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")
