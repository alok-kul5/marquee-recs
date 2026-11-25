"""Loading MovieLens-1M and building a leak-free temporal split."""
from __future__ import annotations

import io
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def fetch_ml1m(dest: Path = DATA_DIR) -> Path:
    """Download and unpack ML-1M into data/ml-1m (never committed)."""
    target = dest / "ml-1m"
    if (target / "ratings.dat").exists():
        return target
    dest.mkdir(parents=True, exist_ok=True)
    print(f"downloading {ML1M_URL}")
    with urllib.request.urlopen(ML1M_URL) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        zf.extractall(dest)
    return target


def _read_dat(path: Path, names: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, sep="::", engine="python", names=names, encoding="latin-1")


@dataclass
class Dataset:
    ratings: pd.DataFrame  # user, item, rating, ts (dense int ids)
    users: pd.DataFrame    # user, gender, age, occupation
    items: pd.DataFrame    # item, movie_id, title, year, genres (list), genre_vec
    n_users: int
    n_items: int


def load(root: Path) -> Dataset:
    ratings = _read_dat(root / "ratings.dat", ["user_raw", "movie_id", "rating", "ts"])
    users = _read_dat(root / "users.dat", ["user_raw", "gender", "age", "occupation", "zip"])
    movies = _read_dat(root / "movies.dat", ["movie_id", "title", "genres"])

    users = users.sort_values("user_raw").reset_index(drop=True)
    users["user"] = np.arange(len(users))
    movies = movies.sort_values("movie_id").reset_index(drop=True)
    movies["item"] = np.arange(len(movies))

    uid = dict(zip(users.user_raw, users.user))
    iid = dict(zip(movies.movie_id, movies.item))
    ratings["user"] = ratings.user_raw.map(uid)
    ratings["item"] = ratings.movie_id.map(iid)
    ratings = ratings.dropna(subset=["user", "item"])
    ratings = ratings.astype({"user": np.int64, "item": np.int64})
    ratings = ratings[["user", "item", "rating", "ts"]].sort_values("ts", kind="stable")
    ratings = ratings.reset_index(drop=True)

    years = movies.title.str.extract(r"\((\d{4})\)\s*$")[0].astype(float)
    movies["year"] = years.fillna(years.median()).astype(int)
    movies["clean_title"] = movies.title.map(lambda t: re.sub(r"\s*\(\d{4}\)\s*$", "", t))
    movies["genre_list"] = movies.genres.str.split("|")
    gvec = np.zeros((len(movies), len(GENRES)), dtype=np.float32)
    gidx = {g: k for k, g in enumerate(GENRES)}
    for r, gl in enumerate(movies.genre_list):
        for g in gl:
            if g in gidx:
                gvec[r, gidx[g]] = 1.0
    movies["genre_vec"] = list(gvec)

    users["gender"] = (users.gender == "M").astype(int)
    age_codes = sorted(users.age.unique())
    users["age_bucket"] = users.age.map({a: k for k, a in enumerate(age_codes)})
    return Dataset(
        ratings=ratings,
        users=users[["user", "user_raw", "gender", "age", "age_bucket", "occupation"]],
        items=movies[["item", "movie_id", "title", "clean_title", "year", "genres",
                      "genre_list", "genre_vec"]],
        n_users=len(users),
        n_items=len(movies),
    )


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    t_val: int
    t_test: int


def temporal_split(ratings: pd.DataFrame, val_frac: float = 0.1, test_frac: float = 0.1) -> Split:
    """Global time cut: everything before t_val trains, [t_val, t_test) validates,
    [t_test, end] tests. No interaction from the future is ever seen while fitting."""
    ts = ratings.ts.to_numpy()
    t_val = int(np.quantile(ts, 1 - val_frac - test_frac))
    t_test = int(np.quantile(ts, 1 - test_frac))
    train = ratings[ratings.ts < t_val]
    val = ratings[(ratings.ts >= t_val) & (ratings.ts < t_test)]
    test = ratings[ratings.ts >= t_test]
    return Split(train.reset_index(drop=True), val.reset_index(drop=True),
                 test.reset_index(drop=True), t_val, t_test)


def targets(history: pd.DataFrame, future: pd.DataFrame, min_rating: int = 4) -> dict[int, set[int]]:
    """Positive items a user rated >= min_rating in the future window and had not seen before."""
    seen = set(zip(history.user, history.item))
    fut = future[future.rating >= min_rating]
    out: dict[int, set[int]] = {}
    for u, i in zip(fut.user.to_numpy(), fut.item.to_numpy()):
        if (u, i) in seen:
            continue
        out.setdefault(int(u), set()).add(int(i))
    return out


def make_synthetic(root: Path, n_users: int = 300, n_items: int = 200, n_ratings: int = 12000,
                   seed: int = 0) -> Path:
    """Write a tiny MovieLens-shaped dataset for tests and CI (no real data is redistributed)."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    item_genre = rng.integers(0, len(GENRES), size=n_items)
    user_pref = rng.integers(0, len(GENRES), size=n_users)
    pop = rng.zipf(1.6, size=n_items).astype(float)
    pop /= pop.sum()
    rows = []
    t0 = 956_700_000
    for k in range(n_ratings):
        u = int(rng.integers(0, n_users))
        if rng.random() < 0.6:
            pool = np.flatnonzero(item_genre == user_pref[u])
            i = int(rng.choice(pool)) if len(pool) else int(rng.choice(n_items, p=pop))
        else:
            i = int(rng.choice(n_items, p=pop))
        r = int(np.clip(round(rng.normal(4.2 if item_genre[i] == user_pref[u] else 3.0, 0.9)), 1, 5))
        rows.append((u + 1, i + 1, r, t0 + k * 60))
    df = pd.DataFrame(rows, columns=["u", "i", "r", "t"]).drop_duplicates(["u", "i"])
    with open(root / "ratings.dat", "w", encoding="latin-1") as f:
        for u, i, r, t in df.itertuples(index=False):
            f.write(f"{u}::{i}::{r}::{t}\n")
    ages = [1, 18, 25, 35, 45, 50, 56]
    with open(root / "users.dat", "w", encoding="latin-1") as f:
        for u in range(n_users):
            f.write(f"{u + 1}::{'MF'[u % 2]}::{ages[u % 7]}::{u % 21}::00000\n")
    with open(root / "movies.dat", "w", encoding="latin-1") as f:
        for i in range(n_items):
            f.write(f"{i + 1}::Movie {i + 1} ({1950 + i % 50})::{GENRES[item_genre[i]]}\n")
    return root
