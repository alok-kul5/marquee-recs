"""Quick validation-window sweep for the two-tower retriever (train -> val)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402

from marquee import data as D  # noqa: E402
from marquee.baselines import interaction_matrix  # noqa: E402
from marquee.metrics import evaluate  # noqa: E402
from marquee.two_tower import (AnnIndex, Featurizer, TowerConfig, item_embeddings,  # noqa: E402
                               train_two_tower, user_embeddings)

ds = D.load(D.fetch_ml1m())
sp = D.temporal_split(ds.ratings)
truth = D.targets(sp.train, sp.val)
X = interaction_matrix(sp.train, ds.n_users, ds.n_items)
known = np.asarray(X.sum(0)).ravel() > 0
pop = np.asarray(X.sum(0)).ravel()
h = sp.train.sort_values(["user", "ts"])
hist = {u: g.item.tolist() for u, g in h.groupby("user")}
users = sorted(u for u in truth if u in hist)
feat = Featurizer(ds.items, ds.users, known)
for kw in [dict(logq_weight=2.0), dict(logq_weight=1.5, epochs=10), dict(logq_weight=1.5, temperature=0.1)]:
    cfg = TowerConfig(**kw)
    m = train_two_tower(sp.train, feat, cfg, log=lambda *_: None)
    idx = AnnIndex(item_embeddings(m, feat))
    uu = np.array(users)
    q = user_embeddings(m, feat, [hist[u] for u in users], (feat.gender[uu], feat.age[uu], feat.occ[uu]))
    _, I = idx.search(q, 500)
    recs = {u: [int(i) for i in I[r] if i not in set(hist[u])][:50] for r, u in enumerate(users)}
    r = evaluate(recs, truth, pop, ds.n_items)
    print(kw, {k: round(v, 4) for k, v in r.items() if k in ("recall@10", "ndcg@10", "recall@50", "coverage@10")}, flush=True)
    del m, idx
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()
