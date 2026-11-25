"""Measure API latency (p50/p95) against the running FastAPI app in-process."""
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi.testclient import TestClient  # noqa: E402

from serve.app import ART, app, bundle  # noqa: E402


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def main(n=500):
    c = TestClient(app)
    st = bundle()
    s = st["stack"]
    users = [u for u, h in s.hist.items() if len(h) >= 5]
    rng = random.Random(0)
    for _ in range(20):  # warm-up
        c.get(f"/api/users/{rng.choice(users)}")
    server, wall = [], []
    for k in range(n):
        if k % 2:
            liked = rng.sample(range(s.ds.n_items), rng.randint(1, 8))
            t = time.perf_counter()
            r = c.post("/api/recommend", json={"liked": liked, "k": 12})
        else:
            t = time.perf_counter()
            r = c.get(f"/api/users/{rng.choice(users)}?k=12")
        wall.append((time.perf_counter() - t) * 1e3)
        server.append(r.json()["timing"]["total_ms"])
    out = {"requests": n, "p50_ms": pct(wall, 50), "p95_ms": pct(wall, 95), "p99_ms": pct(wall, 99),
           "mean_ms": statistics.mean(wall), "model_p50_ms": pct(server, 50), "model_p95_ms": pct(server, 95)}
    (ART / "latency.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
