"""Runs the whole two-stage pipeline on a tiny synthetic dataset, then hits the API."""
import importlib

import pytest


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    from marquee import run as R
    out = tmp_path_factory.mktemp("art")
    summary = R.main(["--synthetic", "--epochs", "1", "--out", str(out)])
    return summary, out


def test_pipeline_produces_all_systems(run):
    summary, _ = run
    res = summary["results"]
    for name in ["Popularity", "Item-kNN", "ALS (implicit)", "BPR (implicit)",
                 "Two-tower (retrieval only)", "Two-stage (two-tower + LightGBM)"]:
        assert name in res
        assert 0.0 <= res[name]["ndcg@10"] <= 1.0
    assert summary["warm_users"] > 0


def test_api_serves_recommendations(run, monkeypatch):
    _, out = run
    monkeypatch.setenv("MARQUEE_ARTIFACTS", str(out))
    import serve.app as A
    A = importlib.reload(A)
    from fastapi.testclient import TestClient
    c = TestClient(A.app)
    assert c.get("/api/health").json()["ok"]
    r = c.post("/api/recommend", json={"liked": [0, 1, 2], "k": 5}).json()
    assert len(r["recommendations"]) == 5
    assert not {0, 1, 2} & {m["id"] for m in r["recommendations"]}
    cold = c.post("/api/recommend", json={"liked": [], "k": 5}).json()
    assert cold["cold_start"] and len(cold["recommendations"]) == 5
    u = c.get("/api/users/0?k=4").json()
    assert len(u["recommendations"]) == 4
    assert "p50_ms" in c.get("/api/stats").json()["live"]
