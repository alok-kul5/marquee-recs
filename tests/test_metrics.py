import numpy as np

from marquee.metrics import ap_at_k, evaluate, ndcg_at_k, recall_at_k


def test_perfect_ranking_scores_one():
    truth = {1, 2, 3}
    recs = [1, 2, 3, 9, 8]
    assert recall_at_k(recs, truth, 3) == 1.0
    assert abs(ndcg_at_k(recs, truth, 3) - 1.0) < 1e-9
    assert abs(ap_at_k(recs, truth, 3) - 1.0) < 1e-9


def test_miss_scores_zero_and_order_matters():
    assert recall_at_k([7, 8], {1}, 2) == 0.0
    assert ndcg_at_k([1, 9], {1}, 2) > ndcg_at_k([9, 1], {1}, 2)


def test_evaluate_coverage_and_novelty():
    recs = {0: [0, 1], 1: [2, 3]}
    truth = {0: {1}, 1: {9}}
    pop = np.array([100, 50, 1, 1, 0, 0, 0, 0, 0, 0], dtype=float)
    m = evaluate(recs, truth, pop, n_items=10, ks=(2,))
    assert m["coverage@10"] == 0.4
    assert m["recall@2"] == 0.5
    assert m["novelty@10"] > 0
