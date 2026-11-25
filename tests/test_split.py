from marquee import data as D


def test_temporal_split_has_no_leakage(tmp_path):
    ds = D.load(D.make_synthetic(tmp_path / "ml"))
    sp = D.temporal_split(ds.ratings)
    assert sp.train.ts.max() < sp.t_val <= sp.val.ts.min()
    assert sp.val.ts.max() < sp.t_test <= sp.test.ts.min()
    assert len(sp.train) + len(sp.val) + len(sp.test) == len(ds.ratings)


def test_targets_exclude_already_seen_and_low_ratings(tmp_path):
    import pandas as pd
    hist = pd.DataFrame({"user": [0, 0], "item": [1, 2], "rating": [5, 5], "ts": [1, 2]})
    fut = pd.DataFrame({"user": [0, 0, 0], "item": [1, 3, 4], "rating": [5, 5, 2], "ts": [3, 4, 5]})
    assert D.targets(hist, fut) == {0: {3}}
