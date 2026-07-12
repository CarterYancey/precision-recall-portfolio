"""Tests for the pure combinatorics in simulate_model.py.

No network, no market data. Run with:
    uv run python tests/test_simulate_model.py
"""
import math

import numpy as np
import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulate_model import estimate_num_ways, sample_confusion_draws, simulate_selection


def build_df(n_true: int, n_false: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": [f"p{i}" for i in range(n_true)] + [f"n{i}" for i in range(n_false)],
            "label": ["TRUE"] * n_true + ["FALSE"] * n_false,
        }
    )


def ways(df, recall, precision) -> int:
    return estimate_num_ways(df, "name", "label", recall, precision)


def sim(df, recall, precision, **kw):
    return simulate_selection(df, "name", "label", recall, precision, **kw)


def test_basic_feasible_selection() -> None:
    df = build_df(10, 90)

    # recall=0.5 -> TP=5; precision=0.5 -> FP=5.
    res = sim(df, 0.5, 0.5, random_state=1)
    assert (res.tp, res.fp, res.fn, res.tn) == (5, 5, 5, 85), res
    assert res.achieved_recall == 0.5 and res.achieved_precision == 0.5
    assert res.note is None
    assert res.num_ways == math.comb(10, 5) * math.comb(90, 5)
    assert ways(df, 0.5, 0.5) == res.num_ways

    # Selection contains exactly the right mix of positives and negatives.
    positives = {f"p{i}" for i in range(10)}
    chosen = set(res.selected_names)
    assert len(chosen) == 10
    assert len(chosen & positives) == 5

    # Perfect model: all positives, nothing else, exactly one way.
    res = sim(df, 1.0, 1.0, random_state=1)
    assert (res.tp, res.fp) == (10, 0)
    assert set(res.selected_names) == positives
    assert res.num_ways == 1 and ways(df, 1.0, 1.0) == 1

    # random_state makes the draw reproducible.
    a = sim(df, 0.3, 0.6, random_state=7).selected_names
    b = sim(df, 0.3, 0.6, random_state=7).selected_names
    assert a == b


def test_rounding_scheme() -> None:
    df = build_df(10, 90)

    # TP = round(recall * T): 0.33 * 10 = 3.3 -> 3.
    res = sim(df, 0.33, 1.0, random_state=1)
    assert res.tp == 3 and res.fp == 0

    # FP = round(TP * (1-p)/p): TP=3, p=0.75 -> round(1.0) = 1.
    res = sim(df, 0.3, 0.75, random_state=1)
    assert (res.tp, res.fp) == (3, 1)
    assert ways(df, 0.3, 0.75) == math.comb(10, 3) * math.comb(90, 1)

    # Python's banker's rounding is part of the scheme: round(2.5) == 2.
    res = sim(df, 0.25, 1.0, random_state=1)
    assert res.tp == 2, res.tp


def test_infeasible_requests() -> None:
    # Not enough negatives: TP=9, p=0.1 needs FP=81 but only F=5 exist.
    df = build_df(10, 5)
    assert ways(df, 0.9, 0.1) == 0
    res = sim(df, 0.9, 0.1, random_state=1)
    assert res.fp == 5 and res.note is not None  # best effort: capped at F
    assert res.num_ways == math.comb(10, 9) * math.comb(5, 5)
    try:
        sim(df, 0.9, 0.1, strict=True)
        raise AssertionError("strict=True should raise on infeasible request")
    except ValueError:
        pass

    # precision=0 with recall>0 can't happen (any TP makes precision > 0).
    df = build_df(10, 90)
    assert ways(df, 0.5, 0.0) == 0
    res = sim(df, 0.5, 0.0, random_state=1)
    assert res.fp == 0 and res.num_ways == 0 and res.note is not None

    # TP=0 with precision strictly between 0 and 1 is infeasible; the
    # simulation falls back to the empty selection (num_ways counts ways
    # for the *returned* selection, and there is exactly 1 empty one).
    assert ways(df, 0.0, 0.5) == 0
    res = sim(df, 0.0, 0.5, random_state=1)
    assert res.selected_names == [] and res.note is not None
    assert res.num_ways == 1 and res.achieved_precision == 1.0

    # Out-of-range metrics are rejected outright.
    for bad in (-0.1, 1.5, float("nan")):
        for args in ((bad, 0.5), (0.5, bad)):
            try:
                ways(df, *args)
                raise AssertionError(f"metric {bad} should raise")
            except ValueError:
                pass


def test_zero_positive_edge_cases() -> None:
    df = build_df(0, 4)

    # recall > 0 is impossible with no TRUE rows.
    assert ways(df, 0.5, 0.5) == 0
    res = sim(df, 0.5, 0.5, random_state=1)
    assert res.selected_names == [] and res.num_ways == 0

    # recall=0, precision=1: only the empty selection qualifies.
    assert ways(df, 0.0, 1.0) == 1
    res = sim(df, 0.0, 1.0, random_state=1)
    assert res.selected_names == [] and res.num_ways == 1

    # recall=0, precision=0: any non-empty subset of the 4 negatives.
    assert ways(df, 0.0, 0.0) == 2**4 - 1
    res = sim(df, 0.0, 0.0, random_state=1)
    assert res.fp == 1 and res.num_ways == 2**4 - 1

    # recall=0, precision in (0,1): infeasible.
    assert ways(df, 0.0, 0.5) == 0

    # Fully empty dataset can't achieve precision=0.
    empty = build_df(0, 0)
    res = sim(empty, 0.0, 0.0, random_state=1)
    assert res.selected_names == [] and res.num_ways == 0


def test_exclude_names() -> None:
    df = build_df(10, 90)
    excluded = ["p0", "p1"]
    positives = {f"p{i}" for i in range(10)}

    # T stays 10 (recall's denominator), so recall=0.5 still means TP=5 —
    # but the 5 must come from the 8 non-excluded positives.
    res = sim(df, 0.5, 0.5, random_state=1, exclude_names=excluded)
    assert (res.tp, res.fp, res.fn, res.tn) == (5, 5, 5, 85), res
    assert res.achieved_recall == 0.5 and res.achieved_precision == 0.5
    assert res.note is None
    assert not set(res.selected_names) & set(excluded)
    assert len(set(res.selected_names) & positives) == 5
    expected_ways = math.comb(8, 5) * math.comb(90, 5)
    assert res.num_ways == expected_ways
    assert estimate_num_ways(
        df, "name", "label", 0.5, 0.5, exclude_names=excluded
    ) == expected_ways

    # Excluded names never show up, whatever the seed.
    for seed in range(20):
        res = sim(df, 0.5, 0.5, random_state=seed, exclude_names=excluded)
        assert not set(res.selected_names) & set(excluded)

    # TP quota exceeding the shrunken pool: recall=1 needs TP=10, only 8 drawable.
    assert estimate_num_ways(df, "name", "label", 1.0, 1.0, exclude_names=excluded) == 0
    res = sim(df, 1.0, 1.0, random_state=1, exclude_names=excluded)
    assert res.tp == 8 and res.note is not None
    assert res.achieved_recall == 0.8
    assert set(res.selected_names) == positives - set(excluded)
    assert res.num_ways == 1  # comb(8, 8) for the returned (capped) selection
    try:
        sim(df, 1.0, 1.0, strict=True, exclude_names=excluded)
        raise AssertionError("strict=True should raise when the pool can't fill the quota")
    except ValueError:
        pass

    # Excluded negatives shrink the FP pool the same way.
    small = build_df(2, 3)
    assert estimate_num_ways(
        small, "name", "label", 1.0, 0.5, exclude_names=["n0", "n1"]
    ) == 0
    res = sim(small, 1.0, 0.5, random_state=1, exclude_names=["n0", "n1"])
    assert res.fp == 1 and res.note is not None
    assert "n0" not in res.selected_names and "n1" not in res.selected_names

    # Names not in the dataset are ignored; empty exclusion is a no-op.
    assert estimate_num_ways(
        df, "name", "label", 0.5, 0.5, exclude_names=["ghost"]
    ) == ways(df, 0.5, 0.5)
    assert estimate_num_ways(
        df, "name", "label", 0.5, 0.5, exclude_names=[]
    ) == ways(df, 0.5, 0.5)


def test_exclude_names_small_exhaustive() -> None:
    # T=3, F=3, one positive excluded; brute-force subsets that avoid it and
    # hit the (recall, precision) targets under the same rounding scheme.
    df = build_df(3, 3)
    from itertools import combinations

    names = df["name"].tolist()
    positives = set(names[:3])
    excluded = {"p0"}

    def brute(recall, precision) -> int:
        tp_target = round(recall * 3)
        count = 0
        for r in range(len(names) + 1):
            for combo in combinations(names, r):
                if set(combo) & excluded:
                    continue
                tp = len(set(combo) & positives)
                fp = len(combo) - tp
                achieved_p = tp / (tp + fp) if combo else 1.0
                if tp == tp_target and abs(achieved_p - precision) < 1e-9:
                    count += 1
        return count

    for recall, precision in [(1.0, 1.0), (2 / 3, 1.0), (2 / 3, 0.5), (1 / 3, 0.5), (0.0, 1.0)]:
        got = estimate_num_ways(
            df, "name", "label", recall, precision, exclude_names=excluded
        )
        assert got == brute(recall, precision), (recall, precision, got)


def test_num_ways_small_exhaustive() -> None:
    # T=2, F=3; brute-force every subset and compare counts with
    # estimate_num_ways under the same rounding scheme.
    df = build_df(2, 3)
    from itertools import combinations

    names = df["name"].tolist()
    positives = set(names[:2])

    def brute(recall, precision) -> int:
        tp_target = round(recall * 2)
        count = 0
        for r in range(len(names) + 1):
            for combo in combinations(names, r):
                tp = len(set(combo) & positives)
                fp = len(combo) - tp
                achieved_p = tp / (tp + fp) if combo else 1.0
                if tp == tp_target and abs(achieved_p - precision) < 1e-9:
                    count += 1
        return count

    for recall, precision in [(1.0, 1.0), (0.5, 0.5), (0.5, 1.0), (1.0, 0.5), (0.0, 1.0)]:
        assert ways(df, recall, precision) == brute(recall, precision), (recall, precision)


def test_sample_confusion_draws() -> None:
    rng = np.random.default_rng(0)
    pos_idx, neg_idx = sample_confusion_draws(10, 90, 5, 7, 200, rng)
    assert pos_idx.shape == (200, 5) and neg_idx.shape == (200, 7)
    for idx, pool in ((pos_idx, 10), (neg_idx, 90)):
        assert idx.min() >= 0 and idx.max() < pool
        # Without replacement within each draw.
        assert all(len(set(row)) == len(row) for row in idx)

    # Zero quotas and full-pool quotas.
    pos_idx, neg_idx = sample_confusion_draws(10, 90, 0, 90, 3, rng)
    assert pos_idx.shape == (3, 0) and neg_idx.shape == (3, 90)
    assert all(sorted(row) == list(range(90)) for row in neg_idx)
    pos_idx, neg_idx = sample_confusion_draws(4, 6, 4, 0, 3, rng)
    assert all(sorted(row) == list(range(4)) for row in pos_idx)
    pos_idx, neg_idx = sample_confusion_draws(10, 90, 5, 7, 0, rng)
    assert pos_idx.shape == (0, 5) and neg_idx.shape == (0, 7)

    # Same seed -> same draws; the blocked path draws the same way per row
    # (each row consumes one pool-sized batch of keys either way).
    a = sample_confusion_draws(10, 90, 5, 7, 50, np.random.default_rng(7))
    b = sample_confusion_draws(10, 90, 5, 7, 50, np.random.default_rng(7))
    c = sample_confusion_draws(
        10, 90, 5, 7, 50, np.random.default_rng(7), max_block_elems=1
    )
    assert (a[0] == b[0]).all() and (a[1] == b[1]).all()
    assert (a[0] == c[0]).all() and (a[1] == c[1]).all()

    # Uniformity: every pool member should be picked ~ num_draws * k / pool.
    pos_idx, _ = sample_confusion_draws(10, 0, 3, 0, 20000, np.random.default_rng(1))
    counts = np.bincount(pos_idx.ravel(), minlength=10)
    expected = 20000 * 3 / 10
    assert (abs(counts - expected) < 5 * math.sqrt(expected)).all(), counts

    # Quotas outside the pools are rejected.
    for bad in ((11, 7), (5, 91), (-1, 7), (5, -1)):
        try:
            sample_confusion_draws(10, 90, *bad, 5, rng)
            raise AssertionError(f"quotas {bad} should raise")
        except ValueError:
            pass
    try:
        sample_confusion_draws(10, 90, 5, 7, -1, rng)
        raise AssertionError("negative num_draws should raise")
    except ValueError:
        pass


def main() -> None:
    test_basic_feasible_selection()
    test_rounding_scheme()
    test_infeasible_requests()
    test_zero_positive_edge_cases()
    test_exclude_names()
    test_exclude_names_small_exhaustive()
    test_num_ways_small_exhaustive()
    test_sample_confusion_draws()
    print("All simulate_model tests passed.")


if __name__ == "__main__":
    main()
