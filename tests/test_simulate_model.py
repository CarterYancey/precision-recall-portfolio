"""Tests for the pure combinatorics in simulate_model.py.

No network, no market data. Run with:
    uv run python tests/test_simulate_model.py
"""
import math

import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulate_model import estimate_num_ways, simulate_selection


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


def main() -> None:
    test_basic_feasible_selection()
    test_rounding_scheme()
    test_infeasible_requests()
    test_zero_positive_edge_cases()
    test_num_ways_small_exhaustive()
    print("All simulate_model tests passed.")


if __name__ == "__main__":
    main()
