"""Tests for the "miss the super-performers" mode (exclude_top).

Synthetic returns only — no network, no price cache. Run with:
    uv run python tests/test_exclude_top.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")

from pickn import (
    compute_custom_stats,
    resolve_top_exclusion_count,
    simulate_custom_portfolio_distribution,
    sweep_from_returns,
)


def test_resolve_top_exclusion_count() -> None:
    # Off.
    assert resolve_top_exclusion_count(100, None) == 0

    # Fractions of the positive set, ceil so any nonzero fraction bites.
    assert resolve_top_exclusion_count(100, 0.1) == 10
    assert resolve_top_exclusion_count(25, 0.1) == 3   # ceil(2.5)
    assert resolve_top_exclusion_count(5, 0.5) == 3    # ceil(2.5)
    assert resolve_top_exclusion_count(5, 0.01) == 1   # ceil(0.05)
    assert resolve_top_exclusion_count(0, 0.1) == 0    # no positives, nothing to bar

    # Absolute counts, capped at the positive-set size.
    assert resolve_top_exclusion_count(100, 1) == 1
    assert resolve_top_exclusion_count(100, 7.0) == 7
    assert resolve_top_exclusion_count(5, 100) == 5

    # Invalid specs.
    for bad in (0, -1, -0.5, 2.5, float("inf"), float("nan")):
        try:
            resolve_top_exclusion_count(100, bad)
            raise AssertionError(f"exclude_top={bad} should raise")
        except ValueError:
            pass


def make_year_returns() -> pd.Series:
    # 5 positives vs benchmark 0.05 (one 0.50 super-performer), 5 negatives.
    return pd.Series(
        {
            "SUPER": 0.50,
            "GOOD1": 0.20,
            "GOOD2": 0.15,
            "GOOD3": 0.12,
            "GOOD4": 0.10,
            "MEH1": 0.04,
            "MEH2": 0.00,
            "BAD1": -0.05,
            "BAD2": -0.10,
            "BAD3": -0.20,
        }
    )


def test_distribution_excludes_super_performers() -> None:
    year_returns = make_year_returns()
    bmk = 0.05

    # recall=0.2 of T=5 -> TP=1; precision=1.0 -> FP=0. Uniform draws can land
    # on SUPER; with exclude_top=1 the drawable positives top out at GOOD1.
    uniform = simulate_custom_portfolio_distribution(
        year_returns, bmk, 0.2, 1.0, 100, rng=np.random.default_rng(0)
    )
    pessimist = simulate_custom_portfolio_distribution(
        year_returns, bmk, 0.2, 1.0, 100,
        exclude_top=1, rng=np.random.default_rng(0),
    )

    assert uniform["excluded_top"] == 0
    assert pessimist["excluded_top"] == 1

    # num_ways caps the draw count: C(5,1)=5 uniform, C(4,1)=4 excluded.
    assert uniform["count"] == 5
    assert pessimist["count"] == 4

    # Excluded draws can never contain the 0.50 outlier.
    assert pessimist["q95"] <= 0.20 + 1e-12
    assert pessimist["mean"] <= 0.20 + 1e-12

    # Recall's denominator still counts the excluded positive.
    assert abs(pessimist["achieved_recall"] - 0.2) < 1e-12
    assert abs(pessimist["achieved_precision"] - 1.0) < 1e-12

    # Base rate is a labeling property, untouched by exclusion.
    assert pessimist["base_rate"] == uniform["base_rate"] == 0.5

    # Deeper exclusion bites harder: top-2 out -> max drawable is 0.15.
    pessimist2 = simulate_custom_portfolio_distribution(
        year_returns, bmk, 0.2, 1.0, 100,
        exclude_top=2, rng=np.random.default_rng(0),
    )
    assert pessimist2["excluded_top"] == 2
    assert pessimist2["q95"] <= 0.15 + 1e-12

    # Fraction spec: 0.1 of 5 positives -> ceil(0.5) = 1 excluded.
    frac = simulate_custom_portfolio_distribution(
        year_returns, bmk, 0.2, 1.0, 100,
        exclude_top=0.1, rng=np.random.default_rng(0),
    )
    assert frac["excluded_top"] == 1


def test_infeasible_when_pool_cannot_fill_quota() -> None:
    year_returns = make_year_returns()
    # recall=1.0 needs TP=5 but exclusion leaves only 4 drawable positives;
    # achieved recall 0.8 falls outside the ±0.1 tolerance -> no draws.
    stats = simulate_custom_portfolio_distribution(
        year_returns, 0.05, 1.0, 1.0, 100,
        exclude_top=1, rng=np.random.default_rng(0),
    )
    assert stats["count"] == 0
    assert np.isnan(stats["mean"])
    assert abs(stats["achieved_recall"] - 0.8) < 1e-12


def test_threading_through_stats_and_sweep() -> None:
    rng = np.random.default_rng(3)
    years = list(range(2018, 2024))
    tickers = [f"T{i:03d}" for i in range(60)]
    stock_yearly = pd.DataFrame(
        rng.normal(0.08, 0.25, size=(len(years), len(tickers))),
        index=years, columns=tickers,
    )
    stock_yearly.index.name = "year"
    bmk_yearly = pd.Series(0.07, index=years)

    stats = compute_custom_stats(
        stock_yearly, bmk_yearly, 0.2, 0.7, 50,
        exclude_top=0.1, rng=np.random.default_rng(42),
    )
    assert "excluded_top" in stats.columns
    positives_per_year = (stock_yearly.gt(0.07)).sum(axis=1)
    expected = np.ceil(0.1 * positives_per_year)
    assert (stats["excluded_top"] == expected).all()

    # The pessimistic sweep should never beat the uniform one, year by year:
    # exclusion only removes the best names from the TP pool.
    uniform_stats = compute_custom_stats(
        stock_yearly, bmk_yearly, 0.2, 0.7, 200,
        rng=np.random.default_rng(42),
    )
    pessimist_stats = compute_custom_stats(
        stock_yearly, bmk_yearly, 0.2, 0.7, 200,
        exclude_top=0.25, rng=np.random.default_rng(42),
    )
    assert pessimist_stats["mean"].mean() < uniform_stats["mean"].mean()

    df = sweep_from_returns(
        stock_yearly, bmk_yearly, [0.2], [0.7],
        num_simulations=50, model_random_seed=42, exclude_top=0.1,
    )
    assert np.isclose(df.loc[0, "excluded_top_mean"], expected.mean())


def main() -> None:
    test_resolve_top_exclusion_count()
    test_distribution_excludes_super_performers()
    test_infeasible_when_pool_cannot_fill_quota()
    test_threading_through_stats_and_sweep()
    print("All exclude_top tests passed.")


if __name__ == "__main__":
    main()
