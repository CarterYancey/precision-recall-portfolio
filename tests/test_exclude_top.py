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


def test_uniform_baseline_in_sweep() -> None:
    rng = np.random.default_rng(7)
    years = list(range(2016, 2023))
    tickers = [f"T{i:03d}" for i in range(70)]
    stock_yearly = pd.DataFrame(
        rng.normal(0.08, 0.25, size=(len(years), len(tickers))),
        index=years, columns=tickers,
    )
    stock_yearly.index.name = "year"
    bmk_yearly = pd.Series(0.07, index=years)
    grid = dict(num_simulations=100, model_random_seed=42)

    df_uni = sweep_from_returns(stock_yearly, bmk_yearly, [0.2, 0.4], [0.6, 0.8], **grid)
    df_excl = sweep_from_returns(
        stock_yearly, bmk_yearly, [0.2, 0.4], [0.6, 0.8], exclude_top=0.25, **grid
    )

    # Baseline columns exist only when the mode is on.
    uniform_cols = [
        "cagr_custom_q05_uniform", "custom_q05_gap",
        "custom_q05_meets_benchmark_uniform", "custom_q05_meets_ew_benchmark_uniform",
    ]
    for col in uniform_cols:
        assert col not in df_uni.columns
        assert col in df_excl.columns

    # Paired seeding: the embedded baseline reproduces a plain sweep exactly.
    assert np.allclose(
        df_excl["cagr_custom_q05_uniform"], df_uni["cagr_custom_q05"], equal_nan=True
    )
    assert (
        df_excl["custom_q05_meets_benchmark_uniform"]
        == df_uni["custom_q05_meets_benchmark"]
    ).all()
    assert (
        df_excl["custom_q05_meets_ew_benchmark_uniform"]
        == df_uni["custom_q05_meets_ew_benchmark"]
    ).all()

    # gap = uniform - excluded, NaN-propagating.
    both = df_excl["cagr_custom_q05_uniform"] - df_excl["cagr_custom_q05"]
    assert np.allclose(df_excl["custom_q05_gap"], both, equal_nan=True)


def test_uniform_baseline_in_study() -> None:
    """Fabricate a price cache (as in smoke_test.py) so run_top_n_study runs
    offline, and check the uniform baseline rides along when exclude_top is on."""
    import os
    import tempfile

    from pickn import run_top_n_study

    tickers = [f"T{i:02d}" for i in range(30)]
    all_cols = tickers + ["SPY"]
    dates = pd.bdate_range("2020-01-02", "2021-12-31")
    rng = np.random.default_rng(0)
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, size=(len(dates), len(all_cols))), axis=0)),
        index=dates,
        columns=all_cols,
    )

    common = dict(
        n_values=[5], year_start=2020, year_end=2021, tickers=tickers,
        model_recall=0.2, model_precision=0.7, num_simulations=100,
        model_random_seed=42,
    )
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            os.makedirs("data/cache", exist_ok=True)
            prices.to_csv("data/cache/adj_close_cache.csv", index_label="Date")
            res_uni = run_top_n_study(**common)
            res_excl = run_top_n_study(**common, exclude_top=0.1)
        finally:
            os.chdir(cwd)

    # Off: no baseline frame, no *_uniform summary columns.
    assert res_uni.custom_stats_uniform is None
    assert "avg_custom_mean_gap" not in res_uni.summary.columns

    # On: baseline frame present, unexcluded, and (same seed) it reproduces
    # the plain run's distribution.
    baseline = res_excl.custom_stats_uniform
    assert baseline is not None
    assert (baseline["excluded_top"] == 0).all()
    assert np.allclose(baseline["mean"], res_uni.custom_stats["mean"], equal_nan=True)
    assert (res_excl.custom_stats["excluded_top"] >= 1).all()

    gap = (baseline["mean"] - res_excl.custom_stats["mean"]).mean()
    for col in (
        "cagr_custom_mean_uniform", "cagr_custom_q05_uniform",
        "cagr_custom_q95_uniform", "avg_custom_mean_return_uniform",
        "avg_custom_mean_gap",
    ):
        assert col in res_excl.summary.columns
    assert np.allclose(res_excl.summary["avg_custom_mean_gap"], gap)


def main() -> None:
    test_resolve_top_exclusion_count()
    test_distribution_excludes_super_performers()
    test_infeasible_when_pool_cannot_fill_quota()
    test_threading_through_stats_and_sweep()
    test_uniform_baseline_in_sweep()
    test_uniform_baseline_in_study()
    print("All exclude_top tests passed.")


if __name__ == "__main__":
    main()
