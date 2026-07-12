"""Tests for multi-year holding-horizon support: formation-year scheduling,
h-year window returns (universe and benchmark), per-annum threshold
compounding, CAGR annualization, and the horizon-aware screen/sweep/study.

Pure pandas / fabricated price cache — no network. Run with:
    uv run python tests/test_horizon.py
"""
import os
import tempfile
from pathlib import Path

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pickn import (
    annualize_window_return,
    compound_annual_threshold,
    compute_cagr,
    compute_calendar_year_returns,
    formation_years,
    screen_label_criteria,
    summarize_screen,
    sweep_from_returns,
    year_universe_returns,
)


def test_formation_years() -> None:
    # Horizon 1: every year, both modes identical.
    assert formation_years(2012, 2015, 1) == [2012, 2013, 2014, 2015]
    assert formation_years(2012, 2015, 1, overlapping=True) == [2012, 2013, 2014, 2015]

    # Non-overlapping steps by h; windows must fit inside [start, end].
    # 2012..2024 with h=3: last formation is 2022 (covers 2022-2024).
    assert formation_years(2012, 2024, 3) == [2012, 2015, 2018, 2021]
    assert formation_years(2012, 2024, 3, overlapping=True) == list(range(2012, 2023))

    # Exactly one window.
    assert formation_years(2020, 2022, 3) == [2020]

    # No window fits, or nonsense horizons -> loud errors.
    for bad_call in (
        lambda: formation_years(2020, 2021, 3),
        lambda: formation_years(2020, 2022, 0),
        lambda: formation_years(2020, 2022, -1),
        lambda: formation_years(2020, 2022, 1.5),
    ):
        try:
            bad_call()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


def test_threshold_compounding() -> None:
    # h=1 is the identity in both directions.
    assert abs(compound_annual_threshold(0.1, 1) - 0.1) < 1e-12
    assert abs(annualize_window_return(0.1, 1) - 0.1) < 1e-12

    # '>10% annual over 3 years' means a window total of 1.1^3 - 1.
    assert abs(compound_annual_threshold(0.1, 3) - (1.1**3 - 1.0)) < 1e-12
    # 0% annual stays 0% total at any horizon.
    assert compound_annual_threshold(0.0, 5) == 0.0

    # Round trip.
    assert abs(annualize_window_return(compound_annual_threshold(0.07, 4), 4) - 0.07) < 1e-12
    # Total loss annualizes to -100%, not a complex number.
    assert annualize_window_return(-1.0, 3) == -1.0


def test_compute_cagr_years_per_period() -> None:
    # One 3-year window returning 33.1% total is a 10% CAGR.
    one_window = pd.Series([1.1**3 - 1.0], index=[2020])
    assert abs(compute_cagr(one_window, years_per_period=3) - 0.1) < 1e-12

    # Two chained 2-year windows: growth 1.21 * 0.81 over 4 years.
    two_windows = pd.Series([0.21, -0.19], index=[2020, 2022])
    expected = (1.21 * 0.81) ** (1.0 / 4.0) - 1.0
    assert abs(compute_cagr(two_windows, years_per_period=2) - expected) < 1e-12

    # Default keeps the original per-year behavior.
    assert abs(compute_cagr(pd.Series([0.21])) - 0.21) < 1e-12


def build_adj() -> pd.DataFrame:
    """Three years of synthetic daily prices exercising the window rules."""
    days = pd.bdate_range("2020-01-02", "2022-12-30")
    n = len(days)
    adj = pd.DataFrame(index=days, dtype=float)

    # FULL: trades the whole window, 100 -> 180.
    adj["FULL"] = np.linspace(100.0, 180.0, n)

    # MIDDELIST: delists halfway through the 3-year window (in year 2);
    # must be kept at its last available price under the same rule that
    # covers mid-year delistings at horizon 1.
    half = n // 2
    mid = np.full(n, np.nan)
    mid[:half] = np.linspace(100.0, 40.0, half)
    adj["MIDDELIST"] = mid

    # LATE: data only begins in the window's second half (reused symbol).
    late = np.full(n, np.nan)
    late[half:] = np.linspace(50.0, 60.0, n - half)
    adj["LATE"] = late

    adj["SPY"] = np.linspace(300.0, 390.0, n)
    return adj


def test_window_universe_returns() -> None:
    adj = build_adj()
    universe = ["FULL", "MIDDELIST", "LATE", "GONE"]
    returns, stats = year_universe_returns(adj, 2020, universe, horizon=3)

    assert abs(returns["FULL"] - 0.8) < 1e-9, returns["FULL"]

    # Delisting *inside the horizon* (year 2 of 3): kept at last print.
    assert "MIDDELIST" in returns.index, "in-horizon delisting was dropped"
    assert abs(returns["MIDDELIST"] - (40.0 / 100.0 - 1.0)) < 1e-9

    # Late data start is still excluded, measured from the window start.
    assert "LATE" not in returns.index

    assert stats["members"] == 4
    assert stats["no_data"] == 1        # GONE
    assert stats["late_start"] == 1     # LATE
    assert stats["partial_year"] == 1   # MIDDELIST (partial *window*)
    assert stats["full_year"] == 1      # FULL
    assert stats["used"] == 2

    # horizon=1 restricts to the formation year only (old behavior):
    # MIDDELIST trades all of 2020, so it's a full-year name there.
    r1, s1 = year_universe_returns(adj, 2020, universe, horizon=1)
    assert s1["partial_year"] == 0 and s1["full_year"] == 2
    year_mask = adj.index.year == 2020
    exp = adj.loc[year_mask, "FULL"].iloc[-1] / adj.loc[year_mask, "FULL"].iloc[0] - 1.0
    assert abs(r1["FULL"] - exp) < 1e-9


def test_benchmark_window_returns() -> None:
    adj = build_adj()
    out = compute_calendar_year_returns(adj[["SPY"]], [2020], horizon=3)
    assert list(out.index) == [2020]
    assert abs(out.loc[2020, "SPY"] - 0.3) < 1e-9

    # A window with no data is skipped, same as before.
    empty = compute_calendar_year_returns(adj[["SPY"]], [2010], horizon=3)
    assert empty.empty


def make_window_returns(num_windows: int = 4, num_tickers: int = 60, seed: int = 1):
    """Synthetic formation-year x ticker window returns plus a benchmark."""
    rng = np.random.default_rng(seed)
    years = [2010 + 2 * i for i in range(num_windows)]  # h=2, non-overlapping
    tickers = [f"T{i:03d}" for i in range(num_tickers)]
    stock_windows = pd.DataFrame(
        rng.normal(0.16, 0.35, size=(num_windows, num_tickers)),
        index=years,
        columns=tickers,
    )
    stock_windows.index.name = "year"
    bmk_windows = pd.Series(0.15, index=years)
    return stock_windows, bmk_windows


def test_screen_with_horizon() -> None:
    stock, bmk = make_window_returns()
    screen = screen_label_criteria(stock, bmk, ["0.1", "bmk+0.05"], horizon=2)

    # Absolute criterion: threshold compounds to the window total.
    absolute = screen[screen["criterion"] == ">10%"]
    assert np.allclose(absolute["threshold"], 1.1**2 - 1.0)

    # Benchmark-relative: offset applied to the annualized benchmark return,
    # then compounded back. bmk window 0.15 -> annualized sqrt(1.15)-1.
    relative = screen[screen["criterion"] == ">benchmark+5%"]
    expected = (np.sqrt(1.15) + 0.05) ** 2 - 1.0
    assert np.allclose(relative["threshold"], expected)

    # Membership of the positive set follows the compounded threshold.
    y0 = stock.index[0]
    row = absolute[absolute["year"] == y0].iloc[0]
    manual = (stock.loc[y0].dropna() > (1.1**2 - 1.0)).sum()
    assert row["num_positive"] == manual

    # Summary CAGRs are annualized: benchmark 0.15 per 2-year window.
    summary = summarize_screen(screen, horizon=2)
    assert np.isclose(summary["cagr_benchmark"].iloc[0], 1.15**0.5 - 1.0)

    # horizon=1 reproduces the original thresholds exactly.
    screen_h1 = screen_label_criteria(stock, bmk, ["bmk+0.05"], horizon=1)
    assert np.allclose(screen_h1["threshold"], 0.15 + 0.05)


def test_sweep_with_horizon() -> None:
    stock, bmk = make_window_returns()
    df = sweep_from_returns(
        stock, bmk, [0.3], [0.6],
        num_simulations=50, model_random_seed=7, horizon=2,
    )
    assert len(df) == 1
    # Benchmark CAGR is annualized over windows * horizon years.
    expected = compute_cagr(bmk, years_per_period=2)
    assert np.isclose(df["cagr_benchmark"].iloc[0], expected)
    assert np.isclose(expected, 1.15**0.5 - 1.0)
    # q05 CAGR (when feasible) must also be annualized: it can't exceed the
    # non-annualized value for positive returns. Just check it's finite here.
    assert np.isfinite(df["cagr_custom_q05"].iloc[0])


def test_study_with_horizon_offline() -> None:
    """End-to-end study on a fabricated price cache: h=2 non-overlapping and
    overlapping, checking window bookkeeping and CAGR annualization."""
    from pickn import run_top_n_study

    tickers = [f"T{i:02d}" for i in range(16)]
    all_cols = tickers + ["SPY"]
    dates = pd.bdate_range("2018-01-02", "2021-12-31")
    rng = np.random.default_rng(3)
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, size=(len(dates), len(all_cols))), axis=0)),
        index=dates,
        columns=all_cols,
    )

    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        try:
            os.makedirs("data/cache", exist_ok=True)
            prices.to_csv("data/cache/adj_close_cache.csv", index_label="Date")

            res = run_top_n_study(
                n_values=[3],
                year_start=2018,
                year_end=2021,
                tickers=tickers,
                model_recall=0.3,
                model_precision=0.6,
                num_simulations=30,
                model_random_seed=11,
                horizon=2,
            )
            # Two non-overlapping 2-year windows: 2018-2019 and 2020-2021.
            assert list(res.yearly.index) == [2018, 2020]
            assert int(res.summary["horizon"].iloc[0]) == 2
            assert bool(res.summary["overlapping_windows"].iloc[0]) is False
            # Benchmark CAGR annualized over 2 windows x 2 years.
            growth = (1.0 + res.yearly["benchmark_return"]).prod()
            assert np.isclose(
                res.summary["cagr_benchmark"].iloc[0], growth ** (1 / 4) - 1.0
            )
            # Window return matches first/last trading day of the window.
            w = prices.loc["2018":"2019", "SPY"]
            assert np.isclose(
                res.yearly.loc[2018, "benchmark_return"], w.iloc[-1] / w.iloc[0] - 1.0
            )

            res_ov = run_top_n_study(
                n_values=[3],
                year_start=2018,
                year_end=2021,
                tickers=tickers,
                model_recall=0.3,
                model_precision=0.6,
                num_simulations=30,
                model_random_seed=11,
                horizon=2,
                overlapping=True,
            )
            # Rolling cohorts: 2018, 2019, 2020 (last window ends 2021).
            assert list(res_ov.yearly.index) == [2018, 2019, 2020]
            assert bool(res_ov.summary["overlapping_windows"].iloc[0]) is True
            # Shared cohorts must agree between the two modes.
            assert np.isclose(
                res_ov.yearly.loc[2018, "benchmark_return"],
                res.yearly.loc[2018, "benchmark_return"],
            )
        finally:
            os.chdir(cwd)


if __name__ == "__main__":
    test_formation_years()
    test_threshold_compounding()
    test_compute_cagr_years_per_period()
    test_window_universe_returns()
    test_benchmark_window_returns()
    test_screen_with_horizon()
    test_sweep_with_horizon()
    test_study_with_horizon_offline()
    print("All horizon tests passed.")
