"""Tests for the survivorship-bias handling in year_universe_returns.

Pure pandas, no network. Run with:
    uv run python tests/test_year_universe_returns.py
"""
import numpy as np
import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pickn import year_universe_returns


def build_adj() -> pd.DataFrame:
    """One year of synthetic daily prices exercising every coverage case."""
    days = pd.bdate_range("2020-01-02", "2020-12-31")
    n = len(days)
    adj = pd.DataFrame(index=days, dtype=float)

    # FULL: trades all year, doubles.
    adj["FULL"] = np.linspace(100.0, 200.0, n)

    # DELIST: crashes 100 -> 10 and stops trading at mid-year (a bankruptcy).
    half = n // 2
    delist = np.full(n, np.nan)
    delist[:half] = np.linspace(100.0, 10.0, half)
    adj["DELIST"] = delist

    # LATE: data only begins in the second half (reused ticker / later IPO).
    late = np.full(n, np.nan)
    late[half:] = np.linspace(50.0, 55.0, n - half)
    adj["LATE"] = late

    # EMPTY: column exists but has no prices at all.
    adj["EMPTY"] = np.nan

    # Benchmark keeps the full trading calendar in the index.
    adj["SPY"] = np.linspace(300.0, 330.0, n)
    return adj


def main() -> None:
    adj = build_adj()
    universe = ["FULL", "DELIST", "LATE", "EMPTY", "GONE"]  # GONE: not downloaded at all
    returns, stats = year_universe_returns(adj, 2020, universe)

    # FULL: plain first-to-last return.
    assert abs(returns["FULL"] - 1.0) < 1e-9, returns["FULL"]

    # DELIST is the point of the fix: it must be present, with the crash
    # measured to its last available price, not dropped as NaN.
    assert "DELIST" in returns.index, "mid-year delisting was dropped"
    assert abs(returns["DELIST"] - (10.0 / 100.0 - 1.0)) < 1e-9, returns["DELIST"]

    # LATE must be excluded: its data starting mid-year means the symbol
    # likely belongs to a different company than the year's constituent.
    assert "LATE" not in returns.index, "late-start ticker leaked into universe"

    # No-data tickers can't be recovered; they are counted, not silently lost.
    assert "EMPTY" not in returns.index
    assert "GONE" not in returns.index

    assert stats == {
        "members": 5,
        "no_data": 2,
        "late_start": 1,
        "partial_year": 1,
        "full_year": 1,
        "used": 2,
        "coverage": 2 / 5,
    }, stats

    # Empty year: everything is no_data, coverage NaN-safe.
    empty_returns, empty_stats = year_universe_returns(adj, 1999, universe)
    assert empty_returns.empty
    assert empty_stats["used"] == 0 and np.isnan(empty_stats["coverage"])

    # A single-print ticker (one non-NaN price) can't produce a return.
    adj_one = adj.copy()
    one = np.full(len(adj_one), np.nan)
    one[0] = 42.0
    adj_one["ONE"] = one
    r_one, s_one = year_universe_returns(adj_one, 2020, ["ONE"])
    assert r_one.empty and s_one["no_data"] == 1

    print("All year_universe_returns tests passed.")


if __name__ == "__main__":
    main()
