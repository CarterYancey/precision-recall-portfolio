"""Tests for the equal-weight universe benchmark and the optional `weights`
aggregation in the portfolio-return functions.

Synthetic returns only — no network, no price cache. Run with:
    uv run python tests/test_benchmarks_and_weights.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pickn import (
    bottom_n_portfolio_return,
    equal_weight_benchmark_returns,
    model_selected_portfolio_return,
    portfolio_return,
    top_n_portfolio_return,
)


def test_equal_weight_benchmark_returns() -> None:
    stock_yearly = pd.DataFrame(
        {
            "A": [0.10, 0.20, np.nan],
            "B": [0.30, np.nan, np.nan],
            "C": [-0.10, 0.40, 0.05],
        },
        index=[2020, 2021, 2022],
    )
    ew = equal_weight_benchmark_returns(stock_yearly)

    # Per-year mean over constituents *with data* — NaNs (tickers not in that
    # year's universe / no price history) are skipped, not zero-filled.
    assert np.isclose(ew[2020], (0.10 + 0.30 - 0.10) / 3)
    assert np.isclose(ew[2021], (0.20 + 0.40) / 2)
    assert np.isclose(ew[2022], 0.05)
    assert list(ew.index) == [2020, 2021, 2022]


def test_portfolio_return_equal_weight_default() -> None:
    returns = pd.Series([0.1, 0.2, 0.3], index=["A", "B", "C"])
    assert np.isclose(portfolio_return(returns), 0.2)


def test_portfolio_return_weighted() -> None:
    returns = pd.Series([0.1, 0.3], index=["A", "B"])
    weights = pd.Series({"A": 3.0, "B": 1.0})
    assert np.isclose(portfolio_return(returns, weights), (0.1 * 3 + 0.3 * 1) / 4)

    # Missing or non-positive weights drop the name and renormalize.
    weights_partial = pd.Series({"A": 2.0, "B": 0.0})
    assert np.isclose(portfolio_return(returns, weights_partial), 0.1)
    weights_missing = pd.Series({"B": 5.0})
    assert np.isclose(portfolio_return(returns, weights_missing), 0.3)

    # No usable weights at all -> NaN, not a crash.
    assert np.isnan(portfolio_return(returns, pd.Series({"Z": 1.0})))


def test_top_bottom_n_with_weights() -> None:
    year_returns = pd.Series(
        [0.50, 0.20, 0.00, -0.30], index=["A", "B", "C", "D"]
    )
    # Selection is by return regardless of weights.
    assert np.isclose(top_n_portfolio_return(year_returns, 2), 0.35)
    assert np.isclose(bottom_n_portfolio_return(year_returns, 2), -0.15)

    caps = pd.Series({"A": 1.0, "B": 3.0, "C": 1.0, "D": 1.0})
    assert np.isclose(
        top_n_portfolio_return(year_returns, 2, weights=caps),
        (0.50 * 1 + 0.20 * 3) / 4,
    )
    assert np.isclose(
        bottom_n_portfolio_return(year_returns, 2, weights=caps),
        (0.00 * 1 - 0.30 * 1) / 2,
    )


def test_model_selected_with_weights() -> None:
    year_returns = pd.Series([0.10, 0.40, -0.20], index=["A", "B", "C"])
    assert np.isclose(
        model_selected_portfolio_return(year_returns, ["A", "B"]), 0.25
    )
    caps = pd.Series({"A": 1.0, "B": 4.0})
    assert np.isclose(
        model_selected_portfolio_return(year_returns, ["A", "B"], weights=caps),
        (0.10 * 1 + 0.40 * 4) / 5,
    )
    # Existing behavior unchanged: empty selection or no data -> NaN.
    assert np.isnan(model_selected_portfolio_return(year_returns, []))
    assert np.isnan(model_selected_portfolio_return(year_returns, ["Z"]))


def main() -> None:
    test_equal_weight_benchmark_returns()
    test_portfolio_return_equal_weight_default()
    test_portfolio_return_weighted()
    test_top_bottom_n_with_weights()
    test_model_selected_with_weights()
    print("All benchmark/weights tests passed.")


if __name__ == "__main__":
    main()
