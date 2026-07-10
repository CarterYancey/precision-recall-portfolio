"""Tests for the (recall, precision) sweep and its heatmap plot.

Synthetic returns only — no network, no price cache. Run with:
    uv run python tests/test_sweep.py
"""
import tempfile
from pathlib import Path

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")

from pickn import compute_cagr, plot_sweep_heatmap, sweep_from_returns


def make_returns(num_years: int = 6, num_tickers: int = 80, seed: int = 0):
    """Synthetic year x ticker returns plus a benchmark series."""
    rng = np.random.default_rng(seed)
    years = list(range(2015, 2015 + num_years))
    tickers = [f"T{i:03d}" for i in range(num_tickers)]
    stock_yearly = pd.DataFrame(
        rng.normal(0.08, 0.25, size=(num_years, num_tickers)),
        index=years,
        columns=tickers,
    )
    stock_yearly.index.name = "year"
    bmk_yearly = pd.Series(0.07, index=years)
    return stock_yearly, bmk_yearly


def test_sweep_grid_shape_and_flags() -> None:
    stock_yearly, bmk_yearly = make_returns()
    recalls = [0.1, 0.3]
    precisions = [0.5, 0.7, 0.9]

    df = sweep_from_returns(
        stock_yearly, bmk_yearly, recalls, precisions,
        num_simulations=50, model_random_seed=42,
    )

    assert len(df) == len(recalls) * len(precisions)
    assert list(df.columns) == [
        "recall", "precision", "achieved_recall_mean", "achieved_precision_mean",
        "base_rate_mean", "base_rate_min", "base_rate_max", "precision_edge_mean",
        "cagr_custom_q05", "cagr_benchmark", "custom_q05_meets_benchmark",
    ]
    # Full cross product, recall-major order (matches the CSV consumers expect).
    assert df[["recall", "precision"]].values.tolist() == [
        [r, p] for r in recalls for p in precisions
    ]

    bmk_cagr = compute_cagr(bmk_yearly)
    assert (df["cagr_benchmark"] == bmk_cagr).all()
    for _, row in df.iterrows():
        expected = (
            not pd.isna(row["cagr_custom_q05"])
            and row["cagr_custom_q05"] >= row["cagr_benchmark"]
        )
        assert row["custom_q05_meets_benchmark"] == expected


def test_base_rate_reporting() -> None:
    stock_yearly, bmk_yearly = make_returns()
    df = sweep_from_returns(
        stock_yearly, bmk_yearly, [0.1, 0.3], [0.5, 0.9],
        num_simulations=20, model_random_seed=5,
    )

    # Prevalence depends only on the labeling threshold, so it must be
    # identical across grid cells and match a direct recomputation.
    per_year = pd.Series(
        {
            y: float((stock_yearly.loc[y].dropna() > bmk_yearly[y]).mean())
            for y in stock_yearly.index
        }
    )
    assert np.allclose(df["base_rate_mean"], per_year.mean())
    assert np.allclose(df["base_rate_min"], per_year.min())
    assert np.allclose(df["base_rate_max"], per_year.max())
    assert np.allclose(
        df["precision_edge_mean"],
        df["achieved_precision_mean"] - df["base_rate_mean"],
        atol=1e-9,
    )

    # A fixed absolute threshold changes the prevalence accordingly.
    df_abs = sweep_from_returns(
        stock_yearly, bmk_yearly, [0.3], [0.9],
        num_simulations=20, model_random_seed=5, label_threshold=0.5,
    )
    per_year_abs = pd.Series(
        {
            y: float((stock_yearly.loc[y].dropna() > 0.5).mean())
            for y in stock_yearly.index
        }
    )
    assert np.allclose(df_abs["base_rate_mean"], per_year_abs.mean())


def test_sweep_seed_reproducible() -> None:
    stock_yearly, bmk_yearly = make_returns()
    kwargs = dict(num_simulations=50, model_random_seed=7)
    a = sweep_from_returns(stock_yearly, bmk_yearly, [0.2], [0.6, 0.8], **kwargs)
    b = sweep_from_returns(stock_yearly, bmk_yearly, [0.2], [0.6, 0.8], **kwargs)
    pd.testing.assert_frame_equal(a, b)

    # Cells reseed independently, so a cell's result must not depend on the
    # rest of the grid (this held in the old per-cell-study sweep too).
    c = sweep_from_returns(stock_yearly, bmk_yearly, [0.2], [0.8], **kwargs)
    pd.testing.assert_frame_equal(
        a[a["precision"] == 0.8].reset_index(drop=True), c
    )


def test_perfect_precision_beats_benchmark() -> None:
    # At precision 1.0 every selected stock beats the benchmark by construction,
    # so the 5th-percentile portfolio must too.
    stock_yearly, bmk_yearly = make_returns()
    df = sweep_from_returns(
        stock_yearly, bmk_yearly, [0.3], [1.0],
        num_simulations=50, model_random_seed=1,
    )
    row = df.iloc[0]
    assert row["achieved_precision_mean"] == 1.0
    assert row["cagr_custom_q05"] > row["cagr_benchmark"]
    assert bool(row["custom_q05_meets_benchmark"])


def test_heatmap_written_with_nan_cell() -> None:
    stock_yearly, bmk_yearly = make_returns()
    df = sweep_from_returns(
        stock_yearly, bmk_yearly, [0.1, 0.3], [0.5, 0.9],
        num_simulations=20, model_random_seed=3,
    )
    # Simulate an infeasible pair: no draws, so no q05.
    df.loc[df.index[-1], "cagr_custom_q05"] = np.nan

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "heatmap.png"
        plot_sweep_heatmap(df, output_path=str(out))
        assert out.exists() and out.stat().st_size > 0

        # An all-NaN grid must still render (flat gray) rather than crash.
        all_nan = df.assign(cagr_custom_q05=np.nan)
        out2 = Path(tmp) / "heatmap_all_nan.png"
        plot_sweep_heatmap(all_nan, output_path=str(out2))
        assert out2.exists() and out2.stat().st_size > 0


def main() -> None:
    test_sweep_grid_shape_and_flags()
    test_base_rate_reporting()
    test_sweep_seed_reproducible()
    test_perfect_precision_beats_benchmark()
    test_heatmap_written_with_nan_cell()
    print("All sweep tests passed.")


if __name__ == "__main__":
    main()
