"""Tests for the criterion feasibility screen (perfect-precision necessary
condition) and its per-criterion summary.

Synthetic returns only — no network, no price cache. Run with:
    uv run python tests/test_screen.py
"""
from pathlib import Path

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pickn import (
    compute_cagr,
    criterion_label,
    parse_criterion,
    screen_label_criteria,
    summarize_screen,
)


def make_tiny_returns():
    """Hand-checkable 2-year x 4-ticker universe with one NaN."""
    stock_yearly = pd.DataFrame(
        {
            "A": [0.30, -0.20],
            "B": [0.05, -0.05],
            "C": [-0.10, 0.15],
            "D": [0.50, np.nan],
        },
        index=[2020, 2021],
    )
    stock_yearly.index.name = "year"
    bmk_yearly = pd.Series({2020: 0.10, 2021: -0.02})
    return stock_yearly, bmk_yearly


def test_parse_criterion_and_labels() -> None:
    assert parse_criterion("0") == ("absolute", 0.0)
    assert parse_criterion("0.1") == ("absolute", 0.1)
    assert parse_criterion("-0.05") == ("absolute", -0.05)
    assert parse_criterion("bmk") == ("benchmark", 0.0)
    assert parse_criterion("bmk+0.1") == ("benchmark", 0.1)
    assert parse_criterion("bmk-0.05") == ("benchmark", -0.05)
    assert parse_criterion(" BMK+0.1 ") == ("benchmark", 0.1)

    assert criterion_label("absolute", 0.0) == ">0%"
    assert criterion_label("absolute", 0.1) == ">10%"
    assert criterion_label("benchmark", 0.0) == ">benchmark"
    assert criterion_label("benchmark", 0.1) == ">benchmark+10%"
    assert criterion_label("benchmark", -0.05) == ">benchmark-5%"


def test_screen_values() -> None:
    stock_yearly, bmk_yearly = make_tiny_returns()
    screen = screen_label_criteria(
        stock_yearly, bmk_yearly, ["0", "0.1", "bmk", "bmk+0.1"]
    )

    assert list(screen.columns) == [
        "criterion", "year", "threshold", "num_positive", "base_rate",
        "positive_mean_return", "benchmark_return", "ew_benchmark_return",
        "excess", "excess_ew",
    ]
    # 4 criteria x 2 years
    assert len(screen) == 8

    def row(criterion, year):
        matched = screen[(screen["criterion"] == criterion) & (screen["year"] == year)]
        assert len(matched) == 1
        return matched.iloc[0]

    # Equal-weight benchmark: mean over that year's universe with data.
    ew_2020 = (0.30 + 0.05 - 0.10 + 0.50) / 4
    ew_2021 = (-0.20 - 0.05 + 0.15) / 3  # D is NaN in 2021

    # >0% in 2020: positives {A, B, D}, mean (0.30 + 0.05 + 0.50) / 3.
    r = row(">0%", 2020)
    assert r["num_positive"] == 3
    assert np.isclose(r["base_rate"], 3 / 4)
    assert np.isclose(r["positive_mean_return"], (0.30 + 0.05 + 0.50) / 3)
    assert np.isclose(r["excess"], (0.30 + 0.05 + 0.50) / 3 - 0.10)
    assert np.isclose(r["ew_benchmark_return"], ew_2020)
    assert np.isclose(r["excess_ew"], (0.30 + 0.05 + 0.50) / 3 - ew_2020)

    # >0% in 2021: D is NaN so the universe is {A, B, C}; only C is positive.
    r = row(">0%", 2021)
    assert r["num_positive"] == 1
    assert np.isclose(r["base_rate"], 1 / 3)
    assert np.isclose(r["positive_mean_return"], 0.15)
    assert np.isclose(r["ew_benchmark_return"], ew_2021)
    assert np.isclose(r["excess_ew"], 0.15 - ew_2021)

    # >10% in 2020: strictly greater, so B (0.05) is out and A, D are in.
    r = row(">10%", 2020)
    assert r["num_positive"] == 2
    assert np.isclose(r["positive_mean_return"], (0.30 + 0.50) / 2)

    # >benchmark resolves to that year's benchmark return.
    r = row(">benchmark", 2020)
    assert np.isclose(r["threshold"], 0.10)
    assert r["num_positive"] == 2
    r = row(">benchmark", 2021)
    assert np.isclose(r["threshold"], -0.02)
    assert r["num_positive"] == 1  # only C (0.15); B (-0.05) is below

    # >benchmark+10% shifts the threshold by the offset.
    r = row(">benchmark+10%", 2020)
    assert np.isclose(r["threshold"], 0.20)
    r = row(">benchmark+10%", 2021)
    assert np.isclose(r["threshold"], 0.08)


def test_screen_empty_positive_set() -> None:
    stock_yearly, bmk_yearly = make_tiny_returns()
    screen = screen_label_criteria(stock_yearly, bmk_yearly, ["0.6"])

    assert (screen["num_positive"] == 0).all()
    assert (screen["base_rate"] == 0.0).all()
    assert screen["positive_mean_return"].isna().all()
    assert screen["excess"].isna().all()
    assert screen["excess_ew"].isna().all()

    summary = summarize_screen(screen)
    r = summary.loc[">60%"]
    assert r["years_no_positive"] == 2
    assert pd.isna(r["cagr_perfect"])
    assert not r["passes_screen"]
    assert not r["passes_screen_ew"]


def test_summary_verdicts() -> None:
    stock_yearly, bmk_yearly = make_tiny_returns()
    screen = screen_label_criteria(stock_yearly, bmk_yearly, ["bmk", "0"])
    summary = summarize_screen(screen)

    # Criteria keep their input order.
    assert list(summary.index) == [">benchmark", ">0%"]

    # >benchmark positives beat the benchmark by construction, every year.
    r = summary.loc[">benchmark"]
    assert r["years"] == 2
    assert r["years_beating_benchmark"] == 2
    assert r["avg_excess"] > 0
    assert r["passes_screen"]

    # cagr_perfect matches compounding the per-year positive-set means.
    per_year = screen[screen["criterion"] == ">0%"].set_index("year")
    expected = compute_cagr(per_year["positive_mean_return"])
    assert np.isclose(summary.loc[">0%", "cagr_perfect"], expected)
    assert np.isclose(
        summary.loc[">0%", "cagr_benchmark"],
        compute_cagr(bmk_yearly),
    )

    # Equal-weight verdicts: the >0% positive-set mean beats the universe
    # mean both years by construction (it drops the negatives).
    r = summary.loc[">0%"]
    assert r["years_beating_ew_benchmark"] == 2
    assert r["avg_excess_ew"] > 0
    assert r["passes_screen_ew"]
    assert np.isclose(
        r["cagr_ew_benchmark"],
        compute_cagr(per_year["ew_benchmark_return"]),
    )
    assert np.isclose(
        r["avg_ew_benchmark_return"],
        per_year["ew_benchmark_return"].mean(),
    )


def test_parsed_tuples_accepted() -> None:
    stock_yearly, bmk_yearly = make_tiny_returns()
    from_specs = screen_label_criteria(stock_yearly, bmk_yearly, ["bmk+0.1"])
    from_tuples = screen_label_criteria(
        stock_yearly, bmk_yearly, [("benchmark", 0.1)]
    )
    pd.testing.assert_frame_equal(from_specs, from_tuples)


def main() -> None:
    test_parse_criterion_and_labels()
    test_screen_values()
    test_screen_empty_positive_set()
    test_summary_verdicts()
    test_parsed_tuples_accepted()
    print("All screen tests passed.")


if __name__ == "__main__":
    main()
