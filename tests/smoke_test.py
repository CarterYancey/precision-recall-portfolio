"""Offline smoke test: fabricate an adj_close_cache.csv so download_adj_close
never hits the network, then run run_top_n_study and sweep_recall_precision_pairs
and check the summary/sweep cells are scalars (not embedded Series/sets)."""
import os
import sys

sys.path.insert(0, "/home/user/precision-recall-portfolio")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

tickers = [f"T{i:02d}" for i in range(20)]
all_cols = tickers + ["SPY"]
dates = pd.bdate_range("2020-01-02", "2021-12-31")
rng = np.random.default_rng(0)
prices = pd.DataFrame(
    100 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, size=(len(dates), len(all_cols))), axis=0)),
    index=dates,
    columns=all_cols,
)
prices.to_csv("adj_close_cache.csv", index_label="Date")

from pickn import run_top_n_study, sweep_recall_precision_pairs

res = run_top_n_study(
    n_values=[2, 5],
    year_start=2020,
    year_end=2021,
    tickers=tickers,
    model_recall=0.3,
    model_precision=0.5,
    num_simulations=50,
    model_random_seed=42,
)

print("\n=== custom_stats ===")
print(res.custom_stats)
print("\n=== summary ===")
print(res.summary.round(4))

# every summary cell must be scalar
for col in res.summary.columns:
    for v in res.summary[col]:
        assert np.isscalar(v) or isinstance(v, (int, float, np.floating, np.integer)), (
            f"non-scalar cell in summary[{col!r}]: {type(v)}"
        )
assert "achieved_recall" in res.custom_stats.columns
assert "achieved_precision" in res.custom_stats.columns
assert res.custom_stats["achieved_recall"].notna().any()

df = sweep_recall_precision_pairs(
    [0.3], [0.5],
    n_values=[2],
    year_start=2020,
    year_end=2021,
    tickers=tickers,
    num_simulations=50,
    model_random_seed=42,
)
for col in df.columns:
    for v in df[col]:
        assert not isinstance(v, (set, pd.Series, list)), f"non-scalar in sweep[{col!r}]: {type(v)}"

out = pd.read_csv("Precision_Recall_Tradeoff.csv")
print("\n=== Precision_Recall_Tradeoff.csv ===")
print(out)
assert not out.astype(str).apply(lambda s: s.str.contains(r"[{}]").any()).any(), "stringified sets in CSV"
print("\nALL CHECKS PASSED")

