"""Sanity checks for the simulate_custom_portfolio_distribution fixes."""
import numpy as np
import pandas as pd

from pickn import simulate_custom_portfolio_distribution

rng_returns = np.random.default_rng(42)
# 200 fake tickers with yearly returns centered near 8%
year_returns = pd.Series(
    rng_returns.normal(0.08, 0.25, size=200),
    index=[f"T{i}" for i in range(200)],
)

# 1) benchmark_return is respected: labeling threshold changes the positive
#    pool, so mean simulated return should differ across benchmarks.
stats_low = simulate_custom_portfolio_distribution(
    year_returns, benchmark_return=0.0, recall=0.2, precision=0.7,
    num_simulations=300, rng=np.random.default_rng(0),
)
stats_high = simulate_custom_portfolio_distribution(
    year_returns, benchmark_return=0.20, recall=0.2, precision=0.7,
    num_simulations=300, rng=np.random.default_rng(0),
)
n_pos_low = int((year_returns > 0.0).sum())
n_pos_high = int((year_returns > 0.20).sum())
print(f"positives @ 0% threshold: {n_pos_low}, @ +20% threshold: {n_pos_high}")
print("stats_low :", {k: round(v, 4) if isinstance(v, float) else v for k, v in stats_low.items()})
print("stats_high:", {k: round(v, 4) if isinstance(v, float) else v for k, v in stats_high.items()})
assert n_pos_low != n_pos_high
assert stats_low["mean"] != stats_high["mean"], "benchmark_return still ignored!"
assert stats_high["mean"] > stats_low["mean"], "higher threshold should select higher-return positives"

# 2) Feasible request runs the full number of simulations (no early abandon).
assert stats_low["count"] == 300, stats_low["count"]
assert stats_high["count"] == 300, stats_high["count"]

# 3) Infeasible request: with 1 pos / 1 neg and precision=0.75, FP rounds to 0
#    so achieved precision is 1.0, outside 0.75±0.1 -> count=0, NaN stats.
tiny = pd.Series([0.5, -0.5], index=["A", "B"])  # 1 positive, 1 negative at bmk=0
stats_infeasible = simulate_custom_portfolio_distribution(
    tiny, benchmark_return=0.0, recall=1.0, precision=0.75,
    num_simulations=50, rng=np.random.default_rng(0),
)
print("infeasible:", stats_infeasible)
assert stats_infeasible["count"] == 0
assert np.isnan(stats_infeasible["mean"])

# 4) num_ways cap: 2 positives, choose 1 (recall=0.5, precision=1.0) -> 2 ways,
#    so count must be capped at 2 even though 50 sims were requested.
small = pd.Series([0.4, 0.6, -0.2, -0.3], index=["A", "B", "C", "D"])
stats_cap = simulate_custom_portfolio_distribution(
    small, benchmark_return=0.0, recall=0.5, precision=1.0,
    num_simulations=50, rng=np.random.default_rng(0),
)
print("num_ways cap:", stats_cap)
assert stats_cap["count"] == 2, stats_cap["count"]

print("\nAll checks passed.")

# 5) label_threshold: fixed threshold overrides the benchmark for labeling.
#    Same benchmark_return, different label_threshold -> different label pools.
stats_bmk_labeled = simulate_custom_portfolio_distribution(
    year_returns, benchmark_return=0.08, recall=0.2, precision=0.7,
    num_simulations=300, rng=np.random.default_rng(0),
)
stats_fixed_10pct = simulate_custom_portfolio_distribution(
    year_returns, benchmark_return=0.08, recall=0.2, precision=0.7,
    num_simulations=300, label_threshold=0.10, rng=np.random.default_rng(0),
)
stats_explicit_bmk = simulate_custom_portfolio_distribution(
    year_returns, benchmark_return=0.10, recall=0.2, precision=0.7,
    num_simulations=300, rng=np.random.default_rng(0),
)
print("labeled vs bmk 8%   :", {k: round(v, 4) for k, v in stats_bmk_labeled.items()})
print("labeled vs fixed 10%:", {k: round(v, 4) for k, v in stats_fixed_10pct.items()})
assert stats_fixed_10pct["mean"] != stats_bmk_labeled["mean"]
# label_threshold=0.10 must be equivalent to labeling against a 10% benchmark
assert stats_fixed_10pct == stats_explicit_bmk, "fixed threshold should match equivalent benchmark labeling"

print("label_threshold checks passed.")
