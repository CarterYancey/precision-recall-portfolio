# TODO / recommended improvements

## Correctness bugs

- [x] **`simulate_custom_portfolio_distribution` ignores its `benchmark_return` argument** (`pickn.py`): the line `benchmark_return = 0.1` overwrote the actual SPY return for the year. Fixed: the labeling target is now an explicit `label_threshold` parameter (threaded through `run_top_n_study` and `sweep_recall_precision_pairs`) — set it to a fixed value like `0.1` to simulate a classifier targeting ">10% return", or leave it `None` to target that year's benchmark return. Portfolio performance is always compared against the actual benchmark.
- [x] **Early `break` skews the simulated distribution** (`simulate_custom_portfolio_distribution`): the loop appended `None` and `break`ed when achieved metrics fell outside the ±0.1 tolerance. Fixed: a single probe draw now validates feasibility before the loop (achieved metrics are deterministic given the label counts), and the loop runs `min(num_simulations, num_ways)` clean draws.
- [x] **`summary` rows embed whole Series**: `run_top_n_study` put `custom_stats["achieved_recall"]`/`["achieved_precision"]` (per-year Series) into scalar summary cells, which is why `Precision_Recall_Tradeoff.csv` contained stringified sets like `"{0.13, 0.14, ...}"`. Fixed: per-year achieved metrics live in `custom_stats` columns; the summary carries scalar mean/min/max aggregates (`custom_recall_mean` etc.), and the sweep reports the requested `recall`/`precision` targets plus `achieved_*_mean`. (The committed `Precision_Recall_Tradeoff.csv` still has the old format until the sweep is re-run.)
- [x] **`bottom_n_portfolio_return` docstring** says "top-n" — copy-paste error. Fixed.
- [x] **Stale module docstring in `pickn.py`**: it still said "Uses *current* S&P 500 constituents from Wikipedia by default -> survivorship bias", but the default is now the point-in-time historical CSV. Fixed (with a note that missing-price delistings still leak survivorship bias).

## Robustness / data quality

- [ ] **Cache validation in `download_adj_close`** checks tickers only, not the date range; a cache built for 2012–2024 silently truncates a 2000–2024 request. Validate date coverage (and consider a per-ticker incremental update rather than full re-download when one ticker is missing).
- [ ] **Intra-year membership changes are ignored**: `get_sp500_tickers_by_year` uses only the first snapshot of each year. Consider using membership as of each year's first trading day, or handling additions/removals mid-year.
- [x] **Missing-data handling**: delisted tickers with no yfinance history were silently dropped, re-introducing survivorship bias into the "point-in-time" study. Partially fixed via `year_universe_returns`: mid-year delistings are now kept (return measured to last available price), tickers whose data starts mid-year are excluded (reused-symbol guard), and a per-year coverage report (`StudyResult.coverage`, also printed) counts the no-data drops so the residual bias is visible. Remaining work tracked below.
- [ ] **Source delisted-stock prices from a survivorship-bias-free provider** (CRSP, Sharadar/Nasdaq Data Link, Norgate, EODHD). A 2012-universe spot check found ~80% of since-departed constituents have no data on Yahoo for their constituent years, so early-year universes are missing roughly a third of members — disproportionately the worst outcomes. No yfinance-side change can fix this.
- [ ] `datetime.utcnow()` in `constituents.py` is deprecated since Python 3.12 — use `datetime.now(timezone.utc)`.
- [ ] `constituents.py` and `pickn.py` are disconnected: `pickn.py` reads the committed CSV while `constituents.py` writes its own cache format (`year,ticker` rows). Either wire `get_sp500_tickers_by_year` to accept the constituents cache, or document `constituents.py` as a data-regeneration tool.

## Features / analysis (original planned enhancements)

- [ ] Sweep across multiple recall and precision values to understand performance sensitivity. *(Partially done: `sweep_recall_precision_pairs` exists and its output is now clean scalars, but it redundantly re-runs the full study — including top-N computation — per grid cell.)*
- [ ] Compute total expected returns and their variance for the hypothetical model-driven portfolios to quantify the recall/precision needed for consistent outperformance (hypothesis: sufficiently high precision can beat the market even with low recall).
- [ ] Add transaction-cost / turnover assumptions to make the simulated portfolios more realistic.
- [ ] Consider cap-weighted (not just equal-weighted) portfolio variants.
- [ ] Plot the recall/precision sweep as a heatmap (precision × recall → q05 CAGR vs. benchmark).

## Engineering hygiene

- [ ] Add a test suite (start with `simulate_model.py` — it's pure and fast: feasibility edge cases, rounding scheme, `estimate_num_ways` counts). *(Started: `tests/test_year_universe_returns.py` covers the survivorship-handling return computation with plain asserts.)*
- [ ] Add a CLI (argparse) to `pickn.py` instead of editing the `__main__` block to change `n_values`/years/recall/precision.
- [x] Commit a `uv.lock` for reproducible environments.
- [ ] Fill in the `description` field in `pyproject.toml` (currently "Add your description here").
- [ ] Add a LICENSE file.
- [ ] Remove commented-out debug prints in `simulate_custom_portfolio_distribution` and the commented bottom-N plotting block in `plot_results` once decisions are final.
