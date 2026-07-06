# TODO / recommended improvements

## Robustness / data quality

- [ ] **Cache validation in `download_adj_close`** checks tickers only, not the date range; a cache built for 2012–2024 silently truncates a 2000–2024 request. Validate date coverage (and consider a per-ticker incremental update rather than full re-download when one ticker is missing).
- [ ] **Intra-year membership changes are ignored**: `get_sp500_tickers_by_year` uses only the first snapshot of each year. Consider using membership as of each year's first trading day, or handling additions/removals mid-year.
- [x] **Missing-data handling**: delisted tickers with no yfinance history were silently dropped, re-introducing survivorship bias into the "point-in-time" study. Partially fixed via `year_universe_returns`: mid-year delistings are now kept (return measured to last available price), tickers whose data starts mid-year are excluded (reused-symbol guard), and a per-year coverage report (`StudyResult.coverage`, also printed) counts the no-data drops so the residual bias is visible. Remaining work tracked below.
- [ ] **Source delisted-stock prices from a survivorship-bias-free provider** (CRSP, Sharadar/Nasdaq Data Link, Norgate, EODHD). A 2012-universe spot check found ~80% of since-departed constituents have no data on Yahoo for their constituent years, so early-year universes are missing roughly a third of members — disproportionately the worst outcomes. No yfinance-side change can fix this.

## Features / analysis (original planned enhancements)

- [ ] Sweep across multiple recall and precision values to understand performance sensitivity. *(Partially done: `sweep_recall_precision_pairs` exists and its output is now clean scalars, but it redundantly re-runs the full study — including top-N computation — per grid cell.)*
- [ ] Compute total expected returns and their variance for the hypothetical model-driven portfolios to quantify the recall/precision needed for consistent outperformance (hypothesis: sufficiently high precision can beat the market even with low recall).
- [ ] Consider cap-weighted (not just equal-weighted) portfolio variants.
- [ ] Plot the recall/precision sweep as a heatmap (precision × recall → q05 CAGR vs. benchmark).

## Engineering hygiene

- [ ] Add a test suite (start with `simulate_model.py` — it's pure and fast: feasibility edge cases, rounding scheme, `estimate_num_ways` counts). *(Started: `tests/test_year_universe_returns.py` covers the survivorship-handling return computation with plain asserts.)*
- [ ] Add a CLI (argparse) to `pickn.py` instead of editing the `__main__` block to change `n_values`/years/recall/precision.
- [ ] Fill in the `description` field in `pyproject.toml` (currently "Add your description here").
- [ ] Remove commented-out debug prints in `simulate_custom_portfolio_distribution` and the commented bottom-N plotting block in `plot_results` once decisions are final.
