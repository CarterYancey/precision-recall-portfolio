# TODO / recommended improvements

## Robustness / data quality

- [x] **Cache validation in `download_adj_close`** checks tickers only, not the date range; a cache built for 2012–2024 silently truncates a 2000–2024 request. Fixed: `cache_covers_dates` rejects caches whose date span doesn't cover the request (7-day tolerance for non-trading-day endpoints, future `end` clamped to today), triggering a full re-download; the cache slice also now uses yfinance's end-exclusive convention so cached and fresh results match. Covered by `tests/test_download_adj_close_cache.py`. *Remaining nice-to-have: per-ticker/per-period incremental update instead of a full re-download on any mismatch.*
- [ ] **Intra-year membership changes are ignored**: `get_sp500_tickers_by_year` uses only the first snapshot of each year. Consider using membership as of each year's first trading day, or handling additions/removals mid-year.
- [x] **Missing-data handling**: delisted tickers with no yfinance history were silently dropped, re-introducing survivorship bias into the "point-in-time" study. Partially fixed via `year_universe_returns`: mid-year delistings are now kept (return measured to last available price), tickers whose data starts mid-year are excluded (reused-symbol guard), and a per-year coverage report (`StudyResult.coverage`, also printed) counts the no-data drops so the residual bias is visible. Remaining work tracked below.
- [ ] **Source delisted-stock prices from a survivorship-bias-free provider** (CRSP, Sharadar/Nasdaq Data Link, Norgate, EODHD). A 2012-universe spot check found ~80% of since-departed constituents have no data on Yahoo for their constituent years, so early-year universes are missing roughly a third of members — disproportionately the worst outcomes. No yfinance-side change can fix this. *Provider research done — see `docs/delisted-data-providers.md`: recommendation is the Sharadar Core US Equities Bundle (permaticker IDs solve ticker reuse; SF1 gives point-in-time fundamentals for delisted names; price history starts 1998). Remaining: confirm pricing behind the Nasdaq Data Link login, subscribe, and implement the SEP-backed loader + permaticker mapping sketched in the doc.*

## Features / analysis (original planned enhancements)

- [ ] Sweep across multiple recall and precision values to understand performance sensitivity. *(Partially done: `sweep_recall_precision_pairs` exists and its output is now clean scalars, but it redundantly re-runs the full study — including top-N computation — per grid cell.)*
- [ ] Compute total expected returns and their variance for the hypothetical model-driven portfolios to quantify the recall/precision needed for consistent outperformance (hypothesis: sufficiently high precision can beat the market even with low recall).
- [ ] Consider cap-weighted (not just equal-weighted) portfolio variants.
- [ ] Plot the recall/precision sweep as a heatmap (precision × recall → q05 CAGR vs. benchmark).

## Engineering hygiene

- [x] Add a test suite (start with `simulate_model.py` — it's pure and fast: feasibility edge cases, rounding scheme, `estimate_num_ways` counts). *(Done: `tests/test_simulate_model.py` covers feasibility edge cases, the rounding scheme incl. banker's rounding, zero-positive datasets, strict mode, and brute-force-verified `estimate_num_ways` counts; `tests/test_year_universe_returns.py` covers survivorship handling; `tests/test_download_adj_close_cache.py` covers cache validation with a stubbed yfinance. All plain-assert, no network.)*
- [ ] Add a CLI (argparse) to `pickn.py` instead of editing the `__main__` block to change `n_values`/years/recall/precision.
- [ ] Fill in the `description` field in `pyproject.toml` (currently "Add your description here").
- [ ] Remove commented-out debug prints in `simulate_custom_portfolio_distribution` and the commented bottom-N plotting block in `plot_results` once decisions are final.
