# Notes for AI assistants and new developers

## ⚠️ Do NOT read the large data files into context

- **`SP500_HistoricalComponents_withChanges.csv` (~5.3 MB, 2,701 data rows).** Each row is a change date plus a single comma-separated string of ~500 tickers. Reading it wholesale wastes enormous amounts of tokens and tells you nothing the schema doesn't. If you need to inspect it, use shell commands instead, e.g.:
  ```bash
  head -1 SP500_HistoricalComponents_withChanges.csv          # header: date,tickers
  cut -d, -f1 SP500_HistoricalComponents_withChanges.csv | sed -n '2p;$p'   # date range: 1996-01-02 .. 2025-11-11
  ```
- **`adj_close_cache.csv`** (gitignored, generated at runtime) can grow to tens of MB — same rule applies.
- `SP500Current.csv` (~505 lines, one ticker per line) and `Precision_Recall_Tradeoff.csv` (~100 lines) are small enough to read if needed.

## Project map

- `pickn.py` — main study pipeline (`run_top_n_study`), yearly-label simulation (`simulate_year_model_selection`), grid sweep (`sweep_recall_precision_pairs`), criterion feasibility screen (`screen_criteria_feasibility`), plotting, and the yfinance download/cache layer (`download_adj_close`). Entry point: `uv run python pickn.py` — an argparse CLI with `study` (default when no subcommand is given), `sweep`, and `screen` subcommands; see `--help` on each.
- `simulate_model.py` — pure combinatorics/sampling, no network or market data. `simulate_selection` builds a random selection hitting a target (recall, precision); `estimate_num_ways` counts how many such selections exist. `pickn.py` imports from here; nothing imports `pickn.py`.
- `constituents.py` — standalone CLI for fetching point-in-time S&P 500 membership from Wikipedia revision history. Not imported by the other modules; `pickn.py` reads the committed CSV instead.

## Conventions and gotchas

- Tickers are normalized by replacing `.` with `-` (e.g. `BRK.B` → `BRK-B`) to match yfinance symbols. Keep this consistent anywhere tickers enter the system.
- The "positive" class label is the string `"TRUE"` (not a boolean) — see `positive_label` parameters.
- `get_sp500_tickers_by_year` takes only the **first** membership snapshot of each calendar year; intra-year index changes are ignored.
- Survivorship bias: per-year stock returns go through `year_universe_returns` (`pickn.py`), which keeps mid-year delistings (return measured to last available price), excludes tickers whose data only starts mid-year (reused-symbol guard), and counts constituents with no data in `StudyResult.coverage`. yfinance has no history for most delisted names, so the no-data drops remain an upward bias — don't "clean up" these guards or the coverage report.
- The cache check in `download_adj_close` verifies both ticker coverage and date coverage (`cache_covers_dates`, 7-day tolerance because the cache index holds trading days only, future `end` clamped to today); a cache built for a narrower period triggers a full re-download instead of being silently sliced. The cache slice is end-exclusive, matching yfinance's `end` convention.
- The hypothetical classifier's labeling target in `simulate_custom_portfolio_distribution` is controlled by `label_threshold` (threaded through `run_top_n_study` and `sweep_recall_precision_pairs`): `None` labels stocks against that year's benchmark return; a fixed value (e.g. `0.1`) simulates a model that picks stocks returning more than that absolute threshold. Performance is always *compared* against the actual benchmark — only the classification target changes.
- Plots are written to the repo root: `topNAndCustom_vs_spy.png` and `topNAndCustom_growth.png` from `study`, `recall_precision_heatmap.png` and `recall_precision_heatmap_ew.png` (same grid measured against the equal-weight universe benchmark) from `sweep` (skip with `--no-plots`); PNGs and the price cache are gitignored.
- Two benchmarks wherever performance is reported: the cap-weighted `--benchmark` ticker (SPY) and the equal-weighted universe mean (`equal_weight_benchmark_returns` = `stock_yearly.mean(axis=1)`, bias-consistent with the portfolios). Excess vs SPY mixes selection skill with the equal-weight effect; excess vs the EW benchmark isolates skill. Labeling with `label_threshold=None` still targets the cap-weighted benchmark's return — only the performance comparison gained the second baseline.
- Portfolio aggregation (`portfolio_return`, backing top-N/bottom-N/model-selected returns) accepts an optional ticker-indexed `weights` series for future cap-weighted variants; nothing passes it yet because yfinance lacks historical share counts (blocked on Sharadar — see TODO.md).
- The sweep computes prices and per-year universe returns once (`prepare_universe_returns`) and re-runs only the classifier simulation per (recall, precision) grid cell (`compute_custom_stats`, looped by `sweep_from_returns`). Don't reintroduce a per-cell `run_top_n_study` call — top-N results don't depend on the grid.
- "Miss the super-performers" mode: `exclude_top` (CLI `--exclude-top` on `study`/`sweep`, threaded down to `simulate_custom_portfolio_distribution`) bars the top-k (value >= 1) or top fraction (value in (0,1), `0.1` = top decile) of each year's positives *by return* from the TP pool. The mechanism is the `exclude_names` parameter on `simulate_selection`/`estimate_num_ways` (`simulate_model.py`): excluded names stay in the confusion-matrix denominators — an excluded positive is a forced false negative, so `TP = round(recall*T)` is unchanged and must be filled from ordinary positives — but can't be drawn, and ways-counting uses the shrunken pools. Don't "simplify" by dropping excluded rows from the labeled df; that would silently shrink recall's denominator. Per-year counts surface as `excluded_top` (`custom_stats`, yearly table) and `excluded_top_mean` (summary, sweep rows); the pessimism gap is read by comparing runs with and without the flag.
- Base-rate reporting: `custom_stats` carries a per-year `base_rate` (label prevalence `T/(T+F)` — the precision a random picker gets for free), surfaced as `label_base_rate` in the study's yearly table and as `base_rate_mean/min/max` + `precision_edge_mean` in sweep rows. It depends only on the labeling threshold, never on (recall, precision) — keep it that way, it's the interpretability baseline for the whole study.
- The `screen` subcommand (`screen_label_criteria`/`summarize_screen`) is a *necessary-condition* check: per-year mean return of all stocks meeting a criterion = the precision-1/recall-1 portfolio. Criterion specs: plain numbers are absolute thresholds, `bmk`±offset is benchmark-relative (`parse_criterion`) — the `bmk` threshold is always the cap-weighted benchmark, while excess/verdicts are reported against both benchmarks (`excess`/`passes_screen` vs cap-weighted, `excess_ew`/`passes_screen_ew` vs the equal-weight universe mean). It deliberately runs no selection simulation; don't wire it into the sweep loop.
- Tests live in `tests/` as plain-assert scripts (no pytest dependency), each runnable standalone and none needing network: `for t in tests/test_*.py; do uv run python "$t"; done`. A full `pickn.py` run needs network access to yfinance and can take a long time uncached; prefer testing the pure functions (`simulate_model.py`, `year_universe_returns`) when validating changes.

## Environment

- Python >= 3.12, dependencies managed with `uv` (`uv sync`, then `uv run python ...`). See `pyproject.toml`.
