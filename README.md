## Overview

The **pickN_simulation** project explores how portfolios composed of the top-performing S&P 500 stocks compare to an SPY benchmark across calendar years. It also contains utilities for simulating selections that target specific recall/precision trade-offs and for labeling tickers by yearly SPY outperformance.

The repository currently includes:

- `pickn.py`: downloads year-specific S&P 500 constituent prices, computes top-N/bottom-N/custom-N yearly returns, and plots them against SPY.
- `simulate_model.py`: simulates model selections with requested recall and precision, returning the achieved metrics and combinatorial counts of possible selections.
- Pre-generated plots (`topN_vs_spy.png`, `topAndBottomN_vs_spy.png`, `topAndCustomN_vs_spy.png`) and S&P 500 constituent lists.

## Setup

1. Sync dependencies with `uv` (creates/uses `.venv` automatically):
   ```bash
   uv sync
   ```
2. Activate the environment if you prefer to run scripts directly:
   ```bash
   source .venv/bin/activate
   ```
   You can also prefix commands with `uv run` instead of activating, e.g. `uv run python pickn.py`.

## Running the top-N study

Execute `pickn.py` to download data (via `yfinance`), run the study, and generate a plot:

```bash
uv run python pickn.py
```

By default the script:
- Uses year-specific S&P 500 membership from `SP500_HistoricalComponents_withChanges.csv` to reduce survivorship bias.
- Compares multiple `n_values` (configured near the bottom of the script) against the SPY benchmark.
- Runs repeated model simulations to summarize custom portfolio return distributions (mean, std, 5–95% band).
- Saves a plot to `topAndCustomN_vs_spy.png` and prints summary tables to stdout.

To customize the analysis, adjust the parameters passed to `run_top_n_study` in `pickn.py` (e.g., change `n_values`, `year_start`/`year_end`, or supply your own ticker universe). You can also provide `tickers_by_year` to override the per-year membership, or a fixed `tickers` list for backward compatibility.

## Simulating recall/precision selections

`simulate_model.py` provides `simulate_selection` for exploring achievable recall/precision pairs on tabular data. A minimal example is included in the module’s `__main__` block:

```bash
python simulate_model.py
```

This prints the requested vs. achieved metrics, the sampled names, and the number of combinatorial ways a selection can satisfy the request under the rounding scheme documented in the module docstrings.

### Labeling a year by SPY outperformance

Use `simulate_year_model_selection` in `simulate_model.py` to build a yearly dataset where each ticker is labeled `TRUE` when its calendar-year return beats SPY, and then run the selection simulation:

```bash
uv run python -c "from pickn import simulate_year_model_selection; print(simulate_year_model_selection(2010, recall=.1, precision=.9, cache_path='adj_close_cache.csv'))"
```

This helper downloads adjusted close prices for the requested tickers/year, computes calendar-year returns, labels tickers by benchmark outperformance, and returns a `SimulationResult`.

## Caching adjusted close data

`download_adj_close` accepts a `cache_path` (default `adj_close_cache.csv`). When the cache file exists and contains the required tickers/date range, the data is loaded locally instead of re-downloading from `yfinance`. When the cache is missing or insufficient, the download runs and the adjusted close data is persisted for reuse.

## Notes and caveats

- `yfinance` is used for historical price data; results depend on data availability and may vary slightly over time.

## Planned enhancements

1. Sweep across multiple recall and precision values to understand performance sensitivity.
2. Compute total expected returns and their variance for the hypothetical model-driven portfolios to quantify the recall/precision needed for consistent outperformance (hypothesis: sufficiently high precision can beat the market even with low recall).
