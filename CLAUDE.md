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

- `pickn.py` — main study pipeline (`run_top_n_study`), yearly-label simulation (`simulate_year_model_selection`), grid sweep (`sweep_recall_precision_pairs`), plotting, and the yfinance download/cache layer (`download_adj_close`). Entry point: `uv run python pickn.py`.
- `simulate_model.py` — pure combinatorics/sampling, no network or market data. `simulate_selection` builds a random selection hitting a target (recall, precision); `estimate_num_ways` counts how many such selections exist. `pickn.py` imports from here; nothing imports `pickn.py`.
- `constituents.py` — standalone CLI for fetching point-in-time S&P 500 membership from Wikipedia revision history. Not imported by the other modules; `pickn.py` reads the committed CSV instead.

## Conventions and gotchas

- Tickers are normalized by replacing `.` with `-` (e.g. `BRK.B` → `BRK-B`) to match yfinance symbols. Keep this consistent anywhere tickers enter the system.
- The "positive" class label is the string `"TRUE"` (not a boolean) — see `positive_label` parameters.
- `get_sp500_tickers_by_year` takes only the **first** membership snapshot of each calendar year; intra-year index changes are ignored.
- The cache check in `download_adj_close` verifies ticker coverage but **not** date coverage; a stale cache with a shorter date range is silently sliced. Delete `adj_close_cache.csv` when changing the study period.
- `simulate_custom_portfolio_distribution` currently **hardcodes `benchmark_return = 0.1`**, ignoring the SPY return passed in. This is a known issue (see `TODO.md`) — don't "helpfully" rely on the parameter being used until it's fixed.
- Plots are written to the repo root as `topNAndCustom_vs_spy.png` and `topNAndCustom_growth.png`; PNGs and the price cache are gitignored.
- There is no test suite yet. A full `pickn.py` run needs network access to yfinance and can take a long time uncached; prefer testing `simulate_model.py` functions (pure, fast) when validating changes.

## Environment

- Python >= 3.12, dependencies managed with `uv` (`uv sync`, then `uv run python ...`). See `pyproject.toml`.
