# pickN simulation — precision/recall portfolio study

## Overview

This project explores how portfolios composed of the top-performing S&P 500 stocks compare to an SPY benchmark across calendar years, and how good a stock-picking model would need to be (in precision/recall terms) to beat the market. It contains utilities for simulating model selections that target specific recall/precision trade-offs and for labeling tickers by yearly SPY outperformance.

**Core question:** if a hypothetical model picks stocks with recall *r* and precision *p* (where "positive" means "beat SPY that year"), what distribution of portfolio returns does it achieve — and what (*r*, *p*) is enough to consistently outperform?

## Repository layout

| Path | What it is |
|------|------------|
| `pickn.py` | Main study: downloads year-specific S&P 500 constituent prices, computes top-N/bottom-N/model-simulated yearly returns, and plots them against SPY. Also home of `simulate_year_model_selection` and `sweep_recall_precision_pairs`. |
| `simulate_model.py` | Standalone module: `simulate_selection` constructs a random selection hitting a requested (recall, precision) on labeled tabular data; `estimate_num_ways` counts the combinatorial ways such a selection can exist. No market-data dependencies. |
| `constituents.py` | Optional utility (not imported by the other scripts): fetches point-in-time S&P 500 membership from historical Wikipedia revisions and caches it to `sp500_constituents_wikipedia_by_year.csv`. Run as a CLI: `uv run python constituents.py --year-start 2000 --year-end 2024`. |
| `SP500_HistoricalComponents_withChanges.csv` | **Large (~5.3 MB, ~2,700 rows).** Point-in-time S&P 500 membership, 1996-01-02 through 2025-11-11. One row per change date; columns are `date` and `tickers` (a single comma-separated string of ~500 tickers). Used by `get_sp500_tickers_by_year` to reduce survivorship bias. Don't open it casually — see `CLAUDE.md`. |
| `SP500Current.csv` | Current S&P 500 ticker list (one `Symbol` column, ~505 rows). Used by `get_sp500_tickers_from_csv` as a simpler, survivorship-biased universe. |
| `Precision_Recall_Tradeoff.csv` | Output of `sweep_recall_precision_pairs`: for each (recall, precision) pair, the 5th-percentile CAGR of the simulated portfolios vs. the benchmark CAGR. |
| `adj_close_cache.csv` | Generated at runtime (gitignored). Cached adjusted-close prices from yfinance. |
| `*.png` | Generated at runtime (gitignored). `topNAndCustom_vs_spy.png` (yearly returns) and `topNAndCustom_growth.png` (growth of $100). |
| `TODO.md` | Known issues and planned improvements. |
| `CLAUDE.md` | Notes for AI coding assistants working in this repo. |

## Setup

Requires Python >= 3.12.

1. Sync dependencies with [`uv`](https://docs.astral.sh/uv/) (creates/uses `.venv` automatically):
   ```bash
   uv sync
   ```
2. Either activate the environment (`source .venv/bin/activate`) or prefix commands with `uv run`, e.g. `uv run python pickn.py`.

Without `uv`, a plain `pip install yfinance pandas numpy matplotlib lxml html5lib requests` into any 3.12+ environment also works.

## Running the top-N study

```bash
uv run python pickn.py            # same as `uv run python pickn.py study`
```

The first run downloads price data from yfinance for every historical constituent (thousands of tickers) — expect it to take a while and produce a large `adj_close_cache.csv`. Subsequent runs load from the cache.

With default options, the script:

- Uses year-specific S&P 500 membership from `SP500_HistoricalComponents_withChanges.csv` (first membership snapshot of each year) to reduce survivorship bias.
- Studies years **2012–2024** with `n_values = [100, 250]` against the SPY benchmark.
- Runs repeated model simulations (default: recall 0.2, precision 0.7, up to 1000 draws per year) to summarize the simulated portfolio's return distribution (mean, std, 5–95% band).
- Prints a per-year universe coverage / survivorship report (constituents with no price data, mid-year delistings kept at last price, mid-year data starts excluded), an N-level summary table, and recent yearly metrics to stdout. The yearly table includes `label_base_rate` — the label prevalence `T/(T+F)`, i.e. the precision a dart-throwing picker gets for free that year — and the summary reports `label_base_rate_mean/min/max` plus `custom_precision_edge_mean` (achieved precision minus base rate), so precision reads as skill over chance rather than an absolute number.
- Saves two plots: `topNAndCustom_vs_spy.png` (yearly returns with the simulated-model band) and `topNAndCustom_growth.png` (value of $100 reinvested annually).

To customize, pass CLI flags to the `study` subcommand (see `uv run python pickn.py study --help` for the full list):

```bash
uv run python pickn.py study --n-values 10 50 100 --year-start 2000 --year-end 2024 \
    --recall 0.3 --precision 0.8 --num-simulations 2000 --seed 42
```

Other useful flags: `--benchmark` (default SPY), `--label-threshold` (label stocks against a fixed absolute return instead of the year's benchmark), `--initial-investment` (growth-plot starting value), and `--no-plots`. For programmatic use — e.g. supplying `tickers_by_year` to override per-year membership, or a fixed `tickers` list for a static universe — call `run_top_n_study` directly.

## Sweeping recall/precision pairs

The `sweep` subcommand evaluates a grid of (recall, precision) pairs, reporting for each whether the 5th-percentile CAGR of the simulated portfolios meets or exceeds the benchmark CAGR. Results are printed and written to `Precision_Recall_Tradeoff.csv` (override with `--output`).

```bash
uv run python pickn.py sweep --recalls 0.1 0.2 0.3 --precisions 0.5 0.6 0.7 0.8 0.9
```

The grid defaults to recalls `0.1 0.2 0.3` × precisions `0.5 0.6 0.7 0.8`, and the shared study flags (`--year-start`/`--year-end`, `--benchmark`, `--seed`, `--label-threshold`, `--num-simulations`) apply here too — see `uv run python pickn.py sweep --help`.

Each output row also carries the label base rate (`base_rate_mean/min/max` — prevalence `T/(T+F)`, identical across grid cells since it depends only on the labeling threshold) and `precision_edge_mean` (achieved precision minus base rate): "precision *p* suffices" is only evidence of a strong model where that edge is large, and with a fixed absolute `--label-threshold` the base rate collapses in bear years, which the per-year column in `custom_stats` makes visible.

Prices and per-year universe returns are computed once and shared across the grid; only the classifier simulation is re-run per cell. The first (uncached) run is still slow because of the price download.

## Screening labeling criteria (`screen`)

Before asking what (recall, precision) a criterion needs, check whether the criterion can work at all: the `screen` subcommand computes, per year, the return of the **perfect-classifier portfolio** (precision = 1.0, recall = 1.0 — the equal-weighted mean of every stock meeting the criterion). If even that portfolio doesn't beat the benchmark, no precision level rescues the criterion.

```bash
uv run python pickn.py screen --criteria 0 0.1 bmk bmk+0.1
```

Criterion specs are absolute return thresholds (`0`, `0.1` for >10%) or benchmark-relative (`bmk`, `bmk+0.1`, `bmk-0.05`). The command prints a per-year table (criterion × {threshold, positives, base rate, positive-set mean return, benchmark return, excess}, also written to `Criterion_Feasibility.csv`, override with `--output`) and a per-criterion summary with `cagr_perfect` vs `cagr_benchmark` and a `passes_screen` verdict. Passing is a **necessary condition only**: a low-recall draw centers on the same mean but with real variance, and right-skewed positive returns put the typical draw below it — the risk analysis starts where the screen ends. This needs no selection simulation, so it's fast once prices are cached.

## Simulating recall/precision selections

`simulate_model.py` provides `simulate_selection` for exploring achievable recall/precision pairs on any labeled tabular data. A minimal example is included in the module's `__main__` block:

```bash
uv run python simulate_model.py
```

This prints the requested vs. achieved metrics, the sampled names, and the number of combinatorial ways a selection can satisfy the request under the rounding scheme documented in the module docstrings. Infeasible requests either raise (`strict=True`) or return a best-effort selection with an explanatory `note`.

### Labeling a year by SPY outperformance

Use `simulate_year_model_selection` in **`pickn.py`** to build a yearly dataset where each ticker is labeled `TRUE` when its calendar-year return beats SPY, then run the selection simulation:

```bash
uv run python -c "from pickn import simulate_year_model_selection; print(simulate_year_model_selection(2010, recall=.1, precision=.9, cache_path='adj_close_cache.csv'))"
```

This downloads adjusted-close prices for that year's constituents, computes calendar-year returns, labels tickers by benchmark outperformance, and returns a `SimulationResult`.

## Caching adjusted close data

`download_adj_close` accepts a `cache_path` (default `adj_close_cache.csv`). When the cache file exists and contains **all requested tickers**, data is loaded locally and sliced to the requested date range instead of re-downloading from yfinance. Note the check is ticker-based only — if the cached file covers a narrower date range than requested, you'll silently get the narrower range (delete the cache to force a fresh download). When any ticker is missing, the full download runs and overwrites the cache.

## Notes and caveats

- **Survivorship bias is the biggest caveat, and it is only partially fixable with yfinance.** The point-in-time membership file fixes *universe* selection, but Yahoo has no price history for most delisted names (a spot check of the 2012 constituents that have since left the index found ~80% with no 2012 data — including exactly the bankruptcies and blowups a study most needs). What the code does about it:
  - Tickers that stop trading **mid-year** are kept, with their return measured to the last available price (previously they became NaN and were silently dropped) — see `year_universe_returns`.
  - Tickers whose data only **begins** mid-year are excluded, since a constituent that "starts trading" mid-year usually means the symbol was later reused by a different company.
  - Tickers with **no data at all** cannot be recovered; `run_top_n_study` prints a per-year coverage report and returns it as `StudyResult.coverage` so the residual bias is at least measured, not hidden. Absolute return levels (especially top-N and early years) should be read as **upper bounds**. A survivorship-bias-free source (CRSP, Sharadar, Norgate, EODHD) is the only complete fix.
- "Top-N" portfolios are formed **ex post** (with hindsight) — they are an upper bound, not a strategy.
- The hypothetical classifier's labeling target is controlled by `label_threshold` (`None` = beat that year's benchmark; a fixed value like `0.1` = ">10% return").
- This is research code, not investment advice.

## Future work

See [`TODO.md`](TODO.md) for known issues and planned enhancements.
