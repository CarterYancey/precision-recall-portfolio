# pickN simulation — precision/recall portfolio study

## Overview

This project explores how portfolios composed of the top-performing S&P 500 stocks compare to an SPY benchmark across calendar years, and how good a stock-picking model would need to be (in precision/recall terms) to beat the market. It contains utilities for simulating model selections that target specific recall/precision trade-offs and for labeling tickers by yearly SPY outperformance.

**Core question:** if a hypothetical model picks stocks with recall *r* and precision *p* (where "positive" means "beat SPY that year"), what distribution of portfolio returns does it achieve — and what (*r*, *p*) is enough to consistently outperform?

## Repository layout

| Path | What it is |
|------|------------|
| `pickn.py` | Main study: fetches year-specific S&P 500 constituent prices from Sharadar, computes top-N/bottom-N/model-simulated yearly returns, and plots them against SPY. Also home of `simulate_year_model_selection` and `sweep_recall_precision_pairs`. |
| `simulate_model.py` | Standalone module: `simulate_selection` constructs a random selection hitting a requested (recall, precision) on labeled tabular data; `estimate_num_ways` counts the combinatorial ways such a selection can exist. No market-data dependencies. |
| `constituents.py` | Legacy utility (not imported by the other scripts): fetches point-in-time S&P 500 membership from historical Wikipedia revisions and caches it to `sp500_constituents_wikipedia_by_year.csv`. Run as a CLI: `uv run python constituents.py --year-start 2000 --year-end 2024`. |
| `data/SP500_history.csv` | **Large (~2.5 MB, ~600 rows).** Point-in-time S&P 500 membership, 1996-01-01 through 2026-07-11, in Sharadar's ticker namespace. One row per membership-*change* date; columns are `date` and `tickers` (the full ~500-ticker list as a python list literal). Used by `get_sp500_tickers_by_year` to reduce survivorship bias. Don't open it casually — see `CLAUDE.md`. |
| `data/cache/` | Generated at runtime (gitignored). Everything fetched from Sharadar, chiefly `adj_close_cache.csv` (cached adjusted closes). |
| `docs/sharadar_docs.md` | How the Sharadar/Nasdaq Data Link API is used: tables, filters, pagination, and the price-adjustment methodology. |
| `Precision_Recall_Tradeoff.csv` | Output of `sweep_recall_precision_pairs`: for each (recall, precision) pair, the 5th-percentile CAGR of the simulated portfolios vs. the benchmark CAGR (cap-weighted SPY and the equal-weighted universe mean). |
| `*.png` | Generated at runtime (gitignored). `topNAndCustom_vs_spy.png` (yearly returns) and `topNAndCustom_growth.png` (growth of $100) from `study`; `recall_precision_heatmap.png` and `recall_precision_heatmap_ew.png` (q05 CAGR excess vs the cap-weighted / equal-weight benchmark) from `sweep`. |
| `TODO.md` | Known issues and planned improvements. |
| `CLAUDE.md` | Notes for AI coding assistants working in this repo. |

## Setup

Requires Python >= 3.12.

1. Sync dependencies with [`uv`](https://docs.astral.sh/uv/) (creates/uses `.venv` automatically):
   ```bash
   uv sync
   ```
2. Either activate the environment (`source .venv/bin/activate`) or prefix commands with `uv run`, e.g. `uv run python pickn.py`.
3. Put your Nasdaq Data Link API key (with a Sharadar subscription covering SEP, and SFP for the SPY benchmark) in `~/.nasdaq/data_link_apikey` — the client picks it up automatically.

Without `uv`, a plain `pip install nasdaq-data-link pandas numpy matplotlib lxml html5lib requests` into any 3.12+ environment also works.

## Running the top-N study

```bash
uv run python pickn.py            # same as `uv run python pickn.py study`
```

The first run downloads price data from Sharadar for every historical constituent (thousands of tickers) — expect it to take a while and produce a large `data/cache/adj_close_cache.csv`. Subsequent runs load from the cache.

With default options, the script:

- Uses year-specific S&P 500 membership from `data/SP500_history.csv` (the snapshot in force on each year's January 1) to reduce survivorship bias.
- Studies years **2012–2024** with `n_values = [100, 250]` against the SPY benchmark.
- Runs repeated model simulations (default: recall 0.2, precision 0.7, up to 1000 draws per year) to summarize the simulated portfolio's return distribution (mean, std, 5–95% band).
- Prints a per-year universe coverage / survivorship report (constituents with no price data, mid-year delistings kept at last price, mid-year data starts excluded), an N-level summary table, and recent yearly metrics to stdout. The yearly table includes `label_base_rate` — the label prevalence `T/(T+F)`, i.e. the precision a dart-throwing picker gets for free that year — and the summary reports `label_base_rate_mean/min/max` plus `custom_precision_edge_mean` (achieved precision minus base rate), so precision reads as skill over chance rather than an absolute number.
- Reports performance against **two benchmarks**: the cap-weighted `--benchmark` ticker (SPY) and the **equal-weighted universe mean** (per-year mean return of all constituents with data). The simulated portfolios are equal-weighted, so excess over SPY mixes selection skill with the equal-weight/size effect; excess over the equal-weight benchmark isolates skill. The yearly table carries `ew_benchmark_return` and per-N `top{n}_excess_ew`, the summary `avg_ew_benchmark_return`/`avg_excess_ew`/`cagr_ew_benchmark`, and both plots draw the equal-weight benchmark as a dashed line. (Labeling with the default `label_threshold` still targets the cap-weighted benchmark's return.)
- Saves two plots: `topNAndCustom_vs_spy.png` (yearly returns with the simulated-model band) and `topNAndCustom_growth.png` (value of $100 reinvested annually).

To customize, pass CLI flags to the `study` subcommand (see `uv run python pickn.py study --help` for the full list):

```bash
uv run python pickn.py study --n-values 10 50 100 --year-start 2000 --year-end 2024 \
    --recall 0.3 --precision 0.8 --num-simulations 2000 --seed 42
```

Other useful flags: `--benchmark` (default SPY), `--label-threshold` (label stocks against a fixed absolute per-annum return instead of the benchmark's), `--exclude-top` (see below), `--horizon`/`--overlapping` (multi-year holding windows — see the dedicated section below), `--initial-investment` (growth-plot starting value), and `--no-plots`.

**"Miss the super-performers" mode (`--exclude-top`).** Positive-class returns are heavily right-skewed: a few super-performers carry a large share of the positive-set mean, and a realistic model should not be credited with finding them. `--exclude-top` bars the best positives *by return* from the simulated model's picks — a value >= 1 is a count (top-K positives), a value in (0,1) a fraction of that year's positives (`0.1` = top decile). Excluded stocks still count toward recall's denominator (they become forced false negatives), so the TP quota is filled from ordinary criterion-meeting stocks only. The per-year excluded count appears as `excluded_top` in the yearly table (`excluded_top_mean` in the summary). The same run also simulates the **uniform-draw baseline** (no exclusion, same seed — so the two distributions differ only by the exclusion): the study prints a per-year comparison table (excluded vs uniform mean/q05 and the `mean_gap`), the summary gains `cagr_custom_{mean,q05,q95}_uniform`, `avg_custom_mean_return_uniform`, and `avg_custom_mean_gap`, both plots draw the uniform mean as a dotted line, and `StudyResult.custom_stats_uniform` carries the full per-year baseline. The gap measures how much of the apparent edge depends on catching outliers nobody should count on catching — no second run needed. For programmatic use — e.g. supplying `tickers_by_year` to override per-year membership, or a fixed `tickers` list for a static universe — call `run_top_n_study` directly.

## Sweeping recall/precision pairs

The `sweep` subcommand evaluates a grid of (recall, precision) pairs, reporting for each whether the 5th-percentile CAGR of the simulated portfolios meets or exceeds the benchmark CAGR — against both the cap-weighted benchmark (`cagr_benchmark`, `custom_q05_meets_benchmark`) and the equal-weighted universe mean (`cagr_ew_benchmark`, `custom_q05_meets_ew_benchmark`). Results are printed and written to `Precision_Recall_Tradeoff.csv` (override with `--output`), and plotted as two heatmaps: `recall_precision_heatmap.png` (excess vs the cap-weighted benchmark) and `recall_precision_heatmap_ew.png` (excess vs the equal-weight universe); skip with `--no-plots`.

```bash
uv run python pickn.py sweep --recalls 0.1 0.2 0.3 --precisions 0.5 0.6 0.7 0.8 0.9
```

The grid defaults to recalls `0.1 0.2 0.3` × precisions `0.5 0.6 0.7 0.8`, and the shared study flags (`--year-start`/`--year-end`, `--benchmark`, `--horizon`/`--overlapping`, `--seed`, `--label-threshold`, `--exclude-top`, `--num-simulations`) apply here too — see `uv run python pickn.py sweep --help`. `--exclude-top` applies the "miss the super-performers" mode (see the study section) to every grid cell, with the average per-year excluded count reported as `excluded_top_mean`; each cell also runs the uniform-draw baseline with the same seed, adding `cagr_custom_q05_uniform`, `custom_q05_gap` (uniform minus excluded), and `custom_q05_meets_benchmark_uniform`/`custom_q05_meets_ew_benchmark_uniform` — so one sweep shows how much of the feasible region depends on catching outliers.

Each output row also carries the label base rate (`base_rate_mean/min/max` — prevalence `T/(T+F)`, identical across grid cells since it depends only on the labeling threshold) and `precision_edge_mean` (achieved precision minus base rate): "precision *p* suffices" is only evidence of a strong model where that edge is large, and with a fixed absolute `--label-threshold` the base rate collapses in bear years, which the per-year column in `custom_stats` makes visible.

Prices and per-year universe returns are computed once and shared across the grid; only the classifier simulation is re-run per cell. The first (uncached) run is still slow because of the price download.

## Screening labeling criteria (`screen`)

Before asking what (recall, precision) a criterion needs, check whether the criterion can work at all: the `screen` subcommand computes, per year, the return of the **perfect-classifier portfolio** (precision = 1.0, recall = 1.0 — the equal-weighted mean of every stock meeting the criterion). If even that portfolio doesn't beat the benchmark, no precision level rescues the criterion.

```bash
uv run python pickn.py screen --criteria 0 0.1 bmk bmk+0.1
```

Criterion specs are absolute return thresholds (`0`, `0.1` for >10%) or benchmark-relative (`bmk`, `bmk+0.1`, `bmk-0.05`); `bmk` always refers to the cap-weighted benchmark. The command prints a per-year table (criterion × {threshold, positives, base rate, positive-set mean return, both benchmark returns, excess vs each}, also written to `Criterion_Feasibility.csv`, override with `--output`) and a per-criterion summary with `cagr_perfect` vs `cagr_benchmark` and `cagr_ew_benchmark`, plus paired verdicts: `passes_screen` (beats cap-weighted SPY) and `passes_screen_ew` (beats the equal-weighted universe mean — the perfect portfolio is itself equal-weighted, so only this one isolates selection value from the weighting scheme). Passing is a **necessary condition only**: a low-recall draw centers on the same mean but with real variance, and right-skewed positive returns put the typical draw below it — the risk analysis starts where the screen ends. This needs no selection simulation, so it's fast once prices are cached.

## Multi-year holding horizons (`--horizon`, `--overlapping`)

All three subcommands accept `--horizon H` to study criteria like ">0% annual returns over the next 3 years" instead of single calendar years. Each cohort is formed at the start of a **formation year** — membership is the S&P 500 snapshot in force at formation, exactly as before — and held **buy-and-hold** through year formation+H−1: returns run from the formation year's first trading day to the window's last, and stocks that delist inside the window are kept at their last available price (the same rule that covers mid-year delistings at horizon 1; they show up under `partial_year` in the coverage report). All per-"year" tables become per-window tables indexed by formation year.

```bash
uv run python pickn.py screen --horizon 3 --criteria 0 0.1 bmk       # ">X% annual over 3 years"
uv run python pickn.py sweep  --horizon 3 --overlapping --recalls 0.2 --precisions 0.6 0.8
```

Two window schemes are available:

- **Non-overlapping (default):** formation years step by H (e.g. 2012, 2015, 2018, … for H=3), so chaining the window returns is a realizable "rebalance every H years" path and the reported CAGRs are path CAGRs.
- **`--overlapping`:** a cohort forms every year (rolling windows). More sample windows, but consecutive windows share H−1 calendar years, so the cohorts are autocorrelated: CAGRs are then geometric means of annualized per-cohort returns — a summary statistic, not a realizable single path — and the study skips the growth-of-$100 plot for that reason.

Thresholds stay **per-annum** everywhere and are compounded internally: `--label-threshold 0.1 --horizon 3` labels stocks returning more than 1.1³−1 ≈ 33.1% over the window, and screen criteria work the same way (`bmk±offset` adds the offset to the benchmark's *annualized* window return before compounding back). All CAGR columns are annualized by the horizon, so results at different horizons are directly comparable. `--horizon 1` (the default) reproduces the original calendar-year study exactly.

## Simulating recall/precision selections

`simulate_model.py` provides `simulate_selection` for exploring achievable recall/precision pairs on any labeled tabular data. A minimal example is included in the module's `__main__` block:

```bash
uv run python simulate_model.py
```

This prints the requested vs. achieved metrics, the sampled names, and the number of combinatorial ways a selection can satisfy the request under the rounding scheme documented in the module docstrings. Infeasible requests either raise (`strict=True`) or return a best-effort selection with an explanatory `note`.

### Labeling a year by SPY outperformance

Use `simulate_year_model_selection` in **`pickn.py`** to build a yearly dataset where each ticker is labeled `TRUE` when its calendar-year return beats SPY, then run the selection simulation:

```bash
uv run python -c "from pickn import simulate_year_model_selection; print(simulate_year_model_selection(2010, recall=.1, precision=.9))"
```

This downloads adjusted-close prices for that year's constituents, computes calendar-year returns, labels tickers by benchmark outperformance, and returns a `SimulationResult`.

## Caching adjusted close data

Everything fetched from Sharadar is stored under `data/cache/` and served from there whenever possible. `download_adj_close` accepts a `cache_path` (default `data/cache/adj_close_cache.csv`) and checks both **date coverage** (`cache_covers_dates`) and **ticker coverage**:

- Dates and tickers covered → served entirely from the cache, no network.
- Dates covered, some tickers missing → only the missing tickers are fetched (over the cache's full date span) and merged in. Tickers known to have no data are cached as empty columns and never re-requested.
- Dates not covered → full re-download of the requested tickers, overwriting the cache. There is deliberately no incremental date extension: Sharadar's `closeadj` is backward-adjusted, so price rows fetched at different times sit on different adjustment bases and must not be spliced together per ticker.

## Notes and caveats

- **Survivorship bias is much reduced with Sharadar, but not zero.** The point-in-time membership file fixes *universe* selection, and Sharadar SEP includes delisted names (unlike Yahoo Finance, which this project previously used and which was missing ~80% of since-departed constituents in a 2012 spot check). What the code still does about the residue:
  - Tickers that stop trading **mid-year** are kept, with their return measured to the last available price — see `year_universe_returns`.
  - Tickers whose data only **begins** mid-year are excluded, since a constituent that "starts trading" mid-year usually means the symbol was later reused by a different company.
  - Tickers with **no data at all** (mostly years before SEP's history starts around 1998, or symbols absent from SEP) are dropped; `run_top_n_study` prints a per-year coverage report and returns it as `StudyResult.coverage` so the residual bias is measured, not hidden. Early-year absolute return levels should still be read with the coverage report in hand.
- "Top-N" portfolios are formed **ex post** (with hindsight) — they are an upper bound, not a strategy.
- The hypothetical classifier's labeling target is controlled by `label_threshold` (`None` = beat that year's benchmark; a fixed value like `0.1` = ">10% return").
- This is research code, not investment advice.

## Future work

See [`TODO.md`](TODO.md) for known issues and planned enhancements.
