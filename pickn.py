"""
Top-N winners vs S&P 500 (1-year horizons, by calendar year)

Notes / assumptions:
- "Top-performing stocks for that year" is determined ex post (with hindsight).
- Uses point-in-time historical S&P 500 membership by default
  (SP500_HistoricalComponents_withChanges.csv, first snapshot of each year).
- Survivorship bias handling (see year_universe_returns):
  * Tickers that stop trading mid-year (delistings/bankruptcies) are kept,
    with the return measured to their last available price, instead of being
    silently dropped for lacking a year-end price.
  * Tickers whose data only begins mid-year are excluded (guards against
    reused ticker symbols mapping to a different, later company).
  * Tickers with no price data at all cannot be recovered from yfinance and
    are dropped; per-year counts are reported in StudyResult.coverage so the
    residual bias is visible. Because Yahoo lacks history for most delisted
    names (e.g. Enron, Bear Stearns), dropped names skew toward the worst
    outcomes and results remain upward-biased. A survivorship-bias-free
    price source (CRSP, Sharadar, Norgate, ...) is the only full fix.
- Uses adjusted close (includes splits + dividends where provider supports it).
- S&P 500 benchmark is proxied by SPY total return via adjusted close.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# pip install yfinance pandas numpy lxml
import yfinance as yf

from simulate_model import SimulationResult, simulate_selection

def get_sp500_tickers_from_csv(path="SP500Current.csv"):
    df = pd.read_csv(path)
    tickers = (
        df["Symbol"]
        .astype(str)
        .str.strip()
        .str.replace(".", "-", regex=False)
        .tolist()
    )
    return sorted(set(tickers))


def get_sp500_tickers_by_year(path="SP500_HistoricalComponents_withChanges.csv") -> Dict[int, List[str]]:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    first_by_year = df.groupby(df["date"].dt.year, as_index=False).first()
    tickers_by_year = {}
    for _, row in first_by_year.iterrows():
        tickers = [
            t.strip().replace(".", "-")
            for t in str(row["tickers"]).split(",")
            if t.strip()
        ]
        tickers_by_year[int(row["date"].year)] = tickers
    return tickers_by_year

def cache_covers_dates(
    cached_index: pd.DatetimeIndex,
    start: str,
    end: str,
    *,
    tolerance: pd.Timedelta = pd.Timedelta(days=7),
    today: Optional[pd.Timestamp] = None,
) -> bool:
    """
    True if a cache whose (trading-day) index spans [min, max] covers the
    requested [start, end) window.

    The cache holds trading days only, so exact endpoint matches are not
    possible: the first trading day on/after `start` can fall a few days
    later (weekends, New Year holidays), and the last trading day is
    strictly before the exclusive `end`. `tolerance` absorbs that calendar
    slack; a gap beyond it means the cache was built for a narrower period.
    A requested `end` in the future is clamped to `today`, since no cache
    can contain prices that don't exist yet.
    """
    if len(cached_index) == 0:
        return False
    today = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today)
    end_ts = min(pd.Timestamp(end), today)
    return (
        cached_index.min() <= pd.Timestamp(start) + tolerance
        and cached_index.max() >= end_ts - tolerance
    )


def download_adj_close(
    tickers: List[str],
    start: str,
    end: str,
    auto_adjust: bool = False,
    cache_path: str = "adj_close_cache.csv",
) -> pd.DataFrame:
    """
    Download daily prices and return Adjusted Close as wide dataframe.
    yfinance returns 'Adj Close' if auto_adjust=False.
    If auto_adjust=True, 'Close' is already adjusted; here we keep auto_adjust=False and use 'Adj Close'.
    """
    cache_file = Path(cache_path)
    if cache_file.exists():
        cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        cached.columns = [str(c) for c in cached.columns]
        cached = cached.sort_index()
        requested = {str(t) for t in tickers}
        has_all_tickers = requested.issubset(set(cached.columns))
        covers_dates = cache_covers_dates(cached.index, start, end)
        if has_all_tickers and covers_dates:
            print("Loading from cache file...")
            # Match yfinance's convention: start inclusive, end exclusive.
            in_range = (cached.index >= pd.Timestamp(start)) & (cached.index < pd.Timestamp(end))
            return cached.loc[in_range, sorted(requested)]
        if has_all_tickers and not covers_dates:
            print(
                f"Cache date range {cached.index.min().date()}..{cached.index.max().date()} "
                f"does not cover requested {start}..{end}; re-downloading."
            )

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="column",
        threads=True,
    )

    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            adj = data["Adj Close"].copy()
        elif "Close" in data.columns.get_level_values(0):
            # fallback if provider didn't deliver adj close
            adj = data["Close"].copy()
        else:
            raise ValueError("Could not find Adj Close/Close in downloaded data.")
    else:
        # single ticker
        adj = data.rename(columns={"Adj Close": tickers[0], "Close": tickers[0]})

    # Ensure column names are tickers
    adj.columns = [str(c) for c in adj.columns]
    adj = adj.sort_index()
    adj.to_csv(cache_file, index_label="Date")
    return adj


def year_endpoints(trading_index: pd.DatetimeIndex, year: int) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Get first and last trading day for a given calendar year from available data.
    Returns None if year not covered.
    """
    mask = trading_index.year == year
    if not mask.any():
        return None
    idx = trading_index[mask]
    return idx[0], idx[-1]


def compute_calendar_year_returns(adj: pd.DataFrame, years: Iterable[int]) -> pd.DataFrame:
    """
    For each year, compute total return from first trading day to last trading day.
    Returns a DataFrame indexed by year with columns = tickers.
    """
    rets = {}
    for y in years:
        endpoints = year_endpoints(adj.index, y)
        if endpoints is None:
            continue
        d0, d1 = endpoints
        p0 = adj.loc[d0]
        p1 = adj.loc[d1]
        r = (p1 / p0) - 1.0
        rets[y] = r
    out = pd.DataFrame(rets).T  # years x tickers
    out.index.name = "year"
    return out


def year_universe_returns(
    adj: pd.DataFrame,
    year: int,
    tickers: List[str],
    *,
    max_start_lag: int = 10,
    max_end_lead: int = 10,
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Calendar-year returns for a point-in-time universe, without silently
    dropping mid-year delistings.

    For each ticker the return uses the first and last *available* prices
    within the year, so a stock that stops trading in June (bankruptcy,
    acquisition) stays in the universe with its return measured to its final
    print instead of becoming NaN. Rules:

    - No price data in the year at all -> dropped, counted as ``no_data``.
      (This is the residual survivorship bias yfinance forces on us.)
    - First price more than `max_start_lag` trading days after the year's
      first trading day -> excluded, counted as ``late_start``. The ticker
      was a constituent at the year start, so data appearing only mid-year
      usually means the symbol was reused by a different company later.
    - Last price more than `max_end_lead` trading days before the year's
      last trading day -> kept, counted as ``partial_year`` (delisted
      mid-year, return measured to the last available price).
    - Otherwise counted as ``full_year``.

    Returns (returns indexed by ticker, coverage stats dict).
    """
    mask = adj.index.year == year
    stats = {
        "members": len(tickers),
        "no_data": 0,
        "late_start": 0,
        "partial_year": 0,
        "full_year": 0,
    }
    if not mask.any():
        stats["used"] = 0
        stats["coverage"] = np.nan
        return pd.Series(dtype=float), stats

    window = adj.loc[mask]
    days = window.index
    start_cutoff = days[min(max_start_lag, len(days) - 1)]
    end_cutoff = days[max(len(days) - 1 - max_end_lead, 0)]

    returns = {}
    for t in tickers:
        col = window[t].dropna() if t in window.columns else pd.Series(dtype=float)
        if len(col) < 2:
            stats["no_data"] += 1
            continue
        if col.index[0] > start_cutoff:
            stats["late_start"] += 1
            continue
        if col.index[-1] < end_cutoff:
            stats["partial_year"] += 1
        else:
            stats["full_year"] += 1
        returns[t] = float(col.iloc[-1] / col.iloc[0] - 1.0)

    stats["used"] = stats["partial_year"] + stats["full_year"]
    stats["coverage"] = stats["used"] / stats["members"] if stats["members"] else np.nan
    return pd.Series(returns, dtype=float), stats


def simulate_year_model_selection(
    year: int,
    recall: float,
    precision: float,
    *,
    benchmark: str = "SPY",
    tickers: Optional[List[str]] = None,
    tickers_by_year: Optional[Dict[int, List[str]]] = None,
    positive_label: str = "TRUE",
    random_state: Optional[int] = None,
    strict: bool = False,
    cache_path: str = "adj_close_cache.csv",
) -> SimulationResult:
    """
    Build year-specific labels based on S&P 500 outperformance and run simulate_selection.

    A stock is labeled TRUE if its calendar-year return exceeds the benchmark's return.
    """
    if tickers_by_year is None:
        if tickers is None:
            tickers_by_year = get_sp500_tickers_by_year()
        else:
            tickers_by_year = {year: tickers}

    tickers_for_year = tickers_by_year.get(year)
    if not tickers_for_year:
        raise ValueError(f"No tickers available for year {year}.")

    tickers_all = sorted(set(tickers_for_year + [benchmark]))
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"
    adj = download_adj_close(tickers_all, start, end, cache_path=cache_path)
    bmk_returns = compute_calendar_year_returns(adj[[benchmark]], [year])
    if bmk_returns.empty or year not in bmk_returns.index:
        raise ValueError(f"Could not compute returns for year {year}.")
    spy_return = bmk_returns.loc[year, benchmark]
    if pd.isna(spy_return):
        raise ValueError(f"Missing benchmark return for {benchmark} in year {year}.")

    stock_universe = [t for t in tickers_for_year if t != benchmark]
    stock_returns, cov = year_universe_returns(adj, year, stock_universe)
    if stock_returns.empty:
        raise ValueError(f"No stock returns available for year {year}.")
    if cov["no_data"] or cov["late_start"]:
        print(
            f"[{year}] universe coverage {cov['coverage']:.1%}: "
            f"{cov['no_data']} constituents had no price data (survivorship bias), "
            f"{cov['late_start']} excluded for data starting mid-year, "
            f"{cov['partial_year']} delisted mid-year kept at last price."
        )

    labels = np.where(stock_returns > spy_return, positive_label, "FALSE")
    df = pd.DataFrame(
        {
            "ticker": stock_returns.index.astype(str),
            "label": labels,
        }
    )

    return simulate_selection(
        df,
        "ticker",
        "label",
        recall,
        precision,
        positive_label=positive_label,
        random_state=random_state,
        strict=strict,
    )


def top_n_portfolio_return(year_returns: pd.Series, n: int) -> float:
    """
    Equal-weight return of top-n stocks for that year (ignores missing).
    """
    yr = year_returns.dropna()
    if yr.empty:
        return np.nan
    n_eff = min(n, len(yr))
    top = yr.nlargest(n_eff)
    return float(top.mean())

def bottom_n_portfolio_return(year_returns: pd.Series, n: int) -> float:
    """
    Equal-weight return of bottom-n stocks for that year (ignores missing).
    """
    yr = year_returns.dropna()
    if yr.empty:
        return np.nan
    n_eff = min(n, len(yr))
    bottom = yr.nsmallest(n_eff)
    return float(bottom.mean())

def model_selected_portfolio_return(
    year_returns: pd.Series,
    selected_names: List[str],
) -> float:
    """
    Equal-weight return of model-selected stocks for that year (ignores missing).
    """
    if not selected_names:
        return np.nan
    selected = year_returns.reindex(selected_names).dropna()
    if selected.empty:
        return np.nan
    return float(selected.mean())


def simulate_custom_portfolio_distribution(
    year_returns: pd.Series,
    benchmark_return: float,
    recall: float,
    precision: float,
    num_simulations: int,
    *,
    label_threshold: Optional[float] = None,
    positive_label: str = "TRUE",
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """
    Run repeated simulate_selection calls for a given year and return distribution stats
    for the model-selected portfolio returns.

    The hypothetical classifier labels a stock "positive" when its return exceeds
    `label_threshold` if given (a fixed absolute target, e.g. 0.1 for "picks stocks
    that return >10%"), otherwise that year's `benchmark_return`. Performance of the
    resulting portfolios is always measured against the actual benchmark downstream;
    only the classification target changes.

    The returned stats include ``base_rate``, the label prevalence T/(T+F) —
    the precision a random picker gets for free that year, so achieved
    precision is only evidence of skill to the extent it exceeds it.
    """
    if rng is None:
        rng = np.random.default_rng()
    threshold = benchmark_return if label_threshold is None else label_threshold
    labels = np.where(year_returns > threshold, positive_label, "FALSE")
    base_rate = float((labels == positive_label).mean()) if labels.size else np.nan
    df = pd.DataFrame({"ticker": year_returns.index.astype(str), "label": labels})

    def draw_selection():
        random_state = int(rng.integers(0, 2**31 - 1))
        return simulate_selection(
            df,
            "ticker",
            "label",
            recall,
            precision,
            positive_label=positive_label,
            random_state=random_state,
            strict=False,
        )

    # Achieved metrics depend only on the label counts (draws only vary which
    # names are sampled), so a single probe tells us whether the requested
    # (recall, precision) is reachable within tolerance for this year.
    probe = draw_selection()
    achieved_recall = probe.achieved_recall
    achieved_precision = probe.achieved_precision
    feasible = (
        recall - 0.1 <= achieved_recall <= recall + 0.1
        and precision - 0.1 <= achieved_precision <= precision + 0.1
    )

    returns = []
    if feasible:
        returns.append(model_selected_portfolio_return(year_returns, probe.selected_names))
        # No point drawing more often than there are distinct selections.
        max_draws = min(num_simulations, probe.num_ways)
        for _ in range(1, max_draws):
            model_selection = draw_selection()
            returns.append(model_selected_portfolio_return(year_returns, model_selection.selected_names))

    returns_arr = np.array(returns, dtype=float)
    returns_arr = returns_arr[~np.isnan(returns_arr)]
    if returns_arr.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "q05": np.nan,
            "q95": np.nan,
            "count": 0,
            "achieved_recall": achieved_recall,
            "achieved_precision": achieved_precision,
            "base_rate": base_rate,
        }

    return {
        "mean": float(np.mean(returns_arr)),
        "std": float(np.std(returns_arr, ddof=1)) if returns_arr.size > 1 else 0.0,
        "q05": float(np.quantile(returns_arr, 0.05)),
        "q95": float(np.quantile(returns_arr, 0.95)),
        "count": int(returns_arr.size),
        "achieved_recall": achieved_recall,
        "achieved_precision": achieved_precision,
        "base_rate": base_rate,
    }


@dataclass
class StudyResult:
    yearly: pd.DataFrame          # year x metrics
    by_n: pd.DataFrame            # year x N (portfolio returns)
    by_n_bottom: pd.DataFrame            # year x N (portfolio returns)
    custom_stats: pd.DataFrame    # year x custom distribution stats
    summary: pd.DataFrame         # N-level summary stats
    coverage: pd.DataFrame        # year x universe coverage / survivorship stats


def prepare_universe_returns(
    year_start: int,
    year_end: int,
    benchmark: str = "SPY",
    tickers: Optional[List[str]] = None,
    tickers_by_year: Optional[Dict[int, List[str]]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Download prices once and compute the inputs every study/sweep run shares:
    per-year universe returns (year x ticker), benchmark calendar-year returns,
    and the coverage/survivorship report (also printed).

    None of this depends on the simulated classifier's (recall, precision), so
    a grid sweep calls it once and reuses the result for every cell.
    """
    if tickers_by_year is None:
        if tickers is None:
            tickers_by_year = get_sp500_tickers_by_year()
        else:
            tickers_by_year = {
                y: tickers for y in range(year_start, year_end + 1)
            }

    tickers_union = {
        ticker for tickers_for_year in tickers_by_year.values()
        for ticker in tickers_for_year
    }
    all_tickers = sorted(tickers_union | {benchmark})

    start = f"{year_start}-01-01"
    end = f"{year_end}-12-31"
    adj = download_adj_close(all_tickers, start=start, end=end, auto_adjust=False)

    if benchmark not in adj.columns:
        raise ValueError(f"Benchmark {benchmark} not in downloaded columns.")
    adj_bmk = adj[[benchmark]].copy()
    adj_stk = adj.drop(columns=[benchmark], errors="ignore")

    years = range(year_start, year_end + 1)

    stock_yearly_rows = {}
    coverage_rows = {}
    for y in years:
        tickers_for_year = tickers_by_year.get(y)
        if not tickers_for_year:
            continue
        year_rets, cov = year_universe_returns(adj_stk, y, tickers_for_year)
        if year_rets.empty:
            continue
        stock_yearly_rows[y] = year_rets
        coverage_rows[y] = cov
    stock_yearly = pd.DataFrame(stock_yearly_rows).T
    stock_yearly.index.name = "year"
    coverage = pd.DataFrame(coverage_rows).T
    coverage.index.name = "year"
    print("\n=== Universe coverage / survivorship report ===")
    print(coverage.to_string(float_format=lambda v: f"{v:.3f}"))
    print(
        "no_data tickers were index members but have no yfinance history "
        "(mostly delistings) and are excluded — results are upward-biased "
        "in proportion to these counts. partial_year tickers stopped trading "
        "mid-year and are included at their last available price."
    )
    bmk_yearly = compute_calendar_year_returns(adj_bmk, years)[benchmark]
    return stock_yearly, bmk_yearly, coverage


def compute_custom_stats(
    stock_yearly: pd.DataFrame,
    bmk_yearly: pd.Series,
    recall: float,
    precision: float,
    num_simulations: int,
    *,
    label_threshold: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Per-year distribution stats for the simulated classifier's portfolios
    (year x mean/std/q05/q95/count/achieved metrics/base_rate). This is the
    only part of the pipeline that depends on (recall, precision) — except
    ``base_rate`` (label prevalence T/(T+F)), which depends only on the
    labeling threshold and is reported so precision can be read as skill
    over the dart-throwing baseline.
    """
    if rng is None:
        rng = np.random.default_rng()
    custom_stats = pd.DataFrame(
        index=stock_yearly.index,
        columns=[
            "mean", "std", "q05", "q95", "count",
            "achieved_recall", "achieved_precision", "base_rate",
        ],
        dtype=float,
    )
    for y in stock_yearly.index:
        benchmark_return = bmk_yearly.get(y)
        if pd.isna(benchmark_return):
            continue
        stats = simulate_custom_portfolio_distribution(
            stock_yearly.loc[y].dropna(),
            benchmark_return,
            recall,
            precision,
            num_simulations,
            label_threshold=label_threshold,
            rng=rng,
        )
        for key, value in stats.items():
            custom_stats.loc[y, key] = value
    return custom_stats


def run_top_n_study(
    n_values: Iterable[int] = (1, 5, 10, 20, 50, 100),
    year_start: int = 2000,
    year_end: int = 2024,
    benchmark: str = "SPY",
    tickers: Optional[List[str]] = None,
    tickers_by_year: Optional[Dict[int, List[str]]] = None,
    model_recall: float = .2,
    model_precision: float = 0.7,
    num_simulations: int = 1000,
    model_random_seed: Optional[int] = None,
    label_threshold: Optional[float] = None,
) -> StudyResult:
    """
    Main pipeline.

    `label_threshold` sets the return the hypothetical classifier tries to identify
    stocks as exceeding: a fixed absolute value (e.g. 0.1 for ">10% per year"), or
    None to target beating that year's benchmark return. Portfolio performance is
    always compared against the actual benchmark either way.
    """
    n_values = sorted(set(int(n) for n in n_values))
    print("model_recall: ", model_recall)
    print("model_precision: ", model_precision)
    stock_yearly, bmk_yearly, coverage = prepare_universe_returns(
        year_start,
        year_end,
        benchmark=benchmark,
        tickers=tickers,
        tickers_by_year=tickers_by_year,
    )

    # Compute top-N returns per year
    by_n = pd.DataFrame(index=stock_yearly.index, columns=n_values, dtype=float)
    for y in stock_yearly.index:
        row = stock_yearly.loc[y]
        for n in n_values:
            by_n.loc[y, n] = top_n_portfolio_return(row, n)
    # Compute bottom-N returns per year
    by_n_bottom = pd.DataFrame(index=stock_yearly.index, columns=n_values, dtype=float)
    for y in stock_yearly.index:
        row = stock_yearly.loc[y]
        for n in n_values:
            by_n_bottom.loc[y, n] = bottom_n_portfolio_return(row, n)
    # Compute custom-N returns per year
    custom_stats = compute_custom_stats(
        stock_yearly,
        bmk_yearly,
        model_recall,
        model_precision,
        num_simulations,
        label_threshold=label_threshold,
        rng=np.random.default_rng(model_random_seed),
    )

    # Assemble metrics table
    yearly = pd.DataFrame(index=stock_yearly.index)
    yearly["benchmark_return"] = bmk_yearly.reindex(yearly.index)
    # Label prevalence T/(T+F): the precision a dart-throwing picker gets for
    # free that year, next to the results so precision reads as skill-over-chance.
    yearly["label_base_rate"] = custom_stats["base_rate"]

    # Add excess returns and "top-N share of gains" diagnostics
    # Share-of-gains: sum of top-N stock returns / sum of all positive stock returns (simple proxy)
    # This is not a cap-weighted market attribution; it's a concentration indicator.
    pos_sum = stock_yearly.clip(lower=0).sum(axis=1)
    for n in n_values:
        yearly[f"top{n}_return"] = by_n[n]
        yearly[f"top{n}_excess"] = by_n[n] - yearly["benchmark_return"]

        # concentration proxy
        topn_sum = stock_yearly.apply(lambda r: r.dropna().nlargest(min(n, r.dropna().shape[0])).clip(lower=0).sum(), axis=1)
        yearly[f"top{n}_share_of_positive_gains"] = np.where(pos_sum > 0, topn_sum / pos_sum, np.nan)

    # Summaries across years
    summary_rows = []
    benchmark_cagr = compute_cagr(yearly["benchmark_return"])
    custom_mean_cagr = compute_cagr(custom_stats["mean"]) if "mean" in custom_stats.columns else np.nan
    custom_q05_cagr = compute_cagr(custom_stats["q05"]) if "mean" in custom_stats.columns else np.nan
    custom_q95_cagr = compute_cagr(custom_stats["q95"]) if "mean" in custom_stats.columns else np.nan
    achieved_recall = custom_stats["achieved_recall"]
    achieved_precision = custom_stats["achieved_precision"]
    base_rate = custom_stats["base_rate"]
    for n in n_values:
        r = by_n[n]
        r2 = by_n_bottom[n]
        excess = r - yearly["benchmark_return"]
        summary_rows.append({
            "N": n,
            "years": int(r.notna().sum()),
            "avg_topN_return": float(r.mean()),
            "avg_benchmark_return": float(yearly["benchmark_return"].mean()),
            "avg_excess": float(excess.mean()),
            "avg_bottomN_return": float(r2.mean()),
            "cagr_topN": compute_cagr(r),
            "cagr_bottomN": compute_cagr(r2),
            "cagr_benchmark": benchmark_cagr,
            "cagr_custom_mean": custom_mean_cagr,
            "cagr_custom_q05": custom_q05_cagr,
            "cagr_custom_q95": custom_q95_cagr,
            "custom_recall_mean": float(achieved_recall.mean()),
            "custom_recall_min": float(achieved_recall.min()),
            "custom_recall_max": float(achieved_recall.max()),
            "custom_precision_mean": float(achieved_precision.mean()),
            "custom_precision_min": float(achieved_precision.min()),
            "custom_precision_max": float(achieved_precision.max()),
            "label_base_rate_mean": float(base_rate.mean()),
            "label_base_rate_min": float(base_rate.min()),
            "label_base_rate_max": float(base_rate.max()),
            # Skill over chance: achieved precision minus the free baseline.
            "custom_precision_edge_mean": float((achieved_precision - base_rate).mean()),
        })
    summary = pd.DataFrame(summary_rows).set_index("N")

    custom_summary = {
        "avg_custom_mean_return": float(custom_stats["mean"].mean()),
        "avg_custom_std_return": float(custom_stats["std"].mean()),
        "avg_custom_q05_return": float(custom_stats["q05"].mean()),
        "avg_custom_q95_return": float(custom_stats["q95"].mean()),
    }
    summary = summary.assign(**custom_summary)

    return StudyResult(
        yearly=yearly,
        by_n=by_n,
        by_n_bottom=by_n_bottom,
        custom_stats=custom_stats,
        summary=summary,
        coverage=coverage,
    )


def sweep_from_returns(
    stock_yearly: pd.DataFrame,
    bmk_yearly: pd.Series,
    recall_values: Iterable[float],
    precision_values: Iterable[float],
    *,
    num_simulations: int = 5000,
    model_random_seed: Optional[int] = None,
    label_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Grid-evaluate (recall, precision) pairs on precomputed per-year returns.
    Only the classifier simulation is repeated per cell; the price data,
    universe returns, and benchmark CAGR are shared across the whole grid.

    Each cell reseeds its own RNG from `model_random_seed`, so a seeded sweep
    is reproducible cell-by-cell regardless of grid shape or order.

    Rows carry the label base rate (prevalence T/(T+F), the precision a random
    picker gets for free; identical across cells since it depends only on the
    labeling threshold) and ``precision_edge_mean`` (achieved precision minus
    base rate, averaged over years) — "precision p suffices" is only evidence
    of a strong model where that edge is large.
    """
    cagr_benchmark = compute_cagr(bmk_yearly.reindex(stock_yearly.index))
    rows = []
    for recall in recall_values:
        for precision in precision_values:
            custom_stats = compute_custom_stats(
                stock_yearly,
                bmk_yearly,
                recall,
                precision,
                num_simulations,
                label_threshold=label_threshold,
                rng=np.random.default_rng(model_random_seed),
            )
            cagr_custom_q05 = compute_cagr(custom_stats["q05"])
            rows.append(
                {
                    "recall": recall,
                    "precision": precision,
                    "achieved_recall_mean": float(custom_stats["achieved_recall"].mean()),
                    "achieved_precision_mean": float(custom_stats["achieved_precision"].mean()),
                    "base_rate_mean": float(custom_stats["base_rate"].mean()),
                    "base_rate_min": float(custom_stats["base_rate"].min()),
                    "base_rate_max": float(custom_stats["base_rate"].max()),
                    "precision_edge_mean": float(
                        (custom_stats["achieved_precision"] - custom_stats["base_rate"]).mean()
                    ),
                    "cagr_custom_q05": cagr_custom_q05,
                    "cagr_benchmark": cagr_benchmark,
                    "custom_q05_meets_benchmark": bool(
                        not pd.isna(cagr_custom_q05)
                        and not pd.isna(cagr_benchmark)
                        and cagr_custom_q05 >= cagr_benchmark
                    ),
                }
            )
    return pd.DataFrame(rows)


def sweep_recall_precision_pairs(
    recall_values: Iterable[float],
    precision_values: Iterable[float],
    *,
    year_start: int = 2000,
    year_end: int = 2024,
    benchmark: str = "SPY",
    tickers: Optional[List[str]] = None,
    tickers_by_year: Optional[Dict[int, List[str]]] = None,
    num_simulations: int = 5000,
    model_random_seed: Optional[int] = None,
    label_threshold: Optional[float] = None,
    output_csv: Optional[str] = "Precision_Recall_Tradeoff.csv",
) -> pd.DataFrame:
    """
    Evaluate a grid of (recall, precision) pairs and report where custom q05 CAGR
    meets or exceeds the benchmark CAGR. Prices and per-year universe returns are
    computed once and reused for every grid cell.

    See run_top_n_study for the meaning of `label_threshold`. Pass
    `output_csv=None` to skip writing the results to disk.
    """
    stock_yearly, bmk_yearly, _ = prepare_universe_returns(
        year_start,
        year_end,
        benchmark=benchmark,
        tickers=tickers,
        tickers_by_year=tickers_by_year,
    )
    df = sweep_from_returns(
        stock_yearly,
        bmk_yearly,
        recall_values,
        precision_values,
        num_simulations=num_simulations,
        model_random_seed=model_random_seed,
        label_threshold=label_threshold,
    ).round(4)
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 50)
    print(df)
    if output_csv:
        df.to_csv(output_csv)
    return df


DEFAULT_SCREEN_CRITERIA = ("0", "0.1", "bmk", "bmk+0.1")


def parse_criterion(spec: str) -> Tuple[str, float]:
    """
    Parse a criterion spec into (kind, value).

    A plain number is an absolute return threshold ("0", "0.1" for >10%);
    "bmk" with an optional offset ("bmk+0.1", "bmk-0.05") is relative to each
    year's benchmark return. Returns ("absolute", threshold) or
    ("benchmark", offset).
    """
    text = spec.strip().lower()
    if text.startswith("bmk"):
        rest = text[len("bmk"):]
        return "benchmark", float(rest) if rest else 0.0
    return "absolute", float(text)


def criterion_label(kind: str, value: float) -> str:
    """Human-readable criterion name, e.g. '>10%' or '>benchmark+10%'."""
    if kind == "benchmark":
        return ">benchmark" if value == 0 else f">benchmark{value * 100:+g}%"
    return f">{value * 100:g}%"


def screen_label_criteria(
    stock_yearly: pd.DataFrame,
    bmk_yearly: pd.Series,
    criteria: Iterable = DEFAULT_SCREEN_CRITERIA,
) -> pd.DataFrame:
    """
    Perfect-precision necessary-condition screen for candidate labeling criteria.

    For each criterion, the precision = 1.0 / recall = 1.0 portfolio is the
    equal-weighted mean of every stock meeting the criterion that year — the
    best any classifier trained on that criterion can average. If even this
    portfolio fails to beat the benchmark, no precision level rescues the
    criterion. One pass over precomputed returns; no selection simulation.

    Passing is necessary, not sufficient: a low-recall draw centers on the
    same mean but with real variance, and right-skewed positive returns put
    the typical draw below it — the risk analysis starts where this ends.

    `criteria` items are spec strings (see parse_criterion) or already-parsed
    (kind, value) tuples. Returns a long DataFrame, one row per
    (criterion, year): threshold, num_positive, base_rate,
    positive_mean_return, benchmark_return, excess.
    """
    rows = []
    for spec in criteria:
        kind, value = parse_criterion(spec) if isinstance(spec, str) else spec
        name = criterion_label(kind, value)
        for y in stock_yearly.index:
            bmk = bmk_yearly.get(y)
            if pd.isna(bmk):
                continue
            returns = stock_yearly.loc[y].dropna()
            if returns.empty:
                continue
            threshold = bmk + value if kind == "benchmark" else value
            positives = returns[returns > threshold]
            mean_pos = float(positives.mean()) if len(positives) else np.nan
            rows.append(
                {
                    "criterion": name,
                    "year": int(y),
                    "threshold": float(threshold),
                    "num_positive": int(len(positives)),
                    "base_rate": float(len(positives) / len(returns)),
                    "positive_mean_return": mean_pos,
                    "benchmark_return": float(bmk),
                    "excess": mean_pos - float(bmk),
                }
            )
    return pd.DataFrame(rows)


def summarize_screen(screen_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-criterion verdict on the necessary condition: does the perfect
    (precision=1, recall=1) portfolio beat the benchmark across the period?

    ``cagr_perfect`` compounds only years with a non-empty positive set
    (compute_cagr drops NaN years, mirroring the study's current no-pick
    behavior), so ``years_no_positive`` is reported to keep that visible.
    """
    rows = []
    for name, grp in screen_df.groupby("criterion", sort=False):
        by_year = grp.set_index("year")
        cagr_perfect = compute_cagr(by_year["positive_mean_return"])
        cagr_benchmark = compute_cagr(by_year["benchmark_return"])
        rows.append(
            {
                "criterion": name,
                "years": int(len(grp)),
                "years_no_positive": int((grp["num_positive"] == 0).sum()),
                "base_rate_mean": float(grp["base_rate"].mean()),
                "base_rate_min": float(grp["base_rate"].min()),
                "avg_positive_mean_return": float(grp["positive_mean_return"].mean()),
                "avg_benchmark_return": float(grp["benchmark_return"].mean()),
                "avg_excess": float(grp["excess"].mean()),
                "years_beating_benchmark": int((grp["excess"] > 0).sum()),
                "cagr_perfect": cagr_perfect,
                "cagr_benchmark": cagr_benchmark,
                "passes_screen": bool(
                    not pd.isna(cagr_perfect)
                    and not pd.isna(cagr_benchmark)
                    and cagr_perfect > cagr_benchmark
                ),
            }
        )
    return pd.DataFrame(rows).set_index("criterion")


def screen_criteria_feasibility(
    criteria: Iterable = DEFAULT_SCREEN_CRITERIA,
    *,
    year_start: int = 2000,
    year_end: int = 2024,
    benchmark: str = "SPY",
    tickers: Optional[List[str]] = None,
    tickers_by_year: Optional[Dict[int, List[str]]] = None,
    output_csv: Optional[str] = "Criterion_Feasibility.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download/prepare universe returns and run the perfect-precision screen.
    Returns (per-year table, per-criterion summary); writes the per-year
    table to `output_csv` unless None.
    """
    stock_yearly, bmk_yearly, _ = prepare_universe_returns(
        year_start,
        year_end,
        benchmark=benchmark,
        tickers=tickers,
        tickers_by_year=tickers_by_year,
    )
    screen = screen_label_criteria(stock_yearly, bmk_yearly, criteria)
    summary = summarize_screen(screen)
    if output_csv:
        screen.to_csv(output_csv, index=False)
    return screen, summary


def plot_results(res: StudyResult, n_values):
    plt.figure(figsize=(12, 7))

    # Benchmark
    spy = res.yearly["benchmark_return"].copy()
    plt.plot(spy.index, spy.values, linewidth=3, label="SPY", color="black")

    # Top-N
    for n in n_values:
        if n in res.by_n.columns:
            plt.plot(res.by_n.index, res.by_n[n], linewidth=1.8, label=f"Top {n}", alpha=0.9)
        '''
        if n in res.by_n_bottom.columns:
            plt.plot(res.by_n_bottom.index, res.by_n_bottom[n], linewidth=1.8, label=f"Bottom {n}", alpha=0.9)
        '''
    if "mean" in res.custom_stats.columns:
        plt.plot(
            res.custom_stats.index,
            res.custom_stats["mean"],
            linewidth=2.2,
            label="Custom (mean)",
            color="#7a3db8",
        )
        if "q05" in res.custom_stats.columns and "q95" in res.custom_stats.columns:
            plt.fill_between(
                res.custom_stats.index,
                res.custom_stats["q05"],
                res.custom_stats["q95"],
                color="#7a3db8",
                alpha=0.2,
                label="Custom (5-95% range)",
            )

    plt.axhline(0.0, linestyle="--", linewidth=1, alpha=0.6)
    plt.title("Top-N S&P 500 Stocks vs SPY (1-Year Calendar Returns)")
    plt.xlabel("Year")
    plt.ylabel("Total Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("topNAndCustom_vs_spy.png", dpi=150, bbox_inches="tight")
    plt.close()


def compute_cagr(return_series: pd.Series) -> float:
    series = return_series.dropna()
    if series.empty:
        return np.nan
    growth = (1.0 + series).prod()
    years = series.shape[0]
    if years == 0 or growth <= 0:
        return np.nan
    return float(growth ** (1.0 / years) - 1.0)


def compute_growth_series(return_series: pd.Series, initial_investment: float = 100.0) -> pd.Series:
    series = return_series.dropna().sort_index()
    if series.empty:
        return pd.Series(dtype=float)
    cumulative = (1.0 + series).cumprod() * initial_investment
    cumulative.name = f"${initial_investment:.0f} investment"
    return cumulative


def plot_investment_growth(res: StudyResult, n_values, initial_investment: float = 100.0):
    plt.figure(figsize=(12, 7))

    spy_growth = compute_growth_series(res.yearly["benchmark_return"], initial_investment)
    plt.plot(spy_growth.index, spy_growth.values, linewidth=3, label="SPY", color="black")

    for n in n_values:
        if n in res.by_n.columns:
            growth = compute_growth_series(res.by_n[n], initial_investment)
            plt.plot(growth.index, growth.values, linewidth=1.8, label=f"Top {n}", alpha=0.9)

    if "mean" in res.custom_stats.columns:
        custom_growth = compute_growth_series(res.custom_stats["mean"], initial_investment)
        plt.plot(
            custom_growth.index,
            custom_growth.values,
            linewidth=2.2,
            label="Custom (mean)",
            color="#7a3db8",
        )

    plt.title(f"Value of a ${initial_investment:.0f} Investment Reinvested Each Year")
    plt.xlabel("Year")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("topNAndCustom_growth.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_sweep_heatmap(
    sweep_df: pd.DataFrame,
    output_path: str = "recall_precision_heatmap.png",
) -> None:
    """
    Heatmap of the sweep grid: precision (rows) x recall (columns), colored by
    q05 CAGR minus benchmark CAGR. Blue cells are pairs whose 5th-percentile
    portfolio CAGR beats the benchmark, red cells fall short, the gray midpoint
    is dead even. Cells where the pair was infeasible (no reachable selection,
    so no q05) render in flat gray with "n/a".
    """
    from matplotlib.colors import LinearSegmentedColormap

    excess = sweep_df["cagr_custom_q05"] - sweep_df["cagr_benchmark"]
    grid = (
        sweep_df.assign(excess=excess)
        .pivot(index="precision", columns="recall", values="excess")
        .sort_index()
        .sort_index(axis=1)
    )
    values = np.ma.masked_invalid(grid.to_numpy(dtype=float))
    # Symmetric range so the gray midpoint always sits at exactly zero excess.
    span = float(np.abs(values).max()) if values.count() else 0.0
    span = max(span, 1e-6)

    cmap = LinearSegmentedColormap.from_list(
        "excess_diverging", ["#e34948", "#f0efec", "#2a78d6"]
    )
    cmap.set_bad("#e1e0d9")

    n_rows, n_cols = grid.shape
    fig, ax = plt.subplots(
        figsize=(2.2 + 1.1 * n_cols, 1.6 + 0.75 * n_rows)
    )
    mesh = ax.pcolormesh(
        values,
        cmap=cmap,
        vmin=-span,
        vmax=span,
        edgecolors="white",
        linewidth=2,
    )

    for i in range(n_rows):
        for j in range(n_cols):
            if values.mask is not np.ma.nomask and values.mask[i, j]:
                label, ink = "n/a", "#52514e"
            else:
                v = float(values[i, j])
                label = f"{v:+.1%}"
                r, g, b, _ = cmap((v + span) / (2 * span))
                ink = "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.5 else "#0b0b0b"
            ax.text(j + 0.5, i + 0.5, label, ha="center", va="center",
                    color=ink, fontsize=9)

    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels([f"{r:g}" for r in grid.columns])
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels([f"{p:g}" for p in grid.index])
    ax.set_xlabel("Target recall")
    ax.set_ylabel("Target precision")
    ax.set_title("Worst-case (q05) CAGR vs benchmark by (recall, precision)")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("q05 CAGR − benchmark CAGR")
    cbar.ax.yaxis.set_major_formatter(lambda v, _: f"{v:+.0%}")
    cbar.outline.set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--year-start", type=int, default=2012,
        help="First calendar year of the study (default: %(default)s).",
    )
    common.add_argument(
        "--year-end", type=int, default=2024,
        help="Last calendar year of the study, inclusive (default: %(default)s).",
    )
    common.add_argument(
        "--benchmark", default="SPY",
        help="Benchmark ticker (default: %(default)s).",
    )

    model_common = argparse.ArgumentParser(add_help=False)
    model_common.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for the simulated classifier (default: nondeterministic).",
    )
    model_common.add_argument(
        "--label-threshold", type=float, default=None,
        help="Absolute return the hypothetical classifier labels stocks against "
             "(e.g. 0.1 for '>10%% per year'). Omit to label against each year's "
             "benchmark return. Performance is always compared to the benchmark.",
    )

    parser = argparse.ArgumentParser(
        prog="pickn",
        description="Top-N winners vs S&P 500 study, (recall, precision) sweep, and "
                    "labeling-criterion feasibility screen. Running with no subcommand "
                    "is equivalent to 'study' with defaults.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    study = sub.add_parser(
        "study", parents=[common, model_common],
        help="Run the top-N study once for a single (recall, precision) pair and write plots.",
    )
    study.add_argument(
        "--n-values", type=int, nargs="+", default=[100, 250], metavar="N",
        help="Top-N portfolio sizes to evaluate (default: %(default)s).",
    )
    study.add_argument(
        "--recall", type=float, default=0.2,
        help="Target recall of the simulated classifier (default: %(default)s).",
    )
    study.add_argument(
        "--precision", type=float, default=0.7,
        help="Target precision of the simulated classifier (default: %(default)s).",
    )
    study.add_argument(
        "--num-simulations", type=int, default=1000,
        help="Random portfolio draws per year (default: %(default)s).",
    )
    study.add_argument(
        "--initial-investment", type=float, default=100.0,
        help="Starting value for the growth plot (default: %(default)s).",
    )
    study.add_argument(
        "--no-plots", action="store_true",
        help="Skip writing the PNG plots.",
    )

    sweep = sub.add_parser(
        "sweep", parents=[common, model_common],
        help="Run the study over a grid of (recall, precision) pairs and report "
             "where custom q05 CAGR meets the benchmark.",
    )
    sweep.add_argument(
        "--recalls", type=float, nargs="+", default=[0.1, 0.2, 0.3],
        metavar="R", help="Recall grid values (default: %(default)s).",
    )
    sweep.add_argument(
        "--precisions", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8],
        metavar="P", help="Precision grid values (default: %(default)s).",
    )
    sweep.add_argument(
        "--num-simulations", type=int, default=5000,
        help="Random portfolio draws per year per pair (default: %(default)s).",
    )
    sweep.add_argument(
        "--output", default="Precision_Recall_Tradeoff.csv",
        help="CSV path for the sweep results (default: %(default)s).",
    )
    sweep.add_argument(
        "--no-plots", action="store_true",
        help="Skip writing the heatmap PNG.",
    )

    screen = sub.add_parser(
        "screen", parents=[common],
        help="Feasibility screen: per-year return of the perfect (precision=1, "
             "recall=1) portfolio for each candidate labeling criterion. A "
             "criterion whose perfect portfolio can't beat the benchmark can't "
             "be rescued by any precision level.",
    )
    screen.add_argument(
        "--criteria", nargs="+", default=list(DEFAULT_SCREEN_CRITERIA), metavar="SPEC",
        help="Criterion specs: a number for an absolute return threshold "
             "(0, 0.1 for >10%%) or benchmark-relative 'bmk', 'bmk+0.1', "
             "'bmk-0.05' (default: %(default)s).",
    )
    screen.add_argument(
        "--output", default="Criterion_Feasibility.csv",
        help="CSV path for the per-year screen table (default: %(default)s).",
    )

    return parser


def run_study_command(args: argparse.Namespace) -> None:
    res = run_top_n_study(
        n_values=args.n_values,
        year_start=args.year_start,
        year_end=args.year_end,
        benchmark=args.benchmark,
        model_recall=args.recall,
        model_precision=args.precision,
        num_simulations=args.num_simulations,
        model_random_seed=args.seed,
        label_threshold=args.label_threshold,
    )

    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 50)

    print("\n=== N-level summary ===")
    print(res.summary.round(4))

    print("\n=== Sample yearly output (last 5 years) ===")
    print(res.yearly.tail(5).round(4))
    res.summary.to_csv('res_summary.csv', index=False)
    res.yearly.to_csv('res_yearly.csv', index=False)

    if not args.no_plots:
        plot_results(res, args.n_values)
        plot_investment_growth(res, args.n_values, initial_investment=args.initial_investment)


def run_sweep_command(args: argparse.Namespace) -> None:
    df = sweep_recall_precision_pairs(
        recall_values=args.recalls,
        precision_values=args.precisions,
        year_start=args.year_start,
        year_end=args.year_end,
        benchmark=args.benchmark,
        num_simulations=args.num_simulations,
        model_random_seed=args.seed,
        label_threshold=args.label_threshold,
        output_csv=args.output,
    )
    if not args.no_plots:
        plot_sweep_heatmap(df)


def run_screen_command(args: argparse.Namespace) -> None:
    screen, summary = screen_criteria_feasibility(
        args.criteria,
        year_start=args.year_start,
        year_end=args.year_end,
        benchmark=args.benchmark,
        output_csv=args.output,
    )
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 50)
    print("\n=== Criterion feasibility screen (precision=1, recall=1 portfolio) ===")
    print(screen.set_index(["criterion", "year"]).round(4).to_string())
    print("\n=== Screen summary by criterion ===")
    print(summary.round(4).to_string())
    print(
        "\npasses_screen: the perfect-classifier CAGR beats the benchmark CAGR. "
        "This is a necessary condition only — low-recall draws spread around the "
        "same yearly mean with real variance, and right-skewed positive returns "
        "put the typical draw below it."
    )


def main(argv: Optional[List[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        argv = ["study"]
    args = build_parser().parse_args(argv)
    if args.command == "sweep":
        run_sweep_command(args)
    elif args.command == "screen":
        run_screen_command(args)
    else:
        run_study_command(args)


if __name__ == "__main__":
    main()
