"""
Utilities for loading point-in-time S&P 500 constituents.

The helpers fetch per-date membership from historical Wikipedia revisions
and cache the results locally for reproducibility.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List

import argparse

import pandas as pd
import requests

# Cache next to the codebase so repeated runs reuse downloaded memberships.
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "sp500_constituents_wikipedia_by_year.csv"

WIKI_PAGE_TITLE = "List_of_S%26P_500_companies"
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {
    # Wikipedia throttles/blocks generic clients; a descriptive UA improves reliability.
    "User-Agent": "MarketSims-constituents-fetch/1.0 (research use; contact maintainer)",
}


def _normalize_ticker(symbol: str) -> str:
    return str(symbol).strip().replace(".", "-")


def _get_revision_id_for_date(target_date: date, timeout: float = 10.0) -> int:
    """
    Find the Wikipedia revision that was current on or before the target_date.
    """
    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "rvprop": "ids|timestamp",
        "rvlimit": 1,
        "rvstart": f"{target_date.isoformat()}T23:59:59Z",
        "rvdir": "older",
        "titles": WIKI_PAGE_TITLE,
    }
    resp = requests.get(WIKI_API_URL, params=params, timeout=timeout, headers=WIKI_HEADERS)
    resp.raise_for_status()
    data = resp.json()
    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), None)
    if not page or "revisions" not in page:
        raise RuntimeError(f"No Wikipedia revision found for {target_date}")
    return page["revisions"][0]["revid"]


def _download_constituents_for_revision(revision_id: int, timeout: float = 10.0) -> List[str]:
    """
    Pull the constituents table for a given Wikipedia revision id.
    """
    url = f"https://en.wikipedia.org/w/index.php?title={WIKI_PAGE_TITLE}&oldid={revision_id}"
    tables = pd.read_html(url, storage_options={"User-Agent": WIKI_HEADERS["User-Agent"]})
    if not tables:
        raise RuntimeError(f"Could not parse constituents table from revision {revision_id}")
    table = tables[0]
    if "Symbol" not in table.columns:
        # Wikipedia sometimes labels the ticker column slightly differently.
        symbol_col = next((c for c in table.columns if "symbol" in str(c).lower()), None)
        if symbol_col is None:
            raise RuntimeError(f"No Symbol column found in revision {revision_id}")
        symbols = table[symbol_col]
    else:
        symbols = table["Symbol"]
    tickers = [_normalize_ticker(sym) for sym in symbols if str(sym).strip()]
    return sorted(set(tickers))


def _load_cache(cache_path: Path) -> Dict[int, List[str]]:
    if not cache_path.exists():
        return {}
    df = pd.read_csv(cache_path)
    return {
        int(year): sorted({_normalize_ticker(t) for t in df.loc[df["year"] == year, "ticker"]})
        for year in sorted(df["year"].unique())
    }


def _save_cache(cache_path: Path, memberships: Dict[int, List[str]]) -> None:
    rows = []
    for year, tickers in memberships.items():
        for t in sorted(set(tickers)):
            rows.append({"year": int(year), "ticker": _normalize_ticker(t)})
    df = pd.DataFrame(rows).sort_values(["year", "ticker"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)


def get_yearly_constituents_from_wikipedia(
    years: Iterable[int],
    cache_path: Path = DEFAULT_CACHE_PATH,
    refresh_missing: bool = True,
    timeout: float = 10.0,
) -> Dict[int, List[str]]:
    """
    Build a mapping of year -> tickers using historical Wikipedia revisions.

    Results are cached to avoid repeated network calls. If the cache is missing
    some requested years and refresh_missing=True, those years are fetched,
    merged with any cached data, and written back to disk.
    """
    years = sorted(set(int(y) for y in years))
    cached = _load_cache(cache_path)
    memberships: Dict[int, List[str]] = {**cached}

    missing_years = [y for y in years if y not in memberships]
    if refresh_missing and missing_years:
        for y in missing_years:
            rev_id = _get_revision_id_for_date(date(y, 1, 1), timeout=timeout)
            tickers = _download_constituents_for_revision(rev_id, timeout=timeout)
            memberships[y] = tickers
        _save_cache(cache_path, memberships)

    return {y: memberships[y] for y in years if y in memberships}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate point-in-time S&P 500 constituent cache from Wikipedia.")
    parser.add_argument(
        "--year-start",
        type=int,
        default=2000,
        help="First year (inclusive) to fetch membership for (default: 2000).",
    )
    parser.add_argument(
        "--year-end",
        type=int,
        default=datetime.utcnow().year,
        help="Last year (inclusive) to fetch membership for (default: current year).",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help=f"Where to write the cache CSV (default: {DEFAULT_CACHE_PATH}).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds for Wikipedia requests (default: 10).",
    )
    args = parser.parse_args()

    years = range(args.year_start, args.year_end + 1)
    memberships = get_yearly_constituents_from_wikipedia(
        years,
        cache_path=args.cache_path,
        refresh_missing=True,
        timeout=args.timeout,
    )
    print(f"Wrote {len(memberships)} years to cache: {args.cache_path}")


if __name__ == "__main__":
    main()
