"""Tests for the cache validation and Sharadar fetch layer in download_adj_close.

Pure pandas plus a stubbed nasdaqdatalink.get_table, no network. Run with:
    uv run python tests/test_download_adj_close_cache.py
"""
import tempfile
import types
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickn
from pickn import cache_covers_dates, download_adj_close, fetch_adj_close


TODAY = pd.Timestamp("2026-07-06")


def test_cache_covers_dates() -> None:
    idx = pd.bdate_range("2012-01-03", "2024-12-31")

    # Empty cache never covers anything.
    assert not cache_covers_dates(pd.DatetimeIndex([]), "2012-01-01", "2013-01-01", today=TODAY)

    # Exact-period cache covers its own request, including the trading-day
    # slack at both endpoints (Jan 1-2 holidays, exclusive end date).
    assert cache_covers_dates(idx, "2012-01-01", "2025-01-01", today=TODAY)
    assert cache_covers_dates(idx, "2015-01-01", "2020-01-01", today=TODAY)

    # The bug this guards against: a 2012-2024 cache must NOT satisfy a
    # request starting in 2000.
    assert not cache_covers_dates(idx, "2000-01-01", "2025-01-01", today=TODAY)

    # Nor a request extending past its last date.
    assert not cache_covers_dates(idx, "2012-01-01", "2026-01-01", today=TODAY)

    # A requested end in the future is clamped to today: a fresh cache built
    # through yesterday still counts as covering an open-ended request.
    fresh = pd.bdate_range("2012-01-03", TODAY - pd.Timedelta(days=1))
    assert cache_covers_dates(fresh, "2012-01-01", "2030-01-01", today=TODAY)

    # But slack has limits: a month-wide gap is a real coverage hole.
    assert not cache_covers_dates(idx, "2011-12-01", "2025-01-01", today=TODAY)


def make_cache(path: Path, tickers, start, end) -> pd.DataFrame:
    days = pd.bdate_range(start, end)
    df = pd.DataFrame(
        {t: float(i + 1) for i, t in enumerate(tickers)}, index=days
    )
    df.to_csv(path, index_label="Date")
    return df


# What the fake Sharadar knows: SEP holds equities, SFP holds funds.
FAKE_TABLES = {
    "SHARADAR/SEP": {"AAA": 10.0, "BBB": 20.0, "CCC": 30.0},
    "SHARADAR/SFP": {"SPY": 400.0},
}


def install_get_table_stub(calls: list) -> None:
    """Replace pickn's nasdaqdatalink handle with a recorder returning
    long-form (ticker, date, closeadj) rows like the real datatables API."""

    def fake_get_table(table, ticker=None, date=None, qopts=None, paginate=None):
        calls.append({"table": table, "tickers": list(ticker), "date": dict(date)})
        prices = FAKE_TABLES[table]
        days = pd.bdate_range(date["gte"], min(pd.Timestamp(date["lte"]), TODAY))
        rows = [
            {"ticker": t, "date": d, "closeadj": prices[t]}
            for t in ticker
            if t in prices
            for d in days
        ]
        return pd.DataFrame(rows, columns=["ticker", "date", "closeadj"])

    pickn.nasdaqdatalink = types.SimpleNamespace(get_table=fake_get_table)


def test_fetch_adj_close_shapes() -> None:
    calls: list = []
    install_get_table_stub(calls)

    # Equities from SEP; the declared fund falls back to SFP; a ticker with
    # no data anywhere still gets an (all-NaN) column.
    out = fetch_adj_close(
        ["AAA", "SPY", "ZZZ"], "2015-01-01", "2016-01-01", fund_tickers=["SPY"]
    )
    assert list(out.columns) == ["AAA", "SPY", "ZZZ"]
    assert (out["AAA"] == 10.0).all()
    assert (out["SPY"] == 400.0).all()
    assert out["ZZZ"].isna().all()
    # end is exclusive
    assert out.index.max() < pd.Timestamp("2016-01-01")

    # SFP was queried only for the missing declared fund, never for ZZZ.
    sfp_calls = [c for c in calls if c["table"] == "SHARADAR/SFP"]
    assert len(sfp_calls) == 1 and sfp_calls[0]["tickers"] == ["SPY"]

    # An undeclared fund gets no SFP retry: funds are opt-in because a
    # reused stock symbol must not silently pick up a fund's prices.
    calls.clear()
    out = fetch_adj_close(["AAA", "SPY"], "2015-01-01", "2016-01-01")
    assert out["SPY"].isna().all()
    assert all(c["table"] == "SHARADAR/SEP" for c in calls)


def test_download_adj_close_cache_paths() -> None:
    calls: list = []
    install_get_table_stub(calls)

    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "cache" / "adj_close_cache.csv"
        cache.parent.mkdir()

        # 1. Cache covers tickers and dates -> served from cache, no download.
        make_cache(cache, ["AAA", "BBB"], "2012-01-03", "2024-12-31")
        out = download_adj_close(["AAA", "BBB"], "2015-01-01", "2020-01-01", cache_path=str(cache))
        assert calls == [], "network hit despite a valid cache"
        assert list(out.columns) == ["AAA", "BBB"]
        assert out.index.min() >= pd.Timestamp("2015-01-01")
        assert out.index.max() < pd.Timestamp("2020-01-01")

        # 2. Cache has the tickers but a narrower date range -> full
        # re-download (closeadj is backward-adjusted; no in-place extension).
        out = download_adj_close(["AAA", "BBB"], "2000-01-01", "2020-01-01", cache_path=str(cache))
        assert len(calls) == 1, "stale-date cache was silently served"
        assert (out["AAA"] == 10.0).all()
        # The rewritten cache now covers the wider period.
        rewritten = pd.read_csv(cache, index_col=0, parse_dates=True)
        assert rewritten.index.min() <= pd.Timestamp("2000-01-10")

        # 3. Cache covers the dates but misses a ticker -> only the missing
        # ticker is fetched and merged; existing columns are kept.
        calls.clear()
        make_cache(cache, ["AAA", "BBB"], "2012-01-03", "2024-12-31")
        out = download_adj_close(["AAA", "CCC"], "2015-01-01", "2020-01-01", cache_path=str(cache))
        assert len(calls) == 1, "expected exactly one incremental fetch"
        assert calls[0]["tickers"] == ["CCC"], "fetched more than the missing ticker"
        # Missing tickers are fetched over the cache's own span so the merged
        # cache stays valid for everything it covered before.
        assert pd.Timestamp(calls[0]["date"]["gte"]) <= pd.Timestamp("2012-01-03")
        assert (out["CCC"] == 30.0).all()
        merged = pd.read_csv(cache, index_col=0, parse_dates=True)
        assert set(merged.columns) == {"AAA", "BBB", "CCC"}

        # 4. A no-data ticker is cached as an all-NaN column and never
        # re-requested.
        calls.clear()
        download_adj_close(["AAA", "ZZZ"], "2015-01-01", "2020-01-01", cache_path=str(cache))
        assert len(calls) == 1 and calls[0]["tickers"] == ["ZZZ"]
        calls.clear()
        out = download_adj_close(["AAA", "ZZZ"], "2015-01-01", "2020-01-01", cache_path=str(cache))
        assert calls == [], "known-empty ticker was re-requested"
        assert out["ZZZ"].isna().all()


def main() -> None:
    test_cache_covers_dates()
    test_fetch_adj_close_shapes()
    test_download_adj_close_cache_paths()
    print("All download_adj_close cache tests passed.")


if __name__ == "__main__":
    main()
