"""Tests for the cache validation in download_adj_close.

Pure pandas plus a stubbed yfinance download, no network. Run with:
    uv run python tests/test_download_adj_close_cache.py
"""
import tempfile
import types
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickn
from pickn import cache_covers_dates, download_adj_close


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


def install_download_stub(calls: list) -> None:
    """Replace pickn's yfinance handle with a recorder returning fake prices."""

    def fake_download(tickers, start, end, **kwargs):
        calls.append({"tickers": list(tickers), "start": start, "end": end})
        days = pd.bdate_range(start, min(pd.Timestamp(end), TODAY))
        cols = pd.MultiIndex.from_product([["Adj Close"], list(tickers)])
        return pd.DataFrame(42.0, index=days, columns=cols)

    pickn.yf = types.SimpleNamespace(download=fake_download)


def test_download_adj_close_cache_paths() -> None:
    calls: list = []
    install_download_stub(calls)

    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "adj_close_cache.csv"

        # 1. Cache covers tickers and dates -> served from cache, no download.
        make_cache(cache, ["AAA", "BBB"], "2012-01-03", "2024-12-31")
        out = download_adj_close(["AAA", "BBB"], "2015-01-01", "2020-01-01", cache_path=str(cache))
        assert calls == [], "network hit despite a valid cache"
        assert list(out.columns) == ["AAA", "BBB"]
        assert out.index.min() >= pd.Timestamp("2015-01-01")
        assert out.index.max() < pd.Timestamp("2020-01-01")

        # 2. Cache has the tickers but a narrower date range -> re-download.
        out = download_adj_close(["AAA", "BBB"], "2000-01-01", "2020-01-01", cache_path=str(cache))
        assert len(calls) == 1, "stale-date cache was silently served"
        assert (out == 42.0).all().all()
        # The rewritten cache now covers the wider period.
        rewritten = pd.read_csv(cache, index_col=0, parse_dates=True)
        assert rewritten.index.min() <= pd.Timestamp("2000-01-10")

        # 3. Cache missing a requested ticker -> re-download.
        make_cache(cache, ["AAA"], "2012-01-03", "2024-12-31")
        download_adj_close(["AAA", "CCC"], "2015-01-01", "2020-01-01", cache_path=str(cache))
        assert len(calls) == 2, "missing-ticker cache was silently served"


def main() -> None:
    test_cache_covers_dates()
    test_download_adj_close_cache_paths()
    print("All download_adj_close cache tests passed.")


if __name__ == "__main__":
    main()
