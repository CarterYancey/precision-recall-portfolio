"""Tests for get_sp500_tickers_by_year against the data/SP500_history.csv
format: change-dated rows whose tickers column is a python list literal.

No network. Run with:
    uv run python tests/test_constituents_loader.py
"""
import tempfile
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pickn import SP500_HISTORY_CSV, get_sp500_tickers_by_year


def write_history(path: Path, rows) -> None:
    # Quote the list literal like the real file does.
    lines = ["date,tickers"] + [f'{date},"{tickers}"' for date, tickers in rows]
    path.write_text("\n".join(lines) + "\n")


def test_as_of_january_first_semantics() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "history.csv"
        write_history(
            csv,
            [
                ("1996-01-01", ["AAA", "BRK.B", "CCC"]),
                ("1996-06-15", ["AAA", "BRK.B", "DDD"]),  # mid-1996 change
                ("1998-03-01", ["AAA", "EEE"]),           # change after 1998 starts
            ],
        )
        by_year = get_sp500_tickers_by_year(str(csv))

        # 1996: the snapshot in force on Jan 1 is the first row, not the
        # mid-year change.
        assert by_year[1996] == ["AAA", "BRK.B", "CCC"]
        # 1997 has no rows of its own: membership is the last 1996 snapshot.
        assert by_year[1997] == ["AAA", "BRK.B", "DDD"]
        # 1998's only row is dated March, i.e. AFTER Jan 1 — the year-start
        # snapshot is still the 1996-06-15 list.
        assert by_year[1998] == ["AAA", "BRK.B", "DDD"]
        # No years past the file's last row.
        assert set(by_year) == {1996, 1997, 1998}

        # Tickers are Sharadar-native: dots preserved, no '-' rewriting.
        assert "BRK.B" in by_year[1996]


def test_years_before_first_row_are_absent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "history.csv"
        write_history(
            csv,
            [
                ("1996-01-19", ["AAA"]),  # first snapshot AFTER Jan 1 1996
                ("1997-02-01", ["BBB"]),
            ],
        )
        by_year = get_sp500_tickers_by_year(str(csv))
        # 1996's Jan-1 membership is unknown, so the year is skipped rather
        # than backfilled from a later snapshot.
        assert 1996 not in by_year
        assert by_year[1997] == ["AAA"]


def test_real_committed_file() -> None:
    by_year = get_sp500_tickers_by_year(str(REPO_ROOT / SP500_HISTORY_CSV))
    assert 1996 in by_year and 2026 in by_year
    for year, tickers in by_year.items():
        assert 400 <= len(tickers) <= 520, f"{year}: implausible count {len(tickers)}"
        assert len(tickers) == len(set(tickers))
    # Sharadar-native share-class tickers keep their dot.
    assert "BF.B" in by_year[1996]


def main() -> None:
    test_as_of_january_first_semantics()
    test_years_before_first_row_are_absent()
    test_real_committed_file()
    print("All constituents loader tests passed.")


if __name__ == "__main__":
    main()
