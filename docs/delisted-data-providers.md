# Sourcing survivorship-bias-free price data (delisted stocks)

Research for the TODO item *"Source delisted-stock prices from a
survivorship-bias-free provider."* Compared July 2026.

## What this project actually needs

1. **Daily adjusted closes for every S&P 500 member since 1996-01-02**
   (the start of `SP500_HistoricalComponents_withChanges.csv`), *including*
   the roughly one-third of early-year constituents that have since
   delisted and have no Yahoo/yfinance history.
2. **Correct handling of reused tickers.** A delisted company's symbol is
   frequently recycled by an unrelated new listing (and occasionally a
   company relists). Symbol-keyed data silently splices two companies
   together — this is exactly what the reused-symbol guard in
   `year_universe_returns` defends against today. The provider must key
   history by a *permanent* security ID or by delist-date-stamped symbols,
   not by raw ticker.
3. **Fundamentals for delisted names** (income statement / balance sheet,
   ideally point-in-time as-reported) so a future theoretical
   stock-selection model can be trained without reintroducing the same
   bias on the features side.
4. **Linux/headless-friendly Python access** — the pipeline is
   `uv run python pickn.py` on Linux; no GUI, no Windows.
5. Individual-hobbyist budget.

## Candidates

### Sharadar (Nasdaq Data Link) — SEP prices + SF1 fundamentals ⭐ recommended

- **Coverage:** 21,000+ active *and delisted* US tickers; prices
  ([SEP](https://data.nasdaq.com/databases/SEP)) from **1998**; fundamentals
  ([SF1](https://data.nasdaq.com/databases/SF1)) from **~1990**, explicitly
  marketed as "nearly completely free from survivorship bias"
  ([sharadar.com](https://sharadar.com/)).
- **Ticker reuse:** solved properly with the **`permaticker`** — a unique,
  unchanging security ID. The `TICKERS` table maps symbol↔permaticker with
  listing periods, and `ACTIONS` records delistings (with reason), ticker
  changes, splits, spinoffs, and acquisitions.
- **Fundamentals for delisted stocks:** yes — SF1 is the standout here.
  As-reported, point-in-time dimensions (ARQ/ART etc.), covering dead
  companies. None of the other affordable options matches this.
- **Also included:** an S&P 500 historical-constituents table (SP500) in the
  [Core US Equities Bundle](https://data.nasdaq.com/databases/SFA), which
  could cross-check or replace the committed Wikipedia-derived CSV.
- **Access:** plain REST/bulk-CSV via Nasdaq Data Link (`nasdaq-data-link`
  Python package) — works headless on Linux; bulk export endpoints make a
  local cache (like today's `adj_close_cache.csv`) easy.
- **Cost:** pricing sits behind a Nasdaq Data Link login (per-dataset
  non-professional subscriptions, bundle discount); historically on the
  order of a few hundred USD/year for SEP+SF1. Verify before committing.
- **Gap:** price history starts **1998**, so 1996–97 remain
  yfinance-biased. Options: start the study at 1998, or splice those two
  years and report them as biased in `StudyResult.coverage`.

### Norgate Data — Platinum US Stocks (best pure-price alternative)

- **Coverage:** major-exchange US equities back to **1990**, delisted
  securities included at the Platinum/Diamond levels
  ([packages](https://norgatedata.com/stockmarketpackages.php)), plus
  **built-in point-in-time index constituency** (historical S&P 500
  membership flags) — this would eliminate both the Wikipedia CSV *and* the
  "first snapshot of the year" approximation in
  `get_sp500_tickers_by_year`.
- **Ticker reuse:** solved cleanly — delisted securities carry a
  delist-date suffix (e.g. Sun Microsystems = `JAVA-201001`), stored under
  their final name/ticker, so recycled symbols can never collide
  ([FAQ](https://norgatedata.com/data-package-faq.php)).
- **Cost:** Platinum US Stocks ≈ **USD 630/yr** (346.50/6mo)
  ([prices](https://norgatedata.com/prices.php)).
- **Fundamentals:** only a thin set of snapshot-style fields — not
  point-in-time, not usable for training a model on dead companies. Fails
  requirement 3.
- **Blocker for this repo:** data is delivered through the **Norgate Data
  Updater, a Windows-only desktop app**
  ([system requirements](https://norgatedata.com/system-requirements.php));
  the [`norgatedata`](https://pypi.org/project/norgatedata/) Python package
  reads NDU's local database, so Linux/cloud use requires a Windows VM and
  an export step. Excellent data, awkward fit for requirement 4.

### EODHD — ALL-IN-ONE plan

- **Coverage:** 30+ years claimed, with
  [delisted-company endpoints](https://eodhd.com/financial-apis/delisted-stock-companies-data)
  for EOD prices *and* fundamentals; simple REST, Linux-friendly.
- **Ticker reuse:** handled by renaming — a recycled symbol's previous
  owner becomes `ACR_old.US` etc. Workable but weaker than a permanent ID:
  the mapping is maintained by convention, historical symbol→company
  resolution is fiddlier, and community reports on delisted-data depth and
  quality (especially pre-2005) are mixed.
- **Cost:** delisted data is **only** in the ALL-IN-ONE plan at
  **$99.99/mo** ([pricing](https://eodhd.com/pricing)) — the $19.99/$29.99
  plans exclude it, so the effective price (~$1,200/yr) exceeds both
  Sharadar and Norgate.

### FirstRate Data — one-time purchase

- ~16,000 tickers incl. 7,000+ delisted, flat files, one-time fee with
  optional update subscription ([stock data](https://firstratedata.com/stock-data)).
- **Starts January 2000** — misses 1996–99 — and has no fundamentals.
  Only interesting if the study period is trimmed and budget dominates.

### CRSP (via WRDS) — the gold standard, noted for completeness

Permanent `PERMNO` IDs, coverage to 1926, and — uniquely — **delisting
returns** that capture the final-price-to-proceeds loss. Compustat
point-in-time fundamentals link via CCM. But licensing is institutional;
it is not realistically available (or affordable) to an individual.
Referenced here because its delisting-return concept matters (see caveat
below).

### Ruled out

- **Polygon.io** — delisted-ticker data is
  [spotty](https://polygon.io/knowledge-base/article/what-does-polygon-do-with-delisted-tickers)
  and ticker-centric; renamings must be stitched manually.
- **Tiingo** — merges history across renamings, so when a ticker is
  recycled the old company's data is lost
  ([symbology docs](https://www.tiingo.com/documentation/appendix/symbology)) —
  precisely the failure mode we must avoid.
- **Financial Modeling Prep** — its
  ["survivorship-bias-free" API](https://site.financialmodelingprep.com/developer/docs/survivorship-bias-api)
  is marked legacy; delisted depth/ID discipline unclear.
- **QuantConnect** — excellent bias-free data, but locked inside their
  platform; can't feed this local pipeline.

## Comparison summary

| | Prices from | Delisted prices | Reused-ticker handling | Delisted fundamentals | Linux API | Cost (indicative) |
|---|---|---|---|---|---|---|
| **Sharadar (SEP+SF1)** | 1998 | ✅ | ✅ permaticker | ✅ point-in-time | ✅ REST/bulk | ~few hundred $/yr (login to confirm) |
| **Norgate Platinum** | 1990 | ✅ | ✅ date-suffixed symbols | ❌ thin snapshot only | ⚠️ Windows-only updater | $630/yr |
| **EODHD ALL-IN-ONE** | ~1990s (depth varies) | ✅ | ⚠️ `_old` renaming | ✅ (quality unverified) | ✅ REST | $99.99/mo |
| **FirstRate Data** | 2000 | ✅ | ✅ kept separate | ❌ | ✅ flat files | one-time + optional updates |
| **CRSP/WRDS** | 1926 | ✅ + delisting returns | ✅ PERMNO | ✅ via Compustat | ✅ | institutional only |

## Recommendation

**Sharadar Core US Equities Bundle (SEP + SF1 + TICKERS/ACTIONS/SP500).**
It is the only affordable option that satisfies the fundamentals
requirement, its permaticker is the most robust answer to the
relisted/recycled-ticker problem, and its REST/bulk access drops straight
into the existing Linux pipeline. Accept the 1998 price-history start
(begin the study there, or keep 1996–97 flagged as biased).

If fundamentals were dropped as a requirement, **Norgate Platinum** would
win on price-data quality (1990 start + native point-in-time S&P 500
membership), but its Windows-only updater is a poor match for this repo's
headless workflow.

## Integration sketch (Sharadar)

1. Add a `download_adj_close`-shaped loader backed by SEP bulk download,
   cached locally like `adj_close_cache.csv`.
2. Resolve each constituent ticker *as of its membership year* to a
   permaticker via the TICKERS table (normalizing `.`→`-` as today), then
   key all returns by permaticker — this replaces, and strengthens, the
   reused-symbol guard in `year_universe_returns`.
3. Keep `StudyResult.coverage`: it should now show near-zero `no_data`
   drops, which is itself the acceptance test for the migration.
4. Optionally validate the committed constituents CSV against Sharadar's
   SP500 table.

## Remaining bias even with delisted prices

Measuring a delisted stock's return "to last available price" still
overstates outcomes when shareholders received less than the last quote
(bankruptcies). CRSP solves this with delisting returns; with Sharadar,
approximate it using the `ACTIONS` delist reason — e.g. force a −100%
(or configurable haircut) final return for bankruptcy/exchange-expulsion
delistings, while treating acquisitions at the final price as accurate.
