# Getting data for the local simulator

The local simulator needs a panel of US equity data. Both sources below are
free and need no API key. Neither can be reached from a sandboxed cloud
session with a restricted network policy — run these on your own machine.

## Prices (Stooq)

One request for the whole US market:

```bash
python -m wqb fetch prices --dest data/prices
```

That downloads `https://static.stooq.com/db/h/d_us_txt.zip` and writes one
`SYMBOL.csv` per instrument with at least 300 rows of history. Expect a few
thousand symbols and a few hundred MB.

To try the pipeline on a smaller slice first:

```bash
python -m wqb fetch prices --dest data/prices --max-symbols 500
```

If the bulk archive is unavailable, fall back to per-symbol requests (slower,
rate-limited to roughly two per second):

```bash
python -m wqb fetch prices --dest data/prices --symbols AAPL MSFT NVDA AMZN
```

**Survivorship bias.** Stooq's symbol list is the set of instruments trading
*now*, so companies that went bankrupt or were acquired are missing. Any
backtest over it is therefore flattered — the losers have been removed. This
matters less for ranking candidates against each other, which is what the
screen is for, than for believing an absolute Sharpe.

## Fundamentals (SEC EDGAR)

This is the part worth the effort. Fundamental and analyst fields are the
least-mined region of the platform, so they are where uncorrelated alphas are
most likely to remain — and SEC XBRL is the only free source of real
accounting data.

```bash
python -m wqb fetch fundamentals --symbols-from data/prices --dest data/fundamentals --contact you@example.com
```

Symbols are taken from whatever prices you already fetched. SEC asks automated
clients to identify themselves with a contact address in the `User-Agent`;
`--contact` sets it, and their fair-access policy asks for a real one if you
are pulling thousands of filings. Requests are rate-limited to under 10/second
as they require.

The fetcher pulls these XBRL tags: `Assets`, `Liabilities`,
`StockholdersEquity`, `Revenues`, `NetIncomeLoss`, `OperatingIncomeLoss`,
`CashAndCashEquivalentsAtCarryingValue`, `EarningsPerShareDiluted`,
`CommonStockSharesOutstanding`.

**Each value is keyed to its filing date, not the period end.** A quarter
ending 30 September is not public until the filing in early November, so
aligning on the period end would hand the backtest six weeks of information
nobody could have traded on. `load_fundamentals()` then shifts by one more day
by default, as a cushion against a filing landing after the close.

## Using both

```bash
python -m wqb screen --data data/prices --fundamentals data/fundamentals
```

```bash
python -m wqb search --data data/prices --fundamentals data/fundamentals \
  --fields assets revenues netincomeloss earningspersharediluted
```

Fundamental field names are lowercased XBRL tags, so `Assets` becomes
`assets`. `python -m wqb screen --data ... --fundamentals ...` prints the field
list it loaded.

## Trying it without any data

Every local command accepts `--synthetic`, which generates a panel instead of
loading one:

```bash
python -m wqb screen --synthetic --top 10
```

Useful for checking the pipeline runs. The results are not alphas — the data is
random — and the command says so on stderr.

## Other sources

Not implemented, for reference:

- **Nasdaq Data Link / Alpha Vantage / Polygon free tiers** — too rate-limited
  for a few thousand symbols (25–500 requests a day).
- **yfinance** — widely used and convenient, but it scrapes an undocumented
  endpoint, breaks without notice, and its terms are unclear for bulk use.
- **Sharadar / EODHD** — good fundamentals, paid.

If you add one, write the parser as a pure function and keep the network call
in `wqb/local/fetch.py`, which is how the existing ones stay testable offline.
