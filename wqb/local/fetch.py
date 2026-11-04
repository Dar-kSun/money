"""Fetch free market data for the local simulator.

Two sources, both free and without an API key:

* **Stooq** publishes daily OHLCV for US equities, as one bulk archive or per
  symbol. This covers everything the price/volume templates need.
* **SEC EDGAR** publishes XBRL company facts for every US filer, which is the
  only free source of real fundamentals. Fundamental fields are also the
  least-mined region of the platform, so this is where uncrowded alphas are
  most likely to live.

Network access is deliberately confined to this module; parsing is pure and
tested separately, so a provider changing a header does not require guessing
about the rest of the pipeline.

SEC asks automated clients to identify themselves with a contact address in
the User-Agent. ``contact`` defaults to a generic string -- set it to your own
address if you are going to pull a lot, since that is what their fair-access
policy asks for.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

log = logging.getLogger(__name__)

STOOQ_BULK_URL = "https://static.stooq.com/db/h/d_us_txt.zip"
STOOQ_SYMBOL_URL = "https://stooq.com/q/d/l/?s={symbol}.us&i=d"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

DEFAULT_CONTACT = "wqb-local-sim (contact unset)"

# XBRL tags worth pulling: slow, widely reported, and economically meaningful.
SEC_TAGS: tuple[str, ...] = (
    "Assets", "Liabilities", "StockholdersEquity", "Revenues",
    "NetIncomeLoss", "OperatingIncomeLoss", "CashAndCashEquivalentsAtCarryingValue",
    "EarningsPerShareDiluted", "CommonStockSharesOutstanding",
)


class FetchError(RuntimeError):
    """A source could not be reached or returned something unusable."""


@dataclass
class RateLimit:
    """Minimum seconds between requests; SEC asks for no more than 10/second."""

    interval: float = 0.15
    _last: float = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self._last = time.monotonic()


def _get(url: str, *, contact: str = DEFAULT_CONTACT, timeout: float = 60.0) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": contact, "Accept-Encoding": "gzip, identity"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
            if response.headers.get("Content-Encoding") == "gzip":
                import gzip
                payload = gzip.decompress(payload)
            return payload
    except urllib.error.HTTPError as exc:
        raise FetchError(f"{url} returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise FetchError(
            f"could not reach {url}: {exc}. If this is a sandboxed "
            f"environment, the host may be blocked by a network policy."
        ) from exc


# ------------------------------------------------------------------- stooq
def parse_stooq_txt(text: str) -> list[dict[str, str]]:
    """Parse one Stooq ``.txt`` file.

    The bulk archive uses an angle-bracketed header
    (``<TICKER>,<PER>,<DATE>,...``) with ``YYYYMMDD`` dates; the per-symbol
    endpoint uses a plain ``Date,Open,...`` header with ISO dates. Both are
    normalised to ISO rows here.
    """
    rows: list[dict[str, str]] = []
    reader = csv.reader(io.StringIO(text.strip()))
    try:
        header = [h.strip().strip("<>").upper() for h in next(reader)]
    except StopIteration:
        return rows

    index = {name: i for i, name in enumerate(header)}
    needed = ("DATE", "OPEN", "HIGH", "LOW", "CLOSE")
    if any(key not in index for key in needed):
        return rows

    for raw in reader:
        if len(raw) < len(header):
            continue
        date = raw[index["DATE"]].strip()
        if len(date) == 8 and date.isdigit():
            date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        if len(date) != 10:
            continue
        try:
            values = {k.title(): float(raw[index[k]]) for k in needed[1:]}
        except (ValueError, IndexError):
            continue
        volume = 0.0
        for key in ("VOL", "VOLUME"):
            if key in index:
                try:
                    volume = float(raw[index[key]])
                except (ValueError, IndexError):
                    volume = 0.0
                break
        rows.append({"Date": date, **{k: str(v) for k, v in values.items()},
                     "Volume": str(volume)})
    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["Date", "Open", "High", "Low", "Close", "Volume"]
        )
        writer.writeheader()
        writer.writerows(rows)


def fetch_stooq_bulk(
    dest: str | Path,
    *,
    min_rows: int = 300,
    max_symbols: int | None = None,
    url: str = STOOQ_BULK_URL,
) -> int:
    """Download the whole US daily archive and write one CSV per symbol.

    One request for the entire market, which is far kinder than thousands of
    per-symbol calls. Returns the number of symbols written.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s", url)
    payload = _get(url)

    written = 0
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():
            if not name.lower().endswith(".txt"):
                continue
            symbol = Path(name).stem.split(".")[0].upper()
            try:
                text = archive.read(name).decode("utf-8", errors="replace")
            except KeyError:                       # pragma: no cover - defensive
                continue
            rows = parse_stooq_txt(text)
            if len(rows) < min_rows:
                continue
            _write_csv(dest / f"{symbol}.csv", rows)
            written += 1
            if max_symbols and written >= max_symbols:
                break
    if not written:
        raise FetchError(
            f"archive had no symbol with at least {min_rows} rows; "
            f"the format at {url} may have changed"
        )
    log.info("wrote %d symbols to %s", written, dest)
    return written


def fetch_stooq_symbols(
    symbols: Iterable[str],
    dest: str | Path,
    *,
    min_rows: int = 300,
    rate: RateLimit | None = None,
) -> int:
    """Per-symbol fallback for when the bulk archive is unavailable."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    rate = rate or RateLimit(interval=0.5)
    written = 0

    for symbol in symbols:
        rate.wait()
        try:
            text = _get(STOOQ_SYMBOL_URL.format(symbol=symbol.lower())).decode(
                "utf-8", errors="replace"
            )
        except FetchError as exc:
            log.warning("skipping %s: %s", symbol, exc)
            continue
        rows = parse_stooq_txt(text)
        if len(rows) >= min_rows:
            _write_csv(dest / f"{symbol.upper()}.csv", rows)
            written += 1
    return written


# --------------------------------------------------------------- sec edgar
def parse_sec_tickers(payload: bytes) -> dict[str, int]:
    """Map ticker -> CIK from SEC's company_tickers.json."""
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise FetchError("company_tickers.json was not valid JSON") from exc
    rows = data.values() if isinstance(data, dict) else data
    out: dict[str, int] = {}
    for row in rows:
        if isinstance(row, dict) and row.get("ticker") and row.get("cik_str") is not None:
            try:
                out[str(row["ticker"]).upper()] = int(row["cik_str"])
            except (TypeError, ValueError):
                continue
    if not out:
        raise FetchError("company_tickers.json contained no usable rows")
    return out


def parse_company_facts(
    payload: bytes, tags: Sequence[str] = SEC_TAGS
) -> dict[str, list[tuple[str, float]]]:
    """Extract (filing date, value) series per XBRL tag from company facts.

    ``filed`` is used rather than the period end date on purpose: it is when
    the number became public, and using the period end would leak information
    weeks before anyone could trade on it.
    """
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise FetchError("companyfacts response was not valid JSON") from exc

    facts = (data.get("facts") or {}).get("us-gaap") or {}
    out: dict[str, list[tuple[str, float]]] = {}

    for tag in tags:
        entry = facts.get(tag)
        if not isinstance(entry, dict):
            continue
        points: dict[str, float] = {}
        for unit_rows in (entry.get("units") or {}).values():
            if not isinstance(unit_rows, list):
                continue
            for row in unit_rows:
                if not isinstance(row, dict):
                    continue
                filed, value = row.get("filed"), row.get("val")
                if not filed or value is None:
                    continue
                try:
                    points[str(filed)[:10]] = float(value)
                except (TypeError, ValueError):
                    continue
        if points:
            out[tag] = sorted(points.items())
    return out


def fetch_sec_fundamentals(
    symbols: Sequence[str],
    dest: str | Path,
    *,
    contact: str = DEFAULT_CONTACT,
    tags: Sequence[str] = SEC_TAGS,
    rate: RateLimit | None = None,
) -> int:
    """Write one CSV of fundamental series per symbol.

    Each file is ``Date,<tag>,...`` with one row per filing date. Load these
    alongside the price panel and forward-fill, since a quarterly figure
    stays current until the next filing.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    rate = rate or RateLimit(interval=0.15)       # SEC: at most 10 requests/sec

    cik_by_ticker = parse_sec_tickers(_get(SEC_TICKERS_URL, contact=contact))
    written = 0

    for symbol in symbols:
        cik = cik_by_ticker.get(symbol.upper())
        if cik is None:
            log.warning("no CIK for %s", symbol)
            continue
        rate.wait()
        try:
            facts = parse_company_facts(
                _get(SEC_FACTS_URL.format(cik=cik), contact=contact), tags
            )
        except FetchError as exc:
            log.warning("skipping %s: %s", symbol, exc)
            continue
        if not facts:
            continue

        dates = sorted({d for series in facts.values() for d, _ in series})
        lookup = {tag: dict(series) for tag, series in facts.items()}
        path = dest / f"{symbol.upper()}.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["Date", *facts])
            for date in dates:
                writer.writerow([date, *(lookup[t].get(date, "") for t in facts)])
        written += 1
    return written
