"""Panels to simulate over: synthetic generators and local-file loaders.

The synthetic generators exist for testing. A backtester is easiest to get
subtly wrong -- a sign flip, an off-by-one that leaks the future -- and real
data cannot tell you which. A panel with a known answer can: a mean-reverting
market must make reversion alphas profitable and momentum alphas lose, and a
planted signal must pay off only at the delay it was planted for.

For real work, :func:`load_csv_panel` reads a directory of per-symbol OHLCV
CSVs, which is the shape the free bulk sources hand you. See
``docs/data-sources.md`` for how to fetch them.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .panel import Panel


# --------------------------------------------------------------- synthetic
def synthetic_panel(
    n_dates: int = 800,
    n_instruments: int = 60,
    *,
    autocorr: float = 0.0,
    n_sectors: int = 6,
    daily_vol: float = 0.02,
    market_vol: float = 0.008,
    seed: int = 0,
) -> Panel:
    """A panel with controllable return autocorrelation.

    ``autocorr`` sets the AR(1) coefficient of each instrument's daily return:

    * ``autocorr < 0`` -- mean reverting. Yesterday's winners fall back, so
      reversion alphas should earn a positive Sharpe and momentum a negative
      one. This is the regime the context document describes for USA/D1.
    * ``autocorr > 0`` -- trending, and the signs reverse.
    * ``autocorr == 0`` -- unpredictable; every alpha should score near zero.
    """
    rng = np.random.default_rng(seed)
    T, N = n_dates, n_instruments

    innovations = rng.normal(0.0, daily_vol, size=(T, N))
    returns = np.zeros((T, N))
    returns[0] = innovations[0]
    for t in range(1, T):
        returns[t] = autocorr * returns[t - 1] + innovations[t]

    # A common factor, so neutralisation has something to remove.
    market = rng.normal(0.0, market_vol, size=(T, 1))
    returns = returns + market

    close = 50.0 * np.exp(np.cumsum(returns, axis=0))
    spread = 1.0 + rng.normal(0.0, 0.002, size=(T, N))
    volume = rng.lognormal(mean=13.0, sigma=0.6, size=(T, N))
    shares = rng.lognormal(mean=18.0, sigma=0.5, size=(1, N))

    sectors = np.tile(rng.integers(0, n_sectors, size=(1, N)), (T, 1)).astype(np.int64)

    return Panel.from_mapping(
        dates=_business_days(T),
        instruments=[f"S{i:04d}" for i in range(N)],
        fields={
            "close": close,
            "open": close * spread,
            "high": close * (1.0 + np.abs(rng.normal(0.0, 0.004, (T, N)))),
            "low": close * (1.0 - np.abs(rng.normal(0.0, 0.004, (T, N)))),
            "vwap": close * (1.0 + rng.normal(0.0, 0.001, (T, N))),
            "volume": volume,
            "returns": np.vstack([np.full((1, N), np.nan), returns[1:]]),
            "cap": close * shares,
        },
        groups={
            "sector": sectors,
            "industry": sectors,
            "subindustry": sectors,
            "market": np.zeros((T, N), dtype=np.int64),
        },
    )


def planted_panel(
    n_dates: int = 600,
    n_instruments: int = 50,
    *,
    exploitable_at_delay: int = 1,
    strength: float = 0.9,
    seed: int = 0,
) -> Panel:
    """A panel with a ``signal`` field that genuinely predicts future returns.

    The signal is planted so it pays off at exactly one delay. An alpha on
    ``signal`` run at ``delay == exploitable_at_delay`` should earn a large
    Sharpe; run at any other delay it should earn nothing.

    That pair of outcomes is what pins down the delay semantics. A position
    held on day ``t`` is built from ``signal[t - delay]`` and earns the return
    from ``t`` to ``t+1``, so the signal must be planted ``delay + 1`` rows
    ahead of the return it drives. Profiting at the *wrong* delay would mean
    the engine is reading information it should not have.
    """
    rng = np.random.default_rng(seed)
    T, N = n_dates, n_instruments

    noise = rng.normal(0.0, 0.02, size=(T, N))
    signal = rng.normal(0.0, 1.0, size=(T, N))

    returns = np.array(noise)
    # returns[t] is the move from t-1 to t, and must be driven by the signal
    # from delay+1 rows earlier for a delayed position to capture it.
    offset = int(exploitable_at_delay) + 1
    for t in range(T):
        source = t - offset
        if 0 <= source < T:
            returns[t] += strength * 0.02 * signal[source]

    close = 50.0 * np.exp(np.cumsum(returns, axis=0))
    sectors = np.tile(rng.integers(0, 4, size=(1, N)), (T, 1)).astype(np.int64)

    return Panel.from_mapping(
        dates=_business_days(T),
        instruments=[f"P{i:04d}" for i in range(N)],
        fields={
            "close": close,
            "signal": signal,
            "volume": rng.lognormal(13.0, 0.5, size=(T, N)),
            "cap": close * 1e6,
            "returns": np.vstack([np.full((1, N), np.nan), returns[1:]]),
        },
        groups={
            "sector": sectors, "industry": sectors,
            "subindustry": sectors, "market": np.zeros((T, N), dtype=np.int64),
        },
    )


def _business_days(count: int, start: str = "2018-01-01") -> np.ndarray:
    return np.busday_offset(
        np.datetime64(start, "D"), np.arange(count), roll="forward"
    ).astype("datetime64[D]")


# ------------------------------------------------------------------ loaders
@dataclass(frozen=True)
class Bar:
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


def load_csv_panel(
    directory: str | Path,
    *,
    pattern: str = "*.csv",
    min_history: int = 300,
    max_symbols: int | None = None,
    date_col: str = "Date",
    columns: Sequence[str] = ("Open", "High", "Low", "Close", "Volume"),
) -> Panel:
    """Build a panel from a directory of per-symbol OHLCV CSVs.

    Each file is one symbol named after the file stem, with a header row. This
    is the layout the free bulk sources produce (see
    ``docs/data-sources.md``). Symbols with less than ``min_history`` rows are
    dropped; dates are the union across symbols, and gaps are forward-filled
    so a missing day does not look like a price move.
    """
    directory = Path(directory)
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no files matching {pattern!r} in {directory}")

    per_symbol: dict[str, dict[str, Bar]] = {}
    for path in files:
        bars = _read_csv(path, date_col, columns)
        if len(bars) >= min_history:
            per_symbol[path.stem.upper()] = bars
        if max_symbols and len(per_symbol) >= max_symbols:
            break

    if not per_symbol:
        raise ValueError(
            f"no symbol in {directory} had at least {min_history} rows of history"
        )

    all_dates = sorted({d for bars in per_symbol.values() for d in bars})
    symbols = sorted(per_symbol)
    T, N = len(all_dates), len(symbols)
    index = {d: i for i, d in enumerate(all_dates)}

    arrays = {name: np.full((T, N), np.nan) for name in
              ("open", "high", "low", "close", "volume")}
    for col, symbol in enumerate(symbols):
        for date, bar in per_symbol[symbol].items():
            row = index[date]
            arrays["open"][row, col] = bar.open
            arrays["high"][row, col] = bar.high
            arrays["low"][row, col] = bar.low
            arrays["close"][row, col] = bar.close
            arrays["volume"][row, col] = bar.volume

    for name in ("open", "high", "low", "close"):
        arrays[name] = _forward_fill(arrays[name])
    arrays["volume"] = np.nan_to_num(arrays["volume"], nan=0.0)

    close = arrays["close"]
    with np.errstate(invalid="ignore", divide="ignore"):
        returns = np.vstack([np.full((1, N), np.nan), close[1:] / close[:-1] - 1.0])
    returns[~np.isfinite(returns)] = np.nan

    fields = dict(arrays)
    fields["vwap"] = (arrays["high"] + arrays["low"] + close) / 3.0
    fields["returns"] = returns
    # No share counts in OHLCV data, so dollar volume stands in for size when
    # ranking the universe by liquidity.
    fields["cap"] = close * arrays["volume"]

    zeros = np.zeros((T, N), dtype=np.int64)
    return Panel.from_mapping(
        dates=np.array(all_dates, dtype="datetime64[D]"),
        instruments=symbols,
        fields=fields,
        groups={"market": zeros},
    )


def _read_csv(path: Path, date_col: str, columns: Sequence[str]) -> dict[str, Bar]:
    bars: dict[str, Bar] = {}
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            return bars
        lookup = {name.strip().lower(): name for name in reader.fieldnames}

        def column(label: str) -> str | None:
            return lookup.get(label.strip().lower())

        date_key = column(date_col) or column("date")
        keys = [column(c) for c in columns]
        if date_key is None or any(k is None for k in keys):
            return bars

        for row in reader:
            try:
                values = [float(row[k]) for k in keys]  # type: ignore[index]
            except (TypeError, ValueError):
                continue
            date = str(row[date_key])[:10]
            if len(date) == 10 and all(math.isfinite(v) for v in values):
                bars[date] = Bar(date, *values)
    return bars


def _forward_fill(x: np.ndarray) -> np.ndarray:
    """Carry the last observation forward down each column."""
    out = np.array(x, dtype=float, copy=True)
    valid = np.isfinite(out)
    idx = np.where(valid, np.arange(out.shape[0])[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    filled = out[idx, np.arange(out.shape[1])[None, :]]
    seen = np.maximum.accumulate(valid, axis=0)
    return np.where(seen, filled, np.nan)


def add_universe(panel: Panel, size: int = 3000, by: str = "cap") -> Panel:
    """Restrict the panel to the ``size`` largest instruments by ``by``, per day."""
    panel.restrict_universe(panel.top_by(by, size))
    return panel
