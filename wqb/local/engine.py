"""Turn an alpha signal into positions, then into Brain-shaped metrics.

The pipeline mirrors the one Brain's settings describe: delay the signal, keep
only universe members, decay it, neutralise it against a group, truncate
per-instrument weight, scale to book size, then hold those weights for a day.

Metric definitions
------------------
``sharpe``    sqrt(252) * mean(daily pnl) / std(daily pnl)
``turnover``  mean(dollar traded per day) / book size
``fitness``   sharpe * sqrt(|returns| / max(turnover, 0.125))
``drawdown``  deepest peak-to-trough of cumulative pnl, as a fraction of book
``margin``    total pnl / total dollar traded

The fitness formula is corroborated by several independent write-ups of the
platform. ``returns`` is the one definition that is *not* publicly pinned down:
it is an annualised return, but whether the denominator is the whole book or
the long side only is unclear, and it feeds fitness. ``SimConfig.returns_basis``
selects it ("half_book" by default, the common community reading) and
:mod:`wqb.local.calibrate` exists to settle it against real Brain output.

So treat the absolute numbers as uncalibrated. What this is for is *ranking*
candidates cheaply, so that scarce real simulations go to the best few.

A trap worth knowing about
--------------------------
``rank()`` returns values in [0, 1] -- all non-negative -- so an alpha built
from it is **long-only** until something demeans it. With
``neutralization="NONE"`` the book therefore has a full market exposure, and
because geometric prices drift upward by sigma^2/2 per day, *every* expression
scores a positive Sharpe of around 1.5 on data with no predictability at all,
including two mutually opposite alphas. That is a real property of the
settings, not a bug, and it is why the default here is ``SUBINDUSTRY`` as on
Brain. Use ``NONE`` only if you mean to measure a long-biased book.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .evaluator import evaluate_expression
from .operators import Context, op_decay_linear, op_group_neutralize
from .panel import Panel

TRADING_DAYS = 252
TURNOVER_FLOOR = 0.125          # the max(turnover, 0.125) in the fitness formula

Neutralization = Literal["NONE", "MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY"]
ReturnsBasis = Literal["half_book", "book"]


@dataclass(frozen=True)
class SimConfig:
    """Local mirror of the Brain simulation settings."""

    delay: int = 1
    decay: int = 5
    truncation: float = 0.05
    neutralization: Neutralization = "SUBINDUSTRY"
    universe_size: int = 3000
    universe_field: str = "cap"
    book_size: float = 20_000_000.0
    warmup: int = 250               # days dropped so trailing windows are filled
    returns_basis: ReturnsBasis = "half_book"
    price_field: str = "close"


@dataclass
class Metrics:
    sharpe: float
    fitness: float
    turnover: float
    returns: float
    drawdown: float
    margin: float
    long_count: float
    short_count: float
    days: int
    pnl: np.ndarray = field(default_factory=lambda: np.array([]))

    def summary(self) -> str:
        return (
            f"sharpe={self.sharpe:.3f} fitness={self.fitness:.3f} "
            f"turnover={self.turnover:.3f} returns={self.returns:.3f} "
            f"drawdown={self.drawdown:.3f} days={self.days}"
        )

    def as_dict(self) -> dict[str, float]:
        """Metric names matching the Brain in-sample summary."""
        return {
            "sharpe": self.sharpe, "fitness": self.fitness,
            "turnover": self.turnover, "returns": self.returns,
            "drawdown": self.drawdown, "margin": self.margin,
            "longCount": self.long_count, "shortCount": self.short_count,
        }


def build_positions(signal: np.ndarray, panel: Panel, config: SimConfig) -> np.ndarray:
    """Signal -> dollar weights per day, summing to the book size in absolute value."""
    ctx = Context(universe=panel.universe, groups=panel.groups)
    weights = np.array(signal, dtype=float, copy=True)

    # Delay: a position held on day t may only use data through t - delay.
    if config.delay > 0:
        delayed = np.full_like(weights, np.nan)
        if config.delay < weights.shape[0]:
            delayed[config.delay:] = weights[: -config.delay]
        weights = delayed

    weights[~panel.universe] = np.nan

    if config.decay and config.decay > 1:
        weights = op_decay_linear(ctx, weights, config.decay)

    weights = _neutralize(weights, panel, ctx, config.neutralization)

    # Demeaning leaves a zero-sum vector; scale it so |weights| sums to the book.
    weights = _scale_to_book(weights, config.book_size)

    if config.truncation and config.truncation > 0:
        cap = config.truncation * config.book_size
        weights = np.clip(weights, -cap, cap)
        # Clipping changes the gross, so rescale; this is one pass, not Brain's
        # iterate-to-convergence, and can leave a few names at the cap.
        weights = _scale_to_book(weights, config.book_size)

    return np.nan_to_num(weights, nan=0.0)


def _neutralize(
    weights: np.ndarray, panel: Panel, ctx: Context, how: Neutralization
) -> np.ndarray:
    how = (how or "NONE").upper()
    if how == "NONE":
        # Deliberately leaves any long bias in place; see the module docstring.
        return weights
    if how == "MARKET":
        return weights - _row_mean(weights, panel.universe)
    group = how.lower()
    if group not in panel.groups:
        # Fall back to market neutrality rather than silently skipping it.
        return weights - _row_mean(weights, panel.universe)
    return op_group_neutralize(ctx, weights, panel.groups[group])


def _row_mean(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-day mean over in-universe, finite entries; NaN for an empty row.

    Written out rather than using nanmean so an all-NaN warm-up row does not
    emit a RuntimeWarning on every simulation.
    """
    values = np.where(mask, x, np.nan)
    finite = np.isfinite(values)
    count = finite.sum(axis=1, keepdims=True)
    total = np.where(finite, values, 0.0).sum(axis=1, keepdims=True)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _scale_to_book(weights: np.ndarray, book_size: float) -> np.ndarray:
    gross = np.nansum(np.abs(weights), axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        scaled = weights * book_size / np.where(gross > 0, gross, np.nan)
    return scaled


def simulate(
    expression: str,
    panel: Panel,
    config: SimConfig | None = None,
    *,
    signal: np.ndarray | None = None,
) -> Metrics:
    """Evaluate an expression and score it the way Brain reports in-sample.

    Pass ``signal`` to score an already-evaluated array and skip parsing.
    """
    config = config or SimConfig()
    if signal is None:
        signal = evaluate_expression(expression, panel)

    weights = build_positions(signal, panel, config)
    forward = panel.forward_returns(config.price_field)

    # PnL earned on day t by the weights held at the close of day t.
    daily = np.nansum(weights * np.nan_to_num(forward, nan=0.0), axis=1)

    # Dollar traded on day t, versus the previous day's weights.
    traded = np.nansum(np.abs(np.diff(weights, axis=0)), axis=1)
    traded = np.concatenate([[np.nan], traded])

    start = min(max(config.warmup, config.delay), max(panel.n_dates - 2, 0))
    # The final row's forward return is unknown, so it cannot contribute pnl.
    pnl = daily[start:-1]
    traded_window = traded[start:-1]
    weights_window = weights[start:-1]

    return _metrics(pnl, traded_window, weights_window, config)


def _metrics(
    pnl: np.ndarray, traded: np.ndarray, weights: np.ndarray, config: SimConfig
) -> Metrics:
    days = int(pnl.size)
    empty = Metrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, days, pnl)
    if days < 2 or not np.any(np.isfinite(pnl)):
        return empty

    pnl = np.nan_to_num(pnl, nan=0.0)
    std = float(pnl.std(ddof=1))
    mean = float(pnl.mean())
    sharpe = float(np.sqrt(TRADING_DAYS) * mean / std) if std > 0 else 0.0

    book = config.book_size
    denominator = book / 2.0 if config.returns_basis == "half_book" else book
    returns = float(TRADING_DAYS * mean / denominator) if denominator else 0.0

    daily_traded = np.nan_to_num(traded, nan=0.0)
    turnover = float(daily_traded.mean() / book) if book else 0.0

    fitness = (
        float(sharpe * np.sqrt(abs(returns) / max(turnover, TURNOVER_FLOOR)))
        if turnover >= 0 else 0.0
    )

    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
    drawdown = float(np.max(peak - equity) / book) if book else 0.0

    total_traded = float(daily_traded.sum())
    margin = float(pnl.sum() / total_traded) if total_traded > 0 else 0.0

    long_count = float(np.mean(np.sum(weights > 0, axis=1)))
    short_count = float(np.mean(np.sum(weights < 0, axis=1)))

    return Metrics(
        sharpe=sharpe, fitness=fitness, turnover=turnover, returns=returns,
        drawdown=drawdown, margin=margin, long_count=long_count,
        short_count=short_count, days=days, pnl=pnl,
    )


def correlation(left: Metrics | np.ndarray, right: Metrics | np.ndarray) -> float:
    """Correlation of two alphas' daily pnl.

    This is the quantity Brain's self-correlation check measures, computed on
    daily changes rather than a cumulative curve. Checking it locally before
    submitting is far cheaper than finding out from a rejected submission.
    """
    a = left.pnl if isinstance(left, Metrics) else np.asarray(left, dtype=float)
    b = right.pnl if isinstance(right, Metrics) else np.asarray(right, dtype=float)
    n = min(a.size, b.size)
    if n < 3:
        return 0.0
    a, b = np.nan_to_num(a[-n:], nan=0.0), np.nan_to_num(b[-n:], nan=0.0)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
