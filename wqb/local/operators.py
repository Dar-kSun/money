"""FastExpr operator implementations over (dates x instruments) arrays.

Cross-sectional operators (``rank``, ``group_neutralize``, ...) need to know
which instruments are in the universe on each day, so they take a
:class:`Context`. Time-series operators work column-wise and use ``min_periods
= 1``: they compute from whatever history is available rather than returning
NaN until a window fills. The engine discards a warm-up period instead, which
keeps sparse fundamental series usable.

Rolling means, sums and standard deviations are computed from cumulative sums
in O(T*N), because a search evaluates thousands of expressions and a per-day
Python loop dominates the runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Context:
    """Panel-level information the cross-sectional operators need."""

    universe: np.ndarray                      # bool (T, N)
    groups: dict[str, np.ndarray]             # name -> int codes (T, N)

    def group(self, name: str) -> np.ndarray:
        if name not in self.groups:
            raise KeyError(
                f"group {name!r} is not in the panel; available: "
                f"{', '.join(sorted(self.groups)) or 'none'}"
            )
        return self.groups[name]


# --------------------------------------------------------------- helpers
def _masked(x: np.ndarray, ctx: Context) -> np.ndarray:
    """NaN outside the universe, so out-of-universe names never get weight."""
    out = np.array(x, dtype=float, copy=True)
    if out.shape != ctx.universe.shape:
        # Everything normally flows from one panel, so a mismatch is a caller
        # bug; name it rather than letting numpy raise about axis sizes.
        raise ValueError(
            f"cross-sectional operator got shape {out.shape} but the universe "
            f"is {ctx.universe.shape}; both must come from the same panel"
        )
    out[~ctx.universe] = np.nan
    return out


def _rolling_sum_count(x: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware rolling sum and observation count over the last ``d`` rows."""
    valid = np.isfinite(x)
    filled = np.where(valid, x, 0.0)
    zeros = np.zeros((1, x.shape[1]))
    cum_sum = np.concatenate([zeros, np.cumsum(filled, axis=0)], axis=0)
    cum_cnt = np.concatenate([zeros, np.cumsum(valid, axis=0)], axis=0)

    idx = np.arange(1, x.shape[0] + 1)
    lo = np.maximum(idx - d, 0)
    return cum_sum[idx] - cum_sum[lo], cum_cnt[idx] - cum_cnt[lo]


def _sliding(x: np.ndarray, d: int, fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Apply ``fn`` over a trailing window, chunked to bound memory.

    ``fn`` receives an array of shape (rows, N, d) and reduces the last axis.
    """
    T, N = x.shape
    out = np.full((T, N), np.nan)
    padded = np.concatenate([np.full((d - 1, N), np.nan), x], axis=0)
    windows = np.lib.stride_tricks.sliding_window_view(padded, d, axis=0)  # (T, N, d)

    # Cap each chunk at roughly 4M elements so wide panels stay in memory.
    rows = max(1, int(4_000_000 / max(1, N * d)))
    for start in range(0, T, rows):
        stop = min(T, start + rows)
        out[start:stop] = fn(windows[start:stop])
    return out


def _nan_to_nan(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=float)
    return np.where(np.isfinite(out), out, np.nan)


def _as_scalar_int(value, *, minimum: int, label: str) -> int:
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"expected a scalar {label}, got an array")
        value = value.reshape(-1)[0]
    if not np.isfinite(value):
        raise ValueError(f"{label} must be a finite number, got {value}")
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{label} must be >= {minimum}, got {parsed}")
    return parsed


def _as_int(value) -> int:
    """A trailing-window length, which must cover at least one observation."""
    return _as_scalar_int(value, minimum=1, label="window")


def _as_lag(value) -> int:
    """A shift, where zero is legitimate (decay_linear's leading term)."""
    return _as_scalar_int(value, minimum=0, label="lag")


# ------------------------------------------------------- cross-sectional
def op_rank(ctx: Context, x: np.ndarray) -> np.ndarray:
    """Rank across instruments per day, scaled to [0, 1].

    This is what makes an alpha market neutral: it replaces raw magnitudes
    with a uniform spread across the universe.
    """
    values = _masked(x, ctx)
    out = np.full(values.shape, np.nan)
    for t in range(values.shape[0]):
        row = values[t]
        valid = np.flatnonzero(np.isfinite(row))
        if valid.size == 0:
            continue
        if valid.size == 1:
            out[t, valid] = 0.5
            continue
        order = np.argsort(row[valid], kind="stable")
        ranks = np.empty(valid.size, dtype=float)
        ranks[order] = np.arange(valid.size, dtype=float)
        out[t, valid] = ranks / (valid.size - 1)
    return out


def op_zscore(ctx: Context, x: np.ndarray) -> np.ndarray:
    """Cross-sectional standardisation per day."""
    values = _masked(x, ctx)
    mean = np.nanmean(values, axis=1, keepdims=True)
    std = np.nanstd(values, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return _nan_to_nan((values - mean) / np.where(std > 0, std, np.nan))


def op_scale(ctx: Context, x: np.ndarray, target: float = 1.0) -> np.ndarray:
    """Rescale each day so the absolute values sum to ``target``."""
    values = _masked(x, ctx)
    total = np.nansum(np.abs(values), axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return _nan_to_nan(values * float(target) / np.where(total > 0, total, np.nan))


def op_winsorize(ctx: Context, x: np.ndarray, std: float = 4.0) -> np.ndarray:
    """Clip cross-sectional outliers at +/- ``std`` standard deviations."""
    values = _masked(x, ctx)
    mean = np.nanmean(values, axis=1, keepdims=True)
    sigma = np.nanstd(values, axis=1, keepdims=True)
    lo, hi = mean - float(std) * sigma, mean + float(std) * sigma
    return _nan_to_nan(np.clip(values, lo, hi))


def _group_reduce(ctx: Context, x: np.ndarray, codes: np.ndarray, how: str) -> np.ndarray:
    values = _masked(x, ctx)
    out = np.full(values.shape, np.nan)
    for t in range(values.shape[0]):
        row, grp = values[t], codes[t]
        finite = np.isfinite(row) & (grp >= 0)
        if not finite.any():
            continue
        for code in np.unique(grp[finite]):
            sel = finite & (grp == code)
            block = row[sel]
            if how == "mean":
                out[t, sel] = block.mean()
            elif how == "neutralize":
                out[t, sel] = block - block.mean()
            elif how == "rank":
                if block.size == 1:
                    out[t, sel] = 0.5
                else:
                    order = np.argsort(block, kind="stable")
                    ranks = np.empty(block.size, dtype=float)
                    ranks[order] = np.arange(block.size, dtype=float)
                    out[t, sel] = ranks / (block.size - 1)
    return out


def op_group_neutralize(ctx: Context, x: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Subtract the group mean, leaving only within-group deviation."""
    return _group_reduce(ctx, x, _group_codes(ctx, group), "neutralize")


def op_group_rank(ctx: Context, x: np.ndarray, group: np.ndarray) -> np.ndarray:
    return _group_reduce(ctx, x, _group_codes(ctx, group), "rank")


def op_group_mean(ctx: Context, x: np.ndarray, group: np.ndarray) -> np.ndarray:
    return _group_reduce(ctx, x, _group_codes(ctx, group), "mean")


def _group_codes(ctx: Context, group: np.ndarray) -> np.ndarray:
    codes = np.asarray(group)
    if codes.dtype.kind == "f":
        codes = np.where(np.isfinite(codes), codes, -1).astype(np.int64)
    return codes.astype(np.int64)


# ------------------------------------------------------------ time series
def op_ts_delay(ctx: Context, x: np.ndarray, d: int) -> np.ndarray:
    """Shift down by ``d`` rows, so row t holds the value from t-d."""
    d = _as_lag(d)
    arr = np.asarray(x, dtype=float)
    if d == 0:
        return arr.copy()
    out = np.full_like(arr, np.nan)
    if d < arr.shape[0]:
        out[d:] = arr[:-d]
    return out


def op_ts_delta(ctx: Context, x: np.ndarray, d: int) -> np.ndarray:
    return _nan_to_nan(x - op_ts_delay(ctx, x, d))


def op_ts_sum(ctx: Context, x: np.ndarray, d: int) -> np.ndarray:
    total, count = _rolling_sum_count(x, _as_int(d))
    return np.where(count > 0, total, np.nan)


def op_ts_mean(ctx: Context, x: np.ndarray, d: int) -> np.ndarray:
    total, count = _rolling_sum_count(x, _as_int(d))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def op_ts_std_dev(ctx: Context, x: np.ndarray, d: int) -> np.ndarray:
    d = _as_int(d)
    total, count = _rolling_sum_count(x, d)
    sq_total, _ = _rolling_sum_count(np.square(x), d)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = np.maximum(count, 1)
        var = (sq_total - np.square(total) / n) / np.maximum(n - 1, 1)
    return np.where((count > 1) & (var >= 0), np.sqrt(np.maximum(var, 0.0)), np.nan)


def op_ts_zscore(ctx: Context, x: np.ndarray, d: int) -> np.ndarray:
    """Standardise against the instrument's own trailing window."""
    mean = op_ts_mean(ctx, x, d)
    std = op_ts_std_dev(ctx, x, d)
    with np.errstate(invalid="ignore", divide="ignore"):
        return _nan_to_nan((x - mean) / np.where(std > 0, std, np.nan))


def op_ts_min(ctx: Context, x: np.ndarray, d: int) -> np.ndarray:
    return _sliding(x, _as_int(d), lambda w: np.nanmin(w, axis=-1))


def op_ts_max(ctx: Context, x: np.ndarray, d: int) -> np.ndarray:
    return _sliding(x, _as_int(d), lambda w: np.nanmax(w, axis=-1))


def op_ts_rank(ctx: Context, x: np.ndarray, d: int) -> np.ndarray:
    """Where the latest value sits in its own trailing window, in [0, 1]."""
    def rank_last(w: np.ndarray) -> np.ndarray:
        current = w[..., -1]
        valid = np.isfinite(w)
        below = np.sum(np.where(valid, w < current[..., None], False), axis=-1)
        count = np.sum(valid, axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = below / np.maximum(count - 1, 1)
        return np.where((count > 1) & np.isfinite(current), out, np.nan)

    return _sliding(x, _as_int(d), rank_last)


def op_ts_argmax(ctx: Context, x: np.ndarray, d: int) -> np.ndarray:
    d = _as_int(d)

    def argmax(w: np.ndarray) -> np.ndarray:
        allnan = np.all(~np.isfinite(w), axis=-1)
        safe = np.where(np.isfinite(w), w, -np.inf)
        pos = (d - 1) - np.argmax(safe, axis=-1)
        return np.where(allnan, np.nan, pos.astype(float))

    return _sliding(x, d, argmax)


def op_ts_argmin(ctx: Context, x: np.ndarray, d: int) -> np.ndarray:
    d = _as_int(d)

    def argmin(w: np.ndarray) -> np.ndarray:
        allnan = np.all(~np.isfinite(w), axis=-1)
        safe = np.where(np.isfinite(w), w, np.inf)
        pos = (d - 1) - np.argmin(safe, axis=-1)
        return np.where(allnan, np.nan, pos.astype(float))

    return _sliding(x, d, argmin)


def op_ts_corr(ctx: Context, x: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
    """Trailing Pearson correlation between two series, per instrument."""
    d = _as_int(d)
    both = np.isfinite(x) & np.isfinite(y)
    xm = np.where(both, x, np.nan)
    ym = np.where(both, y, np.nan)

    sx, n = _rolling_sum_count(xm, d)
    sy, _ = _rolling_sum_count(ym, d)
    sxx, _ = _rolling_sum_count(np.square(xm), d)
    syy, _ = _rolling_sum_count(np.square(ym), d)
    sxy, _ = _rolling_sum_count(xm * ym, d)

    with np.errstate(invalid="ignore", divide="ignore"):
        nn = np.maximum(n, 1)
        cov = sxy - sx * sy / nn
        vx = sxx - np.square(sx) / nn
        vy = syy - np.square(sy) / nn
        denom = np.sqrt(np.maximum(vx, 0.0) * np.maximum(vy, 0.0))
        corr = cov / np.where(denom > 0, denom, np.nan)
    return np.where(n > 2, _nan_to_nan(corr), np.nan)


def op_decay_linear(ctx: Context, x: np.ndarray, d: int) -> np.ndarray:
    """Linearly weighted moving average, heaviest on the most recent day.

    This is the shape Brain's ``decay`` setting applies, and the reason decay
    lowers turnover: today's weights are blended with recent days'.
    """
    d = _as_int(d)
    weights = np.arange(d, 0, -1, dtype=float)      # d, d-1, ..., 1
    num = np.zeros_like(x, dtype=float)
    den = np.zeros_like(x, dtype=float)
    for lag, weight in enumerate(weights):
        shifted = op_ts_delay(ctx, x, lag)
        valid = np.isfinite(shifted)
        num += np.where(valid, shifted * weight, 0.0)
        den += np.where(valid, weight, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


# ------------------------------------------------------------- arithmetic
def op_abs(ctx: Context, x: np.ndarray) -> np.ndarray:
    return np.abs(x)


def op_log(ctx: Context, x: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return _nan_to_nan(np.log(np.where(np.asarray(x, dtype=float) > 0, x, np.nan)))


def op_sign(ctx: Context, x: np.ndarray) -> np.ndarray:
    return _nan_to_nan(np.sign(x))


def op_sqrt(ctx: Context, x: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return _nan_to_nan(np.sqrt(np.where(np.asarray(x, dtype=float) >= 0, x, np.nan)))


def op_signed_power(ctx: Context, x: np.ndarray, e: float) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    with np.errstate(invalid="ignore"):
        return _nan_to_nan(np.sign(arr) * np.power(np.abs(arr), float(np.ravel(e)[0])))


def op_min(ctx: Context, *args: np.ndarray) -> np.ndarray:
    stacked = np.stack([np.asarray(a, dtype=float) for a in _broadcast(args)])
    return np.nanmin(stacked, axis=0)


def op_max(ctx: Context, *args: np.ndarray) -> np.ndarray:
    stacked = np.stack([np.asarray(a, dtype=float) for a in _broadcast(args)])
    return np.nanmax(stacked, axis=0)


def op_if_else(ctx: Context, cond: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    cond_arr, a_arr, b_arr = _broadcast((cond, a, b))
    truthy = np.isfinite(cond_arr) & (cond_arr != 0)
    return _nan_to_nan(np.where(truthy, a_arr, b_arr))


def _broadcast(args) -> list[np.ndarray]:
    arrays = [np.asarray(a, dtype=float) for a in args]
    shape = np.broadcast_shapes(*[a.shape for a in arrays])
    return [np.broadcast_to(a, shape) for a in arrays]


OPERATORS: dict[str, Callable] = {
    # cross-sectional
    "rank": op_rank,
    "zscore": op_zscore,
    "normalize": op_zscore,
    "scale": op_scale,
    "winsorize": op_winsorize,
    "group_neutralize": op_group_neutralize,
    "group_rank": op_group_rank,
    "group_mean": op_group_mean,
    # time series
    "ts_delay": op_ts_delay,
    "ts_delta": op_ts_delta,
    "ts_sum": op_ts_sum,
    "ts_mean": op_ts_mean,
    "ts_std_dev": op_ts_std_dev,
    "ts_stddev": op_ts_std_dev,
    "ts_zscore": op_ts_zscore,
    "ts_min": op_ts_min,
    "ts_max": op_ts_max,
    "ts_rank": op_ts_rank,
    "ts_argmax": op_ts_argmax,
    "ts_argmin": op_ts_argmin,
    "ts_corr": op_ts_corr,
    "decay_linear": op_decay_linear,
    "ts_decay_linear": op_decay_linear,
    # arithmetic
    "abs": op_abs,
    "log": op_log,
    "sign": op_sign,
    "sqrt": op_sqrt,
    "signed_power": op_signed_power,
    "power": op_signed_power,
    "min": op_min,
    "max": op_max,
    "if_else": op_if_else,
}
