"""Alpha expression generation (Phase 2).

The spec's key finding is that in USA / D1 / TOP3000 raw momentum loses money,
so every template here is **mean-reverting**: the signal is inverted with
``-1 *`` and wrapped in a cross-sectional ``rank()`` to force a market-neutral
long/short spread across the universe.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Iterable, Iterator

# Price/volume fields that exist in every USA equity universe.
DEFAULT_FIELDS: tuple[str, ...] = (
    "close",
    "open",
    "high",
    "low",
    "volume",
    "vwap",
    "returns",
    "cap",
)

DEFAULT_WINDOWS: tuple[int, ...] = (5, 10, 20, 60)


@dataclass(frozen=True)
class Template:
    """A parameterised alpha shape."""

    name: str
    expression: str          # format string over {field} and {d}
    rationale: str
    needs_window: bool = True

    def render(self, field: str, window: int) -> str:
        return self.expression.format(field=field, d=window)


# The blueprint from the spec comes first; the rest are variations on the same
# mean-reversion idea, each still inverted and still cross-sectionally ranked.
TEMPLATES: tuple[Template, ...] = (
    Template(
        "reversion_ma",
        "rank(-1 * ({field} / ts_mean({field}, {d}) - 1))",
        "The validated blueprint: long below the moving average, short above it.",
    ),
    Template(
        "reversion_zscore",
        "rank(-1 * ts_zscore({field}, {d}))",
        "Same reversion, scaled by trailing volatility so the signal is comparable across names.",
    ),
    Template(
        "reversion_delta",
        "rank(-1 * ts_delta({field}, {d}))",
        "Short-horizon reversal: fade the last d-day change.",
    ),
    Template(
        "reversion_ts_rank",
        "rank(-1 * ts_rank({field}, {d}))",
        "Fade names sitting high in their own trailing distribution.",
    ),
    Template(
        "reversion_range",
        "rank(-1 * (({field} - ts_min({field}, {d})) / "
        "(ts_max({field}, {d}) - ts_min({field}, {d}) + 1e-9)))",
        "Stochastic-oscillator reversion: fade position within the d-day range.",
    ),
    Template(
        "reversion_decay_linear",
        "rank(-1 * decay_linear({field} / ts_mean({field}, {d}) - 1, {d}))",
        "Blueprint smoothed by a linear decay, which lowers turnover further.",
    ),
    Template(
        "reversion_vol_scaled",
        "rank(-1 * ts_zscore({field}, {d}) / (1 + ts_std_dev(returns, {d})))",
        "Reversion damped in high-volatility names to protect drawdown.",
    ),
    Template(
        "reversion_group_neutral",
        "rank(group_neutralize(-1 * ts_zscore({field}, {d}), subindustry))",
        "Reversion measured against subindustry peers rather than the whole market.",
    ),
)

TEMPLATES_BY_NAME = {t.name: t for t in TEMPLATES}

# The spec's failed example: momentum that is NOT inverted.
_MOMENTUM_PATTERN = re.compile(
    r"(?<![\w.])(?:\w+)\s*/\s*ts_mean\s*\([^)]*\)\s*-\s*1", re.IGNORECASE
)


def looks_like_naive_momentum(expression: str) -> bool:
    """True when an expression repeats the spec's documented failure mode.

    ``close / ts_mean(close, 20) - 1`` buys price spikes in a mean-reverting
    market and scored a negative Sharpe (-0.87). An expression that contains
    that ratio with no inversion and no ranking is the same mistake.
    """
    expr = expression.strip()
    if not _MOMENTUM_PATTERN.search(expr):
        return False
    inverted = "-1 *" in expr.replace(" ", " ") or "-1*" in expr.replace(" ", "")
    ranked = "rank(" in expr
    return not (inverted or ranked)


@dataclass(frozen=True)
class Candidate:
    expression: str
    template: str
    field: str
    window: int

    @property
    def tags(self) -> list[str]:
        return ["auto", self.template, self.field, f"d{self.window}"]


def generate(
    fields: Iterable[str] = DEFAULT_FIELDS,
    windows: Iterable[int] = DEFAULT_WINDOWS,
    templates: Iterable[Template] = TEMPLATES,
    *,
    limit: int | None = None,
) -> Iterator[Candidate]:
    """Yield the template x field x window grid, de-duplicated.

    Ordered template-major so the validated blueprint is tried across every
    field before the more speculative shapes get a turn.
    """
    seen: set[str] = set()
    count = 0
    field_list = list(fields)
    window_list = list(windows)

    for template, field, window in itertools.product(templates, field_list, window_list):
        expression = template.render(field, window)
        normalized = re.sub(r"\s+", "", expression)
        if normalized in seen:
            continue
        seen.add(normalized)
        yield Candidate(expression=expression, template=template.name, field=field, window=window)
        count += 1
        if limit is not None and count >= limit:
            return


def grid_size(
    fields: Iterable[str] = DEFAULT_FIELDS,
    windows: Iterable[int] = DEFAULT_WINDOWS,
    templates: Iterable[Template] = TEMPLATES,
) -> int:
    return sum(1 for _ in generate(fields, windows, templates))


def resolve_templates(names: Iterable[str] | None) -> tuple[Template, ...]:
    """Map template names to Templates, raising on an unknown name."""
    if not names:
        return TEMPLATES
    resolved, unknown = [], []
    for name in names:
        template = TEMPLATES_BY_NAME.get(name)
        (resolved.append(template) if template else unknown.append(name))
    if unknown:
        raise ValueError(
            f"Unknown template(s): {', '.join(unknown)}. "
            f"Available: {', '.join(TEMPLATES_BY_NAME)}"
        )
    return tuple(resolved)
