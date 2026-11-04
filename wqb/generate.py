"""Alpha expression generation.

The context file's finding is that in USA / D1 / TOP3000 raw *price* momentum
loses money, so price templates must invert the signal. That lesson is
class-specific and the earlier version of this module over-applied it: it
forced every template to be inverted mean-reversion, which rules out drift on
fundamental and analyst data -- the least-crowded and best-documented region of
the space (post-earnings-announcement drift, analyst-revision momentum).

Templates therefore carry a :class:`~wqb.fields.Direction` and the set of field
classes they apply to. A price field only ever gets reversion templates; a
fundamental or analyst field only ever gets drift templates; classes with no
strong prior get both and the data decides.

One invariant holds everywhere: the output is wrapped in a cross-sectional
``rank()`` (or an explicit group neutralisation), which is what forces the
market-neutral long/short spread.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

from .fields import Direction, FieldClass, FieldInfo, classify

# Price/volume fields that exist in every USA equity universe. These are also
# the most heavily mined fields on the platform -- see wqb.fields.crowding --
# so they are a baseline, not a destination.
DEFAULT_FIELDS: tuple[str, ...] = (
    "close", "open", "high", "low", "volume", "vwap", "returns", "cap",
)

DEFAULT_WINDOWS: tuple[int, ...] = (5, 10, 20, 60)

# Lookback windows that make sense per class. Fundamental and ownership data
# updates quarterly, so a 5-day window over it is sampling noise, not signal.
WINDOWS_BY_CLASS: dict[FieldClass, tuple[int, ...]] = {
    FieldClass.PRICE: (5, 10, 20, 60),
    FieldClass.VOLUME: (5, 10, 20, 60),
    FieldClass.FUNDAMENTAL: (60, 120, 250),
    FieldClass.ANALYST: (20, 60, 120),
    FieldClass.OWNERSHIP: (60, 120, 250),
    FieldClass.SENTIMENT: (5, 10, 20, 60),
    FieldClass.OPTION: (5, 10, 20, 60),
    FieldClass.MODEL: (20, 60, 120),
    FieldClass.OTHER: (10, 20, 60),
}

_REVERSION_CLASSES = frozenset({
    FieldClass.PRICE, FieldClass.VOLUME, FieldClass.SENTIMENT,
    FieldClass.OPTION, FieldClass.OWNERSHIP, FieldClass.MODEL, FieldClass.OTHER,
})
_DRIFT_CLASSES = frozenset({
    FieldClass.FUNDAMENTAL, FieldClass.ANALYST, FieldClass.SENTIMENT,
    FieldClass.OPTION, FieldClass.OWNERSHIP, FieldClass.MODEL, FieldClass.OTHER,
})


@dataclass(frozen=True)
class Template:
    """A parameterised alpha shape."""

    name: str
    expression: str          # format string over {field} and {d}
    rationale: str
    direction: Direction
    applies_to: frozenset[FieldClass]

    def render(self, field: str, window: int) -> str:
        return self.expression.format(field=field, d=window)

    def accepts(self, field_class: FieldClass) -> bool:
        return field_class in self.applies_to


# --------------------------------------------------------------- reversion
# For price, the validated blueprint comes first; the rest vary the reversion
# measure. Every one inverts and ranks.
REVERSION_TEMPLATES: tuple[Template, ...] = (
    Template(
        "reversion_ma",
        "rank(-1 * ({field} / ts_mean({field}, {d}) - 1))",
        "The validated blueprint: long below the moving average, short above it.",
        Direction.REVERSION, _REVERSION_CLASSES,
    ),
    Template(
        "reversion_zscore",
        "rank(-1 * ts_zscore({field}, {d}))",
        "Reversion scaled by trailing volatility, so names are comparable.",
        Direction.REVERSION, _REVERSION_CLASSES,
    ),
    Template(
        "reversion_delta",
        "rank(-1 * ts_delta({field}, {d}))",
        "Short-horizon reversal: fade the last d-day change.",
        Direction.REVERSION, _REVERSION_CLASSES,
    ),
    Template(
        "reversion_ts_rank",
        "rank(-1 * ts_rank({field}, {d}))",
        "Fade names sitting high in their own trailing distribution.",
        Direction.REVERSION, _REVERSION_CLASSES,
    ),
    Template(
        "reversion_range",
        "rank(-1 * (({field} - ts_min({field}, {d})) / "
        "(ts_max({field}, {d}) - ts_min({field}, {d}) + 1e-9)))",
        "Stochastic-oscillator reversion: fade position within the d-day range.",
        Direction.REVERSION, _REVERSION_CLASSES,
    ),
    Template(
        "reversion_decay_linear",
        "rank(-1 * decay_linear({field} / ts_mean({field}, {d}) - 1, {d}))",
        "Blueprint smoothed by a linear decay, which lowers turnover further.",
        Direction.REVERSION, _REVERSION_CLASSES,
    ),
    Template(
        "reversion_vol_scaled",
        "rank(-1 * ts_zscore({field}, {d}) / (1 + ts_std_dev(returns, {d})))",
        "Reversion damped in high-volatility names to protect drawdown.",
        Direction.REVERSION, _REVERSION_CLASSES,
    ),
    Template(
        "reversion_group_neutral",
        "rank(group_neutralize(-1 * ts_zscore({field}, {d}), subindustry))",
        "Reversion measured against subindustry peers, not the whole market.",
        Direction.REVERSION, _REVERSION_CLASSES,
    ),
)

# ------------------------------------------------------------------- drift
# Slow data persists. Note `drift_ma` is the *same algebra* as the expression
# that scored -0.87 on price -- uninverted. That is the point: the sign is a
# property of the data, not of the formula.
DRIFT_TEMPLATES: tuple[Template, ...] = (
    Template(
        "drift_delta",
        "rank(ts_delta({field}, {d}))",
        "Follow the change: revisions and accounting improvements persist.",
        Direction.DRIFT, _DRIFT_CLASSES,
    ),
    Template(
        "drift_ma",
        "rank({field} / ts_mean({field}, {d}) - 1)",
        "Level versus its own trailing mean, uninverted -- drift, not reversal.",
        Direction.DRIFT, _DRIFT_CLASSES,
    ),
    Template(
        "drift_zscore",
        "rank(ts_zscore({field}, {d}))",
        "Standardised level: buy names improving relative to their own history.",
        Direction.DRIFT, _DRIFT_CLASSES,
    ),
    Template(
        "drift_scaled_by_cap",
        "rank(ts_delta({field}, {d}) / (cap + 1e-9))",
        "Size-normalised change, so large firms do not dominate the ranking.",
        Direction.DRIFT, _DRIFT_CLASSES,
    ),
    Template(
        "drift_group_neutral",
        "rank(group_neutralize(ts_zscore({field}, {d}), subindustry))",
        "Drift measured against subindustry peers, isolating idiosyncratic news.",
        Direction.DRIFT, _DRIFT_CLASSES,
    ),
    Template(
        "drift_persistence",
        "rank(ts_mean(ts_delta({field}, {d}), {d}))",
        "Averaged change: smooths a noisy or sparsely updated series.",
        Direction.DRIFT, _DRIFT_CLASSES,
    ),
)

TEMPLATES: tuple[Template, ...] = REVERSION_TEMPLATES + DRIFT_TEMPLATES
TEMPLATES_BY_NAME = {t.name: t for t in TEMPLATES}


# ------------------------------------------------------------- validation
_MA_RATIO = re.compile(
    r"(?<![\w.])([A-Za-z_]\w*)\s*/\s*ts_mean\s*\(\s*\1\s*,[^)]*\)\s*-\s*1", re.IGNORECASE
)


def looks_like_naive_momentum(expression: str) -> bool:
    """True when an expression repeats the documented *price* failure mode.

    ``close / ts_mean(close, 20) - 1`` buys price spikes in a mean-reverting
    market and scored a Sharpe of about -0.87. The same algebra over a
    fundamental or analyst field is a legitimate drift signal, so this only
    fires for price-like fields -- flagging it everywhere would rule out the
    most promising part of the search space.
    """
    expr = expression.strip()
    match = _MA_RATIO.search(expr)
    if not match:
        return False
    if classify(match.group(1)) not in (FieldClass.PRICE, FieldClass.VOLUME):
        return False
    compact = expr.replace(" ", "")
    return "-1*" not in compact and "rank(" not in expr


def validate_expression(expression: str) -> list[str]:
    """Problems that can be spotted without spending a simulation.

    Used both by the CLI's single-expression path and to screen machine- or
    LLM-proposed candidates before they reach the API.
    """
    problems: list[str] = []
    expr = expression.strip()

    if not expr:
        return ["expression is empty"]
    if expr.count("(") != expr.count(")"):
        problems.append("unbalanced parentheses")
    if looks_like_naive_momentum(expr):
        problems.append(
            "uninverted price momentum: the documented failure mode "
            "(Sharpe about -0.87); invert it or wrap it in rank()"
        )
    if "rank(" not in expr and "group_neutralize(" not in expr and "zscore(" not in expr:
        problems.append(
            "no cross-sectional normalisation (rank/group_neutralize/zscore), "
            "so positions are not market neutral"
        )
    return problems


# -------------------------------------------------------------- generation
@dataclass(frozen=True)
class Candidate:
    expression: str
    template: str
    field: str
    window: int
    field_class: FieldClass = FieldClass.OTHER
    direction: Direction = Direction.BOTH
    source: str = "template"      # "template" or "llm"
    hypothesis: str = ""

    @property
    def tags(self) -> list[str]:
        return [self.source, self.template, self.field, f"d{self.window}"]


def _as_field_info(field: str | FieldInfo) -> FieldInfo:
    return field if isinstance(field, FieldInfo) else FieldInfo(id=field)


def templates_for(field_class: FieldClass) -> tuple[Template, ...]:
    """Templates whose direction prior matches this class."""
    return tuple(t for t in TEMPLATES if t.accepts(field_class))


def windows_for(field_class: FieldClass, override: Sequence[int] | None = None) -> tuple[int, ...]:
    if override:
        return tuple(override)
    return WINDOWS_BY_CLASS.get(field_class, DEFAULT_WINDOWS)


def generate(
    fields: Iterable[str | FieldInfo] = DEFAULT_FIELDS,
    windows: Sequence[int] | None = None,
    templates: Iterable[Template] | None = None,
    *,
    limit: int | None = None,
) -> Iterator[Candidate]:
    """Yield the field x template x window grid, de-duplicated.

    Each field only draws templates its class accepts, and -- unless
    ``windows`` is given explicitly -- windows appropriate to its update
    frequency. Ordered field-major so each field is explored across shapes
    before moving on, which keeps a quota-limited run from spending everything
    on one field.
    """
    seen: set[str] = set()
    count = 0
    allowed = tuple(templates) if templates is not None else None

    for raw in fields:
        info = _as_field_info(raw)
        klass = info.field_class
        pool = allowed if allowed is not None else templates_for(klass)
        pool = tuple(t for t in pool if t.accepts(klass))

        for template, window in itertools.product(pool, windows_for(klass, windows)):
            expression = template.render(info.id, window)
            normalized = re.sub(r"\s+", "", expression)
            if normalized in seen:
                continue
            seen.add(normalized)
            yield Candidate(
                expression=expression,
                template=template.name,
                field=info.id,
                window=window,
                field_class=klass,
                direction=template.direction,
            )
            count += 1
            if limit is not None and count >= limit:
                return


def grid_size(
    fields: Iterable[str | FieldInfo] = DEFAULT_FIELDS,
    windows: Sequence[int] | None = None,
    templates: Iterable[Template] | None = None,
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
