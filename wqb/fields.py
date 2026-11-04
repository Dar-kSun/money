"""Data-field intelligence: economic class and how heavily mined a field is.

Two signals drive the search:

**Class** decides which templates even make sense. The context file's finding
that momentum loses money is specific to *price* in a liquid universe, where
short-horizon reversal dominates. It does not generalise: slow fundamental and
analyst series are the classic home of *drift* (post-earnings-announcement
drift, analyst-revision momentum). Applying the price lesson everywhere would
rule out the most promising fields.

**Crowding** decides which fields are worth spending quota on. BRAIN reports
``userCount`` and ``alphaCount`` per field -- how many users have used it and
how many alphas reference it. Price/volume is the most heavily mined region of
the platform, so a passing price alpha is also the most likely to be rejected
for correlation with the existing pool. Low-count fields are where
uncorrelated alphas remain.

NOTE: the ``userCount`` / ``alphaCount`` response keys are taken from community
tooling, not confirmed against the live API. :func:`FieldInfo.from_api` accepts
several spellings and leaves the counts ``None`` when absent, so ranking
degrades to coverage order rather than crashing.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Sequence


class FieldClass(str, Enum):
    PRICE = "price"
    VOLUME = "volume"
    FUNDAMENTAL = "fundamental"
    ANALYST = "analyst"
    SENTIMENT = "sentiment"
    OPTION = "option"
    OWNERSHIP = "ownership"
    MODEL = "model"
    OTHER = "other"


class Direction(str, Enum):
    """Which way a signal is expected to point."""

    REVERSION = "reversion"   # fade the move
    DRIFT = "drift"           # follow the move
    BOTH = "both"             # no strong prior; let the data decide


# Economic prior per class. These are priors, not conclusions -- the runner
# records what actually scores and the allocator reweights accordingly.
DIRECTION_PRIOR: dict[FieldClass, Direction] = {
    # Short-horizon reversal dominates liquid equity prices at D1. This is the
    # -0.87 lesson, and it is confined to these two classes.
    FieldClass.PRICE: Direction.REVERSION,
    FieldClass.VOLUME: Direction.REVERSION,
    # Slow-moving accounting data persists; value/quality effects drift.
    FieldClass.FUNDAMENTAL: Direction.DRIFT,
    # Analyst revisions are serially correlated (revision momentum, PEAD).
    FieldClass.ANALYST: Direction.DRIFT,
    # News/sentiment drifts briefly, then reverses; horizon decides.
    FieldClass.SENTIMENT: Direction.BOTH,
    FieldClass.OPTION: Direction.BOTH,
    FieldClass.OWNERSHIP: Direction.BOTH,
    FieldClass.MODEL: Direction.BOTH,
    FieldClass.OTHER: Direction.BOTH,
}

# Dataset-id prefixes BRAIN uses, checked before field-name keywords because a
# dataset is the more reliable signal.
_DATASET_PREFIXES: tuple[tuple[str, FieldClass], ...] = (
    ("pv", FieldClass.PRICE),
    ("fundamental", FieldClass.FUNDAMENTAL),
    ("analyst", FieldClass.ANALYST),
    ("news", FieldClass.SENTIMENT),
    ("socialmedia", FieldClass.SENTIMENT),
    ("sentiment", FieldClass.SENTIMENT),
    ("option", FieldClass.OPTION),
    ("institution", FieldClass.OWNERSHIP),
    ("insider", FieldClass.OWNERSHIP),
    ("shortinterest", FieldClass.OWNERSHIP),
    ("model", FieldClass.MODEL),
    ("risk", FieldClass.MODEL),
    ("macro", FieldClass.OTHER),
)

_PRICE_FIELDS = frozenset({"close", "open", "high", "low", "vwap", "returns", "cap", "adv20"})
_VOLUME_FIELDS = frozenset({"volume", "dollar_volume", "turnover", "sharesout"})

_KEYWORDS: tuple[tuple[FieldClass, tuple[str, ...]], ...] = (
    (FieldClass.ANALYST, ("analyst", "estimate", "est_", "revision", "recommend",
                          "target_price", "surprise", "consensus", "anl")),
    (FieldClass.FUNDAMENTAL, ("asset", "liabilit", "revenue", "income", "equity", "debt",
                              "cash", "eps", "sales", "ebit", "capex", "inventory",
                              "book", "margin", "earnings", "dividend", "accrual", "fnd")),
    (FieldClass.SENTIMENT, ("news", "sentiment", "buzz", "social", "tweet", "headline")),
    (FieldClass.OPTION, ("option", "implied", "_iv", "put_", "call_", "skew")),
    (FieldClass.OWNERSHIP, ("ownership", "holder", "institution", "insider",
                            "short_interest", "float")),
    (FieldClass.MODEL, ("risk", "beta", "factor_", "exposure", "model")),
)


@dataclass(frozen=True)
class FieldInfo:
    id: str
    description: str = ""
    dataset: str = ""
    type: str = "MATRIX"
    coverage: float | None = None
    user_count: int | None = None
    alpha_count: int | None = None

    @classmethod
    def from_api(cls, payload: dict[str, Any]) -> "FieldInfo":
        """Build from a ``/data-fields`` result row, tolerating key spellings."""
        dataset = payload.get("dataset")
        if isinstance(dataset, dict):
            dataset_id = str(dataset.get("id") or "")
        else:
            dataset_id = str(dataset or "")
        return cls(
            id=str(payload.get("id") or ""),
            description=str(payload.get("description") or ""),
            dataset=dataset_id,
            type=str(payload.get("type") or "MATRIX"),
            coverage=_num(payload.get("coverage")),
            user_count=_int(_first(payload, "userCount", "user_count", "users")),
            alpha_count=_int(_first(payload, "alphaCount", "alpha_count", "alphas")),
        )

    @property
    def field_class(self) -> FieldClass:
        return classify(self.id, self.description, self.dataset)

    @property
    def direction(self) -> Direction:
        return DIRECTION_PRIOR[self.field_class]

    @property
    def crowding(self) -> float | None:
        """How heavily mined this field is; higher means more crowded.

        Counts are heavy-tailed, so they are compared on a log scale. ``None``
        when the API reported neither count.
        """
        if self.user_count is None and self.alpha_count is None:
            return None
        users = math.log1p(max(0, self.user_count or 0))
        alphas = math.log1p(max(0, self.alpha_count or 0))
        # alphaCount is the more direct measure of how mined the field is;
        # userCount says how many people looked at all.
        return 0.65 * alphas + 0.35 * users


def classify(field_id: str, description: str = "", dataset: str = "") -> FieldClass:
    """Best-effort economic class for a field.

    Dataset wins over name, and an exact price/volume name wins over a keyword
    that happens to appear in a description.
    """
    fid = (field_id or "").strip().lower()

    if fid in _PRICE_FIELDS:
        return FieldClass.PRICE
    if fid in _VOLUME_FIELDS:
        return FieldClass.VOLUME

    ds = (dataset or "").strip().lower()
    for prefix, klass in _DATASET_PREFIXES:
        if ds.startswith(prefix):
            return klass

    haystack = f"{fid} {(description or '').lower()}"
    for klass, keywords in _KEYWORDS:
        if any(word in haystack for word in keywords):
            return klass
    return FieldClass.OTHER


def usable(field: FieldInfo, *, min_coverage: float = 0.5) -> bool:
    """Whether a field has enough coverage to build an alpha on.

    A field present for a tenth of the universe produces a concentrated,
    high-turnover alpha that fails Brain's own checks, so it is not worth
    simulating however uncrowded it is.
    """
    if field.type.upper() not in ("MATRIX", "VECTOR"):
        return False
    return field.coverage is None or field.coverage >= min_coverage


def rank_by_crowding(
    fields: Iterable[FieldInfo],
    *,
    min_coverage: float = 0.5,
    limit: int | None = None,
) -> list[FieldInfo]:
    """Least-crowded usable fields first.

    Fields with no reported counts sort last -- not because they are crowded,
    but because an unknown is not evidence of opportunity and quota should go
    to a field that is measurably underexplored.
    """
    candidates = [f for f in fields if usable(f, min_coverage=min_coverage)]
    ranked = sorted(
        candidates,
        key=lambda f: (
            f.crowding is None,
            f.crowding if f.crowding is not None else 0.0,
            -(f.coverage or 0.0),
            f.id,
        ),
    )
    return ranked[:limit] if limit else ranked


def group_by_class(fields: Iterable[FieldInfo]) -> dict[FieldClass, list[FieldInfo]]:
    grouped: dict[FieldClass, list[FieldInfo]] = {}
    for field in fields:
        grouped.setdefault(field.field_class, []).append(field)
    return grouped


def _first(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    parsed = _num(value)
    return None if parsed is None else int(parsed)
