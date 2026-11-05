"""Structural diversity, as a proxy for correlation.

Self-correlation against the existing alpha pool is the usual reason a
submission is refused, and the practical finding from people mining this
platform is that changing a lookback window, a weight or the neutralisation
barely moves correlation -- a different data source or a different economic
mechanism is what moves it.

That has a direct consequence for a template grid: ``rank(-1 * ts_zscore(close,
5))`` and ``rank(-1 * ts_zscore(close, 60))`` are two rows in the grid but one
idea, and submitting both wastes a slot. This module reduces an expression to a
structural signature so near-duplicates can be collapsed:

* :func:`family` keeps the fields and abstracts the constants -- the unit of
  submission diversity, since window variants of one shape collapse together.
* :func:`skeleton` abstracts the fields to their class as well -- the unit of
  *mechanism* diversity, for comparing one idea across different data.

It is a proxy, not a correlation measurement. Two different families can still
correlate; only Brain's own check settles that. What this does guarantee is
that quota and submission slots are not spent re-testing one idea.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from .fields import classify

# Grouping operands are structural, not data: keep them verbatim.
GROUP_OPERANDS = frozenset({
    "market", "sector", "industry", "subindustry", "exchange", "country",
    "sector_industry", "densify",
})

_TOKEN = re.compile(r"[A-Za-z_]\w*|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\S")


def _canonicalise(expression: str, *, abstract_fields: bool) -> str:
    """Reduce an expression to a signature string.

    Operator names and grouping operands are kept, numeric constants collapse
    to ``N``, and data fields are either kept (family) or replaced by their
    economic class (skeleton).
    """
    tokens = _TOKEN.findall(expression or "")
    out: list[str] = []

    for index, token in enumerate(tokens):
        if re.fullmatch(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", token):
            out.append("N")
            continue
        if not re.fullmatch(r"[A-Za-z_]\w*", token):
            out.append(token)
            continue

        is_call = index + 1 < len(tokens) and tokens[index + 1] == "("
        lowered = token.lower()
        if is_call or lowered in GROUP_OPERANDS:
            out.append(lowered)          # operator or grouping operand
        elif abstract_fields:
            out.append(classify(lowered).value.upper())
        else:
            out.append(lowered)
    return "".join(out)


def family(expression: str) -> str:
    """Signature that ignores constants but keeps the fields.

    Window variants of one shape share a family, so they count once for
    submission diversity.
    """
    return _canonicalise(expression, abstract_fields=False)


def skeleton(expression: str) -> str:
    """Signature that also abstracts fields to their class.

    The same mechanism applied to different data shares a skeleton but not a
    family -- useful for checking a book is not all one idea.
    """
    return _canonicalise(expression, abstract_fields=True)


def same_family(left: str, right: str) -> bool:
    return family(left) == family(right)


def family_counts(expressions: Iterable[str]) -> Counter[str]:
    return Counter(family(e) for e in expressions)


def skeleton_counts(expressions: Iterable[str]) -> Counter[str]:
    return Counter(skeleton(e) for e in expressions)


@dataclass(frozen=True)
class DiversityReport:
    total: int
    families: int
    skeletons: int

    @property
    def family_ratio(self) -> float:
        """Distinct ideas per alpha; 1.0 means every alpha is its own idea."""
        return self.families / self.total if self.total else 0.0

    def describe(self) -> str:
        return (
            f"{self.total} alpha(s), {self.families} distinct famil(ies), "
            f"{self.skeletons} distinct mechanism(s) "
            f"({self.family_ratio:.0%} unique)"
        )


def report(expressions: Sequence[str]) -> DiversityReport:
    """Summarise how many genuinely distinct ideas a set of alphas contains.

    A book can look diverse by count and be a handful of ideas in disguise;
    this is the number that matters when the gate is correlation.
    """
    items = list(expressions)
    return DiversityReport(
        total=len(items),
        families=len(set(family(e) for e in items)),
        skeletons=len(set(skeleton(e) for e in items)),
    )


def select_diverse(
    expressions: Iterable[str],
    *,
    max_per_family: int = 1,
    max_per_skeleton: int | None = None,
) -> list[str]:
    """Pick a structurally varied subset, preserving input order.

    Order matters: pass the best alphas first (e.g. sorted by Sharpe) so the
    strongest member of each family is the one kept.
    """
    fam_seen: Counter[str] = Counter()
    skel_seen: Counter[str] = Counter()
    chosen: list[str] = []

    for expression in expressions:
        fam, skel = family(expression), skeleton(expression)
        if fam_seen[fam] >= max_per_family:
            continue
        if max_per_skeleton is not None and skel_seen[skel] >= max_per_skeleton:
            continue
        fam_seen[fam] += 1
        skel_seen[skel] += 1
        chosen.append(expression)
    return chosen


def is_novel(expression: str, existing: Iterable[str]) -> bool:
    """Whether an expression introduces a family not already present."""
    return family(expression) not in {family(e) for e in existing}
