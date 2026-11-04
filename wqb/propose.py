"""Hypothesis-driven alpha proposals from Claude.

Where this earns its place
--------------------------
Mutation search is good at refining a shape and hopeless at inventing one: it
can turn a 20-day window into 60, but it will not decide that analyst revisions
should be scaled by dispersion because crowded names react differently. That
step needs to read what a field *means*, and a language model is the only cheap
way to do it.

What it is deliberately not used for is predicting whether an alpha will score.
A model's guess at a Sharpe is worthless; the local simulator measures it for
free. So the division of labour is: the model proposes mechanisms grounded in
field semantics, and :mod:`wqb.local` decides which ones survive.

Everything returned is treated as untrusted input. Expressions are validated
and, when a panel is supplied, evaluated before being accepted, so a
hallucinated operator or a malformed formula is dropped with a reason rather
than reaching the API.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field as dc_field
from typing import Any, Sequence

from .diversity import family, is_novel
from .fields import Direction, FieldInfo
from .generate import validate_expression
from .local.operators import OPERATORS

log = logging.getLogger(__name__)

MODEL = "claude-opus-5-5"
MAX_TOKENS = 16_000

PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A FastExpr alpha using only the listed "
                                       "operators and fields.",
                    },
                    "hypothesis": {
                        "type": "string",
                        "description": "The economic reason this should predict "
                                       "returns, in one or two sentences.",
                    },
                    "mechanism": {
                        "type": "string",
                        "description": "Short label for the effect, e.g. "
                                       "'analyst revision drift'.",
                    },
                    "fields": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["expression", "hypothesis", "mechanism", "fields"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["proposals"],
    "additionalProperties": False,
}


@dataclass
class Proposal:
    expression: str
    hypothesis: str
    mechanism: str
    fields: list[str] = dc_field(default_factory=list)

    @property
    def family(self) -> str:
        return family(self.expression)


@dataclass
class ProposalBatch:
    accepted: list[Proposal] = dc_field(default_factory=list)
    rejected: list[tuple[str, str]] = dc_field(default_factory=list)

    def summary(self) -> str:
        return f"{len(self.accepted)} accepted, {len(self.rejected)} rejected"


class ProposerUnavailable(RuntimeError):
    """The Anthropic SDK is not installed, or no credential is configured."""


SYSTEM_RULES = """\
You design quantitative alpha expressions for the WorldQuant BRAIN platform.

Target environment: USA equities, 1-day delay, TOP3000 universe. An alpha must
reach Sharpe > 1.25 and Fitness >= 1.0, with turnover between 1% and 70%, and
must not correlate with alphas already in the platform's pool.

Hard rules:
1. Use ONLY the operators listed below. An operator that is not listed cannot
   be evaluated and the proposal is discarded.
2. Use ONLY the data fields listed below, by their exact id.
3. Every expression must be cross-sectionally normalised -- wrap it in rank(),
   zscore(), or group_neutralize(..., subindustry) -- so the book is market
   neutral. An un-normalised expression is discarded.
4. For PRICE and VOLUME fields, only mean-reverting formulations work in this
   environment: close / ts_mean(close, 20) - 1 was measured at a Sharpe of
   about -0.87 because it buys spikes in a mean-reverting market. Invert it.
5. For FUNDAMENTAL and ANALYST fields the opposite holds: these drift
   (post-earnings-announcement drift, analyst-revision momentum), so do NOT
   invert them.
6. Match the lookback to how often the data updates. A 5-day window over
   quarterly accounting data samples noise.

What makes a proposal valuable: a distinct economic mechanism, not a variant of
a shape already tried. Changing a window or a neutralisation barely moves
correlation with existing alphas; a different data source or a different
economic reason does. Prefer the least-crowded fields offered.

For each proposal give the expression and the economic hypothesis behind it.
Do not predict its Sharpe -- it will be measured.\
"""


def _operator_catalogue() -> str:
    return ", ".join(sorted(OPERATORS))


def _field_catalogue(fields: Sequence[FieldInfo]) -> str:
    lines = []
    for info in fields:
        bits = [f"{info.id}", f"class={info.field_class.value}"]
        if info.direction is not Direction.BOTH:
            bits.append(f"expected={info.direction.value}")
        if info.crowding is not None:
            bits.append(f"crowding={info.crowding:.1f}")
        if info.coverage is not None:
            bits.append(f"coverage={info.coverage:.2f}")
        if info.description:
            bits.append(f"- {info.description[:90]}")
        lines.append("  " + " ".join(bits))
    return "\n".join(lines)


def build_prompt(
    fields: Sequence[FieldInfo],
    *,
    count: int,
    tried: Sequence[tuple[str, str]] = (),
    found: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], str]:
    """Return (system blocks, user message).

    The operator and field catalogues are stable across calls in a run and go
    in a cached system block; what has been tried changes every round and goes
    in the user turn, after the cache breakpoint.
    """
    stable = (
        f"{SYSTEM_RULES}\n\n"
        f"Available operators:\n  {_operator_catalogue()}\n\n"
        f"Available fields (lower crowding means less mined by other users):\n"
        f"{_field_catalogue(fields)}"
    )
    system = [{
        "type": "text",
        "text": stable,
        "cache_control": {"type": "ephemeral"},
    }]

    parts = [f"Propose {count} alpha expressions."]
    if found:
        parts.append(
            "Already working, so propose mechanisms uncorrelated with these:\n"
            + "\n".join(f"  {e}" for e in found[:25])
        )
    if tried:
        parts.append(
            "Already tried and rejected, with the reason. Do not repeat these "
            "shapes:\n" + "\n".join(f"  {e}  -> {why}" for e, why in tried[:40])
        )
    return system, "\n\n".join(parts)


def propose(
    fields: Sequence[FieldInfo],
    *,
    count: int = 10,
    tried: Sequence[tuple[str, str]] = (),
    found: Sequence[str] = (),
    client: Any | None = None,
    model: str = MODEL,
    effort: str = "high",
    panel: Any | None = None,
) -> ProposalBatch:
    """Ask Claude for alpha hypotheses, then validate what comes back.

    ``client`` is injectable so this is testable without network access or a
    credential. ``panel`` is optional: when given, each expression is also
    evaluated against it, which catches references to fields the panel does
    not hold before any quota is spent.
    """
    if not fields:
        raise ValueError("propose() needs at least one field to work with")

    client = client or _default_client()
    system, user = build_prompt(fields, count=count, tried=tried, found=found)

    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}],
        thinking={"type": "adaptive"},
        output_config={
            "effort": effort,
            "format": {"type": "json_schema", "schema": PROPOSAL_SCHEMA},
        },
    )

    if getattr(response, "stop_reason", None) == "refusal":
        raise ProposerUnavailable(
            "the model declined this request; check the prompt for anything "
            "that could read as disallowed"
        )

    return _validate(
        _extract(response), allowed_fields={f.id for f in fields},
        found=found, panel=panel,
    )


def _default_client() -> Any:
    try:
        import anthropic
    except ImportError as exc:                 # pragma: no cover - env specific
        raise ProposerUnavailable(
            "the proposer needs the Anthropic SDK: pip install anthropic. "
            "The rest of the pipeline works without it."
        ) from exc
    try:
        return anthropic.Anthropic()
    except Exception as exc:                   # pragma: no cover - env specific
        raise ProposerUnavailable(
            f"could not construct an Anthropic client ({exc}). Set "
            f"ANTHROPIC_API_KEY, or sign in with `ant auth login`."
        ) from exc


def _extract(response: Any) -> list[dict[str, Any]]:
    """Pull the proposal list out of a structured response."""
    texts = [
        block.text for block in getattr(response, "content", []) or []
        if getattr(block, "type", None) == "text" and getattr(block, "text", None)
    ]
    if not texts:
        return []
    try:
        payload = json.loads("".join(texts))
    except json.JSONDecodeError:
        log.warning("proposer returned content that was not valid JSON")
        return []
    rows = payload.get("proposals") if isinstance(payload, dict) else payload
    return [r for r in (rows or []) if isinstance(r, dict)]


def _validate(
    rows: Sequence[dict[str, Any]],
    *,
    allowed_fields: set[str],
    found: Sequence[str],
    panel: Any | None,
) -> ProposalBatch:
    batch = ProposalBatch()
    seen: set[str] = {family(e) for e in found}

    for row in rows:
        expression = str(row.get("expression") or "").strip()
        if not expression:
            batch.rejected.append(("", "no expression"))
            continue

        problems = validate_expression(expression)
        if problems:
            batch.rejected.append((expression, "; ".join(problems)))
            continue

        unknown = _unknown_names(expression, allowed_fields)
        if unknown:
            batch.rejected.append(
                (expression, f"uses unavailable field(s): {', '.join(sorted(unknown))}")
            )
            continue

        if panel is not None:
            try:
                from .local.evaluator import ExpressionError, evaluate_expression
                evaluate_expression(expression, panel)
            except ExpressionError as exc:
                batch.rejected.append((expression, f"does not evaluate: {exc}"))
                continue

        fam = family(expression)
        if fam in seen:
            batch.rejected.append((expression, "duplicate of an existing family"))
            continue
        seen.add(fam)

        batch.accepted.append(Proposal(
            expression=expression,
            hypothesis=str(row.get("hypothesis") or "").strip(),
            mechanism=str(row.get("mechanism") or "").strip(),
            fields=[str(f) for f in (row.get("fields") or [])],
        ))
    return batch


def _unknown_names(expression: str, allowed: set[str]) -> set[str]:
    from .local.evaluator import expression_fields
    from .diversity import GROUP_OPERANDS
    try:
        referenced = expression_fields(expression)
    except Exception:
        return set()
    return {
        name for name in referenced
        if name not in allowed
        and name not in GROUP_OPERANDS
        and name not in OPERATORS
    }
