import json
from dataclasses import dataclass

import pytest

from wqb.fields import FieldInfo
from wqb.local.data import synthetic_panel
from wqb.propose import (
    MODEL,
    PROPOSAL_SCHEMA,
    ProposerUnavailable,
    build_prompt,
    propose,
)

FIELDS = [
    FieldInfo(id="close", coverage=1.0, user_count=50_000, alpha_count=900_000),
    FieldInfo(id="anl_eps_rev", dataset="analyst4", description="EPS revisions",
              coverage=0.8, user_count=90, alpha_count=200),
]


@dataclass
class FakeBlock:
    text: str
    type: str = "text"


class FakeResponse:
    def __init__(self, payload, stop_reason="end_turn"):
        body = payload if isinstance(payload, str) else json.dumps(payload)
        self.content = [FakeBlock(body)]
        self.stop_reason = stop_reason


class FakeClient:
    """Records the request and replays a canned response."""

    def __init__(self, payload, stop_reason="end_turn"):
        self.payload = payload
        self.stop_reason = stop_reason
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(self.payload, self.stop_reason)


def batch_of(*expressions, **extra):
    return {"proposals": [
        {"expression": e, "hypothesis": "because", "mechanism": "test",
         "fields": ["close"], **extra}
        for e in expressions
    ]}


# ------------------------------------------------------------ request shape
def test_request_uses_the_documented_api_shape():
    client = FakeClient(batch_of("rank(-1 * ts_zscore(close, 20))"))
    propose(FIELDS, count=3, client=client)

    call = client.calls[0]
    assert call["model"] == MODEL
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"]["effort"] == "high"
    assert call["output_config"]["format"] == {
        "type": "json_schema", "schema": PROPOSAL_SCHEMA
    }
    assert "max_tokens" in call


def test_stable_catalogue_is_cached():
    """Operators and fields repeat across a run; feedback does not."""
    client = FakeClient(batch_of("rank(close)"))
    propose(FIELDS, count=1, client=client, tried=[("rank(volume)", "low sharpe")])
    system = client.calls[0]["system"]
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    assert "rank(volume)" not in system[0]["text"], "feedback must not break the cache"
    assert "rank(volume)" in client.calls[0]["messages"][0]["content"]


def test_prompt_states_the_direction_rules_and_crowding():
    system, user = build_prompt(FIELDS, count=4)
    text = system[0]["text"]
    assert "-0.87" in text                      # the documented price failure
    assert "do NOT" in text or "do not invert" in text.lower()
    assert "crowding=12.7" in text              # close is heavily mined
    assert "class=analyst expected=drift" in text


def test_prompt_lists_only_evaluable_operators():
    from wqb.local.operators import OPERATORS
    text = build_prompt(FIELDS, count=1)[0][0]["text"]
    assert "ts_zscore" in text and "group_neutralize" in text
    for name in OPERATORS:
        assert name in text


def test_no_fields_is_an_error():
    with pytest.raises(ValueError, match="at least one field"):
        propose([], client=FakeClient(batch_of()))


# --------------------------------------------------------------- validation
def test_valid_proposal_is_accepted():
    client = FakeClient(batch_of("rank(-1 * ts_zscore(close, 20))"))
    result = propose(FIELDS, client=client)
    assert len(result.accepted) == 1
    assert result.accepted[0].hypothesis == "because"
    assert result.accepted[0].mechanism == "test"


def test_hallucinated_operator_is_rejected():
    """A model may invent an operator; it must not reach the API."""
    client = FakeClient(batch_of("rank(ts_sorcery(close, 20))"))
    result = propose(FIELDS, client=client, panel=synthetic_panel(60, 6, seed=0))
    assert result.accepted == []
    assert "does not evaluate" in result.rejected[0][1]


def test_unavailable_field_is_rejected():
    client = FakeClient(batch_of("rank(ts_zscore(made_up_field, 20))"))
    result = propose(FIELDS, client=client)
    assert result.accepted == []
    assert "unavailable field" in result.rejected[0][1]


def test_unnormalised_expression_is_rejected():
    """Without cross-sectional normalisation the book is not market neutral."""
    client = FakeClient(batch_of("ts_delta(close, 5)"))
    result = propose(FIELDS, client=client)
    assert result.accepted == []
    assert "normalisation" in result.rejected[0][1]


def test_uninverted_price_momentum_is_rejected():
    """Rule 4 in the prompt, enforced in code rather than trusted."""
    client = FakeClient({"proposals": [{
        "expression": "close / ts_mean(close, 20) - 1",
        "hypothesis": "momentum", "mechanism": "momentum", "fields": ["close"],
    }]})
    result = propose(FIELDS, client=client)
    assert result.accepted == []
    assert "momentum" in result.rejected[0][1]


def test_group_operands_are_not_treated_as_unknown_fields():
    client = FakeClient(batch_of(
        "rank(group_neutralize(ts_zscore(close, 20), subindustry))"
    ))
    assert len(propose(FIELDS, client=client).accepted) == 1


def test_duplicate_families_within_a_batch_are_dropped():
    client = FakeClient(batch_of(
        "rank(-1 * ts_zscore(close, 20))",
        "rank(-1 * ts_zscore(close, 60))",      # same family, different window
    ))
    result = propose(FIELDS, client=client)
    assert len(result.accepted) == 1
    assert "duplicate" in result.rejected[0][1]


def test_proposals_matching_existing_alphas_are_dropped():
    client = FakeClient(batch_of("rank(-1 * ts_zscore(close, 20))"))
    result = propose(FIELDS, client=client,
                     found=["rank(-1 * ts_zscore(close, 5))"])
    assert result.accepted == []
    assert "duplicate" in result.rejected[0][1]


def test_panel_evaluation_catches_fields_the_panel_lacks():
    panel = synthetic_panel(60, 6, seed=0)      # has no analyst field
    client = FakeClient(batch_of("rank(ts_delta(anl_eps_rev, 60))"))
    result = propose(FIELDS, client=client, panel=panel)
    assert result.accepted == []
    assert "does not evaluate" in result.rejected[0][1]


# ------------------------------------------------------------- robustness
def test_non_json_response_yields_nothing_rather_than_raising():
    result = propose(FIELDS, client=FakeClient("I cannot do that"))
    assert result.accepted == [] and result.rejected == []


def test_empty_proposal_list_is_handled():
    assert propose(FIELDS, client=FakeClient({"proposals": []})).accepted == []


def test_bare_list_response_is_accepted():
    client = FakeClient([{"expression": "rank(close)", "hypothesis": "h",
                          "mechanism": "m", "fields": ["close"]}])
    assert len(propose(FIELDS, client=client).accepted) == 1


def test_missing_expression_is_rejected():
    client = FakeClient({"proposals": [{"hypothesis": "h", "mechanism": "m"}]})
    result = propose(FIELDS, client=client)
    assert result.rejected == [("", "no expression")]


def test_refusal_is_surfaced_clearly():
    client = FakeClient(batch_of("rank(close)"), stop_reason="refusal")
    with pytest.raises(ProposerUnavailable, match="declined"):
        propose(FIELDS, client=client)


def test_batch_summary_reads_sensibly():
    client = FakeClient(batch_of("rank(-1 * ts_zscore(close, 20))", "ts_delta(close, 5)"))
    assert propose(FIELDS, client=client).summary() == "1 accepted, 1 rejected"
