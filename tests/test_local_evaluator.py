import numpy as np
import pytest

from wqb.local.data import synthetic_panel
from wqb.local.evaluator import (
    ExpressionError,
    evaluate_expression,
    expression_fields,
)


@pytest.fixture(scope="module")
def panel():
    return synthetic_panel(60, 8, seed=3)


# ------------------------------------------------------------- evaluation
def test_bare_field_returns_the_panel_array(panel):
    assert np.allclose(evaluate_expression("close", panel), panel.get("close"))


def test_arithmetic_and_operators(panel):
    got = evaluate_expression("rank(-1 * (close / ts_mean(close, 20) - 1))", panel)
    assert got.shape == panel.shape
    finite = got[np.isfinite(got)]
    assert finite.min() >= 0.0 and finite.max() <= 1.0


def test_constant_broadcasts_to_panel_shape(panel):
    assert evaluate_expression("1", panel).shape == panel.shape


def test_group_operand_resolves(panel):
    got = evaluate_expression("group_neutralize(close, subindustry)", panel)
    assert got.shape == panel.shape


def test_division_by_zero_becomes_nan_not_inf(panel):
    got = evaluate_expression("close / 0", panel)
    assert not np.isfinite(got).any()


def test_comparison_and_if_else(panel):
    got = evaluate_expression("if_else(close > 0, 1, -1)", panel)
    assert set(np.unique(got[np.isfinite(got)])) <= {1.0, -1.0}


def test_nested_operators(panel):
    got = evaluate_expression(
        "rank(ts_mean(ts_delta(close, 1), 5) / (1 + ts_std_dev(returns, 20)))", panel
    )
    assert got.shape == panel.shape


# --------------------------------------------------------------- sandbox
def test_unknown_field_is_an_error(panel):
    with pytest.raises(ExpressionError, match="unknown field 'nope'"):
        evaluate_expression("rank(nope)", panel)


def test_unknown_operator_is_an_error(panel):
    with pytest.raises(ExpressionError, match="unknown operator 'ts_magic'"):
        evaluate_expression("ts_magic(close, 5)", panel)


def test_syntax_error_is_reported(panel):
    with pytest.raises(ExpressionError, match="syntax error"):
        evaluate_expression("rank(close", panel)


def test_empty_expression_is_an_error(panel):
    with pytest.raises(ExpressionError, match="empty"):
        evaluate_expression("   ", panel)


@pytest.mark.parametrize("hostile", [
    "__import__('os').system('echo pwned')",
    "close.__class__",
    "(1).__class__.__bases__",
    "[c for c in close]",
    "lambda: close",
    "open('/etc/passwd').read()",
    "globals()",
    "close[0]",
    "exec('import os')",
])
def test_sandbox_rejects_non_alpha_syntax(panel, hostile):
    """Expressions may come from an LLM or a config file, so the evaluator must
    not be a path to arbitrary execution. Nothing is eval'd as text: only the
    node types an alpha needs are walked."""
    with pytest.raises(ExpressionError):
        evaluate_expression(hostile, panel)


def test_attribute_access_is_rejected_before_lookup(panel):
    with pytest.raises(ExpressionError, match="not allowed"):
        evaluate_expression("close.mean", panel)


def test_absurdly_long_expression_is_rejected(panel):
    with pytest.raises(ExpressionError, match="limit"):
        evaluate_expression("close + " * 2000 + "close", panel)


def test_overly_complex_expression_is_rejected(panel):
    nested = "close"
    for _ in range(200):
        nested = f"ts_mean({nested}, 2)"
    with pytest.raises(ExpressionError, match="too complex"):
        evaluate_expression(nested, panel)


def test_string_constant_is_rejected(panel):
    with pytest.raises(ExpressionError, match="unsupported constant"):
        evaluate_expression("rank('close')", panel)


# ----------------------------------------------------------- introspection
def test_expression_fields_lists_data_dependencies():
    found = expression_fields("rank(-1 * ts_zscore(close, 20) / (1 + ts_std_dev(returns, 20)))")
    assert found == {"close", "returns"}


def test_expression_fields_excludes_operators_and_constants():
    assert expression_fields("ts_mean(close, 20)") == {"close"}


def test_expression_fields_includes_group_operands():
    assert expression_fields("group_neutralize(close, subindustry)") == {
        "close", "subindustry"
    }
