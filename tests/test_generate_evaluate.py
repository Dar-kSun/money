import pytest

from wqb.evaluate import evaluate, failed_platform_checks
from wqb.generate import (
    DEFAULT_FIELDS,
    TEMPLATES,
    generate,
    grid_size,
    looks_like_naive_momentum,
    resolve_templates,
)
from wqb.simulate import SimulationResult


def make_result(sharpe=2.0, fitness=1.5, status="COMPLETE", checks=None):
    return SimulationResult(
        expression="x", status=status, alpha_id="A1",
        metrics={"sharpe": sharpe, "fitness": fitness},
        checks=checks or [],
    )


# ----------------------------------------------------------------- generation
def test_blueprint_is_generated_first():
    first = next(generate())
    assert first.expression == "rank(-1 * (close / ts_mean(close, 5) - 1))"
    assert first.template == "reversion_ma"
    assert first.field == "close"
    # The exact blueprint from the spec is in the grid at the 20-day window.
    blueprint = "rank(-1 * (close / ts_mean(close, 20) - 1))"
    assert blueprint in [c.expression for c in generate()]


def test_every_template_is_mean_reverting():
    """No template may reproduce the momentum form the spec rules out."""
    for template in TEMPLATES:
        expr = template.render("close", 20)
        assert not looks_like_naive_momentum(expr), template.name
        assert "rank(" in expr, template.name


def test_detects_the_documented_failure_mode():
    assert looks_like_naive_momentum("close / ts_mean(close, 20) - 1")
    assert looks_like_naive_momentum("volume / ts_mean(volume, 5) - 1")


def test_does_not_flag_the_blueprint():
    assert not looks_like_naive_momentum("rank(-1 * (close / ts_mean(close, 20) - 1))")
    assert not looks_like_naive_momentum("-1 * (close / ts_mean(close, 20) - 1)")


def test_grid_is_deduplicated_and_sized():
    assert grid_size(["close"], [20], TEMPLATES[:1]) == 1
    assert grid_size(["close", "close"], [20, 20], TEMPLATES[:1]) == 1
    assert grid_size(DEFAULT_FIELDS, [5, 10], TEMPLATES) == len(DEFAULT_FIELDS) * 2 * len(TEMPLATES)


def test_limit_caps_generation():
    assert len(list(generate(limit=5))) == 5


def test_resolve_templates_rejects_unknown_name():
    assert resolve_templates(None) == TEMPLATES
    assert resolve_templates(["reversion_ma"])[0].name == "reversion_ma"
    with pytest.raises(ValueError, match="nonsense"):
        resolve_templates(["nonsense"])


def test_candidate_tags():
    candidate = next(generate(["close"], [20], TEMPLATES[:1]))
    assert candidate.tags == ["auto", "reversion_ma", "close", "d20"]


# ----------------------------------------------------------------- evaluation
def test_passes_when_above_both_thresholds():
    assert evaluate(make_result(sharpe=1.26, fitness=1.0)).passed


def test_sharpe_threshold_is_strictly_greater():
    """The directive says Sharpe > 1.25, so exactly 1.25 must fail."""
    verdict = evaluate(make_result(sharpe=1.25, fitness=1.5))
    assert not verdict.passed
    assert "sharpe 1.250 <= 1.25" in verdict.reason


def test_fitness_threshold_is_inclusive():
    """The directive says Fitness >= 1.0, so exactly 1.0 must pass."""
    assert evaluate(make_result(sharpe=2.0, fitness=1.0)).passed


def test_fails_just_below_fitness():
    verdict = evaluate(make_result(sharpe=2.0, fitness=0.999))
    assert not verdict.passed
    assert "fitness" in verdict.reason


def test_negative_sharpe_momentum_alpha_fails():
    verdict = evaluate(make_result(sharpe=-0.87, fitness=-0.4))
    assert not verdict.passed
    assert len(verdict.reasons) == 2


def test_incomplete_simulation_never_passes():
    result = SimulationResult(expression="x", status="ERROR", message="boom")
    assert not evaluate(result).passed


def test_missing_metrics_do_not_crash():
    verdict = evaluate(make_result(sharpe=None, fitness=None))
    assert not verdict.passed
    assert "no sharpe reported" in verdict.reason


def test_platform_check_failure_blocks_pass():
    checks = [
        {"name": "LOW_SHARPE", "result": "PASS"},
        {"name": "CONCENTRATED_WEIGHT", "result": "FAIL"},
    ]
    verdict = evaluate(make_result(checks=checks))
    assert not verdict.passed
    assert verdict.failed_checks == ("CONCENTRATED_WEIGHT",)
    assert failed_platform_checks(checks) == ("CONCENTRATED_WEIGHT",)


def test_platform_checks_can_be_waived():
    checks = [{"name": "MATCHES_PYRAMID", "result": "FAIL"}]
    assert evaluate(make_result(checks=checks), require_platform_checks=False).passed


def test_custom_thresholds():
    assert evaluate(make_result(sharpe=1.0, fitness=0.8), min_sharpe=0.5, min_fitness=0.5).passed
