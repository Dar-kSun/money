import pytest

from wqb.evaluate import evaluate, failed_platform_checks
from wqb.fields import Direction, FieldClass, FieldInfo
from wqb.generate import (
    DEFAULT_FIELDS,
    DRIFT_TEMPLATES,
    REVERSION_TEMPLATES,
    TEMPLATES,
    generate,
    grid_size,
    looks_like_naive_momentum,
    resolve_templates,
    templates_for,
    validate_expression,
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


def test_every_template_is_cross_sectionally_normalised():
    """The one universal invariant: positions must be market neutral."""
    for template in TEMPLATES:
        expr = template.render("close", 20)
        assert "rank(" in expr or "group_neutralize(" in expr, template.name
        assert validate_expression(expr) == [], template.name


def test_price_fields_only_draw_reversion_templates():
    """The -0.87 lesson: price momentum must never be generated."""
    for klass in (FieldClass.PRICE, FieldClass.VOLUME):
        for template in templates_for(klass):
            assert template.direction is Direction.REVERSION, (klass, template.name)


def test_fundamental_and_analyst_only_draw_drift_templates():
    """Drift is the documented effect on slow data (PEAD, revision momentum),
    so forcing reversion there would rule out the least-crowded region."""
    for klass in (FieldClass.FUNDAMENTAL, FieldClass.ANALYST):
        pool = templates_for(klass)
        assert pool, klass
        for template in pool:
            assert template.direction is Direction.DRIFT, (klass, template.name)


def test_detects_the_documented_failure_mode():
    assert looks_like_naive_momentum("close / ts_mean(close, 20) - 1")
    assert looks_like_naive_momentum("volume / ts_mean(volume, 5) - 1")


def test_same_algebra_is_allowed_on_slow_data():
    """The sign is a property of the data, not the formula: the expression that
    fails on price is a legitimate drift signal on analyst revisions."""
    assert not looks_like_naive_momentum(
        "anl_eps_revision / ts_mean(anl_eps_revision, 60) - 1"
    )
    assert not looks_like_naive_momentum("total_assets / ts_mean(total_assets, 250) - 1")


def test_does_not_flag_the_blueprint():
    assert not looks_like_naive_momentum("rank(-1 * (close / ts_mean(close, 20) - 1))")
    assert not looks_like_naive_momentum("-1 * (close / ts_mean(close, 20) - 1)")


def test_windows_match_field_update_frequency():
    """A 5-day window over quarterly accounting data samples noise."""
    fundamental = FieldInfo(id="total_assets", dataset="fundamental6")
    windows = {c.window for c in generate([fundamental])}
    assert min(windows) >= 60, windows
    price_windows = {c.window for c in generate(["close"])}
    assert 5 in price_windows


def test_grid_is_deduplicated_and_sized():
    assert grid_size(["close"], [20], REVERSION_TEMPLATES[:1]) == 1
    assert grid_size(["close", "close"], [20, 20], REVERSION_TEMPLATES[:1]) == 1
    # Price fields draw only the reversion half of the template set.
    assert grid_size(DEFAULT_FIELDS, [5, 10], TEMPLATES) == (
        len(DEFAULT_FIELDS) * 2 * len(REVERSION_TEMPLATES)
    )


def test_drift_templates_are_not_offered_to_price_fields():
    assert grid_size(["close"], [20], DRIFT_TEMPLATES) == 0


def test_limit_caps_generation():
    assert len(list(generate(limit=5))) == 5


def test_resolve_templates_rejects_unknown_name():
    assert resolve_templates(None) == TEMPLATES
    assert resolve_templates(["reversion_ma"])[0].name == "reversion_ma"
    with pytest.raises(ValueError, match="nonsense"):
        resolve_templates(["nonsense"])


def test_candidate_tags():
    candidate = next(generate(["close"], [20], REVERSION_TEMPLATES[:1]))
    assert candidate.tags == ["template", "reversion_ma", "close", "d20"]
    assert candidate.field_class is FieldClass.PRICE


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
