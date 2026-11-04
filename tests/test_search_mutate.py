import ast
import random

import pytest

from wqb.local.data import synthetic_panel
from wqb.local.evaluator import ExpressionError, evaluate_expression
from wqb.search.mutate import (
    SERIES_ARITY,
    crossover,
    mutate,
    mutation_space,
    simplify,
)

BLUEPRINT = "rank(-1 * (close / ts_mean(close, 20) - 1))"
GROUPED = "rank(group_neutralize(ts_zscore(close, 20), subindustry))"
PARENTS = [BLUEPRINT, "zscore(ts_delta(volume, 5))", GROUPED]


@pytest.fixture(scope="module")
def panel():
    return synthetic_panel(120, 8, seed=1)


@pytest.fixture
def space():
    return mutation_space(["close", "volume", "returns", "vwap"])


# ------------------------------------------------------------------- mutate
def test_mutation_changes_the_expression(space):
    rng = random.Random(0)
    assert mutate(BLUEPRINT, space, rng) != BLUEPRINT


def test_mutation_explores_several_distinct_forms(space):
    rng = random.Random(0)
    variants = {mutate(BLUEPRINT, space, rng) for _ in range(40)}
    assert len(variants) >= 8


def test_every_mutant_stays_evaluable(panel, space):
    """Mutation operates on the syntax tree, so offspring are well formed by
    construction rather than by luck."""
    rng = random.Random(7)
    for _ in range(150):
        child = mutate(random.Random(rng.random()).choice(PARENTS), space, rng)
        evaluate_expression(child, panel)       # must not raise


def test_every_crossover_stays_evaluable(panel, space):
    rng = random.Random(11)
    for _ in range(150):
        child = crossover(rng.choice(PARENTS), rng.choice(PARENTS), rng)
        evaluate_expression(child, panel)       # must not raise


def test_mutation_never_replaces_an_operator_name_with_an_expression(space):
    """Regression: indices from ast.walk (breadth first) were applied during a
    depth-first transform, so edits landed on the wrong node and produced
    things like zscore(x)(x)."""
    rng = random.Random(3)
    for _ in range(120):
        child = mutate(BLUEPRINT, space, rng)
        tree = ast.parse(child, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                assert isinstance(node.func, ast.Name), child


def test_crossover_never_grafts_into_a_window_argument(space):
    """A window must stay a scalar; ts_mean(close, <array>) cannot evaluate."""
    rng = random.Random(5)
    for _ in range(150):
        child = crossover(rng.choice(PARENTS), rng.choice(PARENTS), rng)
        for node in ast.walk(ast.parse(child, mode="eval")):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            limit = SERIES_ARITY.get(node.func.id)
            if limit is None:
                continue
            for extra in node.args[limit:]:
                assert isinstance(extra, (ast.Constant, ast.Name)), child


def test_unparseable_input_is_returned_unchanged(space):
    assert mutate("rank(close", space, random.Random(0)) == "rank(close"
    assert crossover("rank(close", "close", random.Random(0)) == "rank(close"


def test_window_mutation_respects_field_update_frequency():
    """Quarterly data should not be offered a 2-day window."""
    space = mutation_space(["total_assets"])
    windows = space.windows_for("rank(ts_mean(total_assets, 250))")
    assert min(windows) >= 20


def test_fields_in_excludes_operators_and_groups(space):
    assert space.fields_in(GROUPED) == {"close"}


def test_single_field_space_still_mutates(panel):
    """With nothing to swap to, other mutation kinds must still apply."""
    space = mutation_space(["close"])
    rng = random.Random(0)
    assert mutate(BLUEPRINT, space, rng) != BLUEPRINT


# ----------------------------------------------------------------- simplify
def test_simplify_collapses_idempotent_nesting():
    assert simplify("rank(rank(close))") == "rank(close)"
    assert simplify("sign(sign(rank(rank(close))))") == "sign(rank(close))"


def test_simplify_leaves_distinct_operators_alone():
    assert simplify("rank(zscore(close))") == "rank(zscore(close))"


def test_simplify_is_safe_on_bad_input():
    assert simplify("rank(close") == "rank(close"


def test_flip_is_an_involution(space):
    """Repeated sign flips must not accumulate into '--1 * x'."""
    rng = random.Random(0)
    seen = {mutate("close", space, rng) for _ in range(60)}
    assert not any("--" in s for s in seen)
