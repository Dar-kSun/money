import numpy as np
import pytest

from wqb.local.data import planted_panel, synthetic_panel
from wqb.local.engine import SimConfig, correlation
from wqb.search import EvolveConfig, evolve, mutation_space

SIM = SimConfig(universe_size=40, warmup=60, decay=1,
                neutralization="MARKET", delay=1)
PRICE_SEEDS = [
    "rank(-1 * (close / ts_mean(close, 20) - 1))",
    "rank(ts_delta(volume, 5))",
]


def test_search_discovers_a_planted_signal_unprompted():
    """The end-to-end property that matters.

    The panel contains a `signal` field that genuinely predicts returns, but
    the search is seeded only with price expressions. It has to reach the
    field on its own, through field mutation, and have the local simulator
    tell it that it worked.
    """
    panel = planted_panel(700, 40, exploitable_at_delay=1, strength=0.5, seed=0)
    space = mutation_space(["close", "volume", "signal", "returns"])

    book = evolve(panel, PRICE_SEEDS, space, SIM,
                  EvolveConfig(population=30, generations=8, elite=6,
                               max_book=6, seed=1))

    assert book, "search returned nothing"
    assert any("signal" in found.expression for found in book)
    best = max(book, key=lambda f: f.metrics.sharpe)
    assert "signal" in best.expression
    assert best.metrics.sharpe > 5


def test_nothing_is_accepted_from_an_unpredictable_panel():
    """The counterpart: with no signal to find, the search must come back
    empty rather than fitting noise hard enough to clear the thresholds."""
    panel = synthetic_panel(700, 40, autocorr=0.0, market_vol=0.0, seed=3)
    space = mutation_space(["close", "volume", "returns", "vwap"])

    book = evolve(panel, PRICE_SEEDS, space, SIM,
                  EvolveConfig(population=20, generations=5, elite=5,
                               max_book=5, seed=2, min_sharpe=1.25,
                               min_fitness=1.0))
    assert book == []


# ------------------------------------------------------------- constraints
@pytest.fixture(scope="module")
def book():
    panel = planted_panel(700, 40, exploitable_at_delay=1, strength=0.5, seed=0)
    space = mutation_space(["close", "volume", "signal", "returns"])
    return evolve(panel, PRICE_SEEDS, space, SIM,
                  EvolveConfig(population=30, generations=8, elite=6,
                               max_book=8, seed=1))


def test_accepted_alphas_respect_turnover_bounds(book):
    """Brain requires turnover inside roughly 1%-70%; an alpha outside that is
    refused however good its Sharpe looks locally."""
    for found in book:
        assert 0.01 <= found.metrics.turnover <= 0.70, found.expression


def test_accepted_alphas_clear_the_thresholds(book):
    for found in book:
        assert found.metrics.sharpe > 1.25
        assert found.metrics.fitness >= 1.0


def test_book_contains_no_two_highly_correlated_alphas(book):
    """The whole point of the novelty discount: a book of correlated alphas
    cannot be submitted, so it must not be what the search returns."""
    for i, a in enumerate(book):
        for b in book[i + 1:]:
            assert abs(correlation(a.metrics, b.metrics)) <= 0.7 + 1e-9, (
                a.expression, b.expression
            )


def test_book_contains_no_duplicate_families(book):
    from wqb.diversity import family
    families = [family(f.expression) for f in book]
    assert len(families) == len(set(families))


def test_book_is_ordered_best_first(book):
    scores = [f.score for f in book]
    assert scores == sorted(scores, reverse=True)


def test_correlation_penalty_demotes_a_near_duplicate():
    """A second copy of an idea must score below the first."""
    panel = planted_panel(500, 30, exploitable_at_delay=1, strength=0.5, seed=1)
    space = mutation_space(["close", "signal"])
    book = evolve(panel, ["rank(ts_delta(signal, 5))"], space, SIM,
                  EvolveConfig(population=16, generations=4, elite=4,
                               max_book=4, seed=0))
    for found in book[1:]:
        assert found.correlation_to_book <= 0.7


# ------------------------------------------------------------- robustness
def test_evolve_requires_a_seed():
    panel = synthetic_panel(200, 10, seed=0)
    with pytest.raises(ValueError, match="at least one seed"):
        evolve(panel, [], mutation_space(["close"]), SIM, EvolveConfig())


def test_invalid_seeds_do_not_crash_the_search():
    """Seeds can come from a config file or a model, so a broken one must be
    scored as unusable rather than aborting the run."""
    panel = planted_panel(400, 30, exploitable_at_delay=1, seed=0)
    space = mutation_space(["close", "signal"])
    book = evolve(panel, ["rank(close", "ts_bogus(close, 5)", "rank(signal)"],
                  space, SIM,
                  EvolveConfig(population=12, generations=3, elite=4,
                               max_book=3, seed=0))
    assert isinstance(book, list)


def test_progress_callback_sees_every_candidate():
    panel = planted_panel(300, 20, exploitable_at_delay=1, seed=0)
    space = mutation_space(["close", "signal"])
    seen = []
    evolve(panel, ["rank(signal)"], space, SIM,
           EvolveConfig(population=8, generations=2, elite=3, max_book=2, seed=0),
           progress=lambda gen, book, scored: seen.append(scored.expression))
    assert len(seen) >= 8


def test_search_stops_once_the_book_is_full():
    panel = planted_panel(500, 30, exploitable_at_delay=1, strength=0.6, seed=0)
    space = mutation_space(["close", "signal", "volume"])
    book = evolve(panel, ["rank(ts_delta(signal, 5))"], space, SIM,
                  EvolveConfig(population=20, generations=30, elite=5,
                               max_book=2, seed=0))
    assert len(book) <= 2


def test_results_are_reproducible_for_a_given_seed():
    panel = planted_panel(400, 25, exploitable_at_delay=1, seed=0)
    space = mutation_space(["close", "signal", "volume"])
    cfg = EvolveConfig(population=14, generations=3, elite=4, max_book=3, seed=42)
    first = evolve(panel, PRICE_SEEDS, space, SIM, cfg)
    second = evolve(panel, PRICE_SEEDS, space, SIM, cfg)
    assert [f.expression for f in first] == [f.expression for f in second]
