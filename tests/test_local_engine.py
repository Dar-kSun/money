import numpy as np
import pytest

from wqb.local.data import planted_panel, synthetic_panel
from wqb.local.engine import (
    TURNOVER_FLOOR,
    Metrics,
    SimConfig,
    build_positions,
    correlation,
    simulate,
)

BLUEPRINT = "rank(-1 * (close / ts_mean(close, 20) - 1))"
REVERSAL = "rank(-1 * ts_delta(close, 1))"
MOMENTUM = "rank(ts_delta(close, 1))"


def mean_sharpe(expression, *, autocorr, delay, seeds=6, T=1200, N=120):
    # MARKET neutralization makes the book dollar neutral. With "NONE" a rank
    # alpha is long-only and inherits the drift of geometric prices, which
    # swamps the signal -- see test_rank_without_neutralization_is_long_only.
    cfg = SimConfig(
        universe_size=N, warmup=60, delay=delay, decay=1, neutralization="MARKET"
    )
    out = []
    for seed in range(seeds):
        panel = synthetic_panel(T, N, autocorr=autocorr, market_vol=0.0, seed=seed)
        out.append(simulate(expression, panel, cfg).sharpe)
    return float(np.mean(out))


# ===================================================================== signs
# With returns following AR(1), r_t = phi*r_{t-1} + eps, the sign a delayed
# alpha can capture is a sharp prediction:
#   delay=0: weights at t are built from r_t and earn r_{t+1} = phi*r_t  -> phi
#   delay=1: weights at t are built from r_{t-1} and earn r_{t+1}        -> phi^2
# So for phi < 0 the profitable side *flips* between the two delays. Matching
# that is strong evidence the delay is applied in the right direction and that
# no future information leaks in.
def test_reversion_wins_at_delay_zero_in_a_mean_reverting_market():
    assert mean_sharpe(REVERSAL, autocorr=-0.5, delay=0) > 3
    assert mean_sharpe(MOMENTUM, autocorr=-0.5, delay=0) < -3


def test_momentum_wins_at_delay_one_because_the_sign_squares():
    """phi^2 > 0, so at delay=1 the predictable component is positive even
    though the market mean-reverts."""
    assert mean_sharpe(MOMENTUM, autocorr=-0.5, delay=1) > 3
    assert mean_sharpe(REVERSAL, autocorr=-0.5, delay=1) < -3


def test_momentum_wins_at_delay_zero_in_a_trending_market():
    assert mean_sharpe(MOMENTUM, autocorr=+0.5, delay=0) > 3
    assert mean_sharpe(REVERSAL, autocorr=+0.5, delay=0) < -3


def test_nothing_is_predictable_in_a_random_walk():
    """The key look-ahead guard: with no autocorrelation there is nothing to
    find, so any strategy scoring well here would mean the engine is reading
    the future."""
    for expression in (REVERSAL, MOMENTUM, BLUEPRINT):
        for delay in (0, 1):
            score = mean_sharpe(expression, autocorr=0.0, delay=delay, seeds=8)
            assert abs(score) < 1.0, (expression, delay, score)


# ============================================================ planted signal
@pytest.mark.parametrize("planted_for", [0, 1, 2])
def test_planted_signal_pays_off_only_at_its_own_delay(planted_for):
    """The decisive look-ahead test.

    A signal planted to be exploitable at one delay must earn a large Sharpe
    at exactly that delay and nothing at any other. Profiting at a *smaller*
    delay would mean the engine reads information too early; profiting at a
    larger one would mean the signal is stale yet still paying.
    """
    panel = planted_panel(600, 50, exploitable_at_delay=planted_for, seed=0)
    scores = {}
    for delay in (0, 1, 2):
        cfg = SimConfig(universe_size=50, warmup=30, decay=1,
                        neutralization="MARKET", delay=delay)
        scores[delay] = simulate("rank(signal)", panel, cfg).sharpe

    assert scores[planted_for] > 20, scores
    for delay, score in scores.items():
        if delay != planted_for:
            assert abs(score) < 2.0, (delay, scores)


# ============================================================== construction
@pytest.fixture(scope="module")
def panel():
    return synthetic_panel(400, 40, autocorr=-0.1, seed=11)


def test_gross_exposure_equals_book_size(panel):
    cfg = SimConfig(universe_size=40, warmup=60, truncation=0.0)
    weights = build_positions(
        np.asarray(simulate_signal(BLUEPRINT, panel)), panel, cfg
    )
    gross = np.abs(weights).sum(axis=1)[cfg.warmup:]
    assert np.allclose(gross[gross > 0], cfg.book_size, rtol=1e-6)


def simulate_signal(expression, panel):
    from wqb.local.evaluator import evaluate_expression
    return evaluate_expression(expression, panel)


def test_positions_are_market_neutral_after_neutralization(panel):
    cfg = SimConfig(universe_size=40, warmup=60, neutralization="MARKET",
                    truncation=0.0)
    weights = build_positions(simulate_signal(BLUEPRINT, panel), panel, cfg)
    net = weights.sum(axis=1)[cfg.warmup:]
    assert np.allclose(net, 0.0, atol=cfg.book_size * 1e-6)


def test_subindustry_neutralization_zeroes_each_group(panel):
    cfg = SimConfig(universe_size=40, warmup=60,
                    neutralization="SUBINDUSTRY", truncation=0.0)
    weights = build_positions(simulate_signal(BLUEPRINT, panel), panel, cfg)
    codes = panel.groups["subindustry"][-1]
    for code in np.unique(codes):
        sel = codes == code
        assert abs(weights[-2, sel].sum()) < cfg.book_size * 1e-6


def test_truncation_caps_individual_weights(panel):
    cfg = SimConfig(universe_size=40, warmup=60, truncation=0.05)
    weights = build_positions(simulate_signal(BLUEPRINT, panel), panel, cfg)
    cap = cfg.truncation * cfg.book_size
    # One rescale pass after clipping can leave a little headroom above the cap.
    assert np.abs(weights).max() <= cap * 1.5


def test_decay_lowers_turnover(panel):
    """This is why the context document says decay 4-5 lifts fitness: fitness
    divides by turnover."""
    base = SimConfig(universe_size=40, warmup=60, decay=1)
    decayed = SimConfig(universe_size=40, warmup=60, decay=5)
    assert simulate(BLUEPRINT, panel, decayed).turnover < \
           simulate(BLUEPRINT, panel, base).turnover


def test_out_of_universe_names_never_get_weight():
    panel = synthetic_panel(200, 20, seed=5)
    mask = np.ones(panel.shape, dtype=bool)
    mask[:, 10:] = False
    panel.restrict_universe(mask)
    weights = build_positions(simulate_signal(BLUEPRINT, panel),
                              panel, SimConfig(warmup=30))
    assert np.allclose(weights[:, 10:], 0.0)


# =================================================================== metrics
def test_fitness_matches_the_published_formula(panel):
    m = simulate(BLUEPRINT, panel, SimConfig(universe_size=40, warmup=60))
    expected = m.sharpe * np.sqrt(abs(m.returns) / max(m.turnover, TURNOVER_FLOOR))
    assert np.isclose(m.fitness, expected)


def test_turnover_floor_applies_to_a_very_slow_alpha():
    """max(turnover, 0.125) means a near-zero-turnover alpha does not get
    unbounded fitness."""
    m = Metrics(sharpe=2.0, fitness=0.0, turnover=0.0, returns=0.5,
                drawdown=0.0, margin=0.0, long_count=0, short_count=0, days=10)
    expected = 2.0 * np.sqrt(0.5 / TURNOVER_FLOOR)
    assert np.isclose(expected, 4.0)


def test_sharpe_is_annualised(panel):
    m = simulate(BLUEPRINT, panel, SimConfig(universe_size=40, warmup=60))
    daily = m.pnl
    expected = np.sqrt(252) * daily.mean() / daily.std(ddof=1)
    assert np.isclose(m.sharpe, expected)


def test_drawdown_is_non_negative(panel):
    assert simulate(BLUEPRINT, panel, SimConfig(universe_size=40, warmup=60)).drawdown >= 0


def test_returns_basis_changes_the_reported_return(panel):
    """The one metric definition that is not publicly pinned down, so it is a
    config switch and calibration decides it."""
    half = simulate(BLUEPRINT, panel,
                    SimConfig(universe_size=40, warmup=60, returns_basis="half_book"))
    full = simulate(BLUEPRINT, panel,
                    SimConfig(universe_size=40, warmup=60, returns_basis="book"))
    assert np.isclose(half.returns, 2 * full.returns)
    assert np.isclose(half.sharpe, full.sharpe)   # sharpe is unaffected


def test_metrics_dict_uses_brain_key_names(panel):
    keys = simulate(BLUEPRINT, panel, SimConfig(universe_size=40, warmup=60)).as_dict()
    assert {"sharpe", "fitness", "turnover", "returns", "longCount"} <= set(keys)


def test_too_short_a_window_returns_zeroed_metrics():
    panel = synthetic_panel(30, 10, seed=1)
    m = simulate(BLUEPRINT, panel, SimConfig(warmup=200))
    assert m.sharpe == 0.0 and m.fitness == 0.0


# =============================================================== correlation
def test_identical_alphas_correlate_perfectly(panel):
    cfg = SimConfig(universe_size=40, warmup=60)
    a = simulate(BLUEPRINT, panel, cfg)
    assert np.isclose(correlation(a, a), 1.0)


def test_opposite_alphas_correlate_negatively(panel):
    cfg = SimConfig(universe_size=40, warmup=60)
    a = simulate(BLUEPRINT, panel, cfg)
    b = simulate("rank(close / ts_mean(close, 20) - 1)", panel, cfg)
    assert correlation(a, b) < -0.9


def test_window_variants_correlate_highly(panel):
    """The reason family-level dedup exists: changing the lookback barely
    changes the pnl stream, which is what the submission check measures."""
    cfg = SimConfig(universe_size=40, warmup=60)
    a = simulate("rank(-1 * ts_zscore(close, 20))", panel, cfg)
    b = simulate("rank(-1 * ts_zscore(close, 25))", panel, cfg)
    assert correlation(a, b) > 0.8


def test_correlation_of_a_flat_series_is_zero():
    assert correlation(np.zeros(10), np.zeros(10)) == 0.0


def test_rank_without_neutralization_is_long_only():
    """A named trap, not a bug.

    rank() yields values in [0, 1], so without neutralization every weight is
    positive and the book is fully long. Because geometric prices drift up by
    sigma^2/2 a day, that book earns a positive Sharpe on data with no
    predictability whatsoever -- and so does its exact opposite. Anyone
    screening with neutralization="NONE" would read that drift as skill.
    """
    panel = synthetic_panel(400, 60, autocorr=0.0, market_vol=0.0, seed=0)
    cfg = SimConfig(universe_size=60, warmup=60, decay=1, neutralization="NONE")
    weights = build_positions(simulate_signal(REVERSAL, panel), panel, cfg)

    row = weights[200]
    assert (row > 0).sum() > 0 and (row < 0).sum() == 0
    assert np.isclose(row.sum() / np.abs(row).sum(), 1.0)

    # Both an alpha and its opposite score positively on unpredictable data.
    up = simulate(REVERSAL, panel, cfg).sharpe
    down = simulate(MOMENTUM, panel, cfg).sharpe
    assert up > 0.5 and down > 0.5

    # Dollar-neutral weighting removes it.
    neutral = SimConfig(universe_size=60, warmup=60, decay=1, neutralization="MARKET")
    row_n = build_positions(simulate_signal(REVERSAL, panel), panel, neutral)[200]
    assert abs(row_n.sum() / np.abs(row_n).sum()) < 1e-9
