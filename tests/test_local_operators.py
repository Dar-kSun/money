import numpy as np
import pytest

from wqb.local.operators import (
    Context,
    op_decay_linear,
    op_group_neutralize,
    op_rank,
    op_scale,
    op_ts_corr,
    op_ts_delay,
    op_ts_delta,
    op_ts_max,
    op_ts_mean,
    op_ts_min,
    op_ts_rank,
    op_ts_std_dev,
    op_ts_sum,
    op_ts_zscore,
    op_winsorize,
    op_zscore,
)

T, N = 10, 4


@pytest.fixture
def ctx():
    return Context(
        universe=np.ones((T, N), dtype=bool),
        groups={"subindustry": np.zeros((T, N), dtype=np.int64)},
    )


@pytest.fixture
def x():
    rng = np.random.default_rng(7)
    return rng.normal(size=(T, N))


# ------------------------------------------------------------ time series
def test_ts_mean_matches_numpy(ctx, x):
    got = op_ts_mean(ctx, x, 3)
    assert np.allclose(got[-1], x[-3:].mean(axis=0))
    assert np.allclose(got[2], x[0:3].mean(axis=0))


def test_ts_mean_uses_available_history_early(ctx, x):
    """min_periods is 1, so early rows compute from a partial window rather
    than returning NaN -- sparse fundamental series stay usable."""
    assert np.allclose(op_ts_mean(ctx, x, 5)[0], x[0])


def test_ts_sum_matches_numpy(ctx, x):
    assert np.allclose(op_ts_sum(ctx, x, 4)[-1], x[-4:].sum(axis=0))


def test_ts_std_dev_matches_numpy_ddof1(ctx, x):
    assert np.allclose(op_ts_std_dev(ctx, x, 5)[-1], x[-5:].std(axis=0, ddof=1))


def test_ts_std_dev_of_constant_is_zero_not_nan(ctx):
    out = op_ts_std_dev(ctx, np.ones((T, N)), 4)
    assert np.allclose(out[-1], 0.0)


def test_ts_delay_shifts_down(ctx, x):
    assert np.allclose(op_ts_delay(ctx, x, 2)[5], x[3])
    assert np.all(np.isnan(op_ts_delay(ctx, x, 2)[0]))


def test_ts_delay_zero_is_identity(ctx, x):
    """decay_linear's leading term is a zero lag, so this must be allowed."""
    assert np.allclose(op_ts_delay(ctx, x, 0), x)


def test_ts_delta_is_difference(ctx, x):
    assert np.allclose(op_ts_delta(ctx, x, 3)[-1], x[-1] - x[-4])


def test_ts_min_max(ctx, x):
    assert np.allclose(op_ts_min(ctx, x, 4)[-1], x[-4:].min(axis=0))
    assert np.allclose(op_ts_max(ctx, x, 4)[-1], x[-4:].max(axis=0))


def test_ts_zscore_standardises_against_own_history(ctx, x):
    expected = (x[-1] - x[-5:].mean(axis=0)) / x[-5:].std(axis=0, ddof=1)
    assert np.allclose(op_ts_zscore(ctx, x, 5)[-1], expected)


def test_ts_rank_is_one_for_a_rising_series(ctx):
    rising = np.arange(T, dtype=float)[:, None] * np.ones((1, N))
    assert np.allclose(op_ts_rank(ctx, rising, 5)[-1], 1.0)


def test_ts_rank_is_zero_for_a_falling_series(ctx):
    falling = -np.arange(T, dtype=float)[:, None] * np.ones((1, N))
    assert np.allclose(op_ts_rank(ctx, falling, 5)[-1], 0.0)


def test_ts_corr_of_identical_series_is_one(ctx, x):
    assert np.allclose(op_ts_corr(ctx, x, x, 6)[-1], 1.0)


def test_ts_corr_of_negated_series_is_minus_one(ctx, x):
    assert np.allclose(op_ts_corr(ctx, x, -x, 6)[-1], -1.0)


def test_decay_linear_of_a_constant_is_that_constant(ctx):
    assert np.allclose(op_decay_linear(ctx, np.full((T, N), 3.0), 5)[-1], 3.0)


def test_decay_linear_weights_recent_days_more(ctx):
    """Weights are d, d-1, ... 1, so a jump shows up heavily weighted."""
    series = np.zeros((T, N))
    series[-1] = 1.0
    out = op_decay_linear(ctx, series, 4)[-1]
    expected = 4.0 / (4 + 3 + 2 + 1)
    assert np.allclose(out, expected)


def test_window_must_be_positive(ctx, x):
    with pytest.raises(ValueError, match="window must be >= 1"):
        op_ts_mean(ctx, x, 0)


# -------------------------------------------------------- cross-sectional
def _row_ctx(n: int = 4) -> Context:
    return Context(universe=np.ones((1, n), dtype=bool), groups={})


def test_rank_scales_to_unit_interval():
    out = op_rank(_row_ctx(), np.array([[3.0, 1.0, 2.0, 4.0]]))
    assert np.allclose(out[0], [2 / 3, 0.0, 1 / 3, 1.0])


def test_rank_ignores_nan_and_keeps_it_nan():
    out = op_rank(_row_ctx(), np.array([[3.0, np.nan, 1.0, 2.0]]))
    assert np.isnan(out[0, 1])
    assert np.allclose(out[0, [2, 3, 0]], [0.0, 0.5, 1.0])


def test_rank_excludes_out_of_universe_names():
    """An alpha must not take positions outside the universe."""
    ctx = Context(universe=np.array([[True, True, False]]), groups={})
    out = op_rank(ctx, np.array([[1.0, 2.0, 99.0]]))
    assert np.isnan(out[0, 2])
    assert np.allclose(out[0, :2], [0.0, 1.0])


def test_zscore_has_zero_mean(ctx, x):
    out = op_zscore(ctx, x)
    assert np.allclose(np.nanmean(out, axis=1), 0.0, atol=1e-12)


def test_scale_normalises_gross_exposure(ctx, x):
    out = op_scale(ctx, x, 2.0)
    assert np.allclose(np.nansum(np.abs(out), axis=1), 2.0)


def test_winsorize_clips_an_outlier():
    row = np.array([[1.0, 1.0, 1.0, 100.0]])
    out = op_winsorize(_row_ctx(), row, std=1.0)
    assert out[0, 3] < 100.0


def test_group_neutralize_zeroes_each_group_mean():
    ctx = Context(
        universe=np.ones((1, 4), dtype=bool),
        groups={"subindustry": np.array([[0, 0, 1, 1]], dtype=np.int64)},
    )
    out = op_group_neutralize(ctx, np.array([[1.0, 3.0, 10.0, 20.0]]),
                              ctx.groups["subindustry"])
    assert np.allclose(out[0], [-1.0, 1.0, -5.0, 5.0])
    assert np.isclose(out[0, :2].sum(), 0.0)
    assert np.isclose(out[0, 2:].sum(), 0.0)


def test_group_neutralize_ignores_unknown_group_code():
    ctx = Context(
        universe=np.ones((1, 3), dtype=bool),
        groups={"subindustry": np.array([[0, 0, -1]], dtype=np.int64)},
    )
    out = op_group_neutralize(ctx, np.array([[1.0, 3.0, 99.0]]), ctx.groups["subindustry"])
    assert np.isnan(out[0, 2])


def test_shape_mismatch_is_reported_clearly(ctx):
    """A cross-sectional operator and its universe must come from one panel."""
    with pytest.raises(ValueError, match="same panel"):
        op_rank(ctx, np.array([[1.0, 2.0, 3.0, 4.0]]))
