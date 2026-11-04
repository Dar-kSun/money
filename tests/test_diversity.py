from wqb.diversity import (
    family,
    family_counts,
    is_novel,
    report,
    same_family,
    select_diverse,
    skeleton,
)

MA_REVERSION_5 = "rank(-1 * (close / ts_mean(close, 5) - 1))"
MA_REVERSION_20 = "rank(-1 * (close / ts_mean(close, 20) - 1))"
MA_DRIFT_20 = "rank(close / ts_mean(close, 20) - 1)"
ZSCORE_CLOSE = "rank(-1 * ts_zscore(close, 20))"
ZSCORE_VOLUME = "rank(-1 * ts_zscore(volume, 20))"
ANALYST_DRIFT = "rank(ts_delta(anl_eps_revision, 60))"


# ------------------------------------------------------------------- family
def test_window_variants_share_a_family():
    """The core case: the grid produces these as separate rows, but changing a
    lookback barely moves correlation, so they are one idea."""
    assert same_family(MA_REVERSION_5, MA_REVERSION_20)


def test_whitespace_does_not_affect_the_family():
    assert same_family(ZSCORE_CLOSE, "rank(-1*ts_zscore(close,20))")


def test_different_fields_are_different_families():
    assert not same_family(ZSCORE_CLOSE, ZSCORE_VOLUME)


def test_inversion_changes_the_family():
    """Reversion and drift on the same field are opposite bets, not duplicates."""
    assert not same_family(MA_REVERSION_20, MA_DRIFT_20)


def test_different_operators_are_different_families():
    assert not same_family(MA_REVERSION_20, ZSCORE_CLOSE)


def test_float_and_scientific_constants_collapse():
    assert same_family("rank(ts_zscore(close, 20) / 1e-9)", "rank(ts_zscore(close, 5) / 0.5)")


# ----------------------------------------------------------------- skeleton
def test_skeleton_abstracts_the_field_to_its_class():
    assert skeleton(ZSCORE_CLOSE) == "rank(-N*ts_zscore(PRICE,N))"


def test_same_mechanism_on_different_data_shares_no_family_but_differs_in_class():
    """Price and volume are different classes, so neither family nor skeleton
    collapses -- the mechanism is portable but the data is not the same."""
    assert not same_family(ZSCORE_CLOSE, ZSCORE_VOLUME)
    assert skeleton(ZSCORE_CLOSE) != skeleton(ZSCORE_VOLUME)


def test_skeleton_collapses_two_fields_of_one_class():
    """close and vwap are both price, so the same shape over them is one
    mechanism even though it is two families."""
    a = "rank(-1 * ts_zscore(close, 20))"
    b = "rank(-1 * ts_zscore(vwap, 20))"
    assert skeleton(a) == skeleton(b)
    assert not same_family(a, b)


def test_grouping_operands_are_structural_not_data():
    expr = "rank(group_neutralize(ts_zscore(close, 20), subindustry))"
    assert "subindustry" in skeleton(expr)


# ------------------------------------------------------------------- report
def test_report_exposes_a_book_that_is_fewer_ideas_than_it_looks():
    book = [MA_REVERSION_5, MA_REVERSION_20, ZSCORE_CLOSE, ZSCORE_VOLUME]
    result = report(book)
    assert result.total == 4
    assert result.families == 3        # the two window variants collapse
    assert result.family_ratio == 0.75
    assert "3 distinct famil" in result.describe()


def test_report_handles_an_empty_book():
    result = report([])
    assert result.total == 0 and result.family_ratio == 0.0


def test_family_counts():
    counts = family_counts([MA_REVERSION_5, MA_REVERSION_20, ZSCORE_CLOSE])
    assert sorted(counts.values()) == [1, 2]


# ----------------------------------------------------------------- selection
def test_select_diverse_keeps_one_per_family():
    chosen = select_diverse([MA_REVERSION_5, MA_REVERSION_20, ZSCORE_CLOSE])
    assert chosen == [MA_REVERSION_5, ZSCORE_CLOSE]


def test_select_diverse_preserves_order_so_the_best_survives():
    """Callers pass best-first (e.g. by Sharpe), so the kept member of a
    family must be the one that came first."""
    chosen = select_diverse([MA_REVERSION_20, MA_REVERSION_5])
    assert chosen == [MA_REVERSION_20]


def test_select_diverse_can_allow_several_per_family():
    chosen = select_diverse(
        [MA_REVERSION_5, MA_REVERSION_20, ZSCORE_CLOSE], max_per_family=2
    )
    assert len(chosen) == 3


def test_select_diverse_can_cap_per_mechanism():
    """Capping skeletons stops a book being one mechanism across many fields."""
    close_z = "rank(-1 * ts_zscore(close, 20))"
    vwap_z = "rank(-1 * ts_zscore(vwap, 20))"
    chosen = select_diverse([close_z, vwap_z], max_per_skeleton=1)
    assert chosen == [close_z]


def test_is_novel():
    assert is_novel(ANALYST_DRIFT, [MA_REVERSION_5, ZSCORE_CLOSE])
    assert not is_novel(MA_REVERSION_20, [MA_REVERSION_5])
