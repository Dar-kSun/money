from wqb.fields import (
    Direction,
    FieldClass,
    FieldInfo,
    classify,
    group_by_class,
    rank_by_crowding,
    usable,
)


# ------------------------------------------------------------ classification
def test_price_and_volume_names_are_exact():
    for name in ("close", "open", "high", "low", "vwap", "returns", "cap"):
        assert classify(name) is FieldClass.PRICE, name
    assert classify("volume") is FieldClass.VOLUME


def test_dataset_wins_over_name():
    assert classify("x1", dataset="fundamental6") is FieldClass.FUNDAMENTAL
    assert classify("x2", dataset="analyst4") is FieldClass.ANALYST
    assert classify("x3", dataset="news12") is FieldClass.SENTIMENT
    assert classify("x4", dataset="option8") is FieldClass.OPTION
    assert classify("x5", dataset="institutions1") is FieldClass.OWNERSHIP


def test_keywords_classify_when_dataset_is_missing():
    assert classify("eps_estimate_revision") is FieldClass.ANALYST
    assert classify("total_liabilities") is FieldClass.FUNDAMENTAL
    assert classify("news_sentiment_score") is FieldClass.SENTIMENT
    assert classify("implied_vol_skew") is FieldClass.OPTION


def test_unknown_field_is_other():
    assert classify("zzz_unknown_42") is FieldClass.OTHER


def test_price_name_beats_a_misleading_description():
    """An exact price name must not be reclassified by description text."""
    assert classify("close", description="revenue and earnings data") is FieldClass.PRICE


# --------------------------------------------------------------- directions
def test_direction_priors_encode_the_economics():
    # The -0.87 lesson applies here and only here.
    assert FieldInfo(id="close").direction is Direction.REVERSION
    assert FieldInfo(id="volume").direction is Direction.REVERSION
    # Slow data drifts: PEAD and analyst-revision momentum.
    assert FieldInfo(id="x", dataset="fundamental6").direction is Direction.DRIFT
    assert FieldInfo(id="x", dataset="analyst4").direction is Direction.DRIFT
    # No strong prior elsewhere.
    assert FieldInfo(id="zzz").direction is Direction.BOTH


# ------------------------------------------------------------------ parsing
def test_from_api_reads_nested_dataset_and_counts():
    field = FieldInfo.from_api({
        "id": "anl4_rev", "description": "revisions",
        "dataset": {"id": "analyst4"}, "type": "MATRIX",
        "coverage": 0.83, "userCount": 120, "alphaCount": 450,
    })
    assert field.dataset == "analyst4"
    assert field.field_class is FieldClass.ANALYST
    assert field.user_count == 120 and field.alpha_count == 450


def test_from_api_tolerates_snake_case_and_flat_dataset():
    field = FieldInfo.from_api({
        "id": "f", "dataset": "fundamental6", "user_count": 5, "alpha_count": 9
    })
    assert field.dataset == "fundamental6"
    assert (field.user_count, field.alpha_count) == (5, 9)


def test_from_api_survives_missing_counts():
    """The count keys are unverified against the live API, so their absence
    must degrade gracefully rather than crash."""
    field = FieldInfo.from_api({"id": "f", "coverage": 0.9})
    assert field.user_count is None and field.alpha_count is None
    assert field.crowding is None


def test_from_api_ignores_garbage_numbers():
    field = FieldInfo.from_api({"id": "f", "coverage": "n/a", "alphaCount": "many"})
    assert field.coverage is None and field.alpha_count is None


# ----------------------------------------------------------------- crowding
def test_crowding_is_higher_for_heavily_mined_fields():
    crowded = FieldInfo(id="close", user_count=50_000, alpha_count=900_000)
    fresh = FieldInfo(id="anl_rev", dataset="analyst4", user_count=120, alpha_count=300)
    assert crowded.crowding > fresh.crowding


def test_crowding_weights_alpha_count_above_user_count():
    """alphaCount measures how mined a field is; userCount only who looked."""
    many_alphas = FieldInfo(id="a", user_count=10, alpha_count=10_000)
    many_users = FieldInfo(id="b", user_count=10_000, alpha_count=10)
    assert many_alphas.crowding > many_users.crowding


def test_rank_puts_least_crowded_first():
    fields = [
        FieldInfo(id="close", coverage=1.0, user_count=50_000, alpha_count=900_000),
        FieldInfo(id="mid", coverage=0.9, user_count=500, alpha_count=2_000),
        FieldInfo(id="fresh", coverage=0.8, user_count=10, alpha_count=20),
    ]
    assert [f.id for f in rank_by_crowding(fields)] == ["fresh", "mid", "close"]


def test_unmeasured_fields_sort_last():
    """An unknown count is not evidence of opportunity."""
    fields = [
        FieldInfo(id="unknown", coverage=0.9),
        FieldInfo(id="measured", coverage=0.9, user_count=10, alpha_count=20),
    ]
    assert [f.id for f in rank_by_crowding(fields)] == ["measured", "unknown"]


def test_low_coverage_fields_are_excluded():
    """A sparse field yields a concentrated alpha that fails platform checks,
    however uncrowded it is."""
    sparse = FieldInfo(id="sparse", coverage=0.05, user_count=1, alpha_count=1)
    dense = FieldInfo(id="dense", coverage=0.95, user_count=900, alpha_count=9_000)
    assert not usable(sparse)
    assert [f.id for f in rank_by_crowding([sparse, dense])] == ["dense"]


def test_group_field_type_is_not_usable_directly():
    assert not usable(FieldInfo(id="sector", type="GROUP", coverage=1.0))
    assert usable(FieldInfo(id="close", type="MATRIX", coverage=1.0))


def test_rank_respects_limit():
    fields = [FieldInfo(id=f"f{i}", coverage=0.9, alpha_count=i) for i in range(10)]
    assert len(rank_by_crowding(fields, limit=3)) == 3


def test_group_by_class():
    fields = [FieldInfo(id="close"), FieldInfo(id="x", dataset="analyst4")]
    grouped = group_by_class(fields)
    assert grouped[FieldClass.PRICE][0].id == "close"
    assert grouped[FieldClass.ANALYST][0].id == "x"
