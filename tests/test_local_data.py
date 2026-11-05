import numpy as np
import pytest

from wqb.local.data import add_universe, load_csv_panel, planted_panel, synthetic_panel


def test_synthetic_panel_shape_and_fields():
    p = synthetic_panel(120, 15, seed=2)
    assert p.shape == (120, 15)
    for name in ("close", "open", "high", "low", "vwap", "volume", "returns", "cap"):
        assert name in p.fields, name
    for name in ("market", "sector", "industry", "subindustry"):
        assert name in p.groups, name


def test_synthetic_autocorrelation_is_actually_induced():
    """The sign tests depend on this, so verify the generator does what it says."""
    for target in (-0.5, 0.0, 0.5):
        p = synthetic_panel(3000, 30, autocorr=target, market_vol=0.0, seed=4)
        r = p.get("returns")[1:]
        measured = np.mean([
            np.corrcoef(r[:-1, i], r[1:, i])[0, 1] for i in range(r.shape[1])
        ])
        assert abs(measured - target) < 0.08, (target, measured)


def test_synthetic_prices_are_positive():
    p = synthetic_panel(200, 10, seed=1)
    assert np.all(p.get("close") > 0)


def test_dates_are_business_days_in_order():
    p = synthetic_panel(50, 5)
    assert np.all(np.diff(p.dates.astype("datetime64[D]")).astype(int) > 0)
    weekdays = ((p.dates.astype("datetime64[D]").astype(int) - 4) % 7)
    assert np.all(weekdays < 5)


def test_planted_panel_has_a_signal_field():
    p = planted_panel(200, 10, seed=0)
    assert "signal" in p.fields


def test_forward_returns_align_with_close():
    p = synthetic_panel(30, 4, seed=0)
    close = p.get("close")
    fwd = p.forward_returns()
    assert np.allclose(fwd[5], close[6] / close[5] - 1)
    assert np.all(np.isnan(fwd[-1])), "the last day's forward return is unknowable"


def test_add_universe_limits_membership_per_day():
    p = synthetic_panel(60, 20, seed=0)
    add_universe(p, size=5, by="cap")
    assert np.all(p.universe.sum(axis=1) == 5)


# ------------------------------------------------------------- csv loading
def write_csv(path, dates, closes, volume=1000.0):
    path.write_text(
        "Date,Open,High,Low,Close,Volume\n"
        + "".join(
            f"{d},{c},{c * 1.01},{c * 0.99},{c},{volume}\n"
            for d, c in zip(dates, closes)
        )
    )


def test_load_csv_panel_round_trip(tmp_path):
    dates = [f"2024-01-{d:02d}" for d in range(1, 21)]
    write_csv(tmp_path / "AAA.csv", dates, [10.0 + i for i in range(20)])
    write_csv(tmp_path / "BBB.csv", dates, [50.0 - i for i in range(20)])

    panel = load_csv_panel(tmp_path, min_history=10)
    assert panel.instruments == ["AAA", "BBB"]
    assert panel.shape == (20, 2)
    assert np.isclose(panel.get("close")[0, 0], 10.0)
    assert np.isclose(panel.get("close")[-1, 1], 31.0)


def test_load_csv_panel_unions_dates_and_forward_fills(tmp_path):
    write_csv(tmp_path / "AAA.csv", ["2024-01-01", "2024-01-02", "2024-01-03"], [1.0, 2.0, 3.0])
    write_csv(tmp_path / "BBB.csv", ["2024-01-01", "2024-01-03"], [9.0, 11.0])

    panel = load_csv_panel(tmp_path, min_history=2)
    close = panel.get("close")
    assert panel.shape == (3, 2)
    # BBB has no bar on the 2nd; carrying the price forward avoids inventing
    # a round trip that never happened.
    assert np.isclose(close[1, 1], 9.0)


def test_load_csv_panel_skips_short_histories(tmp_path):
    write_csv(tmp_path / "LONG.csv", [f"2024-01-{d:02d}" for d in range(1, 16)],
              [float(i) for i in range(1, 16)])
    write_csv(tmp_path / "SHORT.csv", ["2024-01-01", "2024-01-02"], [1.0, 2.0])

    panel = load_csv_panel(tmp_path, min_history=10)
    assert panel.instruments == ["LONG"]


def test_load_csv_panel_tolerates_lowercase_headers(tmp_path):
    (tmp_path / "AAA.csv").write_text(
        "date,open,high,low,close,volume\n"
        + "".join(f"2024-01-{d:02d},1,1,1,{d},100\n" for d in range(1, 13))
    )
    panel = load_csv_panel(tmp_path, min_history=10)
    assert panel.instruments == ["AAA"]


def test_load_csv_panel_skips_unparseable_rows(tmp_path):
    (tmp_path / "AAA.csv").write_text(
        "Date,Open,High,Low,Close,Volume\n"
        + "".join(f"2024-01-{d:02d},1,1,1,{d},100\n" for d in range(1, 13))
        + "2024-01-13,,,,,\n"
    )
    panel = load_csv_panel(tmp_path, min_history=10)
    assert panel.shape[0] == 12


def test_load_csv_panel_errors_on_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="no files matching"):
        load_csv_panel(tmp_path)


def test_load_csv_panel_errors_when_nothing_has_enough_history(tmp_path):
    write_csv(tmp_path / "AAA.csv", ["2024-01-01"], [1.0])
    with pytest.raises(ValueError, match="at least"):
        load_csv_panel(tmp_path, min_history=300)


# ------------------------------------------------------- fundamental merge
def test_load_fundamentals_aligns_and_forward_fills(tmp_path):
    """A quarterly figure stays current until the next filing, so it must be
    carried forward across trading days rather than left as a single spike."""
    import numpy as np
    from wqb.local.data import load_fundamentals

    panel = synthetic_panel(40, 2, seed=0)
    panel.instruments[:] = ["AAA", "BBB"]
    first = str(panel.dates.astype("datetime64[D]")[5])
    (tmp_path / "AAA.csv").write_text(f"Date,Assets\n{first},100\n")

    load_fundamentals(panel, tmp_path, lag_days=0)
    assets = panel.get("assets")
    assert np.isnan(assets[4, 0])          # nothing known before the filing
    assert assets[5, 0] == 100.0
    assert assets[20, 0] == 100.0          # carried forward
    assert np.all(np.isnan(assets[:, 1]))  # BBB filed nothing


def test_load_fundamentals_lags_the_filing_date(tmp_path):
    import numpy as np
    from wqb.local.data import load_fundamentals

    panel = synthetic_panel(40, 1, seed=0)
    panel.instruments[:] = ["AAA"]
    day = str(panel.dates.astype("datetime64[D]")[5])
    (tmp_path / "AAA.csv").write_text(f"Date,Assets\n{day},100\n")

    load_fundamentals(panel, tmp_path, lag_days=3)
    assert np.isnan(panel.get("assets")[5, 0])
    assert panel.get("assets")[8, 0] == 100.0


def test_load_fundamentals_ignores_unknown_symbols(tmp_path):
    from wqb.local.data import load_fundamentals

    panel = synthetic_panel(30, 1, seed=0)
    panel.instruments[:] = ["AAA"]
    (tmp_path / "ZZZ.csv").write_text("Date,Assets\n2018-01-05,100\n")
    load_fundamentals(panel, tmp_path)
    assert "assets" not in panel.fields


def test_load_fundamentals_errors_on_empty_directory(tmp_path):
    import pytest
    from wqb.local.data import load_fundamentals
    panel = synthetic_panel(10, 1, seed=0)
    with pytest.raises(FileNotFoundError):
        load_fundamentals(panel, tmp_path)
