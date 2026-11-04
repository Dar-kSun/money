"""The local commands must run with no credentials and no network."""

import pytest

from wqb.cli import build_parser, main


def run(argv):
    return main(argv)


def test_parser_exposes_the_local_commands():
    parser = build_parser()
    for command in ("fetch", "screen", "search", "propose", "calibrate"):
        assert parser.parse_args([command] + (["prices"] if command == "fetch" else
                                              ["x"] if command == "calibrate" else []))


def test_screen_runs_on_synthetic_data(capsys):
    code = run(["screen", "--synthetic", "--universe", "40", "--warmup", "80",
                "--fields", "close", "--windows", "20", "--top", "3"])
    assert code == 0
    out = capsys.readouterr().out
    assert "sharpe" in out and "evaluated" in out


def test_screen_warns_about_fields_absent_from_the_panel(capsys):
    run(["screen", "--synthetic", "--universe", "40", "--warmup", "80",
         "--fields", "close", "not_a_field", "--windows", "20"])
    assert "absent from the panel" in capsys.readouterr().err


def test_screen_fails_clearly_when_no_field_is_usable():
    with pytest.raises(SystemExit, match="none of the requested fields"):
        run(["screen", "--synthetic", "--universe", "40", "--warmup", "80",
             "--fields", "nope"])


def test_screen_points_at_fetch_when_data_is_missing(tmp_path):
    with pytest.raises(SystemExit, match="fetch"):
        run(["screen", "--data", str(tmp_path / "absent")])


def test_search_runs_on_synthetic_data(capsys):
    code = run(["search", "--synthetic", "--universe", "40", "--warmup", "80",
                "--population", "8", "--generations", "2", "--book", "2"])
    assert code == 0
    assert "evaluated locally (no quota spent)" in capsys.readouterr().out


def test_calibrate_reports_both_returns_bases(capsys):
    code = run(["calibrate", "--synthetic", "--universe", "40", "--warmup", "80",
                "rank(-1 * ts_zscore(close, 20))"])
    assert code == 0
    out = capsys.readouterr().out
    assert "returns_basis=half_book" in out and "returns_basis=book" in out


def test_calibrate_flags_a_problematic_expression(capsys):
    run(["calibrate", "--synthetic", "--universe", "40", "--warmup", "80",
         "ts_delta(close, 5)"])
    assert "normalisation" in capsys.readouterr().err


def test_propose_reports_cleanly_without_a_credential(capsys, monkeypatch):
    """Without the SDK or a key the command must explain itself, not traceback."""
    monkeypatch.setattr(
        "wqb.propose._default_client",
        lambda: (_ for _ in ()).throw(
            __import__("wqb.propose", fromlist=["ProposerUnavailable"])
            .ProposerUnavailable("no credential")
        ),
    )
    code = run(["propose", "--synthetic", "--universe", "40", "--warmup", "80"])
    assert code == 2
    assert "Proposer unavailable" in capsys.readouterr().err


def test_fetch_defaults_the_destination_per_kind(monkeypatch):
    seen = {}

    def fake_bulk(dest, **kwargs):
        seen["dest"] = str(dest)
        return 3

    monkeypatch.setattr("wqb.local.fetch.fetch_stooq_bulk", fake_bulk)
    assert run(["fetch", "prices"]) == 0
    assert seen["dest"] == "data/prices"


def test_fetch_reports_a_blocked_host_without_tracebacking(monkeypatch, capsys):
    from wqb.local.fetch import FetchError

    def fake_bulk(dest, **kwargs):
        raise FetchError("could not reach stooq: blocked by network policy")

    monkeypatch.setattr("wqb.local.fetch.fetch_stooq_bulk", fake_bulk)
    assert run(["fetch", "prices", "--dest", "x"]) == 1
    assert "Fetch failed" in capsys.readouterr().err
