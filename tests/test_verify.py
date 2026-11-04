"""The verifier must be trustworthy before it runs for real, so it is tested
against a mocked API that returns realistic payloads."""

import pytest
import responses

from wqb.verify import PROBE_EXPRESSION, compare_local, verify

BASE = "https://api.worldquantbrain.com"

IS_SUMMARY = {
    "sharpe": 1.81, "fitness": 1.32, "turnover": 0.21, "returns": 0.14,
    "drawdown": 0.05, "margin": 0.0012,
    "checks": [{"name": "LOW_SHARPE", "result": "PASS"}],
}


def register_success(crowding=True, is_summary=None):
    responses.add(responses.POST, f"{BASE}/authentication",
                  json={"user": {"id": "AB12345"}}, status=201)
    row = {"id": "close", "description": "close price", "coverage": 1.0}
    if crowding:
        row |= {"userCount": 50_000, "alphaCount": 900_000}
    responses.add(responses.GET, f"{BASE}/data-fields",
                  json={"count": 1, "results": [row]}, status=200)
    responses.add(responses.POST, f"{BASE}/simulations", status=201,
                  headers={"Location": f"{BASE}/simulations/SIM1"})
    responses.add(responses.GET, f"{BASE}/simulations/SIM1", status=200,
                  json={"status": "COMPLETE", "alpha": "ALPHA1"})
    responses.add(responses.GET, f"{BASE}/alphas/ALPHA1", status=200,
                  json={"is": is_summary if is_summary is not None else IS_SUMMARY})


@responses.activate
def test_all_checks_pass_against_a_conforming_api(client, no_sleep):
    register_success()
    report = verify(client=client)
    assert report.ok, report.render()
    assert report.metrics["sharpe"] == 1.81
    assert "All" in report.render()


@responses.activate
def test_missing_crowding_counts_are_reported_not_fatal(client, no_sleep):
    """fields.py ranking degrades to coverage order; everything else holds."""
    register_success(crowding=False)
    report = verify(client=client)
    failed = [c for c in report.checks if not c.ok]
    assert len(failed) == 1
    assert "crowding" in failed[0].name
    assert "coverage order" in failed[0].detail


@responses.activate
def test_unexpected_metric_names_are_flagged(client, no_sleep):
    """The likeliest contract mismatch: different in-sample summary keys."""
    register_success(is_summary={"sharpeRatio": 1.8, "fit": 1.2, "checks": []})
    report = verify(client=client)
    assert not report.ok
    names = [c.name for c in report.checks if not c.ok]
    assert any("metric names" in n for n in names)


@responses.activate
def test_biometric_requirement_stops_early_with_the_url(client):
    responses.add(responses.POST, f"{BASE}/authentication", status=401,
                  headers={"WWW-Authenticate": "persona"},
                  json={"location": "https://platform.worldquantbrain.com/bio/x"})
    report = verify(client=client)
    assert not report.ok
    assert "biometric" in report.checks[0].detail
    assert len(report.checks) == 1, "must not continue past a failed login"


@responses.activate
def test_bad_credentials_stop_early(client, no_sleep):
    responses.add(responses.POST, f"{BASE}/authentication", status=401)
    report = verify(client=client)
    assert not report.ok and len(report.checks) == 1


@responses.activate
def test_no_simulate_spends_no_simulation(client, no_sleep):
    register_success()
    verify(client=client, skip_simulation=True)
    assert not [c for c in responses.calls if c.request.url.endswith("/simulations")]


@responses.activate
def test_simulation_failure_is_reported(client, no_sleep):
    responses.add(responses.POST, f"{BASE}/authentication", json={}, status=201)
    responses.add(responses.GET, f"{BASE}/data-fields",
                  json={"count": 0, "results": []}, status=200)
    responses.add(responses.POST, f"{BASE}/simulations", status=400,
                  body="unknown operator")
    report = verify(client=client)
    assert not report.ok
    assert any("unknown operator" in c.detail for c in report.checks)


@responses.activate
def test_missing_credentials_are_reported_without_a_client(monkeypatch):
    monkeypatch.delenv("WQB_EMAIL", raising=False)
    monkeypatch.delenv("WQB_PASSWORD", raising=False)
    report = verify()
    assert not report.ok
    assert "credentials" in report.checks[0].name


def test_compare_local_tabulates_both_metric_sets():
    from wqb.local.data import synthetic_panel
    from wqb.local.engine import SimConfig

    panel = synthetic_panel(400, 20, seed=0)
    out = compare_local(PROBE_EXPRESSION, IS_SUMMARY, panel,
                        SimConfig(universe_size=20, warmup=60))
    assert "brain" in out and "local" in out
    assert "sharpe" in out and "1.8100" in out
    assert "RANKS" in out, "must say one expression cannot settle calibration"
