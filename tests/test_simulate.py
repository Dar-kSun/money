import pytest
import responses

from wqb.client import BrainError
from wqb.config import SimSettings, simulation_payload
from wqb.simulate import SimulationTimeout, run_simulation, start_simulation

BASE = "https://api.worldquantbrain.com"
EXPR = "rank(-1 * (close / ts_mean(close, 20) - 1))"


def test_payload_matches_phase3_spec():
    payload = simulation_payload(EXPR)
    assert payload["type"] == "REGULAR"
    assert payload["regular"] == EXPR
    s = payload["settings"]
    assert s["instrumentType"] == "EQUITY"
    assert s["region"] == "USA"
    assert s["universe"] == "TOP3000"
    assert s["delay"] == 1
    assert s["decay"] == 5
    assert s["neutralization"] == "SUBINDUSTRY"
    assert s["truncation"] == 0.05


def test_payload_rejects_empty_expression():
    with pytest.raises(ValueError):
        simulation_payload("   ")


def test_settings_are_overridable():
    payload = simulation_payload(EXPR, SimSettings(decay=4, universe="TOP1000"))
    assert payload["settings"]["decay"] == 4
    assert payload["settings"]["universe"] == "TOP1000"


@responses.activate
def test_start_simulation_returns_location(client):
    responses.add(responses.POST, f"{BASE}/authentication", json={}, status=201)
    responses.add(
        responses.POST, f"{BASE}/simulations", status=201,
        headers={"Location": f"{BASE}/simulations/SIM1"},
    )
    assert start_simulation(client, EXPR) == f"{BASE}/simulations/SIM1"


@responses.activate
def test_start_simulation_falls_back_to_body_id(client):
    responses.add(responses.POST, f"{BASE}/authentication", json={}, status=201)
    responses.add(responses.POST, f"{BASE}/simulations", json={"id": "SIM9"}, status=201)
    assert start_simulation(client, EXPR).endswith("/simulations/SIM9")


@responses.activate
def test_start_simulation_raises_on_rejection(client, no_sleep):
    responses.add(responses.POST, f"{BASE}/authentication", json={}, status=201)
    responses.add(responses.POST, f"{BASE}/simulations", status=400, body="bad expression")
    with pytest.raises(BrainError, match="bad expression"):
        start_simulation(client, EXPR)


@responses.activate
def test_run_simulation_polls_then_grades(client, no_sleep):
    responses.add(responses.POST, f"{BASE}/authentication", json={}, status=201)
    responses.add(
        responses.POST, f"{BASE}/simulations", status=201,
        headers={"Location": f"{BASE}/simulations/SIM1"},
    )
    # First poll: still running, Retry-After tells us how long to wait.
    responses.add(
        responses.GET, f"{BASE}/simulations/SIM1", status=200,
        headers={"Retry-After": "3"}, body="",
    )
    responses.add(
        responses.GET, f"{BASE}/simulations/SIM1", status=200,
        json={"status": "COMPLETE", "alpha": "ALPHA1"},
    )
    responses.add(
        responses.GET, f"{BASE}/alphas/ALPHA1", status=200,
        json={"is": {
            "sharpe": 1.81, "fitness": 1.32, "turnover": 0.21, "returns": 0.14,
            "drawdown": 0.05, "margin": 0.0012,
            "checks": [{"name": "LOW_SHARPE", "result": "PASS"}],
        }},
    )

    result = run_simulation(client, EXPR)
    assert result.ok
    assert result.alpha_id == "ALPHA1"
    assert result.sharpe == 1.81 and result.fitness == 1.32
    assert result.checks[0]["name"] == "LOW_SHARPE"
    assert 3.0 in no_sleep


@responses.activate
def test_run_simulation_reports_failure(client, no_sleep):
    responses.add(responses.POST, f"{BASE}/authentication", json={}, status=201)
    responses.add(
        responses.POST, f"{BASE}/simulations", status=201,
        headers={"Location": f"{BASE}/simulations/SIM2"},
    )
    responses.add(
        responses.GET, f"{BASE}/simulations/SIM2", status=200,
        json={"status": "ERROR", "message": "unknown operator foo"},
    )
    result = run_simulation(client, EXPR)
    assert not result.ok
    assert "unknown operator" in result.message


@responses.activate
def test_poll_times_out(client, no_sleep):
    responses.add(responses.POST, f"{BASE}/authentication", json={}, status=201)
    responses.add(
        responses.POST, f"{BASE}/simulations", status=201,
        headers={"Location": f"{BASE}/simulations/SIM3"},
    )
    responses.add(
        responses.GET, f"{BASE}/simulations/SIM3", status=200,
        headers={"Retry-After": "1"}, body="",
    )
    with pytest.raises(SimulationTimeout):
        run_simulation(client, EXPR, timeout=-1)
