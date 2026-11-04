import responses

from wqb.config import RunConfig
from wqb.generate import TEMPLATES, generate
from wqb.runner import clone_client, run_batch
from wqb.store import ResultStore

BASE = "https://api.worldquantbrain.com"


def register_simulation(expr_index, sharpe, fitness):
    """Register one full submit->poll->alpha cycle."""
    sim_id = f"SIM{expr_index}"
    alpha_id = f"ALPHA{expr_index}"
    responses.add(
        responses.POST, f"{BASE}/simulations", status=201,
        headers={"Location": f"{BASE}/simulations/{sim_id}"},
    )
    responses.add(
        responses.GET, f"{BASE}/simulations/{sim_id}", status=200,
        json={"status": "COMPLETE", "alpha": alpha_id},
    )
    responses.add(
        responses.GET, f"{BASE}/alphas/{alpha_id}", status=200,
        json={"is": {"sharpe": sharpe, "fitness": fitness, "turnover": 0.2, "checks": []}},
    )


@responses.activate
def test_run_batch_separates_winners_from_losers(client, no_sleep, tmp_path, capsys):
    responses.add(responses.POST, f"{BASE}/authentication", json={}, status=201)
    client.authenticate()
    # Two candidates: the first passes, the second is below the Sharpe bar.
    register_simulation(1, sharpe=1.90, fitness=1.40)
    register_simulation(2, sharpe=0.40, fitness=0.20)

    candidates = list(generate(["close"], [5, 10], TEMPLATES[:1]))
    assert len(candidates) == 2

    with ResultStore(tmp_path / "db.sqlite") as store:
        report = run_batch(
            client, candidates, store,
            RunConfig(concurrency=1, poll_interval=0),
        )
        assert report.simulated == 2
        assert report.passed == 1
        assert report.failed == 1
        assert len(report.winners) == 1
        assert store.counts() == {"total": 2, "passed": 1, "submitted": 0}
        # Only the winner reaches the success file.
        assert len(store.success_file.read_text().strip().splitlines()) == 1

    out = capsys.readouterr().out
    assert "[PASS]" in out


@responses.activate
def test_run_batch_skips_already_simulated(client, no_sleep, tmp_path):
    responses.add(responses.POST, f"{BASE}/authentication", json={}, status=201)
    client.authenticate()
    candidates = list(generate(["close"], [5], TEMPLATES[:1]))

    with ResultStore(tmp_path / "db.sqlite") as store:
        store.record(expression=candidates[0].expression, status="COMPLETE", metrics={})
        report = run_batch(client, candidates, store, RunConfig(concurrency=1))
        assert report.skipped == 1
        assert report.simulated == 0
        # No simulation was requested at all.
        assert not [c for c in responses.calls if c.request.url.endswith("/simulations")]


@responses.activate
def test_run_batch_records_api_errors_without_dying(client, no_sleep, tmp_path):
    responses.add(responses.POST, f"{BASE}/authentication", json={}, status=201)
    client.authenticate()
    responses.add(responses.POST, f"{BASE}/simulations", status=400, body="bad operator")

    candidates = list(generate(["close"], [5], TEMPLATES[:1]))
    with ResultStore(tmp_path / "db.sqlite") as store:
        report = run_batch(client, candidates, store, RunConfig(concurrency=1))
        assert report.errored == 1
        assert report.passed == 0
        assert store.top() == []          # nothing gradeable
        assert store.counts()["total"] == 1  # but the attempt is remembered


@responses.activate
def test_max_alphas_caps_the_run(client, no_sleep, tmp_path):
    responses.add(responses.POST, f"{BASE}/authentication", json={}, status=201)
    client.authenticate()
    register_simulation(1, sharpe=1.9, fitness=1.4)

    with ResultStore(tmp_path / "db.sqlite") as store:
        report = run_batch(
            client, generate(), store, RunConfig(concurrency=1, max_alphas=1)
        )
        assert report.simulated == 1


def test_clone_client_shares_cookies_without_reauthenticating(client):
    client.session.cookies.set("t", "jwt-value")
    client.authenticated = True
    sibling = clone_client(client)
    assert sibling is not client
    assert sibling.session is not client.session   # own connection pool per thread
    assert sibling.session.cookies.get("t") == "jwt-value"
    assert sibling.authenticated is True
    assert sibling.credentials == client.credentials
