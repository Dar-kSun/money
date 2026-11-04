import pytest
import responses

from wqb.client import AuthError, BiometricRequired, BrainClient, BrainError
from wqb.config import ConfigError, Credentials

BASE = "https://api.worldquantbrain.com"


@responses.activate
def test_authenticate_accepts_201(client):
    responses.add(
        responses.POST, f"{BASE}/authentication",
        json={"user": {"id": "AB12345"}}, status=201,
    )
    client.authenticate()
    assert client.authenticated is True
    assert client.user_id == "AB12345"
    # Credentials must travel as HTTP Basic auth, per Phase 1.
    assert responses.calls[0].request.headers["Authorization"].startswith("Basic ")


@responses.activate
def test_authenticate_rejects_bad_password(client):
    responses.add(responses.POST, f"{BASE}/authentication", status=401)
    with pytest.raises(AuthError):
        client.authenticate()
    assert client.authenticated is False


@responses.activate
def test_authenticate_detects_biometric_requirement(client):
    responses.add(
        responses.POST, f"{BASE}/authentication", status=401,
        headers={"WWW-Authenticate": "persona"},
        json={"location": "https://platform.worldquantbrain.com/biometrics/xyz"},
    )
    with pytest.raises(BiometricRequired, match="biometrics/xyz"):
        client.authenticate()


def test_credentials_require_env(monkeypatch):
    monkeypatch.delenv("WQB_EMAIL", raising=False)
    monkeypatch.delenv("WQB_PASSWORD", raising=False)
    with pytest.raises(ConfigError, match="WQB_EMAIL"):
        Credentials.from_env()


def test_credentials_repr_hides_password(creds):
    assert "secret" not in repr(creds)


@responses.activate
def test_request_reauthenticates_once_on_401(client, no_sleep):
    responses.add(responses.POST, f"{BASE}/authentication", json={}, status=201)
    responses.add(responses.GET, f"{BASE}/alphas/A1", status=401)
    responses.add(responses.GET, f"{BASE}/alphas/A1", json={"id": "A1"}, status=200)

    assert client.alpha("A1") == {"id": "A1"}
    # one initial auth + the renewal
    auth_calls = [c for c in responses.calls if c.request.url.endswith("/authentication")]
    assert len(auth_calls) == 2


@responses.activate
def test_request_retries_429_honouring_retry_after(client, no_sleep):
    responses.add(responses.POST, f"{BASE}/authentication", json={}, status=201)
    responses.add(
        responses.GET, f"{BASE}/alphas/A1", status=429, headers={"Retry-After": "7"}
    )
    responses.add(responses.GET, f"{BASE}/alphas/A1", json={"id": "A1"}, status=200)

    assert client.alpha("A1")["id"] == "A1"
    assert 7.0 in no_sleep


@responses.activate
def test_request_gives_up_after_max_retries(client, no_sleep):
    responses.add(responses.POST, f"{BASE}/authentication", json={}, status=201)
    for _ in range(4):
        responses.add(responses.GET, f"{BASE}/alphas/A1", status=503)
    with pytest.raises(BrainError):
        client.alpha("A1")


@responses.activate
def test_data_fields_paginates_and_filters(client):
    responses.add(responses.POST, f"{BASE}/authentication", json={}, status=201)
    page1 = {"count": 3, "results": [{"id": "close"}, {"id": "volume"}]}
    page2 = {"count": 3, "results": [{"id": "returns"}]}
    responses.add(responses.GET, f"{BASE}/data-fields", json=page1, status=200)
    responses.add(responses.GET, f"{BASE}/data-fields", json=page2, status=200)

    fields = client.data_fields(limit=2)
    assert [f["id"] for f in fields] == ["close", "volume", "returns"]
    first = responses.calls[1].request
    assert "universe=TOP3000" in first.url and "delay=1" in first.url
    assert "type=MATRIX" in first.url
