import pytest

from wqb.client import BrainClient
from wqb.config import Credentials

BASE = "https://api.worldquantbrain.com"


@pytest.fixture
def creds():
    return Credentials(email="quant@example.com", password="secret")


@pytest.fixture
def client(creds):
    return BrainClient(creds, base_url=BASE, max_retries=2)


@pytest.fixture
def no_sleep(monkeypatch):
    """Make every backoff/poll sleep instant."""
    slept = []
    for module in ("wqb.client", "wqb.simulate", "wqb.submit"):
        monkeypatch.setattr(f"{module}.time.sleep", lambda s: slept.append(s))
    return slept
