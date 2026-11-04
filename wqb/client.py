"""Authenticated HTTP client for the WorldQuant Brain API.

Phase 1 of the spec: ``requests.Session`` plus session-based HTTP Basic
authentication. Everything else in the package goes through this class so that
re-authentication, throttling and ``Retry-After`` handling live in one place.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Iterator
from urllib.parse import urljoin

import requests

from .config import API_BASE, Credentials

log = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class BrainError(RuntimeError):
    """Any non-recoverable API failure."""


class AuthError(BrainError):
    """Authentication failed and cannot be retried without user action."""


class BiometricRequired(AuthError):
    """Brain wants a biometric/persona check completed in a browser first."""


class BrainClient:
    """Thin, retrying wrapper over the Brain REST API."""

    def __init__(
        self,
        credentials: Credentials | None = None,
        *,
        base_url: str = API_BASE,
        session: requests.Session | None = None,
        timeout: float = 30.0,
        max_retries: int = 4,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.credentials = credentials
        self.session = session or requests.Session()
        self.session.headers.setdefault("Accept", "application/json")
        self.timeout = timeout
        self.max_retries = max_retries
        self.authenticated = False
        self._user: dict[str, Any] = {}

    # ------------------------------------------------------------------ auth
    def authenticate(self) -> dict[str, Any]:
        """POST /authentication with Basic auth; Brain returns 201 on success.

        The JWT lands in the session cookie jar, so every later request is
        cookie-authenticated and needs no ``auth=`` argument.
        """
        creds = self.credentials or Credentials.from_env()
        self.credentials = creds
        url = urljoin(self.base_url, "authentication")
        resp = self.session.post(url, auth=(creds.email, creds.password), timeout=self.timeout)

        if resp.status_code == 201:
            self.authenticated = True
            self._user = _json_or_empty(resp)
            log.info("Authentication successful (user=%s)", self.user_id or "unknown")
            return self._user

        if resp.status_code == 401:
            # Brain signals an unfinished biometric enrolment via WWW-Authenticate.
            if "persona" in resp.headers.get("WWW-Authenticate", "").lower():
                where = _json_or_empty(resp).get("location") or (
                    "https://platform.worldquantbrain.com/biometrics"
                )
                raise BiometricRequired(
                    "Brain requires a biometric/persona check before API access. "
                    f"Complete it in a browser at {where}, then re-run."
                )
            raise AuthError(
                "Authentication rejected (401). Check WQB_EMAIL / WQB_PASSWORD; "
                "an account can also be locked after repeated failures."
            )

        raise AuthError(f"Authentication failed: HTTP {resp.status_code} {resp.text[:300]}")

    @property
    def user_id(self) -> str | None:
        user = self._user.get("user")
        if isinstance(user, dict):
            return user.get("id")
        return None

    def _ensure_auth(self) -> None:
        if not self.authenticated:
            self.authenticate()

    # --------------------------------------------------------------- request
    def request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        """Issue a request, re-authenticating once on 401 and backing off on 429/5xx."""
        self._ensure_auth()
        url = path if path.startswith("http") else urljoin(self.base_url, path.lstrip("/"))
        kwargs.setdefault("timeout", self.timeout)
        reauthed = False

        for attempt in range(self.max_retries + 1):
            resp = self.session.request(method, url, **kwargs)

            if resp.status_code == 401 and not reauthed:
                # The JWT expires after a few hours; renew it and retry once.
                log.info("Session expired, re-authenticating")
                self.authenticated = False
                self.authenticate()
                reauthed = True
                continue

            if resp.status_code in RETRYABLE_STATUS and attempt < self.max_retries:
                delay = self._retry_delay(resp, attempt)
                log.warning(
                    "HTTP %s from %s %s; retrying in %.1fs (attempt %d/%d)",
                    resp.status_code, method, url, delay, attempt + 1, self.max_retries,
                )
                time.sleep(delay)
                continue

            return resp

        return resp  # pragma: no cover - loop always returns

    def _retry_delay(self, resp: requests.Response, attempt: int) -> float:
        """Honour Retry-After when present, else exponential backoff with jitter."""
        retry_after = resp.headers.get("Retry-After")
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
        return min(60.0, (2 ** attempt)) + random.uniform(0, 1)

    def get_json(self, path: str, **kwargs: Any) -> dict[str, Any]:
        resp = self.request("GET", path, **kwargs)
        if not resp.ok:
            raise BrainError(f"GET {path} failed: HTTP {resp.status_code} {resp.text[:300]}")
        return _json_or_empty(resp)

    # ----------------------------------------------------------- data fields
    def datasets(
        self,
        *,
        instrument_type: str = "EQUITY",
        region: str = "USA",
        delay: int = 1,
        universe: str = "TOP3000",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        params = {
            "instrumentType": instrument_type,
            "region": region,
            "delay": delay,
            "universe": universe,
            "limit": limit,
        }
        return list(self._paginate("data-sets", params))

    def data_fields(
        self,
        *,
        instrument_type: str = "EQUITY",
        region: str = "USA",
        delay: int = 1,
        universe: str = "TOP3000",
        dataset_id: str | None = None,
        field_type: str | None = "MATRIX",
        search: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List usable data fields for the target environment.

        ``field_type="MATRIX"`` keeps the result to per-stock-per-day fields,
        which are the ones the Phase 2 templates can consume directly.
        """
        params: dict[str, Any] = {
            "instrumentType": instrument_type,
            "region": region,
            "delay": delay,
            "universe": universe,
            "limit": limit,
        }
        if dataset_id:
            params["dataset.id"] = dataset_id
        if field_type:
            params["type"] = field_type
        if search:
            params["search"] = search
        return list(self._paginate("data-fields", params))

    def _paginate(self, path: str, params: dict[str, Any]) -> Iterator[dict[str, Any]]:
        offset, limit = 0, int(params.get("limit", 50))
        while True:
            page = self.get_json(path, params={**params, "offset": offset})
            results = page.get("results") or []
            yield from results
            offset += limit
            if len(results) < limit or offset >= int(page.get("count") or 0):
                return

    # ---------------------------------------------------------------- alphas
    def alpha(self, alpha_id: str) -> dict[str, Any]:
        return self.get_json(f"alphas/{alpha_id}")

    def update_alpha(self, alpha_id: str, **fields: Any) -> dict[str, Any]:
        """PATCH an alpha's metadata (name, color, tags, description)."""
        resp = self.request("PATCH", f"alphas/{alpha_id}", json=fields)
        if not resp.ok:
            raise BrainError(f"PATCH alphas/{alpha_id} failed: HTTP {resp.status_code}")
        return _json_or_empty(resp)


def _json_or_empty(resp: requests.Response) -> dict[str, Any]:
    """Brain sends empty bodies while a job is in flight; treat those as {}."""
    if not resp.content:
        return {}
    try:
        data = resp.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {"results": data}
