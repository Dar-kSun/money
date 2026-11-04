"""Simulation submission and polling (Phases 3 and 4)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .client import BrainClient, BrainError
from .config import SimSettings, simulation_payload

log = logging.getLogger(__name__)

TERMINAL_OK = frozenset({"COMPLETE", "COMPLETED", "WARNING"})
TERMINAL_BAD = frozenset({"ERROR", "FAIL", "FAILED"})


class SimulationTimeout(BrainError):
    """A simulation did not reach a terminal state in time."""


@dataclass
class SimulationResult:
    expression: str
    status: str
    alpha_id: str | None = None
    progress_url: str | None = None
    message: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    checks: list[dict[str, Any]] = field(default_factory=list)
    elapsed: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status in TERMINAL_OK and self.alpha_id is not None

    @property
    def sharpe(self) -> float | None:
        return _as_float(self.metrics.get("sharpe"))

    @property
    def fitness(self) -> float | None:
        return _as_float(self.metrics.get("fitness"))

    @property
    def turnover(self) -> float | None:
        return _as_float(self.metrics.get("turnover"))

    @property
    def returns(self) -> float | None:
        return _as_float(self.metrics.get("returns"))

    @property
    def drawdown(self) -> float | None:
        return _as_float(self.metrics.get("drawdown"))

    @property
    def margin(self) -> float | None:
        return _as_float(self.metrics.get("margin"))

    def summary(self) -> str:
        if not self.ok:
            return f"{self.status}: {self.message or 'no detail'}"
        return (
            f"sharpe={_fmt(self.sharpe)} fitness={_fmt(self.fitness)} "
            f"turnover={_fmt(self.turnover)} returns={_fmt(self.returns)}"
        )


def start_simulation(
    client: BrainClient, expression: str, settings: SimSettings | None = None
) -> str:
    """POST the alpha to /simulations and return the progress URL."""
    payload = simulation_payload(expression, settings)
    resp = client.request("POST", "simulations", json=payload)

    if resp.status_code not in (200, 201):
        detail = resp.text[:300]
        raise BrainError(
            f"Simulation request rejected: HTTP {resp.status_code} {detail}"
        )

    progress_url = resp.headers.get("Location")
    if not progress_url:
        # Some responses carry the id in the body instead of the header.
        body = resp.json() if resp.content else {}
        sim_id = body.get("id") if isinstance(body, dict) else None
        if not sim_id:
            raise BrainError("Simulation accepted but no Location header or id was returned")
        progress_url = f"{client.base_url}simulations/{sim_id}"
    return progress_url


def poll_simulation(
    client: BrainClient,
    progress_url: str,
    *,
    expression: str = "",
    poll_interval: float = 5.0,
    timeout: float = 1800.0,
) -> SimulationResult:
    """Poll a simulation to completion, honouring the Retry-After header.

    While a simulation runs, Brain answers with an empty body and a
    ``Retry-After`` header; that header is the authoritative backoff.
    """
    started = time.monotonic()
    while True:
        resp = client.request("GET", progress_url)
        elapsed = time.monotonic() - started

        if not resp.ok:
            return SimulationResult(
                expression=expression,
                status="ERROR",
                progress_url=progress_url,
                message=f"HTTP {resp.status_code} while polling: {resp.text[:200]}",
                elapsed=elapsed,
            )

        retry_after = resp.headers.get("Retry-After")
        body = resp.json() if resp.content else {}
        status = str(body.get("status") or "").upper()

        if retry_after and not status:
            wait = _safe_float(retry_after, poll_interval)
        elif status in TERMINAL_OK or status in TERMINAL_BAD:
            return _finalize(client, body, expression, progress_url, elapsed)
        elif retry_after:
            wait = _safe_float(retry_after, poll_interval)
        else:
            wait = poll_interval

        if elapsed > timeout:
            raise SimulationTimeout(
                f"Simulation {progress_url} still {status or 'in progress'} after {elapsed:.0f}s"
            )
        time.sleep(wait)


def _finalize(
    client: BrainClient,
    body: dict[str, Any],
    expression: str,
    progress_url: str,
    elapsed: float,
) -> SimulationResult:
    status = str(body.get("status") or "").upper()
    alpha_id = body.get("alpha")
    result = SimulationResult(
        expression=expression or (body.get("regular") or {}).get("code", ""),
        status=status,
        alpha_id=alpha_id,
        progress_url=progress_url,
        message=body.get("message"),
        elapsed=elapsed,
    )
    if status in TERMINAL_BAD or not alpha_id:
        if not result.message:
            result.message = "simulation failed without a message"
        return result

    # Pull the In-Sample summary that Phase 4 grades on.
    alpha = client.alpha(alpha_id)
    is_summary = alpha.get("is") or {}
    result.metrics = {k: v for k, v in is_summary.items() if k != "checks"}
    result.checks = list(is_summary.get("checks") or [])
    return result


def run_simulation(
    client: BrainClient,
    expression: str,
    settings: SimSettings | None = None,
    *,
    poll_interval: float = 5.0,
    timeout: float = 1800.0,
) -> SimulationResult:
    """Submit one expression and block until it has an In-Sample verdict."""
    progress_url = start_simulation(client, expression, settings)
    log.debug("Simulating %s -> %s", expression, progress_url)
    return poll_simulation(
        client,
        progress_url,
        expression=expression,
        poll_interval=poll_interval,
        timeout=timeout,
    )


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any, default: float) -> float:
    parsed = _as_float(value)
    return default if parsed is None else max(0.0, parsed)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"
