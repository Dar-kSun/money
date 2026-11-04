"""The Phase 4 automation loop: generate -> simulate -> evaluate -> log."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Iterable

import requests

from .client import BrainClient, BrainError
from .config import RunConfig
from .evaluate import Verdict, evaluate
from .generate import Candidate
from .simulate import SimulationResult, run_simulation
from .store import ResultStore

log = logging.getLogger(__name__)


def clone_client(client: BrainClient) -> BrainClient:
    """A sibling client sharing the authenticated cookies.

    ``requests.Session`` is not thread-safe, so each worker thread gets its own
    session and connection pool; copying the cookie jar means only one
    authentication request is spent for the whole run.
    """
    session = requests.Session()
    session.headers.update(client.session.headers)
    session.cookies.update(client.session.cookies)
    sibling = BrainClient(
        client.credentials,
        base_url=client.base_url,
        session=session,
        timeout=client.timeout,
        max_retries=client.max_retries,
    )
    sibling.authenticated = client.authenticated
    return sibling


@dataclass
class RunReport:
    simulated: int = 0
    passed: int = 0
    failed: int = 0
    errored: int = 0
    skipped: int = 0
    winners: list[tuple[Candidate, SimulationResult]] = field(default_factory=list)

    def line(self) -> str:
        return (
            f"simulated={self.simulated} passed={self.passed} "
            f"below_threshold={self.failed} errored={self.errored} skipped={self.skipped}"
        )


ProgressFn = Callable[[Candidate, SimulationResult, Verdict], None]


def _default_progress(candidate: Candidate, result: SimulationResult, verdict: Verdict) -> None:
    mark = "PASS" if verdict.passed else ("ERR " if not result.ok else "    ")
    detail = result.summary() if result.ok else (result.message or result.status)
    print(f"[{mark}] {candidate.expression}\n       {detail}", flush=True)
    if result.ok and not verdict.passed:
        print(f"       rejected: {verdict.reason}", flush=True)


def run_batch(
    client: BrainClient,
    candidates: Iterable[Candidate],
    store: ResultStore,
    config: RunConfig | None = None,
    *,
    progress: ProgressFn | None = _default_progress,
    resume: bool = True,
) -> RunReport:
    """Simulate every candidate, grade it, and persist the outcome.

    Runs ``config.concurrency`` simulations at a time. Candidates already in
    the store are skipped when ``resume`` is set, so an interrupted run can be
    restarted without spending simulation quota twice.
    """
    config = config or RunConfig()
    report = RunReport()
    seen = store.seen() if resume else set()

    todo: list[Candidate] = []
    for candidate in candidates:
        if candidate.expression in seen:
            report.skipped += 1
            continue
        todo.append(candidate)
        if config.max_alphas is not None and len(todo) >= config.max_alphas:
            break

    if not todo:
        log.info("Nothing to simulate (%d skipped as already seen)", report.skipped)
        return report

    print(
        f"Simulating {len(todo)} alpha(s), {config.concurrency} at a time "
        f"(skipping {report.skipped} already simulated)\n",
        flush=True,
    )

    lock = threading.Lock()
    local = threading.local()

    def worker(candidate: Candidate) -> tuple[Candidate, SimulationResult, Verdict]:
        if not hasattr(local, "client"):
            local.client = clone_client(client)
        try:
            result = run_simulation(
                local.client,
                candidate.expression,
                config.settings,
                poll_interval=config.poll_interval,
                timeout=config.poll_timeout,
            )
        except BrainError as exc:
            result = SimulationResult(
                expression=candidate.expression, status="ERROR", message=str(exc)
            )
        verdict = evaluate(
            result, min_sharpe=config.min_sharpe, min_fitness=config.min_fitness
        )
        with lock:
            store.record(
                expression=candidate.expression,
                template=candidate.template,
                field=candidate.field,
                window=candidate.window,
                status=result.status,
                alpha_id=result.alpha_id,
                metrics=result.metrics,
                passed=verdict.passed,
                reason=verdict.reason,
            )
        return candidate, result, verdict

    with ThreadPoolExecutor(max_workers=max(1, config.concurrency)) as pool:
        futures = {pool.submit(worker, c): c for c in todo}
        try:
            for future in as_completed(futures):
                candidate, result, verdict = future.result()
                report.simulated += 1
                if verdict.passed:
                    report.passed += 1
                    report.winners.append((candidate, result))
                elif not result.ok:
                    report.errored += 1
                else:
                    report.failed += 1
                if progress:
                    progress(candidate, result, verdict)
        except KeyboardInterrupt:
            print("\nInterrupted; cancelling queued simulations. Progress is saved.", flush=True)
            for future in futures:
                future.cancel()
            raise

    return report
