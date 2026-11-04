"""Out-of-Sample submission workflow.

Submission is irreversible and public: a submitted alpha is scored against
out-of-sample data and joins the user's Brain portfolio. So the default here is
a dry run -- ``submit_alpha`` only sends the POST when ``confirm=True``, and the
CLI requires an explicit ``--yes`` flag to set it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .client import BrainClient
from .simulate import _safe_float

log = logging.getLogger(__name__)


@dataclass
class SubmissionOutcome:
    alpha_id: str
    submitted: bool
    status: str
    checks: list[dict[str, Any]] = field(default_factory=list)
    message: str = ""

    @property
    def failed_checks(self) -> list[str]:
        return [
            str(c.get("name") or "UNKNOWN")
            for c in self.checks
            if str(c.get("result", "")).upper() == "FAIL"
        ]

    @property
    def pending_checks(self) -> list[str]:
        return [
            str(c.get("name") or "UNKNOWN")
            for c in self.checks
            if str(c.get("result", "")).upper() == "PENDING"
        ]

    def describe(self) -> str:
        if self.submitted:
            return f"{self.alpha_id}: SUBMITTED"
        failed = self.failed_checks
        if failed:
            return f"{self.alpha_id}: blocked by {', '.join(failed)}"
        return f"{self.alpha_id}: {self.status} {self.message}".strip()


def check_submission(
    client: BrainClient,
    alpha_id: str,
    *,
    poll_interval: float = 5.0,
    timeout: float = 600.0,
) -> SubmissionOutcome:
    """Read Brain's submission checks for an alpha without submitting it.

    ``GET /alphas/{id}/check`` runs the same gate the submit button applies
    (correlation with existing alphas, self-correlation, IS thresholds, ...).
    """
    started = time.monotonic()
    while True:
        resp = client.request("GET", f"alphas/{alpha_id}/check")
        if resp.status_code == 403:
            return SubmissionOutcome(
                alpha_id, False, "NOT_ELIGIBLE",
                message="account is not permitted to submit this alpha",
            )
        if not resp.ok:
            return SubmissionOutcome(
                alpha_id, False, "ERROR",
                message=f"HTTP {resp.status_code} {resp.text[:200]}",
            )

        body = resp.json() if resp.content else {}
        checks = list(((body.get("is") or {}).get("checks")) or [])
        retry_after = resp.headers.get("Retry-After")

        if checks and not retry_after:
            failed = [c for c in checks if str(c.get("result", "")).upper() == "FAIL"]
            status = "FAIL" if failed else "PASS"
            return SubmissionOutcome(alpha_id, False, status, checks=checks)

        if time.monotonic() - started > timeout:
            return SubmissionOutcome(
                alpha_id, False, "TIMEOUT", checks=checks,
                message=f"checks still running after {timeout:.0f}s",
            )
        time.sleep(_safe_float(retry_after, poll_interval))


def submit_alpha(
    client: BrainClient,
    alpha_id: str,
    *,
    confirm: bool = False,
    poll_interval: float = 5.0,
    timeout: float = 900.0,
) -> SubmissionOutcome:
    """Submit an alpha out-of-sample.

    With ``confirm=False`` (the default) this only runs the pre-submission
    checks and reports what *would* happen; nothing is sent to Brain's submit
    endpoint. Submission is permanent, so the caller has to opt in.
    """
    precheck = check_submission(
        client, alpha_id, poll_interval=poll_interval, timeout=min(timeout, 600.0)
    )

    if not confirm:
        precheck.status = f"DRY_RUN({precheck.status})"
        precheck.message = (
            precheck.message or "pre-submission checks only; pass --yes to submit for real"
        )
        return precheck

    if precheck.failed_checks:
        return precheck  # never burn a submission slot on an alpha Brain will reject

    resp = client.request("POST", f"alphas/{alpha_id}/submit")
    if resp.status_code == 403:
        return SubmissionOutcome(
            alpha_id, False, "NOT_ELIGIBLE",
            message="submission refused (403): quota exhausted or alpha not eligible",
        )
    if resp.status_code not in (200, 201):
        return SubmissionOutcome(
            alpha_id, False, "ERROR",
            message=f"HTTP {resp.status_code} {resp.text[:200]}",
        )

    # Brain processes the submission asynchronously; poll until the checks settle.
    started = time.monotonic()
    while True:
        poll = client.request("GET", f"alphas/{alpha_id}/submit")
        body = poll.json() if poll.content else {}
        checks = list(((body.get("is") or {}).get("checks")) or [])
        retry_after = poll.headers.get("Retry-After")

        if poll.status_code in (200, 201) and not retry_after:
            failed = [c for c in checks if str(c.get("result", "")).upper() == "FAIL"]
            if failed:
                return SubmissionOutcome(alpha_id, False, "FAIL", checks=checks)
            log.info("Alpha %s submitted out-of-sample", alpha_id)
            return SubmissionOutcome(alpha_id, True, "SUBMITTED", checks=checks)

        if time.monotonic() - started > timeout:
            return SubmissionOutcome(
                alpha_id, False, "TIMEOUT", checks=checks,
                message="submission still processing; check the Brain UI",
            )
        time.sleep(_safe_float(retry_after, poll_interval))
