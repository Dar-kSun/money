"""One-shot verification of the live Brain API contract.

Every assumption in this package about Brain's API was written from
documentation, not from an observed response: the environment it was built in
blocks ``api.worldquantbrain.com``. This module checks each assumption against
the real service in one run and reports which held, so the first session with
network access settles the contract instead of discovering it one traceback at
a time.

It also seeds calibration: the same expression is simulated remotely and
locally, and the two metric sets are printed side by side.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

from .client import AuthError, BiometricRequired, BrainClient, BrainError
from .config import ConfigError, Credentials, SimSettings
from .evaluate import evaluate
from .simulate import run_simulation

# The validated blueprint: cheap, known-good, and the thing most worth
# comparing local metrics against.
PROBE_EXPRESSION = "rank(-1 * (close / ts_mean(close, 20) - 1))"

IS_KEYS = ("sharpe", "fitness", "turnover", "returns", "drawdown", "margin")


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""

    def line(self) -> str:
        return f"[{'PASS' if self.ok else 'FAIL'}] {self.name}" + (
            f"\n       {self.detail}" if self.detail else ""
        )


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, ok: bool, detail: str = "") -> Check:
        check = Check(name, ok, detail)
        self.checks.append(check)
        return check

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def render(self) -> str:
        lines = [c.line() for c in self.checks]
        failed = [c for c in self.checks if not c.ok]
        lines.append("")
        lines.append(
            f"{len(self.checks) - len(failed)}/{len(self.checks)} checks passed"
            if failed else f"All {len(self.checks)} checks passed."
        )
        if failed:
            lines.append(
                "\nThe failures above are contract mismatches, not bugs in your "
                "data. Fix the named module and add a test using the actual "
                "payload printed above."
            )
        return "\n".join(lines)


def verify(
    *,
    client: BrainClient | None = None,
    expression: str = PROBE_EXPRESSION,
    settings: SimSettings | None = None,
    skip_simulation: bool = False,
) -> Report:
    """Exercise auth, data-fields and simulation against the live API."""
    report = Report()

    # --- authentication ---------------------------------------------------
    if client is None:
        try:
            client = BrainClient(Credentials.from_env())
        except ConfigError as exc:
            report.add("credentials present", False, str(exc))
            return report
    if not client.authenticated:
        try:
            client.authenticate()
        except BiometricRequired as exc:
            report.add("authentication", False, f"biometric check required: {exc}")
            return report
        except (AuthError, BrainError) as exc:
            report.add("authentication", False, str(exc))
            return report
    report.add("authentication (POST /authentication -> 201)", True,
               f"user={client.user_id or 'id not reported'}")

    # --- data fields ------------------------------------------------------
    try:
        fields = client.data_fields(limit=10)
    except BrainError as exc:
        report.add("GET /data-fields", False, str(exc))
        fields = []
    else:
        report.add("GET /data-fields", True, f"{len(fields)} field(s) returned")
        if fields:
            sample = fields[0]
            report.add(
                "data-field rows carry an id",
                "id" in sample,
                f"keys: {', '.join(sorted(sample)[:12])}",
            )
            # These drive crowding-aware field selection and are the least
            # certain assumption in the package.
            crowding = [k for k in ("userCount", "alphaCount", "user_count",
                                    "alpha_count") if k in sample]
            report.add(
                "crowding counts present (userCount/alphaCount)",
                bool(crowding),
                f"found {', '.join(crowding)}" if crowding else
                "absent: wqb/fields.py ranking falls back to coverage order",
            )

    if skip_simulation:
        return report

    # --- simulation -------------------------------------------------------
    try:
        result = run_simulation(client, expression, settings)
    except BrainError as exc:
        report.add("POST /simulations + poll", False, str(exc))
        return report

    report.add(
        "POST /simulations + poll to terminal state",
        result.ok,
        f"status={result.status} alpha={result.alpha_id or '-'}"
        + (f" message={result.message}" if result.message else ""),
    )
    if not result.ok:
        return report

    present = [k for k in IS_KEYS if k in result.metrics]
    report.add(
        "in-sample summary uses the expected metric names",
        len(present) >= 3,
        f"found {', '.join(present) or 'none'}; "
        f"all keys: {', '.join(sorted(result.metrics)[:14])}",
    )
    report.add(
        "platform checks returned",
        bool(result.checks),
        f"{len(result.checks)} check(s)" if result.checks else
        "none returned; wqb/evaluate.py cannot gate on them",
    )

    verdict = evaluate(result)
    report.metrics = dict(result.metrics)
    report.add(
        "grading runs end to end",
        True,
        f"{'PASS' if verdict.passed else 'FAIL'}: {verdict.reason}",
    )
    return report


def compare_local(
    expression: str,
    remote_metrics: dict[str, Any],
    panel: Any,
    config: Any = None,
) -> str:
    """Print Brain's metrics next to the local simulator's for one expression.

    This is the first data point for calibration. What matters is not that the
    numbers match -- they will not, on different data -- but whether they are
    close enough in *ordering* for local screening to be worth anything, which
    needs a dozen or so expressions, not one.
    """
    from .local.engine import simulate as local_simulate

    local = local_simulate(expression, panel, config).as_dict()
    rows = ["", f"Local vs Brain for {expression}", ""]
    rows.append(f"  {'metric':<10} {'brain':>12} {'local':>12}   ratio")
    for key in IS_KEYS:
        remote_value = _num(remote_metrics.get(key))
        local_value = _num(local.get(key))
        ratio = (
            f"{local_value / remote_value:>7.2f}x"
            if remote_value not in (None, 0.0) and local_value is not None
            else "      -"
        )
        rows.append(
            f"  {key:<10} {_fmt(remote_value):>12} {_fmt(local_value):>12}  {ratio}"
        )
    rows.append("")
    rows.append(
        "  One expression cannot settle calibration. Repeat for ~20 and check\n"
        "  that the local Sharpe RANKS them the way Brain does."
    )
    return "\n".join(rows)


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"
