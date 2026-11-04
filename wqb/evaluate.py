"""Grading of In-Sample results against the System Directive thresholds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import MIN_FITNESS, MIN_SHARPE
from .simulate import SimulationResult


@dataclass(frozen=True)
class Verdict:
    passed: bool
    reasons: tuple[str, ...]
    failed_checks: tuple[str, ...] = ()

    @property
    def reason(self) -> str:
        return "; ".join(self.reasons) if self.reasons else "meets all thresholds"


def failed_platform_checks(checks: list[dict[str, Any]]) -> tuple[str, ...]:
    """Names of Brain's own IS checks that came back FAIL.

    Brain runs its own gate (low Sharpe, high turnover, concentrated weight,
    correlation with existing alphas, ...). A FAIL there blocks submission no
    matter how good Sharpe and Fitness look, so it is part of the verdict.
    """
    failed = []
    for check in checks or []:
        if str(check.get("result", "")).upper() == "FAIL":
            failed.append(str(check.get("name") or "UNKNOWN"))
    return tuple(failed)


def evaluate(
    result: SimulationResult,
    *,
    min_sharpe: float = MIN_SHARPE,
    min_fitness: float = MIN_FITNESS,
    require_platform_checks: bool = True,
) -> Verdict:
    """Apply ``Sharpe > min_sharpe`` and ``Fitness >= min_fitness``.

    Note the asymmetry, which is taken from the directive verbatim: Sharpe is a
    strict ``>`` while Fitness is an inclusive ``>=``.
    """
    if not result.ok:
        return Verdict(False, (f"simulation did not complete ({result.status})",))

    reasons: list[str] = []
    sharpe, fitness = result.sharpe, result.fitness

    if sharpe is None:
        reasons.append("no sharpe reported")
    elif not sharpe > min_sharpe:
        reasons.append(f"sharpe {sharpe:.3f} <= {min_sharpe}")

    if fitness is None:
        reasons.append("no fitness reported")
    elif not fitness >= min_fitness:
        reasons.append(f"fitness {fitness:.3f} < {min_fitness}")

    failed = failed_platform_checks(result.checks)
    if failed and require_platform_checks:
        reasons.append(f"platform checks failed: {', '.join(failed)}")

    return Verdict(not reasons, tuple(reasons), failed)
