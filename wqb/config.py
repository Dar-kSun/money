"""Configuration for the WorldQuant Brain automation CLI.

Phase 3 of the spec fixes the simulation environment (USA / D1 / TOP3000).
Those values live in :data:`DEFAULT_SETTINGS` and are overridable per run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

API_BASE = os.getenv("WQB_API_BASE", "https://api.worldquantbrain.com")

# Pass criteria from the System Directive.
MIN_SHARPE = float(os.getenv("WQB_MIN_SHARPE", "1.25"))
MIN_FITNESS = float(os.getenv("WQB_MIN_FITNESS", "1.0"))

DATA_DIR = Path(os.getenv("WQB_DATA_DIR", "data")).expanduser()


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Credentials:
    email: str
    password: str

    @classmethod
    def from_env(cls) -> "Credentials":
        email = os.getenv("WQB_EMAIL")
        password = os.getenv("WQB_PASSWORD")
        missing = [n for n, v in (("WQB_EMAIL", email), ("WQB_PASSWORD", password)) if not v]
        if missing:
            raise ConfigError(
                "Missing credentials: %s. Export them or put them in a .env file "
                "(see .env.example); never commit them." % ", ".join(missing)
            )
        return cls(email=email, password=password)  # type: ignore[arg-type]

    def __repr__(self) -> str:  # keep the password out of tracebacks and logs
        return f"Credentials(email={self.email!r}, password=***)"


@dataclass(frozen=True)
class SimSettings:
    """The Phase 3 simulation settings block."""

    instrumentType: str = "EQUITY"
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = 1
    decay: int = 5           # 4-5 cuts turnover, which lifts Fitness over 1.0
    neutralization: str = "SUBINDUSTRY"
    truncation: float = 0.05
    pasteurization: str = "ON"
    unitHandling: str = "VERIFY"
    nanHandling: str = "OFF"
    language: str = "FASTEXPR"
    visualization: bool = False
    testPeriod: str = "P0Y0M"

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_SETTINGS = SimSettings()


def simulation_payload(expression: str, settings: SimSettings | None = None) -> dict[str, Any]:
    """Build the ``POST /simulations`` body for a single alpha expression."""
    expression = expression.strip()
    if not expression:
        raise ValueError("alpha expression must not be empty")
    return {
        "type": "REGULAR",
        "settings": (settings or DEFAULT_SETTINGS).to_payload(),
        "regular": expression,
    }


@dataclass
class RunConfig:
    """Knobs for the Phase 4 automation loop."""

    concurrency: int = 3            # Brain throttles concurrent simulations
    poll_interval: float = 5.0      # fallback when no Retry-After header
    poll_timeout: float = 1800.0    # give up on a single simulation after 30 min
    max_alphas: int | None = None   # cap per run; None means the whole grid
    settings: SimSettings = field(default_factory=SimSettings)
    min_sharpe: float = MIN_SHARPE
    min_fitness: float = MIN_FITNESS
