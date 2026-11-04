"""WorldQuant Brain alpha automation."""

from .client import AuthError, BiometricRequired, BrainClient, BrainError
from .config import Credentials, RunConfig, SimSettings, simulation_payload
from .evaluate import evaluate
from .generate import Candidate, generate, looks_like_naive_momentum
from .runner import run_batch
from .simulate import run_simulation
from .store import ResultStore
from .submit import check_submission, submit_alpha

__version__ = "0.1.0"

__all__ = [
    "AuthError", "BiometricRequired", "BrainClient", "BrainError",
    "Credentials", "RunConfig", "SimSettings", "simulation_payload",
    "Candidate", "ResultStore", "check_submission", "evaluate", "generate",
    "looks_like_naive_momentum", "run_batch", "run_simulation", "submit_alpha",
]
