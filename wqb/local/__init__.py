"""Local simulation: evaluate alpha expressions without spending Brain quota.

Brain's simulation quota is the binding constraint on any automated search, so
the search needs a free fitness signal. This package evaluates an expression
over a locally held panel of data and reports Brain-shaped metrics, which lets
thousands of candidates be screened offline and only the best few spend a real
simulation.

It is a *ranking* tool, not a replica. Absolute numbers will differ from
Brain's -- different data vendor, different universe construction, and at least
one metric definition that is not publicly documented (see
:mod:`wqb.local.engine`). What makes it useful is rank correlation with Brain's
own scores, which is why :mod:`wqb.local.calibrate` exists.
"""

from .engine import Metrics, SimConfig, simulate
from .evaluator import ExpressionError, evaluate_expression
from .panel import Panel

__all__ = [
    "ExpressionError", "Metrics", "Panel", "SimConfig",
    "evaluate_expression", "simulate",
]
