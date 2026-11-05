"""Parse and evaluate a FastExpr alpha expression over a :class:`Panel`.

FastExpr's surface syntax -- function calls, arithmetic, bare field names -- is
a subset of Python expression syntax, so the expression is parsed with
:mod:`ast` and walked directly. Nothing is ``eval``'d as text: the walker
handles exactly the node types an alpha needs, so a name that is not a panel
field or a known operator is an error rather than a lookup into the
interpreter's namespace. There is no attribute access, no subscripting, no
comprehension and no call to anything outside the operator table.
"""

from __future__ import annotations

import ast
from typing import Any

import numpy as np

from .operators import Context, OPERATORS
from .panel import Panel

MAX_EXPRESSION_CHARS = 4000
MAX_AST_NODES = 500


class ExpressionError(ValueError):
    """The expression is malformed, or uses an unknown field or operator."""


_ALLOWED_NODES = (
    ast.Expression, ast.Call, ast.Name, ast.Load, ast.Constant,
    ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.Pow, ast.Mod, ast.USub, ast.UAdd,
    ast.Compare, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
    ast.keyword,
)

_BINOPS = {
    ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
    ast.Div: np.divide, ast.Pow: np.power, ast.Mod: np.mod,
}
_COMPARES = {
    ast.Lt: np.less, ast.LtE: np.less_equal, ast.Gt: np.greater,
    ast.GtE: np.greater_equal, ast.Eq: np.equal, ast.NotEq: np.not_equal,
}


def evaluate_expression(expression: str, panel: Panel) -> np.ndarray:
    """Evaluate ``expression`` and return the raw signal, shape (T, N).

    The result is the alpha *before* any portfolio construction -- delay,
    neutralisation, decay and truncation are applied by
    :func:`wqb.local.engine.simulate`.
    """
    if not isinstance(expression, str) or not expression.strip():
        raise ExpressionError("expression is empty")
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise ExpressionError(
            f"expression is {len(expression)} chars, limit {MAX_EXPRESSION_CHARS}"
        )

    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"syntax error: {exc.msg}") from exc

    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_AST_NODES:
        raise ExpressionError(f"expression too complex ({len(nodes)} nodes)")
    for node in nodes:
        if not isinstance(node, _ALLOWED_NODES):
            raise ExpressionError(
                f"{type(node).__name__} is not allowed in an alpha expression"
            )

    ctx = Context(universe=panel.universe, groups=panel.groups)
    result = _eval(tree.body, panel, ctx)

    if np.isscalar(result) or np.ndim(result) == 0:
        result = np.full(panel.shape, float(result))
    arr = np.asarray(result, dtype=float)
    if arr.shape != panel.shape:
        arr = np.broadcast_to(arr, panel.shape).copy()
    return arr


def _eval(node: ast.AST, panel: Panel, ctx: Context) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ExpressionError(f"unsupported constant {node.value!r}")
        return float(node.value)

    if isinstance(node, ast.Name):
        name = node.id
        if panel.has(name):
            return panel.get(name)
        if name in OPERATORS:
            raise ExpressionError(f"operator {name!r} used without arguments")
        raise ExpressionError(
            f"unknown field {name!r}; the panel holds: "
            f"{', '.join(panel.field_names) or 'nothing'}"
        )

    if isinstance(node, ast.UnaryOp):
        operand = _eval(node.operand, panel, ctx)
        return -np.asarray(operand, dtype=float) if isinstance(node.op, ast.USub) else operand

    if isinstance(node, ast.BinOp):
        fn = _BINOPS.get(type(node.op))
        if fn is None:
            raise ExpressionError(f"unsupported operator {type(node.op).__name__}")
        left = np.asarray(_eval(node.left, panel, ctx), dtype=float)
        right = np.asarray(_eval(node.right, panel, ctx), dtype=float)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            out = fn(left, right)
        return np.where(np.isfinite(out), out, np.nan)

    if isinstance(node, ast.Compare):
        if len(node.ops) != 1:
            raise ExpressionError("chained comparisons are not supported")
        fn = _COMPARES.get(type(node.ops[0]))
        if fn is None:
            raise ExpressionError(f"unsupported comparison {type(node.ops[0]).__name__}")
        left = np.asarray(_eval(node.left, panel, ctx), dtype=float)
        right = np.asarray(_eval(node.comparators[0], panel, ctx), dtype=float)
        with np.errstate(invalid="ignore"):
            return fn(left, right).astype(float)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ExpressionError("only direct operator calls are allowed")
        name = node.func.id
        operator = OPERATORS.get(name)
        if operator is None:
            raise ExpressionError(
                f"unknown operator {name!r}; supported: {', '.join(sorted(OPERATORS))}"
            )
        args = [_eval(a, panel, ctx) for a in node.args]
        kwargs = {kw.arg: _eval(kw.value, panel, ctx) for kw in node.keywords if kw.arg}
        try:
            return operator(ctx, *args, **kwargs)
        except ExpressionError:
            raise
        except KeyError as exc:
            raise ExpressionError(str(exc).strip("'\"")) from exc
        except TypeError as exc:
            raise ExpressionError(f"{name}: {exc}") from exc
        except ValueError as exc:
            raise ExpressionError(f"{name}: {exc}") from exc

    raise ExpressionError(f"{type(node).__name__} is not allowed in an alpha expression")


def expression_fields(expression: str) -> set[str]:
    """Field names an expression references, without evaluating it."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"syntax error: {exc.msg}") from exc
    called = {
        n.func.id for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    return {
        n.id for n in ast.walk(tree)
        if isinstance(n, ast.Name) and n.id not in called and n.id not in OPERATORS
    }
