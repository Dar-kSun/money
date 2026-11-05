"""Structural edits to an alpha expression, via its syntax tree.

Mutating strings with regexes breaks as soon as expressions nest. Instead an
expression is parsed with :mod:`ast`, a node is rewritten, and the tree is
unparsed. Every operand of the grammar is therefore valid by construction, and
the same safety checks the evaluator applies still gate the result.
"""

from __future__ import annotations

import ast
import random
from dataclasses import dataclass, field
from typing import Sequence

from ..fields import Direction, FieldClass, FieldInfo, classify
from ..generate import WINDOWS_BY_CLASS
from ..local.operators import OPERATORS

# Operators grouped by signature, so a swap keeps the expression well formed.
TS_WINDOW_OPS = (
    "ts_mean", "ts_std_dev", "ts_zscore", "ts_delta", "ts_sum",
    "ts_rank", "ts_min", "ts_max", "ts_argmax", "ts_argmin", "decay_linear",
)
CS_UNARY_OPS = ("rank", "zscore", "winsorize", "scale")
ELEMENTWISE_UNARY_OPS = ("abs", "log", "sign", "sqrt")
GROUP_OPS = ("group_neutralize", "group_rank", "group_mean")

DEFAULT_WINDOW_CHOICES = (2, 3, 5, 10, 20, 40, 60, 120, 250)

# How many leading arguments of an operator are series. The remaining ones are
# scalars (a window, an exponent) or a grouping operand, and grafting an array
# into those positions produces an expression that cannot evaluate -- so they
# are not value positions, and are edited only by their own mutation kind.
SERIES_ARITY: dict[str, int] = {
    **{op: 1 for op in TS_WINDOW_OPS},
    **{op: 1 for op in GROUP_OPS},
    "ts_delay": 1,
    "ts_corr": 2,
    "signed_power": 1,
    "power": 1,
    "scale": 1,
    "winsorize": 1,
}


@dataclass
class MutationSpace:
    """What a mutation is allowed to reach for."""

    fields: tuple[str, ...]
    windows: tuple[int, ...] = DEFAULT_WINDOW_CHOICES
    groups: tuple[str, ...] = ("subindustry", "sector", "industry", "market")
    allow_direction_flip: bool = True

    def windows_for(self, expression: str) -> tuple[int, ...]:
        """Windows appropriate to the slowest field the expression uses.

        Quarterly data does not reward a 2-day window, so a mutation on a
        fundamental expression should not propose one.
        """
        classes = {classify(f) for f in self.fields_in(expression)}
        if not classes:
            return self.windows
        allowed: set[int] = set()
        for klass in classes:
            allowed.update(WINDOWS_BY_CLASS.get(klass, self.windows))
        slow = {FieldClass.FUNDAMENTAL, FieldClass.ANALYST, FieldClass.OWNERSHIP}
        if classes & slow:
            allowed = {w for w in allowed if w >= 20} or set(self.windows)
        return tuple(sorted(allowed)) or self.windows

    def fields_in(self, expression: str) -> set[str]:
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError:
            return set()
        called = {
            n.func.id for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        return {
            n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name)
            and n.id not in called and n.id not in OPERATORS
            and n.id not in self.groups
        }


def mutation_space(
    fields: Sequence[str | FieldInfo], **kwargs
) -> MutationSpace:
    names = tuple(f.id if isinstance(f, FieldInfo) else str(f) for f in fields)
    return MutationSpace(fields=names, **kwargs)


class _Replace(ast.NodeTransformer):
    """Swap one node for another, matched by object identity.

    Identity matters: an earlier version picked nodes by their index in
    ``ast.walk`` order (breadth first) but replaced them during a
    ``NodeTransformer`` traversal (depth first), so edits landed on whichever
    node happened to share the index -- replacing a whole subexpression with a
    field name, or nesting a call inside its own argument.
    """

    def __init__(self, target: ast.AST, replacement: ast.AST) -> None:
        self.target = target
        self.replacement = replacement
        self.applied = False

    def visit(self, node: ast.AST) -> ast.AST:
        if node is self.target:
            self.applied = True
            return self.replacement
        return self.generic_visit(node)


def _value_children(node: ast.AST) -> list[ast.AST]:
    """Children of ``node`` that sit in a value position.

    A ``Call``'s ``func`` is excluded: it names the operator, and substituting
    an expression there produces nonsense like ``ts_mean(close, 20)(x)``. So
    are its scalar arguments -- a window or an exponent -- since grafting an
    array into ``ts_mean(close, <array>)`` cannot evaluate.
    """
    if isinstance(node, ast.Call):
        name = node.func.id if isinstance(node.func, ast.Name) else ""
        limit = SERIES_ARITY.get(name)
        return list(node.args if limit is None else node.args[:limit])
    if isinstance(node, ast.BinOp):
        return [node.left, node.right]
    if isinstance(node, ast.UnaryOp):
        return [node.operand]
    if isinstance(node, ast.Compare):
        return [node.left, *node.comparators]
    return []


def _value_nodes(tree: ast.Expression, *, include_root: bool = True) -> list[ast.AST]:
    out: list[ast.AST] = [tree.body] if include_root else []
    stack = [tree.body]
    while stack:
        for child in _value_children(stack.pop()):
            out.append(child)
            stack.append(child)
    return out


def _is_field(node: ast.AST, space: MutationSpace) -> bool:
    return (
        isinstance(node, ast.Name)
        and node.id not in OPERATORS
        and node.id not in space.groups
    )


def _candidates(kind: str, tree: ast.Expression, space: MutationSpace) -> list[ast.AST]:
    values = _value_nodes(tree)
    if kind == "window":
        return [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Constant)
            and isinstance(n.value, (int, float))
            and not isinstance(n.value, bool)
        ]
    if kind == "operator":
        return [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        ]
    if kind == "field":
        return [n for n in values if _is_field(n, space)]
    if kind == "group":
        return [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Name) and n.id in space.groups
        ]
    return []


def mutate(
    expression: str,
    space: MutationSpace,
    rng: random.Random | None = None,
) -> str:
    """Return a structurally altered copy of ``expression``.

    One edit is applied per call: a window change, an operator swap, a field
    swap, a sign flip, a new wrapping operator, or a group change. Returns the
    input unchanged when it cannot be parsed or no edit applies.
    """
    rng = rng or random.Random()
    source = expression.strip()
    try:
        ast.parse(source, mode="eval")
    except SyntaxError:
        return expression

    kinds = ["window", "operator", "field", "wrap", "group"]
    if space.allow_direction_flip:
        kinds.append("flip")
    rng.shuffle(kinds)

    for kind in kinds:
        mutated = _apply(kind, source, space, rng)
        if mutated:
            mutated = simplify(mutated)
            if mutated != source:
                return mutated
    return expression


def _apply(
    kind: str, source: str, space: MutationSpace, rng: random.Random
) -> str | None:
    tree = ast.parse(source, mode="eval")

    if kind == "flip":
        return _unparse(_flip(tree.body))

    if kind == "wrap":
        op = rng.choice(CS_UNARY_OPS + ELEMENTWISE_UNARY_OPS)
        return _unparse(ast.Call(
            func=ast.Name(id=op, ctx=ast.Load()), args=[tree.body], keywords=[]
        ))

    nodes = _candidates(kind, tree, space)
    rng.shuffle(nodes)
    for node in nodes:
        replacement = _rewrite(kind, node, space, rng, source)
        if replacement is None:
            continue
        transformer = _Replace(node, replacement)
        mutated = transformer.visit(tree)
        if not transformer.applied:
            continue
        try:
            return _unparse(mutated.body)
        except Exception:                      # pragma: no cover - defensive
            return None
    return None


def _rewrite(
    kind: str, node: ast.AST, space: MutationSpace, rng: random.Random, source: str
) -> ast.AST | None:
    if kind == "window" and isinstance(node, ast.Constant):
        pool = [w for w in space.windows_for(source) if float(w) != float(node.value)]
        return ast.Constant(float(rng.choice(pool))) if pool else None

    if kind == "operator" and isinstance(node, ast.Call):
        name = node.func.id                     # type: ignore[union-attr]
        arity = len(node.args)
        for group in (TS_WINDOW_OPS, CS_UNARY_OPS, ELEMENTWISE_UNARY_OPS, GROUP_OPS):
            if name not in group:
                continue
            # Keep the arity: a windowed operator needs its window argument.
            pool = [o for o in group if o != name]
            if not pool:
                return None
            return ast.Call(
                func=ast.Name(id=rng.choice(pool), ctx=ast.Load()),
                args=list(node.args), keywords=list(node.keywords),
            )
        return None

    if kind == "field" and isinstance(node, ast.Name):
        pool = [f for f in space.fields if f != node.id]
        return ast.Name(id=rng.choice(pool), ctx=ast.Load()) if pool else None

    if kind == "group" and isinstance(node, ast.Name):
        pool = [g for g in space.groups if g != node.id]
        return ast.Name(id=rng.choice(pool), ctx=ast.Load()) if pool else None

    return None


def crossover(
    left: str, right: str, rng: random.Random | None = None
) -> str:
    """Graft a random subexpression of ``right`` into ``left``.

    Both ends are restricted to value positions, so an operator name is never
    donated or overwritten.
    """
    rng = rng or random.Random()
    try:
        left_tree = ast.parse(left.strip(), mode="eval")
        right_tree = ast.parse(right.strip(), mode="eval")
    except SyntaxError:
        return left

    donors = [
        n for n in _value_nodes(right_tree)
        if isinstance(n, (ast.Call, ast.BinOp, ast.UnaryOp))
        or (isinstance(n, ast.Name) and n.id not in OPERATORS)
    ]
    targets = _value_nodes(left_tree, include_root=False) or _value_nodes(left_tree)
    if not donors or not targets:
        return left

    transformer = _Replace(rng.choice(targets), rng.choice(donors))
    mutated = transformer.visit(left_tree)
    if not transformer.applied:
        return left
    try:
        return simplify(_unparse(mutated.body))
    except Exception:                          # pragma: no cover - defensive
        return left


def _flip(node: ast.AST) -> ast.AST:
    """Toggle the sign of an expression.

    Stripping an existing ``-1 *`` rather than stacking another one keeps flip
    an involution; otherwise repeated flips accumulate into ``--1 * x``.
    """
    if (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Mult)
        and isinstance(node.left, ast.Constant)
        and isinstance(node.left.value, (int, float))
        and float(node.left.value) == -1.0
    ):
        return node.right
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return node.operand
    return ast.BinOp(left=ast.Constant(-1.0), op=ast.Mult(), right=node)


# Operators that are idempotent: applying one twice adds nothing.
_IDEMPOTENT = frozenset({"rank", "zscore", "normalize", "sign", "abs"})


class _Simplify(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in _IDEMPOTENT
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Call)
            and isinstance(node.args[0].func, ast.Name)
            and node.args[0].func.id == node.func.id
        ):
            return node.args[0]
        return node


def simplify(expression: str) -> str:
    """Collapse redundant structure, e.g. ``rank(rank(x))`` to ``rank(x)``.

    Mutation and crossover readily stack an idempotent operator on itself.
    The result is harmless but obscures what the alpha does, and a cleaner
    expression is also likelier to be accepted verbatim by the platform.
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        return expression
    for _ in range(4):                     # a few passes to reach a fixed point
        before = ast.dump(tree)
        tree = _Simplify().visit(tree)
        if ast.dump(tree) == before:
            break
    try:
        return _unparse(tree.body)
    except Exception:                      # pragma: no cover - defensive
        return expression


def _unparse(node: ast.AST) -> str:
    return ast.unparse(ast.fix_missing_locations(node))
