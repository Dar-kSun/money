"""Evolutionary search over expressions, scored by the local simulator.

The objective is not raw fitness. Correlated alphas are refused at submission,
so a candidate's score is its local fitness *discounted by how correlated its
pnl is with the alphas already found*. An uncorrelated alpha of middling
fitness is worth more than another copy of the best one, and plain fitness
maximisation reliably produces the latter.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

from ..diversity import family
from ..generate import validate_expression
from ..local.engine import Metrics, SimConfig, correlation, simulate
from ..local.evaluator import ExpressionError, evaluate_expression
from ..local.panel import Panel
from .mutate import MutationSpace, crossover, mutate

log = logging.getLogger(__name__)


@dataclass
class EvolveConfig:
    population: int = 40
    generations: int = 15
    elite: int = 8
    crossover_rate: float = 0.3
    min_sharpe: float = 1.25
    min_fitness: float = 1.0
    # Above this pnl correlation with something already found, a candidate is
    # treated as the same alpha and not kept.
    max_correlation: float = 0.7
    novelty_power: float = 1.0
    max_book: int = 25
    # Brain requires turnover inside roughly 1%-70%. Screening without this
    # produces alphas that look excellent locally and are refused on sight.
    min_turnover: float = 0.01
    max_turnover: float = 0.70
    seed: int = 0


@dataclass
class Found:
    expression: str
    metrics: Metrics
    score: float
    correlation_to_book: float
    generation: int

    def summary(self) -> str:
        return (
            f"sharpe={self.metrics.sharpe:+.2f} fitness={self.metrics.fitness:.2f} "
            f"turnover={self.metrics.turnover:.3f} corr={self.correlation_to_book:.2f}"
        )


@dataclass
class _Scored:
    expression: str
    metrics: Metrics | None
    score: float
    error: str = ""


ProgressFn = Callable[[int, list[Found], "_Scored"], None]


def evolve(
    panel: Panel,
    seeds: Sequence[str],
    space: MutationSpace,
    sim_config: SimConfig | None = None,
    config: EvolveConfig | None = None,
    *,
    progress: ProgressFn | None = None,
) -> list[Found]:
    """Search for a set of mutually uncorrelated, locally strong alphas.

    Returns the accepted book, best first. Every candidate is evaluated
    locally, so this costs no Brain quota.
    """
    sim_config = sim_config or SimConfig()
    config = config or EvolveConfig()
    rng = random.Random(config.seed)

    book: list[Found] = []
    cache: dict[str, _Scored] = {}
    population = _seed_population(seeds, space, rng, config)

    for generation in range(config.generations):
        scored: list[_Scored] = []
        for expression in population:
            result = _score(expression, panel, sim_config, book, config, cache)
            scored.append(result)
            if progress:
                progress(generation, book, result)
            _maybe_accept(result, book, config, generation, panel, sim_config)

        scored.sort(key=lambda s: s.score, reverse=True)
        if len(book) >= config.max_book:
            break
        population = _breed(scored, space, rng, config)

    book.sort(key=lambda f: f.score, reverse=True)
    return book


def _seed_population(
    seeds: Sequence[str], space: MutationSpace, rng: random.Random, config: EvolveConfig
) -> list[str]:
    population = [s for s in dict.fromkeys(seeds) if s.strip()]
    if not population:
        raise ValueError("evolve() needs at least one seed expression")
    while len(population) < config.population:
        parent = rng.choice(population[: max(1, len(seeds))])
        population.append(mutate(parent, space, rng))
    return population[: config.population]


def _score(
    expression: str,
    panel: Panel,
    sim_config: SimConfig,
    book: list[Found],
    config: EvolveConfig,
    cache: dict[str, _Scored],
) -> _Scored:
    """Local fitness, discounted by correlation with the current book."""
    key = expression.strip()
    if key in cache:
        cached = cache[key]
        # The penalty depends on the book, which grows, so re-derive the score.
        if cached.metrics is None:
            return cached
        return _Scored(
            expression=key,
            metrics=cached.metrics,
            score=_adjusted(cached.metrics, book, config),
        )

    problems = validate_expression(key)
    if problems:
        result = _Scored(key, None, float("-inf"), "; ".join(problems))
        cache[key] = result
        return result

    try:
        signal = evaluate_expression(key, panel)
    except ExpressionError as exc:
        result = _Scored(key, None, float("-inf"), str(exc))
        cache[key] = result
        return result

    if not np.isfinite(signal).any():
        result = _Scored(key, None, float("-inf"), "signal is entirely NaN")
        cache[key] = result
        return result

    metrics = simulate(key, panel, sim_config, signal=signal)
    cache[key] = _Scored(key, metrics, 0.0)
    return _Scored(key, metrics, _adjusted(metrics, book, config))


def _adjusted(metrics: Metrics, book: list[Found], config: EvolveConfig) -> float:
    base = float(metrics.fitness)
    if not np.isfinite(base):
        return float("-inf")
    novelty = 1.0 - _max_correlation(metrics, book)
    return base * max(novelty, 0.0) ** config.novelty_power


def _max_correlation(metrics: Metrics, book: list[Found]) -> float:
    if not book:
        return 0.0
    return max(abs(correlation(metrics, found.metrics)) for found in book)


def _maybe_accept(
    result: _Scored,
    book: list[Found],
    config: EvolveConfig,
    generation: int,
    panel: Panel,
    sim_config: SimConfig,
) -> None:
    metrics = result.metrics
    if metrics is None or len(book) >= config.max_book:
        return
    if metrics.sharpe <= config.min_sharpe or metrics.fitness < config.min_fitness:
        return
    if not config.min_turnover <= metrics.turnover <= config.max_turnover:
        return

    fam = family(result.expression)
    if any(family(found.expression) == fam for found in book):
        return

    corr = _max_correlation(metrics, book)
    if corr > config.max_correlation:
        return

    book.append(Found(
        expression=result.expression,
        metrics=metrics,
        score=result.score,
        correlation_to_book=corr,
        generation=generation,
    ))
    log.info("accepted %s (%s)", result.expression, book[-1].summary())


def _breed(
    scored: list[_Scored], space: MutationSpace, rng: random.Random, config: EvolveConfig
) -> list[str]:
    viable = [s for s in scored if np.isfinite(s.score)]
    if not viable:
        # Everything failed to evaluate; restart from whatever parsed.
        viable = scored[: config.elite]
    elite = [s.expression for s in viable[: max(1, config.elite)]]

    children: list[str] = list(elite)
    families = {family(e) for e in elite}
    attempts = 0

    while len(children) < config.population and attempts < config.population * 10:
        attempts += 1
        parent = rng.choice(elite)
        if len(elite) > 1 and rng.random() < config.crossover_rate:
            child = crossover(parent, rng.choice(elite), rng)
        else:
            child = mutate(parent, space, rng)

        fam = family(child)
        if fam in families:
            continue
        families.add(fam)
        children.append(child)

    while len(children) < config.population:
        children.append(mutate(rng.choice(elite), space, rng))
    return children[: config.population]
