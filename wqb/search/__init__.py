"""Search over expression space, using the local simulator as fitness.

With a free evaluator the search can afford thousands of candidates, so it
becomes worth optimising rather than enumerating. Two pieces:

* :mod:`wqb.search.mutate` edits expressions structurally, by parsing them and
  rewriting the syntax tree.
* :mod:`wqb.search.evolve` runs a population over those edits, scoring each
  candidate on local fitness *discounted by how correlated it is with what has
  already been found*.

That discount is the point. Raw fitness alone converges on many variants of
one idea, and the platform refuses correlated submissions, so an uncorrelated
alpha of middling fitness is worth more than a tenth copy of the best one.
"""

from .evolve import EvolveConfig, Found, evolve
from .mutate import crossover, mutate, mutation_space

__all__ = [
    "EvolveConfig", "Found", "crossover", "evolve", "mutate", "mutation_space",
]
