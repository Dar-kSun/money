# How to automate alpha generation

The question was whether to use LLMs or train our own model. The short answer
is **neither, as the primary engine** — and the reason is that both framings
skip the actual bottleneck.

## 1. The bottleneck is not generating expressions

It is cheap to generate a million alpha expressions. Two things are scarce:

1. **Evaluations.** BRAIN's simulation quota is limited and a simulation takes
   tens of seconds. Any search that needs thousands of evaluations is dead on
   arrival against the API.
2. **Uncorrelated ideas.** A submission is refused if it correlates with
   alphas already in the pool, and the platform reports somewhere between
   ~375,000 and 550,000 users. Price and volume are the most heavily mined
   fields on it. A `rank(-ts_zscore(close, 20))` that clears Sharpe 1.25 is
   also near-certain to be a duplicate of something already submitted.

The second point is the one that reframes everything. The practical finding
from people mining this platform is that **changing a lookback window, a
weight, or the neutralization barely moves correlation — a different data
source or a different economic mechanism is what moves it**
([wq-alpha-research](https://github.com/QuantML-Research/wq-alpha-research)).

The original brief's grid varied exactly those things: fields `close`,
`volume`, `returns` crossed with windows 5/10/20/60. That grid is a good
harness and a poor strategy. It explores the most crowded region of the
platform along the axes that least affect correlation.

So the first fix was not a better generator. It was to **make correlation a
first-class objective** and to **spend quota only on measurably underexplored
fields**.

## 2. Why not train our own model

Training a model to propose alphas needs either labelled examples
(expression → performance) or a fitness function to learn against. At the
start we had neither: zero examples, and the only fitness signal was a
quota-limited API. Reinforcement-learning approaches in the literature
([AlphaRLM](https://arxiv.org/html/2610.14959v3)) need thousands of scored
evaluations, which is precisely the resource we lack.

That was the correct answer *before* a local evaluator existed. It is now
partly obsolete — see §4.

## 3. Why not an LLM as the primary engine either

An LLM is good at reading what a data field *means* and proposing a mechanism
grounded in it. It is bad at predicting whether an expression will score; its
guess at a Sharpe carries no information. So using a model as the engine means
paying for generation and still having no way to tell which of its outputs are
any good — you are back to spending quota to find out.

The academic work converges on the same split rather than on "LLM generates
alphas": a **model proposes**, a **search allocates budget**, and a **separate
verified evaluator decides**. That is the shape of
[Alpha Jungle (LLM + MCTS)](https://arxiv.org/abs/2505.11122),
[AlphaAgent](https://arxiv.org/pdf/2502.16789) (which targets alpha decay
explicitly and reports an 81% higher effective-factor ratio), and
[AgonAlpha](https://github.com/AutoResearch-Factory/AgonAlpha) (a proposer, a
reviewer that re-runs evaluations, and a tree-search scheduler over
candidates). Reported results favour LLM-guided search over plain genetic
programming on sample efficiency and resistance to decay.

**Caveat worth stating plainly:** every one of those comparisons is run by the
authors of the method being promoted, on their own data and compute budget.
There is no independent head-to-head benchmark. Treat the architecture as
well-motivated, not as proven.

## 4. What actually unlocks this: a free evaluator

Every system above has a **local backtester**. They evaluate thousands of
candidates on their own data for free and never touch a rate-limited API. That
single difference is why their search budgets are feasible and ours was not.

So the highest-value thing to build was not a generator at all. It was
`wqb/local` — a panel of data, the FastExpr operators, a sandboxed expression
evaluator, and BRAIN's metrics:

```
fitness = sharpe * sqrt(|returns| / max(turnover, 0.125))
```

(corroborated across several independent write-ups of the platform).

With that in place the economics invert. Candidates cost microseconds instead
of quota, so search becomes worth optimising — and **genetic programming
becomes viable after all**, which is what `wqb/search/evolve.py` is. The
earlier "don't train our own" conclusion was downstream of not having a
fitness function; the local simulator is that function.

**This is also the main open risk.** The local numbers are *uncalibrated*: a
different data vendor, a different universe construction, and one metric
definition (`returns`) that is not publicly documented and which feeds
fitness. The local simulator's job is **ranking**, not reproducing BRAIN. See
§7.

## 5. The architecture

```
  free, unlimited                                    quota-limited
  ─────────────────────────────────────────────      ──────────────
  field selection ──► proposal ──► local search ──►  BRAIN sim ──► submit
  (crowding-ranked)   (templates   (evolve, with     (confirm      (checks,
                       + Claude)    novelty          the best        then --yes)
                                    discount)        few)
```

**Field selection** (`wqb/fields.py`) ranks fields by how heavily mined they
are, using `alphaCount`/`userCount` from the data-fields endpoint, and
classifies each into an economic class that decides the signal direction.

**Direction is class-dependent**, which was a real bug in the first version.
The `-0.87` momentum result is specific to *price*: short-horizon reversal
dominates liquid equities at D1. Fundamental and analyst data does the
opposite — post-earnings-announcement drift and analyst-revision momentum are
among the better-documented effects in finance. The first implementation
forced inverted mean-reversion on every template and had a test enforcing it,
which ruled out the least-crowded and most promising region of the space. The
same algebra is now a failure mode on `close` and a legitimate signal on
`anl_eps_revision`.

**Proposal** is both template expansion and `wqb/propose.py`, which asks Claude
for mechanisms grounded in field descriptions. Its output is validated and
locally evaluated before it can reach the API.

**Local search** (`wqb/search/`) mutates expressions by rewriting their syntax
tree and scores each candidate on **local fitness discounted by correlation
with what has already been found**. That discount is the whole point: plain
fitness maximisation reliably returns ten variants of one idea, and ten
correlated alphas are one submission.

**BRAIN** is touched only at the end, to confirm the few survivors.

## 6. Does this make money?

Honestly: it is a real but modest and competitive income path, not a quick win.

Compensation is merit-based and tiered. Third-party reporting of WorldQuant's
consultant page puts Grandmaster potential around $8,000/quarter and Master
around $2,000/quarter, with consultant invitations gated behind roughly 10,000
points and Gold status, and explicitly not guaranteed
([quantt.co.uk review](https://www.quantt.co.uk/resources/worldquant-brain),
[consultant page](https://worldquantbrain.com/consultant)). I could not verify
those figures against WorldQuant's own page, so treat them as indicative.

What that means for the plan: the payoff scales with **accepted, uncorrelated**
alphas, not with alphas simulated. A pipeline that produces 500 correlated
price alphas earns nothing. One that produces five genuinely distinct
fundamental alphas a month is on the track that pays. Every design choice above
follows from that.

## 7. What I am least sure about

In rough order of how much it would change things:

1. **The local simulator is uncalibrated.** If local Sharpe does not
   rank-correlate with BRAIN's, the screen is worthless and the whole pipeline
   degrades to the original grid. **This is the first thing to measure** —
   `python -m wqb calibrate` exists for it. Run ~20 expressions both locally
   and on BRAIN and check the rank correlation.
2. **The `returns` definition** feeds fitness and is a guess between two
   candidates. `calibrate` prints both; pick the one matching BRAIN.
3. **The BRAIN API contract is still unverified** — no call has been made
   against the live service from here, because the host is blocked by this
   environment's network policy.
4. **`alphaCount`/`userCount`** as crowding signals come from community tooling
   and are not confirmed against the live API.
5. **Structural diversity is a proxy for correlation**, not a measurement. Two
   different families can still correlate; only BRAIN's check settles it.
6. **Free price data is split-adjusted-ish at best.** Survivorship bias is real:
   Stooq's current symbol list excludes delisted names, which flatters any
   backtest. It biases levels more than rankings, which is tolerable for a
   screen and not for a published result.

## 8. Recommended next steps

1. `python -m wqb fetch prices` and run `calibrate` on ~20 expressions against
   BRAIN. **Do not skip this**; everything downstream rests on it.
2. `python -m wqb fetch fundamentals` — this is the uncrowded region, and the
   reason to bother with SEC data at all.
3. `python -m wqb search` over fundamental and analyst fields, then confirm the
   top few with `python -m wqb simulate`.
4. Only then add `propose` to the loop, so a model's suggestions are judged by
   a screen you have already shown to work.
