# CLAUDE.md

Automated alpha discovery for **WorldQuant Brain**: generate candidate alphas,
simulate them against the Brain API, grade them, and submit the ones that pass.

The end goal is a portfolio of submitted alphas clearing **Sharpe > 1.25** and
**Fitness >= 1.0**.

## Status: the harness exists — extend it, don't rebuild it

The `wqb` package already implements authentication, generation, local
simulation, search, grading, persistence and the submission workflow. Read the
relevant module before writing new code; most tasks are a change inside one of
these, not a new script.

**Read [`STRATEGY.md`](STRATEGY.md) first.** It explains why the pipeline is
shaped the way it is — in particular why correlation, not expression volume,
is the binding constraint, and why the search runs locally before it touches
Brain.

**Remote side** (spends Brain quota):

| Module | Role |
| --- | --- |
| `wqb/config.py` | Credentials, the fixed simulation settings block, run knobs |
| `wqb/client.py` | `BrainClient`: auth, token renewal, backoff, pagination |
| `wqb/simulate.py` | Submit one simulation, poll it to an in-sample verdict |
| `wqb/evaluate.py` | The Sharpe/Fitness gate plus Brain's own IS checks |
| `wqb/runner.py` | The concurrent, resumable discovery loop |
| `wqb/store.py` | Thread-safe SQLite results store and JSONL success log |
| `wqb/submit.py` | Pre-submission checks and the gated OS submission |

**Local side** (free and unlimited — do the work here first):

| Module | Role |
| --- | --- |
| `wqb/local/panel.py` | Aligned (dates x instruments) data, with a per-day universe |
| `wqb/local/operators.py` | FastExpr operator implementations over numpy |
| `wqb/local/evaluator.py` | Sandboxed expression parsing and evaluation |
| `wqb/local/engine.py` | Positions and Brain-shaped metrics; `correlation()` |
| `wqb/local/data.py` | Synthetic panels, CSV loading, fundamental merge |
| `wqb/local/fetch.py` | Free data: Stooq prices, SEC EDGAR fundamentals |
| `wqb/search/mutate.py` | Structural edits to an expression via its syntax tree |
| `wqb/search/evolve.py` | Population search scored on fitness x novelty |
| `wqb/fields.py` | Field class, direction prior, and crowding rank |
| `wqb/diversity.py` | Family/skeleton signatures as a correlation proxy |
| `wqb/generate.py` | Direction-aware templates and the expression grid |
| `wqb/propose.py` | Claude proposes mechanisms; output is validated |
| `wqb/cli.py`, `wqb/localcli.py` | `python -m wqb` subcommands |

**All HTTP goes through `BrainClient`.** It already handles re-auth on an
expired token, `Retry-After`, and 429/5xx backoff. Do not create a bare
`requests.Session` or re-roll the auth handshake in new code.

## Commands

```bash
pip install -r requirements-dev.txt
```

```bash
cp .env.example .env   # then fill in WQB_EMAIL / WQB_PASSWORD
```

```bash
python -m pytest
```

```bash
python -m wqb auth
```

```bash
python -m wqb verify --no-simulate   # check the live API contract, spends no simulation
```

```bash
python -m wqb templates --expressions
```

```bash
python -m wqb run --dry-run --fields close volume returns --windows 5 10 20
```

```bash
python -m wqb run --fields close volume returns vwap --windows 5 10 20 60 --concurrency 3
```

```bash
python -m wqb run --screen --data data/prices --screen-top 20   # screen locally, simulate the best
```

```bash
python -m wqb results --top 20
```

```bash
python -m wqb submit --all          # checks only; --yes to actually submit
```

### The local pipeline — free, no credentials, no quota

```bash
python -m wqb fetch prices --dest data/prices
```

```bash
python -m wqb fetch fundamentals --symbols-from data/prices --dest data/fundamentals --contact you@example.com
```

```bash
python -m wqb screen --data data/prices --top 25
```

```bash
python -m wqb search --data data/prices --fundamentals data/fundamentals --generations 20
```

```bash
python -m wqb calibrate --data data/prices 'rank(-1 * ts_zscore(close, 20))'
```

Add `--synthetic` to any of these to run on generated data without downloading
anything; the results are not alphas and the command says so. See
[`docs/data-sources.md`](docs/data-sources.md).

## Domain knowledge (the part that is expensive to rediscover)

Target environment: **USA / D1 (1-day delay) / TOP3000**.

In this environment **raw momentum loses money.** This expression was tested and
scored a Sharpe of about **-0.87**:

```
close / ts_mean(close, 20) - 1
```

It fails because it buys price spikes in a market that mean-reverts. Do not
propose variations of it; `generate.looks_like_naive_momentum()` exists to catch
this shape, and `wqb simulate` warns when a hand-written expression matches it.

**This lesson is specific to price.** Short-horizon reversal dominates liquid
equities at D1, but slow data does the opposite: post-earnings-announcement
drift and analyst-revision momentum are well documented. So the *same algebra*
is a failure mode on `close` and a legitimate signal on `anl_eps_revision`.
`wqb/fields.py` assigns each field a class and a direction prior, and templates
declare which classes they apply to — price and volume draw only reversion,
fundamental and analyst only drift, and the uncertain classes draw both. An
earlier version forced reversion everywhere and had a test enforcing it, which
ruled out the least-crowded region of the search space. Do not reinstate that.

The working blueprint:

```
rank(-1 * (close / ts_mean(close, 20) - 1))
```

Two changes make the difference, and both must survive any edit:

* **Inversion (`-1 *`)** — buys stocks trading *below* their moving average and
  shorts those above it.
* **Cross-sectional `rank()`** — forces a market-neutral long/short spread
  across the 3000-name universe.

`decay: 5` in the settings matters independently: cutting turnover is what lifts
Fitness above 1.0. Values of 4–5 are the useful range.

## Simulation settings

Fixed in `config.SimSettings` and overridable per run via CLI flags. Changing a
default here changes every result's comparability, so prefer a flag:

```
type: REGULAR
instrumentType: EQUITY   region: USA        universe: TOP3000
delay: 1                 decay: 5           neutralization: SUBINDUSTRY
truncation: 0.05
```

## Conventions

* **Thresholds are asymmetric and deliberate**: Sharpe is strict (`> 1.25`),
  Fitness is inclusive (`>= 1.0`). Both boundaries are pinned by tests — this
  reads like an inconsistency and must not be "tidied".
* **Brain's own IS checks are part of the verdict.** A `FAIL` there blocks
  submission regardless of Sharpe and Fitness.
* **Credentials come from the environment or `.env`** (gitignored). Never commit
  them, never log them, never write them into the results DB.
* **Submission is opt-in.** It is irreversible and consumes a submission slot.
  `submit_alpha()` defaults to `confirm=False` and only runs pre-checks; the CLI
  requires `--yes`. Do not add an auto-submit path that bypasses this, and keep
  the rule that an alpha failing pre-checks is never POSTed.
* **The store is thread-safe on purpose.** The loop writes from worker threads,
  so the SQLite connection uses `check_same_thread=False` behind a lock. Writing
  to the DB directly from a new thread without going through `ResultStore` will
  break. Also note `requests.Session` is not thread-safe — `runner.clone_client`
  gives each worker its own session sharing the auth cookies.
* **Resume is quota protection.** Every attempt is recorded, and a re-run skips
  expressions already simulated. Avoid `--no-resume` unless re-simulation is the
  point.
* **Tests never hit the network.** They use `responses` for HTTP, injected fake
  clients for the proposer, and the `no_sleep` fixture to make backoff instant.
  Keep it that way so the suite stays fast and runnable without credentials.
* **Screen locally before spending quota.** A Brain simulation is the scarce
  resource. Anything that can be decided from `wqb/local` should be, and new
  search ideas belong on the local side first.
* **Correlation is an objective, not a final check.** The search scores
  candidates on local fitness *discounted by* correlation with what it has
  already found, because ten correlated alphas are one submission. Do not
  "simplify" that back to plain fitness maximisation.
* **The expression evaluator is a sandbox.** Expressions arrive from config
  files and from a language model, so `wqb/local/evaluator.py` walks only the
  AST node types an alpha needs and never evals text. Do not add attribute
  access, subscripting, or a general `eval` fast path.
* **Anything a model returns is untrusted.** `wqb/propose.py` validates,
  field-checks, locally evaluates and de-duplicates every proposal before it
  can reach the API.
* **Fundamentals key off the filing date.** The period end is weeks before the
  number is public; aligning on it leaks information nobody could trade on.
* **Local metrics are uncalibrated.** They are for *ranking*. `returns` is a
  genuine unknown between two definitions and it feeds fitness — see
  `wqb/local/engine.py` and the `calibrate` command.
* **`rank()` is non-negative, so `neutralization="NONE"` gives a long-only
  book** that scores a positive Sharpe on pure noise. There is a test naming
  this; do not screen with `NONE` and read the drift as skill.

## Known gap: the live API contract is unverified

The client was written against the documented API, but **no call has been made
against the live service** — the environment it was built in blocked
`api.worldquantbrain.com`. Endpoint paths and response field names are therefore
unconfirmed.

`api.worldquantbrain.com` is denied at the *network layer* in that
environment — the proxy refuses CONNECT with a 403, so no credential changes
it; the host has to be allowed in the cloud environment's network settings.
On your own machine there is no such restriction.

One command runs every contract check and names which assumption broke:

```bash
python -m wqb verify --no-simulate
```

```bash
python -m wqb verify --compare-local --data data/prices
```

The first checks authentication and `data-fields` without spending a
simulation. The second runs the full path and prints Brain's metrics beside the
local simulator's. Most likely mismatch spots:

* the in-sample summary key (`is`) and its metric names (`sharpe`, `fitness`,
  `turnover`), read in `simulate._finalize`
* the submission check path, assumed to be `GET /alphas/{id}/check` then
  `POST /alphas/{id}/submit`, in `submit.py`
* `data-fields` / `data-sets` query parameters in `client.py`

When a real response disagrees with the code, fix the code and add a test with
the **actual** payload — the mocked fixtures are only as good as their source.

## Extending

**A new alpha template** — add a `Template` to `REVERSION_TEMPLATES` or
`DRIFT_TEMPLATES` in `generate.py`, with a one-line rationale, a `Direction`,
and the set of field classes it applies to. The invariant the tests enforce is
that every template is **cross-sectionally normalised**
(`test_every_template_is_cross_sectionally_normalised`) — *not* that it is
mean-reverting. Two further tests check that price/volume only ever draw
reversion templates and fundamental/analyst only drift. Then:

```bash
python -m wqb run --dry-run --templates <name>
```

**New data fields** — `python -m wqb fields --dataset <id>` lists them with
coverage; `wqb/fields.py` ranks them by crowding. The built-in `DEFAULT_FIELDS`
is only price/volume, which is the most mined region of the platform.
Fundamental and analyst datasets are where uncorrelated alphas remain.

**A new operator** — implement it in `wqb/local/operators.py` and register it in
`OPERATORS`, or the local simulator cannot evaluate expressions that use it and
the search will discard them. `wqb/propose.py` builds its operator list from
that same registry, so a model is never told about an operator we cannot score.

**Operator availability on Brain varies by region and delay.** If a *remote*
simulation returns "unknown operator", that is the cause, not a syntax bug —
and note it may be an operator the local simulator happily evaluates.

## Suggested next steps

1. **Calibrate the local simulator.** Run ~20 expressions through
   `python -m wqb calibrate` and the same ones through `python -m wqb simulate`,
   and check that local Sharpe *rank-correlates* with Brain's. Everything
   downstream rests on this, and it also settles the `returns` definition. If
   the ranks do not correlate, fix that before building anything else.
2. **Validate the live API contract** (below) while doing it.
3. **Fetch fundamentals and search there.** Price/volume is the most mined
   region of the platform; fundamental and analyst fields are where
   uncorrelated alphas remain.
4. **Record Brain's failed check names into the store**, so the search can
   learn which regions get refused for correlation rather than rediscovering
   them.
5. **Then add `propose` to the loop**, so a model's suggestions are judged by a
   screen already shown to work.

## Caveats worth keeping in mind

* The thresholds are **in-sample**. The grid tests hundreds of combinations, so
  some will clear 1.25 by chance alone; a pass is a gate for submission, not
  evidence an alpha will hold up out-of-sample. `data/alphas.db` keeps every
  attempt, so the true hit rate is always checkable.
* Brain throttles concurrent simulations. `--concurrency 3` is a safe default and
  the client backs off on 429 automatically.
* If Brain requires a biometric/persona check, the CLI reports it with the URL to
  complete in a browser — it is not a credentials problem.
