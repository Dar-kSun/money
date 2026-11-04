# CLAUDE.md

Automated alpha discovery for **WorldQuant Brain**: generate candidate alphas,
simulate them against the Brain API, grade them, and submit the ones that pass.

The end goal is a portfolio of submitted alphas clearing **Sharpe > 1.25** and
**Fitness >= 1.0**.

## Status: the harness exists — extend it, don't rebuild it

The `wqb` package already implements authentication, generation, simulation,
grading, persistence and the submission workflow. Read the relevant module
before writing new code; most tasks are a change inside one of these, not a new
script.

| Module | Role |
| --- | --- |
| `wqb/config.py` | Credentials, the fixed simulation settings block, run knobs |
| `wqb/client.py` | `BrainClient`: auth, token renewal, backoff, pagination |
| `wqb/generate.py` | Mean-reversion templates and the expression grid |
| `wqb/simulate.py` | Submit one simulation, poll it to an in-sample verdict |
| `wqb/evaluate.py` | The Sharpe/Fitness gate plus Brain's own IS checks |
| `wqb/runner.py` | The concurrent, resumable discovery loop |
| `wqb/store.py` | Thread-safe SQLite results store and JSONL success log |
| `wqb/submit.py` | Pre-submission checks and the gated OS submission |
| `wqb/cli.py` | `python -m wqb` subcommands |

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
python -m wqb templates --expressions
```

```bash
python -m wqb run --dry-run --fields close volume returns --windows 5 10 20
```

```bash
python -m wqb run --fields close volume returns vwap --windows 5 10 20 60 --concurrency 3
```

```bash
python -m wqb results --top 20
```

```bash
python -m wqb submit --all          # checks only; --yes to actually submit
```

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
* **Tests never hit the network.** They use `responses` for HTTP and the
  `no_sleep` fixture to make backoff instant. Keep it that way so the suite stays
  fast and runnable without credentials.

## Known gap: the live API contract is unverified

The client was written against the documented API, but **no call has been made
against the live service** — the environment it was built in blocked
`api.worldquantbrain.com`. Endpoint paths and response field names are therefore
unconfirmed.

Validate cheaply before any large run, in this order:

```bash
python -m wqb auth
```

```bash
python -m wqb fields --limit 5
```

```bash
python -m wqb simulate 'rank(-1 * ts_zscore(close, 20))'
```

If something is wrong it will surface here. Most likely spots:

* the in-sample summary key (`is`) and its metric names (`sharpe`, `fitness`,
  `turnover`), read in `simulate._finalize`
* the submission check path, assumed to be `GET /alphas/{id}/check` then
  `POST /alphas/{id}/submit`, in `submit.py`
* `data-fields` / `data-sets` query parameters in `client.py`

When a real response disagrees with the code, fix the code and add a test with
the **actual** payload — the mocked fixtures are only as good as their source.

## Extending

**A new alpha template** — add a `Template` to `TEMPLATES` in `generate.py` with
a one-line rationale. It must be mean-reverting and ranked; the
`test_every_template_is_mean_reverting` test enforces this. Then
`python -m wqb run --dry-run --templates <name>` to eyeball the rendered
expressions before spending quota.

**New data fields** — discover them with `python -m wqb fields --dataset <id>`
and pass them to `run --fields`. The built-in `DEFAULT_FIELDS` list is only
price/volume; fundamental datasets are where less-crowded alphas tend to be.

**Operators used so far**: `rank`, `ts_mean`, `ts_zscore`, `ts_delta`, `ts_rank`,
`ts_min`, `ts_max`, `ts_std_dev`, `decay_linear`, `group_neutralize`. Operator
availability varies by region and delay — if a simulation returns an "unknown
operator" error, that is the cause, not a syntax bug.

## Suggested next steps

1. **Validate against the live API** (above) and correct any contract mismatch.
2. **One full baseline run** over price/volume fields, then read
   `python -m wqb results --top 20` to see which templates and windows actually
   score, rather than widening the search blindly.
3. **Expand into fundamental datasets** — the price/volume grid is heavily
   crowded, and self-correlation with existing alphas is the most common
   submission blocker.
4. **Handle correlation explicitly.** Brain rejects alphas correlated with the
   user's existing submissions. Worth recording the failed check names from
   `submit` into the store so the search can learn to avoid that region.
5. **Only then consider auto-submission** of pre-check-passing alphas, keeping
   the explicit opt-in.

## Caveats worth keeping in mind

* The thresholds are **in-sample**. The grid tests hundreds of combinations, so
  some will clear 1.25 by chance alone; a pass is a gate for submission, not
  evidence an alpha will hold up out-of-sample. `data/alphas.db` keeps every
  attempt, so the true hit rate is always checkable.
* Brain throttles concurrent simulations. `--concurrency 3` is a safe default and
  the client backs off on 429 automatically.
* If Brain requires a biometric/persona check, the CLI reports it with the URL to
  complete in a browser — it is not a credentials problem.
