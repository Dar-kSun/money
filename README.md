# money — WorldQuant Brain alpha automation

A CLI that discovers alpha expressions for **USA / D1 / TOP3000**, grades them
on **Sharpe > 1.25** and **Fitness >= 1.0**, and runs the out-of-sample
submission checks.

It has two halves. A **local** half that screens thousands of candidates for
free, and a **remote** half that spends Brain's limited simulation quota only
on the survivors.

- [`STRATEGY.md`](STRATEGY.md) — why the pipeline is shaped this way, and the
  answer to "LLMs or train our own?"
- [`CLAUDE.md`](CLAUDE.md) — working conventions and open questions.
- [`docs/data-sources.md`](docs/data-sources.md) — getting free data.

## Quick start

```bash
pip install -r requirements-dev.txt
```

```bash
python -m wqb screen --synthetic --top 10
```

That runs the whole local pipeline on generated data, with no credentials and
no downloads, to check the install. Then fetch real data:

```bash
python -m wqb fetch prices --dest data/prices
```

```bash
python -m wqb search --data data/prices
```

Both are free. Only `auth`, `verify`, `simulate`, `run`, `fields` and `submit`
talk to Brain.

`python -m wqb propose` (Claude suggests alpha mechanisms) additionally needs an
optional dependency and an Anthropic key — everything else works without both:

```bash
pip install -r requirements-optional.txt
```

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
```

```bash
pip install -r requirements-dev.txt
```

```bash
cp .env.example .env
```

Put your Brain credentials in `.env` (it is gitignored). `WQB_EMAIL` and
`WQB_PASSWORD` are read from the environment or from `.env`; they are never
written to the results database or the logs.

## Use

Verify credentials:

```bash
python -m wqb auth
```

See the alpha templates and what each one is for:

```bash
python -m wqb templates --expressions
```

Preview the expression grid without spending simulation quota:

```bash
python -m wqb run --dry-run --fields close volume returns --windows 5 10 20
```

Run the discovery loop (three simulations at a time):

```bash
python -m wqb run --fields close volume returns vwap --windows 5 10 20 60 --concurrency 3
```

Inspect what has been found:

```bash
python -m wqb results --top 20
```

Discover fundamental data fields to widen the search:

```bash
python -m wqb fields --dataset fundamental6 --limit 40
```

Simulate a single expression:

```bash
python -m wqb simulate 'rank(-1 * ts_zscore(close, 20))'
```

## Submission is opt-in

Submitting an alpha is irreversible and consumes a submission slot, so
`wqb submit` **only runs Brain's pre-submission checks** by default and prints
what would happen:

```bash
python -m wqb submit --all
```

Add `--yes` to actually submit. Even then, an alpha whose pre-checks fail is
never sent, so a slot is not wasted on a rejection:

```bash
python -m wqb submit --all --yes
```

## Why the alphas look the way they do

The context file records that in this environment raw momentum loses money:
`close / ts_mean(close, 20) - 1` scored a Sharpe of about **-0.87**, because it
buys price spikes in a mean-reverting market. Every template here is therefore
inverted and cross-sectionally ranked, following the validated blueprint:

```
rank(-1 * (close / ts_mean(close, 20) - 1))
```

* `-1 *` buys stocks trading **below** their moving average and shorts those above.
* `rank()` forces a market-neutral long/short spread across the 3000-name universe.

`wqb simulate` warns if an expression you pass by hand reproduces that
documented failure mode. The eight templates vary the reversion measure
(moving-average ratio, z-score, delta, trailing rank, range position, decayed,
volatility-scaled, subindustry-relative) across the fields and lookback windows
you choose.

Simulation settings are fixed to the Phase 3 values — `decay: 5` matters, since
cutting turnover is what lifts Fitness above 1.0 — and each is overridable
(`--decay`, `--universe`, `--neutralization`, `--truncation`, ...).

## Results and resuming

Every attempt is written to `data/alphas.db`, and passing alphas are also
appended to `data/successful_alphas.jsonl`. A re-run skips expressions already
simulated, so an interrupted run resumes without spending quota twice; use
`--no-resume` to force re-simulation.

## Layout

| File | Role |
| --- | --- |
| `wqb/local/` | The free simulator: panel, operators, evaluator, metrics, data |
| `wqb/search/` | Expression mutation and population search |
| `wqb/fields.py` | Field class, direction prior, crowding rank |
| `wqb/diversity.py` | Family signatures, used as a correlation proxy |
| `wqb/generate.py` | Direction-aware templates and the expression grid |
| `wqb/propose.py` | Claude proposes mechanisms; every output is validated |
| `wqb/client.py` | Authenticated Brain session, re-auth, backoff, pagination |
| `wqb/simulate.py` | Submit a simulation and poll it to an IS verdict |
| `wqb/evaluate.py` | The Sharpe/Fitness gate plus Brain's own IS checks |
| `wqb/runner.py` | The concurrent, resumable remote loop |
| `wqb/store.py` | Thread-safe SQLite results store and success log |
| `wqb/submit.py` | Pre-submission checks and the gated OS submission |
| `wqb/cli.py`, `wqb/localcli.py` | The subcommands |

## Tests

```bash
python -m pytest
```

265 tests, none of which touch the network. They cover the API contract
against mocked HTTP (auth handshake, token renewal, backoff, `Retry-After`
polling, the exact simulation payload), both threshold boundaries, concurrent
store writes, and the guarantee that a dry-run submission never POSTs.

The local simulator is verified against synthetic panels with known answers,
because real data cannot tell you whether a backtester has a sign flip or an
off-by-one that leaks the future:

* A signal planted to pay off at one delay earns a Sharpe above 20 at exactly
  that delay and under 2 at every other one, in both directions.
* Under AR(1) returns the profitable side flips with the delay (a delay-1 alpha
  can only capture phi-squared), and the engine reproduces that.
* On a random walk nothing scores — the look-ahead guard.
* The search finds a planted signal it was not seeded with, and returns an
  empty book when run on unpredictable data.

## Notes

* The thresholds are in-sample. Passing them is a necessary condition for
  submission, not a prediction of out-of-sample performance.
* Brain throttles concurrent simulations; `--concurrency 3` is a safe default
  and the client backs off on 429 automatically.
* If Brain asks for a biometric/persona check, the CLI says so and gives the
  URL — complete it in a browser, then re-run.
