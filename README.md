# money — WorldQuant Brain alpha automation

A CLI that authenticates against the Brain API, generates mean-reversion alphas
for **USA / D1 / TOP3000**, simulates them, grades them on **Sharpe > 1.25** and
**Fitness >= 1.0**, and runs the out-of-sample submission checks.

[`CLAUDE.md`](CLAUDE.md) holds the strategy context, conventions and open
questions — read it before changing how alphas are generated or graded.

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
| `wqb/config.py` | Credentials, the Phase 3 settings block, run knobs |
| `wqb/client.py` | Authenticated session, re-auth, `Retry-After` backoff, pagination |
| `wqb/generate.py` | Mean-reversion templates and the expression grid |
| `wqb/simulate.py` | Submit a simulation and poll it to an IS verdict |
| `wqb/evaluate.py` | The Sharpe/Fitness gate plus Brain's own IS checks |
| `wqb/runner.py` | The Phase 4 loop, concurrent and resumable |
| `wqb/store.py` | Thread-safe SQLite results store and success log |
| `wqb/submit.py` | Pre-submission checks and the gated OS submission |
| `wqb/cli.py` | Argument parsing and the subcommands |

## Tests

```bash
python -m pytest
```

55 tests cover the API contract against mocked HTTP: the 201 auth handshake,
biometric-required detection, re-auth on an expired token, 429/5xx backoff,
`Retry-After` polling, the exact Phase 3 payload, both threshold boundaries
(Sharpe is strict, Fitness is inclusive), concurrent store writes, and the
guarantee that a dry-run submission never POSTs.

## Notes

* The thresholds are in-sample. Passing them is a necessary condition for
  submission, not a prediction of out-of-sample performance.
* Brain throttles concurrent simulations; `--concurrency 3` is a safe default
  and the client backs off on 429 automatically.
* If Brain asks for a biometric/persona check, the CLI says so and gives the
  URL — complete it in a browser, then re-run.
