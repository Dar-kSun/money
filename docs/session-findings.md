# Findings and open threads

Written at the end of the session that built the local pipeline, for whoever
picks this up next (including a future Claude session with no memory of it).
[`STRATEGY.md`](../STRATEGY.md) has the reasoning; this has the measurements,
the mistakes, and what is still unresolved.

## Measured facts

**Correctness of the local simulator** (synthetic panels, known answers):

- A signal planted to be exploitable at delay *d* earns Sharpe ≈ 85 at exactly
  delay *d* and |Sharpe| < 2 at every other delay, for d ∈ {0, 1, 2}. The
  diagonal is clean in both directions, which pins the delay semantics and
  rules out leakage either way.
- Under AR(1) returns with φ = −0.5 the profitable side flips with the delay:
  reversion wins at delay 0 (corr = φ), momentum at delay 1 (corr = φ² > 0).
  The engine reproduces this sign flip, which is a non-obvious prediction.
- On a random walk with dollar-neutral weighting, every alpha scores |Sharpe|
  < 1 (t between −0.6 and +0.3 over 12 seeds). Nothing is found where nothing
  exists.

**Performance** (10-year history, this container):

| Panel width | sec/expression | expressions/hour |
| --- | --- | --- |
| 500 names | 0.33 | ~10,800 |
| 3000 names | 4.93 | ~730 |

So screen on a subsample and confirm finalists on the full panel.
`--max-symbols 500` is the default for `run --screen` for this reason.

**Offspring validity**: 400 consecutive mutations and crossovers all evaluated
without a single rejection, after the node-selection fixes below.

**Subsample screening preserves ranking.** Scoring the same 32 expressions on
column subsets of one 1200-name panel (autocorr −0.30, so real signal is
present):

| Subsample | Spearman vs full panel | Top-5 overlap | Speedup |
| --- | --- | --- | --- |
| 250 names | +0.922 | 3/5 | 4.2× |
| 500 names | +0.951 | 4/5 | 2.4× |

So a narrow panel is a sound *coarse* filter — the ordering largely survives —
but top-5 overlap of 3–4 out of 5 means it is not precise enough to pick the
single best candidate. **Screen wide, then re-rank the finalists on the full
panel before spending quota.**

Two caveats on how this was measured, because the first attempt got it wrong.
The subsets must be columns of *one* panel: comparing separately generated
panels of different widths conflates "fewer names" with "different data", and
on a weak-signal panel that comparison returned Spearman −0.62, which is
noise being correlated against noise rather than evidence against
subsampling. The panel also needs genuine signal, or there is no real ordering
for the subsample to agree with.

## Mistakes worth not repeating

**1. Over-generalising the −0.87 result.** The brief's finding that
`close / ts_mean(close, 20) - 1` loses money is specific to *price*. The first
implementation turned it into a universal rule — every template inverted, with
a test enforcing it — which ruled out drift on fundamental and analyst data,
the least-crowded region of the space. Direction is now a property of the field
class. Do not reinstate a global reversion rule.

**2. `rank()` is non-negative, so `neutralization="NONE"` is long-only.**
Geometric prices drift up by σ²/2 per day, so with `NONE` *every* expression
scores Sharpe ≈ +1.5 on data with no predictability — including two mutually
opposite alphas. I nearly diagnosed this as an engine bug. There is a named
test for it (`test_rank_without_neutralization_is_long_only`). Never screen
with `NONE` and read the drift as skill.

**3. An underpowered test looks like a broken engine.** My first sign test used
φ = −0.15 over ~580 days, where the standard error of an annualised Sharpe is
≈ 0.66. All six numbers were noise and I briefly read the pattern as a bug.
Check the standard error before concluding anything from a backtest number:
SE(Sharpe) ≈ √(252/T).

**4. Breadth-first node indices, depth-first replacement.** `ast.walk` is
breadth-first; `NodeTransformer` is depth-first. Selecting a node by index in
one order and replacing it in the other lands the edit on an unrelated node —
it produced `zscore(x)(x)` and replaced whole subexpressions with field names.
Replacement is now by object identity.

**5. Vectorising was slower.** Replacing the per-day loops in `rank()` and the
group reductions with whole-panel array operations gave identical results and
was *not* faster: a wash for `group_neutralize`, 3.7× slower for `group_rank`.
Those loops already do bulk numpy work per row, so the vectorised form only
traded CPU for memory bandwidth on 60 MB temporaries. It was reverted. If you
try again, measure first — and `rank()` at 3000×2520 is ~0.5s, so the ceiling
on that optimisation is small.

## Open threads, most important first

1. **Calibrate the local simulator against Brain.** Untested. Run ~20
   expressions through `wqb calibrate` and `wqb simulate`, and check the
   *rank* correlation of local Sharpe against Brain's. If it is weak, the
   screen is worthless and everything downstream degrades to the original
   grid. Nothing else matters as much as this number.
2. **Settle the `returns` definition.** It is a guess between two candidates
   (`half_book` vs `book`) and it feeds fitness. `calibrate` prints both; pick
   whichever lands closer to Brain's reported fitness and set
   `--returns-basis` from then on.
3. **Verify the live Brain API contract.** No call has ever been made against
   it from this environment — `api.worldquantbrain.com` is denied at the
   *network layer*: the proxy refuses CONNECT with a 403, so credentials
   cannot help. The hosts have to be allowed in the environment's network
   settings (session title bar → cloud environment menu → Edit → Network
   access). `python -m wqb verify` now runs the whole contract check in one
   pass and names which assumption broke. Likely mismatch sites: the `is`
   summary keys in `simulate._finalize`, the submission paths in `submit.py`,
   and the `data-fields` query parameters.

   Hosts to allow: `api.worldquantbrain.com`, `static.stooq.com`,
   `stooq.com`, `www.sec.gov`, `data.sec.gov`.
4. **Re-rank screen finalists on the full panel.** The subsample result above
   says the ordering survives but the top of it is not reliable to within a
   place or two, so `run --screen` ideally re-scores its shortlist at full
   width before choosing what to simulate. Not implemented; currently it
   simulates the subsample's top N directly.
5. **Record Brain's failed check names into the store.** The search currently
   cannot learn which regions get refused for correlation, so it may keep
   proposing them.
6. **Survivorship bias in the free price data.** Stooq's symbol list is what
   trades *today*, so losers are missing and every backtest over it is
   flattered. It biases levels more than rankings, which is tolerable for a
   screen and not for anything you would publish.

## Things that are easy to get wrong here

- Fundamental data must align on the **filing date**, not the period end. A
  quarter ending 30 September is not public until early November; aligning on
  the period end hands the backtest six weeks of impossible foresight.
- Brain checks turnover is inside roughly 1%–70%. An alpha outside that band
  is refused however good its Sharpe looks, so the local screen enforces it.
- The Sharpe threshold is strict (`> 1.25`) and the Fitness one inclusive
  (`>= 1.0`). That asymmetry is from the brief and is pinned by tests. It
  reads like a bug and is not.
- `wqb/local/evaluator.py` is a sandbox because expressions arrive from config
  files and from a language model. It walks only the AST node types an alpha
  needs and never evals text. Do not add an `eval` fast path.
