"""Command line entry point: ``python -m wqb ...``"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from .client import AuthError, BiometricRequired, BrainClient, BrainError
from .config import ConfigError, Credentials, RunConfig, SimSettings
from .evaluate import evaluate
from .generate import (
    DEFAULT_FIELDS,
    DEFAULT_WINDOWS,
    TEMPLATES,
    Candidate,
    generate,
    looks_like_naive_momentum,
    resolve_templates,
)
from .localcli import HANDLERS as LOCAL_HANDLERS
from .localcli import register as register_local
from .runner import run_batch
from .simulate import run_simulation
from .store import ResultStore
from .submit import check_submission, submit_alpha


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader so credentials need not be exported by hand."""
    env_path = Path(path)
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wqb",
        description="Automate WorldQuant Brain alpha discovery (USA / D1 / TOP3000).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.add_argument("--db", default=None, help="path to the results SQLite file")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("auth", help="verify credentials against the Brain API")

    p_tpl = sub.add_parser("templates", help="list the built-in alpha templates")
    p_tpl.add_argument("--expressions", action="store_true", help="show a rendered example")

    p_fields = sub.add_parser("fields", help="list available data fields")
    p_fields.add_argument("--dataset", help="restrict to one dataset id")
    p_fields.add_argument("--search", help="substring filter applied by the API")
    p_fields.add_argument("--type", default="MATRIX", help="field type (default MATRIX)")
    p_fields.add_argument("--limit", type=int, default=50, help="max fields to print")
    p_fields.add_argument("--datasets", action="store_true", help="list datasets instead")

    p_sim = sub.add_parser("simulate", help="simulate one expression and print its IS result")
    p_sim.add_argument("expression")

    p_run = sub.add_parser("run", help="run the generate/simulate/evaluate loop")
    p_run.add_argument("--fields", nargs="+", default=list(DEFAULT_FIELDS))
    p_run.add_argument("--windows", nargs="+", type=int, default=list(DEFAULT_WINDOWS))
    p_run.add_argument("--templates", nargs="+", help="template names (default: all)")
    p_run.add_argument("--limit", type=int, help="max alphas to simulate this run")
    p_run.add_argument("--concurrency", type=int, default=3)
    p_run.add_argument("--decay", type=int, default=5)
    p_run.add_argument("--universe", default="TOP3000")
    p_run.add_argument("--region", default="USA")
    p_run.add_argument("--delay", type=int, default=1)
    p_run.add_argument("--neutralization", default="SUBINDUSTRY")
    p_run.add_argument("--truncation", type=float, default=0.05)
    p_run.add_argument("--min-sharpe", type=float, default=None)
    p_run.add_argument("--min-fitness", type=float, default=None)
    p_run.add_argument("--no-resume", action="store_true", help="re-simulate seen expressions")
    p_run.add_argument("--dry-run", action="store_true", help="print the grid, simulate nothing")
    # Pre-screen locally so quota goes only to candidates that already look good.
    p_run.add_argument("--screen", action="store_true",
                       help="score candidates locally first and simulate only the best")
    p_run.add_argument("--screen-top", type=int, default=20,
                       help="how many of the locally best candidates to simulate")
    p_run.add_argument("--data", default="data/prices",
                       help="price data for --screen")
    p_run.add_argument("--fundamentals", default=None)
    p_run.add_argument("--synthetic", action="store_true",
                       help="screen against generated data (for trying it out)")
    p_run.add_argument("--min-history", type=int, default=300)
    p_run.add_argument("--max-symbols", type=int, default=500,
                       help="panel width for screening; a subsample ranks "
                            "candidates far faster than the full universe")
    p_run.add_argument("--warmup", type=int, default=250)
    p_run.add_argument("--returns-basis", default="half_book",
                       choices=("half_book", "book"))

    p_res = sub.add_parser("results", help="show stored results")
    p_res.add_argument("--top", type=int, default=20)
    p_res.add_argument("--passed", action="store_true", help="only passing alphas")

    p_sub = sub.add_parser("submit", help="run submission checks (and optionally submit)")
    p_sub.add_argument("alpha_id", nargs="?", help="alpha id; omit to use stored winners")
    p_sub.add_argument("--all", action="store_true", help="process every stored winner")
    p_sub.add_argument(
        "--yes", action="store_true",
        help="actually submit. Without this flag only the checks are run.",
    )

    # Local pipeline: these cost no Brain quota and need no credentials.
    register_local(sub)
    return parser


def settings_from_args(args: argparse.Namespace) -> SimSettings:
    return SimSettings(
        region=args.region,
        universe=args.universe,
        delay=args.delay,
        decay=args.decay,
        neutralization=args.neutralization,
        truncation=args.truncation,
    )


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    if getattr(args, "command", None) == "fetch" and not args.dest:
        args.dest = f"data/{args.what}"
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.command == "templates":
        return cmd_templates(args)
    if args.command in LOCAL_HANDLERS:
        # Local commands run offline: no authentication, no quota.
        try:
            return LOCAL_HANDLERS[args.command](args)
        except KeyboardInterrupt:
            return 130

    store = ResultStore(args.db)

    if args.command == "run" and args.dry_run:
        return cmd_run_dry(args)
    if args.command == "results":
        return cmd_results(args, store)

    try:
        client = BrainClient(Credentials.from_env())
        client.authenticate()
    except BiometricRequired as exc:
        print(f"Biometric check required: {exc}", file=sys.stderr)
        return 3
    except (AuthError, ConfigError) as exc:
        print(f"Authentication failed: {exc}", file=sys.stderr)
        return 2

    try:
        if args.command == "auth":
            print(f"Authentication successful (user={client.user_id or 'unknown'})")
            return 0
        if args.command == "fields":
            return cmd_fields(args, client)
        if args.command == "simulate":
            return cmd_simulate(args, client, store)
        if args.command == "run":
            return cmd_run(args, client, store)
        if args.command == "submit":
            return cmd_submit(args, client, store)
    except BrainError as exc:
        print(f"API error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        store.close()

    return 0


def cmd_templates(args: argparse.Namespace) -> int:
    for template in TEMPLATES:
        print(f"{template.name}\n    {template.rationale}")
        if args.expressions:
            print(f"    e.g. {template.render('close', 20)}")
    return 0


def cmd_fields(args: argparse.Namespace, client: BrainClient) -> int:
    if args.datasets:
        for ds in client.datasets():
            print(f"{ds.get('id'):<28} {str(ds.get('name'))[:60]}")
        return 0
    fields = client.data_fields(
        dataset_id=args.dataset, field_type=args.type, search=args.search
    )
    print(f"{len(fields)} field(s)\n")
    for f in fields[: args.limit]:
        coverage = f.get("coverage")
        cov = f"{float(coverage):.2f}" if isinstance(coverage, (int, float)) else "n/a"
        print(f"{str(f.get('id')):<32} cov={cov:<6} {str(f.get('description'))[:60]}")
    return 0


def cmd_simulate(args: argparse.Namespace, client: BrainClient, store: ResultStore) -> int:
    expression = args.expression
    if looks_like_naive_momentum(expression):
        print(
            "Warning: this looks like the momentum form the context file flags as a\n"
            "         known failure in USA/D1/TOP3000 (Sharpe around -0.87). Consider\n"
            "         inverting it and wrapping it in rank().\n",
            file=sys.stderr,
        )
    result = run_simulation(client, expression)
    verdict = evaluate(result)
    print(f"status : {result.status}")
    print(f"alpha  : {result.alpha_id or '-'}")
    print(f"metrics: {result.summary()}")
    print(f"verdict: {'PASS' if verdict.passed else 'FAIL'} ({verdict.reason})")
    store.record(
        expression=expression,
        status=result.status,
        alpha_id=result.alpha_id,
        metrics=result.metrics,
        passed=verdict.passed,
        reason=verdict.reason,
    )
    return 0 if verdict.passed else 1


def _candidates(args: argparse.Namespace) -> list[Candidate]:
    templates = resolve_templates(args.templates)
    return list(generate(args.fields, args.windows, templates, limit=args.limit))


def cmd_run_dry(args: argparse.Namespace) -> int:
    candidates = _candidates(args)
    for c in candidates:
        print(c.expression)
    print(f"\n{len(candidates)} expression(s); nothing simulated (--dry-run)")
    return 0


def screen_locally(args: argparse.Namespace, candidates: list[Candidate]) -> list[Candidate]:
    """Rank candidates with the local simulator and keep the best few.

    A Brain simulation is the scarce resource, so spending it on a candidate
    that already looks poor locally is waste. Screening is free, so this
    strictly dominates simulating the grid in order -- provided the local
    screen ranks the way Brain does, which `wqb calibrate` is for.
    """
    import argparse as _argparse
    import re as _re

    from .localcli import load_panel, sim_config
    from .local.engine import simulate as local_simulate
    from .local.evaluator import ExpressionError

    # The remote `--universe` is a Brain universe NAME ("TOP3000"); the local
    # panel needs an instrument count. Derive one from the name so the screen
    # targets the same universe, bounded by the panel actually loaded.
    digits = _re.sub(r"\D", "", str(args.universe)) or "3000"
    width = int(digits)
    if args.max_symbols:
        width = min(width, args.max_symbols)
    local_args = _argparse.Namespace(**{**vars(args), "universe": width})

    panel = load_panel(local_args)
    config = sim_config(local_args)
    print(
        f"Screening {len(candidates)} candidate(s) locally over "
        f"{panel.n_dates} days x {panel.n_instruments} instruments...",
        flush=True,
    )

    scored: list[tuple[float, Candidate]] = []
    skipped = 0
    for candidate in candidates:
        try:
            metrics = local_simulate(candidate.expression, panel, config)
        except ExpressionError:
            skipped += 1
            continue
        # Reject locally hopeless turnover before it costs a simulation.
        if not 0.01 <= metrics.turnover <= 0.70:
            continue
        scored.append((metrics.sharpe, candidate))

    scored.sort(key=lambda row: row[0], reverse=True)
    kept = [candidate for _, candidate in scored[: args.screen_top]]
    print(
        f"  {len(scored)} screened"
        + (f", {skipped} unevaluable" if skipped else "")
        + f"; simulating the top {len(kept)} on Brain"
        + (f" (best local sharpe {scored[0][0]:.2f})" if scored else "")
        + "\n",
        flush=True,
    )
    return kept


def cmd_run(args: argparse.Namespace, client: BrainClient, store: ResultStore) -> int:
    config = RunConfig(
        concurrency=args.concurrency,
        max_alphas=args.limit,
        settings=settings_from_args(args),
    )
    if args.min_sharpe is not None:
        config.min_sharpe = args.min_sharpe
    if args.min_fitness is not None:
        config.min_fitness = args.min_fitness

    candidates = _candidates(args)
    if args.screen:
        candidates = screen_locally(args, candidates)
        if not candidates:
            print("Nothing survived the local screen; no quota spent.")
            return 0

    report = run_batch(
        client, candidates, store, config, resume=not args.no_resume
    )
    print(f"\n{report.line()}")
    if report.winners:
        print(f"\nPassing alphas (also appended to {store.success_file}):")
        for candidate, result in sorted(
            report.winners, key=lambda w: w[1].sharpe or 0, reverse=True
        ):
            print(f"  {candidate.expression}\n    {result.summary()}  alpha={result.alpha_id}")
        print("\nRun `python -m wqb submit --all` to check them for submission.")
    return 0


def cmd_results(args: argparse.Namespace, store: ResultStore) -> int:
    counts = store.counts()
    print(
        f"{counts['total']} simulated, {counts['passed']} passing, "
        f"{counts['submitted']} submitted\n"
    )
    rows = store.winners(args.top) if args.passed else store.top(args.top)
    if not rows:
        print("No results yet. Run `python -m wqb run` first.")
        return 0
    print(f"{'sharpe':>7} {'fitness':>8} {'turnover':>9}  expression")
    for row in rows:
        print(
            f"{_f(row['sharpe']):>7} {_f(row['fitness']):>8} {_f(row['turnover']):>9}  "
            f"{row['expression']}"
        )
    return 0


def cmd_submit(args: argparse.Namespace, client: BrainClient, store: ResultStore) -> int:
    if args.alpha_id:
        targets = [{"alpha_id": args.alpha_id, "expression": args.alpha_id}]
    else:
        targets = store.pending_submission()
        if not args.all:
            targets = targets[:1]
    if not targets:
        print("No stored alphas are awaiting submission.")
        return 0

    if not args.yes:
        print(
            f"Dry run: checking {len(targets)} alpha(s). "
            "Nothing will be submitted without --yes.\n"
        )

    exit_code = 0
    for target in targets:
        alpha_id = target["alpha_id"]
        outcome = (
            submit_alpha(client, alpha_id, confirm=True)
            if args.yes
            else check_submission(client, alpha_id)
        )
        print(outcome.describe())
        for check in outcome.checks:
            result = str(check.get("result", "")).upper()
            if result != "PASS":
                print(f"    {check.get('name')}: {result} {check.get('message') or ''}".rstrip())
        if outcome.submitted:
            store.mark_submitted(target["expression"])
        elif args.yes:
            exit_code = 1
    return exit_code


def _f(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"  # type: ignore[arg-type]


if __name__ == "__main__":
    raise SystemExit(main())
