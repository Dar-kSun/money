"""CLI handlers for the local pipeline: fetch, screen, search, propose, calibrate."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .diversity import report as diversity_report
from .diversity import select_diverse
from .fields import FieldInfo, rank_by_crowding
from .generate import DEFAULT_FIELDS, generate, resolve_templates, validate_expression
from .local.data import add_universe, load_csv_panel, load_fundamentals, synthetic_panel
from .local.engine import SimConfig, Metrics, correlation, simulate
from .local.evaluator import ExpressionError
from .local.panel import Panel

log = logging.getLogger(__name__)


# ------------------------------------------------------------------- panel
def load_panel(args: argparse.Namespace) -> Panel:
    """Build the panel a local command runs over.

    ``--synthetic`` exists so every command can be exercised before any data
    has been downloaded; results from it are meaningless as alphas.
    """
    if getattr(args, "synthetic", False):
        print("Using a synthetic panel: results are NOT real alphas.\n", file=sys.stderr)
        panel = synthetic_panel(1200, 120, autocorr=-0.05, seed=0)
    else:
        directory = Path(args.data)
        if not directory.is_dir():
            raise SystemExit(
                f"no data directory at {directory}. Run `python -m wqb fetch "
                f"prices --dest {directory}` first, or pass --synthetic to try "
                f"the pipeline on generated data."
            )
        panel = load_csv_panel(
            directory, min_history=args.min_history, max_symbols=args.max_symbols
        )
        if getattr(args, "fundamentals", None):
            load_fundamentals(panel, args.fundamentals)
    add_universe(panel, size=args.universe, by="cap")
    return panel


def sim_config(args: argparse.Namespace) -> SimConfig:
    return SimConfig(
        delay=args.delay,
        decay=args.decay,
        truncation=args.truncation,
        neutralization=args.neutralization,
        universe_size=args.universe,
        warmup=args.warmup,
        returns_basis=args.returns_basis,
    )


def _print_panel(panel: Panel) -> None:
    print(
        f"Panel: {panel.n_dates} days x {panel.n_instruments} instruments, "
        f"{len(panel.field_names)} fields "
        f"({panel.dates[0]} to {panel.dates[-1]})\n"
    )


# ------------------------------------------------------------------- fetch
def cmd_fetch(args: argparse.Namespace) -> int:
    from .local.fetch import (
        FetchError, fetch_sec_fundamentals, fetch_stooq_bulk, fetch_stooq_symbols,
    )

    try:
        if args.what == "prices":
            dest = Path(args.dest)
            if args.symbols:
                count = fetch_stooq_symbols(args.symbols, dest,
                                            min_rows=args.min_history)
            else:
                count = fetch_stooq_bulk(dest, min_rows=args.min_history,
                                         max_symbols=args.max_symbols)
            print(f"Wrote {count} symbol file(s) to {dest}")
            return 0

        symbols = args.symbols or [
            p.stem for p in sorted(Path(args.symbols_from).glob("*.csv"))
        ]
        if not symbols:
            raise SystemExit(
                f"no symbols given and none found in {args.symbols_from}; "
                f"fetch prices first or pass --symbols"
            )
        if args.contact == "unset":
            print(
                "Note: SEC asks automated clients to identify themselves with a\n"
                "contact address. Pass --contact 'you@example.com' if you are\n"
                "going to pull a lot of filings.\n", file=sys.stderr,
            )
        count = fetch_sec_fundamentals(
            symbols[: args.max_symbols] if args.max_symbols else symbols,
            Path(args.dest),
            contact=args.contact if args.contact != "unset" else
            "wqb-local-sim (contact unset)",
        )
        print(f"Wrote fundamentals for {count} symbol(s) to {args.dest}")
        return 0
    except FetchError as exc:
        print(f"Fetch failed: {exc}", file=sys.stderr)
        return 1


# ------------------------------------------------------------------ screen
def cmd_screen(args: argparse.Namespace) -> int:
    """Score the template grid locally and print it ranked. Costs no quota."""
    panel = load_panel(args)
    _print_panel(panel)
    config = sim_config(args)

    available = set(panel.field_names)
    requested = args.fields or list(DEFAULT_FIELDS)
    fields = [f for f in requested if f in available]
    missing = [f for f in requested if f not in available]
    if missing:
        print(f"Skipping fields absent from the panel: {', '.join(missing)}\n",
              file=sys.stderr)
    if not fields:
        raise SystemExit(
            f"none of the requested fields are in the panel. It holds: "
            f"{', '.join(sorted(available))}"
        )

    templates = resolve_templates(args.templates)
    rows: list[tuple[Metrics, str]] = []
    failures = 0

    for candidate in generate(fields, args.windows, templates, limit=args.limit):
        try:
            metrics = simulate(candidate.expression, panel, config)
        except ExpressionError as exc:
            failures += 1
            log.debug("skipped %s: %s", candidate.expression, exc)
            continue
        rows.append((metrics, candidate.expression))

    if not rows:
        print("Nothing evaluated.", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: r[0].sharpe, reverse=True)
    passing = [
        (m, e) for m, e in rows
        if m.sharpe > args.min_sharpe and m.fitness >= args.min_fitness
        and 0.01 <= m.turnover <= 0.70
    ]

    print(f"{'sharpe':>7} {'fitness':>8} {'turnover':>9} {'returns':>8}  expression")
    for metrics, expression in rows[: args.top]:
        print(f"{metrics.sharpe:>7.2f} {metrics.fitness:>8.2f} "
              f"{metrics.turnover:>9.3f} {metrics.returns:>8.3f}  {expression}")

    print(f"\n{len(rows)} evaluated, {len(passing)} clear the local thresholds"
          + (f", {failures} failed to evaluate" if failures else ""))
    if passing:
        print(diversity_report([e for _, e in passing]).describe())
        print("\nThese are LOCAL estimates on local data. Confirm the best few "
              "with `python -m wqb simulate '<expression>'` before trusting them.")
    return 0


# ------------------------------------------------------------------ search
def cmd_search(args: argparse.Namespace) -> int:
    from .search import EvolveConfig, evolve, mutation_space

    panel = load_panel(args)
    _print_panel(panel)

    available = [f for f in panel.field_names if f not in
                 ("market", "sector", "industry", "subindustry")]
    fields = [f for f in (args.fields or available) if f in set(available)]
    if not fields:
        raise SystemExit("no usable fields in the panel")

    seeds = args.seeds or [
        c.expression for c in generate(fields[:4], [20], limit=4)
    ] or ["rank(-1 * ts_zscore(close, 20))"]

    space = mutation_space(fields)
    config = EvolveConfig(
        population=args.population, generations=args.generations,
        elite=args.elite, max_book=args.book, seed=args.seed,
        min_sharpe=args.min_sharpe, min_fitness=args.min_fitness,
        max_correlation=args.max_correlation,
    )

    evaluated = [0]

    def progress(generation: int, book, scored) -> None:
        evaluated[0] += 1
        if args.verbose and scored.metrics:
            print(f"  gen{generation} sharpe={scored.metrics.sharpe:+.2f} "
                  f"{scored.expression}", file=sys.stderr)

    print(f"Searching from {len(seeds)} seed(s) over {len(fields)} field(s)...")
    book = evolve(panel, seeds, space, sim_config(args), config, progress=progress)

    print(f"\n{evaluated[0]} candidates evaluated locally (no quota spent); "
          f"{len(book)} kept\n")
    if not book:
        print("Nothing cleared the thresholds. That is a real answer: either "
              "the data holds no signal these shapes can reach, or the "
              "thresholds need relaxing to explore (--min-sharpe).")
        return 0

    for found in book:
        print(f"  {found.expression}")
        print(f"     {found.summary()}")
    print(f"\n{diversity_report([f.expression for f in book]).describe()}")
    print("\nConfirm on Brain before submitting:")
    for found in book[:3]:
        print(f"  python -m wqb simulate '{found.expression}'")
    return 0


# ----------------------------------------------------------------- propose
def cmd_propose(args: argparse.Namespace) -> int:
    from .propose import ProposerUnavailable, propose

    panel = load_panel(args) if (args.data or args.synthetic) else None
    if panel is not None:
        _print_panel(panel)

    fields = [
        FieldInfo(id=name) for name in (
            args.fields or (panel.field_names if panel else list(DEFAULT_FIELDS))
        )
        if name not in ("market", "sector", "industry", "subindustry")
    ]

    try:
        batch = propose(fields, count=args.count, panel=panel)
    except ProposerUnavailable as exc:
        print(f"Proposer unavailable: {exc}", file=sys.stderr)
        return 2

    print(f"{batch.summary()}\n")
    for proposal in batch.accepted:
        print(f"  {proposal.expression}")
        print(f"     {proposal.mechanism}: {proposal.hypothesis}")
        if panel is not None:
            metrics = simulate(proposal.expression, panel, sim_config(args))
            print(f"     local: {metrics.summary()}")
    for expression, why in batch.rejected:
        print(f"  [rejected] {expression or '(empty)'}\n     {why}", file=sys.stderr)
    return 0


# --------------------------------------------------------------- calibrate
def cmd_calibrate(args: argparse.Namespace) -> int:
    """Print local metrics for one expression, for comparison with Brain's.

    The local absolute numbers are uncalibrated -- notably the definition of
    `returns`, which feeds fitness. Run the same expression on Brain with
    `python -m wqb simulate` and compare: if the local Sharpe ranks candidates
    the same way Brain does, the screen is doing its job even when the levels
    differ.
    """
    problems = validate_expression(args.expression)
    if problems:
        print("Expression has problems: " + "; ".join(problems), file=sys.stderr)
        # An unbalanced expression cannot evaluate at all, so there is nothing
        # to calibrate; the others are warnings worth simulating through.
        if any("parenthes" in problem for problem in problems):
            return 1

    panel = load_panel(args)
    _print_panel(panel)

    for basis in ("half_book", "book"):
        config = sim_config(args)
        config = SimConfig(**{**config.__dict__, "returns_basis": basis})
        try:
            metrics = simulate(args.expression, panel, config)
        except ExpressionError as exc:
            print(f"Cannot evaluate the expression: {exc}", file=sys.stderr)
            return 1
        print(f"returns_basis={basis:10s} {metrics.summary()}")

    print(
        "\nNow run the same expression on Brain and compare:\n"
        f"  python -m wqb simulate '{args.expression}'\n"
        "Pick the returns_basis whose fitness lands closer to Brain's, and set\n"
        "it with --returns-basis from then on."
    )
    return 0


# ------------------------------------------------------------------ parser
def add_local_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", default="data/prices",
                        help="directory of per-symbol OHLCV CSVs")
    parser.add_argument("--fundamentals", default=None,
                        help="directory of SEC fundamental CSVs to merge in")
    parser.add_argument("--synthetic", action="store_true",
                        help="use generated data (for trying the pipeline only)")
    parser.add_argument("--min-history", type=int, default=300)
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--universe", type=int, default=3000)
    parser.add_argument("--delay", type=int, default=1)
    parser.add_argument("--decay", type=int, default=5)
    parser.add_argument("--truncation", type=float, default=0.05)
    parser.add_argument("--neutralization", default="SUBINDUSTRY")
    parser.add_argument("--warmup", type=int, default=250)
    parser.add_argument("--returns-basis", default="half_book",
                        choices=("half_book", "book"))
    parser.add_argument("--min-sharpe", type=float, default=1.25)
    parser.add_argument("--min-fitness", type=float, default=1.0)


def register(sub: argparse._SubParsersAction) -> None:
    fetch = sub.add_parser("fetch", help="download free market data")
    fetch.add_argument("what", choices=("prices", "fundamentals"))
    fetch.add_argument("--dest", default=None)
    fetch.add_argument("--symbols", nargs="+", default=None)
    fetch.add_argument("--symbols-from", default="data/prices",
                       help="take the symbol list from this price directory")
    fetch.add_argument("--contact", default="unset",
                       help="contact address for the SEC User-Agent")
    fetch.add_argument("--min-history", type=int, default=300)
    fetch.add_argument("--max-symbols", type=int, default=None)

    screen = sub.add_parser("screen", help="score the template grid locally (free)")
    add_local_arguments(screen)
    screen.add_argument("--fields", nargs="+", default=None)
    screen.add_argument("--windows", nargs="+", type=int, default=None)
    screen.add_argument("--templates", nargs="+", default=None)
    screen.add_argument("--limit", type=int, default=None)
    screen.add_argument("--top", type=int, default=25)

    search = sub.add_parser("search", help="evolutionary search over expressions (free)")
    add_local_arguments(search)
    search.add_argument("--fields", nargs="+", default=None)
    search.add_argument("--seeds", nargs="+", default=None)
    search.add_argument("--population", type=int, default=40)
    search.add_argument("--generations", type=int, default=15)
    search.add_argument("--elite", type=int, default=8)
    search.add_argument("--book", type=int, default=15)
    search.add_argument("--max-correlation", type=float, default=0.7)
    search.add_argument("--seed", type=int, default=0)

    propose_p = sub.add_parser("propose", help="ask Claude for alpha hypotheses")
    add_local_arguments(propose_p)
    propose_p.add_argument("--fields", nargs="+", default=None)
    propose_p.add_argument("--count", type=int, default=10)

    calibrate = sub.add_parser(
        "calibrate", help="compare local metrics with Brain's for one expression"
    )
    add_local_arguments(calibrate)
    calibrate.add_argument("expression")


HANDLERS = {
    "fetch": cmd_fetch,
    "screen": cmd_screen,
    "search": cmd_search,
    "propose": cmd_propose,
    "calibrate": cmd_calibrate,
}
