"""CLI entry point for AccelBench."""

from __future__ import annotations

import argparse
import importlib
import logging
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="accelbench",
        description="AccelBench — benchmark harness for accelerator operations agents",
    )
    sub = parser.add_subparsers(dest="command")

    # Run command
    run_parser = sub.add_parser("run", help="Run benchmark tasks")
    run_parser.add_argument(
        "--config", required=True, help="Path to accelerator-gym YAML config"
    )
    run_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    run_parser.add_argument(
        "--adapter",
        default="accelbench.adapters.claude_sdk.ClaudeSDKAdapter",
        help="Fully qualified adapter class name",
    )
    run_parser.add_argument(
        "--tasks", default=None, help="Comma-separated task IDs (e.g. 1.1,1.2,3.5)"
    )
    run_parser.add_argument(
        "--tier", type=int, default=None, help="Run only tasks from this tier"
    )
    run_parser.add_argument(
        "--output", default=None, help="Path to save JSON report"
    )
    run_parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging"
    )

    # List command
    list_parser = sub.add_parser("list", help="List available tasks")
    list_parser.add_argument(
        "--tier", type=int, default=None, help="Filter by tier"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "list":
        _cmd_list(args)
    elif args.command == "run":
        _cmd_run(args)


def _cmd_list(args):
    from accelbench.tasks import ALL_TASKS

    tasks = ALL_TASKS
    if args.tier is not None:
        tasks = [t for t in tasks if t.tier == args.tier]

    print(f"{'ID':6s} {'Name':45s} {'Tier':5s} {'Budget':7s} Abilities")
    print("-" * 90)
    for t in tasks:
        print(
            f"{t.id:6s} {t.name:45s} {t.tier:5d} {t.budget:7d} {', '.join(t.abilities)}"
        )
    print(f"\nTotal: {len(tasks)} tasks")


def _cmd_run(args):
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )

    # Import adapter class
    adapter = _load_adapter(args.adapter)

    # Parse task IDs
    task_ids = args.tasks.split(",") if args.tasks else None

    from accelbench.harness import run_benchmark
    from accelbench.report import generate_report, print_report, save_report

    record = run_benchmark(
        config_path=args.config,
        adapter=adapter,
        seed=args.seed,
        task_ids=task_ids,
        tier=args.tier,
    )

    report = generate_report(record)
    print_report(report)

    if args.output:
        save_report(report, args.output)


def _load_adapter(fqn: str):
    """Import and instantiate an adapter from a fully qualified class name."""
    module_path, class_name = fqn.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()


if __name__ == "__main__":
    main()
