"""CLI entry point for AccelBench."""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from typing import Any


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
        default=None,
        help="Fully qualified adapter class name (default: LiteLLMAdapter)",
    )
    run_parser.add_argument(
        "--adapter-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra keyword arguments for the adapter constructor (repeatable)",
    )
    run_parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Model name for the default LiteLLM adapter (e.g. anthropic/claude-sonnet-4-20250514, gemini/gemini-2.5-pro)",
    )
    run_parser.add_argument(
        "--tasks", default=None, help="Comma-separated task IDs (e.g. 1.1,1.2,3.5)"
    )
    run_parser.add_argument(
        "--tier", type=int, default=None, help="Run only tasks from this tier"
    )
    run_parser.add_argument(
        "--output-dir", default=None, help="Directory for report and per-task trajectory files"
    )
    run_parser.add_argument(
        "--timeout", type=int, default=600,
        help="Wall-clock timeout in seconds per task (default: 600)",
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
    if args.adapter:
        adapter_kwargs = _parse_adapter_args(args.adapter_arg)
        adapter = _load_adapter(args.adapter, **adapter_kwargs)
    else:
        from accelbench.adapters.litellm import LiteLLMAdapter
        adapter = LiteLLMAdapter(model=args.model)

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
        output_dir=args.output_dir,
        timeout=args.timeout,
    )

    report = generate_report(record)
    print_report(report)

    if args.output_dir:
        import os
        os.makedirs(args.output_dir, exist_ok=True)
        save_report(report, os.path.join(args.output_dir, "report.json"))


def _parse_adapter_args(raw: list[str]) -> dict[str, str]:
    """Parse KEY=VALUE strings into a dict."""
    result = {}
    for item in raw:
        if "=" not in item:
            raise SystemExit(f"Invalid --adapter-arg (expected KEY=VALUE): {item}")
        key, value = item.split("=", 1)
        result[key] = value
    return result


def _load_adapter(fqn: str, **kwargs: Any):
    """Import and instantiate an adapter from a fully qualified class name."""
    module_path, class_name = fqn.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)


if __name__ == "__main__":
    main()
