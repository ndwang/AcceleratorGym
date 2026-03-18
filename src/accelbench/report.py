"""Scoring and report generation."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from accelbench.tasks import TASKS_BY_ID
from accelbench.types import RunRecord


def generate_report(record: RunRecord) -> dict[str, Any]:
    """Generate a structured scoring report from a benchmark run."""
    total = len(record.results)
    passed = sum(1 for r in record.results if r.passed)
    failed = sum(1 for r in record.results if not r.passed and not r.error)
    errors = sum(1 for r in record.results if r.error)

    # Per-tier breakdown
    tier_stats: dict[int, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "passed": 0, "failed": 0, "errors": 0}
    )
    for r in record.results:
        task = TASKS_BY_ID.get(r.task_id)
        tier = task.tier if task else 0
        tier_stats[tier]["total"] += 1
        if r.passed:
            tier_stats[tier]["passed"] += 1
        elif r.error:
            tier_stats[tier]["errors"] += 1
        else:
            tier_stats[tier]["failed"] += 1

    # Per-ability pass rates
    ability_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "passed": 0}
    )
    for r in record.results:
        task = TASKS_BY_ID.get(r.task_id)
        if not task:
            continue
        for ability in task.abilities:
            ability_stats[ability]["total"] += 1
            if r.passed:
                ability_stats[ability]["passed"] += 1

    ability_rates = {
        ability: stats["passed"] / stats["total"] if stats["total"] > 0 else 0.0
        for ability, stats in sorted(ability_stats.items())
    }

    # Per-task details
    task_details = []
    for r in record.results:
        task = TASKS_BY_ID.get(r.task_id)
        detail: dict[str, Any] = {
            "task_id": r.task_id,
            "name": task.name if task else "unknown",
            "tier": task.tier if task else 0,
            "passed": r.passed,
            "tool_calls": r.tool_calls,
            "budget": r.budget,
            "efficiency": round(r.efficiency, 3),
            "wall_time": round(r.wall_time, 2),
        }
        if r.error:
            detail["error"] = r.error
        task_details.append(detail)

    return {
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "pass_rate": round(passed / total, 3) if total > 0 else 0.0,
        },
        "per_tier": {
            str(tier): dict(stats)
            for tier, stats in sorted(tier_stats.items())
        },
        "per_ability": ability_rates,
        "tasks": task_details,
        "metadata": {
            "seed": record.seed,
            "config_path": record.config_path,
            "adapter": record.adapter_name,
        },
    }


def print_report(report: dict[str, Any]) -> None:
    """Print a human-readable summary to stdout."""
    s = report["summary"]
    print(f"\n{'='*60}")
    print(f"AccelBench Results: {s['passed']}/{s['total']} passed ({s['pass_rate']:.0%})")
    print(f"{'='*60}")

    print("\nPer-Tier Breakdown:")
    for tier, stats in sorted(report["per_tier"].items()):
        t = int(tier)
        label = {1: "Direct", 2: "Procedural", 3: "Adaptive", 4: "Complex"}.get(t, f"Tier {t}")
        print(f"  Tier {t} ({label}): {stats['passed']}/{stats['total']}")

    print("\nAbility Pass Rates:")
    for ability, rate in report["per_ability"].items():
        print(f"  {ability:15s}: {rate:.0%}")

    print("\nTask Details:")
    for t in report["tasks"]:
        status = "PASS" if t["passed"] else "FAIL"
        err = f" [{t['error'][:40]}...]" if t.get("error") else ""
        print(
            f"  {t['task_id']:5s} {t['name']:40s} {status:4s}  "
            f"tools: {t['tool_calls']:3d}/{t['budget']:3d}  "
            f"time: {t['wall_time']:6.1f}s{err}"
        )

    print()


def save_report(report: dict[str, Any], path: str) -> None:
    """Save report as JSON."""
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {path}")
