"""Terminal viewer for AccelBench trace files."""

from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

_USE_COLOR = True


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if not hasattr(sys.stdout, "isatty"):
        return False
    return sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def _bold(t: str) -> str:
    return _c("1", t)


def _dim(t: str) -> str:
    return _c("2", t)


def _green(t: str) -> str:
    return _c("32", t)


def _red(t: str) -> str:
    return _c("31", t)


def _yellow(t: str) -> str:
    return _c("33", t)


def _cyan(t: str) -> str:
    return _c("36", t)


def _magenta(t: str) -> str:
    return _c("35", t)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _status(passed: bool, error: str | None = None) -> str:
    if error:
        return _red("ERR")
    return _green("PASS") if passed else _red("FAIL")


def _indent(text: str, prefix: str = "  ") -> str:
    return textwrap.indent(text, prefix)


def _truncate(text: str, max_lines: int) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    shown = lines[:max_lines]
    omitted = len(lines) - max_lines
    shown.append(_dim(f"  ... ({omitted} more lines)"))
    return "\n".join(shown)


def _format_tool_args(tool: str, arguments: dict[str, Any]) -> str:
    if tool == "query_devices":
        return arguments.get("sql", "")
    elif tool == "get_variables":
        names = arguments.get("names", [])
        if len(names) <= 6:
            return ", ".join(names)
        return ", ".join(names[:5]) + f" ... (+{len(names) - 5} more)"
    elif tool == "set_variables":
        values = arguments.get("values", {})
        parts = [f"{k} = {v}" for k, v in values.items()]
        return ", ".join(parts)
    elif tool == "browse_devices":
        path = arguments.get("path", "/")
        depth = arguments.get("depth", 1)
        return f"path={path}, depth={depth}"
    elif tool == "reset":
        return ""
    return json.dumps(arguments, default=str)


def _format_tool_result(tool: str, result: str, max_lines: int) -> str:
    if result.startswith("Error"):
        return _red(result)
    # For query results, try to summarize
    if tool == "query_devices":
        try:
            data = json.loads(result)
            count = data.get("count", "?")
            rows = data.get("rows", [])
            if count == 0:
                return _dim("(no rows)")
            # Show a compact summary of rows
            lines = [f"{count} rows:"]
            for row in rows[:max_lines - 1]:
                parts = [f"{k}={v}" for k, v in row.items()]
                lines.append("  " + ", ".join(parts))
            if len(rows) > max_lines - 1:
                lines.append(_dim(f"  ... ({len(rows) - max_lines + 1} more rows)"))
            return "\n".join(lines)
        except (json.JSONDecodeError, AttributeError):
            pass
    if tool == "browse_devices":
        try:
            data = json.loads(result)
            lines = []
            for item in data:
                name = item.get("name", "?")
                kind = item.get("type", "")
                lines.append(f"  {name} ({kind})" if kind else f"  {name}")
            if not lines:
                return _dim("(empty)")
            return "\n".join(lines[:max_lines])
        except (json.JSONDecodeError, TypeError):
            pass
    return _truncate(result, max_lines)


# ---------------------------------------------------------------------------
# View a single trace file
# ---------------------------------------------------------------------------

def view_trace(path: str, max_lines: int = 20) -> None:
    with open(path) as f:
        data = json.load(f)

    task_id = data.get("task_id", "?")
    task_name = data.get("task_name", "")
    tier = data.get("tier", "?")
    passed = data.get("passed", False)
    error = data.get("error")
    tool_calls = data.get("tool_calls", 0)
    budget = data.get("budget", 0)
    efficiency = data.get("efficiency", 0)
    wall_time = data.get("wall_time", 0)
    failure_reason = data.get("failure_reason")

    # Header
    status = _status(passed, error)
    print()
    print(
        f"{_bold(f'Task {task_id}: {task_name}')} "
        f"(Tier {tier})  {status}"
    )
    detail_parts = [
        f"Tools: {tool_calls}/{budget}",
        f"Efficiency: {efficiency:.0%}",
        f"Time: {wall_time:.1f}s",
    ]
    if failure_reason:
        detail_parts.append(f"Reason: {_yellow(failure_reason)}")
    print(_dim(" | ".join(detail_parts)))

    if error:
        print(f"\n{_red('Error:')} {error}")

    # Prompt
    prompt = data.get("prompt", "")
    if prompt:
        print(f"\n{_bold('Prompt:')}")
        print(_indent(_truncate(prompt, max_lines), "  "))

    # Trace
    trace = data.get("trace", [])
    if trace:
        print()
        step = 0
        for entry in trace:
            if entry.get("role") == "assistant":
                # Reasoning entry
                content = entry.get("content", "")
                print(f"{_dim('─'*60)}")
                print(f"  {_magenta('Agent:')}")
                print(_indent(content, "    "))
                print()
            elif "tool" in entry:
                step += 1
                tool = entry["tool"]
                arguments = entry.get("arguments", {})
                result = entry.get("result", "")

                print(f"{_dim('─'*60)}")
                formatted_args = _format_tool_args(tool, arguments)
                print(f"  {_cyan(f'[{step}]')} {_bold(tool)}")
                if formatted_args:
                    print(_indent(formatted_args, "    "))
                print()
                formatted_result = _format_tool_result(tool, result, max_lines)
                print(f"  {_dim('Result:')}")
                print(_indent(formatted_result, "    "))
                print()

    # Answer
    answer = data.get("extracted_answer")
    if answer is not None:
        print(f"{_dim('─'*60)}")
        print(f"  {_bold('Answer:')} {json.dumps(answer, default=str)}")

    # Response (if different from answer)
    response = data.get("response", "")
    if response and response != json.dumps(answer):
        print(f"\n{_bold('Response:')}")
        print(_indent(_truncate(response, max_lines), "  "))

    print()


# ---------------------------------------------------------------------------
# View a run directory (summary table)
# ---------------------------------------------------------------------------

def view_run(directory: str) -> None:
    dirpath = Path(directory)

    # Try to load report.json for metadata
    report_path = dirpath / "report.json"
    report: dict[str, Any] | None = None
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)

    # Load all trace files
    traces_dir = dirpath / "traces"
    if not traces_dir.exists():
        # Maybe the user pointed directly at a traces dir
        traces_dir = dirpath

    trace_files = sorted(traces_dir.glob("task_*.json"))
    if not trace_files:
        print(f"No trace files found in {directory}")
        return

    traces = []
    for tf in trace_files:
        with open(tf) as f:
            traces.append(json.load(f))

    # Sort by task_id
    def _sort_key(t: dict) -> tuple:
        parts = t.get("task_id", "0.0").split(".")
        return tuple(int(p) for p in parts)
    traces.sort(key=_sort_key)

    # Header
    total = len(traces)
    passed = sum(1 for t in traces if t.get("passed"))
    failed = sum(1 for t in traces if not t.get("passed") and not t.get("error"))
    errors = sum(1 for t in traces if t.get("error"))

    print()
    name = dirpath.name
    if report and report.get("metadata", {}).get("model"):
        name += f"  ({report['metadata']['model']})"
    print(_bold(f"AccelBench Run: {name}"))
    parts = [f"{total} tasks"]
    if passed:
        parts.append(_green(f"{passed} passed"))
    if failed:
        parts.append(_red(f"{failed} failed"))
    if errors:
        parts.append(_red(f"{errors} errors"))
    print(" | ".join(parts))

    # Per-tier summary
    tiers: dict[int, list] = {}
    for t in traces:
        tier = t.get("tier", 0)
        tiers.setdefault(tier, []).append(t)

    if len(tiers) > 1:
        print()
        tier_labels = {1: "Direct", 2: "Procedural", 3: "Adaptive", 4: "Complex"}
        for tier in sorted(tiers):
            tier_traces = tiers[tier]
            tp = sum(1 for t in tier_traces if t.get("passed"))
            label = tier_labels.get(tier, f"Tier {tier}")
            status = _green(f"{tp}/{len(tier_traces)}") if tp == len(tier_traces) else f"{tp}/{len(tier_traces)}"
            print(f"  Tier {tier} ({label}): {status}")

    # Task table
    print()
    for t in traces:
        task_id = t.get("task_id", "?")
        task_name = t.get("task_name", "")
        is_passed = t.get("passed", False)
        error = t.get("error")
        tc = t.get("tool_calls", 0)
        bud = t.get("budget", 0)
        wt = t.get("wall_time", 0)
        fr = t.get("failure_reason")

        status = _status(is_passed, error)
        reason = ""
        if fr:
            reason = f"  ({_yellow(fr)})"
        elif error:
            # Shorten error for table display
            short_err = error[:60] + "..." if len(error) > 60 else error
            reason = f"  ({_dim(short_err)})"

        print(
            f"  {task_id:5s} {task_name:40s} {status}  "
            f"tools: {tc:3d}/{bud:3d}  "
            f"time: {wt:6.1f}s{reason}"
        )

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def view(path: str, max_lines: int = 20) -> None:
    global _USE_COLOR
    _USE_COLOR = _supports_color()

    p = Path(path)
    if p.is_file() and p.suffix == ".json":
        view_trace(path, max_lines=max_lines)
    elif p.is_dir():
        view_run(path)
    else:
        print(f"Error: {path} is not a JSON file or directory")
        sys.exit(1)
