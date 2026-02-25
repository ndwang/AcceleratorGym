"""Interactive CLI for manually testing accelerator-gym tools."""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import sys
from pathlib import Path
from typing import Any

from accelerator_gym.core.machine import Machine

logger = logging.getLogger(__name__)

COMMANDS = {
    "browse":    "browse [path] [depth]     Browse the device tree",
    "query":     "query <sql>               Run a SQL query against device metadata",
    "get":       "get <variable>            Read a variable value",
    "gets":      "gets <var1> <var2> ...    Read multiple variables",
    "set":       "set <variable> <value>    Write a variable value",
    "sets":      "sets <var>=<val> ...      Write multiple variables atomically",
    "state":     "state                     Snapshot of all readable variables",
    "reset":     "reset                     Reset machine to initial state",
    "help":      "help                      Show this help message",
    "quit":      "quit                      Exit the CLI",
}

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

_NO_COLOR = os.environ.get("NO_COLOR") is not None


def _supports_color() -> bool:
    if _NO_COLOR:
        return False
    if sys.platform == "win32":
        return os.environ.get("TERM") or os.environ.get("WT_SESSION") or os.environ.get("ANSICON")
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


_COLOR = _supports_color()


def _ansi(code: str, text: str) -> str:
    if not _COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def _bold(text: str) -> str:
    return _ansi("1", text)


def _dim(text: str) -> str:
    return _ansi("2", text)


def _cyan(text: str) -> str:
    return _ansi("36", text)


def _green(text: str) -> str:
    return _ansi("32", text)


def _yellow(text: str) -> str:
    return _ansi("33", text)


def _red(text: str) -> str:
    return _ansi("31", text)


def _magenta(text: str) -> str:
    return _ansi("35", text)


# ---------------------------------------------------------------------------
# Unicode support detection & symbols
# ---------------------------------------------------------------------------

def _supports_unicode() -> bool:
    """Check if stdout can handle Unicode box-drawing / icon characters."""
    try:
        encoding = sys.stdout.encoding or "ascii"
        "\u2514\u2500\u251c\u2502\u25c6\u2714".encode(encoding)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


_UNICODE = _supports_unicode()

# Tree connectors
_T_MID = "\u251c\u2500\u2500 " if _UNICODE else "|-- "
_T_END = "\u2514\u2500\u2500 " if _UNICODE else "`-- "
_T_PIPE = "\u2502   " if _UNICODE else "|   "
_T_BLANK = "    "

# Table separator
_HSEP = "\u2500" if _UNICODE else "-"

# Em dash for title
_EMDASH = "\u2014" if _UNICODE else "--"


# ---------------------------------------------------------------------------
# Browse formatting
# ---------------------------------------------------------------------------

# Path depths: 0=root(systems), 1=system(types), 2=type(devices), 3=device(attrs)
_LEAF_DEPTH = 3


def _fmt_browse(result: dict[str, Any]) -> str:
    """Format browse output as a tree with box-drawing lines and colors."""
    if "error" in result:
        return _red(f"Error: {result['message']}")

    # Leaf attribute (browsed to /sys/type/dev/attr)
    if "variable" in result:
        lines = [_bold(_cyan(result["path"]))]
        lines.append(f"  {_dim('variable:')}  {_yellow(result['variable'])}")
        if result.get("description"):
            lines.append(f"  {_dim('desc:')}      {result['description']}")
        if result.get("units"):
            lines.append(f"  {_dim('units:')}     {_magenta(result['units'])}")
        r, w = result.get("read", False), result.get("write", False)
        lines.append(f"  {_dim('read:')}      {_green('yes') if r else _red('no')}")
        lines.append(f"  {_dim('write:')}     {_green('yes') if w else _red('no')}")
        if "limits" in result:
            lo, hi = result["limits"]
            lines.append(f"  {_dim('limits:')}    [{lo}, {hi}]")
        return "\n".join(lines)

    # Tree node
    path = result.get("path", "/")
    depth = len([p for p in path.strip("/").split("/") if p])
    lines = [_bold(_cyan(path))]
    children = result.get("children", [])
    _fmt_tree(lines, children, depth, prefix="")
    return "\n".join(lines)


def _fmt_tree(
    lines: list[str],
    children: list[Any],
    parent_depth: int,
    prefix: str,
) -> None:
    """Recursively format children with box-drawing connectors."""
    child_depth = parent_depth + 1
    total = len(children)

    for i, child in enumerate(children):
        is_last = i == total - 1
        connector = _T_END if is_last else _T_MID
        continuation = _T_BLANK if is_last else _T_PIPE

        if isinstance(child, str):
            is_leaf = child_depth > _LEAF_DEPTH
            label = child if is_leaf else _bold(child)
            lines.append(f"{prefix}{_dim(connector)}{label}")

        elif isinstance(child, dict):
            if "variable" in child:
                # Full attribute metadata inline
                name = child["name"]
                rw = "RW" if child.get("read") and child.get("write") else (
                    "R" if child.get("read") else "W" if child.get("write") else ""
                )
                units = _magenta(f"[{child['units']}]") if child.get("units") else ""
                desc = _dim(child["description"]) if child.get("description") else ""
                parts = [_yellow(name)]
                if rw:
                    parts.append(_green(rw))
                if units:
                    parts.append(units)
                if desc:
                    parts.append(desc)
                lines.append(f"{prefix}{_dim(connector)}{'  '.join(parts)}")
            else:
                name = child.get("name", "")
                desc = child.get("description", "")
                nested = child.get("children")
                is_leaf = child_depth > _LEAF_DEPTH
                label = name if is_leaf else _bold(name)
                if desc:
                    label += f"  {_dim(desc)}"
                lines.append(f"{prefix}{_dim(connector)}{label}")
                if nested is not None:
                    _fmt_tree(lines, nested, child_depth, prefix + _dim(continuation))


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def _fmt_table(rows: list[dict[str, Any]]) -> str:
    """Format a list of dicts as an aligned table with colored header."""
    if not rows:
        return _dim("(no results)")
    columns = list(rows[0].keys())
    widths = {col: len(col) for col in columns}
    str_rows = []
    for row in rows:
        str_row = {col: str(row[col]) for col in columns}
        for col in columns:
            widths[col] = max(widths[col], len(str_row[col]))
        str_rows.append(str_row)
    header = "  ".join(_bold(col.ljust(widths[col])) for col in columns)
    separator = _dim("  ".join(_HSEP * widths[col] for col in columns))
    lines = [header, separator]
    for str_row in str_rows:
        cells = []
        for col in columns:
            cells.append(str_row[col].ljust(widths[col]))
        lines.append("  ".join(cells))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Value formatting
# ---------------------------------------------------------------------------

def _fmt_values(values: dict[str, float], variables: dict | None = None) -> str:
    """Format variable name=value pairs in aligned columns."""
    if not values:
        return _dim("(no values)")
    name_width = max(len(n) for n in values)
    lines = []
    for name, val in values.items():
        units = ""
        if variables:
            var = variables.get(name)
            if var and var.units:
                units = f"  {_magenta(f'[{var.units}]')}"
        lines.append(f"  {_cyan(f'{name:<{name_width}}')}  {_dim('=')} {_green(str(val))}{units}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class CLI:
    def __init__(self, machine: Machine) -> None:
        self._machine = machine

    def run(self) -> None:
        title = _bold(f"accelerator-gym CLI {_EMDASH} {self._machine._config.name}")
        print(title)
        print(f"{_green(str(len(self._machine.variables)))} variables loaded")
        print(f'Type {_bold("help")} for commands.\n')

        while True:
            try:
                line = input(_bold("ag> ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            try:
                parts = shlex.split(line)
            except ValueError as e:
                print(_red(f"Parse error: {e}"))
                continue
            cmd, args = parts[0], parts[1:]
            handler = getattr(self, f"cmd_{cmd}", None)
            if handler is None:
                print(_red(f"Unknown command: {cmd}.") + " Type 'help' for commands.")
                continue
            try:
                handler(args)
            except Exception as e:
                print(_red(f"Error: {e}"))

    def cmd_help(self, args: list[str]) -> None:
        for key, desc in COMMANDS.items():
            # Highlight the command name portion
            print(f"  {_bold(_cyan(key))}{desc[len(key):]}")

    def cmd_quit(self, args: list[str]) -> None:
        raise SystemExit(0)

    def cmd_browse(self, args: list[str]) -> None:
        path = args[0] if args else "/"
        depth = int(args[1]) if len(args) > 1 else 1
        result = self._machine.catalog.browse(path, depth)
        print(_fmt_browse(result))

    def cmd_query(self, args: list[str]) -> None:
        if not args:
            print(f"Usage: query {_dim('<sql>')}")
            return
        sql = " ".join(args)
        try:
            rows = self._machine.catalog.query(sql)
            print(_fmt_table(rows))
            print(f"\n{_dim(f'({len(rows)} rows)')}")
        except ValueError as e:
            print(_red(f"Query error: {e}"))

    def cmd_get(self, args: list[str]) -> None:
        if not args:
            print(f"Usage: get {_dim('<variable>')}")
            return
        name = args[0]
        var = self._machine.variables.get(name)
        if var is None:
            print(_red(f"Unknown variable: {name}"))
            return
        value = self._machine.get(name)
        units = f"  {_magenta(f'[{var.units}]')}" if var.units else ""
        print(f"  {_cyan(name)} {_dim('=')} {_green(str(value))}{units}")

    def cmd_gets(self, args: list[str]) -> None:
        if not args:
            print(f"Usage: gets {_dim('<var1> <var2> ...')}")
            return
        values = self._machine.get_many(args)
        print(_fmt_values(values, self._machine.variables))

    def cmd_set(self, args: list[str]) -> None:
        if len(args) < 2:
            print(f"Usage: set {_dim('<variable> <value>')}")
            return
        name = args[0]
        value = float(args[1])
        self._machine.set(name, value)
        var = self._machine.variables.get(name)
        units = f"  {_magenta(f'[{var.units}]')}" if var and var.units else ""
        print(f"  {_cyan(name)} {_dim('<-')} {_yellow(str(value))}{units}")

    def cmd_sets(self, args: list[str]) -> None:
        if not args:
            print(f"Usage: sets {_dim('<var>=<val> ...')}")
            return
        values: dict[str, float] = {}
        for pair in args:
            if "=" not in pair:
                print(_red(f"Invalid format: {pair}. Use var=value"))
                return
            k, v = pair.split("=", 1)
            values[k] = float(v)
        self._machine.set_many(values)
        for name, val in values.items():
            var = self._machine.variables.get(name)
            units = f"  {_magenta(f'[{var.units}]')}" if var and var.units else ""
            print(f"  {_cyan(name)} {_dim('<-')} {_yellow(str(val))}{units}")

    def cmd_state(self, args: list[str]) -> None:
        readable = [n for n, v in self._machine.variables.items() if v.readable]
        values = self._machine.get_many(readable)
        print(_fmt_values(values, self._machine.variables))

    def cmd_reset(self, args: list[str]) -> None:
        self._machine.reset()
        print(f"  Machine reset to initial state.")


CONFIG_FILENAME = "accelerator-gym.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive CLI for testing accelerator-gym tools. "
        "Run from a directory with accelerator-gym.yaml or pass --config.",
    )
    parser.add_argument("--config", default=None, help="Path to YAML config (auto-detected if omitted)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )

    if args.config:
        config_path = args.config
    else:
        path = Path.cwd() / CONFIG_FILENAME
        if not path.exists():
            print(f"ERROR: {CONFIG_FILENAME} not found in {Path.cwd()}.", file=sys.stderr)
            print("Provide --config or create an accelerator-gym.yaml file.", file=sys.stderr)
            sys.exit(1)
        config_path = str(path)

    try:
        machine = Machine.from_config(config_path)
    except Exception as e:
        print(f"ERROR: Failed to initialize machine: {e}", file=sys.stderr)
        sys.exit(1)

    CLI(machine).run()


if __name__ == "__main__":
    main()
