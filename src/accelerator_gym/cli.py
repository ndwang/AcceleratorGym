"""Interactive CLI for manually testing accelerator-gym tools."""

from __future__ import annotations

import argparse
import json
import logging
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


def _fmt(obj: Any) -> str:
    """Pretty-print a result dict."""
    return json.dumps(obj, indent=2)


class CLI:
    def __init__(self, machine: Machine) -> None:
        self._machine = machine

    def run(self) -> None:
        print(f"accelerator-gym CLI — {self._machine._config.name}")
        print(f"{len(self._machine.variables)} variables loaded")
        print('Type "help" for commands.\n')

        while True:
            try:
                line = input("ag> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            try:
                parts = shlex.split(line)
            except ValueError as e:
                print(f"Parse error: {e}")
                continue
            cmd, args = parts[0], parts[1:]
            handler = getattr(self, f"cmd_{cmd}", None)
            if handler is None:
                print(f"Unknown command: {cmd}. Type 'help' for commands.")
                continue
            try:
                handler(args)
            except Exception as e:
                print(f"Error: {e}")

    def cmd_help(self, args: list[str]) -> None:
        for desc in COMMANDS.values():
            print(f"  {desc}")

    def cmd_quit(self, args: list[str]) -> None:
        raise SystemExit(0)

    def cmd_browse(self, args: list[str]) -> None:
        path = args[0] if args else "/"
        depth = int(args[1]) if len(args) > 1 else 1
        result = self._machine.catalog.browse(path, depth)
        print(_fmt(result))

    def cmd_query(self, args: list[str]) -> None:
        if not args:
            print("Usage: query <sql>")
            return
        sql = " ".join(args)
        try:
            rows = self._machine.catalog.query(sql)
            print(_fmt({"rows": rows, "count": len(rows)}))
        except ValueError as e:
            print(f"Query error: {e}")

    def cmd_get(self, args: list[str]) -> None:
        if not args:
            print("Usage: get <variable>")
            return
        name = args[0]
        var = self._machine.variables.get(name)
        if var is None:
            print(f"Unknown variable: {name}")
            return
        value = self._machine.get(name)
        result: dict[str, Any] = {"name": name, "value": value}
        if var.units:
            result["units"] = var.units
        print(_fmt(result))

    def cmd_gets(self, args: list[str]) -> None:
        if not args:
            print("Usage: gets <var1> <var2> ...")
            return
        values = self._machine.get_many(args)
        print(_fmt({"values": values}))

    def cmd_set(self, args: list[str]) -> None:
        if len(args) < 2:
            print("Usage: set <variable> <value>")
            return
        name = args[0]
        value = float(args[1])
        self._machine.set(name, value)
        print(_fmt({"success": True, "name": name, "value": value}))

    def cmd_sets(self, args: list[str]) -> None:
        if not args:
            print("Usage: sets <var>=<val> ...")
            return
        values: dict[str, float] = {}
        for pair in args:
            if "=" not in pair:
                print(f"Invalid format: {pair}. Use var=value")
                return
            k, v = pair.split("=", 1)
            values[k] = float(v)
        self._machine.set_many(values)
        print(_fmt({"success": True, "values": values}))

    def cmd_state(self, args: list[str]) -> None:
        readable = [n for n, v in self._machine.variables.items() if v.readable]
        values = self._machine.get_many(readable)
        print(_fmt({"variables": values}))

    def cmd_reset(self, args: list[str]) -> None:
        self._machine.reset()
        print(_fmt({"success": True}))


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
