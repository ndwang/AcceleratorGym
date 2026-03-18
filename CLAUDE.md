# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AcceleratorGym is a unified interface for AI agents to monitor and control particle accelerators. It exposes a device tree via an MCP (Model Context Protocol) server, with pluggable backends for simulation (Bmad/Tao) and control systems (EPICS).

## Build & Run

```bash
pip install -e ".[test]"           # Install in dev mode with test deps
pip install -e ".[bmad]"           # Include Bmad/Tao backend

python -m pytest                   # Run all tests
python -m pytest tests/test_variable.py          # Single test file
python -m pytest tests/test_variable.py::test_name -v  # Single test

accelerator-gym                    # Start MCP server (needs accelerator-gym.yaml in cwd)
ag-cli                             # Interactive CLI for manual testing
```

## Architecture

The layered architecture enforces a strict flow: **MCP Server → Machine → Catalog/Variable → Backend**.

- **Machine** (`core/machine.py`) — Central orchestrator. Owns the variable registry, routes reads/writes to the backend, and enforces all validation (types, permissions, limits). Never bypass this layer.
- **Catalog** (`core/catalog.py`) — Builds an in-memory SQLite database from the YAML device tree. Provides tree browsing (filesystem-like paths: `/system/type/device/attribute`) and read-only SQL queries against device metadata.
- **Variable** (`core/variable.py`) — Dataclass for a single named parameter with metadata (units, limits, read/write permissions). Validation rejects booleans, enforces inclusive limit bounds, and checks permissions.
- **Backend** (`backends/base.py`) — Abstract interface (`connect/disconnect/get/set/set_many/reset/discover_devices`). Backends are lazily imported; optional deps (pytao, pyepics) are checked at runtime. Bmad backend auto-discovers elements from the lattice via `tao.lat_list()` when the YAML config has no `devices:` section.
- **MCP Server** (`server.py`) — Exposes 5 tools to AI agents: `browse_devices`, `query_devices`, `get_variables`, `set_variables`, `reset`.
- **CLI** (`cli.py`) — Interactive shell with commands: browse, query, get, gets, set, sets, reset.

### Key Design Decisions

- **Variable names are flat** (e.g., `QF:K1`) even though the tree is hierarchical. Backend `resolve_variable_name()` maps to backend-native names.
- **Bmad variable routing is attribute-based** — lattice-computed attributes (`orbit.x`, `beta.a`, `phi.a`, etc.) route to `lat::{attr}[{element}]`, global parameters route to `lat::{attr}[0]`, element control attributes route to `ele::{element}[{attr}]`. Twiss attributes are added directly to each element, not in a separate optics subtree.
- **Bmad element attributes are configurable** — defaults cover common knobs (K1 for quads, kick for correctors, etc.). Override via `element_attributes` in backend settings for lattice-specific attributes.
- **Atomic batch writes** — `set_many()` validates all values before applying any.
- **Read-only SQL** — Catalog queries only allow SELECT; schema is locked via PRAGMA after init.
- **Logging goes to stderr** so it doesn't interfere with the MCP stdout protocol.

## Git

Do not co-author in git commit messages.
