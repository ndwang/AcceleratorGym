# AcceleratorGym

A unified interface for AI agents to monitor and control particle accelerators. AcceleratorGym exposes a device tree via an [MCP](https://modelcontextprotocol.io/) (Model Context Protocol) server, with a pluggable backend for simulation (Bmad/Tao).

It also includes **AccelBench**, a benchmark harness with 27 tasks across 4 difficulty tiers for evaluating how well AI agents operate accelerator control systems.

## Installation

Requires Python ≥ 3.10.

```bash
pip install -e .                   # Core package
pip install -e ".[bmad]"           # + Bmad/Tao simulation backend
pip install -e ".[bench]"          # + AccelBench dependencies (litellm)
pip install -e ".[test]"           # + Test dependencies (pytest)
```

## Quick Start

### 1. Define a machine

Create an `accelerator-gym.yaml` configuration file describing your accelerator:

```yaml
machine:
  name: my-accelerator
  description: A simple FODO lattice

backend:
  type: bmad
  init_file: tao.init

devices:
  magnets:
    quadrupole:
      QF:
        description: Focusing quadrupole
        s_position: 1.0
        attributes:
          K1:
            description: Quadrupole strength
            units: 1/m^2
            limits: [-10, 10]
            read: true
            write: true
```

With the Bmad backend, you can omit the `devices` section entirely — elements are auto-discovered from the lattice.

See the `examples/` directory for working configurations (mock, FODO cell, BNL Booster ring).

### 2. Start the MCP server

```bash
accelerator-gym                    # Reads accelerator-gym.yaml from cwd
accelerator-gym --config path.yaml # Or specify a config file
```

The server exposes five tools to AI agents over MCP:

| Tool | Description |
|------|-------------|
| `browse_devices` | Navigate the device hierarchy (filesystem-like paths) |
| `query_devices` | Run read-only SQL queries against device metadata |
| `get_variables` | Read one or more variable values |
| `set_variables` | Atomically write variable values (validates all before applying) |
| `reset` | Reset the machine to its initial state |

### 3. Or use the interactive CLI

```bash
ag-cli                             # Auto-discovers accelerator-gym.yaml
ag-cli --config path.yaml
```

Commands: `browse`, `query`, `get`, `gets`, `set`, `sets`, `reset`. See [docs/CLI.md](docs/CLI.md) for the full reference.

## Architecture

```
MCP Server / CLI
       │
    Machine          ← Central orchestrator, validation, variable registry
    ├── Catalog      ← In-memory SQLite of the device tree
    ├── Variable[]   ← Parameter metadata (units, limits, permissions)
    └── Backend      ← Abstract interface to the accelerator
         ├── BmadBackend   (Bmad/Tao simulation)
         └── MockBackend   (in-memory, for testing)
```

**Key design points:**
- Variable names are flat (`QF:K1`) even though the tree is hierarchical (`/magnets/quadrupole/QF/K1`).
- Batch writes are atomic — all values validated before any are applied.
- SQL queries are read-only (SELECT only).
- Logging goes to stderr to avoid interfering with the MCP stdout protocol.

## AccelBench

AccelBench evaluates AI agents on 27 accelerator control tasks across 4 tiers of increasing difficulty:

| Tier | Name | Tasks | Budget | What it tests |
|------|------|-------|--------|---------------|
| 1 | Direct | 6 | 3 | Basic I/O — read and write variables |
| 2 | Procedural | 7 | 5–12 | Multi-step sequences with deterministic procedures |
| 3 | Adaptive | 8 | 15–40 | Observation-based iteration and discovery |
| 4 | Complex | 6 | 80–200 | Multi-procedure tasks with competing objectives |

### Running benchmarks

```bash
accelbench list                    # List all tasks (--tier N to filter)
accelbench run --config bench.yaml # Run benchmark suite
```

Agents implement a simple adapter interface:

```python
class AgentAdapter:
    def run(self, prompt: str, tools: list[dict], call_tool: Callable) -> str:
        ...
```

Built-in adapters: **LiteLLM** (100+ LLM providers) and **Claude Code** (CLI subprocess).

See [docs/AccelBench.md](docs/AccelBench.md) for full documentation, including how to write custom adapters and add new tasks.

## Testing

```bash
pip install -e ".[test]"
python -m pytest                   # All tests
python -m pytest tests/test_machine.py -v  # Single file
```
