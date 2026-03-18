# AccelBench

Benchmark harness for evaluating AI agents on particle accelerator operations. Defines 27 tasks across 4 difficulty tiers, runs agents against a Bmad simulation backend, and produces scored reports.

## Quick Start

```bash
pip install -e ".[bench,bmad]"     # Install with benchmark + Bmad deps

accelbench list                    # Show all 27 tasks
accelbench list --tier 1           # Show only Tier 1 tasks

accelbench run --config examples/booster/accelerator-gym.yaml --seed 42
accelbench run --config ... --tasks 1.1,1.2,1.3
accelbench run --config ... --tier 1
accelbench run --config ... --output results.json
```

## Architecture

```
Agent  ──►  Adapter  ──►  InstrumentedMachine  ──►  Machine  ──►  Backend
               │                  │
               │            counts calls,
               │            enforces budget
               │
         call_tool(name, args) -> str
```

The harness is agent-agnostic. Any agent can be benchmarked by implementing a thin Python adapter. The `InstrumentedMachine` wraps the real `Machine`, counting each tool-level call and enforcing per-task budgets. One call to `get_variables(["a", "b"])` counts as 1 tool call, not 2.

## Tasks

27 tasks across 4 tiers, testing 7 abilities:

| Tier | Name | Tasks | Budget | What it tests |
|------|------|-------|--------|---------------|
| 1 | Direct | 1.1–1.6 | 3 | Basic interface competence — read, write, browse, query |
| 2 | Procedural | 2.1–2.7 | 5–12 | Known multi-step procedures with deterministic sequencing |
| 3 | Adaptive | 3.1–3.8 | 15–40 | Next action depends on observations; requires discovery and iteration |
| 4 | Complex | 4.1–4.6 | 80–200 | Multi-procedure compositions with competing objectives |

Each task defines:

- **Prompt template** with placeholders filled at runtime from randomized setup
- **Setup function** that configures the machine state (perturbations, etc.)
- **Verify function** that checks the agent's answer and/or machine state
- **Budget** — maximum tool calls before automatic failure
- **Abilities** — tagged capabilities tested (io, discovery, analysis, measurement, physics, optimization, diagnosis)

## Writing an Adapter

Implement the `AgentAdapter` protocol:

```python
from accelbench.adapters.base import AgentAdapter

class MyAgent:
    def run(self, prompt: str, tools: list[dict], call_tool) -> str:
        """
        Args:
            prompt: Natural language task description.
            tools: JSON schemas for the 5 MCP tools (browse_devices,
                   query_devices, get_variables, set_variables, reset).
            call_tool: Execute a tool — call_tool("get_variables", {"names": ["QF:K1"]}) -> str

        Returns:
            Final text response. Must contain a JSON answer inside a
            ```json fenced code block.
        """
        # Your agent logic here
        response = call_tool("get_variables", {"names": ["QF:K1"]})
        return '```json\n{"value": 0.5}\n```'
```

Then run it:

```bash
accelbench run --config path/to/config.yaml \
    --adapter mypackage.agent.MyAgent
```

The adapter class is imported and instantiated with no arguments. If your adapter needs constructor parameters, subclass it with defaults or use a wrapper.

## Built-in Adapters

### `accelbench.adapters.claude_sdk.ClaudeSDKAdapter`

Reference implementation using the Anthropic Python SDK (callback mode). Sends the prompt and tool schemas to Claude via the Messages API, executes tool calls in a loop, and returns the final text response.

Requires the `ANTHROPIC_API_KEY` environment variable and the `anthropic` package (`pip install -e ".[bench]"`).

```bash
accelbench run --config ... --adapter accelbench.adapters.claude_sdk.ClaudeSDKAdapter
```

### `accelbench.adapters.claude_code.ClaudeCodeAdapter`

Runs Claude Code CLI against an MCP server subprocess. Starts an `accelerator-gym` MCP server, writes a temporary MCP config, and pipes the prompt to the `claude` CLI via stdin.

Requires the `claude` CLI to be installed and configured.

## CLI Reference

### `accelbench list`

List available benchmark tasks.

```
accelbench list                    # All tasks
accelbench list --tier 2           # Only Tier 2
```

Output columns: ID, Name, Tier, Budget, Abilities.

### `accelbench run`

Run benchmark tasks and produce a scored report.

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | (required) | Path to `accelerator-gym.yaml` |
| `--seed` | 42 | Random seed for reproducible setups |
| `--adapter` | `...ClaudeSDKAdapter` | Fully qualified adapter class name |
| `--tasks` | all | Comma-separated task IDs (e.g. `1.1,2.3`) |
| `--tier` | all | Run only tasks from this tier |
| `--output` | none | Path to save JSON report |
| `--debug` | off | Enable debug logging on stderr |

## Report Format

The report (printed to stdout and optionally saved as JSON) contains:

- **Summary**: total/passed/failed/error counts and pass rate
- **Per-tier breakdown**: pass counts for each difficulty tier
- **Per-ability pass rates**: how well the agent performs on each capability
- **Per-task details**: tool calls used, budget, efficiency score, wall time, pass/fail, errors

Example output:

```
============================================================
AccelBench Results: 6/6 passed (100%)
============================================================

Per-Tier Breakdown:
  Tier 1 (Direct): 6/6

Ability Pass Rates:
  analysis       : 100%
  discovery      : 100%
  io             : 100%

Task Details:
  1.1   Read a Parameter                          PASS  tools:   2/  3  time:    3.2s
  1.2   Set a Parameter                           PASS  tools:   2/  3  time:    2.8s
  ...
```

## Verification

Tasks use two verification strategies:

- **Report tasks**: The agent returns information. Verification checks `result` (extracted JSON) against ground truth computed from the machine.
- **Action tasks**: The agent changes machine state. Verification reads back the machine state after the agent finishes.

Ground truth is computed from the Bmad backend directly. Tolerances are generous — the benchmark tests reasoning ability, not numerical precision.

## Design Values

Some verification functions need reference (unperturbed) values. The `Env.get_design(name)` method reads design values from Tao using the `|design` suffix, without needing a separate machine instance.

```python
# In a verify function:
design_k1 = env.get_design("ele::QF[K1]")  # Unperturbed value
current_k1 = env.machine.get("QF:K1")       # Current (possibly perturbed) value
```

## Tool Call Counting

- All calls count, including failed or redundant calls
- Exceeding the budget is an automatic failure
- Efficiency score: `max(0, 1 - calls_used / budget)` — a bonus metric, not pass/fail
- Budget enforcement returns `"Error: tool call budget exceeded"` for any call beyond the limit

## Adding New Tasks

Add a task definition to the appropriate tier file in `src/accelbench/tasks/`:

```python
def _setup_X_Y(env: Env) -> dict[str, Any]:
    """Configure machine state, return template params."""
    return {"element": "QF", "target": 0.5}

def _verify_X_Y(result: dict, env: Env, setup: dict) -> bool:
    """Check agent's answer and/or machine state."""
    return abs(env.machine.get("QF:K1") - setup["target"]) < 1e-8

TIER_TASKS.append(TaskDef(
    id="X.Y",
    name="My New Task",
    tier=X,
    prompt_template="Set K1 of {element} to {target}.",
    budget=5,
    abilities=["io"],
    setup=_setup_X_Y,
    verify=_verify_X_Y,
))
```

The task is automatically picked up by `ALL_TASKS` via the tier module imports.
