# AccelBench

Benchmark harness for evaluating AI agents on particle accelerator operations. Defines 27 tasks across 4 difficulty tiers, runs agents against a Bmad simulation backend, and produces scored reports.

## Quick Start

```bash
pip install -e ".[bench,bmad]"     # Install with benchmark + Bmad deps

accelbench list                    # Show all 27 tasks
accelbench list --tier 1           # Show only Tier 1 tasks

accelbench run --config examples/booster/accelerator-gym.yaml --seed 42
accelbench run --config ... --model anthropic/claude-sonnet-4-20250514
accelbench run --config ... --model gemini/gemini-2.5-pro
accelbench run --config ... --tasks 1.1,1.2,1.3
accelbench run --config ... --tier 1
accelbench run --config ... --output-dir results/
accelbench run --config ... --workers 8          # Run tasks in parallel (non-Bmad backends only)
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

See [AccelBench-tasks.md](AccelBench-tasks.md) for the full task spec — tool interface, device tree structure, verification protocol, and per-task details.

## Implementing a Custom Adapter

To benchmark your own agent, implement a single Python class with one method: `run()`. The benchmark calls this method once per task, providing everything the agent needs to interact with the accelerator.

### What your adapter receives

Your `run()` method gets three arguments:

1. **`prompt`** — A natural language task description (e.g., "What is K1 of QF?" or "Correct the horizontal orbit to minimize the RMS across all BPMs.").

2. **`tools`** — A list of JSON schemas describing the 5 available accelerator tools:
   - `browse_devices(path, depth)` — Navigate the device tree to discover what's available
   - `query_devices(sql)` — Run SQL queries against device metadata
   - `get_variables(names)` — Read one or more variable values
   - `set_variables(values)` — Write one or more variable values (atomic)
   - `reset()` — Reset the machine to its initial state

3. **`call_tool`** — A function your agent calls to execute tools. It takes a tool name and arguments dict, and returns a string result:
   ```python
   result = call_tool("get_variables", {"names": ["QF:K1", "QD:K1"]})
   # result: "QF:K1 = 0.5 (1/m^2)\nQD:K1 = -0.3 (1/m^2)"
   ```

### What your adapter returns

A string containing the agent's final answer as a JSON object inside a fenced code block tagged `json`:

    Based on my analysis, the K1 value is 0.5.

    ```json
    {"value": 0.5}
    ```

The benchmark extracts the JSON and passes it to the task's verification function.

### Minimal example

````python
class MyAdapter:
    def run(self, prompt, tools, call_tool):
        # Your agent decides what to do based on the prompt and tools.
        # It calls call_tool() to interact with the accelerator.
        result = call_tool("get_variables", {"names": ["QF:K1"]})

        # Parse the result and return a JSON answer.
        value = float(result.split("=")[1].split("(")[0].strip())
        return f'```json\n{{"value": {value}}}\n```'
````

### Wrapping an LLM-based agent

Most agents are LLM-powered tool-calling loops. The adapter bridges `call_tool` to whatever interface the agent expects:

```python
class MyLLMAgentAdapter:
    def __init__(self, model="gpt-4o"):
        self.model = model

    def run(self, prompt, tools, call_tool):
        # Initialize your agent with the tools
        agent = MyAgent(model=self.model)

        # Convert AccelBench tools to your agent's format.
        # The key contract: when your agent wants to call a tool,
        # it must go through call_tool() so the benchmark can
        # count the call and enforce the budget.
        for tool_schema in tools:
            fn = tool_schema["function"]
            agent.register_tool(
                name=fn["name"],
                description=fn["description"],
                handler=lambda name, args: call_tool(name, args),
            )

        # Run the agent and return its final response
        return agent.chat(prompt)
```

### Wrapping a framework-based agent (LangChain, etc.)

```python
from langchain.tools import StructuredTool

class LangChainAdapter:
    def run(self, prompt, tools, call_tool):
        # Convert call_tool into LangChain tools
        lc_tools = []
        for schema in tools:
            fn = schema["function"]
            name = fn["name"]
            lc_tools.append(StructuredTool.from_function(
                func=lambda args, _name=name: call_tool(_name, args),
                name=name,
                description=fn["description"],
            ))

        agent = create_your_agent(lc_tools)
        return agent.invoke(prompt)
```

### Running your adapter

```bash
# Install your adapter package so it's importable
pip install -e ./my-agent

# Run the benchmark
accelbench run --config examples/booster/accelerator-gym.yaml \
    --adapter my_agent.adapter.MyLLMAgentAdapter \
    --adapter-arg model=gpt-4o \
    --output-dir results/my_agent
```

The `--adapter-arg` flag passes keyword arguments to your adapter's constructor. It is repeatable:

```bash
accelbench run --config ... \
    --adapter my_agent.MyAdapter \
    --adapter-arg model=gpt-4o \
    --adapter-arg temperature=0.5 \
    --adapter-arg api_base=http://localhost:8000
```

All values are passed as strings. Your constructor should convert types as needed.

### Important rules for adapters

- **Always use `call_tool`** to interact with the accelerator. Never bypass it — this is how the benchmark counts tool calls and enforces budgets.
- **Return a JSON answer** in a fenced code block tagged `json`. If the benchmark can't extract JSON from your response, the task fails.
- **Don't cache or share state between tasks.** Each `run()` call is independent. The benchmark creates a fresh machine for each task.
- **Budget-exceeded errors are returned as strings.** When the budget is exhausted, `call_tool` returns `"Error: tool call budget exceeded"`. Your agent should produce a final answer when it sees this.

## Built-in Adapters

There are two ways an adapter can interact with the accelerator, depending on whether the agent runs in the same process or as a subprocess.

### In-process adapters (default)

The agent runs in the harness process and calls `call_tool()` directly. The `InstrumentedMachine` counts calls, enforces the budget, and records a trace — all transparently. Verification reads machine state directly after the agent finishes.

**`accelbench.adapters.litellm.LiteLLMAdapter`** is the default in-process adapter, using [LiteLLM](https://docs.litellm.ai/docs/providers) for provider-agnostic model access. Supports 100+ providers — pass any model via `--model`:

```bash
# OpenAI (default)
OPENAI_API_KEY=... accelbench run --config ...

# Anthropic
ANTHROPIC_API_KEY=... accelbench run --config ... --model anthropic/claude-sonnet-4-20250514

# Gemini
GEMINI_API_KEY=... accelbench run --config ... --model gemini/gemini-2.5-pro

# Mistral
MISTRAL_API_KEY=... accelbench run --config ... --model mistral/mistral-large-latest

# Local Ollama
accelbench run --config ... --model ollama/llama3
```

Requires `pip install -e ".[bench]"` and the appropriate API key env var for your provider.

### Subprocess adapters (trace replay)

Some agents manage their own server connection and can't use `call_tool()` directly. For these, the adapter launches a `bench_server` subprocess — a standalone MCP server that handles task setup, budget enforcement, and trace recording internally. The agent connects to this server instead.

After the agent finishes, the adapter reads back a trace file containing all tool calls the agent made. The harness replays `set_variables` and `reset` calls from this trace onto its own Machine to reconstruct the post-agent state for verification.

A subprocess adapter exposes a `last_trace` property that returns the trace data. The harness detects this automatically:

```python
class MySubprocessAdapter:
    @property
    def last_trace(self):
        """Return trace dict with 'call_count' and 'trace' keys, or None."""
        return self._last_trace

    def set_task_context(self, task_id, seed, budget):
        """Called by the harness before each run() with task parameters."""
        ...
```

**`accelbench.adapters.claude_code.ClaudeCodeAdapter`** is the built-in subprocess adapter. It launches `bench_server` as an MCP server, writes a temporary MCP config, and pipes the prompt to the `claude` CLI:

```bash
accelbench run --config examples/booster/accelerator-gym.yaml \
    --adapter accelbench.adapters.claude_code.ClaudeCodeAdapter \
    --adapter-arg config_path=examples/booster/accelerator-gym.yaml \
    --adapter-arg model=claude-sonnet-4-20250514
```

Requires the `claude` CLI to be installed and configured.

## CLI Reference

### `accelbench list`

List available benchmark tasks.

```
accelbench list                    # All tasks
accelbench list --tier 2           # Only Tier 2
```

Output columns: ID, Name, Tier, Budget.

### `accelbench run`

Run benchmark tasks and produce a scored report.

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | (required) | Path to `accelerator-gym.yaml` |
| `--seed` | 42 | Random seed for reproducible setups |
| `--model` | `gpt-4o` | Model name for the default LiteLLM adapter |
| `--adapter` | `LiteLLMAdapter` | Fully qualified adapter class name (overrides `--model`) |
| `--adapter-arg` | none | `KEY=VALUE` constructor args for custom adapters (repeatable) |
| `--tasks` | all | Comma-separated task IDs (e.g. `1.1,2.3`) |
| `--tier` | all | Run only tasks from this tier |
| `--output-dir` | none | Directory for report and per-task trajectory files |
| `--timeout` | 600 | Wall-clock timeout in seconds per task |
| `--workers` | 1 | Number of tasks to run in parallel |
| `--debug` | off | Enable debug logging on stderr |

### Parallel execution

Use `--workers N` to run N tasks concurrently via threads:

```bash
accelbench run --config ... --workers 8
```

**Limitation:** The Bmad backend uses pytao, which does not support multiple Tao instances in the same process. Since the harness creates a Machine (and thus a Tao instance) per task for setup and verification, `--workers` > 1 will crash with Bmad. Use `--workers 1` (the default) for Bmad-backed configs. Other backends that support concurrent instances work fine with parallel execution.

## Output Format

A summary is always printed to stdout. When `--output-dir` is given, the harness writes:

```
results/
├── report.json              # Aggregate scoring report
└── traces/
    ├── task_1_1.json         # Full trajectory for task 1.1
    ├── task_1_2.json
    └── ...
```

### `report.json` — Aggregate Scores

Contains summary stats, per-tier breakdown, and per-task score lines (no traces — kept lean for quick comparison).

### `traces/task_X_Y.json` — Per-Task Trajectories

Full record of a single task run:

````json
{
  "task_id": "1.1",
  "task_name": "Read a Parameter",
  "tier": 1,
  "passed": true,
  "tool_calls": 2,
  "budget": 3,
  "efficiency": 0.333,
  "wall_time": 3.21,
  "error": null,
  "prompt": "What is K1 of QVA1? ...",
  "response": "The K1 of QVA1 is ... ```json\n{\"value\": 0.5}\n```",
  "extracted_answer": {"value": 0.5},
  "setup_data": {"element": "QVA1", "variable": "QVA1:K1"},
  "trace": [
    {"tool": "browse_devices", "arguments": {"path": "/", "depth": 2}, "result": "..."},
    {"tool": "get_variables", "arguments": {"names": ["QVA1:K1"]}, "result": "QVA1:K1 = 0.5 (1/m^2)"}
  ]
}
````

### Console Output

```
============================================================
AccelBench Results: 6/6 passed (100%)
============================================================

Per-Tier Breakdown:
  Tier 1 (Direct): 6/6

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

## Tool Call Counting

- All calls count, including failed or redundant calls
- Exceeding the budget is an automatic failure
- Efficiency score: `max(0, 1 - calls_used / budget)` — a bonus metric, not pass/fail
- Budget enforcement returns `"Error: tool call budget exceeded"` for any call beyond the limit

## No Built-in Code Execution

AccelBench provides only 5 domain-specific tools: `browse_devices`, `query_devices`, `get_variables`, `set_variables`, and `reset`. It does **not** provide a code execution environment (Python interpreter, shell, etc.).

Higher-tier tasks may involve data volumes or calculations that are infeasible to perform manually — orbit response matrices with hundreds of BPMs, high-dimensional optimization, numerical physics calculations. Agents with code execution capabilities (e.g., Claude Code with bash, Codex with its sandbox) will have an advantage on these tasks, and **that is by design**. Code execution is an intrinsic harness capability, and we want to measure the agent's ability to leverage it.

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
    setup=_setup_X_Y,
    verify=_verify_X_Y,
))
```

The task is automatically picked up by `ALL_TASKS` via the tier module imports.

### Design Values

Some verification functions need reference (unperturbed) values. The `Env.get_design(name)` method reads design values from Tao using the `|design` suffix, without needing a separate machine instance.

```python
# In a verify function:
design_k1 = env.get_design("ele::QF[K1]")  # Unperturbed value
current_k1 = env.machine.get("QF:K1")       # Current (possibly perturbed) value
```
