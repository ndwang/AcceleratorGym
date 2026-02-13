# AcceleratorGym Design Document

## 1. Overview

AcceleratorGym is a Python library that provides a unified interface for AI agents
to monitor and control particle accelerators. It abstracts over different backends
(simulations like Bmad, real control systems like EPICS) behind a single API,
enabling LLM-based agents to interact with any accelerator through tool calling.

### Goals

- **Unified interface**: One API regardless of whether the backend is a simulation
  or a real machine.
- **LLM-agent-first**: The primary consumer is an LLM agent using function/tool
  calling. The interface must be discoverable, self-describing, and return
  structured data.
- **Config-driven**: A YAML configuration file has final authority over what
  variables are exposed to the agent, their metadata, and their safety limits.
- **Backend-swappable**: Switching from simulation to real hardware is a config
  change, not a code change.
- **Extensible**: New backends, new agent adapters (e.g., Gymnasium for RL), and
  custom safety constraints can be added without modifying the core.

### Non-Goals (for now)

- Multi-agent / concurrent access.
- Real-time streaming or event-driven interfaces.
- Gymnasium / RL adapter (deferred to future work).
- Built-in reward functions or optimization loops.

## 2. Architecture

```
┌──────────────────────────────────────────────────┐
│                  LLM Agent                       │
│         (Claude, GPT, etc. via tool use)         │
└───────────────────────┬──────────────────────────┘
                        │ tool calls / results
                        ▼
┌──────────────────────────────────────────────────┐
│                 LLMInterface                     │
│  - Generates tool schemas from Machine metadata  │
│  - Dispatches tool calls to Machine methods      │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│                   Machine                        │
│  - Variable registry (config ∪ discovery)        │
│  - get / set / get_many / set_many / reset       │
│  - Validates limits before delegating to backend │
└──────────┬───────────────────────┬───────────────┘
           │                       │
     loads │                       │ delegates
           ▼                       ▼
┌────────────────────┐  ┌─────────────────────────┐
│    YAML Config     │  │    Backend (abstract)   │
│  - variable defs   │  │  ┌─ BmadBackend         │
│  - discovery rules │  │  ├─ EPICSBackend        │
│  - backend config  │  │  └─ ...                 │
└────────────────────┘  └─────────────────────────┘
```

### Layer Responsibilities

| Layer | Responsibility |
|---|---|
| **LLMInterface** | Translate between LLM tool-calling conventions and Machine methods. Generate self-describing tool schemas. Format responses. |
| **Machine** | Central orchestrator. Owns the variable registry. Enforces safety limits. Routes reads/writes to the backend. |
| **Config** | Declares which variables exist, their metadata, limits, and whether backend discovery is enabled. Final authority on what is exposed. |
| **Backend** | Communicates with the actual accelerator (simulated or real). Handles physics computation, hardware I/O, and state management. |

## 3. Core Concepts

### 3.1 Variable

A `Variable` describes a single named parameter that can be read and optionally
written.

```python
@dataclass
class Variable:
    name: str                       # Unique identifier, e.g. "Q1:K1"
    description: str = ""           # Human-readable description
    dtype: str = "float"            # "float", "int", "bool", "array"
    units: str | None = None        # Physical units, e.g. "1/m^2"
    read_only: bool = False         # True for observables (BPMs, etc.)
    limits: tuple[float, float] | None = None  # (min, max), enforced on write
```

Variables are the fundamental abstraction. Everything the agent can see or do is
expressed as reading or writing variables.

### 3.2 Backend

The `Backend` is an abstract base class that each data source implements.

```python
class Backend(ABC):
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the accelerator/simulation."""

    @abstractmethod
    def disconnect(self) -> None:
        """Release resources and close connections."""

    @abstractmethod
    def get(self, name: str) -> float | int | bool | np.ndarray:
        """Read the current value of a variable.

        The backend must return values consistent with the current state.
        For simulations, this means re-computing if settings have changed
        since the last read (e.g., re-tracking in Bmad).
        """

    @abstractmethod
    def set(self, name: str, value: float | int | bool | np.ndarray) -> None:
        """Write a value to a variable.

        The backend may defer computation until the next get() call.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset to the initial state."""

    @abstractmethod
    def discover_variables(self) -> list[Variable]:
        """Return all variables the backend can provide.

        Called only when discovery is enabled in config. The returned
        variables provide baseline metadata that config can override.
        """
```

#### Backend Contract: Consistency on Read

For **Bmad**: after `set("Q1:K1", 2.3)`, a subsequent `get("BPM1:X")` must
reflect the new quadrupole strength. The backend tracks whether settings have
changed (dirty flag) and re-runs beam tracking before returning observable values.

For **EPICS**: `get()` reads the current process variable value from the control
system. Consistency is handled by the real machine and its feedback loops.

This keeps the agent's mental model simple: set things, read things, the physics
happens transparently.

### 3.3 Machine

The `Machine` is the central orchestrator and the only class agents interact with
(directly or via adapters).

```python
class Machine:
    def __init__(self, backend: Backend, config: MachineConfig):
        """Initialize with a backend and configuration."""

    @classmethod
    def from_config(cls, config_path: str) -> "Machine":
        """Construct a Machine from a YAML config file."""

    @property
    def variables(self) -> dict[str, Variable]:
        """All variables exposed to the agent."""

    def get(self, name: str) -> float | int | bool | np.ndarray:
        """Read a variable value."""

    def set(self, name: str, value: float | int | bool | np.ndarray) -> None:
        """Write a variable value. Validates limits before delegating."""

    def get_many(self, names: list[str]) -> dict[str, Any]:
        """Read multiple variables at once."""

    def set_many(self, values: dict[str, Any]) -> None:
        """Write multiple variables. All-or-nothing validation."""

    def reset(self) -> None:
        """Reset the machine to its initial state."""
```

#### Variable Registry Construction

The variable registry is built at initialization time:

1. Load config from YAML.
2. If `variables.discover` is present in config, call
   `backend.discover_variables()` and filter results through the include/exclude
   patterns.
3. Merge explicitly defined variables from `variables.definitions` on top of
   discovered variables. Explicit definitions override discovered metadata.
4. The resulting registry is the **complete and final** set of variables exposed
   to the agent. Anything not in the registry does not exist from the agent's
   perspective.

#### Limit Enforcement

When `set()` or `set_many()` is called, the Machine checks the value against
`variable.limits` before forwarding to the backend. If the value is out of
bounds, a `ValueError` is raised and the backend is never called. This ensures
safety rules are enforced uniformly regardless of which backend is active.

For `set_many()`, validation is all-or-nothing: all values are checked before
any are written. If any value violates limits, none are applied.

### 3.4 LLMInterface

The `LLMInterface` adapts the Machine for LLM tool calling.

```python
class LLMInterface:
    def __init__(self, machine: Machine):
        """Wrap a Machine for LLM consumption."""

    def get_tools(self) -> list[dict]:
        """Return OpenAI-compatible tool/function schemas."""

    def execute(self, tool_name: str, arguments: dict) -> dict:
        """Execute a tool call and return a structured result."""
```

#### Tool Definitions

| Tool | Parameters | Returns | Description |
|---|---|---|---|
| `list_variables` | `filter?: str` | List of variable descriptors | List available variables, optionally filtered by glob pattern |
| `get_variable` | `name: str` | `{name, value, units}` | Read a single variable |
| `get_variables` | `names: list[str]` | `{values: {name: value, ...}}` | Read multiple variables |
| `set_variable` | `name: str, value: number` | `{success, name, value}` | Write a single variable |
| `set_variables` | `values: {name: value, ...}` | `{success, values}` | Write multiple variables atomically |
| `get_state` | — | `{variables: {name: value, ...}}` | Snapshot of all readable variables |
| `reset` | — | `{success}` | Reset machine to initial state |

Tool schemas are generated dynamically from the Machine's variable registry,
including descriptions, types, and valid ranges in the schema. This makes the
tools self-documenting for the LLM.

#### Response Format

All tool responses are dicts with structured data. Errors are returned as
`{error: str, details: ...}` rather than raised as exceptions, so the LLM
can reason about failures.

## 4. Configuration

The YAML configuration file is the single source of truth for what the agent
can see and do.

### 4.1 Full Schema

```yaml
# Machine metadata
machine:
  name: str                    # Human-readable name
  description: str             # What this machine/setup is

# Backend configuration
backend:
  type: str                    # Backend type: "bmad", "epics", etc.
  # Backend-specific keys follow:
  # For bmad:
  lattice_file: str            # Path to .bmad lattice file
  # For epics:
  # prefix: str               # PV prefix

# Variable configuration
variables:
  # Optional: discover variables from the backend
  discover:
    include:                   # Glob patterns to include
      - "Q*:K1"
      - "BPM*:X"
      - "BPM*:Y"
    exclude:                   # Glob patterns to exclude (optional)
      - "BPM0:*"

  # Explicit variable definitions (override discovered metadata)
  definitions:
    "Q1:K1":
      description: "Quadrupole 1 integrated strength"
      dtype: float             # optional, default "float"
      units: "1/m"            # optional
      read_only: false         # optional, default false
      limits: [-5.0, 5.0]     # optional
    "BPM1:X":
      description: "Horizontal beam position at BPM 1"
      read_only: true
      units: "mm"
```

### 4.2 Resolution Rules

1. **No `discover` section**: only variables listed under `definitions` are
   exposed. Backend discovery is never called.
2. **`discover` with `include`**: `backend.discover_variables()` is called.
   Only variables whose names match at least one `include` pattern are kept.
3. **`discover` with `exclude`**: matched variables are removed after include
   filtering.
4. **`definitions` overlay**: for any variable that exists in both discovery
   results and `definitions`, the `definitions` values take precedence
   field-by-field. Fields not specified in `definitions` retain their
   discovered values.
5. **`definitions`-only variables**: a variable listed in `definitions` but
   not found in discovery is still created, with the backend expected to
   handle it. This allows defining variables that the discovery mechanism
   misses.

## 5. Project Structure

```
accelerator_gym/
├── pyproject.toml                 # Package metadata, dependencies
├── DESIGN.md                      # This document
├── examples/
│   └── fodo/
│       ├── fodo.yaml              # Example machine config
│       └── fodo.bmad              # Example Bmad lattice
├── src/
│   └── accelerator_gym/
│       ├── __init__.py            # Public API exports
│       ├── core/
│       │   ├── __init__.py
│       │   ├── machine.py         # Machine class
│       │   ├── variable.py        # Variable dataclass
│       │   └── config.py          # YAML config loading & validation
│       ├── backends/
│       │   ├── __init__.py        # Backend registry
│       │   ├── base.py            # Abstract Backend base class
│       │   ├── bmad.py            # Bmad/Tao backend
│       │   └── epics.py           # EPICS Channel Access backend
│       └── agents/
│           ├── __init__.py
│           └── llm.py             # LLMInterface
└── tests/
    ├── conftest.py
    ├── test_machine.py
    ├── test_config.py
    ├── test_llm_interface.py
    └── backends/
        └── test_bmad.py
```

Uses `src/` layout to avoid import ambiguity.

## 6. Error Handling

Errors are divided into two categories based on audience:

### Agent-Facing Errors (returned, not raised)

These are problems the agent can reason about and recover from:
- Variable not found
- Value out of limits
- Setting a read-only variable
- Backend returned an unexpected value

The `LLMInterface` catches these and returns structured error responses:
```python
{"error": "limit_violation", "variable": "Q1:K1", "value": 12.0,
 "limits": [-5.0, 5.0], "message": "Value 12.0 exceeds upper limit 5.0"}
```

### System Errors (raised as exceptions)

These are infrastructure problems the agent cannot fix:
- Config file not found or invalid
- Backend connection failure
- Backend communication timeout

These raise Python exceptions (`FileNotFoundError`, `ConnectionError`, etc.)
for the host application to handle.

## 7. Future Extensions

These are explicitly out of scope now but the architecture accommodates them:

- **Gymnasium adapter**: Wrap `Machine` in a `gymnasium.Env` with configurable
  observation/action spaces and reward functions. The `get_many`/`set_many`
  methods map naturally to vectorized obs/action.
- **Multi-agent access**: Add a locking/transaction layer in `Machine` between
  the public API and the backend delegation.
- **Streaming/events**: Add an event bus or callback system alongside the
  polling interface.
- **Safety constraints**: Add a pluggable constraint system beyond simple
  min/max limits (e.g., correlated constraints across multiple variables).
- **Logging/audit trail**: Machine can log all get/set calls for
  reproducibility and debugging.
