# AccelBench Task Coverage

## Overview

This benchmark evaluates AI agents on accelerator physics operations tasks against a Bmad simulation backend. All tasks are programmatically verifiable — each has a natural-language prompt, a verification function, and a maximum tool-call budget.

Task difficulty correlates with two factors: the degree of adaptive reasoning required, and the degree of discovery required. Lower tiers give exact element names and map to fixed procedures. Higher tiers use vague or symptomatic descriptions and require the agent to discover targets through search, adapt based on intermediate results, and compose multiple sub-procedures.

| Tier | Name | Tasks | What it tests |
|------|------|-------|---------------|
| 1 | Direct | 10 | Single tool call — read, write, browse, query, reset |
| 2 | Procedural | 8 | Deterministic multi-step procedures |
| 3 | Adaptive | 7 | Discovery or iterative optimization based on observations |
| 4 | Complex | 2 | Multi-procedure compositions with competing objectives |

---

## Tool Interface

The agent interacts with the accelerator through five MCP tools. There are no physics-specific convenience tools — the agent must compose procedures from these primitives.

```
browse_devices(path, depth)     Navigate the device tree to discover what exists
query_devices(sql)              SQL against device/attribute metadata
get_variables(names)            Read one or more variable values
set_variables(values)           Write one or more variables (atomic, all-or-nothing)
reset()                         Restore machine to initial state
```

### Device Tree Structure

The device tree exposes actuators, sensors, and model-computed quantities through the same interface:

```
/magnets/quadrupole/{name}/K1           read+write   Quadrupole strength
/magnets/sbend/{name}/ANGLE             read+write   Dipole bend angle
/magnets/hkicker/{name}/KICK            read+write   Horizontal corrector kick
/magnets/vkicker/{name}/KICK            read+write   Vertical corrector kick
/magnets/sextupole/{name}/K2            read+write   Sextupole strength
/diagnostics/monitor/{name}/orbit.x     read-only    Horizontal orbit at BPM
/diagnostics/monitor/{name}/orbit.y     read-only    Vertical orbit at BPM

# Global lattice parameters:
/global/lattice/params/tune.a            read-only    Horizontal tune
/global/lattice/params/tune.b            read-only    Vertical tune
/global/lattice/params/chrom.a           read-only    Horizontal chromaticity
/global/lattice/params/chrom.b           read-only    Vertical chromaticity
/global/lattice/params/e_tot             read-only    Beam energy

# Twiss attributes — added to EVERY element (quads, dipoles, correctors, BPMs, ...):
{any element path}/beta.a               read-only    Horizontal beta function
{any element path}/beta.b               read-only    Vertical beta function
{any element path}/alpha.a              read-only    Horizontal alpha function
{any element path}/alpha.b              read-only    Vertical alpha function
{any element path}/phi.a                read-only    Horizontal betatron phase
{any element path}/phi.b                read-only    Vertical betatron phase
{any element path}/eta.x                read-only    Horizontal dispersion
```

Twiss attributes (beta, phase advance, dispersion) are attached directly to every element. This means a quadrupole has both its control knob (`K1`) and its computed optics (`beta.a`, `phi.a`, etc.) in the same place.

Variable names used by `get_variables` / `set_variables` are flat identifiers (e.g., `QF:K1`, `params:tune.a`, `BPM1:orbit.x`) that do **not** match tree paths. The tree path `/magnets/quadrupole/QF/K1` corresponds to variable `QF:K1`. An agent must discover the mapping — browse or query to find a device, then read/write using its variable name from the `variable` column in the attributes table.

### SQL Schema

```sql
devices(device_id, system, device_type, s_position, tree_path)
attributes(device_id, attribute_name, description, value, unit,
           readable, writable, lower_limit, upper_limit, variable)
```

---

## Verification Protocol

Every task uses a uniform verification signature:

```python
def verify(result: dict, env: Env) -> bool:
    """
    result: structured dict extracted from the agent's final response
    env.machine: Machine instance (current state, post-agent)
    env.get_design(name): read unperturbed reference values
    """
```

Tasks fall into two categories:

- **Report tasks**: Agent must return information. Verification checks `result` against ground truth computed from `env`.
- **Action tasks**: Agent must change machine state. Verification reads back `env.machine` state. `result` may contain supplementary info (e.g., diagnosis) but the primary check is on machine state.

### Answer Extraction

The agent's final response must include a JSON object inside a fenced code block tagged `json`:

    Based on my analysis, the K1 value is 0.5.

    ```json
    {"value": 0.5}
    ```

The harness extracts this JSON and passes it to the task's verification function as `result`. Each task's prompt template specifies the expected keys (e.g., `{"value": <number>}`, `{"count": <integer>}`, `{"status": "done"}`). If the harness cannot extract valid JSON, the task fails.

---

## Tier 1: Direct (Budget: 3)

**Goal:** Verify the agent can use each interface primitive in isolation.

There are 5 MCP tools: `browse_devices`, `query_devices`, `get_variables`,
`set_variables`, `reset`. Tier 1 should cover each tool, plus basic variations
(single vs. batch, magnet vs. diagnostic).

Each task must be **truly single-step**: one tool call, testing one thing.
The prompt must provide enough context that no prior discovery is needed.
**Prompts must give the full variable name** (e.g., `QF:K1`, not just `QF`)
so the agent doesn't need to guess or discover the naming convention.

### Tasks (10)

| ID   | Name              | MCP Tool       | Prompt (template)                                              | Notes |
|------|-------------------|----------------|----------------------------------------------------------------|-------|
| 1.1  | Read a Variable   | get_variables  | "What is the value of `{variable}`?"                           | Randomize across magnet settings, diagnostics, and global parameters. |
| 1.2  | Batch Read        | get_variables  | "Read the values of `{var1}`, `{var2}`, and `{var3}`."         | Pick variables from different categories. |
| 1.3  | Set a Variable    | set_variables  | "Set `{variable}` to {target}."                                | Single write. |
| 1.4  | Batch Set         | set_variables  | "Set `{var1}` to {val1} and `{var2}` to {val2}."              | Batch write in a single call. |
| 1.5  | Reset Machine     | reset          | "Reset the machine to its default state."                      | Verify perturbed variable returns to design value. |
| 1.6  | Count Aggregation | query_devices  | "How many devices of type `{device_type}` are in the ring?"    | COUNT + WHERE. Returns a single number. |
| 1.7  | Attribute Lookup  | query_devices  | "What is the s-position of device `{device_id}`?"              | WHERE on primary key. Returns a single value. |
| 1.8  | Range Filter      | query_devices  | "List all devices with s-position between {lo} and {hi} meters." | Range condition. Returns a result set. |
| 1.9  | Browse Root       | browse_devices | "What are the top-level categories in the device tree?"        | Root-level tree navigation. |
| 1.10 | Browse Leaf       | browse_devices | "What attributes are available for the device at `{path}`?"    | Leaf-level inspection. Prompt gives exact tree path. |

### Design Notes

- All prompts give exact variable names, device IDs, paths, and type strings.
  No discovery or name construction required.
- The single-read task (1.1) randomizes across variable categories (magnet
  setting, diagnostic, global parameter) so one task covers the variety that
  previously required three separate tasks (old 1.1, 1.3, 1.5).

---

## Tier 2: Procedural (Budget: 5–12)

**Goal:** Known multi-step procedures with deterministic sequencing.

### Tasks (8)

Tier 2 tasks are deterministic multi-step procedures. The agent knows what to
do in advance — no discovery or adaptive reasoning required. Prompts spell out
the exact procedure. Variable names are provided where needed.

Five distinct procedural patterns, with physics variety:

| ID  | Name                           | Pattern              | Prompt (template)                                              | Notes |
|-----|--------------------------------|----------------------|----------------------------------------------------------------|-------|
| 2.1 | Maximum Beta Function          | Batch read + compute | "Read the horizontal beta function at all elements and report the maximum value and where it occurs." | Argmax. Tests optics knowledge. |
| 2.2 | Orbit RMS                      | Batch read + compute | "Read the horizontal orbit at all BPMs and report the RMS."   | RMS. Requires calculator for many values. |
| 2.3 | Corrector Orbit Response       | Set-read-restore     | "Apply a kick of 0.1 mrad to `{corrector_var}`, read the horizontal orbit at all BPMs, then restore the corrector. Report the orbit change." | Tests corrector → orbit relationship. |
| 2.4 | Quad Tune Shift                | Set-read-restore     | "Change `{quad_var}` by +1%, read the horizontal tune, then restore. Report the tune change." | Tests quad → tune relationship. |
| 2.5 | Increase Magnet Strength       | Read-then-set        | "Read the current value of `{variable}`, increase it by {pct}%, and set the new value." | Read informs the write. |
| 2.6 | Zero Correctors and Read Orbit | Batch set + batch read | "Set all horizontal correctors to zero, then read the horizontal orbit at all BPMs." | Coordinate many writes then observe. |
| 2.7 | Corrector Scan                 | Predetermined scan   | "Scan `{corrector_var}` from 0 to {max_kick} mrad in {n} steps. At each step, read the horizontal orbit at `{bpm_var}`. Report the list of (kick, orbit) pairs. Restore the corrector when done." | Longest Tier 2 task. |
| 2.8 | Find and Zero a Corrector      | Query + set          | "One of the horizontal correctors has a nonzero kick. Find it using `query_devices` and set it to zero." | Discovery via SQL, then act. Moved from old Tier 3. |

### Notes

- All prompts give full variable names except 2.8, which requires a query to
  discover the corrector name. This is the only Tier 2 task with a discovery
  step, but the discovery is deterministic (one query suffices).

---

## Tier 3: Adaptive (Budget: 15–40)

**Goal:** Agent must either discover the machine structure before acting, or
iteratively optimize toward a target. Two categories:

1. **Discovery → act:** Prompts use natural language (not exact variable
   names). Agent must navigate browse/query to find the right devices and
   variables, then reason about what to read or write.
2. **Iterative optimization:** Agent is told what knobs to use but must close
   a feedback loop — adjust, observe, repeat until target is met.

### Tasks (7)

| ID  | Name                        | Category    | Prompt (template)                                              | Notes |
|-----|-----------------------------|-------------|----------------------------------------------------------------|-------|
| 3.1 | Find the Tune               | Discovery   | "What is the tune of the ring?"                                | Must find `params:tune.a` without being told the variable name. |
| 3.2 | List Elements and Settings  | Discovery   | "List all horizontal correctors and their current kick values." | Discover element type + attribute, batch read. |
| 3.3 | Nearest Quad to BPM         | Discovery   | "Which quadrupole is closest to `{bpm}`?"                      | Cross-reference s-positions across element types. |
| 3.4 | Elements in a Range         | Discovery   | "What magnets are between `{bpm1}` and `{bpm2}`?"             | Range query with type filtering. |
| 3.5 | Tune Adjustment             | Optimization| "Adjust the horizontal tune to 4.82 ± 0.005 by modifying the QF and QD in one cell. Keep their K1 changes equal and opposite." | Quad K1 pair → tune. Symmetric adjustment preserves lattice structure. |
| 3.6 | Chromaticity Adjustment     | Optimization| "Adjust the horizontal chromaticity to +1.0 ± 0.3 by changing all SF sextupoles by the same amount." | SF family ΔK2 → horizontal chromaticity. Requires sextupoles in lattice. |
| 3.7 | Local Orbit Correction      | Optimization| "Reduce the horizontal orbit at `{bpm}` to below 0.1 mm using the nearest horizontal corrector." | Single corrector kick → orbit at one BPM. |

---

## Tier 4: Complex (Budget: 80–200)

**Goal:** Long multi-step procedures combining discovery, measurement, and
physics reasoning. These tasks require chaining multiple Tier 2/3 skills.

### Tasks (2)

| ID  | Name                    | Prompt (template)                                              | Notes |
|-----|-------------------------|----------------------------------------------------------------|-------|
| 4.1 | Full ORM Measurement    | "Measure the full horizontal orbit response matrix. For each corrector, apply a small kick, record the orbit change at all BPMs, then restore the corrector." | Discovery + long set-read-restore loop over all correctors. |
| 4.2 | Local Closed Bump       | "Create a closed orbit bump producing +3 mm horizontal displacement at `{bpm}`, closing within 0.2 mm at all other BPMs." | Discovery + multi-corrector physics. Merged from old 3.8/4.5. |

### Notes

- Tier is intentionally small for now. More tasks to be added.
- Global orbit correction and multi-objective tasks (orbit + tune) require
  matrix math or an optimizer — not feasible by iterative hand-tuning with
  10 correctors. Deferred to a potential Tier 5.

---

## Implementation Notes

### Lattice Configuration

- Base lattice: FODO 10-cell ring (Bmad model)
- All computed quantities (Twiss, orbit, tunes, chromaticities) are exposed as read-only variables in the device tree, backed by the Bmad/Tao backend
- Error seeds are randomized per run (corrector kicks, quad perturbations)
- Element names in prompts are parameterized and drawn from the actual lattice at runtime
- Each run generates a configuration file recording all perturbations for reproducibility

### Setup and Perturbations

Tasks that require a perturbed machine state declare a **Setup** block. The harness applies perturbations before the agent starts and records them for verification. Common perturbations:

- Corrector kicks (orbit distortion)
- Quadrupole gradient errors (tune shift, beta beating)
- Misalignments (x/y offset)

### Verification

- Ground truth is computed from Bmad directly, not from agent actions
- For action tasks, verification reads back the final machine state
- Tolerances are generous — the benchmark tests reasoning ability, not numerical precision

### Tool Call Counting

- All calls count, including failed or redundant calls
- Exceeding the max tool call budget is an automatic failure
- Efficiency score: `max(0, 1 - calls_used / max_calls)` (bonus metric, not pass/fail)

### Timeout

- 5 minutes wall-clock per task
- Agent must not prompt for human clarification (automatic failure)

---

## Cross-Cutting Gaps

| Gap                        | Tier | Notes                                              |
|----------------------------|------|----------------------------------------------------|
| Longitudinal dynamics      | 2–4  | No RF/energy tasks; energy is read-only in Bmad    |
| Coupling                   | 3–4  | No cross-plane measurement or correction           |
| Tier 4 is thin             | 4    | Only 2 tasks; need more complex compositions       |
| Tier 5 (optimizer-level)   | 5    | Global orbit correction, multi-objective control — require matrix math or code execution |
