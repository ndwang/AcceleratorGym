# AccelAgent Benchmark v2

## Evaluation Tasks for Accelerator Operations Agents

**Target Lattice**: BNL Booster Ring (Bmad model)

---

## Overview

This benchmark evaluates AI agents on accelerator physics operations tasks against a Bmad simulation backend. All tasks are programmatically verifiable — each has a natural-language prompt, a verification function, and a maximum tool-call budget.

Task difficulty correlates with two factors: the degree of adaptive reasoning required, and the degree of discovery required. Lower tiers give exact element names and map to fixed procedures. Higher tiers use vague or symptomatic descriptions and require the agent to discover targets through search, adapt based on intermediate results, and compose multiple sub-procedures. This is the fundamental gap between planning agents (which emit a fixed action sequence upfront) and ReAct agents (which observe, reason, and act in a loop).

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
/magnets/quadrupole/{name}/K1          read+write   Quadrupole strength
/magnets/hkicker/{name}/KICK           read+write   Horizontal corrector kick
/magnets/vkicker/{name}/KICK           read+write   Vertical corrector kick
/magnets/sextupole/{name}/K2           read+write   Sextupole strength
/diagnostics/monitor/{name}/orbit.x    read-only    Horizontal orbit at BPM
/diagnostics/monitor/{name}/orbit.y    read-only    Vertical orbit at BPM
/optics/twiss/{name}/beta.a            read-only    Horizontal beta function
/optics/twiss/{name}/beta.b            read-only    Vertical beta function
/optics/twiss/{name}/phi.a             read-only    Horizontal phase advance
/optics/twiss/{name}/phi.b             read-only    Vertical phase advance
/optics/twiss/{name}/eta.x             read-only    Horizontal dispersion
/optics/lattice/ring/tune.a            read-only    Horizontal tune
/optics/lattice/ring/tune.b            read-only    Vertical tune
/optics/lattice/ring/chrom.a           read-only    Horizontal chromaticity
/optics/lattice/ring/chrom.b           read-only    Vertical chromaticity
/optics/lattice/ring/energy            read-only    Beam energy
/optics/lattice/ring/circumference     read-only    Ring circumference
```

The agent must discover this structure through `browse_devices` and `query_devices`. Variable names (e.g., `QVA1:K1`, `ring:tune.a`, `PUEVA1:orbit.x`) are obtained from the tree metadata, not given in advance.

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
    env.design: Machine instance (reference/design state, read-only)
    """
```

Tasks fall into two categories:

- **Report tasks**: Agent must return information. Verification checks `result` against ground truth computed from `env`.
- **Action tasks**: Agent must change machine state. Verification reads back `env.machine` state. `result` may contain supplementary info (e.g., diagnosis) but the primary check is on machine state.

The harness extracts `result` as a JSON object from the agent's final response.

---

## Tier 1: Direct (budget: 3)

Exact element names given. Tests basic interface competence — tool selection, argument extraction, result interpretation. Both planning and ReAct agents should score 100%.

### 1.1 — Read a Parameter

**Prompt**: "What is K1 of QVA1?"
**Ability**: I/O

```python
def verify(result, env):
    truth = env.machine.get("QVA1:K1")
    return abs(result["value"] - truth) < 1e-8
```

### 1.2 — Set a Parameter

**Prompt**: "Set K1 of QVA1 to 0.5."
**Ability**: I/O

```python
def verify(result, env):
    return abs(env.machine.get("QVA1:K1") - 0.5) < 1e-8
```

### 1.3 — Read a BPM

**Prompt**: "What is the vertical orbit at PUEVA3?"
**Ability**: I/O

```python
def verify(result, env):
    truth = env.machine.get("PUEVA3:orbit.y")
    return abs(result["value"] - truth) < 1e-9
```

### 1.4 — Count Elements by Type

**Prompt**: "How many quadrupoles are in the ring?"
**Ability**: Discovery

```python
def verify(result, env):
    rows = env.machine.catalog.query(
        "SELECT COUNT(*) as n FROM devices WHERE device_type='quadrupole'")
    return result["count"] == rows[0]["n"]
```

### 1.5 — Read a Global Parameter

**Prompt**: "What is the horizontal tune?"
**Ability**: Analysis

```python
def verify(result, env):
    truth = env.machine.get("ring:tune.a")
    return abs(result["value"] - truth) < 1e-4
```

### 1.6 — Identify an Element

**Prompt**: "What type of element is DHCC2?"
**Ability**: Discovery

```python
def verify(result, env):
    rows = env.machine.catalog.query(
        "SELECT device_type FROM devices WHERE device_id='DHCC2'")
    return result["type"] == rows[0]["device_type"]
```

---

## Tier 2: Procedural (budget: 5–15)

Known multi-step procedures with deterministic sequencing. A planning agent can handle these if it knows the recipe and element names are given. No adaptation required — every step is predetermined.

### 2.1 — Orbit RMS

**Prompt**: "Measure the horizontal orbit at all BPMs and report the RMS."
**Ability**: I/O + Analysis
**Budget**: 5

```python
def verify(result, env):
    bpm_vars = [r["variable"] for r in env.machine.catalog.query(
        "SELECT a.variable FROM attributes a JOIN devices d ON a.device_id = d.device_id "
        "WHERE d.device_type='monitor' AND a.attribute_name='orbit.x'")]
    values = env.machine.get_many(bpm_vars)
    truth = np.sqrt(np.mean([v**2 for v in values.values()]))
    return abs(result["rms"] - truth) / truth < 0.01
```

### 2.2 — Single Corrector Orbit Response

**Prompt**: "Measure the horizontal orbit response to corrector DHCC2. Apply a kick of 0.1 mrad, record the orbit change at all BPMs, then restore the corrector."
**Ability**: Measurement
**Budget**: 10

```python
def verify(result, env):
    # Ground truth: compute response column from Bmad
    bpm_vars = _get_bpm_vars(env, "orbit.x")
    initial = env.machine.get_many(bpm_vars)
    env.machine.set("DHCC2:KICK", 0.1e-3)
    kicked = env.machine.get_many(bpm_vars)
    env.machine.set("DHCC2:KICK", 0.0)
    truth = np.array([kicked[v] - initial[v] for v in bpm_vars])
    agent = np.array(result["response"])
    return np.linalg.norm(agent - truth) / np.linalg.norm(truth) < 0.01
```

### 2.3 — Maximum Beta Function

**Prompt**: "What is the maximum horizontal beta function in the ring and where does it occur?"
**Ability**: I/O + Analysis
**Budget**: 8

```python
def verify(result, env):
    twiss_vars = [r["variable"] for r in env.machine.catalog.query(
        "SELECT a.variable FROM attributes a JOIN devices d ON a.device_id = d.device_id "
        "WHERE d.system='optics' AND d.device_type='twiss' AND a.attribute_name='beta.a'")]
    values = env.machine.get_many(twiss_vars)
    max_var = max(values, key=values.get)
    truth_val = values[max_var]
    truth_name = max_var.split(":")[0]
    return (abs(result["value"] - truth_val) / truth_val < 0.02 and
            result["element"] == truth_name)
```

### 2.4 — Phase Advance Between Two Points

**Prompt**: "What is the horizontal phase advance between PUEVA1 and PUEVA5?"
**Ability**: Analysis
**Budget**: 5

```python
def verify(result, env):
    phi_1 = env.machine.get("PUEVA1:phi.a")
    phi_5 = env.machine.get("PUEVA5:phi.a")
    truth = phi_5 - phi_1
    return abs(result["value"] - truth) < 0.01
```

### 2.5 — Total Bending Angle

**Prompt**: "What is the total bending angle from all dipoles? Is it consistent with a full ring (2π)?"
**Ability**: Discovery + Analysis
**Budget**: 10

```python
def verify(result, env):
    dipole_vars = [r["variable"] for r in env.machine.catalog.query(
        "SELECT a.variable FROM attributes a JOIN devices d ON a.device_id = d.device_id "
        "WHERE d.device_type='sbend' AND a.attribute_name='ANGLE'")]
    values = env.machine.get_many(dipole_vars)
    truth = sum(abs(v) for v in values.values())
    angle_ok = abs(result["total_angle"] - truth) / truth < 0.001
    consistent_ok = result["consistent"] == (abs(truth - 2 * np.pi) < 0.01)
    return angle_ok and consistent_ok
```

### 2.6 — Dispersion Measurement

**Prompt**: "Measure the horizontal dispersion at all BPMs by changing the beam energy by ±0.01% and observing the orbit shift."
**Ability**: Measurement
**Budget**: 12

```python
def verify(result, env):
    bpm_vars = _get_bpm_vars(env, "orbit.x")
    truth = np.array([env.machine.get(v.replace("orbit.x", "eta.x")) for v in bpm_vars])
    agent = np.array(result["dispersion"])
    return np.linalg.norm(agent - truth) / np.linalg.norm(truth) < 0.05
```

### 2.7 — Chromaticity Measurement

**Prompt**: "Measure the horizontal chromaticity by shifting the beam energy by ±0.01% and observing the tune change."
**Ability**: Measurement
**Budget**: 10

```python
def verify(result, env):
    truth = env.machine.get("ring:chrom.a")
    return abs(result["chromaticity"] - truth) / abs(truth) < 0.05
```

---

## Tier 3: Adaptive (budget: 10–40)

The next action depends on the result of the previous one. A planning agent cannot pre-determine the action sequence. Prompts become vague — the agent must discover targets through search before it can act.

### 3.1 — Find Most Effective Corrector

**Prompt**: "Which horizontal corrector has the largest effect on the orbit near s = 45 m?"
**Ability**: Discovery + Analysis
**Budget**: 30
**Setup**: None

The agent must find the nearest BPM to s = 45 m, then measure response from each horizontal corrector, then compare. Cannot be pre-planned because it doesn't know which BPM or correctors exist.

```python
def verify(result, env):
    # Find closest BPM to s=45
    bpms = env.machine.catalog.query(
        "SELECT device_id, s_position FROM devices WHERE device_type='monitor' "
        "ORDER BY ABS(s_position - 45) LIMIT 1")
    target_bpm = bpms[0]["device_id"]
    bpm_var = f"{target_bpm}:orbit.x"
    # Measure all corrector responses
    hcors = env.machine.catalog.query(
        "SELECT a.variable FROM attributes a JOIN devices d ON a.device_id = d.device_id "
        "WHERE d.device_type='hkicker' AND a.attribute_name='KICK'")
    best_cor, best_resp = None, 0
    for row in hcors:
        var = row["variable"]
        name = var.split(":")[0]
        initial_orbit = env.machine.get(bpm_var)
        initial_kick = env.machine.get(var)
        env.machine.set(var, initial_kick + 1e-4)
        new_orbit = env.machine.get(bpm_var)
        env.machine.set(var, initial_kick)
        resp = abs(new_orbit - initial_orbit)
        if resp > best_resp:
            best_resp = resp
            best_cor = name
    return result["corrector"] == best_cor
```

### 3.2 — Find a Nonzero Corrector

**Prompt**: "Something is applying a horizontal kick somewhere in the ring. Find the source."
**Ability**: Discovery + I/O
**Budget**: 25
**Setup**: One horizontal corrector set to a known nonzero kick.

```python
def verify(result, env, setup):
    return (result["corrector"] == setup["corrector_name"] and
            abs(result["kick"] - setup["kick_value"]) / abs(setup["kick_value"]) < 0.05)
```

### 3.3 — Gradient Error Detection

**Prompt**: "One of the focusing quadrupoles has an anomalous gradient. Which one?"
**Ability**: Discovery + Diagnosis
**Budget**: 40
**Setup**: One focusing quadrupole K1 perturbed by 5%.

The agent must discover all quadrupoles, determine which are "focusing" (K1 > 0 vs K1 < 0), read their strengths, identify the outlier within the family. Cannot be pre-planned because the agent doesn't know the naming convention or family structure.

```python
def verify(result, env, setup):
    return result["element"] == setup["perturbed_quad"]
```

### 3.4 — Beta Function Outlier

**Prompt**: "Find any BPM where the horizontal beta function is more than 50% above average."
**Ability**: Analysis + Diagnosis
**Budget**: 15
**Setup**: Lattice configured so one BPM location has anomalously high beta (e.g., via quad error nearby).

```python
def verify(result, env):
    twiss_rows = env.machine.catalog.query(
        "SELECT d.device_id, a.variable FROM attributes a "
        "JOIN devices d ON a.device_id = d.device_id "
        "WHERE d.device_type='twiss' AND a.attribute_name='beta.a' "
        "AND d.device_id IN (SELECT device_id FROM devices WHERE device_type='monitor')")
    betas = {r["device_id"]: env.machine.get(r["variable"]) for r in twiss_rows}
    avg = np.mean(list(betas.values()))
    outliers = {name for name, b in betas.items() if b > 1.5 * avg}
    return result["bpm"] in outliers
```

### 3.5 — Tune Adjustment

**Prompt**: "Adjust the horizontal tune to 4.82 ± 0.005."
**Ability**: Physics + Optimization
**Budget**: 30
**Setup**: Tune detuned from 4.82.

The agent must figure out which elements control the horizontal tune (focusing quadrupoles), adjust them, read back the tune, and iterate until converged.

```python
def verify(result, env):
    tune = env.machine.get("ring:tune.a")
    return abs(tune - 4.82) < 0.005
```

### 3.6 — Chromaticity Adjustment

**Prompt**: "Adjust the horizontal chromaticity to +1.0 ± 0.3."
**Ability**: Physics + Optimization
**Budget**: 30
**Setup**: Chromaticity away from target.

```python
def verify(result, env):
    chrom = env.machine.get("ring:chrom.a")
    return abs(chrom - 1.0) < 0.3
```

### 3.7 — Single-BPM Orbit Correction

**Prompt**: "Correct the horizontal orbit at the BPM with the worst reading to within 0.1 mm."
**Ability**: Discovery + Optimization
**Budget**: 25
**Setup**: Orbit distorted by corrector kicks.

The agent must read all BPMs to find the worst one (discovery), then find and apply a correction (optimization). Both steps depend on observed data.

```python
def verify(result, env):
    # Find worst BPM
    bpm_vars = _get_bpm_vars(env, "orbit.x")
    values = env.machine.get_many(bpm_vars)
    worst_var = max(values, key=lambda v: abs(values[v]))
    return abs(env.machine.get(worst_var)) < 0.1e-3
```

### 3.8 — Local Orbit Bump

**Prompt**: "Create a +2 mm horizontal orbit bump near s = 30 m that doesn't affect BPMs more than 5 m away by more than 0.1 mm."
**Ability**: Measurement + Physics
**Budget**: 40
**Setup**: None (clean orbit).

```python
def verify(result, env, setup):
    bpm_rows = env.machine.catalog.query(
        "SELECT device_id, s_position FROM devices WHERE device_type='monitor'")
    target_bpm = min(bpm_rows, key=lambda r: abs(r["s_position"] - 30))
    target_var = f"{target_bpm['device_id']}:orbit.x"
    bump_ok = abs(env.machine.get(target_var) - setup["initial_orbits"][target_var] - 2e-3) < 0.3e-3
    far_bpms_ok = all(
        abs(env.machine.get(f"{r['device_id']}:orbit.x") - setup["initial_orbits"][f"{r['device_id']}:orbit.x"]) < 0.1e-3
        for r in bpm_rows if abs(r["s_position"] - 30) > 5)
    return bump_ok and far_bpms_ok
```

---

## Tier 4: Complex (budget: 50–200)

Multiple sub-procedures composed together. Require sustained reasoning, physics knowledge, and constraint handling. No element names given — the agent must discover everything.

### 4.1 — Full ORM Measurement

**Prompt**: "Measure the full horizontal orbit response matrix and report its dimensions."
**Ability**: Discovery + Measurement + Analysis
**Budget**: 80

The agent must discover all BPMs and all horizontal correctors, then systematically kick each corrector, record orbit changes, and restore. Requires bookkeeping across N correctors.

```python
def verify(result, env):
    n_bpms = env.machine.catalog.query(
        "SELECT COUNT(*) as n FROM devices WHERE device_type='monitor'")[0]["n"]
    n_hcors = env.machine.catalog.query(
        "SELECT COUNT(*) as n FROM devices WHERE device_type='hkicker'")[0]["n"]
    shape_ok = result["rows"] == n_bpms and result["cols"] == n_hcors
    # Verify a sample of matrix entries against ground truth
    agent_orm = np.array(result["matrix"])
    truth_orm = _compute_orm(env, plane="x", kick=1e-4)
    norm_ok = abs(np.linalg.norm(agent_orm - truth_orm, 'fro') /
                  np.linalg.norm(truth_orm, 'fro')) < 0.05
    return shape_ok and norm_ok
```

### 4.2 — Global Orbit Correction

**Prompt**: "Correct the horizontal orbit to minimize the RMS across all BPMs."
**Ability**: Measurement + Physics + Optimization
**Budget**: 100
**Setup**: Orbit distorted by multiple corrector kicks.

```python
def verify(result, env, setup):
    bpm_vars = _get_bpm_vars(env, "orbit.x")
    values = env.machine.get_many(bpm_vars)
    final_rms = np.sqrt(np.mean([v**2 for v in values.values()]))
    return final_rms < setup["initial_rms"] * 0.2
```

### 4.3 — Orbit Correction with Dead Corrector

**Prompt**: "Correct the horizontal orbit to below 1.0 mm RMS. One of the correctors appears to be stuck — find it and work around it."
**Ability**: Diagnosis + Optimization
**Budget**: 120
**Setup**: One corrector locked (write disabled via limits), orbit distorted.

The agent must discover the broken corrector (by trying to set it or noticing it doesn't respond), exclude it, and correct with the remaining correctors.

```python
def verify(result, env, setup):
    bpm_vars = _get_bpm_vars(env, "orbit.x")
    values = env.machine.get_many(bpm_vars)
    rms = np.sqrt(np.mean([v**2 for v in values.values()]))
    stuck_ok = abs(env.machine.get(setup["stuck_corrector_var"])) < 1e-12
    return rms < 1.0e-3 and stuck_ok
```

### 4.4 — Combined Orbit and Tune Correction

**Prompt**: "Get the horizontal orbit below 0.5 mm RMS and the horizontal tune to 4.82 ± 0.01 without changing the vertical tune by more than 0.01."
**Ability**: Physics + Optimization (two competing objectives)
**Budget**: 150
**Setup**: Orbit distorted, tune detuned, vertical tune recorded.

```python
def verify(result, env, setup):
    bpm_vars = _get_bpm_vars(env, "orbit.x")
    values = env.machine.get_many(bpm_vars)
    rms = np.sqrt(np.mean([v**2 for v in values.values()]))
    tune_a = env.machine.get("ring:tune.a")
    tune_b = env.machine.get("ring:tune.b")
    return (rms < 0.5e-3 and
            abs(tune_a - 4.82) < 0.01 and
            abs(tune_b - setup["initial_tune_b"]) < 0.01)
```

### 4.5 — Three-Corrector Closed Bump

**Prompt**: "Create a closed orbit bump producing +3 mm horizontal displacement at the BPM closest to s = 30 m, closing within 0.2 mm at all other BPMs."
**Ability**: Physics + Measurement + Optimization
**Budget**: 100
**Setup**: None (clean orbit).

The agent must select three correctors that can form a closed bump, measure their responses, solve for kick ratios, apply, verify closure, and iterate.

```python
def verify(result, env, setup):
    bpm_rows = env.machine.catalog.query(
        "SELECT device_id, s_position FROM devices WHERE device_type='monitor'")
    target = min(bpm_rows, key=lambda r: abs(r["s_position"] - 30))
    target_var = f"{target['device_id']}:orbit.x"
    bump_ok = abs(env.machine.get(target_var) - setup["initial_orbits"][target_var] - 3e-3) < 0.3e-3
    others_ok = all(
        abs(env.machine.get(f"{r['device_id']}:orbit.x") - setup["initial_orbits"][f"{r['device_id']}:orbit.x"]) < 0.2e-3
        for r in bpm_rows if r["device_id"] != target["device_id"])
    return bump_ok and others_ok
```

### 4.6 — End-to-End Diagnosis and Correction

**Prompt**: "The beam orbit is distorted and the tune is wrong. Diagnose the problem and fix it."
**Ability**: Diagnosis + Physics + Optimization
**Budget**: 200
**Setup**: One quadrupole has a gradient error causing orbit distortion and tune shift.

The most open-ended task. The agent doesn't know what's wrong — it must measure, hypothesize, test, and correct. Requires combining diagnosis (find the bad quad) with correction (fix the tune and orbit).

```python
def verify(result, env, setup):
    tune = env.machine.get("ring:tune.a")
    bpm_vars = _get_bpm_vars(env, "orbit.x")
    values = env.machine.get_many(bpm_vars)
    rms = np.sqrt(np.mean([v**2 for v in values.values()]))
    tune_ok = abs(tune - setup["design_tune"]) < 0.02
    orbit_ok = rms < 1.0e-3
    diagnosis_ok = result.get("faulty_element") == setup["perturbed_quad"]
    return tune_ok and orbit_ok and diagnosis_ok
```

---

## Ability Tags

Each task is tagged with the abilities it tests. After scoring, compute per-ability pass rates for a capability profile.

| Ability | What it measures | Tasks |
|---------|-----------------|-------|
| I/O | Read/write values via the interface | 1.1–1.3, 2.1, 2.2, 3.2, 3.7 |
| Discovery | Navigate/query to find relevant variables | 1.4, 1.6, 2.5, 3.1–3.3, 4.1, 4.3 |
| Analysis | Aggregate, compare, extract statistics | 1.5, 2.1, 2.3–2.5, 3.1, 3.4, 4.1 |
| Measurement | Execute standard measurement procedures | 2.2, 2.6, 2.7, 3.8, 4.1, 4.2 |
| Physics | Apply domain knowledge to select strategy | 3.5, 3.6, 3.8, 4.2, 4.4, 4.5, 4.6 |
| Optimization | Iteratively converge toward a target | 3.5–3.8, 4.2–4.5 |
| Diagnosis | Identify root cause from observations | 3.3, 3.4, 4.3, 4.6 |

---

## Implementation Notes

### Lattice Configuration

- Base lattice: BNL Booster Ring Bmad model
- All computed quantities (Twiss, orbit, tunes, chromaticities) are exposed as read-only variables in the device tree, backed by the Bmad/Tao backend
- Error seeds are randomized per run (corrector kicks, quad perturbations)
- Element names in prompts are parameterized and drawn from the actual lattice at runtime
- Each run generates a configuration file recording all perturbations for reproducibility

### Setup and Perturbations

Tasks that require a perturbed machine state declare a **Setup** block. The harness applies perturbations before the agent starts and records them for verification. Common perturbations:

- Corrector kicks (orbit distortion)
- Quadrupole gradient errors (tune shift, beta beating)
- Locked correctors (constraint tasks)

### Verification

- Ground truth is computed from Bmad directly, not from agent actions
- For action tasks, verification reads back the final machine state
- Tolerances are generous — the benchmark tests reasoning ability, not numerical precision
- Helper function `_get_bpm_vars(env, attr)` queries all BPM variables of a given attribute
- Helper function `_compute_orm(env, plane, kick)` computes the ground truth ORM from Bmad

### Tool Call Counting

- All calls count, including failed or redundant calls
- Exceeding the max tool call budget is an automatic failure
- Efficiency score: `max(0, 1 - calls_used / max_calls)` (bonus metric, not pass/fail)

### Timeout

- 5 minutes wall-clock per task
- Agent must not prompt for human clarification (automatic failure)

### Answer Extraction

The harness extracts the agent's answer as a JSON object from its final response. Tasks define which keys are required (e.g., `result["value"]`, `result["corrector"]`). Action tasks primarily verify machine state; the `result` dict carries supplementary information like diagnosis.
