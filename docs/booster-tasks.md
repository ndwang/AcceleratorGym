# Booster Problem Set — Task Specifications

Separate AccelBench problem set targeting the AGS Booster lattice. All 6 tasks are hard, realistic accelerator operations challenges.

**Lattice:** AGS Booster (48 BPMs, 24 H-correctors, 24 V-correctors, 48 quads in QH/QV families, 54 sextupoles in SH/SV families, overlay family controls).

**Prerequisites:** Overlay support in `bmad.py` (exposes `qh_k1[IQHC]`, `qv_k1[IQVC]`, `sh_k2[ISH]`, `sv_k2[ISV]`, etc.).

---

## B1: Two-Plane Tune Matching

| Field | Value |
|-------|-------|
| **Budget** | 50 |
| **Tier** | 4 (Complex) |
| **Core challenge** | 2D coupled optimization |

### Setup

```python
def setup(env):
    rng = env.rng
    tune_a = env.machine.get(global_variable(env, "tune.a"))
    tune_b = env.machine.get(global_variable(env, "tune.b"))
    target_a = round(tune_a + float(rng.uniform(-0.03, -0.01)) * rng.choice([-1, 1]), 4)
    target_b = round(tune_b + float(rng.uniform(-0.03, -0.01)) * rng.choice([-1, 1]), 4)
    return {"qx_target": target_a, "qy_target": target_b, "tol": 0.005}
```

### Prompt

> Adjust the horizontal tune to {qx_target} +/- {tol} and the vertical tune to {qy_target} +/- {tol} simultaneously. You may use any quadrupoles or overlays.
>
> Return `{"status": "done"}`.

### Verify

```python
def verify(result, env, setup_data):
    tune_a = env.machine.get(global_variable(env, "tune.a"))
    tune_b = env.machine.get(global_variable(env, "tune.b"))
    return (abs(tune_a - setup_data["qx_target"]) <= setup_data["tol"]
            and abs(tune_b - setup_data["qy_target"]) <= setup_data["tol"])
```

### Why it's hard

Changing IQHC (horizontal tune trim) shifts **both** Qx and Qy due to cross-plane coupling. The agent must:
1. Discover the tune-control overlays (`qh_k1`, `qv_k1`) and their variables (IQHC, IQVC)
2. Measure or estimate the 2x2 Jacobian (dQx/dIQHC, dQx/dIQVC, dQy/dIQHC, dQy/dIQVC)
3. Use Newton-like iteration to converge both tunes simultaneously
4. Budget 50 allows ~22 iterations of (set + read), tight for coupled 2D optimization

Without overlays, the agent would need to set 24 individual quads per iteration — infeasible within budget.

---

## B2: Two-Plane Chromaticity Matching

| Field | Value |
|-------|-------|
| **Budget** | 50 |
| **Tier** | 4 (Complex) |
| **Core challenge** | 2D coupled optimization (different domain) |

### Setup

```python
def setup(env):
    rng = env.rng
    chrom_a = env.machine.get(global_variable(env, "chrom.a"))
    chrom_b = env.machine.get(global_variable(env, "chrom.b"))
    target_a = round(chrom_a + float(rng.uniform(1.0, 3.0)) * rng.choice([-1, 1]), 2)
    target_b = round(chrom_b + float(rng.uniform(1.0, 3.0)) * rng.choice([-1, 1]), 2)
    return {"ca_target": target_a, "cb_target": target_b, "tol": 0.5}
```

### Prompt

> Adjust the horizontal chromaticity to {ca_target} +/- {tol} and the vertical chromaticity to {cb_target} +/- {tol} simultaneously using sextupoles or overlays.
>
> Return `{"status": "done"}`.

### Verify

```python
def verify(result, env, setup_data):
    chrom_a = env.machine.get(global_variable(env, "chrom.a"))
    chrom_b = env.machine.get(global_variable(env, "chrom.b"))
    return (abs(chrom_a - setup_data["ca_target"]) <= setup_data["tol"]
            and abs(chrom_b - setup_data["cb_target"]) <= setup_data["tol"])
```

### Why it's hard

Same coupled 2D structure as B1 but in the chromaticity domain. Agent must discover `sh_k2[ISH]` and `sv_k2[ISV]` sextupole overlays (different system, different naming). Tests whether the agent can generalize the Jacobian-based approach from tunes to chromaticity.

---

## B3: Global Orbit Correction

| Field | Value |
|-------|-------|
| **Budget** | 120 |
| **Tier** | 4 (Complex) |
| **Core challenge** | Large-scale ORM-based correction |

### Setup

```python
def setup(env):
    rng = env.rng
    quads = [r["device_id"] for r in query_variables(env, "quadrupole", "K1")]
    bpms = query_variables(env, "monitor", "orbit.x")

    # Apply random quad misalignments (persistent via startup_file)
    errors = apply_random_errors(env, [ErrorSpec("x_offset", 0.5e-3, quads)])

    # Measure initial orbit RMS, scale up if needed
    orbits = [env.machine.get(r["variable"]) for r in bpms]
    rms = math.sqrt(sum(o**2 for o in orbits) / len(orbits))

    if rms < 1e-4:  # < 0.1mm, scale up offsets
        env.machine._backend.disconnect()
        # Rewrite with larger offsets
        errors = apply_random_errors(env, [ErrorSpec("x_offset", 2e-3, quads)])
        orbits = [env.machine.get(r["variable"]) for r in bpms]
        rms = math.sqrt(sum(o**2 for o in orbits) / len(orbits))

    threshold = max(rms * 0.2, 1e-4)  # 20% of initial, min 0.1mm

    return {
        "initial_rms_mm": round(rms * 1e3, 3),
        "threshold_mm": round(threshold * 1e3, 3),
        "_errors": errors,
    }
```

### Prompt

> The horizontal orbit is disturbed (RMS ~ {initial_rms_mm} mm) due to magnet misalignments. Reduce the horizontal orbit RMS at all BPMs to below {threshold_mm} mm using horizontal correctors.
>
> Return `{"status": "done"}`.

### Verify

```python
def verify(result, env, setup_data):
    bpms = query_variables(env, "monitor", "orbit.x")
    orbits = [env.machine.get(r["variable"]) for r in bpms]
    rms = math.sqrt(sum(o**2 for o in orbits) / len(orbits))
    return rms < setup_data["threshold_mm"] * 1e-3
```

### Why it's hard

THE classic accelerator operations task at realistic scale:
- Correctors start at zero — zeroing them does nothing
- Misalignments persist across `reset` (Tao startup file)
- 24 correctors x 48 BPMs = large response matrix
- Full ORM measurement needs ~73 tool calls (1 baseline + 24 x 3 set/read/restore)
- Budget 120 barely fits ORM + discovery + correction application
- Alternative: iterative nearest-corrector approach (less optimal, uses fewer calls)

---

## B4: Local Closed Orbit Bump

| Field | Value |
|-------|-------|
| **Budget** | 100 |
| **Tier** | 4 (Complex) |
| **Core challenge** | Constrained linear algebra |

### Setup

```python
def setup(env):
    rng = env.rng
    bpms = query_variables(env, "monitor", "orbit.x")
    pick = bpms[int(rng.integers(len(bpms)))]
    initial = {r["variable"]: env.machine.get(r["variable"]) for r in bpms}
    return {
        "bpm": pick["device_id"],
        "target_mm": 3.0,
        "closure_mm": 0.2,
        "_bpm_var": pick["variable"],
        "_initial_orbit": initial,
    }
```

### Prompt

> Create a closed orbit bump producing +{target_mm} mm horizontal displacement at `{bpm}`, closing within {closure_mm} mm at all other BPMs.
>
> Return `{"status": "done"}`.

### Verify

```python
def verify(result, env, setup_data):
    bpm_var = setup_data["_bpm_var"]
    initial = setup_data["_initial_orbit"]
    target_m = setup_data["target_mm"] * 1e-3
    closure_m = setup_data["closure_mm"] * 1e-3
    bpms = query_variables(env, "monitor", "orbit.x")

    # Target BPM: displacement must match
    change = env.machine.get(bpm_var) - initial[bpm_var]
    if abs(change - target_m) > closure_m:
        return False

    # All other BPMs: orbit change must be small
    for r in bpms:
        if r["variable"] == bpm_var:
            continue
        change = abs(env.machine.get(r["variable"]) - initial[r["variable"]])
        if change > closure_m:
            return False
    return True
```

### Why it's hard

On the booster with 48 BPMs, closure is significantly harder than on FODO (10 BPMs):
1. Agent must identify correctors near the target BPM (spatial reasoning)
2. Measure their orbit response at the target and neighboring BPMs
3. Solve a constrained linear system: create displacement at one point, cancel everywhere else
4. With 47 BPMs to satisfy closure, more correctors may be needed than a simple 3-corrector bump
5. Budget 100 allows response measurement for ~30 correctors + iteration

---

## B5: Simultaneous Orbit and Tune Correction

| Field | Value |
|-------|-------|
| **Budget** | 200 |
| **Tier** | 4 (Complex) |
| **Core challenge** | Coupled multi-objective optimization |

### Setup

```python
def setup(env):
    rng = env.rng
    quads = [r["device_id"] for r in query_variables(env, "quadrupole", "K1")]
    bpms = query_variables(env, "monitor", "orbit.x")

    # Apply misalignments for orbit distortion
    errors = apply_random_errors(env, [ErrorSpec("x_offset", 0.5e-3, quads)])

    # Perturb tune overlay variables
    # Find the IQHC and IQVC overlay variables
    qh_rows = env.machine.catalog.query(
        "SELECT a.variable FROM attributes a JOIN devices d ON a.device_id = d.device_id "
        "WHERE d.device_type = 'overlay' AND a.attribute_name = 'IQHC'"
    )
    qv_rows = env.machine.catalog.query(
        "SELECT a.variable FROM attributes a JOIN devices d ON a.device_id = d.device_id "
        "WHERE d.device_type = 'overlay' AND a.attribute_name = 'IQVC'"
    )
    if qh_rows:
        iqhc_var = qh_rows[0]["variable"]
        current_iqhc = env.machine.get(iqhc_var)
        env.machine.set(iqhc_var, current_iqhc + float(rng.uniform(-5, 5)))
    if qv_rows:
        iqvc_var = qv_rows[0]["variable"]
        current_iqvc = env.machine.get(iqvc_var)
        env.machine.set(iqvc_var, current_iqvc + float(rng.uniform(-5, 5)))

    # Record current state
    orbits = [env.machine.get(r["variable"]) for r in bpms]
    rms = math.sqrt(sum(o**2 for o in orbits) / len(orbits))
    threshold = max(rms * 0.2, 1e-4)

    tune_a = env.machine.get(global_variable(env, "tune.a"))
    tune_b = env.machine.get(global_variable(env, "tune.b"))
    # Targets: restore tunes to design-like values (small shift from current)
    qx_target = round(tune_a + float(rng.uniform(-0.02, 0.02)), 4)
    qy_target = round(tune_b + float(rng.uniform(-0.02, 0.02)), 4)

    return {
        "initial_rms_mm": round(rms * 1e3, 3),
        "threshold_mm": round(threshold * 1e3, 3),
        "qx_current": round(tune_a, 4),
        "qy_current": round(tune_b, 4),
        "qx_target": qx_target,
        "qy_target": qy_target,
        "tol": 0.005,
    }
```

### Prompt

> The machine has orbit distortion (RMS ~ {initial_rms_mm} mm) and shifted tunes (Qx = {qx_current}, Qy = {qy_current}; targets: Qx = {qx_target} +/- {tol}, Qy = {qy_target} +/- {tol}). Correct the orbit RMS below {threshold_mm} mm AND match both tunes to their targets.
>
> Return `{"status": "done"}`.

### Verify

```python
def verify(result, env, setup_data):
    # Check orbit
    bpms = query_variables(env, "monitor", "orbit.x")
    orbits = [env.machine.get(r["variable"]) for r in bpms]
    rms = math.sqrt(sum(o**2 for o in orbits) / len(orbits))
    if rms > setup_data["threshold_mm"] * 1e-3:
        return False

    # Check tunes
    tune_a = env.machine.get(global_variable(env, "tune.a"))
    tune_b = env.machine.get(global_variable(env, "tune.b"))
    if abs(tune_a - setup_data["qx_target"]) > setup_data["tol"]:
        return False
    if abs(tune_b - setup_data["qy_target"]) > setup_data["tol"]:
        return False
    return True
```

### Why it's hard

Orbit and tune are **coupled** — this is NOT two independent sub-problems:
- Correcting orbit changes where the beam passes through quadrupoles → off-center beam in quads creates feed-down effects → changes effective tune
- Changing tune via quad overlays changes focusing → modifies the closed orbit response
- Agent must iterate: correct orbit → re-check tunes → adjust tunes → re-check orbit → converge
- Budget 200 must cover orbit correction (~80 calls) + tune matching (~30 calls) + 2-3 convergence iterations
- Requires strategic planning: which to fix first, when to re-check, how to handle coupling

---

## B6: 4D Optics Matching

| Field | Value |
|-------|-------|
| **Budget** | 100 |
| **Tier** | 4 (Complex) |
| **Core challenge** | 4D coupled optimization |

### Setup

```python
def setup(env):
    rng = env.rng

    # Find overlay variables for all 4 knobs
    tune_a = env.machine.get(global_variable(env, "tune.a"))
    tune_b = env.machine.get(global_variable(env, "tune.b"))
    chrom_a = env.machine.get(global_variable(env, "chrom.a"))
    chrom_b = env.machine.get(global_variable(env, "chrom.b"))

    # Perturb all 4 overlay variables slightly (IQHC, IQVC, ISH, ISV)
    # (implementation queries overlay variables from catalog and shifts them)

    # Set targets
    qx_target = round(tune_a + float(rng.uniform(-0.02, 0.02)), 4)
    qy_target = round(tune_b + float(rng.uniform(-0.02, 0.02)), 4)
    ca_target = round(chrom_a + float(rng.uniform(-2.0, 2.0)), 2)
    cb_target = round(chrom_b + float(rng.uniform(-2.0, 2.0)), 2)

    return {
        "qx_target": qx_target, "qy_target": qy_target,
        "ca_target": ca_target, "cb_target": cb_target,
        "tune_tol": 0.005, "chrom_tol": 0.5,
    }
```

### Prompt

> Match all four optics parameters simultaneously:
> - Horizontal tune (Qx) to {qx_target} +/- {tune_tol}
> - Vertical tune (Qy) to {qy_target} +/- {tune_tol}
> - Horizontal chromaticity to {ca_target} +/- {chrom_tol}
> - Vertical chromaticity to {cb_target} +/- {chrom_tol}
>
> You may use any overlays or magnets.
>
> Return `{"status": "done"}`.

### Verify

```python
def verify(result, env, setup_data):
    tune_a = env.machine.get(global_variable(env, "tune.a"))
    tune_b = env.machine.get(global_variable(env, "tune.b"))
    chrom_a = env.machine.get(global_variable(env, "chrom.a"))
    chrom_b = env.machine.get(global_variable(env, "chrom.b"))
    return (
        abs(tune_a - setup_data["qx_target"]) <= setup_data["tune_tol"]
        and abs(tune_b - setup_data["qy_target"]) <= setup_data["tune_tol"]
        and abs(chrom_a - setup_data["ca_target"]) <= setup_data["chrom_tol"]
        and abs(chrom_b - setup_data["cb_target"]) <= setup_data["chrom_tol"]
    )
```

### Why it's hard

Scaling from 2D to 4D optimization:
1. Agent must discover all 4 overlay knobs: IQHC, IQVC (tune), ISH, ISV (chromaticity)
2. Must measure or estimate the 4x4 Jacobian — how each knob affects all 4 observables
3. The matrix has cross-coupling: changing IQHC affects tunes AND may weakly affect chromaticity through higher-order effects
4. Budget 100 = ~20 iterations of (set 4 knobs + read 4 observables = 5 calls/iteration)
5. Newton iteration in 4D requires good Jacobian estimate — poor estimates lead to divergence
6. This is the hardest pure optimization task. Most agents solve 2D but fail at 4D.

---

## Summary

| Task | Core challenge | Budget | Key physics |
|------|---------------|--------|-------------|
| B1 | 2D coupled optimization | 50 | Tune matching via quad overlays |
| B2 | 2D coupled optimization (new domain) | 50 | Chromaticity via sextupole overlays |
| B3 | Large-scale ORM-based correction | 120 | Orbit correction with misalignments |
| B4 | Constrained linear algebra | 100 | 3-corrector closed orbit bump |
| B5 | Coupled multi-objective optimization | 200 | Orbit-tune coupling |
| B6 | 4D coupled optimization | 100 | Tune + chromaticity simultaneously |
