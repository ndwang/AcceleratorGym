# Accelerator Gym CLI

Interactive command-line tool for manually testing accelerator-gym tools against a running backend.

## Starting the CLI

From a directory containing `accelerator-gym.yaml`:

```
ag-cli
```

Or point to a config file explicitly:

```
ag-cli --config path/to/accelerator-gym.yaml
```

You can also run the module directly:

```
python -m accelerator_gym.cli --config path/to/accelerator-gym.yaml
```

Add `--debug` for verbose backend/logging output on stderr.

## Commands

### Device Discovery

#### `browse [path] [depth]`

Navigate the device tree using filesystem-style paths. The tree hierarchy is: system > device type > device instance > attribute.

Omit path to start at the root. Use `depth` > 1 to expand multiple levels at once.

```
ag> browse
{"path": "/", "children": ["diagnostics", "magnets"]}

ag> browse /magnets
{"path": "/magnets", "children": ["quadrupole"]}

ag> browse /magnets/quadrupole
{"path": "/magnets/quadrupole", "children": [{"name": "QD", ...}, {"name": "QF", ...}]}

ag> browse /magnets/quadrupole/QF
{"path": "/magnets/quadrupole/QF", "children": ["K1"]}

ag> browse /magnets/quadrupole/QF/K1
{"path": "/magnets/quadrupole/QF/K1", "variable": "ele::QF[K1]", "units": "1/m", "limits": [0.0, 5.0], "read": true, "write": true}

ag> browse / 3
# expands 3 levels deep from root: systems → device types → instances
```

#### `query <sql>`

Run a read-only SQL query against the device metadata database. Only SELECT statements are allowed.

Available tables:

| Table | Columns |
|---|---|
| `systems` | `name` |
| `device_types` | `name`, `system` |
| `devices` | `name`, `device_type`, `system`, `description` |
| `attributes` | `device_name`, `device_type`, `system`, `attr_name`, `variable`, `description`, `units`, `readable`, `writable`, `limit_low`, `limit_high` |

```
ag> query SELECT name FROM systems
ag> query SELECT name FROM devices WHERE device_type='quadrupole'
ag> query SELECT variable, units FROM attributes WHERE writable=1
```

### Reading and Writing Variables

Variable names are backend-specific identifiers shown in `browse` output as the `variable` field.

#### `get <variable>`

Read a single variable.

```
ag> get ele::QF[K1]
{"name": "ele::QF[K1]", "value": 0.5, "units": "1/m"}
```

#### `gets <var1> <var2> ...`

Read multiple variables at once.

```
ag> gets ele::QF[K1] ele::QD[K1]
{"values": {"ele::QF[K1]": 0.5, "ele::QD[K1]": -0.3}}
```

#### `set <variable> <value>`

Write a single variable. Validates limits before applying.

```
ag> set ele::QF[K1] 1.5
{"success": true, "name": "ele::QF[K1]", "value": 1.5}
```

#### `sets <var>=<val> ...`

Write multiple variables atomically (all-or-nothing). If any value violates limits, none are applied.

```
ag> sets ele::QF[K1]=1.5 ele::QD[K1]=-0.8
{"success": true, "values": {"ele::QF[K1]": 1.5, "ele::QD[K1]": -0.8}}
```

### Machine State

#### `state`

Print a snapshot of all readable variable values.

```
ag> state
{"variables": {"ele::QD[K1]": -0.3, "ele::QF[K1]": 1.5, "lat::orbit.x[BPM1]": 0.001}}
```

#### `reset`

Reset the machine to its initial state.

```
ag> reset
{"success": true}
```

### Other

| Command | Description |
|---|---|
| `help` | Show the command summary |
| `quit` | Exit the CLI (Ctrl-C or Ctrl-D also work) |
