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
/
|-- diagnostics
`-- magnets

ag> browse / 2
/
|-- diagnostics
|   `-- monitor
`-- magnets
    `-- quadrupole

ag> browse /magnets 2
/magnets
`-- quadrupole
    |-- QD  Defocusing quadrupole
    `-- QF  Focusing quadrupole

ag> browse /magnets/quadrupole 2
/magnets/quadrupole
|-- QD  Defocusing quadrupole
|   `-- K1
`-- QF  Focusing quadrupole
    `-- K1

ag> browse /magnets/quadrupole/QF 2
/magnets/quadrupole/QF
`-- K1  RW  [1/m]  Integrated strength

ag> browse /magnets/quadrupole/QF/K1
/magnets/quadrupole/QF/K1
  variable:  QF:K1
  desc:      Integrated strength
  units:     1/m
  read:      yes
  write:     yes
  limits:    [0.0, 5.0]
```

Directory nodes are shown in bold (on supported terminals). Browsing a leaf attribute shows its full metadata.

#### `query <sql>`

Run a read-only SQL query against the device metadata database. Only SELECT statements are allowed. Results are displayed as an aligned table.

Available tables:

| Table | Columns |
|---|---|
| `systems` | `name` |
| `device_types` | `name`, `system` |
| `devices` | `name`, `device_type`, `system`, `description` |
| `attributes` | `device_name`, `device_type`, `system`, `attr_name`, `variable`, `description`, `units`, `readable`, `writable`, `limit_low`, `limit_high` |

```
ag> query SELECT attr_name, units FROM attributes LIMIT 3
attr_name  units
---------  -----
K1         1/m
orbit.x    mm
orbit.y    mm

(3 rows)
```

### Reading and Writing Variables

Variable names are backend-specific identifiers shown in `browse` output as the `variable` field.

#### `get <variable>`

Read a single variable.

```
ag> get QF:K1
  QF:K1 = 0.5  [1/m]
```

#### `gets <var1> <var2> ...`

Read multiple variables at once.

```
ag> gets QF:K1 QD:K1
  QF:K1  = 0.5   [1/m]
  QD:K1  = -0.3  [1/m]
```

#### `set <variable> <value>`

Write a single variable. Validates limits before applying.

```
ag> set QF:K1 1.5
  QF:K1 <- 1.5  [1/m]
```

#### `sets <var>=<val> ...`

Write multiple variables atomically (all-or-nothing). If any value violates limits, none are applied.

```
ag> sets QF:K1=1.5 QD:K1=-0.8
  QF:K1 <- 1.5   [1/m]
  QD:K1 <- -0.8  [1/m]
```

### Machine State

#### `state`

Print a snapshot of all readable variable values.

```
ag> state
  BPM1:orbit.x  = 0.001  [mm]
  BPM1:orbit.y  = 0.0    [mm]
  QD:K1         = -0.3   [1/m]
  QF:K1         = 1.5    [1/m]
```

#### `reset`

Reset the machine to its initial state.

```
ag> reset
  Machine reset to initial state.
```

### Other

| Command | Description |
|---|---|
| `help` | Show the command summary |
| `quit` | Exit the CLI (Ctrl-C or Ctrl-D also work) |

## Output Styling

The CLI uses ANSI colors and Unicode box-drawing characters when the terminal supports them:

- **Colors**: paths in cyan, values in green, set values in yellow, units in magenta, descriptions dimmed, errors in red
- **Tree lines**: `├──`, `└──`, `│` connectors for `browse` output
- **Tables**: bold headers with `─` separator lines for `query` output

On terminals without Unicode support (e.g. some Windows consoles), the CLI falls back to ASCII equivalents (`|--`, `` `-- ``, `|`, `-`).

Color output is disabled automatically when `NO_COLOR` environment variable is set, following the [no-color convention](https://no-color.org/).
