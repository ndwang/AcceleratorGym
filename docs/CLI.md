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
├── diagnostics
│   └── monitor
└── magnets
    └── quadrupole

ag> browse /magnets 2
/magnets
└── quadrupole
    ├── QD
    └── QF

ag> browse /magnets/quadrupole 2
/magnets/quadrupole
├── QD
│   └── K1
└── QF
    └── K1

ag> browse /magnets/quadrupole/QF 2
/magnets/quadrupole/QF
└── K1  RW  [1/m]

ag> browse /magnets/quadrupole/QF/K1
/magnets/quadrupole/QF/K1
  variable:  QF:K1
  units:     1/m
  read:      yes
  write:     yes
  limits:    [0.0, 5.0]
```

Directory nodes are shown in bold (on supported terminals). Browsing a leaf attribute shows its full metadata.

#### `query <sql>`

Run a read-only SQL query against the device catalog. Only SELECT statements are allowed. Results are displayed as an aligned table.

Available tables:

| Table | Columns |
|---|---|
| `devices` | `device_id`, `system`, `device_type`, `s_position`, `tree_path` |
| `attributes` | `device_id`, `attribute_name`, `value`, `unit`, `readable`, `writable`, `lower_limit`, `upper_limit`, `variable` |

Join `attributes` with `devices` on `device_id` to filter by system or device type. The `variable` column is the flat name used with `get` and `set` (e.g. `QF:K1`).

```
ag> query SELECT attribute_name, unit FROM attributes LIMIT 3
attribute_name  unit
--------------  ----
K1              1/m
orbit.x         mm
orbit.y         mm

(3 rows)
```

```
ag> query SELECT a.variable, a.unit FROM attributes a JOIN devices d ON a.device_id = d.device_id WHERE d.system = 'magnets'
variable  unit
--------  ----
QF:K1     1/m
QD:K1     1/m

(2 rows)
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
