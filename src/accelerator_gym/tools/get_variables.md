Read one or more variables by their variable names.

Variable names come from browse_devices (the "variable" field when you browse to an
attribute) or from querying the metadata database. Use the variable name exactly as
returned — do not modify the format.

If output_file is provided, results are written as CSV to that path instead of
being returned inline. This saves tokens when reading many variables. The CSV has
columns: name, value, units.
