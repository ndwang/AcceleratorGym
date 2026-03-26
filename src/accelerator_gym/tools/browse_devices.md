Browse the device tree to progressively discover how devices are organized.
The catalog is tree-shaped with filesystem-like paths.

Path levels: "/" -> systems, "/system" -> device types,
"/system/type" -> devices, "/system/type/device" -> attributes,
"/system/type/device/attr" -> attribute metadata.

When you browse to an attribute (or use depth so attributes are included), each
attribute has a "variable" field: that is the exact string to pass to get_variables
and set_variables. Use the variable name exactly as returned — do not modify the format.

Use depth > 1 to see multiple levels at once.
