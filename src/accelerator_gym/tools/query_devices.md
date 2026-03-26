Search, filter, or aggregate devices by their properties using SQL.

Use this to find devices matching specific criteria (e.g. by type, s-position
range, or attribute limits) or to count/aggregate device metadata.

Tables: devices(device_id, system, device_type, s_position, tree_path),
attributes(device_id, attribute_name, description, value, unit, readable, writable,
lower_limit, upper_limit, variable).
JOIN attributes with devices on device_id to filter by system/device_type.
Example: SELECT a.variable, a.unit FROM attributes a JOIN devices d
ON a.device_id = d.device_id WHERE d.system = 'magnets';
