"""In-memory SQLite catalog built from a hierarchical device tree."""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

from accelerator_gym.backends.base import Backend
from accelerator_gym.core.variable import Variable

logger = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE systems (
    name TEXT PRIMARY KEY
);

CREATE TABLE device_types (
    name   TEXT NOT NULL,
    system TEXT NOT NULL REFERENCES systems(name),
    PRIMARY KEY (name, system)
);

CREATE TABLE devices (
    name        TEXT NOT NULL,
    device_type TEXT NOT NULL,
    system      TEXT NOT NULL,
    description TEXT DEFAULT '',
    PRIMARY KEY (name, device_type, system),
    FOREIGN KEY (device_type, system) REFERENCES device_types(name, system)
);

CREATE TABLE attributes (
    device_name TEXT NOT NULL,
    device_type TEXT NOT NULL,
    system      TEXT NOT NULL,
    attr_name   TEXT NOT NULL,
    variable    TEXT NOT NULL UNIQUE,
    description TEXT DEFAULT '',
    units       TEXT,
    readable    INTEGER DEFAULT 1,
    writable    INTEGER DEFAULT 1,
    limit_low   REAL,
    limit_high  REAL,
    PRIMARY KEY (device_name, device_type, system, attr_name),
    FOREIGN KEY (device_name, device_type, system) REFERENCES devices(name, device_type, system)
);
"""


class Catalog:
    """In-memory SQLite catalog built from the device tree YAML.

    Provides tree browsing, SQL queries, and flat variable registry generation.
    """

    def __init__(self, devices: dict[str, Any], backend: Backend) -> None:
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._create_tables()
        self._populate(devices, backend)
        # Cache the tree structure for browsing
        self._tree = devices

    def _create_tables(self) -> None:
        self._conn.executescript(_SCHEMA)

    def _populate(self, devices: dict[str, Any], backend: Backend) -> None:
        for system_name, device_types in devices.items():
            self._conn.execute(
                "INSERT INTO systems (name) VALUES (?)", (system_name,)
            )
            for dtype_name, instances in device_types.items():
                self._conn.execute(
                    "INSERT INTO device_types (name, system) VALUES (?, ?)",
                    (dtype_name, system_name),
                )
                for dev_name, dev_def in instances.items():
                    description = dev_def.get("description", "")
                    self._conn.execute(
                        "INSERT INTO devices (name, device_type, system, description) "
                        "VALUES (?, ?, ?, ?)",
                        (dev_name, dtype_name, system_name, description),
                    )
                    for attr_name, attr_def in dev_def.get("attributes", {}).items():
                        variable = backend.resolve_variable_name(
                            system_name, dtype_name, dev_name, attr_name
                        )
                        limits = attr_def.get("limits")
                        self._conn.execute(
                            "INSERT INTO attributes "
                            "(device_name, device_type, system, attr_name, variable, "
                            "description, units, readable, writable, limit_low, limit_high) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                dev_name,
                                dtype_name,
                                system_name,
                                attr_name,
                                variable,
                                attr_def.get("description", ""),
                                attr_def.get("units"),
                                int(attr_def.get("read", True)),
                                int(attr_def.get("write", True)),
                                limits[0] if limits else None,
                                limits[1] if limits else None,
                            ),
                        )
        self._conn.commit()
        logger.info("Catalog populated from device tree")

    def build_variables(self) -> dict[str, Variable]:
        """Generate a flat variable dict from the catalog for get/set operations."""
        rows = self._conn.execute(
            "SELECT variable, description, units, readable, writable, "
            "limit_low, limit_high FROM attributes"
        ).fetchall()
        variables: dict[str, Variable] = {}
        for row in rows:
            limits = None
            if row["limit_low"] is not None and row["limit_high"] is not None:
                limits = (row["limit_low"], row["limit_high"])
            variables[row["variable"]] = Variable(
                name=row["variable"],
                description=row["description"],
                units=row["units"],
                readable=bool(row["readable"]),
                writable=bool(row["writable"]),
                limits=limits,
            )
        return dict(sorted(variables.items()))

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        """Execute a read-only SQL query. Returns rows as dicts.

        Raises ValueError for non-SELECT statements.
        """
        stripped = sql.strip().upper()
        if not stripped.startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed")
        cursor = self._conn.execute(sql, params)
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def browse(self, path: str, depth: int = 1) -> dict[str, Any]:
        """Navigate the device tree using filesystem-like paths.

        Args:
            path: ``"/"`` for root, ``"/system"``, ``"/system/type"``,
                  ``"/system/type/device"``, or ``"/system/type/device/attr"``.
            depth: How many levels below *path* to include (1 = immediate children).

        Returns:
            Dict with ``path`` and ``children`` (or ``metadata`` for leaf attributes).
        """
        parts = [p for p in path.strip("/").split("/") if p]

        if len(parts) == 0:
            return self._browse_root(depth)
        if len(parts) == 1:
            return self._browse_system(parts[0], depth)
        if len(parts) == 2:
            return self._browse_device_type(parts[0], parts[1], depth)
        if len(parts) == 3:
            return self._browse_device(parts[0], parts[1], parts[2], depth)
        if len(parts) == 4:
            return self._browse_attribute(parts[0], parts[1], parts[2], parts[3])
        return {"error": "invalid_path", "message": f"Path too deep: {path}"}

    def _browse_root(self, depth: int) -> dict[str, Any]:
        rows = self._conn.execute(
            "SELECT name FROM systems ORDER BY name"
        ).fetchall()
        children: list[Any] = [row["name"] for row in rows]
        if depth > 1:
            children = [
                {"name": name, "children": self._browse_system(name, depth - 1)["children"]}
                for name in children
            ]
        return {"path": "/", "children": children}

    def _browse_system(self, system: str, depth: int) -> dict[str, Any]:
        rows = self._conn.execute(
            "SELECT name FROM device_types WHERE system = ? ORDER BY name",
            (system,),
        ).fetchall()
        if not rows:
            return {"error": "not_found", "message": f"System not found: {system}"}
        children: list[Any] = [row["name"] for row in rows]
        if depth > 1:
            children = [
                {"name": name, "children": self._browse_device_type(system, name, depth - 1)["children"]}
                for name in children
            ]
        return {"path": f"/{system}", "children": children}

    def _browse_device_type(
        self, system: str, device_type: str, depth: int
    ) -> dict[str, Any]:
        rows = self._conn.execute(
            "SELECT name, description FROM devices "
            "WHERE system = ? AND device_type = ? ORDER BY name",
            (system, device_type),
        ).fetchall()
        if not rows:
            return {
                "error": "not_found",
                "message": f"Device type not found: {system}/{device_type}",
            }
        children: list[Any] = [
            {"name": row["name"], "description": row["description"]} for row in rows
        ]
        if depth > 1:
            children = [
                {
                    **child,
                    "children": self._browse_device(
                        system, device_type, child["name"], depth - 1
                    )["children"],
                }
                for child in children
            ]
        return {"path": f"/{system}/{device_type}", "children": children}

    def _browse_device(
        self, system: str, device_type: str, device_name: str, depth: int
    ) -> dict[str, Any]:
        rows = self._conn.execute(
            "SELECT attr_name, description, units, readable, writable, "
            "limit_low, limit_high, variable FROM attributes "
            "WHERE system = ? AND device_type = ? AND device_name = ? "
            "ORDER BY attr_name",
            (system, device_type, device_name),
        ).fetchall()
        if not rows:
            return {
                "error": "not_found",
                "message": f"Device not found: {system}/{device_type}/{device_name}",
            }
        children: list[Any] = [row["attr_name"] for row in rows]
        if depth > 1:
            children = [
                {
                    "name": row["attr_name"],
                    **self._attr_row_to_metadata(row),
                }
                for row in rows
            ]
        return {
            "path": f"/{system}/{device_type}/{device_name}",
            "children": children,
        }

    def _browse_attribute(
        self, system: str, device_type: str, device_name: str, attr_name: str
    ) -> dict[str, Any]:
        row = self._conn.execute(
            "SELECT attr_name, description, units, readable, writable, "
            "limit_low, limit_high, variable FROM attributes "
            "WHERE system = ? AND device_type = ? AND device_name = ? AND attr_name = ?",
            (system, device_type, device_name, attr_name),
        ).fetchone()
        if row is None:
            return {
                "error": "not_found",
                "message": f"Attribute not found: {system}/{device_type}/{device_name}/{attr_name}",
            }
        return {
            "path": f"/{system}/{device_type}/{device_name}/{attr_name}",
            **self._attr_row_to_metadata(row),
        }

    @staticmethod
    def _attr_row_to_metadata(row: sqlite3.Row) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "description": row["description"],
            "units": row["units"],
            "read": bool(row["readable"]),
            "write": bool(row["writable"]),
            "variable": row["variable"],
        }
        if row["limit_low"] is not None and row["limit_high"] is not None:
            meta["limits"] = [row["limit_low"], row["limit_high"]]
        return meta
