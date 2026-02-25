"""In-memory SQLite catalog built from a hierarchical device tree."""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

from accelerator_gym.backends.base import Backend
from accelerator_gym.core.variable import Variable

logger = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE devices (
    device_id   TEXT PRIMARY KEY,
    system      TEXT NOT NULL,
    device_type TEXT NOT NULL,
    s_position  REAL,
    tree_path   TEXT NOT NULL
);

CREATE TABLE attributes (
    device_id      TEXT NOT NULL REFERENCES devices(device_id),
    attribute_name TEXT NOT NULL,
    description    TEXT NOT NULL DEFAULT '',
    value          REAL,
    unit           TEXT,
    readable       INTEGER NOT NULL DEFAULT 1,
    writable       INTEGER NOT NULL DEFAULT 1,
    lower_limit    REAL,
    upper_limit    REAL,
    variable       TEXT UNIQUE,
    PRIMARY KEY (device_id, attribute_name)
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
        self._backend = backend
        self._create_tables()
        self._populate(devices, backend)
        # Lock the database to read-only now that population is complete
        self._conn.execute("PRAGMA query_only = ON")
        # Cache the tree structure for browsing
        self._tree = devices

    def _create_tables(self) -> None:
        self._conn.executescript(_SCHEMA)

    def _populate(self, devices: dict[str, Any], backend: Backend) -> None:
        for system_name, device_types in devices.items():
            for dtype_name, instances in device_types.items():
                for dev_name, dev_def in instances.items():
                    tree_path = dev_def.get("tree_path") or (
                        f"{system_name}/{dtype_name}/{dev_name}"
                    )
                    self._conn.execute(
                        "INSERT INTO devices "
                        "(device_id, system, device_type, s_position, tree_path) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            dev_name,
                            system_name,
                            dtype_name,
                            dev_def.get("s_position"),
                            tree_path,
                        ),
                    )
                    for attr_name, attr_def in dev_def.get("attributes", {}).items():
                        variable = backend.resolve_variable_name(
                            system_name, dtype_name, dev_name, attr_name
                        )
                        limits = attr_def.get("limits")
                        self._conn.execute(
                            "INSERT INTO attributes "
                            "(device_id, attribute_name, description, value, unit, readable, writable, "
                            "lower_limit, upper_limit, variable) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                dev_name,
                                attr_name,
                                attr_def.get("description", ""),
                                None,
                                attr_def.get("units"),
                                int(attr_def.get("read", True)),
                                int(attr_def.get("write", True)),
                                limits[0] if limits else None,
                                limits[1] if limits else None,
                                variable,
                            ),
                        )
        self._conn.commit()
        logger.info("Catalog populated from device tree")

    def build_variables(self) -> dict[str, Variable]:
        """Generate a flat variable dict from the catalog for get/set operations."""
        rows = self._conn.execute(
            "SELECT variable, description, unit, readable, writable, lower_limit, upper_limit FROM attributes"
        ).fetchall()
        variables: dict[str, Variable] = {}
        for row in rows:
            limits = None
            if row["lower_limit"] is not None and row["upper_limit"] is not None:
                limits = (row["lower_limit"], row["upper_limit"])
            variables[row["variable"]] = Variable(
                name=row["variable"],
                description=row["description"],
                units=row["unit"],
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
            "SELECT DISTINCT system FROM devices ORDER BY system"
        ).fetchall()
        children: list[Any] = [row["system"] for row in rows]
        if depth > 1:
            children = [
                {"name": name, "children": self._browse_system(name, depth - 1)["children"]}
                for name in children
            ]
        return {"path": "/", "children": children}

    def _browse_system(self, system: str, depth: int) -> dict[str, Any]:
        rows = self._conn.execute(
            "SELECT DISTINCT device_type FROM devices WHERE system = ? ORDER BY device_type",
            (system,),
        ).fetchall()
        if not rows:
            return {"error": "not_found", "message": f"System not found: {system}"}
        children: list[Any] = [row["device_type"] for row in rows]
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
            "SELECT device_id FROM devices "
            "WHERE system = ? AND device_type = ? ORDER BY device_id",
            (system, device_type),
        ).fetchall()
        if not rows:
            return {
                "error": "not_found",
                "message": f"Device type not found: {system}/{device_type}",
            }
        children: list[Any] = [{"name": row["device_id"]} for row in rows]
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
        self, system: str, device_type: str, device_id: str, depth: int
    ) -> dict[str, Any]:
        rows = self._conn.execute(
            "SELECT a.attribute_name, a.unit, a.readable, a.writable, a.lower_limit, a.upper_limit, "
            "a.variable FROM attributes a "
            "JOIN devices d ON a.device_id = d.device_id "
            "WHERE d.system = ? AND d.device_type = ? AND d.device_id = ? "
            "ORDER BY a.attribute_name",
            (system, device_type, device_id),
        ).fetchall()
        if not rows:
            return {
                "error": "not_found",
                "message": f"Device not found: {system}/{device_type}/{device_id}",
            }
        children: list[Any] = [row["attribute_name"] for row in rows]
        if depth > 1:
            children = [
                {
                    "name": row["attribute_name"],
                    **self._attr_row_to_metadata(row),
                }
                for row in rows
            ]
        return {
            "path": f"/{system}/{device_type}/{device_id}",
            "children": children,
        }

    def _browse_attribute(
        self, system: str, device_type: str, device_id: str, attr_name: str
    ) -> dict[str, Any]:
        row = self._conn.execute(
            "SELECT a.attribute_name, a.unit, a.readable, a.writable, a.lower_limit, a.upper_limit, "
            "a.variable FROM attributes a "
            "JOIN devices d ON a.device_id = d.device_id "
            "WHERE d.system = ? AND d.device_type = ? AND d.device_id = ? AND a.attribute_name = ?",
            (system, device_type, device_id, attr_name),
        ).fetchone()
        if row is None:
            return {
                "error": "not_found",
                "message": f"Attribute not found: {system}/{device_type}/{device_id}/{attr_name}",
            }
        return {
            "path": f"/{system}/{device_type}/{device_id}/{attr_name}",
            **self._attr_row_to_metadata(row),
        }

    @staticmethod
    def _attr_row_to_metadata(row: sqlite3.Row) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "units": row["unit"],
            "read": bool(row["readable"]),
            "write": bool(row["writable"]),
            "variable": row["variable"],
        }
        if row["lower_limit"] is not None and row["upper_limit"] is not None:
            meta["limits"] = [row["lower_limit"], row["upper_limit"]]
        return meta
