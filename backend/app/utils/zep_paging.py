"""Compatibility shim — Zep paging utilities are no longer needed.

All graph data is now stored locally in SQLite via LocalGraphStore.
These functions are kept as no-ops to avoid import errors from any
remaining call sites that have not been migrated yet.
"""

from __future__ import annotations

from typing import Any


def fetch_all_nodes(*_args: Any, **_kwargs: Any) -> list[Any]:
    """No-op stub — use LocalGraphStore.get_nodes_by_graph() instead."""
    return []


def fetch_all_edges(*_args: Any, **_kwargs: Any) -> list[Any]:
    """No-op stub — use LocalGraphStore.get_edges_by_graph() instead."""
    return []

