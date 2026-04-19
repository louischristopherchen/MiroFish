"""
本地图谱存储服务（SQLite）
替代 Zep Cloud API，使用本地 SQLite 数据库存储图谱数据

功能：
1. 图谱 CRUD（create, delete, get）
2. 节点/边 CRUD
3. 本体（ontology）存储
4. Episode 存储与处理状态管理
"""

import json
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..config import Config
from ..utils.logger import get_logger

logger = get_logger('mirofish.local_graph_store')

# Default database directory — inside uploads/ so data persists across runs
_DEFAULT_DB_DIR = os.path.join(os.path.dirname(__file__), '../../uploads/graph_db')


class LocalGraphStore:
    """
    本地 SQLite 图谱存储

    Thread-safe: 每个线程使用独立的 SQLite 连接。
    """

    def __init__(self, db_dir: Optional[str] = None):
        self.db_dir = db_dir or _DEFAULT_DB_DIR
        os.makedirs(self.db_dir, exist_ok=True)
        self.db_path = os.path.join(self.db_dir, 'graph_store.db')
        self._local = threading.local()
        self._init_schema()
        logger.info(f"LocalGraphStore initialized: {self.db_path}")

    # ── connection helpers ──────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        """Return a thread-local connection."""
        conn = getattr(self._local, 'conn', None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    @contextmanager
    def _cursor(self):
        conn = self._get_conn()
        cur = conn.cursor()
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # ── schema ──────────────────────────────────────────────

    def _init_schema(self):
        with self._cursor() as cur:
            cur.executescript("""
                CREATE TABLE IF NOT EXISTS graphs (
                    graph_id   TEXT PRIMARY KEY,
                    name       TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    ontology   TEXT DEFAULT '{}',
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS nodes (
                    uuid       TEXT PRIMARY KEY,
                    graph_id   TEXT NOT NULL,
                    name       TEXT NOT NULL,
                    labels     TEXT DEFAULT '[]',       -- JSON array
                    summary    TEXT DEFAULT '',
                    attributes TEXT DEFAULT '{}',       -- JSON object
                    created_at TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY (graph_id) REFERENCES graphs(graph_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_nodes_graph ON nodes(graph_id);
                CREATE INDEX IF NOT EXISTS idx_nodes_name  ON nodes(name);

                CREATE TABLE IF NOT EXISTS edges (
                    uuid              TEXT PRIMARY KEY,
                    graph_id          TEXT NOT NULL,
                    name              TEXT DEFAULT '',
                    fact              TEXT DEFAULT '',
                    source_node_uuid  TEXT NOT NULL,
                    target_node_uuid  TEXT NOT NULL,
                    attributes        TEXT DEFAULT '{}',
                    created_at        TEXT DEFAULT (datetime('now')),
                    valid_at          TEXT,
                    invalid_at        TEXT,
                    expired_at        TEXT,
                    episodes          TEXT DEFAULT '[]',
                    FOREIGN KEY (graph_id) REFERENCES graphs(graph_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_edges_graph  ON edges(graph_id);
                CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_node_uuid);
                CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_node_uuid);

                CREATE TABLE IF NOT EXISTS episodes (
                    uuid       TEXT PRIMARY KEY,
                    graph_id   TEXT NOT NULL,
                    data       TEXT NOT NULL,
                    type       TEXT DEFAULT 'text',
                    processed  INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY (graph_id) REFERENCES graphs(graph_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_episodes_graph ON episodes(graph_id);
            """)

    # ── Graph CRUD ──────────────────────────────────────────

    def create_graph(self, graph_id: str, name: str, description: str = "") -> str:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO graphs (graph_id, name, description) VALUES (?, ?, ?)",
                (graph_id, name, description),
            )
        logger.info(f"Graph created: {graph_id} ({name})")
        return graph_id

    def delete_graph(self, graph_id: str):
        with self._cursor() as cur:
            cur.execute("DELETE FROM graphs WHERE graph_id = ?", (graph_id,))
        logger.info(f"Graph deleted: {graph_id}")

    def get_graph(self, graph_id: str) -> Optional[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM graphs WHERE graph_id = ?", (graph_id,))
            row = cur.fetchone()
        if row is None:
            return None
        return dict(row)

    # ── Ontology ────────────────────────────────────────────

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]):
        with self._cursor() as cur:
            cur.execute(
                "UPDATE graphs SET ontology = ? WHERE graph_id = ?",
                (json.dumps(ontology, ensure_ascii=False), graph_id),
            )
        logger.info(f"Ontology set for graph {graph_id}")

    def get_ontology(self, graph_id: str) -> Dict[str, Any]:
        with self._cursor() as cur:
            cur.execute("SELECT ontology FROM graphs WHERE graph_id = ?", (graph_id,))
            row = cur.fetchone()
        if row is None:
            return {}
        return json.loads(row["ontology"])

    # ── Node CRUD ───────────────────────────────────────────

    def add_node(
        self,
        graph_id: str,
        name: str,
        labels: Optional[List[str]] = None,
        summary: str = "",
        attributes: Optional[Dict[str, Any]] = None,
        node_uuid: Optional[str] = None,
    ) -> str:
        node_uuid = node_uuid or uuid.uuid4().hex
        labels = labels or ["Entity"]
        attributes = attributes or {}
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO nodes (uuid, graph_id, name, labels, summary, attributes)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    node_uuid,
                    graph_id,
                    name,
                    json.dumps(labels, ensure_ascii=False),
                    summary,
                    json.dumps(attributes, ensure_ascii=False),
                ),
            )
        return node_uuid

    def upsert_node(
        self,
        graph_id: str,
        name: str,
        labels: Optional[List[str]] = None,
        summary: str = "",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Insert or update a node, matching by (graph_id, name)."""
        existing = self.find_node_by_name(graph_id, name)
        if existing:
            node_uuid = existing["uuid"]
            # Merge attributes
            old_attrs = json.loads(existing["attributes"]) if isinstance(existing["attributes"], str) else existing["attributes"]
            new_attrs = {**old_attrs, **(attributes or {})}
            # Merge labels
            old_labels = json.loads(existing["labels"]) if isinstance(existing["labels"], str) else existing["labels"]
            merged_labels = list(set(old_labels + (labels or [])))
            # Update summary if new one is longer
            new_summary = summary if len(summary) > len(existing["summary"]) else existing["summary"]
            with self._cursor() as cur:
                cur.execute(
                    """UPDATE nodes SET labels = ?, summary = ?, attributes = ?
                       WHERE uuid = ?""",
                    (
                        json.dumps(merged_labels, ensure_ascii=False),
                        new_summary,
                        json.dumps(new_attrs, ensure_ascii=False),
                        node_uuid,
                    ),
                )
            return node_uuid
        else:
            return self.add_node(graph_id, name, labels, summary, attributes)

    def find_node_by_name(self, graph_id: str, name: str) -> Optional[Dict[str, Any]]:
        """Find a node by exact name (case-insensitive)."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM nodes WHERE graph_id = ? AND LOWER(name) = LOWER(?)",
                (graph_id, name),
            )
            row = cur.fetchone()
        return dict(row) if row else None

    def get_node(self, node_uuid: str) -> Optional[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM nodes WHERE uuid = ?", (node_uuid,))
            row = cur.fetchone()
        if row is None:
            return None
        result = dict(row)
        result["labels"] = json.loads(result["labels"])
        result["attributes"] = json.loads(result["attributes"])
        return result

    def get_nodes_by_graph(self, graph_id: str) -> List[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM nodes WHERE graph_id = ?", (graph_id,))
            rows = cur.fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["labels"] = json.loads(d["labels"])
            d["attributes"] = json.loads(d["attributes"])
            result.append(d)
        return result

    # ── Edge CRUD ───────────────────────────────────────────

    def add_edge(
        self,
        graph_id: str,
        name: str,
        fact: str,
        source_node_uuid: str,
        target_node_uuid: str,
        attributes: Optional[Dict[str, Any]] = None,
        valid_at: Optional[str] = None,
        invalid_at: Optional[str] = None,
        expired_at: Optional[str] = None,
        episode_uuid: Optional[str] = None,
        edge_uuid: Optional[str] = None,
    ) -> str:
        edge_uuid = edge_uuid or uuid.uuid4().hex
        attributes = attributes or {}
        episodes = [episode_uuid] if episode_uuid else []
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO edges
                   (uuid, graph_id, name, fact, source_node_uuid, target_node_uuid,
                    attributes, valid_at, invalid_at, expired_at, episodes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    edge_uuid,
                    graph_id,
                    name,
                    fact,
                    source_node_uuid,
                    target_node_uuid,
                    json.dumps(attributes, ensure_ascii=False),
                    valid_at,
                    invalid_at,
                    expired_at,
                    json.dumps(episodes),
                ),
            )
        return edge_uuid

    def get_edges_by_graph(self, graph_id: str) -> List[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM edges WHERE graph_id = ?", (graph_id,))
            rows = cur.fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["attributes"] = json.loads(d["attributes"])
            d["episodes"] = json.loads(d["episodes"])
            result.append(d)
        return result

    def get_edges_by_node(self, node_uuid: str) -> List[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM edges WHERE source_node_uuid = ? OR target_node_uuid = ?",
                (node_uuid, node_uuid),
            )
            rows = cur.fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["attributes"] = json.loads(d["attributes"])
            d["episodes"] = json.loads(d["episodes"])
            result.append(d)
        return result

    # ── Episode management ──────────────────────────────────

    def add_episode(self, graph_id: str, data: str, ep_type: str = "text") -> str:
        ep_uuid = uuid.uuid4().hex
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO episodes (uuid, graph_id, data, type) VALUES (?, ?, ?, ?)",
                (ep_uuid, graph_id, data, ep_type),
            )
        return ep_uuid

    def mark_episode_processed(self, ep_uuid: str):
        with self._cursor() as cur:
            cur.execute("UPDATE episodes SET processed = 1 WHERE uuid = ?", (ep_uuid,))

    def get_episode(self, ep_uuid: str) -> Optional[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM episodes WHERE uuid = ?", (ep_uuid,))
            row = cur.fetchone()
        return dict(row) if row else None

    def get_unprocessed_episodes(self, graph_id: str) -> List[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM episodes WHERE graph_id = ? AND processed = 0",
                (graph_id,),
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]
