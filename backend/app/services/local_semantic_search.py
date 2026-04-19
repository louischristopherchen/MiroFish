"""
本地语义搜索服务（ChromaDB）
替代 Zep 的 graph.search() 功能

功能：
1. 使用 ChromaDB 存储 edges/nodes 的向量嵌入
2. 支持语义搜索 + 关键词匹配的混合搜索
3. 自动在图谱数据变更时同步向量索引
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from ..utils.logger import get_logger
from .local_graph_store import LocalGraphStore

logger = get_logger('mirofish.local_semantic_search')


def _get_chromadb_client(persist_dir: str):
    """Lazy-import chromadb and return a persistent client."""
    import chromadb
    return chromadb.PersistentClient(path=persist_dir)


class LocalSemanticSearch:
    """
    本地语义搜索引擎

    Wraps ChromaDB for vector storage and similarity search.
    Falls back to keyword matching when ChromaDB is unavailable.
    """

    def __init__(
        self,
        store: LocalGraphStore,
        persist_dir: Optional[str] = None,
    ):
        self.store = store
        self._persist_dir = persist_dir or os.path.join(store.db_dir, 'chroma_data')
        os.makedirs(self._persist_dir, exist_ok=True)
        self._client = None
        self._chroma_available = True
        self._init_chromadb()

    def _init_chromadb(self):
        try:
            self._client = _get_chromadb_client(self._persist_dir)
            logger.info("ChromaDB initialized successfully")
        except Exception as e:
            logger.warning(f"ChromaDB not available, falling back to keyword search: {e}")
            self._chroma_available = False

    # ── collection helpers ──────────────────────────────────

    def _get_edge_collection(self, graph_id: str):
        safe_name = f"edges_{graph_id.replace('-', '_')}"
        # ChromaDB collection names must be 3-63 chars and alphanumeric with underscores
        safe_name = safe_name[:63]
        return self._client.get_or_create_collection(name=safe_name)

    def _get_node_collection(self, graph_id: str):
        safe_name = f"nodes_{graph_id.replace('-', '_')}"
        safe_name = safe_name[:63]
        return self._client.get_or_create_collection(name=safe_name)

    # ── index sync ──────────────────────────────────────────

    def sync_index(self, graph_id: str):
        """
        Rebuild vector index from the current graph data in SQLite.
        Call this after adding new data to the graph.
        """
        if not self._chroma_available:
            return

        self._sync_edges(graph_id)
        self._sync_nodes(graph_id)
        logger.info(f"Vector index synced for graph {graph_id}")

    def _sync_edges(self, graph_id: str):
        collection = self._get_edge_collection(graph_id)
        edges = self.store.get_edges_by_graph(graph_id)
        if not edges:
            return

        ids = []
        documents = []
        metadatas = []
        for edge in edges:
            doc_text = f"{edge.get('name', '')} {edge.get('fact', '')}"
            if not doc_text.strip():
                continue
            ids.append(edge["uuid"])
            documents.append(doc_text)
            metadatas.append({
                "name": edge.get("name", ""),
                "fact": edge.get("fact", ""),
                "source_node_uuid": edge.get("source_node_uuid", ""),
                "target_node_uuid": edge.get("target_node_uuid", ""),
                "graph_id": graph_id,
            })

        if ids:
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def _sync_nodes(self, graph_id: str):
        collection = self._get_node_collection(graph_id)
        nodes = self.store.get_nodes_by_graph(graph_id)
        if not nodes:
            return

        ids = []
        documents = []
        metadatas = []
        for node in nodes:
            doc_text = f"{node.get('name', '')} {node.get('summary', '')}"
            if not doc_text.strip():
                continue
            ids.append(node["uuid"])
            documents.append(doc_text)
            labels = node.get("labels", [])
            if isinstance(labels, str):
                labels = json.loads(labels)
            metadatas.append({
                "name": node.get("name", ""),
                "summary": node.get("summary", ""),
                "labels": json.dumps(labels),
                "graph_id": graph_id,
            })

        if ids:
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def delete_index(self, graph_id: str):
        """Delete vector index for a graph."""
        if not self._chroma_available:
            return
        try:
            edge_name = f"edges_{graph_id.replace('-', '_')}"[:63]
            self._client.delete_collection(edge_name)
        except Exception:
            pass
        try:
            node_name = f"nodes_{graph_id.replace('-', '_')}"[:63]
            self._client.delete_collection(node_name)
        except Exception:
            pass

    # ── search ──────────────────────────────────────────────

    def search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",
    ) -> Dict[str, Any]:
        """
        Hybrid search: ChromaDB vector search + keyword fallback.

        Args:
            graph_id: Graph to search in.
            query: Natural language query.
            limit: Max results.
            scope: "edges", "nodes", or "both".

        Returns:
            dict with keys ``facts``, ``edges``, ``nodes``, ``query``, ``total_count``.
        """
        if self._chroma_available:
            return self._vector_search(graph_id, query, limit, scope)
        else:
            return self._keyword_search(graph_id, query, limit, scope)

    def _vector_search(
        self,
        graph_id: str,
        query: str,
        limit: int,
        scope: str,
    ) -> Dict[str, Any]:
        facts: List[str] = []
        edges: List[Dict[str, Any]] = []
        nodes: List[Dict[str, Any]] = []

        try:
            if scope in ("edges", "both"):
                collection = self._get_edge_collection(graph_id)
                results = collection.query(query_texts=[query], n_results=limit)
                if results and results.get("metadatas"):
                    for meta_list in results["metadatas"]:
                        for meta in meta_list:
                            fact = meta.get("fact", "")
                            if fact:
                                facts.append(fact)
                            edges.append({
                                "uuid": "",
                                "name": meta.get("name", ""),
                                "fact": fact,
                                "source_node_uuid": meta.get("source_node_uuid", ""),
                                "target_node_uuid": meta.get("target_node_uuid", ""),
                            })

            if scope in ("nodes", "both"):
                collection = self._get_node_collection(graph_id)
                results = collection.query(query_texts=[query], n_results=limit)
                if results and results.get("metadatas"):
                    for meta_list in results["metadatas"]:
                        for meta in meta_list:
                            summary = meta.get("summary", "")
                            name = meta.get("name", "")
                            labels_str = meta.get("labels", "[]")
                            labels = json.loads(labels_str) if isinstance(labels_str, str) else labels_str
                            nodes.append({
                                "uuid": "",
                                "name": name,
                                "labels": labels,
                                "summary": summary,
                            })
                            if summary:
                                facts.append(f"[{name}]: {summary}")
        except Exception as e:
            logger.warning(f"ChromaDB search failed, falling back to keyword: {e}")
            return self._keyword_search(graph_id, query, limit, scope)

        return {
            "facts": facts,
            "edges": edges,
            "nodes": nodes,
            "query": query,
            "total_count": len(facts),
        }

    def _keyword_search(
        self,
        graph_id: str,
        query: str,
        limit: int,
        scope: str,
    ) -> Dict[str, Any]:
        """Pure keyword matching fallback."""
        query_lower = query.lower()
        keywords = [
            w.strip()
            for w in query_lower.replace(",", " ").replace("，", " ").split()
            if len(w.strip()) > 1
        ]

        def score(text: str) -> int:
            if not text:
                return 0
            t = text.lower()
            s = 100 if query_lower in t else 0
            for kw in keywords:
                if kw in t:
                    s += 10
            return s

        facts: List[str] = []
        edges_out: List[Dict[str, Any]] = []
        nodes_out: List[Dict[str, Any]] = []

        if scope in ("edges", "both"):
            all_edges = self.store.get_edges_by_graph(graph_id)
            scored = [(score(e.get("fact", "")) + score(e.get("name", "")), e) for e in all_edges]
            scored = [(s, e) for s, e in scored if s > 0]
            scored.sort(key=lambda x: x[0], reverse=True)
            for _, e in scored[:limit]:
                if e.get("fact"):
                    facts.append(e["fact"])
                edges_out.append({
                    "uuid": e.get("uuid", ""),
                    "name": e.get("name", ""),
                    "fact": e.get("fact", ""),
                    "source_node_uuid": e.get("source_node_uuid", ""),
                    "target_node_uuid": e.get("target_node_uuid", ""),
                })

        if scope in ("nodes", "both"):
            all_nodes = self.store.get_nodes_by_graph(graph_id)
            scored = [(score(n.get("name", "")) + score(n.get("summary", "")), n) for n in all_nodes]
            scored = [(s, n) for s, n in scored if s > 0]
            scored.sort(key=lambda x: x[0], reverse=True)
            for _, n in scored[:limit]:
                labels = n.get("labels", [])
                if isinstance(labels, str):
                    labels = json.loads(labels)
                nodes_out.append({
                    "uuid": n.get("uuid", ""),
                    "name": n.get("name", ""),
                    "labels": labels,
                    "summary": n.get("summary", ""),
                })
                if n.get("summary"):
                    facts.append(f"[{n.get('name', '')}]: {n.get('summary', '')}")

        return {
            "facts": facts,
            "edges": edges_out,
            "nodes": nodes_out,
            "query": query,
            "total_count": len(facts),
        }
