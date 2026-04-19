"""
图谱构建服务
接口2：使用本地图谱存储构建 Standalone Graph（替代 Zep Cloud）
"""

import os
import uuid
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

from ..config import Config
from ..models.task import TaskManager, TaskStatus
from .text_processor import TextProcessor
from ..utils.locale import t, get_locale, set_locale
from .local_graph_store import LocalGraphStore
from .llm_entity_extractor import LLMEntityExtractor
from .local_semantic_search import LocalSemanticSearch


@dataclass
class GraphInfo:
    """图谱信息"""
    graph_id: str
    node_count: int
    edge_count: int
    entity_types: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "entity_types": self.entity_types,
        }


class GraphBuilderService:
    """
    图谱构建服务
    使用本地 SQLite + LLM 抽取构建知识图谱（替代 Zep Cloud）
    """
    
    def __init__(self, api_key: Optional[str] = None):
        # api_key kept for backward-compatible call sites; no longer required
        self.store = LocalGraphStore()
        self.extractor = LLMEntityExtractor(self.store)
        self.search_engine = LocalSemanticSearch(self.store)
        self.task_manager = TaskManager()
    
    def build_graph_async(
        self,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str = "MiroFish Graph",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        batch_size: int = 3
    ) -> str:
        """
        异步构建图谱
        
        Args:
            text: 输入文本
            ontology: 本体定义（来自接口1的输出）
            graph_name: 图谱名称
            chunk_size: 文本块大小
            chunk_overlap: 块重叠大小
            batch_size: 每批发送的块数量
            
        Returns:
            任务ID
        """
        # 创建任务
        task_id = self.task_manager.create_task(
            task_type="graph_build",
            metadata={
                "graph_name": graph_name,
                "chunk_size": chunk_size,
                "text_length": len(text),
            }
        )
        
        # Capture locale before spawning background thread
        current_locale = get_locale()

        # 在后台线程中执行构建
        thread = threading.Thread(
            target=self._build_graph_worker,
            args=(task_id, text, ontology, graph_name, chunk_size, chunk_overlap, batch_size, current_locale)
        )
        thread.daemon = True
        thread.start()
        
        return task_id
    
    def _build_graph_worker(
        self,
        task_id: str,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int,
        locale: str = 'zh'
    ):
        """图谱构建工作线程"""
        set_locale(locale)
        try:
            self.task_manager.update_task(
                task_id,
                status=TaskStatus.PROCESSING,
                progress=5,
                message=t('progress.startBuildingGraph')
            )
            
            # 1. 创建图谱
            graph_id = self.create_graph(graph_name)
            self.task_manager.update_task(
                task_id,
                progress=10,
                message=t('progress.graphCreated', graphId=graph_id)
            )
            
            # 2. 设置本体
            self.set_ontology(graph_id, ontology)
            self.task_manager.update_task(
                task_id,
                progress=15,
                message=t('progress.ontologySet')
            )
            
            # 3. 文本分块
            chunks = TextProcessor.split_text(text, chunk_size, chunk_overlap)
            total_chunks = len(chunks)
            self.task_manager.update_task(
                task_id,
                progress=20,
                message=t('progress.textSplit', count=total_chunks)
            )
            
            # 4. 分批用 LLM 抽取实体和关系
            episode_uuids = self.add_text_batches(
                graph_id, chunks, ontology, batch_size,
                lambda msg, prog: self.task_manager.update_task(
                    task_id,
                    progress=20 + int(prog * 0.6),  # 20-80%
                    message=msg
                )
            )
            
            # 5. 同步向量索引
            self.task_manager.update_task(
                task_id,
                progress=85,
                message=t('progress.fetchingGraphInfo')
            )
            self.search_engine.sync_index(graph_id)
            
            # 6. 获取图谱信息
            self.task_manager.update_task(
                task_id,
                progress=90,
                message=t('progress.fetchingGraphInfo')
            )
            
            graph_info = self._get_graph_info(graph_id)
            
            # 完成
            self.task_manager.complete_task(task_id, {
                "graph_id": graph_id,
                "graph_info": graph_info.to_dict(),
                "chunks_processed": total_chunks,
            })
            
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.task_manager.fail_task(task_id, error_msg)
    
    def create_graph(self, name: str) -> str:
        """创建图谱"""
        graph_id = f"mirofish_{uuid.uuid4().hex[:16]}"
        self.store.create_graph(graph_id=graph_id, name=name, description="MiroFish Social Simulation Graph")
        return graph_id
    
    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]):
        """存储图谱本体定义到数据库"""
        self.store.set_ontology(graph_id, ontology)
    
    def add_text_batches(
        self,
        graph_id: str,
        chunks: List[str],
        ontology: Dict[str, Any],
        batch_size: int = 3,
        progress_callback: Optional[Callable] = None
    ) -> List[str]:
        """分批用 LLM 抽取实体/关系并写入数据库，返回 episode uuid 列表"""
        episode_uuids = []
        total_chunks = len(chunks)
        
        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size
            
            if progress_callback:
                progress = (i + len(batch_chunks)) / total_chunks
                progress_callback(
                    t('progress.sendingBatch', current=batch_num, total=total_batches, chunks=len(batch_chunks)),
                    progress
                )
            
            for chunk in batch_chunks:
                # 1. Store episode
                ep_uuid = self.store.add_episode(graph_id, chunk, "text")
                
                # 2. LLM extraction
                try:
                    self.extractor.extract_and_store(
                        graph_id=graph_id,
                        text=chunk,
                        ontology=ontology,
                        episode_uuid=ep_uuid,
                    )
                    self.store.mark_episode_processed(ep_uuid)
                except Exception as e:
                    if progress_callback:
                        progress_callback(t('progress.batchFailed', batch=batch_num, error=str(e)), 0)
                    raise
                
                episode_uuids.append(ep_uuid)
            
            # Small delay between batches to avoid LLM rate limits
            time.sleep(0.5)
        
        return episode_uuids
    
    def _wait_for_episodes(
        self,
        episode_uuids: List[str],
        progress_callback: Optional[Callable] = None,
        timeout: int = 600
    ):
        """
        Check episode processing status.
        With local LLM extraction, episodes are processed synchronously,
        so this is mostly a no-op kept for backward compatibility.
        """
        if not episode_uuids:
            if progress_callback:
                progress_callback(t('progress.noEpisodesWait'), 1.0)
            return
        
        total_episodes = len(episode_uuids)
        completed_count = 0
        
        for ep_uuid in episode_uuids:
            ep = self.store.get_episode(ep_uuid)
            if ep and ep.get("processed"):
                completed_count += 1
        
        if progress_callback:
            progress_callback(
                t('progress.processingComplete', completed=completed_count, total=total_episodes),
                1.0
            )
    
    def _get_graph_info(self, graph_id: str) -> GraphInfo:
        """获取图谱信息"""
        nodes = self.store.get_nodes_by_graph(graph_id)
        edges = self.store.get_edges_by_graph(graph_id)

        # 统计实体类型
        entity_types = set()
        for node in nodes:
            labels = node.get("labels", [])
            for label in labels:
                if label not in ["Entity", "Node"]:
                    entity_types.add(label)

        return GraphInfo(
            graph_id=graph_id,
            node_count=len(nodes),
            edge_count=len(edges),
            entity_types=list(entity_types)
        )
    
    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        """
        获取完整图谱数据（包含详细信息）
        """
        nodes = self.store.get_nodes_by_graph(graph_id)
        edges = self.store.get_edges_by_graph(graph_id)

        # 创建节点映射
        node_map = {}
        for node in nodes:
            node_map[node["uuid"]] = node.get("name", "")
        
        nodes_data = []
        for node in nodes:
            nodes_data.append({
                "uuid": node["uuid"],
                "name": node.get("name", ""),
                "labels": node.get("labels", []),
                "summary": node.get("summary", ""),
                "attributes": node.get("attributes", {}),
                "created_at": node.get("created_at"),
            })
        
        edges_data = []
        for edge in edges:
            edges_data.append({
                "uuid": edge["uuid"],
                "name": edge.get("name", ""),
                "fact": edge.get("fact", ""),
                "fact_type": edge.get("name", ""),
                "source_node_uuid": edge.get("source_node_uuid", ""),
                "target_node_uuid": edge.get("target_node_uuid", ""),
                "source_node_name": node_map.get(edge.get("source_node_uuid", ""), ""),
                "target_node_name": node_map.get(edge.get("target_node_uuid", ""), ""),
                "attributes": edge.get("attributes", {}),
                "created_at": edge.get("created_at"),
                "valid_at": edge.get("valid_at"),
                "invalid_at": edge.get("invalid_at"),
                "expired_at": edge.get("expired_at"),
                "episodes": edge.get("episodes", []),
            })
        
        return {
            "graph_id": graph_id,
            "nodes": nodes_data,
            "edges": edges_data,
            "node_count": len(nodes_data),
            "edge_count": len(edges_data),
        }
    
    def delete_graph(self, graph_id: str):
        """删除图谱"""
        self.search_engine.delete_index(graph_id)
        self.store.delete_graph(graph_id=graph_id)

