"""
LLM 实体抽取器
替代 Zep 的自动实体/关系抽取功能

从文本中使用 LLM 提取实体（节点）和关系（边），
按照给定的 ontology schema 进行结构化抽取，
然后写入 LocalGraphStore。
"""

import json
from typing import Any, Dict, List, Optional

from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger
from .local_graph_store import LocalGraphStore

logger = get_logger('mirofish.llm_entity_extractor')


# ── Extraction prompt ───────────────────────────────────

_SYSTEM_PROMPT = """You are an expert information extraction system. Your task is to extract entities and relationships from the given text according to the provided ontology schema.

## Rules
1. Only extract entities and relationships that match the types defined in the ontology.
2. Each entity must have a `name` (short, canonical identifier) and a `type` matching one of the entity types.
3. Each relationship must have a `name` matching one of the edge types, plus `source` and `target` entity names.
4. Include a short `fact` sentence for each relationship that describes the relationship in natural language.
5. If the entity has attributes defined in the ontology, extract them when available.
6. Deduplicate entities: if the same real-world entity appears with different surface forms, normalize to one canonical name.
7. Return **valid JSON only**, no extra text.

## Output format
```json
{
  "entities": [
    {
      "name": "Entity Name",
      "type": "EntityType",
      "summary": "Brief summary of the entity",
      "attributes": {"attr_key": "attr_value"}
    }
  ],
  "relationships": [
    {
      "name": "RELATIONSHIP_TYPE",
      "source": "Source Entity Name",
      "target": "Target Entity Name",
      "fact": "Natural language description of this relationship"
    }
  ]
}
```
"""


class LLMEntityExtractor:
    """
    使用 LLM 从文本中抽取实体和关系，并写入 LocalGraphStore。
    """

    def __init__(
        self,
        store: LocalGraphStore,
        llm_client: Optional[LLMClient] = None,
    ):
        self.store = store
        self.llm = llm_client or LLMClient()

    # ── public API ──────────────────────────────────────────

    def extract_and_store(
        self,
        graph_id: str,
        text: str,
        ontology: Dict[str, Any],
        episode_uuid: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract entities/relationships from *text* and persist them.

        Args:
            graph_id: Target graph ID.
            text: The source text chunk.
            ontology: The ontology definition (entity_types, edge_types).
            episode_uuid: Optional episode UUID to link edges to.

        Returns:
            dict with keys ``entities_added``, ``edges_added``.
        """
        # 1. Call LLM to extract
        extraction = self._call_llm(text, ontology)

        entities_added = 0
        edges_added = 0

        # 2. Upsert entities
        entity_name_to_uuid: Dict[str, str] = {}
        for ent in extraction.get("entities", []):
            name = ent.get("name", "").strip()
            if not name:
                continue
            ent_type = ent.get("type", "Entity")
            labels = ["Entity", ent_type] if ent_type != "Entity" else ["Entity"]
            summary = ent.get("summary", "")
            attributes = ent.get("attributes", {})
            node_uuid = self.store.upsert_node(
                graph_id=graph_id,
                name=name,
                labels=labels,
                summary=summary,
                attributes=attributes,
            )
            entity_name_to_uuid[name] = node_uuid
            entities_added += 1

        # 3. Add relationships (edges)
        for rel in extraction.get("relationships", []):
            source_name = rel.get("source", "").strip()
            target_name = rel.get("target", "").strip()
            if not source_name or not target_name:
                continue

            # Resolve UUIDs — create stub nodes if needed
            source_uuid = entity_name_to_uuid.get(source_name)
            if source_uuid is None:
                source_uuid = self.store.upsert_node(graph_id, source_name)
                entity_name_to_uuid[source_name] = source_uuid

            target_uuid = entity_name_to_uuid.get(target_name)
            if target_uuid is None:
                target_uuid = self.store.upsert_node(graph_id, target_name)
                entity_name_to_uuid[target_name] = target_uuid

            self.store.add_edge(
                graph_id=graph_id,
                name=rel.get("name", ""),
                fact=rel.get("fact", ""),
                source_node_uuid=source_uuid,
                target_node_uuid=target_uuid,
                episode_uuid=episode_uuid,
            )
            edges_added += 1

        logger.info(
            f"Extracted {entities_added} entities, {edges_added} edges "
            f"from text ({len(text)} chars) into graph {graph_id}"
        )
        return {"entities_added": entities_added, "edges_added": edges_added}

    # ── internals ───────────────────────────────────────────

    def _call_llm(self, text: str, ontology: Dict[str, Any]) -> Dict[str, Any]:
        """Call LLM for extraction and return parsed JSON."""
        ontology_desc = self._build_ontology_description(ontology)

        user_prompt = f"""## Ontology Schema

{ontology_desc}

## Text to extract from

{text}

Extract all entities and relationships from the text above according to the ontology schema. Return valid JSON only."""

        try:
            result = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=4096,
            )
            return result
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return {"entities": [], "relationships": []}

    @staticmethod
    def _build_ontology_description(ontology: Dict[str, Any]) -> str:
        """Build a human-readable ontology description for the prompt."""
        parts: List[str] = []

        parts.append("### Entity Types")
        for et in ontology.get("entity_types", []):
            name = et.get("name", "")
            desc = et.get("description", "")
            attrs = et.get("attributes", [])
            attr_str = ", ".join(a.get("name", "") for a in attrs)
            parts.append(f"- **{name}**: {desc}")
            if attr_str:
                parts.append(f"  Attributes: {attr_str}")

        parts.append("\n### Relationship Types (Edge Types)")
        for edge in ontology.get("edge_types", []):
            name = edge.get("name", "")
            desc = edge.get("description", "")
            sts = edge.get("source_targets", [])
            st_str = ", ".join(f"{s.get('source', '?')} → {s.get('target', '?')}" for s in sts)
            parts.append(f"- **{name}**: {desc}")
            if st_str:
                parts.append(f"  Source→Target: {st_str}")

        return "\n".join(parts)
