"""Entity extraction — extracts entities and relations from text using LLM."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from llm.base import LLMMessage, LLMRequest
from llm.router import ModelRouter, TaskMetadata
from llm.schemas import EntityExtractionOutput, parse_llm_output

logger = logging.getLogger(__name__)


ENTITY_TYPES = [
    "person", "organization", "concept", "technology",
    "process", "metric", "event", "location", "product",
    "framework", "theory", "regulation", "market",
]


@dataclass
class ExtractionResult:
    """Result of entity extraction."""
    entities: list[dict] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)
    total_entities: int = 0
    total_relations: int = 0
    confidence: float = 0.7
    model_used: str = ""
    tokens_used: int = 0
    cost_usd: float = 0.0


@dataclass
class EntityClassification:
    """Classification of an entity mention."""
    text: str
    entity_type: str
    confidence: float = 0.7
    alternatives: list[str] = field(default_factory=list)


@dataclass
class ResolvedEntity:
    """An entity resolved against existing knowledge."""
    name: str
    entity_type: str
    is_new: bool = True
    existing_id: str = ""
    confidence: float = 0.7
    merge_suggested: bool = False


@dataclass
class ValidationResult:
    """Result of extraction validation."""
    valid: bool = True
    issues: list[str] = field(default_factory=list)
    filtered_entities: list[dict] = field(default_factory=list)
    filtered_relations: list[dict] = field(default_factory=list)


class EntityExtractor:
    """Extracts entities and relations from text using LLM.

    Handles extraction, classification, deduplication, and resolution
    against existing knowledge graph entities.
    """

    def __init__(self, router: ModelRouter | None = None, default_model: str = ""):
        self.router = router or ModelRouter()
        self._default_model = default_model or "gpt-4o-mini"

    def extract_from_text(
        self,
        text: str,
        context: str = "",
        max_entities: int = 30,
    ) -> ExtractionResult:
        """Extract entities and relations from text.

        Args:
            text: Text to extract from.
            context: Additional context (domain, task type, etc.).
            max_entities: Maximum entities to extract.

        Returns:
            ExtractionResult with entities and relations.
        """
        entity_types_str = ", ".join(ENTITY_TYPES)
        prompt = f"""Extract all significant entities and their relations from this text.

## Text
{text[:6000]}

{f"## Context{chr(10)}{context}" if context else ""}

## Instructions
1. Extract entities with:
   - name: The entity name
   - entity_type: One of [{entity_types_str}]
   - description: Brief description
   - confidence: 0.0-1.0

2. Extract relations between entities:
   - source_entity: Name of source entity
   - target_entity: Name of target entity
   - relation_type: Type of relation (e.g., "is_part_of", "created_by", "uses", "competes_with", "related_to")
   - confidence: 0.0-1.0

Maximum {max_entities} entities.

Respond as JSON:
{{
    "entities": [
        {{"name": "...", "entity_type": "...", "description": "...", "confidence": 0.8}}
    ],
    "relations": [
        {{"source_entity": "...", "target_entity": "...", "relation_type": "...", "confidence": 0.7}}
    ]
}}
"""
        try:
            request = LLMRequest(
                messages=[LLMMessage.user(prompt)],
                model=self._default_model,
                system_prompt="You are an expert entity extraction system. Be precise and thorough.",
                temperature=0.1,
                max_tokens=3000,
            )
            response = self.router.execute(request, TaskMetadata(
                task_type="extraction",
                complexity="moderate",
                estimated_tokens=3000,
            ))

            parsed = parse_llm_output(response.content, EntityExtractionOutput)
            if parsed:
                entities = [
                    {
                        "name": e.name,
                        "entity_type": e.entity_type,
                        "description": e.description,
                        "confidence": e.confidence,
                        "attributes": e.attributes,
                    }
                    for e in parsed.entities[:max_entities]
                ]
                relations = [
                    {
                        "source": r.source_entity,
                        "target": r.target_entity,
                        "type": r.relation_type,
                        "confidence": r.confidence,
                    }
                    for r in parsed.relations
                ]
            else:
                # Try raw JSON parse
                from llm.schemas import safe_json_loads
                data = safe_json_loads(response.content)

                entities = data.get("entities", [])[:max_entities]
                relations = data.get("relations", [])

            return ExtractionResult(
                entities=entities,
                relations=relations,
                total_entities=len(entities),
                total_relations=len(relations),
                model_used=response.model,
                tokens_used=response.total_tokens,
                cost_usd=response.cost_usd,
            )

        except Exception as e:
            logger.error("Entity extraction failed: %s", e)
            return ExtractionResult()

    def extract_from_task_result(self, task_result: dict) -> ExtractionResult:
        """Extract entities from a task result.

        Args:
            task_result: Task result dict with 'content' key.

        Returns:
            ExtractionResult.
        """
        content = task_result.get("content", "")
        if not content:
            return ExtractionResult()

        context = f"Task: {task_result.get('title', '')}"
        return self.extract_from_text(content, context)

    def classify_entity(self, entity_text: str, context: str = "") -> EntityClassification:
        """Classify a single entity mention.

        Args:
            entity_text: The entity text to classify.
            context: Surrounding context.

        Returns:
            EntityClassification.
        """
        # Heuristic classification based on common patterns
        text_lower = entity_text.lower()

        if any(w in text_lower for w in ["inc", "corp", "ltd", "company", "group"]):
            return EntityClassification(text=entity_text, entity_type="organization", confidence=0.9)

        if any(w in text_lower for w in ["python", "javascript", "api", "sdk", "framework", "library"]):
            return EntityClassification(text=entity_text, entity_type="technology", confidence=0.8)

        if any(w in text_lower for w in ["algorithm", "method", "approach", "pattern", "principle"]):
            return EntityClassification(text=entity_text, entity_type="concept", confidence=0.7)

        return EntityClassification(text=entity_text, entity_type="concept", confidence=0.5)

    def resolve_entity(
        self,
        name: str,
        entity_type: str,
        existing_entities: list[dict] | None = None,
    ) -> ResolvedEntity:
        """Resolve an entity against existing knowledge.

        Checks if the entity already exists (possibly under a different name)
        and suggests merging if appropriate.
        """
        if not existing_entities:
            return ResolvedEntity(name=name, entity_type=entity_type, is_new=True)

        name_lower = name.lower().strip()

        for existing in existing_entities:
            existing_name = existing.get("name", "").lower().strip()

            # Exact match
            if name_lower == existing_name:
                return ResolvedEntity(
                    name=name,
                    entity_type=entity_type,
                    is_new=False,
                    existing_id=existing.get("id", ""),
                    confidence=0.95,
                )

            # Substring match
            if name_lower in existing_name or existing_name in name_lower:
                return ResolvedEntity(
                    name=name,
                    entity_type=entity_type,
                    is_new=False,
                    existing_id=existing.get("id", ""),
                    confidence=0.7,
                    merge_suggested=True,
                )

        return ResolvedEntity(name=name, entity_type=entity_type, is_new=True)

    def batch_extract(self, texts: list[str], context: str = "") -> list[ExtractionResult]:
        """Extract entities from multiple texts."""
        return [self.extract_from_text(text, context) for text in texts]

    def validate_extraction(self, result: ExtractionResult) -> ValidationResult:
        """Validate extraction results — filter low-quality entities."""
        issues = []
        filtered_entities = []
        filtered_relations = []

        for entity in result.entities:
            confidence = entity.get("confidence", 0)
            name = entity.get("name", "")

            if not name or len(name) < 2:
                issues.append(f"Entity name too short: '{name}'")
                continue
            if confidence < 0.4:
                issues.append(f"Low confidence entity filtered: '{name}' ({confidence})")
                continue
            if entity.get("entity_type") not in ENTITY_TYPES:
                entity["entity_type"] = "concept"
                issues.append(f"Unknown entity type for '{name}', defaulting to 'concept'")

            filtered_entities.append(entity)

        # Validate relations reference existing entities
        entity_names = {e.get("name", "").lower() for e in filtered_entities}
        for relation in result.relations:
            source = relation.get("source", "").lower()
            target = relation.get("target", "").lower()
            if source in entity_names and target in entity_names:
                filtered_relations.append(relation)
            else:
                issues.append(f"Relation references unknown entity: {source} -> {target}")

        return ValidationResult(
            valid=len(filtered_entities) > 0,
            issues=issues,
            filtered_entities=filtered_entities,
            filtered_relations=filtered_relations,
        )

    def deduplicate_entities(self, entities: list[dict]) -> list[dict]:
        """Remove duplicate entities from a list."""
        seen = set()
        unique = []
        for entity in entities:
            name = entity.get("name", "").lower().strip()
            if name and name not in seen:
                seen.add(name)
                unique.append(entity)
        return unique
