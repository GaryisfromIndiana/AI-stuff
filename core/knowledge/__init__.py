"""Knowledge system — graph, schemas, quality, resolution, queries."""

from core.knowledge.bridge import KnowledgeBridge, SyncResult
from core.knowledge.entities import EntityExtractor, ExtractionResult
from core.knowledge.graph import GraphEdge, GraphNode, GraphStats, KnowledgeGraph, SubGraph
from core.knowledge.maintenance import KnowledgeGap, KnowledgeMaintainer, KnowledgeReport
from core.knowledge.quality import EntityQualityScore, EntityQualityScorer
from core.knowledge.query import KnowledgeAnswer, KnowledgeQuerier
from core.knowledge.resolution import EntityResolver, ResolutionResult
from core.knowledge.schemas import ENTITY_SCHEMAS, EntitySchema, get_schema, validate_entity

__all__ = [
    "ENTITY_SCHEMAS",
    "EntityExtractor",
    "EntityQualityScore",
    "EntityQualityScorer",
    "EntityResolver",
    "EntitySchema",
    "ExtractionResult",
    "GraphEdge",
    "GraphNode",
    "GraphStats",
    "KnowledgeAnswer",
    "KnowledgeBridge",
    "KnowledgeGap",
    "KnowledgeGraph",
    "KnowledgeMaintainer",
    "KnowledgeQuerier",
    "KnowledgeReport",
    "ResolutionResult",
    "SubGraph",
    "SyncResult",
    "get_schema",
    "validate_entity",
]
