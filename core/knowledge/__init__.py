"""Knowledge system — graph, entity extraction, bridge, maintenance."""

from core.knowledge.graph import KnowledgeGraph, GraphNode, GraphEdge, SubGraph, GraphStats
from core.knowledge.entities import EntityExtractor, ExtractionResult
from core.knowledge.bridge import KnowledgeBridge, SyncResult
from core.knowledge.maintenance import KnowledgeMaintainer, KnowledgeReport, KnowledgeGap

__all__ = [
    "KnowledgeGraph", "GraphNode", "GraphEdge", "SubGraph", "GraphStats",
    "EntityExtractor", "ExtractionResult",
    "KnowledgeBridge", "SyncResult",
    "KnowledgeMaintainer", "KnowledgeReport", "KnowledgeGap",
]
