"""Embedding generation for memory and knowledge graph entries.

Wraps OpenAI embeddings with error handling, truncation, and caching.
Used by MemoryManager.store(), BiTemporalMemory.store_fact(), and
the backfill scheduler job.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# text-embedding-3-small max input is 8191 tokens (~32k chars).
# Truncate to be safe.
_MAX_CHARS = 24_000


def generate_embedding(text: str) -> Optional[list[float]]:
    """Generate an embedding vector for text.

    Returns None on any failure (missing API key, rate limit, etc.)
    so callers can store the memory without an embedding and backfill later.
    """
    if not text or not text.strip():
        return None

    try:
        from llm.openai import OpenAIClient
        client = OpenAIClient()
        truncated = text[:_MAX_CHARS]
        return client.create_embedding(truncated)
    except Exception as e:
        logger.debug("Embedding generation failed (will backfill later): %s", e)
        return None


def generate_embeddings_batch(texts: list[str]) -> list[Optional[list[float]]]:
    """Generate embeddings for multiple texts.

    Returns a list parallel to input — None for any that failed.
    Processes in chunks of 100 (OpenAI batch limit).
    """
    if not texts:
        return []

    try:
        from llm.openai import OpenAIClient
        client = OpenAIClient()
    except Exception as e:
        logger.debug("OpenAI client init failed: %s", e)
        return [None] * len(texts)

    results: list[Optional[list[float]]] = [None] * len(texts)
    chunk_size = 100

    for start in range(0, len(texts), chunk_size):
        chunk = texts[start:start + chunk_size]
        truncated = [t[:_MAX_CHARS] if t else "" for t in chunk]

        # Skip empty strings
        non_empty_indices = [i for i, t in enumerate(truncated) if t.strip()]
        if not non_empty_indices:
            continue

        non_empty_texts = [truncated[i] for i in non_empty_indices]

        try:
            embeddings = client.create_embeddings_batch(non_empty_texts)
            for idx, emb in zip(non_empty_indices, embeddings):
                results[start + idx] = emb
        except Exception as e:
            logger.warning("Batch embedding failed for chunk %d: %s", start, e)

    return results


def backfill_embeddings(empire_id: str, batch_size: int = 50) -> dict:
    """Backfill embeddings for memories and KG entities that don't have them.

    Processes a batch per call to avoid hammering the embedding API.
    Called by the scheduler's embedding_backfill job.
    """
    from db.engine import session_scope
    from db.models import MemoryEntry, KnowledgeEntity
    from sqlalchemy import select, and_, or_, cast, String

    total_filled = 0
    kg_filled = 0

    # 1. Backfill memory entries
    try:
        with session_scope() as session:
            stmt = (
                select(MemoryEntry)
                .where(and_(
                    MemoryEntry.empire_id == empire_id,
                    or_(
                        MemoryEntry.embedding_json.is_(None),
                        cast(MemoryEntry.embedding_json, String) == "null",
                        cast(MemoryEntry.embedding_json, String) == "",
                    ),
                    MemoryEntry.memory_type.in_(["semantic", "experiential", "design"]),
                ))
                .order_by(MemoryEntry.effective_importance.desc())
                .limit(batch_size)
            )
            entries = list(session.execute(stmt).scalars().all())

            if entries:
                texts = [f"{e.title}\n{e.content}" if e.title else e.content for e in entries]
                embeddings = generate_embeddings_batch(texts)

                qdrant_batch = []
                for entry, emb in zip(entries, embeddings):
                    if emb:
                        entry.embedding_json = emb
                        total_filled += 1
                        qdrant_batch.append({
                            "memory_id": entry.id,
                            "embedding": emb,
                            "empire_id": entry.empire_id,
                            "lieutenant_id": entry.lieutenant_id or "",
                            "memory_type": entry.memory_type,
                            "importance": entry.importance_score or 0.5,
                            "decay_factor": entry.decay_factor or 1.0,
                        })

                if qdrant_batch:
                    try:
                        from core.vector.store import VectorStore
                        VectorStore.get_instance(empire_id).upsert_memories_batch(qdrant_batch)
                    except Exception:
                        pass

                logger.info("Backfilled %d/%d memory embeddings", total_filled, len(entries))
    except Exception as e:
        logger.warning("Memory embedding backfill failed: %s", e)

    # 2. Backfill KG entities
    try:
        with session_scope() as session:
            stmt = (
                select(KnowledgeEntity)
                .where(and_(
                    KnowledgeEntity.empire_id == empire_id,
                    or_(
                        KnowledgeEntity.embedding_json.is_(None),
                        cast(KnowledgeEntity.embedding_json, String) == "null",
                        cast(KnowledgeEntity.embedding_json, String) == "",
                    ),
                ))
                .order_by(KnowledgeEntity.importance_score.desc())
                .limit(batch_size)
            )
            entities = list(session.execute(stmt).scalars().all())

            if entities:
                texts = [f"{e.name}: {e.description}" if e.description else e.name for e in entities]
                embeddings = generate_embeddings_batch(texts)

                qdrant_batch = []
                for entity, emb in zip(entities, embeddings):
                    if emb:
                        entity.embedding_json = emb
                        kg_filled += 1
                        qdrant_batch.append({
                            "entity_id": entity.id,
                            "embedding": emb,
                            "empire_id": entity.empire_id,
                            "entity_type": entity.entity_type or "",
                            "name": entity.name or "",
                            "importance": entity.importance_score or 0.5,
                        })

                if qdrant_batch:
                    try:
                        from core.vector.store import VectorStore
                        VectorStore.get_instance(empire_id).upsert_entities_batch(qdrant_batch)
                    except Exception:
                        pass

                logger.info("Backfilled %d/%d KG entity embeddings", kg_filled, len(entities))
    except Exception as e:
        logger.warning("KG embedding backfill failed: %s", e)

    return {
        "memories_backfilled": total_filled,
        "kg_entities_backfilled": kg_filled,
    }
