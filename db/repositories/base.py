"""Generic base repository with CRUD, pagination, filtering, and bulk operations."""

from __future__ import annotations

import logging
from typing import Any, Generic, TypeVar

from sqlalchemy import and_, asc, delete, desc, func, select, update
from sqlalchemy.orm import Session

from db.models import Base

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Base)


class BaseRepository(Generic[T]):
    """Generic repository providing common CRUD and query operations.

    Subclass this with a specific model type:
        class LieutenantRepository(BaseRepository[Lieutenant]):
            model_class = Lieutenant
    """

    model_class: type[T]

    def __init__(self, session: Session):
        self.session = session

    # ── Create ─────────────────────────────────────────────────────────

    def create(self, **kwargs: Any) -> T:
        """Create a new entity.

        Args:
            **kwargs: Field values for the new entity.

        Returns:
            The created entity.
        """
        instance = self.model_class(**kwargs)
        self.session.add(instance)
        self.session.flush()
        return instance

    def create_many(self, items: list[dict[str, Any]]) -> list[T]:
        """Create multiple entities in bulk.

        Args:
            items: List of field value dicts.

        Returns:
            List of created entities.
        """
        instances = [self.model_class(**item) for item in items]
        self.session.add_all(instances)
        self.session.flush()
        return instances

    # ── Read ───────────────────────────────────────────────────────────

    def get(self, entity_id: str) -> T | None:
        """Get entity by primary key.

        Args:
            entity_id: Primary key value.

        Returns:
            Entity or None.
        """
        return self.session.get(self.model_class, entity_id)

    def get_or_raise(self, entity_id: str) -> T:
        """Get entity by ID or raise ValueError.

        Args:
            entity_id: Primary key value.

        Returns:
            Entity.

        Raises:
            ValueError: If entity not found.
        """
        entity = self.get(entity_id)
        if entity is None:
            raise ValueError(f"{self.model_class.__name__} with id {entity_id!r} not found")
        return entity

    def get_many(self, entity_ids: list[str]) -> list[T]:
        """Get multiple entities by IDs.

        Args:
            entity_ids: List of primary key values.

        Returns:
            List of found entities (may be shorter than input).
        """
        if not entity_ids:
            return []
        stmt = select(self.model_class).where(self.model_class.id.in_(entity_ids))
        return list(self.session.execute(stmt).scalars().all())

    def list_all(
        self,
        order_by: str = "created_at",
        order_dir: str = "desc",
        limit: int = 100,
        offset: int = 0,
    ) -> list[T]:
        """List all entities with ordering and pagination.

        Args:
            order_by: Column name to order by.
            order_dir: 'asc' or 'desc'.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            List of entities.
        """
        stmt = select(self.model_class)
        stmt = self._apply_ordering(stmt, order_by, order_dir)
        stmt = stmt.limit(limit).offset(offset)
        return list(self.session.execute(stmt).scalars().all())

    def find(
        self,
        filters: dict[str, Any] | None = None,
        order_by: str = "created_at",
        order_dir: str = "desc",
        limit: int = 100,
        offset: int = 0,
    ) -> list[T]:
        """Find entities matching filter criteria.

        Args:
            filters: Dict of {column_name: value} for exact matches.
            order_by: Column to order by.
            order_dir: 'asc' or 'desc'.
            limit: Maximum results.
            offset: Results to skip.

        Returns:
            List of matching entities.
        """
        stmt = select(self.model_class)
        if filters:
            conditions = []
            for key, value in filters.items():
                if hasattr(self.model_class, key):
                    col = getattr(self.model_class, key)
                    if isinstance(value, list):
                        conditions.append(col.in_(value))
                    elif value is None:
                        conditions.append(col.is_(None))
                    else:
                        conditions.append(col == value)
            if conditions:
                stmt = stmt.where(and_(*conditions))
        stmt = self._apply_ordering(stmt, order_by, order_dir)
        stmt = stmt.limit(limit).offset(offset)
        return list(self.session.execute(stmt).scalars().all())

    def find_one(self, filters: dict[str, Any]) -> T | None:
        """Find a single entity matching filters.

        Args:
            filters: Dict of {column_name: value}.

        Returns:
            Entity or None.
        """
        results = self.find(filters=filters, limit=1)
        return results[0] if results else None

    def exists(self, entity_id: str) -> bool:
        """Check if an entity exists.

        Args:
            entity_id: Primary key value.

        Returns:
            True if entity exists.
        """
        stmt = select(func.count()).select_from(self.model_class).where(self.model_class.id == entity_id)
        return self.session.execute(stmt).scalar() > 0

    def count(self, filters: dict[str, Any] | None = None) -> int:
        """Count entities matching optional filters.

        Args:
            filters: Optional filter criteria.

        Returns:
            Count of matching entities.
        """
        stmt = select(func.count()).select_from(self.model_class)
        if filters:
            for key, value in filters.items():
                if hasattr(self.model_class, key):
                    col = getattr(self.model_class, key)
                    if isinstance(value, list):
                        stmt = stmt.where(col.in_(value))
                    else:
                        stmt = stmt.where(col == value)
        return self.session.execute(stmt).scalar() or 0

    # ── Update ─────────────────────────────────────────────────────────

    def update(self, entity_id: str, **kwargs: Any) -> T | None:
        """Update an entity by ID.

        Args:
            entity_id: Primary key value.
            **kwargs: Fields to update.

        Returns:
            Updated entity or None if not found.
        """
        entity = self.get(entity_id)
        if entity is None:
            return None
        for key, value in kwargs.items():
            if hasattr(entity, key):
                setattr(entity, key, value)
        self.session.flush()
        return entity

    def update_many(self, entity_ids: list[str], **kwargs: Any) -> int:
        """Update multiple entities by IDs.

        Args:
            entity_ids: List of primary key values.
            **kwargs: Fields to update.

        Returns:
            Number of updated entities.
        """
        if not entity_ids:
            return 0
        stmt = (
            update(self.model_class)
            .where(self.model_class.id.in_(entity_ids))
            .values(**kwargs)
        )
        result = self.session.execute(stmt)
        self.session.flush()
        return result.rowcount

    def update_where(self, filters: dict[str, Any], **kwargs: Any) -> int:
        """Update entities matching filters.

        Args:
            filters: Filter criteria.
            **kwargs: Fields to update.

        Returns:
            Number of updated entities.
        """
        stmt = update(self.model_class)
        for key, value in filters.items():
            if hasattr(self.model_class, key):
                stmt = stmt.where(getattr(self.model_class, key) == value)
        stmt = stmt.values(**kwargs)
        result = self.session.execute(stmt)
        self.session.flush()
        return result.rowcount

    # ── Delete ─────────────────────────────────────────────────────────

    def delete(self, entity_id: str) -> bool:
        """Delete an entity by ID.

        Args:
            entity_id: Primary key value.

        Returns:
            True if entity was deleted.
        """
        entity = self.get(entity_id)
        if entity is None:
            return False
        self.session.delete(entity)
        self.session.flush()
        return True

    def delete_many(self, entity_ids: list[str]) -> int:
        """Delete multiple entities by IDs.

        Args:
            entity_ids: List of primary key values.

        Returns:
            Number of deleted entities.
        """
        if not entity_ids:
            return 0
        stmt = delete(self.model_class).where(self.model_class.id.in_(entity_ids))
        result = self.session.execute(stmt)
        self.session.flush()
        return result.rowcount

    def delete_where(self, filters: dict[str, Any]) -> int:
        """Delete entities matching filters.

        Args:
            filters: Filter criteria.

        Returns:
            Number of deleted entities.
        """
        stmt = delete(self.model_class)
        for key, value in filters.items():
            if hasattr(self.model_class, key):
                stmt = stmt.where(getattr(self.model_class, key) == value)
        result = self.session.execute(stmt)
        self.session.flush()
        return result.rowcount

    # ── Pagination ─────────────────────────────────────────────────────

    def paginate(
        self,
        page: int = 1,
        per_page: int = 20,
        filters: dict[str, Any] | None = None,
        order_by: str = "created_at",
        order_dir: str = "desc",
    ) -> PaginatedResult[T]:
        """Get paginated results.

        Args:
            page: Page number (1-based).
            per_page: Items per page.
            filters: Optional filter criteria.
            order_by: Column to order by.
            order_dir: 'asc' or 'desc'.

        Returns:
            PaginatedResult with items and metadata.
        """
        total = self.count(filters)
        offset = (page - 1) * per_page
        items = self.find(
            filters=filters,
            order_by=order_by,
            order_dir=order_dir,
            limit=per_page,
            offset=offset,
        )

        return PaginatedResult(
            items=items,
            total=total,
            page=page,
            per_page=per_page,
            total_pages=(total + per_page - 1) // per_page if per_page > 0 else 0,
        )

    # ── Aggregation ────────────────────────────────────────────────────

    def sum_column(self, column_name: str, filters: dict[str, Any] | None = None) -> float:
        """Sum a numeric column.

        Args:
            column_name: Column to sum.
            filters: Optional filter criteria.

        Returns:
            Sum value.
        """
        col = getattr(self.model_class, column_name)
        stmt = select(func.coalesce(func.sum(col), 0.0))
        if filters:
            for key, value in filters.items():
                if hasattr(self.model_class, key):
                    stmt = stmt.where(getattr(self.model_class, key) == value)
        return float(self.session.execute(stmt).scalar() or 0.0)

    def avg_column(self, column_name: str, filters: dict[str, Any] | None = None) -> float:
        """Average a numeric column.

        Args:
            column_name: Column to average.
            filters: Optional filter criteria.

        Returns:
            Average value.
        """
        col = getattr(self.model_class, column_name)
        stmt = select(func.coalesce(func.avg(col), 0.0))
        if filters:
            for key, value in filters.items():
                if hasattr(self.model_class, key):
                    stmt = stmt.where(getattr(self.model_class, key) == value)
        return float(self.session.execute(stmt).scalar() or 0.0)

    def max_column(self, column_name: str, filters: dict[str, Any] | None = None) -> Any:
        """Get maximum value of a column.

        Args:
            column_name: Column name.
            filters: Optional filter criteria.

        Returns:
            Maximum value or None.
        """
        col = getattr(self.model_class, column_name)
        stmt = select(func.max(col))
        if filters:
            for key, value in filters.items():
                if hasattr(self.model_class, key):
                    stmt = stmt.where(getattr(self.model_class, key) == value)
        return self.session.execute(stmt).scalar()

    def min_column(self, column_name: str, filters: dict[str, Any] | None = None) -> Any:
        """Get minimum value of a column.

        Args:
            column_name: Column name.
            filters: Optional filter criteria.

        Returns:
            Minimum value or None.
        """
        col = getattr(self.model_class, column_name)
        stmt = select(func.min(col))
        if filters:
            for key, value in filters.items():
                if hasattr(self.model_class, key):
                    stmt = stmt.where(getattr(self.model_class, key) == value)
        return self.session.execute(stmt).scalar()

    def distinct_values(self, column_name: str, filters: dict[str, Any] | None = None) -> list[Any]:
        """Get distinct values of a column.

        Args:
            column_name: Column name.
            filters: Optional filter criteria.

        Returns:
            List of distinct values.
        """
        col = getattr(self.model_class, column_name)
        stmt = select(col).distinct()
        if filters:
            for key, value in filters.items():
                if hasattr(self.model_class, key):
                    stmt = stmt.where(getattr(self.model_class, key) == value)
        return [row[0] for row in self.session.execute(stmt).all()]

    # ── Helpers ────────────────────────────────────────────────────────

    def _apply_ordering(self, stmt, order_by: str, order_dir: str = "desc"):
        """Apply ordering to a select statement."""
        if hasattr(self.model_class, order_by):
            col = getattr(self.model_class, order_by)
            if order_dir.lower() == "asc":
                stmt = stmt.order_by(asc(col))
            else:
                stmt = stmt.order_by(desc(col))
        return stmt

    def refresh(self, entity: T) -> T:
        """Refresh entity from database.

        Args:
            entity: Entity to refresh.

        Returns:
            Refreshed entity.
        """
        self.session.refresh(entity)
        return entity

    def flush(self) -> None:
        """Flush pending changes to the database."""
        self.session.flush()

    def commit(self) -> None:
        """Commit the current transaction."""
        self.session.commit()

    def rollback(self) -> None:
        """Rollback the current transaction."""
        self.session.rollback()


class PaginatedResult(Generic[T]):
    """Container for paginated query results."""

    def __init__(
        self,
        items: list[T],
        total: int,
        page: int,
        per_page: int,
        total_pages: int,
    ):
        self.items = items
        self.total = total
        self.page = page
        self.per_page = per_page
        self.total_pages = total_pages

    @property
    def has_next(self) -> bool:
        return self.page < self.total_pages

    @property
    def has_prev(self) -> bool:
        return self.page > 1

    @property
    def next_page(self) -> int | None:
        return self.page + 1 if self.has_next else None

    @property
    def prev_page(self) -> int | None:
        return self.page - 1 if self.has_prev else None

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "page": self.page,
            "per_page": self.per_page,
            "total_pages": self.total_pages,
            "has_next": self.has_next,
            "has_prev": self.has_prev,
            "item_count": len(self.items),
        }

    def __repr__(self) -> str:
        return f"<PaginatedResult(page={self.page}/{self.total_pages}, items={len(self.items)}, total={self.total})>"
