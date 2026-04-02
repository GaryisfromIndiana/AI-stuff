"""Unit tests for the consolidated task routing table."""

from __future__ import annotations

from llm.router import TASK_ROUTING, COMPLEXITY_TIERS, TaskRouting


def test_all_entries_are_task_routing_instances():
    """Every entry in TASK_ROUTING should be a TaskRouting dataclass."""
    for name, routing in TASK_ROUTING.items():
        assert isinstance(routing, TaskRouting), f"{name} is {type(routing)}, not TaskRouting"


def test_all_entries_have_required_fields():
    """Every routing entry must have model, tier, capabilities, and reason."""
    for name, routing in TASK_ROUTING.items():
        assert routing.model, f"{name} has no model"
        assert routing.tier in (1, 2, 3, 4), f"{name} has invalid tier: {routing.tier}"
        assert isinstance(routing.capabilities, list), f"{name} capabilities not a list"
        assert len(routing.capabilities) > 0, f"{name} has no capabilities"
        assert routing.reason, f"{name} has no reason"


def test_tier_distribution():
    """Verify we have jobs at multiple tiers (not all on one model)."""
    tiers = {r.tier for r in TASK_ROUTING.values()}
    assert len(tiers) >= 3, f"Only {len(tiers)} tiers used: {tiers}"


def test_haiku_tier_tasks_are_cheap_operations():
    """Tier 3 (haiku) tasks should be simple, cheap operations."""
    haiku_tasks = {name for name, r in TASK_ROUTING.items() if r.tier == 3}
    # These should all be lightweight — no heavy reasoning tasks
    assert "classification" in haiku_tasks
    assert "extraction" in haiku_tasks
    # And none of the heavy ones
    assert "synthesis" not in haiku_tasks
    assert "evolution" not in haiku_tasks


def test_opus_tier_tasks_are_complex():
    """Tier 1 (opus) tasks should be heavy reasoning tasks."""
    opus_tasks = {name for name, r in TASK_ROUTING.items() if r.tier == 1}
    assert "synthesis" in opus_tasks
    assert "evolution" in opus_tasks
    assert "planning" in opus_tasks


def test_complexity_tiers_cover_all_levels():
    """COMPLEXITY_TIERS should map all 4 complexity levels."""
    assert set(COMPLEXITY_TIERS.keys()) == {"simple", "moderate", "complex", "expert"}
    assert COMPLEXITY_TIERS["simple"] > COMPLEXITY_TIERS["expert"]


def test_models_reference_known_catalog_keys():
    """All models referenced in TASK_ROUTING should exist in MODEL_CATALOG."""
    from config.settings import MODEL_CATALOG
    for name, routing in TASK_ROUTING.items():
        assert routing.model in MODEL_CATALOG, (
            f"Task '{name}' references model '{routing.model}' not in MODEL_CATALOG"
        )


def test_no_duplicate_task_types():
    """Task type names should be unique (enforced by dict, but verify count)."""
    assert len(TASK_ROUTING) == len(set(TASK_ROUTING.keys()))
