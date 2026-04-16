"""Lieutenant system — specialized AI agents powered by ACE."""

from core.lieutenant.base import DebateContribution, Lieutenant, PerformanceStats
from core.lieutenant.manager import FleetStats, LieutenantManager
from core.lieutenant.persona import (
    PERSONA_TEMPLATES,
    PersonaBuilder,
    PersonaConfig,
    create_persona,
    list_persona_templates,
)
from core.lieutenant.registry import LieutenantRegistry, RegistryEntry

__all__ = [
    "PERSONA_TEMPLATES",
    "DebateContribution",
    "FleetStats",
    "Lieutenant",
    "LieutenantManager",
    "LieutenantRegistry",
    "PerformanceStats",
    "PersonaBuilder",
    "PersonaConfig",
    "RegistryEntry",
    "create_persona",
    "list_persona_templates",
]
