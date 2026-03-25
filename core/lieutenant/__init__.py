"""Lieutenant system — specialized AI agents powered by ACE."""

from core.lieutenant.base import Lieutenant, PerformanceStats, DebateContribution
from core.lieutenant.manager import LieutenantManager, FleetStats
from core.lieutenant.persona import (
    PersonaConfig, PersonaBuilder, create_persona,
    list_persona_templates, PERSONA_TEMPLATES,
)
from core.lieutenant.registry import LieutenantRegistry, RegistryEntry

__all__ = [
    "Lieutenant", "PerformanceStats", "DebateContribution",
    "LieutenantManager", "FleetStats",
    "PersonaConfig", "PersonaBuilder", "create_persona",
    "list_persona_templates", "PERSONA_TEMPLATES",
    "LieutenantRegistry", "RegistryEntry",
]
