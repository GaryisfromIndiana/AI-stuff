"""Empire generator — spawns new empires from scratch with one command."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EmpireConfig:
    """Configuration for generating a new empire."""
    name: str = ""
    domain: str = "general"
    description: str = ""
    lieutenants: list[dict] = field(default_factory=list)
    scheduler_config: dict = field(default_factory=dict)
    budget_config: dict = field(default_factory=dict)
    knowledge_domains: list[str] = field(default_factory=list)


@dataclass
class GeneratedEmpire:
    """A fully generated empire ready to run."""
    empire_id: str = ""
    config: EmpireConfig = field(default_factory=EmpireConfig)
    lieutenants_created: int = 0
    database_initialized: bool = False
    launch_ready: bool = False


@dataclass
class EmpireTemplate:
    """Pre-built empire template."""
    name: str
    domain: str
    description: str
    default_lieutenants: list[dict] = field(default_factory=list)
    default_knowledge_domains: list[str] = field(default_factory=list)


# ── Pre-built empire templates ─────────────────────────────────────────

EMPIRE_TEMPLATES: dict[str, EmpireTemplate] = {
    "finance": EmpireTemplate(
        name="Finance Empire",
        domain="finance",
        description="Financial analysis, modeling, and market research",
        default_lieutenants=[
            {"name": "Market Analyst", "template": "financial_modeler", "domain": "finance"},
            {"name": "Risk Assessor", "template": "strategy_advisor", "domain": "risk"},
            {"name": "Data Analyst", "template": "data_scientist", "domain": "data_science"},
            {"name": "Research Lead", "template": "research_analyst", "domain": "research"},
        ],
        default_knowledge_domains=["financial_markets", "economics", "accounting", "risk_management"],
    ),
    "tech": EmpireTemplate(
        name="Tech Empire",
        domain="technology",
        description="Software architecture, code review, technical research",
        default_lieutenants=[
            {"name": "Lead Architect", "template": "technical_architect", "domain": "technology"},
            {"name": "Security Lead", "template": "security_auditor", "domain": "security"},
            {"name": "Research Engineer", "template": "research_analyst", "domain": "research"},
            {"name": "DevOps Lead", "template": "operations_manager", "domain": "operations"},
        ],
        default_knowledge_domains=["software_architecture", "cloud", "security", "devops"],
    ),
    "research": EmpireTemplate(
        name="Research Empire",
        domain="research",
        description="Academic research, literature review, hypothesis testing",
        default_lieutenants=[
            {"name": "Principal Researcher", "template": "research_analyst", "domain": "research"},
            {"name": "Data Scientist", "template": "data_scientist", "domain": "data_science"},
            {"name": "Strategy Analyst", "template": "strategy_advisor", "domain": "strategy"},
            {"name": "Technical Writer", "template": "content_strategist", "domain": "content"},
        ],
        default_knowledge_domains=["research_methodology", "statistics", "literature_review"],
    ),
    "content": EmpireTemplate(
        name="Content Empire",
        domain="content",
        description="Content strategy, writing, editing, SEO",
        default_lieutenants=[
            {"name": "Content Lead", "template": "content_strategist", "domain": "content"},
            {"name": "SEO Analyst", "template": "data_scientist", "domain": "data_science"},
            {"name": "Research Writer", "template": "research_analyst", "domain": "research"},
            {"name": "Strategy Advisor", "template": "strategy_advisor", "domain": "strategy"},
        ],
        default_knowledge_domains=["content_marketing", "seo", "audience_analysis", "copywriting"],
    ),
    "operations": EmpireTemplate(
        name="Operations Empire",
        domain="operations",
        description="Process optimization, automation, efficiency",
        default_lieutenants=[
            {"name": "Operations Lead", "template": "operations_manager", "domain": "operations"},
            {"name": "Process Analyst", "template": "data_scientist", "domain": "data_science"},
            {"name": "Tech Lead", "template": "technical_architect", "domain": "technology"},
            {"name": "Strategy Advisor", "template": "strategy_advisor", "domain": "strategy"},
        ],
        default_knowledge_domains=["process_optimization", "automation", "lean_methodology", "efficiency"],
    ),
}


class EmpireGenerator:
    """Generates new empires from templates or custom configurations.

    One command creates a full empire with config, lieutenants,
    database, and everything needed to start autonomous operation.
    """

    def generate_empire(
        self,
        name: str,
        template: str = "",
        domain: str = "general",
        description: str = "",
        custom_lieutenants: list[dict] | None = None,
    ) -> GeneratedEmpire:
        """Generate a complete new empire.

        Args:
            name: Empire name.
            template: Template name (finance, tech, research, content, operations).
            domain: Domain if not using template.
            description: Empire description.
            custom_lieutenants: Custom lieutenant configs.

        Returns:
            GeneratedEmpire ready to run.
        """
        # Build config from template or custom
        if template and template in EMPIRE_TEMPLATES:
            tmpl = EMPIRE_TEMPLATES[template]
            config = EmpireConfig(
                name=name,
                domain=tmpl.domain,
                description=description or tmpl.description,
                lieutenants=custom_lieutenants or tmpl.default_lieutenants,
                knowledge_domains=tmpl.default_knowledge_domains,
            )
        else:
            config = EmpireConfig(
                name=name,
                domain=domain,
                description=description,
                lieutenants=custom_lieutenants or [],
            )

        # Create empire in database
        empire_id = self._create_empire_db(config)

        # Create lieutenants
        lt_count = self._create_lieutenants(empire_id, config.lieutenants)

        # Initialize database tables
        self._initialize_db()

        result = GeneratedEmpire(
            empire_id=empire_id,
            config=config,
            lieutenants_created=lt_count,
            database_initialized=True,
            launch_ready=True,
        )

        logger.info("Generated empire: %s (id=%s, %d lieutenants)", name, empire_id, lt_count)
        return result

    def clone_empire(
        self,
        source_empire_id: str,
        new_name: str,
    ) -> GeneratedEmpire:
        """Clone an existing empire.

        Args:
            source_empire_id: Source empire to clone.
            new_name: Name for the new empire.

        Returns:
            Cloned empire.
        """
        try:
            from db.engine import get_session
            from db.repositories.empire import EmpireRepository
            from db.repositories.lieutenant import LieutenantRepository

            session = get_session()
            empire_repo = EmpireRepository(session)
            lt_repo = LieutenantRepository(session)

            source = empire_repo.get(source_empire_id)
            if not source:
                raise ValueError(f"Source empire not found: {source_empire_id}")

            # Get lieutenant configs
            source_lts = lt_repo.get_by_empire(source_empire_id)
            lt_configs = [
                {
                    "name": lt.name,
                    "domain": lt.domain,
                    "persona": lt.persona_json,
                }
                for lt in source_lts
            ]

            return self.generate_empire(
                name=new_name,
                domain=source.domain,
                description=f"Cloned from {source.name}",
                custom_lieutenants=lt_configs,
            )

        except Exception as e:
            logger.error("Failed to clone empire: %s", e)
            raise

    def get_templates(self) -> list[dict]:
        """Get available empire templates."""
        return [
            {
                "key": key,
                "name": tmpl.name,
                "domain": tmpl.domain,
                "description": tmpl.description,
                "lieutenant_count": len(tmpl.default_lieutenants),
                "knowledge_domains": tmpl.default_knowledge_domains,
            }
            for key, tmpl in EMPIRE_TEMPLATES.items()
        ]

    def _create_empire_db(self, config: EmpireConfig) -> str:
        """Create the empire record in the database."""
        try:
            from db.engine import session_scope
            from db.models import Empire

            with session_scope() as session:
                empire = Empire(
                    name=config.name,
                    domain=config.domain,
                    description=config.description,
                    config_json={
                        "knowledge_domains": config.knowledge_domains,
                        "scheduler": config.scheduler_config,
                        "budget": config.budget_config,
                    },
                )
                session.add(empire)
                session.flush()
                return empire.id

        except Exception as e:
            logger.error("Failed to create empire DB record: %s", e)
            raise

    def _create_lieutenants(self, empire_id: str, lieutenant_configs: list[dict]) -> int:
        """Create lieutenants for the empire."""
        from core.lieutenant.manager import LieutenantManager

        manager = LieutenantManager(empire_id)
        count = 0

        for lt_config in lieutenant_configs:
            try:
                manager.create_lieutenant(
                    name=lt_config.get("name", f"Lieutenant {count + 1}"),
                    template=lt_config.get("template", ""),
                    domain=lt_config.get("domain", "general"),
                )
                count += 1
            except Exception as e:
                logger.warning("Failed to create lieutenant %s: %s", lt_config.get("name"), e)

        return count

    def _initialize_db(self) -> None:
        """Ensure database tables exist."""
        try:
            from db.engine import init_db
            init_db()
        except Exception as e:
            logger.warning("DB initialization: %s", e)
