# Empire AI

Self-upgrading multi-agent AI system for **autonomous AI research**.

## Purpose
Empire runs autonomous research on the latest AI developments — model releases, papers, techniques, tooling, agent architectures, and industry moves. Lieutenants research independently, debate in War Rooms, and compound knowledge over time.

## Architecture
- **ACE (Autonomous Cognitive Engine)** — 3-agent pipeline (planner → executor → critic) powering every lieutenant
- **4-tier memory** — semantic, experiential, design, episodic
- **Knowledge graph** — entities extracted from every task, compounding over time
- **War Rooms** — multi-lieutenant debate, planning, synthesis
- **Evolution** — propose → review → implement → learn → repeat
- **Ralph Wiggum retry** — error injection, model escalation, sibling context
- **Scheduler daemon** — ticks every 60s, drives learning/evolution/health autonomously
- **Cross-empire registry + knowledge bridge** — insights flow across the network

## Lieutenants
| Name | Domain | Focus |
|---|---|---|
| Model Intelligence | models | LLM releases, benchmarks, pricing, capabilities |
| Research Scout | research | Papers, training techniques, alignment, scaling laws |
| Agent Systems | agents | Multi-agent, tool use, frameworks, MCP |
| Tooling & Infra | tooling | APIs, inference, vector DBs, deployment |
| Industry & Strategy | industry | Company strategy, funding, enterprise adoption |
| Open Source | open_source | Open weight models, HuggingFace, local inference |

## Running
```bash
cd empire && source .venv/bin/activate
python -m web.app          # Web UI on http://localhost:5000
python -m cli.commands serve   # Same via CLI
python -m cli.commands scheduler start  # Start autonomous daemon
```

## Stack
Python 3.12 · SQLite · Flask · SQLAlchemy · Pydantic · Claude · APScheduler

## Architecture & boundaries
See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full module contract. Short version:

```
web/ · cli/   →   core/   →   llm/   →   db/   →   utils/ · config/
```

Arrows go down only. Imports upward are rejected by `lint-imports` in CI. Callers import from each package's public `__init__.py` (enforced by ruff `flake8-tidy-imports`), not from deep paths.

**Before landing a change**: `ruff check . && ruff format --check . && lint-imports && mypy config utils llm/schemas.py`
