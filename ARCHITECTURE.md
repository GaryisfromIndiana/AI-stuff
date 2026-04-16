# Empire Architecture

This document is the **source of truth** for Empire's module structure and dependency rules. Changes that violate the rules here are blocked in CI by `lint-imports`, `ruff`, and `mypy`. If you need to change the rules, update this file *and* the enforcement configs (`.importlinter`, `pyproject.toml`) in the same PR.

## Layer diagram

```
┌───────────────────────────────────────────────────────────┐
│  web/  (Flask API edge)     cli/  (command-line edge)     │  ← Entry points
├───────────────────────────────────────────────────────────┤
│                         core/                             │  ← Orchestration
│   ace · lieutenant · warroom · memory · knowledge ·       │    (13 submodules)
│   evolution · scheduler · routing · retry · search ·      │
│   replication · validation · mcp · directives             │
├───────────────────────────────────────────────────────────┤
│                         llm/                              │  ← Model interface
│            router · clients · caching · schemas           │
├───────────────────────────────────────────────────────────┤
│                          db/                              │  ← Persistence
│              engine · models · repositories               │
├───────────────────────────────────────────────────────────┤
│                         utils/                            │  ← Helpers
├───────────────────────────────────────────────────────────┤
│                         config/                           │  ← Leaf
└───────────────────────────────────────────────────────────┘
```

Arrows go **down only**. A module may import from any layer below it. Nothing imports upward.

## Module contract

| Module    | Purpose                                        | May import from          | Public API entry |
|-----------|------------------------------------------------|--------------------------|------------------|
| `web/`    | Flask routes, middleware, rate limiting        | core, llm, db, utils, config | `web.app`    |
| `cli/`    | Command-line interface                         | core, llm, db, utils, config | `cli` (main) |
| `core/`   | ACE pipeline, lieutenants, war rooms, memory, knowledge, evolution, scheduler | llm, db, utils, config | `core.<submodule>` |
| `llm/`    | Model router, clients, Pydantic schemas        | utils, config            | `llm`            |
| `db/`     | SQLAlchemy models, engine, repositories        | utils, config            | `db`             |
| `utils/`  | Logging, text, formatting helpers              | config                   | `utils`          |
| `config/` | Settings singleton                             | (nothing — leaf)         | `config`         |

## The rules

1. **`web/` and `cli/` are edges.** They own HTTP/CLI concerns. No other module imports from them.
2. **`core/` owns orchestration.** It coordinates lieutenants, pipelines, and state. It never imports `web/` or `cli/`.
3. **`llm/` is the model interface.** All model calls go through it. It exposes Pydantic schemas for structured outputs (`llm/schemas.py`).
4. **`db/` is persistence.** SQLAlchemy models live here. Callers use the `engine` and `repositories` entry points — not raw `session` in most cases.
5. **`utils/` and `config/` are leaves.** They import nothing from the rest of the project. Keep them small.

## Public API discipline

Every top-level package and every `core/*` submodule declares `__all__` in its `__init__.py`. **Callers import from the package root, not from deep paths.** This makes internal refactors safe.

**Good:**
```python
from core.ace import ACEEngine
from llm import LLMClient, parse_llm_output
```

**Bad** (blocked by ruff's `flake8-tidy-imports`):
```python
from core.ace.engine import ACEEngine          # deep import
from llm.schemas import CriticOutput           # should go through llm
```

## Documented exceptions

**Legitimate** (will stay):

- **`core/lieutenant/manager.py` → `db.engine.repo_scope`** — Lieutenant manager scopes its own transactions. Acceptable because it's the one place that owns agent lifecycle writes.
- **`cli/commands.py` → `web.app`** — `cli serve` launches the web server. `cli` and `web` are sibling edges of the same system.

**Known debt**: None outstanding. The three historical upward-import violations (utils→llm, llm→core, db→core) have been inverted — JSON parsing lives in `utils.text`, LLM spend recording uses `register_spend_recorder_factory`, and memory vector search goes through `core.memory.manager` → `core.vector.store` directly (the unused `MemoryRepository.similarity_search` was deleted).

If you find yourself wanting to add a new exception, first ask whether the dependency should instead be expressed through a new abstraction in the lower layer.

## Adding a new module

1. Decide which layer it belongs to. If it's not obvious, the module is probably doing too much.
2. Add a row to the module table above.
3. Add the package to `pyproject.toml`'s `[tool.setuptools.packages.find]`.
4. Define `__all__` in its `__init__.py`.
5. Add an entry to `.importlinter` if you need new boundary rules.
6. Add mypy strictness in `pyproject.toml` under `[[tool.mypy.overrides]]` once the module is stable.

## Enforcement

| Check                  | Tool            | Where                |
|------------------------|-----------------|----------------------|
| Module boundaries      | `import-linter` | `.importlinter`      |
| Deep-import bans       | `ruff` (TID)    | `pyproject.toml`     |
| Lint + format          | `ruff`          | `pyproject.toml`     |
| Type safety            | `mypy`          | `pyproject.toml`     |
| All of the above       | CI              | `.github/workflows/` |
| Pre-commit (optional)  | `pre-commit`    | `.pre-commit-config.yaml` |

Run everything locally before pushing:

```bash
ruff check .
ruff format --check .
lint-imports
mypy core llm db utils config
```
