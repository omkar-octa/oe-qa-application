---
name: python-project-structure
description: Scaffolds and enforces this repo's standard Python project layout - src/ with utils/models/prompts/scripts/tests, pydantic-settings config, pytest-based tests, docs/, and .gitignore conventions. Tagged python, scaffold, project-structure.
when_to_use: Use when starting a new Python project or module in this repo, when adding a new top-level folder (utils, models, prompts, scripts, tests), when creating config.py or other pydantic models, when adding tests, or when the user asks whether something follows the project's conventions.
---

# Python project structure

This repo's Python projects follow a fixed layout. Use this skill to scaffold new
projects and to check that new files land in the right place.

## Layout

```
src/
  main.py

  utils/
    __init__.py
    util1.py

  models/
    __init__.py
    request.py
    response.py
    config.py

  prompts/
    1.prompt
    2.prompt

  scripts/
    script1.ipynb

  tests/
    unit/
      test1.py

    integration/
      test2.py

    fixtures/
      fixture1.py
      fixture1.json

    pytest.ini

docs/
  features.md
  metadata.md

README.md

.gitignore
```

## Rules and guidelines

- **Configuration:** `config.py` uses `pydantic-settings`, linked to a local `.env`
  file. Environment variables are loaded from `.env` into the application at runtime.
- **Data models:** every other model uses `pydantic.BaseModel` for data validation
  and serialisation.
- **Utilities:** `utils/` holds functions shared across the application.
- **Prompts:** `prompts/` holds custom `.prompt` files. Omit this folder on
  projects that have no need for it.
- **Notebooks:** `scripts/` holds exploratory or one-off `.ipynb` notebooks, kept
  separate from application code in `src/`.
- **Tests:** written with `pytest`, split into `unit/` and `integration/`.
  - Test files: `test_*.py`
  - Test functions: `test_*`
  - Shared sample data lives in `tests/fixtures/`
  - `pytest.ini` configures pytest for the project
- **Language:** use British English spelling and grammar throughout, except
  where specific terminology (library names, API fields, etc.) requires
  otherwise.
- **`.gitignore`:** must exclude `.env`, local/temporary files, and
  `__pycache__`, so secrets and generated files never get committed.

## When scaffolding a new project

1. Create the directory tree above under the target root.
2. Add empty `__init__.py` files to `utils/` and `models/`.
3. Create `models/config.py` with a `pydantic-settings` `BaseSettings` subclass
   and a matching `.env` (and `.env.example` if the user wants one committed).
4. Create `tests/pytest.ini` with sane defaults (test paths, `python_files`,
   etc.) if one doesn't already exist.
5. Create `.gitignore` covering `.env`, `__pycache__/`, `*.pyc`, and any
   virtual environment folder in use (e.g. `.venv/`).
6. Only add `prompts/` if the project actually uses prompt files.