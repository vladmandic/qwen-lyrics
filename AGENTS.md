# AI Agent Guide for qwen-lyrics

## Purpose
This file describes how future AI agents should work with the `qwen-lyrics` repository for coding, reviewing, and maintenance tasks.

## Repository overview
- `README.md` describes tool usage: voice separation, lyrics extraction, forced alignment, tuning, ranking, metrics.
- `cli/` contains command-line entrypoints and orchestration logic.
- `src/` contains the main library implementation for alignment, Google integration, lyrics processing, metrics, and data splitting.
- `requirements.txt` lists Python dependencies.
- `samples/` contains example lyrics files and genre samples.

## What to do first
- Read `README.md` and inspect the CLI modules in `cli/`.
- Inspect the library code in `src/` for patterns and existing conventions.
- Identify whether the task affects command-line behavior, library logic, or documentation.

## Agent behavior
- Prefer minimal, targeted edits.
- Preserve existing style, naming, and module structure.
- Avoid adding features beyond the user request unless necessary to fix a bug or make the requested change work correctly.
- Ask the user for clarification when the intent is unclear or the fix may change behavior.

## Coding conventions
- Write idiomatic Python and keep code readable.
- Keep imports organized and avoid unused imports.
- Follow existing project patterns for CLI and library structure.
- When changing CLI code, keep command-line interfaces stable unless the user asked for a change.

## Validation checklist
- Activate the virtual environment before running tools: `source venv/bin/activate`.
- Run syntax and import checks: `python -m py_compile <file>`.
- Run linting on affected files: `python -m ruff check .` and `python -m pylint src/ cli/`.
- If there is no formal test suite, validate behavior manually for changed paths.

## When editing
- Update documentation (`README.md`, `AGENTS.md`) only when it is directly related to the requested change.
- Avoid broad rewrites; make focused changes that solve the issue.
- Keep execution compatible with Python from the repository root.
- If updating library implementation in `src/`, ensure that CLI commands in `cli/` that depend on it are still functional.

## Review checklist
- Does the change satisfy the user request?
- Is the code style consistent with the existing repository?
- Are new or modified files correctly named and placed?
- Is documentation updated only when necessary?
- Are there no obvious syntax errors, lint errors, or import issues?

## Notes
- There is currently no dedicated test suite in the repository. Use manual validation of Python files when needed.
- `AGENT.md` may be empty or present, but `AGENTS.md` should be used by future tools as the agent behavior guide.
