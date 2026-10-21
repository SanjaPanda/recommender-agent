# Codex Project Instructions

## Project Guidelines

- Preserve existing functionality when making changes.
- Follow the active design documented in `docs/Agent.md` and `docs/architecture.md`.
- Keep `src/` limited to application agent and UI code.
- Keep offline RAG processing under the existing `src/rag/` package; do not add standalone experiment code to `src/agent/`.
- Keep tests in `tests/`.
- Keep setup and maintenance scripts in `scripts/`.
- Store temporary Codex-generated artifacts in `.codex-work/`.
- Store permanent engineering observations in `observations/`.
- Avoid unnecessary dependencies, configuration, and CLI arguments.
- Validate relevant functionality after modifications.

## Repository Cleanup

Use the `agent-repo-cleanup` skill when auditing, restructuring, cleaning, or verifying the repository.

Default to AUDIT mode. Do not perform destructive cleanup without approval.

## Technical Content Capture

Use the `project-content-planner` skill in CAPTURE mode at meaningful development milestones.

Record:
- Architecture decisions and trade-offs.
- Debugging discoveries.
- Experiments and evaluation findings.
- Performance measurements.
- Lessons that could become Medium articles or LinkedIn posts.

Maintain `observations/content/ideas.md`.

Do not draft or publish articles during development.

## Documentation and Checkpoints

Update relevant documentation when architecture changes.

Record meaningful verified milestones in `observations/checkpoints/`.

Never claim a test or benchmark passed unless it was actually executed.

## Active Application

- The catalog is exclusively Apparel & Accessories; reject all other source records and constrain every retrieval query to that taxonomy.
- Preserve the current repository layout. Keep only active application code, focused tests, and current setup/architecture/database documentation; retain durable engineering observations under `observations/`.
- Run tests with `.venv/Scripts/python.exe -m unittest discover -s tests -v`.
