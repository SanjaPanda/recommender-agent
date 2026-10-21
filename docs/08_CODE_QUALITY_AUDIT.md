# Code quality audit

This is a source-based audit, not a test or security assessment. Findings describe observable implementation characteristics; recommendations are not implemented. Severity indicates review priority, not a confirmed incident.

## Findings

### Medium — API maps some request validation failures to service unavailable

- **Location:** `src/agent/api.py`, `discover`, `recommend`; graph checks in `src/agent/workflow.py`.
- **Evidence:** `ValueError` is grouped with `RuntimeError` in HTTP 503 handlers. Some graph `ValueError`s may result from user-provided category/selection/action values.
- **Why it matters:** Clients may treat a request rejection as temporary infrastructure failure and retry.
- **Suggested improvement:** Review exception taxonomy and map invalid actions to 4xx while retaining 5xx for provider/database errors.
- **Change risk:** Low to medium; changes API contracts and client handling. Confirm tests and UI assumptions first.

### Medium — Live diagnostic scripts have external effects

- **Location:** `scripts/check_agent_live.py`, `scripts/check_router_edits.py`.
- **Evidence:** The live check launches a Uvicorn child, accesses PostgreSQL/API, creates a session, and writes files under `.data`; router checks call Ollama and write a report.
- **Why it matters:** These checks consume resources and alter local state; they are not pure unit tests.
- **Suggested improvement:** Keep side effects and cleanup boundaries explicit; add dry-run/report controls only if needed by actual workflows.
- **Change risk:** Low to medium; may change diagnostic interface and cleanup behavior.

### Medium — Search fast path relies on lexical cues

- **Location:** `src/agent/workflow.py`, `ROUTER_EDIT_CUES`, `_is_plain_search`, `_can_pass_plain_search`.
- **Evidence:** Regex cues and saved-filter/exclusion checks determine whether a message skips the router.
- **Why it matters:** New phrasing can be classified differently, affecting interpretation and model-call latency.
- **Suggested improvement:** Maintain regression examples for ambiguous, negated, and multi-turn searches; change cue logic only with evidence.
- **Change risk:** Medium; affects routing behavior.

### Medium — CLIP fallback does not cover selected-example vector failures

- **Location:** `src/rag/retrieval.py`, `retrieve_products`.
- **Evidence:** CLIP text encoding errors are caught and logged, but selected examples require stored CLIP image vectors. Nomic/database errors propagate.
- **Why it matters:** A text query can fall back to Nomic-only ranking, while example-based retrieval can fail when vectors are absent.
- **Suggested improvement:** Keep error behavior clear to clients and add coverage for missing model/vector cases.
- **Change risk:** Medium; fallback UX or ranking semantics may change.

### Low — Relevance cutoffs are fixed starting values

- **Location:** `src/agent/relevance.py`, similarity constants.
- **Evidence:** Fixed Nomic, CLIP, and example thresholds gate recommendation results; current architecture notes call them initial values based on a small judged query sample.
- **Why it matters:** Model/catalog changes can shift score distributions and relevance/coverage.
- **Suggested improvement:** Maintain a representative judged query set and measure precision and coverage before changing thresholds.
- **Change risk:** Medium; affects recommendations and fallback frequency.

### Low — Migration runner splits SQL text on semicolons

- **Location:** `scripts/db/apply_migrations.py`, `main`.
- **Evidence:** SQL is split with a regular expression on `;` before statements are executed transactionally.
- **Why it matters:** This works for current simple statements but is not a complete SQL parser for procedural blocks or semicolons inside literals.
- **Suggested improvement:** If migration complexity grows, use a migration tool or explicit statement format.
- **Change risk:** Medium; changes migration execution and possibly dependencies.

### Low — Historical DB notes have different time scope from active design docs

- **Location:** `docs/database.md`, `docs/architecture.md`, `docs/Agent.md`.
- **Evidence:** Database notes contain dated cleanup states while architecture files describe current intended behavior.
- **Why it matters:** Readers could mistake a historical row count/schema for live state.
- **Suggested improvement:** Preserve dates and historical qualifiers; perform live inspection only when authorized and needed.
- **Change risk:** Low; documentation-only.

## Limits of this audit

- No dead/unreachable active source was established. Stale `.pyc` names alone are insufficient evidence.
- No test failures or security vulnerabilities were verified; tests and services were not run.
- Retained schema objects and migrations may be deliberate history; live schema state was not inspected.
- Parameterized SQL and local-only model endpoint checks are visible in source, but this is not a full security review.

No audit recommendation has been implemented.
