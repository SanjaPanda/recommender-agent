-- Durable turn snapshots and cart state, serialized per thread by row locks.
CREATE TABLE IF NOT EXISTS agent_sessions (
    thread_id text PRIMARY KEY,
    state jsonb NOT NULL DEFAULT '{}'::jsonb,
    updated_at timestamptz NOT NULL DEFAULT now()
);
