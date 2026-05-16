-- Platform control plane: tenants (channels), users, login sessions.
-- One Postgres database hosts this schema alongside the RAG schema in init_rag.sql.

CREATE SCHEMA IF NOT EXISTS control;

DROP TABLE IF EXISTS control.sessions     CASCADE;
DROP TABLE IF EXISTS control.users        CASCADE;
DROP TABLE IF EXISTS control.tenant       CASCADE;
DROP TABLE IF EXISTS control.channel      CASCADE;

-- A channel is a tenant. rag_channel_id links it to a row in the RAG schema's
-- public.channels table; NULL means the tenant has no RAG corpus yet.
CREATE TABLE control.channel (
    channelid           SERIAL PRIMARY KEY,
    name                VARCHAR(100) NOT NULL UNIQUE,
    rag_channel_id      UUID UNIQUE,
    metadata            JSONB DEFAULT '{}',
    authorized_features JSONB DEFAULT '[]'
);

-- A user belongs to exactly one channel. authorized_features gates UI
-- capabilities: the value "admin_panel" is what unlocks the admin tabs in
-- the Gradio frontend.
CREATE TABLE control.users (
    userid              SERIAL PRIMARY KEY,
    name                VARCHAR(100) NOT NULL,
    username            VARCHAR(100) NOT NULL UNIQUE,
    password            VARCHAR(255) NOT NULL,
    channelid           INT NOT NULL,
    authorized_features JSONB DEFAULT '[]',
    authorized_data     JSONB DEFAULT '{}',

    CONSTRAINT fk_user_channel
        FOREIGN KEY (channelid)
        REFERENCES control.channel(channelid)
        ON DELETE CASCADE
);

-- One row per login. expires_at is enforced inside the JWT itself; ended_at
-- is written on explicit logout so this table doubles as an audit trail.
CREATE TABLE control.sessions (
    session_id  UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     INT          NOT NULL REFERENCES control.users(userid) ON DELETE CASCADE,
    ip_address  VARCHAR(45),
    login_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
    expires_at  TIMESTAMPTZ  NOT NULL,
    ended_at    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS sessions_user_id_idx ON control.sessions (user_id);
CREATE INDEX IF NOT EXISTS sessions_login_at_idx ON control.sessions (login_at);

-- ── Seed: three demo tenants ──────────────────────────────────────────────────
-- rag_channel_id is NULL on bootstrap; admins link a RAG channel from the UI
-- (Channels tab) which writes the UUID back here.
INSERT INTO control.channel (name, rag_channel_id, metadata, authorized_features)
VALUES
(
    'Epsilon AI',
    NULL,
    '{"industry": "AI Solutions", "country": "Egypt"}',
    '["chatbot", "document_search", "voice_agent"]'
),
(
    'Aman',
    NULL,
    '{"industry": "Financial Services", "country": "Egypt"}',
    '["chatbot", "document_search", "tables_search"]'
),
(
    'Swedy',
    NULL,
    '{"industry": "Industrial Manufacturing", "country": "Egypt"}',
    '["chatbot", "document_search", "tables_search", "voice_agent"]'
);

-- ── Seed: demo users ──────────────────────────────────────────────────────────
-- Passwords are plaintext here for the demo. Only `admin` carries the
-- `admin_panel` feature, so only Epsilon AI has full admin access.
INSERT INTO control.users (name, username, password, channelid, authorized_features, authorized_data)
VALUES
(
    'Omar Yasser',
    'omar',
    '1111',
    (SELECT channelid FROM control.channel WHERE name = 'Epsilon AI'),
    '["chatbot", "document_search"]',
    '{}'
),
(
    'Admin User',
    'admin',
    '1234',
    (SELECT channelid FROM control.channel WHERE name = 'Epsilon AI'),
    '["chatbot", "document_search", "voice_agent", "admin_panel"]',
    '{}'
),
(
    'Aman Admin',
    'aman_admin',
    '1234',
    (SELECT channelid FROM control.channel WHERE name = 'Aman'),
    '["chatbot", "document_search", "tables_search"]',
    '{}'
),
(
    'Aman User',
    'aman_user',
    '1111',
    (SELECT channelid FROM control.channel WHERE name = 'Aman'),
    '["chatbot", "document_search"]',
    '{}'
),
(
    'Swedy Admin',
    'swedy_admin',
    '1234',
    (SELECT channelid FROM control.channel WHERE name = 'Swedy'),
    '["chatbot", "document_search", "tables_search", "voice_agent"]',
    '{}'
),
(
    'Swedy Engineer',
    'swedy_engineer',
    '1111',
    (SELECT channelid FROM control.channel WHERE name = 'Swedy'),
    '["chatbot", "document_search", "tables_search"]',
    '{}'
);
