npx wrangler d1 execute wishuponablock --remote --command "
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS auth_sessions (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  token_hash TEXT NOT NULL UNIQUE,
  expires_at_ms INTEGER NOT NULL,
  last_seen_at_ms INTEGER,
  revoked_at_ms INTEGER,
  created_at_ms INTEGER NOT NULL,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_auth_sessions_user_exp
  ON auth_sessions (user_id, expires_at_ms);

CREATE TABLE IF NOT EXISTS auth_tokens (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  type TEXT NOT NULL CHECK (type IN ('email_verify','password_reset')),
  token_hash TEXT NOT NULL UNIQUE,
  email_norm TEXT,
  expires_at_ms INTEGER NOT NULL,
  consumed_at_ms INTEGER,
  created_at_ms INTEGER NOT NULL,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_auth_tokens_user_type_exp
  ON auth_tokens (user_id, type, expires_at_ms);
"
