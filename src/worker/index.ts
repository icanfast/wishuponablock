type D1PreparedStatement = {
  bind: (...values: unknown[]) => D1PreparedStatement;
  run: () => Promise<unknown>;
};

type D1Database = {
  prepare: (query: string) => D1PreparedStatement;
};

type Env = {
  DB: D1Database;
  ASSETS: { fetch: (request: Request) => Promise<Response> };
  RELEASE_CHANNEL?: string;
  FEATURE_FLAGS?: string;
  MODELS_BUCKET?: unknown;
  RECORDINGS_BUCKET?: unknown;
};

const DAY_MS = 24 * 60 * 60 * 1000;
const SESSION_SLIDING_WINDOW_DAYS = 90;
const SESSION_REVOKED_RETENTION_MS = 30 * DAY_MS;
const TOKEN_CONSUMED_RETENTION_MS = 7 * DAY_MS;
const LEGACY_API_ERROR =
  'Legacy snapshot/label APIs were removed on the 0.3.0 dev branch.';

const jsonResponse = (body: unknown, status = 200): Response =>
  new Response(JSON.stringify(body), {
    status,
    headers: {
      'content-type': 'application/json',
      'access-control-allow-origin': '*',
      'access-control-allow-headers': 'content-type',
    },
  });

const okResponse = (): Response =>
  new Response(null, {
    status: 204,
    headers: {
      'access-control-allow-origin': '*',
      'access-control-allow-headers': 'content-type',
    },
  });

const asString = (value: unknown): string | null =>
  typeof value === 'string' && value.trim() ? value.trim() : null;

const parseFeatureFlags = (raw: unknown): Record<string, unknown> => {
  if (typeof raw !== 'string' || !raw.trim()) {
    return {};
  }
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      return {};
    }
    const out: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(parsed)) {
      if (!key.trim()) continue;
      if (
        value == null ||
        typeof value === 'boolean' ||
        typeof value === 'number' ||
        typeof value === 'string'
      ) {
        out[key] = value;
      }
    }
    return out;
  } catch {
    return {};
  }
};

const cleanupAuthState = async (env: Env): Promise<void> => {
  const nowMs = Date.now();
  const consumedTokenCutoffMs = nowMs - TOKEN_CONSUMED_RETENTION_MS;
  const revokedSessionCutoffMs = nowMs - SESSION_REVOKED_RETENTION_MS;
  try {
    await env.DB.prepare(
      `DELETE FROM auth_tokens
       WHERE expires_at_ms < ?
          OR (consumed_at_ms IS NOT NULL AND consumed_at_ms < ?)`,
    )
      .bind(nowMs, consumedTokenCutoffMs)
      .run();
    await env.DB.prepare(
      `DELETE FROM auth_sessions
       WHERE expires_at_ms < ?
          OR (revoked_at_ms IS NOT NULL AND revoked_at_ms < ?)`,
    )
      .bind(nowMs, revokedSessionCutoffMs)
      .run();
  } catch (error) {
    console.error(
      `[cron] auth cleanup failed (session_window_days=${SESSION_SLIDING_WINDOW_DAYS})`,
      error,
    );
  }
};

const handleFeedback = async (
  env: Env,
  payload: unknown,
): Promise<Response> => {
  if (!payload || typeof payload !== 'object') {
    return jsonResponse({ error: 'Invalid payload.' }, 400);
  }
  const record = payload as {
    createdAt?: string;
    feedback?: string;
    contact?: string | null;
  };

  const feedback =
    typeof record.feedback === 'string' ? record.feedback.trim() : '';
  if (!feedback) {
    return jsonResponse({ error: 'Missing feedback.' }, 400);
  }
  const contact =
    typeof record.contact === 'string' ? record.contact.trim() : '';
  const createdAt =
    typeof record.createdAt === 'string'
      ? record.createdAt
      : new Date().toISOString();

  try {
    await env.DB.prepare(
      `INSERT INTO feedback (created_at, feedback, contact) VALUES (?, ?, ?)`,
    )
      .bind(createdAt, feedback, contact || null)
      .run();
  } catch {
    return jsonResponse(
      {
        error: 'Feedback storage is currently unavailable in this environment.',
      },
      503,
    );
  }

  return okResponse();
};

const isLegacyApiPath = (pathname: string): boolean =>
  pathname.startsWith('/api/snapshots') || pathname.startsWith('/api/labels');

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    if (request.method === 'OPTIONS') {
      return okResponse();
    }

    if (url.pathname.startsWith('/api/runtime/flags')) {
      if (request.method !== 'GET') {
        return jsonResponse({ error: 'Method not allowed.' }, 405);
      }
      const channel = asString(env.RELEASE_CHANNEL) ?? 'unknown';
      return jsonResponse({
        channel,
        flags: parseFeatureFlags(env.FEATURE_FLAGS),
      });
    }

    if (isLegacyApiPath(url.pathname)) {
      return jsonResponse({ error: LEGACY_API_ERROR }, 410);
    }

    if (url.pathname.startsWith('/api/feedback')) {
      if (request.method !== 'POST') {
        return jsonResponse({ error: 'Method not allowed.' }, 405);
      }
      const payload = await request.json().catch(() => null);
      return handleFeedback(env, payload);
    }

    return env.ASSETS.fetch(request);
  },
  async scheduled(_controller: unknown, env: Env): Promise<void> {
    await cleanupAuthState(env);
  },
};
