type D1PreparedStatement = {
  bind: (...values: unknown[]) => D1PreparedStatement;
  run: () => Promise<unknown>;
  all: () => Promise<{ results: Array<Record<string, unknown>> }>;
};

type D1Database = {
  prepare: (query: string) => D1PreparedStatement;
};

type Env = {
  DB: D1Database;
  ASSETS: { fetch: (request: Request) => Promise<Response> };
};

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

const handleSnapshots = async (
  env: Env,
  payload: unknown,
): Promise<Response> => {
  if (!payload || typeof payload !== 'object') {
    return jsonResponse({ error: 'Invalid payload.' }, 400);
  }

  const record = payload as {
    createdAt?: string;
    meta?: Record<string, unknown>;
    board?: unknown;
  };

  if (!record.meta || record.board == null) {
    return jsonResponse({ error: 'Missing meta or board.' }, 400);
  }

  const createdAt =
    typeof record.createdAt === 'string'
      ? record.createdAt
      : new Date().toISOString();

  await env.DB.prepare(
    `INSERT INTO snapshots (created_at, meta, board) VALUES (?, ?, ?)`,
  )
    .bind(createdAt, JSON.stringify(record.meta), JSON.stringify(record.board))
    .run();

  return okResponse();
};

const handleLabels = async (env: Env, payload: unknown): Promise<Response> => {
  if (!payload || typeof payload !== 'object') {
    return jsonResponse({ error: 'Invalid payload.' }, 400);
  }
  const record = payload as { createdAt?: string; data?: unknown };

  if (!record.data) {
    return jsonResponse({ error: 'Missing data.' }, 400);
  }

  const createdAt =
    typeof record.createdAt === 'string'
      ? record.createdAt
      : new Date().toISOString();

  await env.DB.prepare(`INSERT INTO labels (created_at, data) VALUES (?, ?)`)
    .bind(createdAt, JSON.stringify(record.data))
    .run();

  return okResponse();
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

  await env.DB.prepare(
    `INSERT INTO feedback (created_at, feedback, contact) VALUES (?, ?, ?)`,
  )
    .bind(createdAt, feedback, contact || null)
    .run();

  return okResponse();
};

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    if (request.method === 'OPTIONS') {
      return okResponse();
    }

    if (url.pathname.startsWith('/api/snapshots')) {
      if (url.pathname.startsWith('/api/snapshots/random')) {
        if (request.method !== 'GET') {
          return jsonResponse({ error: 'Method not allowed.' }, 405);
        }
        const mode = url.searchParams.get('mode') ?? 'all';
        const trigger = url.searchParams.get('trigger') ?? 'all';
        const build = url.searchParams.get('build') ?? 'all';
        let sql = 'SELECT id, created_at, meta, board FROM snapshots';
        const params: unknown[] = [];
        const clauses: string[] = [];
        if (mode !== 'all') {
          if (mode === 'unknown') {
            clauses.push("json_extract(meta, '$.session.mode.id') IS NULL");
          } else {
            clauses.push("json_extract(meta, '$.session.mode.id') = ?");
            params.push(mode);
          }
        }
        if (trigger !== 'all') {
          if (trigger === 'auto') {
            clauses.push(
              "(json_extract(meta, '$.trigger') IS NULL OR json_extract(meta, '$.trigger') IN ('lock','hold'))",
            );
          } else {
            clauses.push("json_extract(meta, '$.trigger') = ?");
            params.push(trigger);
          }
        }
        if (build !== 'all') {
          if (build === 'unknown') {
            clauses.push(
              "json_extract(meta, '$.session.buildVersion') IS NULL",
            );
          } else {
            clauses.push("json_extract(meta, '$.session.buildVersion') = ?");
            params.push(build);
          }
        }
        if (clauses.length > 0) {
          sql += ` WHERE ${clauses.join(' AND ')}`;
        }
        sql += ' ORDER BY RANDOM() LIMIT 1';
        const result = await env.DB.prepare(sql)
          .bind(...params)
          .all();
        const row = result?.results?.[0];
        if (!row) {
          return jsonResponse({ error: 'No snapshots available.' }, 404);
        }
        let meta: unknown = row.meta;
        let board: unknown = row.board;
        if (typeof meta === 'string') {
          meta = JSON.parse(meta);
        }
        if (typeof board === 'string') {
          board = JSON.parse(board);
        }
        return jsonResponse({
          id: row.id,
          createdAt: row.created_at,
          meta,
          board,
        });
      }
      if (request.method !== 'POST') {
        return jsonResponse({ error: 'Method not allowed.' }, 405);
      }
      const payload = await request.json().catch(() => null);
      return handleSnapshots(env, payload);
    }

    if (url.pathname.startsWith('/api/labels')) {
      if (request.method !== 'POST') {
        return jsonResponse({ error: 'Method not allowed.' }, 405);
      }
      const payload = await request.json().catch(() => null);
      return handleLabels(env, payload);
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
};
