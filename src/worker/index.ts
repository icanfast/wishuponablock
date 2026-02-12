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

const asString = (value: unknown): string | null =>
  typeof value === 'string' && value.trim() ? value.trim() : null;

const asInt = (value: unknown): number | null => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return Math.trunc(value);
  }
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return Math.trunc(parsed);
  }
  return null;
};

type SnapshotMetaPayload = {
  session?: {
    id?: unknown;
    createdAt?: unknown;
    protocolVersion?: unknown;
    rows?: unknown;
    cols?: unknown;
    mode?: {
      id?: unknown;
    };
    buildVersion?: unknown;
    settings?: {
      generator?: {
        type?: unknown;
        ml?: {
          strategy?: unknown;
        };
      };
    };
    model_url?: unknown;
    device_id?: unknown;
    user_id?: unknown;
  };
  sample?: {
    index?: unknown;
    timeMs?: unknown;
    hold?: unknown;
    linesLeft?: unknown;
    level?: unknown;
    score?: unknown;
  };
  trigger?: unknown;
};

const extractSnapshotIndexData = (meta: Record<string, unknown>) => {
  const typed = meta as SnapshotMetaPayload;
  const session = typed.session;
  const sample = typed.sample;
  return {
    sessionId: asString(session?.id),
    sessionCreatedAt: asString(session?.createdAt),
    protocolVersion: asInt(session?.protocolVersion),
    rows: asInt(session?.rows),
    cols: asInt(session?.cols),
    modeId: asString(session?.mode?.id),
    buildVersion: asString(session?.buildVersion),
    generatorType: asString(session?.settings?.generator?.type),
    generatorStrategy: asString(session?.settings?.generator?.ml?.strategy),
    trigger: asString(typed.trigger),
    sampleIndex: asInt(sample?.index),
    sampleTimeMs: asInt(sample?.timeMs),
    sampleHold: asInt(sample?.hold),
    linesLeft: asInt(sample?.linesLeft),
    level: asInt(sample?.level),
    score: asInt(sample?.score),
    modelUrl: asString(session?.model_url),
    deviceId: asString(session?.device_id),
    userId: asString(session?.user_id),
  };
};

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
  const metaJson = JSON.stringify(record.meta);
  const boardJson = JSON.stringify(record.board);

  await env.DB.prepare(
    `INSERT INTO snapshots (created_at, meta, board) VALUES (?, ?, ?)`,
  )
    .bind(createdAt, metaJson, boardJson)
    .run();

  const idResult = await env.DB.prepare(
    `SELECT id FROM snapshots
     WHERE created_at = ? AND meta = ? AND board = ?
     ORDER BY id DESC
     LIMIT 1`,
  )
    .bind(createdAt, metaJson, boardJson)
    .all();
  const snapshotId = asInt(idResult?.results?.[0]?.id);
  if (snapshotId == null) {
    return jsonResponse({ error: 'Snapshot saved, but indexing failed.' }, 500);
  }

  const idx = extractSnapshotIndexData(record.meta);
  await env.DB.prepare(
    `INSERT OR REPLACE INTO snapshots_idx (
      snapshot_id,
      created_at,
      session_id,
      session_created_at,
      protocol_version,
      rows,
      cols,
      mode_id,
      build_version,
      generator_type,
      generator_strategy,
      trigger,
      sample_index,
      sample_time_ms,
      sample_hold,
      lines_left,
      level,
      score,
      model_url,
      device_id,
      user_id
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  )
    .bind(
      snapshotId,
      createdAt,
      idx.sessionId,
      idx.sessionCreatedAt,
      idx.protocolVersion,
      idx.rows,
      idx.cols,
      idx.modeId,
      idx.buildVersion,
      idx.generatorType,
      idx.generatorStrategy,
      idx.trigger,
      idx.sampleIndex,
      idx.sampleTimeMs,
      idx.sampleHold,
      idx.linesLeft,
      idx.level,
      idx.score,
      idx.modelUrl,
      idx.deviceId,
      idx.userId,
    )
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
      if (url.pathname.startsWith('/api/snapshots/builds')) {
        if (request.method !== 'GET') {
          return jsonResponse({ error: 'Method not allowed.' }, 405);
        }
        const sql = `
          SELECT
            COALESCE(json_extract(meta, '$.session.buildVersion'), 'unknown') AS build,
            COUNT(*) AS count
          FROM snapshots
          GROUP BY build
        `;
        const result = await env.DB.prepare(sql).all();
        const builds = (result?.results ?? [])
          .map((row) => ({
            build:
              typeof row.build === 'string' && row.build.trim()
                ? row.build
                : 'unknown',
            count:
              typeof row.count === 'number' && Number.isFinite(row.count)
                ? row.count
                : 0,
          }))
          .sort((a, b) => {
            if (a.build === 'unknown') return 1;
            if (b.build === 'unknown') return -1;
            return a.build.localeCompare(b.build);
          });
        return jsonResponse({ builds });
      }
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
