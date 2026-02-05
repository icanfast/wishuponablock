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
  const { session, sample } = payload as {
    session?: Record<string, unknown>;
    sample?: Record<string, unknown>;
  };
  if (!session || !sample) {
    return jsonResponse({ error: 'Missing session or sample.' }, 400);
  }

  const sessionId = session.id as string | undefined;
  if (!sessionId) {
    return jsonResponse({ error: 'Missing session id.' }, 400);
  }

  const mode = session.mode as { id?: string; options?: unknown } | undefined;
  const now = new Date().toISOString();

  await env.DB.prepare(
    `INSERT OR IGNORE INTO snapshot_sessions
      (id, created_at, protocol_version, rows, cols, piece_order, settings, mode_id, mode_options, comment)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  )
    .bind(
      sessionId,
      session.createdAt ?? now,
      session.protocolVersion ?? 0,
      session.rows ?? 0,
      session.cols ?? 0,
      JSON.stringify(session.pieceOrder ?? []),
      JSON.stringify(session.settings ?? {}),
      mode?.id ?? null,
      JSON.stringify(mode?.options ?? null),
      session.comment ?? null,
    )
    .run();

  await env.DB.prepare(
    `INSERT OR REPLACE INTO snapshot_samples
      (session_id, sample_index, time_ms, board, hold, created_at)
     VALUES (?, ?, ?, ?, ?, ?)`,
  )
    .bind(
      sessionId,
      sample.index ?? 0,
      sample.timeMs ?? 0,
      JSON.stringify(sample.board ?? []),
      sample.hold ?? 0,
      now,
    )
    .run();

  return okResponse();
};

const handleLabels = async (env: Env, payload: unknown): Promise<Response> => {
  if (!payload || typeof payload !== 'object') {
    return jsonResponse({ error: 'Invalid payload.' }, 400);
  }
  const record = payload as Record<string, unknown>;
  const source = (record.source ?? {}) as Record<string, unknown>;

  await env.DB.prepare(
    `INSERT INTO label_records
      (created_at, session_id, file_name, sample_index, shown_count, piece_order, board, hold, labels)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  )
    .bind(
      record.createdAt ?? new Date().toISOString(),
      source.sessionId ?? null,
      source.file ?? null,
      source.sampleIndex ?? null,
      source.shownCount ?? null,
      JSON.stringify(record.pieceOrder ?? []),
      record.board ?? '',
      record.hold ?? null,
      JSON.stringify(record.labels ?? []),
    )
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

    return env.ASSETS.fetch(request);
  },
};
