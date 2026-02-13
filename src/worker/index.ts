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

const BUILD_COUNTS_CACHE_TTL_MS = 30_000;
const LABEL_PROGRESS_CACHE_TTL_MS = 30_000;
const SNAPSHOT_BOUNDS_CACHE_TTL_MS = 10_000;
const VIRTUAL_SESSION_SIZE = 100;

type BuildCountsCache = {
  expiresAt: number;
  builds: Array<{ build: string; count: number }>;
};

type LabelProgressCacheEntry = {
  expiresAt: number;
  labeledBoards: number;
};

let buildCountsCache: BuildCountsCache | null = null;
const labelProgressByBuildCache = new Map<string, LabelProgressCacheEntry>();

const snapshotBoundsCache = new Map<
  string,
  {
    expiresAt: number;
    minId: number;
    maxId: number;
  }
>();

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

const asCount = (value: unknown): number => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return Math.max(0, Math.trunc(value));
  }
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return Math.max(0, Math.trunc(parsed));
    }
  }
  return 0;
};

type SnapshotFilterSql = {
  where: string;
  params: unknown[];
};

const buildSnapshotFilterSql = (
  mode: string,
  trigger: string,
  build: string,
): SnapshotFilterSql => {
  const params: unknown[] = [];
  const clauses: string[] = [];
  if (mode !== 'all') {
    if (mode === 'unknown') {
      clauses.push('idx.mode_id IS NULL');
    } else {
      clauses.push('idx.mode_id = ?');
      params.push(mode);
    }
  }
  if (trigger !== 'all') {
    if (trigger === 'auto') {
      clauses.push("(idx.trigger IS NULL OR idx.trigger IN ('lock','hold'))");
    } else {
      clauses.push('idx.trigger = ?');
      params.push(trigger);
    }
  }
  if (build !== 'all') {
    if (build === 'unknown') {
      clauses.push('idx.build_version IS NULL');
    } else {
      clauses.push('idx.build_version = ?');
      params.push(build);
    }
  }
  return {
    where: clauses.length > 0 ? ` WHERE ${clauses.join(' AND ')}` : '',
    params,
  };
};

const appendFilterClause = (
  filter: SnapshotFilterSql,
  clause: string,
): SnapshotFilterSql => ({
  where: filter.where ? `${filter.where} AND ${clause}` : ` WHERE ${clause}`,
  params: [...filter.params],
});

const appendFilterClauseWithParams = (
  filter: SnapshotFilterSql,
  clause: string,
  params: unknown[],
): SnapshotFilterSql => ({
  where: filter.where ? `${filter.where} AND ${clause}` : ` WHERE ${clause}`,
  params: [...filter.params, ...params],
});

const parseCsvInts = (value: string | null, maxItems = 12): number[] => {
  if (!value) return [];
  const out: number[] = [];
  for (const part of value.split(',')) {
    if (out.length >= maxItems) break;
    const parsed = asInt(part);
    if (parsed == null) continue;
    if (parsed <= 0) continue;
    if (out.includes(parsed)) continue;
    out.push(parsed);
  }
  return out;
};

const parseCsvStrings = (value: string | null, maxItems = 12): string[] => {
  if (!value) return [];
  const out: string[] = [];
  for (const part of value.split(',')) {
    if (out.length >= maxItems) break;
    const normalized = part.trim();
    if (!normalized) continue;
    if (out.includes(normalized)) continue;
    out.push(normalized);
  }
  return out;
};

const randomInt = (min: number, max: number): number => {
  if (max <= min) return min;
  return min + Math.floor(Math.random() * (max - min + 1));
};

const snapshotFilterCacheKey = (mode: string, trigger: string, build: string) =>
  `${mode}|${trigger}|${build}`;

const readSnapshotByFilter = async (
  env: Env,
  filter: SnapshotFilterSql,
  filterKey: string | null,
): Promise<Record<string, unknown> | null> => {
  const now = Date.now();
  let minId: number | null = null;
  let maxId: number | null = null;
  const cachedBounds =
    filterKey != null ? snapshotBoundsCache.get(filterKey) : undefined;
  if (filterKey != null && cachedBounds && cachedBounds.expiresAt > now) {
    minId = cachedBounds.minId;
    maxId = cachedBounds.maxId;
  } else {
    const boundsSql = `
      SELECT
        MIN(idx.snapshot_id) AS min_id,
        MAX(idx.snapshot_id) AS max_id
      FROM snapshots_idx idx
      ${filter.where}
    `;
    const boundsResult = await env.DB.prepare(boundsSql)
      .bind(...filter.params)
      .all();
    const boundsRow = boundsResult?.results?.[0];
    minId = asInt(boundsRow?.min_id);
    maxId = asInt(boundsRow?.max_id);
    if (filterKey != null && minId != null && maxId != null) {
      if (snapshotBoundsCache.size > 128) {
        snapshotBoundsCache.clear();
      }
      snapshotBoundsCache.set(filterKey, {
        minId,
        maxId,
        expiresAt: now + SNAPSHOT_BOUNDS_CACHE_TTL_MS,
      });
    }
  }

  if (minId == null || maxId == null) {
    return null;
  }

  const pivot = randomInt(minId, maxId);
  const seekWherePrefix = filter.where ? `${filter.where} AND` : ' WHERE';
  const seekFromPivotSql = `
    SELECT
      s.id,
      s.created_at,
      s.meta,
      s.board,
      idx.session_id,
      idx.batch_id
    FROM snapshots_idx idx
    JOIN snapshots s ON s.id = idx.snapshot_id
    ${seekWherePrefix} idx.snapshot_id >= ?
    ORDER BY idx.snapshot_id
    LIMIT 1
  `;
  let result = await env.DB.prepare(seekFromPivotSql)
    .bind(...filter.params, pivot)
    .all();
  let row = result?.results?.[0];
  if (!row) {
    const wrapSql = `
      SELECT
        s.id,
        s.created_at,
        s.meta,
        s.board,
        idx.session_id,
        idx.batch_id
      FROM snapshots_idx idx
      JOIN snapshots s ON s.id = idx.snapshot_id
      ${seekWherePrefix} idx.snapshot_id < ?
      ORDER BY idx.snapshot_id DESC
      LIMIT 1
    `;
    result = await env.DB.prepare(wrapSql)
      .bind(...filter.params, pivot)
      .all();
    row = result?.results?.[0];
  }
  return row ?? null;
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

  const insertResult = (await env.DB.prepare(
    `INSERT INTO snapshots (created_at, meta, board) VALUES (?, ?, ?)`,
  )
    .bind(createdAt, metaJson, boardJson)
    .run()) as { meta?: { last_row_id?: unknown } };

  let snapshotId = asInt(insertResult?.meta?.last_row_id);
  if (snapshotId == null) {
    const idResult = await env.DB.prepare(
      `SELECT id FROM snapshots
       WHERE created_at = ? AND meta = ? AND board = ?
       ORDER BY id DESC
       LIMIT 1`,
    )
      .bind(createdAt, metaJson, boardJson)
      .all();
    snapshotId = asInt(idResult?.results?.[0]?.id);
  }
  if (snapshotId == null) {
    return jsonResponse({ error: 'Snapshot saved, but indexing failed.' }, 500);
  }

  const idx = extractSnapshotIndexData(record.meta);
  const batchId =
    idx.sampleIndex != null && idx.sampleIndex >= 0
      ? Math.floor(idx.sampleIndex / VIRTUAL_SESSION_SIZE)
      : null;
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
      batch_id,
      sample_time_ms,
      sample_hold,
      lines_left,
      level,
      score,
      label_count,
      last_labeled_at,
      model_url,
      device_id,
      user_id
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
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
      batchId,
      idx.sampleTimeMs,
      idx.sampleHold,
      idx.linesLeft,
      idx.level,
      idx.score,
      0,
      null,
      idx.modelUrl,
      idx.deviceId,
      idx.userId,
    )
    .run();

  buildCountsCache = null;
  snapshotBoundsCache.clear();

  return okResponse();
};

const handleSnapshotsBatch = async (
  env: Env,
  payload: unknown,
): Promise<Response> => {
  if (!Array.isArray(payload)) {
    return jsonResponse({ error: 'Invalid payload.' }, 400);
  }
  if (payload.length === 0) {
    return okResponse();
  }
  const maxBatchSize = 100;
  if (payload.length > maxBatchSize) {
    return jsonResponse(
      { error: `Batch too large (max ${maxBatchSize}).` },
      413,
    );
  }
  for (const record of payload) {
    const result = await handleSnapshots(env, record);
    if (result.status !== 204) {
      return result;
    }
  }
  return okResponse();
};

const resolveLabelSnapshotRef = async (
  env: Env,
  data: unknown,
): Promise<{
  snapshotId: number | null;
  sessionId: string | null;
  sampleIndex: number | null;
  batchId: number | null;
}> => {
  const payload = (data ?? {}) as {
    source?: {
      snapshotId?: unknown;
      snapshot_id?: unknown;
      sessionId?: unknown;
      sampleIndex?: unknown;
    };
  };
  const source = payload.source;
  const sessionId = asString(source?.sessionId);
  const sampleIndex = asInt(source?.sampleIndex);
  let snapshotId = asInt(source?.snapshotId) ?? asInt(source?.snapshot_id);
  let batchId =
    sampleIndex != null && sampleIndex >= 0
      ? Math.floor(sampleIndex / VIRTUAL_SESSION_SIZE)
      : null;

  if (
    snapshotId == null &&
    sessionId &&
    sampleIndex != null &&
    Number.isFinite(sampleIndex)
  ) {
    const lookup = await env.DB.prepare(
      `SELECT snapshot_id, batch_id
       FROM snapshots_idx
       WHERE session_id = ? AND sample_index = ?
       ORDER BY snapshot_id DESC
       LIMIT 1`,
    )
      .bind(sessionId, sampleIndex)
      .all();
    const row = lookup?.results?.[0];
    snapshotId = asInt(row?.snapshot_id);
    batchId = asInt(row?.batch_id) ?? batchId;
  }

  if (snapshotId != null && batchId == null) {
    const lookup = await env.DB.prepare(
      `SELECT batch_id
       FROM snapshots_idx
       WHERE snapshot_id = ?
       LIMIT 1`,
    )
      .bind(snapshotId)
      .all();
    batchId = asInt(lookup?.results?.[0]?.batch_id);
  }

  return { snapshotId, sessionId, sampleIndex, batchId };
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
  const resolved = await resolveLabelSnapshotRef(env, record.data);

  await env.DB.prepare(
    `INSERT INTO labels (
      created_at,
      data,
      snapshot_id,
      session_id,
      sample_index,
      batch_id
    ) VALUES (?, ?, ?, ?, ?, ?)`,
  )
    .bind(
      createdAt,
      JSON.stringify(record.data),
      resolved.snapshotId,
      resolved.sessionId,
      resolved.sampleIndex,
      resolved.batchId,
    )
    .run();

  if (resolved.snapshotId != null) {
    await env.DB.prepare(
      `UPDATE snapshots_idx
       SET
         label_count = COALESCE(label_count, 0) + 1,
         last_labeled_at = ?
       WHERE snapshot_id = ?`,
    )
      .bind(createdAt, resolved.snapshotId)
      .run();
    snapshotBoundsCache.clear();
  }
  labelProgressByBuildCache.clear();

  return okResponse();
};

const readLabeledBoardCountForBuild = async (
  env: Env,
  build: string,
): Promise<number> => {
  const normalizedBuild = build.trim();
  if (!normalizedBuild) return 0;

  const now = Date.now();
  const cached = labelProgressByBuildCache.get(normalizedBuild);
  if (cached && cached.expiresAt > now) {
    return cached.labeledBoards;
  }

  const result = await env.DB.prepare(
    `SELECT COUNT(*) AS count
     FROM snapshots_idx
     WHERE build_version = ?
       AND COALESCE(label_count, 0) > 0`,
  )
    .bind(normalizedBuild)
    .all();
  const labeledBoards = asCount(result?.results?.[0]?.count);
  if (labelProgressByBuildCache.size > 128) {
    labelProgressByBuildCache.clear();
  }
  labelProgressByBuildCache.set(normalizedBuild, {
    expiresAt: now + LABEL_PROGRESS_CACHE_TTL_MS,
    labeledBoards,
  });
  return labeledBoards;
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
        const now = Date.now();
        if (buildCountsCache && buildCountsCache.expiresAt > now) {
          return jsonResponse({ builds: buildCountsCache.builds });
        }
        const sql = `
          SELECT
            COALESCE(build_version, 'unknown') AS build,
            COUNT(*) AS count
          FROM snapshots_idx
          GROUP BY COALESCE(build_version, 'unknown')
        `;
        const result = await env.DB.prepare(sql).all();
        const builds = (result?.results ?? [])
          .map((row) => ({
            build:
              typeof row.build === 'string' && row.build.trim()
                ? row.build
                : 'unknown',
            count: asCount(row.count),
          }))
          .sort((a, b) => {
            if (a.build === 'unknown') return 1;
            if (b.build === 'unknown') return -1;
            return a.build.localeCompare(b.build);
          });
        buildCountsCache = {
          expiresAt: now + BUILD_COUNTS_CACHE_TTL_MS,
          builds,
        };
        return jsonResponse({ builds });
      }
      if (url.pathname.startsWith('/api/snapshots/random')) {
        if (request.method !== 'GET') {
          return jsonResponse({ error: 'Method not allowed.' }, 405);
        }
        const mode = url.searchParams.get('mode') ?? 'all';
        const trigger = url.searchParams.get('trigger') ?? 'all';
        const build = url.searchParams.get('build') ?? 'all';
        const excludeSnapshotIds = parseCsvInts(
          url.searchParams.get('exclude_snapshot_ids') ??
            url.searchParams.get('excludeSnapshotIds'),
        );
        const excludeSessionIds = parseCsvStrings(
          url.searchParams.get('exclude_session_ids') ??
            url.searchParams.get('excludeSessionIds'),
        );

        const hasExclusions =
          excludeSnapshotIds.length > 0 || excludeSessionIds.length > 0;

        let filter = buildSnapshotFilterSql(mode, trigger, build);
        if (excludeSnapshotIds.length > 0) {
          const placeholders = excludeSnapshotIds.map(() => '?').join(', ');
          filter = appendFilterClauseWithParams(
            filter,
            `idx.snapshot_id NOT IN (${placeholders})`,
            excludeSnapshotIds,
          );
        }
        if (excludeSessionIds.length > 0) {
          const placeholders = excludeSessionIds.map(() => '?').join(', ');
          filter = appendFilterClauseWithParams(
            filter,
            `(idx.session_id IS NULL OR idx.session_id NOT IN (${placeholders}))`,
            excludeSessionIds,
          );
        }
        const filterKey = snapshotFilterCacheKey(mode, trigger, build);
        const unlabeledFilter = appendFilterClause(
          filter,
          'COALESCE(idx.label_count, 0) = 0',
        );
        let row = await readSnapshotByFilter(
          env,
          unlabeledFilter,
          hasExclusions ? null : `${filterKey}|unlabeled`,
        );
        if (!row) {
          row = await readSnapshotByFilter(
            env,
            filter,
            hasExclusions ? null : `${filterKey}|all`,
          );
        }

        if (!row && hasExclusions) {
          const baseFilter = buildSnapshotFilterSql(mode, trigger, build);
          const baseUnlabeledFilter = appendFilterClause(
            baseFilter,
            'COALESCE(idx.label_count, 0) = 0',
          );
          row = await readSnapshotByFilter(
            env,
            baseUnlabeledFilter,
            `${filterKey}|unlabeled`,
          );
          if (!row) {
            row = await readSnapshotByFilter(
              env,
              baseFilter,
              `${filterKey}|all`,
            );
          }
        }
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
          sessionId:
            typeof row.session_id === 'string' && row.session_id.trim()
              ? row.session_id
              : null,
          batchId: asInt(row.batch_id),
          meta,
          board,
        });
      }
      if (request.method !== 'POST') {
        return jsonResponse({ error: 'Method not allowed.' }, 405);
      }
      const payload = await request.json().catch(() => null);
      if (Array.isArray(payload)) {
        return handleSnapshotsBatch(env, payload);
      }
      return handleSnapshots(env, payload);
    }

    if (url.pathname.startsWith('/api/labels')) {
      if (url.pathname.startsWith('/api/labels/progress')) {
        if (request.method !== 'GET') {
          return jsonResponse({ error: 'Method not allowed.' }, 405);
        }
        const build = asString(url.searchParams.get('build'));
        if (!build) {
          return jsonResponse({ error: 'Missing build query param.' }, 400);
        }
        const labeledBoards = await readLabeledBoardCountForBuild(env, build);
        return jsonResponse({ build, labeledBoards });
      }
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
