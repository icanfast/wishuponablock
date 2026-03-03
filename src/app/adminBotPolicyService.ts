import type { BotPolicyArtifact } from './headlessBotService';

export type BotPolicySelector = {
  modeId: string;
  archId?: string;
  queuePolicyId?: string;
};

export type BotPolicyRecord = {
  id: string;
  modeId: string;
  archId: string;
  queuePolicyId: string;
  pipelineId: string;
  pieceSourceProfile: string;
  r2Key: string;
  version: number;
  isPinned: boolean;
  metrics: Record<string, unknown> | null;
  createdAtMs: number;
};

export type BotPolicyListPage = {
  selector: {
    modeId: string;
    archId: string;
    queuePolicyId: string;
  };
  policies: BotPolicyRecord[];
  page: {
    limit: number;
    returned: number;
    nextCursor: string | null;
  };
};

export type BotPolicyCurrentResponse = {
  selector: {
    modeId: string;
    archId: string;
    queuePolicyId: string;
  };
  current: BotPolicyRecord | null;
};

export type BotPolicyPublishPayload = {
  selector: BotPolicySelector;
  policyArtifact: BotPolicyArtifact;
  pipelineId?: string;
  pieceSourceProfile?: string;
  metrics?: Record<string, unknown> | null;
  pin?: boolean;
  setCurrent?: boolean;
};

export type BotPolicyService = {
  getCurrent: (
    selector: BotPolicySelector,
  ) => Promise<BotPolicyCurrentResponse>;
  list: (
    selector: BotPolicySelector,
    options?: { limit?: number; cursor?: string | null },
  ) => Promise<BotPolicyListPage>;
  loadObject: (
    id: string,
  ) => Promise<{ policy: BotPolicyRecord; artifact: BotPolicyArtifact }>;
  publish: (payload: BotPolicyPublishPayload) => Promise<BotPolicyRecord>;
  selectCurrent: (id: string) => Promise<BotPolicyRecord>;
  pin: (id: string) => Promise<BotPolicyRecord>;
  unpin: (id: string) => Promise<BotPolicyRecord>;
};

type BotPolicyServiceOptions = {
  baseUrl: string;
};

const asString = (value: unknown): string | null =>
  typeof value === 'string' && value.trim().length > 0 ? value.trim() : null;

const asInt = (value: unknown): number | null => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return Math.trunc(value);
  }
  if (typeof value === 'string' && value.trim().length > 0) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return Math.trunc(parsed);
  }
  return null;
};

const normalizeAxis = (value: unknown): string | null => {
  if (typeof value !== 'string') return null;
  const normalized = value.trim().toLowerCase();
  if (!normalized) return null;
  if (!/^[a-z0-9_-]{1,64}$/.test(normalized)) return null;
  return normalized;
};

const normalizeMode = (value: unknown): string => {
  const mode = normalizeAxis(value);
  if (!mode) throw new Error('Invalid mode id.');
  return mode;
};

const normalizeBaseUrl = (value: string): string =>
  value.trim().replace(/\/+$/, '');

const withStatusError = (message: string, status: number): Error => {
  const error = new Error(message) as Error & { status?: number };
  error.status = status;
  return error;
};

const parseErrorMessage = async (response: Response): Promise<string> => {
  const fallback = `Request failed (${response.status}).`;
  const payload = (await response.json().catch(() => null)) as {
    error?: unknown;
  } | null;
  return asString(payload?.error) ?? fallback;
};

const parseRecord = (row: unknown): BotPolicyRecord | null => {
  if (!row || typeof row !== 'object' || Array.isArray(row)) return null;
  const obj = row as Record<string, unknown>;
  const id = asString(obj.id);
  const modeId = normalizeAxis(obj.modeId);
  const archId = normalizeAxis(obj.archId);
  const queuePolicyId = normalizeAxis(obj.queuePolicyId);
  const pipelineId = normalizeAxis(obj.pipelineId);
  const pieceSourceProfile = normalizeAxis(obj.pieceSourceProfile);
  const r2Key = asString(obj.r2Key);
  const version = asInt(obj.version);
  const createdAtMs = asInt(obj.createdAtMs);
  if (
    !id ||
    !modeId ||
    !archId ||
    !queuePolicyId ||
    !pipelineId ||
    !pieceSourceProfile ||
    !r2Key ||
    version == null ||
    createdAtMs == null
  ) {
    return null;
  }
  const metrics =
    obj.metrics &&
    typeof obj.metrics === 'object' &&
    !Array.isArray(obj.metrics)
      ? (obj.metrics as Record<string, unknown>)
      : null;
  return {
    id,
    modeId,
    archId,
    queuePolicyId,
    pipelineId,
    pieceSourceProfile,
    r2Key,
    version,
    isPinned: obj.isPinned === true || asInt(obj.isPinned) === 1,
    metrics,
    createdAtMs,
  };
};

const withSelector = (
  path: string,
  selector: BotPolicySelector,
  baseUrl: string,
): URL => {
  const url = new URL(`${baseUrl}${path}`, window.location.origin);
  url.searchParams.set('mode', normalizeMode(selector.modeId));
  const archId = normalizeAxis(selector.archId);
  const queuePolicyId = normalizeAxis(selector.queuePolicyId);
  if (archId) url.searchParams.set('arch', archId);
  if (queuePolicyId) url.searchParams.set('queue_policy', queuePolicyId);
  return url;
};

export function createAdminBotPolicyService(
  options: BotPolicyServiceOptions,
): BotPolicyService {
  const baseUrl = normalizeBaseUrl(options.baseUrl || '/api');

  const request = async (url: URL, init?: RequestInit): Promise<Response> =>
    await fetch(url.toString(), {
      ...init,
      credentials: 'include',
      cache: 'no-store',
      headers: {
        ...(init?.headers ?? {}),
      },
    });

  const parseSelector = (
    payload: Record<string, unknown> | null | undefined,
    fallback: BotPolicySelector,
  ): BotPolicyCurrentResponse['selector'] => {
    const selector =
      payload?.selector &&
      typeof payload.selector === 'object' &&
      !Array.isArray(payload.selector)
        ? (payload.selector as Record<string, unknown>)
        : null;
    return {
      modeId: normalizeAxis(selector?.modeId) ?? normalizeMode(fallback.modeId),
      archId: normalizeAxis(selector?.archId) ?? 'full',
      queuePolicyId: normalizeAxis(selector?.queuePolicyId) ?? 'default',
    };
  };

  return {
    getCurrent: async (selector) => {
      const url = withSelector(
        '/admin/bot/policies/current',
        selector,
        baseUrl,
      );
      const response = await request(url, { method: 'GET' });
      if (!response.ok) {
        throw withStatusError(
          await parseErrorMessage(response),
          response.status,
        );
      }
      const payload = (await response.json().catch(() => null)) as Record<
        string,
        unknown
      > | null;
      return {
        selector: parseSelector(payload, selector),
        current: parseRecord(payload?.current) ?? null,
      };
    },
    list: async (selector, optionsArg) => {
      const url = withSelector('/admin/bot/policies/list', selector, baseUrl);
      if (optionsArg?.limit != null && Number.isFinite(optionsArg.limit)) {
        url.searchParams.set('limit', String(Math.trunc(optionsArg.limit)));
      }
      if (optionsArg?.cursor) {
        url.searchParams.set('cursor', optionsArg.cursor);
      }
      const response = await request(url, { method: 'GET' });
      if (!response.ok) {
        throw withStatusError(
          await parseErrorMessage(response),
          response.status,
        );
      }
      const payload = (await response.json().catch(() => null)) as Record<
        string,
        unknown
      > | null;
      const rawPolicies = Array.isArray(payload?.policies)
        ? payload.policies
        : [];
      const policies: BotPolicyRecord[] = [];
      for (const raw of rawPolicies) {
        const parsed = parseRecord(raw);
        if (parsed) policies.push(parsed);
      }
      const pageObj =
        payload?.page && typeof payload.page === 'object'
          ? (payload.page as Record<string, unknown>)
          : null;
      return {
        selector: parseSelector(payload, selector),
        policies,
        page: {
          limit: asInt(pageObj?.limit) ?? 40,
          returned: asInt(pageObj?.returned) ?? policies.length,
          nextCursor: asString(pageObj?.nextCursor),
        },
      };
    },
    loadObject: async (id) => {
      const normalizedId = asString(id);
      if (!normalizedId) {
        throw new Error('Policy id is required.');
      }
      const url = new URL(
        `${baseUrl}/admin/bot/policies/object`,
        window.location.origin,
      );
      url.searchParams.set('id', normalizedId);
      const response = await request(url, { method: 'GET' });
      if (!response.ok) {
        throw withStatusError(
          await parseErrorMessage(response),
          response.status,
        );
      }
      const artifactPayload = (await response
        .json()
        .catch(() => null)) as unknown;
      if (
        !artifactPayload ||
        typeof artifactPayload !== 'object' ||
        Array.isArray(artifactPayload)
      ) {
        throw new Error('Invalid bot policy object payload.');
      }
      const policy: BotPolicyRecord = {
        id:
          asString(response.headers.get('x-wub-bot-policy-id')) ?? normalizedId,
        modeId:
          normalizeAxis(response.headers.get('x-wub-bot-mode')) ?? 'practice',
        archId: normalizeAxis(response.headers.get('x-wub-bot-arch')) ?? 'full',
        queuePolicyId:
          normalizeAxis(response.headers.get('x-wub-bot-queue-policy')) ??
          'default',
        pipelineId:
          normalizeAxis(
            (artifactPayload as Record<string, unknown>).pipelineId,
          ) ?? 'bot_reinforce_v2',
        pieceSourceProfile:
          normalizeAxis(
            (artifactPayload as Record<string, unknown>).pieceSourceProfile,
          ) ?? 'bag7',
        r2Key: asString(response.headers.get('x-wub-bot-r2-key')) ?? '',
        version: asInt(response.headers.get('x-wub-bot-version')) ?? 1,
        isPinned: false,
        metrics: null,
        createdAtMs:
          asInt(response.headers.get('x-wub-bot-created-at-ms')) ?? Date.now(),
      };
      const artifact = artifactPayload as BotPolicyArtifact;
      return {
        policy,
        artifact,
      };
    },
    publish: async (payload) => {
      const response = await request(
        new URL(
          `${baseUrl}/admin/bot/policies/publish`,
          window.location.origin,
        ),
        {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({
            modeId: normalizeMode(payload.selector.modeId),
            archId: normalizeAxis(payload.selector.archId) ?? 'full',
            queuePolicyId:
              normalizeAxis(payload.selector.queuePolicyId) ?? 'default',
            pipelineId: normalizeAxis(payload.pipelineId) ?? 'bot_reinforce_v2',
            pieceSourceProfile:
              normalizeAxis(payload.pieceSourceProfile) ?? 'bag7',
            policyArtifact: payload.policyArtifact,
            metrics: payload.metrics ?? null,
            pin: payload.pin === true,
            setCurrent: payload.setCurrent === true,
          }),
        },
      );
      if (!response.ok) {
        throw withStatusError(
          await parseErrorMessage(response),
          response.status,
        );
      }
      const body = (await response.json().catch(() => null)) as Record<
        string,
        unknown
      > | null;
      const record = parseRecord(body?.policy);
      if (!record) {
        throw new Error('Invalid bot policy response.');
      }
      return record;
    },
    selectCurrent: async (id) => {
      const normalizedId = asString(id);
      if (!normalizedId) throw new Error('Policy id is required.');
      const response = await request(
        new URL(
          `${baseUrl}/admin/bot/policies/select-current`,
          window.location.origin,
        ),
        {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ id: normalizedId }),
        },
      );
      if (!response.ok) {
        throw withStatusError(
          await parseErrorMessage(response),
          response.status,
        );
      }
      const body = (await response.json().catch(() => null)) as Record<
        string,
        unknown
      > | null;
      const record = parseRecord(body?.policy);
      if (!record) throw new Error('Invalid bot policy response.');
      return record;
    },
    pin: async (id) => {
      const normalizedId = asString(id);
      if (!normalizedId) throw new Error('Policy id is required.');
      const response = await request(
        new URL(`${baseUrl}/admin/bot/policies/pin`, window.location.origin),
        {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ id: normalizedId }),
        },
      );
      if (!response.ok) {
        throw withStatusError(
          await parseErrorMessage(response),
          response.status,
        );
      }
      const body = (await response.json().catch(() => null)) as Record<
        string,
        unknown
      > | null;
      const record = parseRecord(body?.policy);
      if (!record) throw new Error('Invalid bot policy response.');
      return record;
    },
    unpin: async (id) => {
      const normalizedId = asString(id);
      if (!normalizedId) throw new Error('Policy id is required.');
      const response = await request(
        new URL(`${baseUrl}/admin/bot/policies/unpin`, window.location.origin),
        {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ id: normalizedId }),
        },
      );
      if (!response.ok) {
        throw withStatusError(
          await parseErrorMessage(response),
          response.status,
        );
      }
      const body = (await response.json().catch(() => null)) as Record<
        string,
        unknown
      > | null;
      const record = parseRecord(body?.policy);
      if (!record) throw new Error('Invalid bot policy response.');
      return record;
    },
  };
}
