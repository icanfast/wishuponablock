import {
  MAX_TRAJECTORY_SAMPLES_PER_SESSION,
  parseTrajectorySessionV1,
  type TrajectorySessionV1,
} from '../core/trajectoryProtocol';

export type AdminRecordingSummary = {
  id: string;
  userId: string;
  mode: string;
  buildVersion: string;
  r2Key: string;
  startedAtMs: number;
  endedAtMs: number;
  durationMs: number;
  samples: number;
  createdAtMs: number;
  meta: Record<string, unknown> | null;
};

export type AdminRecordingsPage = {
  recordings: AdminRecordingSummary[];
  page: {
    limit: number;
    nextCursor: string | null;
    returned: number;
  };
};

export type AdminRecordingsManifestQuery = {
  mode?: string;
  build?: string;
  userId?: string;
  arch?: string;
  rewardProfileId?: string;
  queuePolicyId?: string;
  pipelineId?: string;
  actorType?: 'human' | 'bot';
  minSamples?: number;
  limit?: number;
  cursor?: string | null;
  startedFromMs?: number;
  startedToMs?: number;
};

export type AdminRecordingsManifestPage = {
  selector: {
    mode?: string;
    build?: string;
    userId?: string;
    arch?: string;
    rewardProfileId?: string;
    queuePolicyId?: string;
    pipelineId?: string;
    actorType?: 'human' | 'bot';
    minSamples?: number;
    startedFromMs?: number;
    startedToMs?: number;
  };
  recordings: AdminRecordingSummary[];
  page: {
    limit: number;
    nextCursor: string | null;
    returned: number;
  };
};

export type AdminRecordingsListQuery = {
  mode?: string;
  build?: string;
  userId?: string;
  limit?: number;
  cursor?: string | null;
  startedFromMs?: number;
  startedToMs?: number;
};

export type AdminRecordingsService = {
  listRecordings: (
    query?: AdminRecordingsListQuery,
  ) => Promise<AdminRecordingsPage>;
  listTrainingManifest: (
    query?: AdminRecordingsManifestQuery,
  ) => Promise<AdminRecordingsManifestPage>;
  loadRecordingObject: (id: string) => Promise<TrajectorySessionV1>;
};

type AdminRecordingsServiceOptions = {
  baseUrl: string;
};

type ListPayload = {
  selector?: Record<string, unknown> | null;
  recordings?: Array<Record<string, unknown>> | null;
  page?: {
    limit?: unknown;
    nextCursor?: unknown;
    returned?: unknown;
  } | null;
};

const asString = (value: unknown): string | null =>
  typeof value === 'string' && value.trim().length > 0 ? value.trim() : null;

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

const asObject = (value: unknown): Record<string, unknown> | null => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  return value as Record<string, unknown>;
};

const parseErrorMessage = async (response: Response): Promise<string> => {
  const fallback = `Request failed (${response.status}).`;
  const payload = (await response.json().catch(() => null)) as {
    error?: unknown;
  } | null;
  return asString(payload?.error) ?? fallback;
};

const withStatusError = (message: string, status: number): Error => {
  const error = new Error(message) as Error & { status?: number };
  error.status = status;
  return error;
};

const normalizeBaseUrl = (value: string): string =>
  value.trim().replace(/\/+$/, '');

const toRecordingSummary = (
  row: Record<string, unknown>,
): AdminRecordingSummary | null => {
  const id = asString(row.id);
  const userId = asString(row.userId);
  const mode = asString(row.mode);
  const buildVersion = asString(row.buildVersion);
  const r2Key = asString(row.r2Key);
  const startedAtMs = asInt(row.startedAtMs);
  const endedAtMs = asInt(row.endedAtMs);
  const durationMs = asInt(row.durationMs);
  const samples = asInt(row.samples);
  const createdAtMs = asInt(row.createdAtMs);
  if (
    !id ||
    !userId ||
    !mode ||
    !buildVersion ||
    !r2Key ||
    startedAtMs == null ||
    endedAtMs == null ||
    durationMs == null ||
    samples == null ||
    createdAtMs == null
  ) {
    return null;
  }
  return {
    id,
    userId,
    mode,
    buildVersion,
    r2Key,
    startedAtMs,
    endedAtMs,
    durationMs,
    samples,
    createdAtMs,
    meta: asObject(row.meta),
  };
};

export function createAdminRecordingsService(
  options: AdminRecordingsServiceOptions,
): AdminRecordingsService {
  const baseUrl = normalizeBaseUrl(options.baseUrl || '/api');
  const listPath = `${baseUrl}/admin/recordings/index`;
  const manifestPath = `${baseUrl}/admin/recordings/export-manifest`;
  const objectPath = `${baseUrl}/admin/recordings/object`;

  return {
    listRecordings: async (query) => {
      const url = new URL(listPath, window.location.origin);
      if (query?.mode) url.searchParams.set('mode', query.mode);
      if (query?.build) url.searchParams.set('build', query.build);
      if (query?.userId) url.searchParams.set('user_id', query.userId);
      if (query?.limit != null && Number.isFinite(query.limit)) {
        url.searchParams.set('limit', String(Math.trunc(query.limit)));
      }
      if (query?.cursor) {
        url.searchParams.set('cursor', query.cursor);
      }
      if (
        query?.startedFromMs != null &&
        Number.isFinite(query.startedFromMs)
      ) {
        url.searchParams.set(
          'started_from_ms',
          String(Math.trunc(query.startedFromMs)),
        );
      }
      if (query?.startedToMs != null && Number.isFinite(query.startedToMs)) {
        url.searchParams.set(
          'started_to_ms',
          String(Math.trunc(query.startedToMs)),
        );
      }
      const response = await fetch(url.toString(), {
        method: 'GET',
        credentials: 'include',
        cache: 'no-store',
      });
      if (!response.ok) {
        throw withStatusError(
          await parseErrorMessage(response),
          response.status,
        );
      }
      const payload = (await response
        .json()
        .catch(() => null)) as ListPayload | null;
      const rawRows = Array.isArray(payload?.recordings)
        ? payload.recordings
        : [];
      const recordings: AdminRecordingSummary[] = [];
      for (const row of rawRows) {
        const parsed = toRecordingSummary(row);
        if (parsed) recordings.push(parsed);
      }
      return {
        recordings,
        page: {
          limit: asInt(payload?.page?.limit) ?? 50,
          nextCursor: asString(payload?.page?.nextCursor),
          returned:
            asInt(payload?.page?.returned) ??
            asInt(payload?.recordings?.length) ??
            recordings.length,
        },
      };
    },
    listTrainingManifest: async (query) => {
      const url = new URL(manifestPath, window.location.origin);
      if (query?.mode) url.searchParams.set('mode', query.mode);
      if (query?.build) url.searchParams.set('build', query.build);
      if (query?.userId) url.searchParams.set('user_id', query.userId);
      if (query?.arch) url.searchParams.set('arch', query.arch);
      if (query?.rewardProfileId) {
        url.searchParams.set('reward_profile', query.rewardProfileId);
      }
      if (query?.queuePolicyId) {
        url.searchParams.set('queue_policy', query.queuePolicyId);
      }
      if (query?.pipelineId) {
        url.searchParams.set('pipeline_id', query.pipelineId);
      }
      if (query?.actorType) {
        url.searchParams.set('actor_type', query.actorType);
      }
      if (query?.minSamples != null && Number.isFinite(query.minSamples)) {
        url.searchParams.set(
          'min_samples',
          String(Math.trunc(query.minSamples)),
        );
      }
      if (query?.limit != null && Number.isFinite(query.limit)) {
        url.searchParams.set('limit', String(Math.trunc(query.limit)));
      }
      if (query?.cursor) {
        url.searchParams.set('cursor', query.cursor);
      }
      if (
        query?.startedFromMs != null &&
        Number.isFinite(query.startedFromMs)
      ) {
        url.searchParams.set(
          'started_from_ms',
          String(Math.trunc(query.startedFromMs)),
        );
      }
      if (query?.startedToMs != null && Number.isFinite(query.startedToMs)) {
        url.searchParams.set(
          'started_to_ms',
          String(Math.trunc(query.startedToMs)),
        );
      }
      const response = await fetch(url.toString(), {
        method: 'GET',
        credentials: 'include',
        cache: 'no-store',
      });
      if (!response.ok) {
        throw withStatusError(
          await parseErrorMessage(response),
          response.status,
        );
      }
      const payload = (await response
        .json()
        .catch(() => null)) as ListPayload | null;
      const rawRows = Array.isArray(payload?.recordings)
        ? payload.recordings
        : [];
      const recordings: AdminRecordingSummary[] = [];
      for (const row of rawRows) {
        const parsed = toRecordingSummary(row);
        if (parsed) recordings.push(parsed);
      }
      const selector = asObject(payload?.selector);
      const actorTypeRaw = asString(selector?.actorType);
      return {
        selector: {
          mode: asString(selector?.mode) ?? undefined,
          build: asString(selector?.build) ?? undefined,
          userId: asString(selector?.userId) ?? undefined,
          arch: asString(selector?.arch) ?? undefined,
          rewardProfileId: asString(selector?.rewardProfileId) ?? undefined,
          queuePolicyId: asString(selector?.queuePolicyId) ?? undefined,
          pipelineId: asString(selector?.pipelineId) ?? undefined,
          actorType:
            actorTypeRaw === 'human' || actorTypeRaw === 'bot'
              ? actorTypeRaw
              : undefined,
          minSamples: asInt(selector?.minSamples) ?? undefined,
          startedFromMs: asInt(selector?.startedFromMs) ?? undefined,
          startedToMs: asInt(selector?.startedToMs) ?? undefined,
        },
        recordings,
        page: {
          limit: asInt(payload?.page?.limit) ?? 100,
          nextCursor: asString(payload?.page?.nextCursor),
          returned:
            asInt(payload?.page?.returned) ??
            asInt(payload?.recordings?.length) ??
            recordings.length,
        },
      };
    },
    loadRecordingObject: async (id) => {
      const normalizedId = asString(id);
      if (!normalizedId) {
        throw new Error('Recording id is required.');
      }
      const url = new URL(objectPath, window.location.origin);
      url.searchParams.set('id', normalizedId);
      const response = await fetch(url.toString(), {
        method: 'GET',
        credentials: 'include',
        cache: 'no-store',
      });
      if (!response.ok) {
        throw withStatusError(
          await parseErrorMessage(response),
          response.status,
        );
      }
      const payload = (await response.json().catch(() => null)) as unknown;
      const parsed = parseTrajectorySessionV1(payload, {
        maxSamples: MAX_TRAJECTORY_SAMPLES_PER_SESSION,
      });
      if (!parsed.ok) {
        throw new Error(`Invalid trajectory payload: ${parsed.error}`);
      }
      return parsed.value;
    },
  };
}
