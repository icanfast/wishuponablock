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
  loadRecordingObject: (id: string) => Promise<TrajectorySessionV1>;
};

type AdminRecordingsServiceOptions = {
  baseUrl: string;
};

type ListPayload = {
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
