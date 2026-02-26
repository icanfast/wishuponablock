import type { TrajectorySessionV1 } from '../core/trajectoryProtocol';

export type TrajectoryUploadResponse = {
  id: string;
  mode: string;
  r2Key: string;
  samples: number;
  createdAtMs: number;
};

export type TrajectoryRecordingHttpError = Error & {
  status: number;
};

export type TrajectoryRecordingService = {
  uploadSession: (
    session: TrajectorySessionV1,
  ) => Promise<TrajectoryUploadResponse>;
};

type TrajectoryRecordingServiceOptions = {
  baseUrl: string;
};

type UploadPayload = {
  recording?: {
    id?: unknown;
    mode?: unknown;
    r2Key?: unknown;
    samples?: unknown;
    createdAtMs?: unknown;
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

const withStatusError = (
  message: string,
  status: number,
): TrajectoryRecordingHttpError => {
  const error = new Error(message) as TrajectoryRecordingHttpError;
  error.status = status;
  return error;
};

const parseErrorMessage = async (response: Response): Promise<string> => {
  const fallback = `Request failed (${response.status}).`;
  let bodyText = '';
  try {
    bodyText = await response.text();
  } catch {
    return fallback;
  }
  if (!bodyText.trim()) return fallback;
  try {
    const payload = JSON.parse(bodyText) as { error?: unknown };
    const message = asString(payload?.error);
    if (message) return message;
  } catch {
    // ignore parse errors
  }
  const detail = bodyText.trim().slice(0, 200);
  return detail ? `${fallback} ${detail}` : fallback;
};

const normalizeBaseUrl = (value: string): string =>
  value.trim().replace(/\/+$/, '');

export function createTrajectoryRecordingService(
  options: TrajectoryRecordingServiceOptions,
): TrajectoryRecordingService {
  const baseUrl = normalizeBaseUrl(options.baseUrl || '/api');
  const uploadPath = `${baseUrl}/recordings/me/trajectory`;

  return {
    uploadSession: async (session) => {
      const response = await fetch(uploadPath, {
        method: 'POST',
        credentials: 'include',
        cache: 'no-store',
        headers: {
          'content-type': 'application/json',
        },
        body: JSON.stringify(session),
      });
      if (!response.ok) {
        throw withStatusError(
          await parseErrorMessage(response),
          response.status,
        );
      }
      const body = (await response
        .json()
        .catch(() => null)) as UploadPayload | null;
      const recording = body?.recording ?? null;
      const id = asString(recording?.id) ?? session.sessionId;
      const mode = asString(recording?.mode) ?? session.modeId;
      const r2Key = asString(recording?.r2Key) ?? '';
      const samples = asInt(recording?.samples) ?? session.samples.length;
      const createdAtMs = asInt(recording?.createdAtMs) ?? Date.now();
      return {
        id,
        mode,
        r2Key,
        samples,
        createdAtMs,
      };
    },
  };
}
