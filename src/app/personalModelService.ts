export type PersonalModelDownload = {
  mode: string;
  arch: string | null;
  rewardProfileId: string | null;
  queuePolicyId: string | null;
  versionId: string | null;
  version: number | null;
  sizeBytes: number;
  updatedAtMs: number | null;
  sha256: string | null;
  bytes: ArrayBuffer;
};

export type PersonalModelUpload = {
  mode: string;
  arch: string | null;
  rewardProfileId: string | null;
  queuePolicyId: string | null;
  versionId: string | null;
  version: number | null;
  sizeBytes: number | null;
  updatedAtMs: number | null;
  sha256: string | null;
  baseGlobalModelId: string | null;
};

export type PersonalGlobalModel = {
  id: string;
  mode: string;
  arch: string | null;
  rewardProfileId: string | null;
  queuePolicyId: string | null;
  pipelineId: string | null;
  label: string | null;
  isDefault: boolean;
  sha256: string | null;
  sizeBytes: number | null;
  createdAtMs: number | null;
  updatedAtMs: number | null;
};

export type PersonalModelResetResult = {
  model: PersonalModelUpload;
  globalModelId: string | null;
  globalModelLabel: string | null;
};

export type PersonalModelHttpError = Error & {
  status: number;
};

export type PersonalModelService = {
  downloadCurrent: (mode: string) => Promise<PersonalModelDownload>;
  uploadCurrent: (
    mode: string,
    payload: ArrayBuffer | ArrayBufferView,
  ) => Promise<PersonalModelUpload>;
  listGlobal: (mode: string) => Promise<PersonalGlobalModel[]>;
  resetCurrentFromGlobal: (
    mode: string,
    globalModelId?: string | null,
  ) => Promise<PersonalModelResetResult>;
};

type PersonalModelServiceOptions = {
  baseUrl: string;
};

type UploadPayload = {
  model?: {
    mode?: unknown;
    arch?: unknown;
    rewardProfileId?: unknown;
    queuePolicyId?: unknown;
    versionId?: unknown;
    version?: unknown;
    sizeBytes?: unknown;
    updatedAtMs?: unknown;
    sha256?: unknown;
    baseGlobalModelId?: unknown;
  } | null;
};

type GlobalListPayload = {
  models?: Array<Record<string, unknown>> | null;
};

type ResetPayload = {
  model?: UploadPayload['model'];
  globalModel?: {
    id?: unknown;
    label?: unknown;
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
): PersonalModelHttpError => {
  const error = new Error(message) as PersonalModelHttpError;
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
    // Ignore parse errors; we will return a generic detail.
  }
  const detail = bodyText.trim().slice(0, 200);
  return detail ? `${fallback} ${detail}` : fallback;
};

const normalizeBaseUrl = (value: string): string =>
  value.trim().replace(/\/+$/, '');

const normalizeMode = (value: string): string => {
  const mode = value.trim().toLowerCase();
  if (!/^[a-z0-9_-]{1,64}$/.test(mode)) {
    throw new Error('Invalid model mode.');
  }
  return mode;
};

const toByteView = (payload: ArrayBuffer | ArrayBufferView): Uint8Array => {
  if (payload instanceof ArrayBuffer) {
    return new Uint8Array(payload);
  }
  return new Uint8Array(payload.buffer, payload.byteOffset, payload.byteLength);
};

export function createPersonalModelService(
  options: PersonalModelServiceOptions,
): PersonalModelService {
  const baseUrl = normalizeBaseUrl(options.baseUrl || '/api');

  const buildPath = (mode: string): string =>
    `${baseUrl}/models/me/current?mode=${encodeURIComponent(normalizeMode(mode))}`;
  const buildGlobalListPath = (mode: string): string =>
    `${baseUrl}/models/global/list?mode=${encodeURIComponent(normalizeMode(mode))}`;
  const buildResetPath = (mode: string): string =>
    `${baseUrl}/models/me/reset?mode=${encodeURIComponent(normalizeMode(mode))}`;

  return {
    downloadCurrent: async (mode) => {
      const response = await fetch(buildPath(mode), {
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
      const bytes = await response.arrayBuffer();
      const modeHeader = asString(response.headers.get('x-wub-model-mode'));
      const arch = asString(response.headers.get('x-wub-model-arch'));
      const rewardProfileId = asString(
        response.headers.get('x-wub-model-reward-profile'),
      );
      const queuePolicyId = asString(
        response.headers.get('x-wub-model-queue-policy'),
      );
      const versionId = asString(
        response.headers.get('x-wub-model-version-id'),
      );
      const version = asInt(response.headers.get('x-wub-model-version'));
      const headerSize = asInt(response.headers.get('x-wub-model-size'));
      const updatedAtMs = asInt(
        response.headers.get('x-wub-model-updated-at-ms'),
      );
      const sha256 = asString(response.headers.get('x-wub-model-sha256'));
      return {
        mode: modeHeader ?? normalizeMode(mode),
        arch,
        rewardProfileId,
        queuePolicyId,
        versionId,
        version,
        sizeBytes: headerSize ?? bytes.byteLength,
        updatedAtMs,
        sha256,
        bytes,
      };
    },
    uploadCurrent: async (mode, payload) => {
      const bytes = toByteView(payload);
      const response = await fetch(buildPath(mode), {
        method: 'PUT',
        credentials: 'include',
        cache: 'no-store',
        headers: {
          'content-type': 'application/octet-stream',
        },
        body: bytes,
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
      const model = body?.model ?? null;
      return {
        mode: asString(model?.mode) ?? normalizeMode(mode),
        arch: asString(model?.arch),
        rewardProfileId: asString(model?.rewardProfileId),
        queuePolicyId: asString(model?.queuePolicyId),
        versionId: asString(model?.versionId),
        version: asInt(model?.version),
        sizeBytes: asInt(model?.sizeBytes),
        updatedAtMs: asInt(model?.updatedAtMs),
        sha256: asString(model?.sha256),
        baseGlobalModelId: asString(model?.baseGlobalModelId),
      };
    },
    listGlobal: async (mode) => {
      const response = await fetch(buildGlobalListPath(mode), {
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
      const body = (await response
        .json()
        .catch(() => null)) as GlobalListPayload | null;
      const rows = Array.isArray(body?.models) ? body.models : [];
      const models: PersonalGlobalModel[] = [];
      for (const row of rows) {
        const id = asString(row.id);
        const resolvedMode = asString(row.mode);
        if (!id || !resolvedMode) continue;
        models.push({
          id,
          mode: resolvedMode,
          arch: asString(row.arch),
          rewardProfileId: asString(row.rewardProfileId),
          queuePolicyId: asString(row.queuePolicyId),
          pipelineId: asString(row.pipelineId),
          label: asString(row.label),
          isDefault: asInt(row.isDefault) === 1 || row.isDefault === true,
          sha256: asString(row.sha256),
          sizeBytes: asInt(row.sizeBytes),
          createdAtMs: asInt(row.createdAtMs),
          updatedAtMs: asInt(row.updatedAtMs),
        });
      }
      return models;
    },
    resetCurrentFromGlobal: async (mode, globalModelId) => {
      const normalizedGlobalModelId = asString(globalModelId);
      const response = await fetch(buildResetPath(mode), {
        method: 'POST',
        credentials: 'include',
        cache: 'no-store',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(
          normalizedGlobalModelId
            ? { globalModelId: normalizedGlobalModelId }
            : {},
        ),
      });
      if (!response.ok) {
        throw withStatusError(
          await parseErrorMessage(response),
          response.status,
        );
      }
      const body = (await response
        .json()
        .catch(() => null)) as ResetPayload | null;
      const model = body?.model ?? null;
      return {
        model: {
          mode: asString(model?.mode) ?? normalizeMode(mode),
          arch: asString(model?.arch),
          rewardProfileId: asString(model?.rewardProfileId),
          queuePolicyId: asString(model?.queuePolicyId),
          versionId: asString(model?.versionId),
          version: asInt(model?.version),
          sizeBytes: asInt(model?.sizeBytes),
          updatedAtMs: asInt(model?.updatedAtMs),
          sha256: asString(model?.sha256),
          baseGlobalModelId: asString(model?.baseGlobalModelId),
        },
        globalModelId: asString(body?.globalModel?.id),
        globalModelLabel: asString(body?.globalModel?.label),
      };
    },
  };
}
