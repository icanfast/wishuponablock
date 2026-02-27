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

export type PersonalModelSelector = {
  arch?: string;
  rewardProfileId?: string;
  queuePolicyId?: string;
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

export type PersonalGlobalModelDownload = {
  id: string;
  mode: string;
  arch: string | null;
  rewardProfileId: string | null;
  queuePolicyId: string | null;
  pipelineId: string | null;
  label: string | null;
  sha256: string | null;
  sizeBytes: number;
  createdAtMs: number | null;
  updatedAtMs: number | null;
  bytes: ArrayBuffer;
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
  downloadCurrent: (
    mode: string,
    selector?: PersonalModelSelector,
  ) => Promise<PersonalModelDownload>;
  uploadCurrent: (
    mode: string,
    payload: ArrayBuffer | ArrayBufferView,
    selector?: PersonalModelSelector,
  ) => Promise<PersonalModelUpload>;
  listGlobal: (
    mode: string,
    selector?: PersonalModelSelector,
  ) => Promise<PersonalGlobalModel[]>;
  downloadGlobalCurrent: (
    mode: string,
    selector?: PersonalModelSelector,
    globalModelId?: string | null,
  ) => Promise<PersonalGlobalModelDownload>;
  resetCurrentFromGlobal: (
    mode: string,
    globalModelId?: string | null,
    selector?: PersonalModelSelector,
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

const normalizeAxis = (value: unknown): string | null => {
  if (typeof value !== 'string') return null;
  const normalized = value.trim().toLowerCase();
  if (!normalized) return null;
  if (!/^[a-z0-9_-]{1,64}$/.test(normalized)) return null;
  return normalized;
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

  const withSelector = (
    path: string,
    mode: string,
    selector?: PersonalModelSelector,
  ): string => {
    const url = new URL(`${baseUrl}${path}`, window.location.origin);
    url.searchParams.set('mode', normalizeMode(mode));
    const arch = normalizeAxis(selector?.arch);
    const rewardProfileId = normalizeAxis(selector?.rewardProfileId);
    const queuePolicyId = normalizeAxis(selector?.queuePolicyId);
    if (arch) url.searchParams.set('arch', arch);
    if (rewardProfileId) {
      url.searchParams.set('reward_profile', rewardProfileId);
    }
    if (queuePolicyId) {
      url.searchParams.set('queue_policy', queuePolicyId);
    }
    return url.toString();
  };

  return {
    downloadCurrent: async (mode, selector) => {
      const response = await fetch(
        withSelector('/models/me/current', mode, selector),
        {
          method: 'GET',
          credentials: 'include',
          cache: 'no-store',
        },
      );
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
    uploadCurrent: async (mode, payload, selector) => {
      const bytes = toByteView(payload);
      const response = await fetch(
        withSelector('/models/me/current', mode, selector),
        {
          method: 'PUT',
          credentials: 'include',
          cache: 'no-store',
          headers: {
            'content-type': 'application/octet-stream',
          },
          body: bytes,
        },
      );
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
    listGlobal: async (mode, selector) => {
      const response = await fetch(
        withSelector('/models/global/list', mode, selector),
        {
          method: 'GET',
          credentials: 'include',
          cache: 'no-store',
        },
      );
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
    downloadGlobalCurrent: async (mode, selector, globalModelId) => {
      const normalizedGlobalModelId = asString(globalModelId);
      const url = new URL(
        withSelector('/models/global/current', mode, selector),
        window.location.origin,
      );
      if (normalizedGlobalModelId) {
        url.searchParams.set('global_model_id', normalizedGlobalModelId);
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
      const bytes = await response.arrayBuffer();
      const id = asString(response.headers.get('x-wub-global-model-id'));
      if (!id) {
        throw new Error('Global model response is missing id metadata.');
      }
      const modeHeader = asString(
        response.headers.get('x-wub-global-model-mode'),
      );
      const arch = asString(response.headers.get('x-wub-global-model-arch'));
      const rewardProfileId = asString(
        response.headers.get('x-wub-global-model-reward-profile'),
      );
      const queuePolicyId = asString(
        response.headers.get('x-wub-global-model-queue-policy'),
      );
      const pipelineId = asString(
        response.headers.get('x-wub-global-model-pipeline'),
      );
      const label = asString(response.headers.get('x-wub-global-model-label'));
      const sha256 = asString(
        response.headers.get('x-wub-global-model-sha256'),
      );
      const sizeHeader = asInt(response.headers.get('x-wub-global-model-size'));
      const createdAtMs = asInt(
        response.headers.get('x-wub-global-model-created-at-ms'),
      );
      const updatedAtMs = asInt(
        response.headers.get('x-wub-global-model-updated-at-ms'),
      );
      return {
        id,
        mode: modeHeader ?? normalizeMode(mode),
        arch,
        rewardProfileId,
        queuePolicyId,
        pipelineId,
        label,
        sha256,
        sizeBytes: sizeHeader ?? bytes.byteLength,
        createdAtMs,
        updatedAtMs,
        bytes,
      };
    },
    resetCurrentFromGlobal: async (mode, globalModelId, selector) => {
      const normalizedGlobalModelId = asString(globalModelId);
      const response = await fetch(
        withSelector('/models/me/reset', mode, selector),
        {
          method: 'POST',
          credentials: 'include',
          cache: 'no-store',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify(
            normalizedGlobalModelId
              ? { globalModelId: normalizedGlobalModelId }
              : {},
          ),
        },
      );
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
