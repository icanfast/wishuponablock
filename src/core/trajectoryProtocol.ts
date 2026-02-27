import { PIECES, type PieceKind } from './types';

export const TRAJECTORY_SESSION_SCHEMA_V1 =
  'wishuponablock.trajectory_session.v1' as const;
export const MIN_TRAJECTORY_SAMPLES_PER_SESSION = 8;
export const MAX_TRAJECTORY_SAMPLES_PER_SESSION = 5000;
export const MAX_TRAJECTORY_ROWS = 40;
export const MAX_TRAJECTORY_COLS = 20;

export type TrajectorySessionMetaV1 = {
  outcome?: string;
  modelSource?: string;
  generatorType?: string;
  modelMode?: string;
  modelVersion?: number | null;
  channel?: string;
  modelArchId?: string;
  rewardProfileId?: string;
  queuePolicyId?: string;
  rewardPolicy?: string;
  rewardPolicyId?: string;
  rewardKind?: string;
  rewardGamma?: number;
  pipelineId?: string;
  pipelineMode?: string;
  modelArch?: string;
  actorType?: 'human' | 'bot';
  actorPolicyId?: string;
  trainingIntent?: string;
  pieceSourceProfile?: string;
};

export type TrajectoryReplayStepV1 = {
  lockPiece: PieceKind;
  lockRotation: number;
  lockX: number;
  lockY: number;
  holdUsed: boolean;
  gameTimeMs: number;
  totalLinesCleared: number;
  score: number;
};

export type TrajectorySessionSampleV1 = {
  id: string;
  createdAtMs: number;
  deliberationMs: number | null;
  boardOccupancy: number[][];
  hold: PieceKind | null;
  action: PieceKind;
  actionIndex: number;
  pieces: PieceKind[];
  logits: number[];
  probabilities: number[];
  inferenceMs: number;
  samplingMs: number;
  totalDecisionMs: number;
  reward: number | null;
  replay?: TrajectoryReplayStepV1;
};

export type TrajectorySessionV1 = {
  schema: typeof TRAJECTORY_SESSION_SCHEMA_V1;
  sessionId: string;
  modeId: string;
  buildVersion: string;
  startedAtMs: number;
  endedAtMs: number;
  durationMs: number;
  samples: TrajectorySessionSampleV1[];
  meta: TrajectorySessionMetaV1 | null;
};

export type ParseTrajectorySessionResult =
  | {
      ok: true;
      value: TrajectorySessionV1;
    }
  | {
      ok: false;
      error: string;
    };

type ParseOptions = {
  minSamples?: number;
  maxSamples?: number;
  maxRows?: number;
  maxCols?: number;
};

const PIECE_SET = new Set<string>(PIECES);

const asObject = (value: unknown): Record<string, unknown> | null => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  return value as Record<string, unknown>;
};

const asString = (
  value: unknown,
  minLength = 1,
  maxLength = 256,
): string | null => {
  if (typeof value !== 'string') return null;
  const trimmed = value.trim();
  if (trimmed.length < minLength || trimmed.length > maxLength) return null;
  return trimmed;
};

const asInt = (
  value: unknown,
  options: { min?: number; max?: number } = {},
): number | null => {
  let parsed: number | null = null;
  if (typeof value === 'number' && Number.isFinite(value)) {
    parsed = Math.trunc(value);
  } else if (typeof value === 'string' && value.trim().length > 0) {
    const maybe = Number(value);
    if (Number.isFinite(maybe)) parsed = Math.trunc(maybe);
  }
  if (parsed == null) return null;
  if (options.min != null && parsed < options.min) return null;
  if (options.max != null && parsed > options.max) return null;
  return parsed;
};

const asFiniteNumber = (
  value: unknown,
  options: { min?: number; max?: number } = {},
): number | null => {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null;
  if (options.min != null && value < options.min) return null;
  if (options.max != null && value > options.max) return null;
  return value;
};

const asBoolean = (value: unknown): boolean | null => {
  if (typeof value !== 'boolean') return null;
  return value;
};

const asPiece = (value: unknown): PieceKind | null => {
  if (typeof value !== 'string') return null;
  const piece = value.trim();
  if (!PIECE_SET.has(piece)) return null;
  return piece as PieceKind;
};

const parseBoardOccupancy = (
  value: unknown,
  maxRows: number,
  maxCols: number,
): number[][] | null => {
  if (!Array.isArray(value) || value.length === 0 || value.length > maxRows) {
    return null;
  }
  const rows: number[][] = [];
  for (const row of value) {
    if (!Array.isArray(row) || row.length === 0 || row.length > maxCols) {
      return null;
    }
    const nextRow: number[] = [];
    for (const cell of row) {
      const n = asFiniteNumber(cell);
      if (n == null) return null;
      nextRow.push(n > 0 ? 1 : 0);
    }
    rows.push(nextRow);
  }
  return rows;
};

const parsePieceArray = (
  value: unknown,
  minLength: number,
  maxLength: number,
): PieceKind[] | null => {
  if (!Array.isArray(value)) return null;
  if (value.length < minLength || value.length > maxLength) return null;
  const pieces: PieceKind[] = [];
  for (const entry of value) {
    const piece = asPiece(entry);
    if (!piece) return null;
    pieces.push(piece);
  }
  return pieces;
};

const parseNumberArray = (
  value: unknown,
  expectedLength: number,
): number[] | null => {
  if (!Array.isArray(value) || value.length !== expectedLength) return null;
  const numbers: number[] = [];
  for (const entry of value) {
    const n = asFiniteNumber(entry);
    if (n == null) return null;
    numbers.push(n);
  }
  return numbers;
};

const parseMeta = (value: unknown): TrajectorySessionMetaV1 | null => {
  if (value == null) return null;
  const obj = asObject(value);
  if (!obj) return null;
  const meta: TrajectorySessionMetaV1 = {};
  const outcome = asString(obj.outcome, 1, 64);
  const modelSource = asString(obj.modelSource, 1, 64);
  const generatorType = asString(obj.generatorType, 1, 64);
  const modelMode = asString(obj.modelMode, 1, 64);
  const channel = asString(obj.channel, 1, 64);
  const modelArchId = asString(obj.modelArchId, 1, 64);
  const rewardProfileId = asString(obj.rewardProfileId, 1, 64);
  const queuePolicyId = asString(obj.queuePolicyId, 1, 64);
  const rewardPolicy = asString(obj.rewardPolicy, 1, 64);
  const rewardPolicyId = asString(obj.rewardPolicyId, 1, 64);
  const rewardKind = asString(obj.rewardKind, 1, 64);
  const pipelineId = asString(obj.pipelineId, 1, 64);
  const pipelineMode = asString(obj.pipelineMode, 1, 64);
  const modelArch = asString(obj.modelArch, 1, 128);
  const actorTypeRaw = asString(obj.actorType, 1, 16);
  const actorType =
    actorTypeRaw === 'human' || actorTypeRaw === 'bot' ? actorTypeRaw : null;
  const actorPolicyId = asString(obj.actorPolicyId, 1, 128);
  const trainingIntent = asString(obj.trainingIntent, 1, 128);
  const pieceSourceProfile = asString(obj.pieceSourceProfile, 1, 64);
  const rewardGamma =
    obj.rewardGamma == null
      ? null
      : asFiniteNumber(obj.rewardGamma, { min: 0, max: 1 });
  const modelVersion =
    obj.modelVersion == null
      ? null
      : asInt(obj.modelVersion, { min: 0, max: 1_000_000_000 });

  if (outcome) meta.outcome = outcome;
  if (modelSource) meta.modelSource = modelSource;
  if (generatorType) meta.generatorType = generatorType;
  if (modelMode) meta.modelMode = modelMode;
  if (channel) meta.channel = channel;
  if (modelArchId) meta.modelArchId = modelArchId;
  if (rewardProfileId) meta.rewardProfileId = rewardProfileId;
  if (queuePolicyId) meta.queuePolicyId = queuePolicyId;
  if (rewardPolicy) meta.rewardPolicy = rewardPolicy;
  if (rewardPolicyId) meta.rewardPolicyId = rewardPolicyId;
  if (rewardKind) meta.rewardKind = rewardKind;
  if (pipelineId) meta.pipelineId = pipelineId;
  if (pipelineMode) meta.pipelineMode = pipelineMode;
  if (modelArch) meta.modelArch = modelArch;
  if (actorType) meta.actorType = actorType;
  if (actorPolicyId) meta.actorPolicyId = actorPolicyId;
  if (trainingIntent) meta.trainingIntent = trainingIntent;
  if (pieceSourceProfile) meta.pieceSourceProfile = pieceSourceProfile;
  if (rewardGamma != null) meta.rewardGamma = rewardGamma;
  if (modelVersion != null) meta.modelVersion = modelVersion;
  if (obj.rewardGamma != null && rewardGamma == null) return null;
  return Object.keys(meta).length > 0 ? meta : null;
};

const parseReplay = (
  value: unknown,
  options: { maxRows: number; maxCols: number },
): TrajectoryReplayStepV1 | null => {
  const obj = asObject(value);
  if (!obj) return null;

  const lockPiece = asPiece(obj.lockPiece);
  const lockRotation = asInt(obj.lockRotation, { min: 0, max: 3 });
  const lockX = asInt(obj.lockX, { min: -8, max: options.maxCols + 8 });
  const lockY = asInt(obj.lockY, { min: -8, max: options.maxRows + 8 });
  const holdUsed = asBoolean(obj.holdUsed);
  const gameTimeMs = asInt(obj.gameTimeMs, {
    min: 0,
    max: 7 * 24 * 60 * 60 * 1000,
  });
  const totalLinesCleared = asInt(obj.totalLinesCleared, {
    min: 0,
    max: 1_000_000,
  });
  const score = asInt(obj.score, { min: 0, max: 1_000_000_000 });
  if (
    !lockPiece ||
    lockRotation == null ||
    lockX == null ||
    lockY == null ||
    holdUsed == null ||
    gameTimeMs == null ||
    totalLinesCleared == null ||
    score == null
  ) {
    return null;
  }
  return {
    lockPiece,
    lockRotation,
    lockX,
    lockY,
    holdUsed,
    gameTimeMs,
    totalLinesCleared,
    score,
  };
};

const parseSample = (
  value: unknown,
  options: { maxRows: number; maxCols: number },
): TrajectorySessionSampleV1 | null => {
  const obj = asObject(value);
  if (!obj) return null;

  const id = asString(obj.id, 1, 128);
  const createdAtMs = asInt(obj.createdAtMs, { min: 0 });
  const boardOccupancy = parseBoardOccupancy(
    obj.boardOccupancy,
    options.maxRows,
    options.maxCols,
  );
  const hold = obj.hold == null ? null : asPiece(obj.hold);
  const action = asPiece(obj.action);
  const pieces = parsePieceArray(obj.pieces, 1, 32);
  const actionIndex = asInt(obj.actionIndex, { min: 0, max: 31 });
  const inferenceMs = asFiniteNumber(obj.inferenceMs, { min: 0 });
  const samplingMs = asFiniteNumber(obj.samplingMs, { min: 0 });
  const totalDecisionMs = asFiniteNumber(obj.totalDecisionMs, { min: 0 });

  if (
    !id ||
    createdAtMs == null ||
    !boardOccupancy ||
    action == null ||
    !pieces ||
    actionIndex == null ||
    inferenceMs == null ||
    samplingMs == null ||
    totalDecisionMs == null
  ) {
    return null;
  }
  if (actionIndex >= pieces.length) return null;
  if (pieces[actionIndex] !== action) return null;

  const logits = parseNumberArray(obj.logits, pieces.length);
  const probabilities = parseNumberArray(obj.probabilities, pieces.length);
  if (!logits || !probabilities) return null;

  const deliberationMs =
    obj.deliberationMs == null
      ? null
      : asInt(obj.deliberationMs, { min: 0, max: 24 * 60 * 60 * 1000 });
  if (obj.deliberationMs != null && deliberationMs == null) return null;

  const reward =
    obj.reward == null
      ? null
      : asFiniteNumber(obj.reward, { min: -1e9, max: 1e9 });
  if (obj.reward != null && reward == null) return null;

  const replay = obj.replay == null ? null : parseReplay(obj.replay, options);
  if (obj.replay != null && replay == null) return null;

  return {
    id,
    createdAtMs,
    deliberationMs,
    boardOccupancy,
    hold,
    action,
    actionIndex,
    pieces,
    logits,
    probabilities,
    inferenceMs,
    samplingMs,
    totalDecisionMs,
    reward,
    ...(replay ? { replay } : {}),
  };
};

export const parseTrajectorySessionV1 = (
  payload: unknown,
  options: ParseOptions = {},
): ParseTrajectorySessionResult => {
  const minSamplesRaw = options.minSamples ?? 1;
  const minSamples = Math.max(1, Math.trunc(minSamplesRaw));
  const maxSamples = options.maxSamples ?? MAX_TRAJECTORY_SAMPLES_PER_SESSION;
  const maxRows = options.maxRows ?? MAX_TRAJECTORY_ROWS;
  const maxCols = options.maxCols ?? MAX_TRAJECTORY_COLS;

  const obj = asObject(payload);
  if (!obj) return { ok: false, error: 'Invalid payload.' };
  if (obj.schema !== TRAJECTORY_SESSION_SCHEMA_V1) {
    return { ok: false, error: 'Unsupported trajectory schema.' };
  }

  const sessionId = asString(obj.sessionId, 8, 128);
  if (!sessionId || !/^[A-Za-z0-9._:-]+$/.test(sessionId)) {
    return { ok: false, error: 'Invalid session id.' };
  }

  const modeId = asString(obj.modeId, 1, 64);
  if (!modeId || !/^[a-z0-9_-]+$/.test(modeId)) {
    return { ok: false, error: 'Invalid mode id.' };
  }

  const buildVersion = asString(obj.buildVersion, 1, 64);
  if (!buildVersion) {
    return { ok: false, error: 'Missing build version.' };
  }

  const startedAtMs = asInt(obj.startedAtMs, { min: 0 });
  const endedAtMs = asInt(obj.endedAtMs, { min: 0 });
  const durationMs = asInt(obj.durationMs, {
    min: 0,
    max: 24 * 60 * 60 * 1000,
  });
  if (startedAtMs == null || endedAtMs == null || durationMs == null) {
    return { ok: false, error: 'Invalid run timing.' };
  }
  if (endedAtMs < startedAtMs) {
    return { ok: false, error: 'Run end cannot precede run start.' };
  }
  const elapsedMs = endedAtMs - startedAtMs;
  if (Math.abs(durationMs - elapsedMs) > 60_000) {
    return { ok: false, error: 'Run duration is inconsistent.' };
  }

  if (!Array.isArray(obj.samples)) {
    return { ok: false, error: 'Missing trajectory samples.' };
  }
  if (obj.samples.length < minSamples) {
    return {
      ok: false,
      error: `Trajectory has too few samples (need >= ${minSamples}).`,
    };
  }
  if (obj.samples.length > maxSamples) {
    return { ok: false, error: 'Trajectory has too many samples.' };
  }

  const samples: TrajectorySessionSampleV1[] = [];
  for (let i = 0; i < obj.samples.length; i += 1) {
    const sample = parseSample(obj.samples[i], { maxRows, maxCols });
    if (!sample) {
      return {
        ok: false,
        error: `Invalid trajectory sample at index ${i}.`,
      };
    }
    samples.push(sample);
  }

  const meta = parseMeta(obj.meta);
  if (obj.meta != null && !meta) {
    return { ok: false, error: 'Invalid trajectory metadata.' };
  }

  return {
    ok: true,
    value: {
      schema: TRAJECTORY_SESSION_SCHEMA_V1,
      sessionId,
      modeId,
      buildVersion,
      startedAtMs,
      endedAtMs,
      durationMs,
      samples,
      meta,
    },
  };
};
