import readline from 'node:readline';

import { BotEnvPool } from './envCore.ts';
import type {
  BridgeRequest,
  BridgeResponse,
  InitPayload,
  JsonObject,
  ResetManyPayload,
  SetCurriculumPayload,
  SetPieceSourcePayload,
  StepManyPayload,
} from './protocol.ts';

let pool: BotEnvPool | null = null;

const writeResponse = (response: BridgeResponse): void => {
  process.stdout.write(`${JSON.stringify(response)}\n`);
};

const writeError = (id: number, error: unknown): void => {
  const message = error instanceof Error ? error.message : String(error);
  writeResponse({ id, ok: false, error: message });
};

const asObject = (value: unknown): JsonObject => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {};
  return value as JsonObject;
};

const parseNumberArray = (value: unknown): number[] => {
  if (!Array.isArray(value)) return [];
  return value.map((item) =>
    typeof item === 'number' && Number.isFinite(item) ? Math.trunc(item) : 0,
  );
};

const ensurePool = (): BotEnvPool => {
  if (!pool)
    throw new Error('Environment pool is not initialized. Call init first.');
  return pool;
};

const handleInit = async (id: number, payload: unknown): Promise<void> => {
  const data = asObject(payload) as unknown as InitPayload;
  pool = await BotEnvPool.create(data);
  writeResponse({
    id,
    ok: true,
    result: {
      env_ids: pool.listEnvIds(),
    },
  });
};

const handleResetMany = (id: number, payload: unknown): void => {
  const data = asObject(payload) as unknown as ResetManyPayload;
  const envIds = parseNumberArray(data.envIds);
  const seeds = parseNumberArray(data.seeds);
  const result = ensurePool().resetMany(envIds, seeds);
  writeResponse({ id, ok: true, result });
};

const handleSetPieceSource = (id: number, payload: unknown): void => {
  const data = asObject(payload) as unknown as SetPieceSourcePayload;
  const pieceSourceProfile = ensurePool().setPieceSource(
    data.pieceSourceProfile,
  );
  writeResponse({
    id,
    ok: true,
    result: {
      piece_source_profile: pieceSourceProfile,
    },
  });
};

const handleSetCurriculum = (id: number, payload: unknown): void => {
  const data = asObject(payload) as unknown as SetCurriculumPayload;
  const config = ensurePool().setCurriculum(data);
  writeResponse({
    id,
    ok: true,
    result: {
      top_k: config.topK,
      bias_strength: config.biasStrength,
      danger_height: config.dangerHeight,
    },
  });
};

const handleStepMany = (id: number, payload: unknown): void => {
  const data = asObject(payload) as unknown as StepManyPayload;
  const envIds = parseNumberArray(data.envIds);
  const actions = parseNumberArray(data.actions);
  if (envIds.length !== actions.length) {
    throw new Error('envIds and actions must have matching lengths.');
  }
  const result = ensurePool().stepMany(envIds, actions);
  writeResponse({ id, ok: true, result });
};

const handlePopTrajectory = (id: number): void => {
  const trajectory = ensurePool().popTrajectorySession();
  writeResponse({
    id,
    ok: true,
    result: {
      trajectory,
    },
  });
};

const handleClose = (id: number): void => {
  writeResponse({ id, ok: true, result: { closed: true } });
  process.exit(0);
};

const handleRequest = async (line: string): Promise<void> => {
  const parsed = JSON.parse(line) as BridgeRequest;
  const id = Number.isFinite(parsed.id) ? Math.trunc(parsed.id) : -1;
  if (id < 0) {
    throw new Error('Invalid request id.');
  }
  switch (parsed.cmd) {
    case 'init':
      await handleInit(id, parsed.payload);
      return;
    case 'set_piece_source':
      handleSetPieceSource(id, parsed.payload);
      return;
    case 'set_curriculum':
      handleSetCurriculum(id, parsed.payload);
      return;
    case 'reset_many':
      handleResetMany(id, parsed.payload);
      return;
    case 'step_many':
      handleStepMany(id, parsed.payload);
      return;
    case 'pop_trajectory':
      handlePopTrajectory(id);
      return;
    case 'close':
      handleClose(id);
      return;
    default:
      throw new Error(
        `Unsupported command: ${String((parsed as { cmd?: unknown }).cmd)}`,
      );
  }
};

const rl = readline.createInterface({
  input: process.stdin,
  crlfDelay: Infinity,
});

rl.on('line', (line) => {
  const trimmed = line.trim();
  if (!trimmed) return;
  void handleRequest(trimmed).catch((error) => {
    let requestId = -1;
    try {
      const parsed = JSON.parse(trimmed) as { id?: unknown };
      if (typeof parsed.id === 'number' && Number.isFinite(parsed.id)) {
        requestId = Math.trunc(parsed.id);
      }
    } catch {
      // Ignore parse error for fallback id.
    }
    writeError(requestId >= 0 ? requestId : 0, error);
  });
});

rl.on('close', () => {
  process.exit(0);
});
