#!/usr/bin/env -S node --enable-source-maps

import { mkdir, readFile, readdir, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { gunzipSync } from 'node:zlib';

import {
  buildModelHeadInput,
  parseWubModelFromJsonText,
} from '../../../src/core/wubModel.ts';
import { collides } from '../../../src/core/piece.ts';
import {
  PIECES,
  type ActivePiece,
  type Board,
  type PieceKind,
} from '../../../src/core/types.ts';
import {
  enumerateTrajectoryExecutorPlacements,
  type TrajectoryExecutorReachablePlacement,
} from '../../../src/core/trajectoryExecutor.ts';
import {
  parseTrajectorySessionV1,
  type TrajectorySessionV1,
} from '../../../src/core/trajectoryProtocol.ts';

const DEFAULT_ACTION_DIM = 192;
const DEFAULT_MAX_NODES = 20_000;
const DEFAULT_OUTPUT_PATH = 'tools/bot_env/output/bc_dataset.json';
const DEFAULT_MODEL_PATH = 'public/models/model_v4.json';
const DEFAULT_MODE_FILTER = 'charcuterie';
const PIECE_INDEX = new Map(PIECES.map((piece, idx) => [piece, idx]));
const SPAWN_X = 3;
const SPAWN_Y = -1;

type CliOptions = {
  inputs: string[];
  outputPath: string;
  modelPath: string;
  modeFilter: string | null;
  actionDim: number;
  maxNodesPerBranch: number;
  returnGamma: number;
  maxSessions: number | null;
  maxRecords: number | null;
};

type BcRecord = {
  obs: number[];
  actionMask: number[];
  actionIndex: number;
  returnToGo: number | null;
  source: {
    sessionId: string;
    sampleId: string;
    sampleIndex: number;
    modeId: string;
    buildVersion: string;
  };
};

type BcDataset = {
  schema: 'wishuponablock.bot_bc_dataset.v1';
  createdAtMs: number;
  modelPath: string;
  modeFilter: string | null;
  obsDim: number;
  actionDim: number;
  records: BcRecord[];
  summary: {
    filesScanned: number;
    sessionsParsed: number;
    sessionsUsed: number;
    records: number;
    skipped: {
      parseFailed: number;
      modeFiltered: number;
      missingInitialState: number;
      noReplaySteps: number;
      missingPreviousState: number;
      invalidActiveSpawn: number;
      noPlacementChoices: number;
      targetNotInActionSpace: number;
      invalidRecord: number;
      maxSessionsReached: number;
      maxRecordsReached: number;
    };
  };
};

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value != null && !Array.isArray(value);

const asInt = (value: string | undefined, fallback: number): number => {
  if (!value) return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(1, Math.trunc(parsed));
};

const printUsage = (): void => {
  console.log(`Usage:
  npx --yes tsx tools/bot_env/ts/buildBcDataset.ts \\
    --input <file-or-dir> [--input <file-or-dir> ...] \\
    [--output ${DEFAULT_OUTPUT_PATH}] \\
    [--model-path ${DEFAULT_MODEL_PATH}] \\
    [--mode ${DEFAULT_MODE_FILTER}] \\
    [--action-dim ${DEFAULT_ACTION_DIM}] \\
    [--max-nodes ${DEFAULT_MAX_NODES}] \\
    [--return-gamma 0.995] \\
    [--max-sessions 1000] \\
    [--max-records 500000]`);
};

const parseArgs = (argv: string[]): CliOptions | null => {
  const inputs: string[] = [];
  let outputPath = DEFAULT_OUTPUT_PATH;
  let modelPath = DEFAULT_MODEL_PATH;
  let modeFilter: string | null = DEFAULT_MODE_FILTER;
  let actionDim = DEFAULT_ACTION_DIM;
  let maxNodesPerBranch = DEFAULT_MAX_NODES;
  let returnGamma = 0.995;
  let maxSessions: number | null = null;
  let maxRecords: number | null = null;

  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--help' || arg === '-h') {
      printUsage();
      return null;
    }
    if (arg === '--input') {
      const value = argv[i + 1];
      if (!value) throw new Error('--input requires a value.');
      inputs.push(value);
      i += 1;
      continue;
    }
    if (arg === '--output') {
      const value = argv[i + 1];
      if (!value) throw new Error('--output requires a value.');
      outputPath = value;
      i += 1;
      continue;
    }
    if (arg === '--model-path') {
      const value = argv[i + 1];
      if (!value) throw new Error('--model-path requires a value.');
      modelPath = value;
      i += 1;
      continue;
    }
    if (arg === '--mode') {
      const value = argv[i + 1];
      if (!value) throw new Error('--mode requires a value.');
      const normalized = value.trim().toLowerCase();
      modeFilter = normalized.length > 0 ? normalized : null;
      i += 1;
      continue;
    }
    if (arg === '--all-modes') {
      modeFilter = null;
      continue;
    }
    if (arg === '--action-dim') {
      actionDim = asInt(argv[i + 1], DEFAULT_ACTION_DIM);
      i += 1;
      continue;
    }
    if (arg === '--max-nodes') {
      maxNodesPerBranch = asInt(argv[i + 1], DEFAULT_MAX_NODES);
      i += 1;
      continue;
    }
    if (arg === '--return-gamma') {
      const parsed = Number(argv[i + 1]);
      if (Number.isFinite(parsed)) {
        returnGamma = clamp(parsed, 0, 1);
      }
      i += 1;
      continue;
    }
    if (arg === '--max-sessions') {
      maxSessions = asInt(argv[i + 1], 1);
      i += 1;
      continue;
    }
    if (arg === '--max-records') {
      maxRecords = asInt(argv[i + 1], 1);
      i += 1;
      continue;
    }
    throw new Error(`Unknown argument: ${arg}`);
  }
  if (inputs.length === 0) {
    throw new Error('At least one --input is required.');
  }
  return {
    inputs,
    outputPath,
    modelPath,
    modeFilter,
    actionDim,
    maxNodesPerBranch,
    returnGamma,
    maxSessions,
    maxRecords,
  };
};

const isJsonFilePath = (filePath: string): boolean =>
  filePath.endsWith('.json') || filePath.endsWith('.json.gz');

const collectInputFiles = async (inputPath: string): Promise<string[]> => {
  const resolved = path.resolve(inputPath);
  const st = await stat(resolved);
  if (st.isFile()) {
    return isJsonFilePath(resolved) ? [resolved] : [];
  }
  if (!st.isDirectory()) return [];
  const out: string[] = [];
  const stack: string[] = [resolved];
  while (stack.length > 0) {
    const current = stack.pop();
    if (!current) break;
    const entries = await readdir(current, { withFileTypes: true });
    for (const entry of entries) {
      const next = path.join(current, entry.name);
      if (entry.isDirectory()) {
        stack.push(next);
        continue;
      }
      if (entry.isFile() && isJsonFilePath(next)) {
        out.push(next);
      }
    }
  }
  return out.sort();
};

const loadJsonUnknown = async (filePath: string): Promise<unknown> => {
  const raw = await readFile(filePath);
  const text = filePath.endsWith('.gz')
    ? gunzipSync(raw).toString('utf8')
    : raw.toString('utf8');
  return JSON.parse(text) as unknown;
};

const toBoardFromOccupancy = (occupancy: number[][]): Board =>
  occupancy.map((row) => row.map((cell) => (cell > 0 ? 'I' : null)));

const spawnActiveForBoard = (
  board: Board,
  kind: PieceKind,
): ActivePiece | null => {
  const active: ActivePiece = {
    k: kind,
    r: 0,
    x: SPAWN_X,
    y: SPAWN_Y,
  };
  for (let i = 0; i < 4; i += 1) {
    if (!collides(board, active)) return active;
    active.y -= 1;
  }
  return collides(board, active) ? null : active;
};

const clamp = (value: number, min: number, max: number): number =>
  Math.min(max, Math.max(min, value));

const buildObservation = (input: {
  model: ReturnType<typeof parseWubModelFromJsonText>;
  board: Board;
  hold: PieceKind | null;
  active: PieceKind;
  next: PieceKind | null;
  canHold: boolean;
  totalLinesCleared: number;
  timeMs: number;
  level: number;
  score: number;
  lineGoal: number | null;
}): number[] => {
  const headInput = buildModelHeadInput(input.model, input.board, input.hold);
  const contextDim = PIECES.length + PIECES.length + 5;
  const out = new Float32Array(headInput.length + contextDim);
  out.set(headInput, 0);
  let offset = headInput.length;

  const activeIdx = PIECE_INDEX.get(input.active);
  if (activeIdx != null) out[offset + activeIdx] = 1;
  offset += PIECES.length;

  const nextIdx = input.next == null ? null : PIECE_INDEX.get(input.next);
  if (nextIdx != null) out[offset + nextIdx] = 1;
  offset += PIECES.length;

  const lineGoal =
    input.lineGoal != null && input.lineGoal > 0 ? input.lineGoal : null;
  const progress =
    lineGoal != null
      ? clamp(input.totalLinesCleared / lineGoal, 0, 2)
      : clamp(input.totalLinesCleared / 80, 0, 2);
  out[offset++] = progress;
  out[offset++] = clamp(input.timeMs / 180_000, 0, 2);
  out[offset++] = clamp(input.level / 20, 0, 2);
  out[offset++] = clamp(input.score / 200_000, 0, 2);
  out[offset++] = input.canHold ? 1 : 0;
  return Array.from(out);
};

const buildReturnToGo = (
  samples: TrajectorySessionV1['samples'],
  gamma: number,
): number[] => {
  const out = new Array<number>(samples.length).fill(0);
  let running = 0;
  for (let i = samples.length - 1; i >= 0; i -= 1) {
    const reward = samples[i];
    const value =
      typeof reward.reward === 'number' && Number.isFinite(reward.reward)
        ? reward.reward
        : 0;
    running = value + gamma * running;
    out[i] = running;
  }
  return out;
};

const normalizeRotation = (value: number): number => {
  const normalized = Math.trunc(value) % 4;
  return normalized < 0 ? normalized + 4 : normalized;
};

const findPlacementIndex = (
  placements: TrajectoryExecutorReachablePlacement[],
  actionDim: number,
  sample: TrajectorySessionV1['samples'][number],
): number => {
  const replay = sample.replay;
  if (!replay) return -1;
  const max = Math.min(actionDim, placements.length);
  const targetR = normalizeRotation(replay.lockRotation);
  for (let i = 0; i < max; i += 1) {
    const placement = placements[i];
    if (
      placement.lockPiece === replay.lockPiece &&
      normalizeRotation(placement.lockRotation) === targetR &&
      placement.lockX === replay.lockX &&
      placement.lockY === replay.lockY &&
      placement.holdUsed === replay.holdUsed
    ) {
      return i;
    }
  }
  return -1;
};

const pushIfRecord = (
  records: BcRecord[],
  record: BcRecord,
  maxRecords: number | null,
): boolean => {
  if (maxRecords != null && records.length >= maxRecords) return false;
  records.push(record);
  return true;
};

const toSessions = (raw: unknown): unknown[] => {
  if (Array.isArray(raw)) return raw;
  if (isObject(raw) && Array.isArray(raw.sessions)) {
    return raw.sessions;
  }
  return [raw];
};

const main = async (): Promise<void> => {
  const options = parseArgs(process.argv);
  if (!options) return;

  const modelText = await readFile(path.resolve(options.modelPath), 'utf8');
  const model = parseWubModelFromJsonText(modelText);

  const discovered = new Set<string>();
  for (const input of options.inputs) {
    for (const filePath of await collectInputFiles(input)) {
      discovered.add(filePath);
    }
  }
  const files = [...discovered].sort();

  const records: BcRecord[] = [];
  const skipped: BcDataset['summary']['skipped'] = {
    parseFailed: 0,
    modeFiltered: 0,
    missingInitialState: 0,
    noReplaySteps: 0,
    missingPreviousState: 0,
    invalidActiveSpawn: 0,
    noPlacementChoices: 0,
    targetNotInActionSpace: 0,
    invalidRecord: 0,
    maxSessionsReached: 0,
    maxRecordsReached: 0,
  };

  let sessionsParsed = 0;
  let sessionsUsed = 0;

  for (const filePath of files) {
    const parsed = await loadJsonUnknown(filePath).catch(() => null);
    if (!parsed) {
      skipped.parseFailed += 1;
      continue;
    }
    const candidates = toSessions(parsed);
    for (const candidate of candidates) {
      const parsedSession = parseTrajectorySessionV1(candidate, {
        minSamples: 1,
      });
      if (!parsedSession.ok) {
        skipped.parseFailed += 1;
        continue;
      }
      const session = parsedSession.value;
      sessionsParsed += 1;
      if (options.modeFilter && session.modeId !== options.modeFilter) {
        skipped.modeFiltered += 1;
        continue;
      }
      if (options.maxSessions != null && sessionsUsed >= options.maxSessions) {
        skipped.maxSessionsReached += 1;
        continue;
      }
      const replayCount = session.samples.reduce(
        (count, sample) => count + (sample.replay ? 1 : 0),
        0,
      );
      if (replayCount <= 0) {
        skipped.noReplaySteps += 1;
        continue;
      }
      if (!session.initialState) {
        skipped.missingInitialState += 1;
        continue;
      }

      let sessionUsed = false;
      const returnToGoByIndex = buildReturnToGo(
        session.samples,
        options.returnGamma,
      );
      for (let i = 0; i < session.samples.length; i += 1) {
        const sample = session.samples[i];
        const replay = sample.replay;
        if (!replay) continue;

        const hasPrev = i > 0;
        const prevSample = hasPrev ? session.samples[i - 1] : null;
        if (!hasPrev && !session.initialState) {
          skipped.missingPreviousState += 1;
          continue;
        }

        const boardBeforeOcc = hasPrev
          ? (prevSample?.boardOccupancy ?? null)
          : session.initialState.boardOccupancy;
        const holdBefore = hasPrev
          ? (prevSample?.hold ?? null)
          : session.initialState.hold;
        const activeBefore = hasPrev
          ? (prevSample?.action ?? session.initialState.active.k)
          : session.initialState.active.k;
        const nextBefore = sample.action;
        if (!boardBeforeOcc) {
          skipped.missingPreviousState += 1;
          continue;
        }

        const boardBefore = toBoardFromOccupancy(boardBeforeOcc);
        const activeSpawn = spawnActiveForBoard(boardBefore, activeBefore);
        if (!activeSpawn) {
          skipped.invalidActiveSpawn += 1;
          continue;
        }
        const nextPieceOnFirstHold =
          holdBefore == null && replay.holdUsed ? replay.lockPiece : nextBefore;
        const placements = enumerateTrajectoryExecutorPlacements({
          board: boardBefore,
          active: activeSpawn,
          hold: holdBefore,
          canHold: true,
          nextPieceOnFirstHold,
          maxNodesPerBranch: options.maxNodesPerBranch,
          allowSoftDrop: true,
        });
        if (placements.length === 0) {
          skipped.noPlacementChoices += 1;
          continue;
        }
        const actionMask = new Array<number>(options.actionDim).fill(0);
        const maxChoices = Math.min(options.actionDim, placements.length);
        for (let j = 0; j < maxChoices; j += 1) actionMask[j] = 1;
        const actionIndex = findPlacementIndex(
          placements,
          options.actionDim,
          sample,
        );
        if (actionIndex < 0 || actionIndex >= options.actionDim) {
          skipped.targetNotInActionSpace += 1;
          continue;
        }
        if (actionMask[actionIndex] <= 0) {
          skipped.targetNotInActionSpace += 1;
          continue;
        }

        const beforeReplay = hasPrev ? prevSample?.replay : null;
        const totalLinesCleared = beforeReplay
          ? Math.max(0, Math.trunc(beforeReplay.totalLinesCleared))
          : Math.max(0, Math.trunc(session.initialState.totalLinesCleared));
        const timeMs = beforeReplay
          ? Math.max(0, Math.trunc(beforeReplay.gameTimeMs))
          : Math.max(0, Math.trunc(session.initialState.timeMs));
        const score = beforeReplay
          ? Math.max(0, Math.trunc(beforeReplay.score))
          : Math.max(0, Math.trunc(session.initialState.score));

        const obs = buildObservation({
          model,
          board: boardBefore,
          hold: holdBefore,
          active: activeBefore,
          next: nextBefore,
          canHold: true,
          totalLinesCleared,
          timeMs,
          level: 1,
          score,
          lineGoal: null,
        });
        if (obs.some((value) => !Number.isFinite(value))) {
          skipped.invalidRecord += 1;
          continue;
        }

        const record: BcRecord = {
          obs,
          actionMask,
          actionIndex,
          returnToGo: Number.isFinite(returnToGoByIndex[i])
            ? returnToGoByIndex[i]
            : null,
          source: {
            sessionId: session.sessionId,
            sampleId: sample.id,
            sampleIndex: i,
            modeId: session.modeId,
            buildVersion: session.buildVersion,
          },
        };
        if (!pushIfRecord(records, record, options.maxRecords)) {
          skipped.maxRecordsReached += 1;
          break;
        }
        sessionUsed = true;
      }
      if (sessionUsed) {
        sessionsUsed += 1;
      }
      if (options.maxRecords != null && records.length >= options.maxRecords) {
        break;
      }
    }
    if (options.maxRecords != null && records.length >= options.maxRecords) {
      break;
    }
  }

  if (records.length === 0) {
    throw new Error('No compatible BC records were produced.');
  }

  const dataset: BcDataset = {
    schema: 'wishuponablock.bot_bc_dataset.v1',
    createdAtMs: Date.now(),
    modelPath: options.modelPath,
    modeFilter: options.modeFilter,
    obsDim: records[0].obs.length,
    actionDim: options.actionDim,
    records,
    summary: {
      filesScanned: files.length,
      sessionsParsed,
      sessionsUsed,
      records: records.length,
      skipped,
    },
  };

  const outputPath = path.resolve(options.outputPath);
  await mkdir(path.dirname(outputPath), { recursive: true });
  await writeFile(outputPath, `${JSON.stringify(dataset, null, 2)}\n`, 'utf8');
  console.log(
    `[bc-dataset] wrote ${records.length} records from ${sessionsUsed}/${sessionsParsed} sessions -> ${outputPath}`,
  );
};

void main().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`[bc-dataset] failed: ${message}`);
  process.exitCode = 1;
});
