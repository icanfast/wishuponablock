#!/usr/bin/env -S node --enable-source-maps

import {
  mkdir,
  open as openFile,
  readFile,
  readdir,
  stat,
} from 'node:fs/promises';
import path from 'node:path';
import { gunzipSync } from 'node:zlib';

import {
  encodeBotObservationFromParts,
  normalizeBotObservationSpace,
  type BotObservationSpace,
} from '../../../src/core/botObservation.ts';
import {
  PLACEMENT_ACTION_DIM,
  PLACEMENT_ACTION_HOLD_STEP_DIM,
  PLACEMENT_ACTION_HOLD_STEP_INDEX,
  placementActionIndexFromFields,
  placementActionIndexFromNoHoldFields,
} from '../../../src/core/placementActionSpace.ts';
import {
  parseWubModelFromJsonText,
  type LoadedModel,
} from '../../../src/core/wubModel.ts';
import { collides } from '../../../src/core/piece.ts';
import {
  type ActivePiece,
  type Board,
  type PieceKind,
} from '../../../src/core/types.ts';
import { enumerateTrajectoryExecutorPlacements } from '../../../src/core/trajectoryExecutor.ts';
import {
  parseTrajectorySessionV1,
  type TrajectorySessionV1,
} from '../../../src/core/trajectoryProtocol.ts';
import type { BotActionSpaceKind } from './protocol.ts';

const DEFAULT_ACTION_SPACE_KIND: BotActionSpaceKind = 'placement_full_v1';
const DEFAULT_MAX_NODES = 20_000;
const DEFAULT_OUTPUT_PATH = 'tools/bot_env/output/bc_dataset.json';
const DEFAULT_MODEL_PATH = 'public/models/model_v4.json';
const SPAWN_X = 3;
const SPAWN_Y = -1;

type CliOptions = {
  inputs: string[];
  outputPath: string;
  modelPath: string;
  observationSpace: BotObservationSpace;
  phaseContextEnabled: boolean;
  modeFilter: string | null;
  actionSpaceKind: BotActionSpaceKind;
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
  observationSpace: BotObservationSpace;
  phaseContextEnabled: boolean;
  modeFilter: string | null;
  actionSpaceKind?: BotActionSpaceKind;
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

const defaultActionDimForActionSpaceKind = (
  actionSpaceKind: BotActionSpaceKind,
): number =>
  actionSpaceKind === 'placement_hold_step_v2'
    ? PLACEMENT_ACTION_HOLD_STEP_DIM
    : PLACEMENT_ACTION_DIM;

const printUsage = (): void => {
  console.log(`Usage:
  npx --yes tsx tools/bot_env/ts/buildBcDataset.ts \\
    --input <file-or-dir> [--input <file-or-dir> ...] \\
    [--output ${DEFAULT_OUTPUT_PATH}] \\
    [--model-path ${DEFAULT_MODEL_PATH}] \\
    [--observation-space raw_v1] \\
    [--phase-context | --no-phase-context] \\
    [--mode <mode_id> | --all-modes] \\
    [--action-space-kind ${DEFAULT_ACTION_SPACE_KIND}] \\
    [--action-dim ${PLACEMENT_ACTION_DIM}] \\
    [--max-nodes ${DEFAULT_MAX_NODES}] \\
    [--return-gamma 0.995] \\
    [--max-sessions 1000] \\
    [--max-records 500000]`);
};

const parseArgs = (argv: string[]): CliOptions | null => {
  const inputs: string[] = [];
  let outputPath = DEFAULT_OUTPUT_PATH;
  let modelPath = DEFAULT_MODEL_PATH;
  let observationSpace: BotObservationSpace = 'raw_v1';
  let phaseContextEnabled = false;
  let modeFilter: string | null = null;
  let actionSpaceKind: BotActionSpaceKind = DEFAULT_ACTION_SPACE_KIND;
  let actionDimOverride: number | null = null;
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
    if (arg === '--observation-space') {
      const value = argv[i + 1];
      if (!value) throw new Error('--observation-space requires a value.');
      observationSpace = normalizeBotObservationSpace(value.trim());
      i += 1;
      continue;
    }
    if (arg === '--all-modes') {
      modeFilter = null;
      continue;
    }
    if (arg === '--action-space-kind') {
      const value = argv[i + 1];
      if (!value) throw new Error('--action-space-kind requires a value.');
      actionSpaceKind =
        value.trim() === 'placement_hold_step_v2'
          ? 'placement_hold_step_v2'
          : 'placement_full_v1';
      i += 1;
      continue;
    }
    if (arg === '--phase-context') {
      phaseContextEnabled = true;
      continue;
    }
    if (arg === '--no-phase-context') {
      phaseContextEnabled = false;
      continue;
    }
    if (arg === '--action-dim') {
      actionDimOverride = asInt(
        argv[i + 1],
        defaultActionDimForActionSpaceKind(actionSpaceKind),
      );
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
    observationSpace,
    phaseContextEnabled,
    modeFilter,
    actionSpaceKind,
    actionDim:
      actionDimOverride ??
      defaultActionDimForActionSpaceKind(actionSpaceKind),
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
  model: LoadedModel | null;
  observationSpace: BotObservationSpace;
  phaseContextEnabled: boolean;
  board: Board;
  hold: PieceKind | null;
  active: PieceKind;
  next: PieceKind | null;
  nextQueue?: Array<PieceKind | null>;
  canHold: boolean;
  totalLinesCleared: number;
  timeMs: number;
  level: number;
  score: number;
  lineGoal: number | null;
}): number[] => {
  return Array.from(
    encodeBotObservationFromParts({
      observationSpace: input.observationSpace,
      model: input.model,
      includePhaseContext: input.phaseContextEnabled,
      parts: {
        board: input.board,
        hold: input.hold,
        active: input.active,
        next: input.next,
        nextQueue: input.nextQueue,
        canHold: input.canHold,
        totalLinesCleared: input.totalLinesCleared,
        timeMs: input.timeMs,
        level: input.level,
        score: input.score,
        lineGoal: input.lineGoal,
      },
    }),
  );
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

const buildVisibleQueueFromSamples = (
  samples: TrajectorySessionV1['samples'],
  startIndex: number,
  maxSlots = 5,
): Array<PieceKind | null> => {
  const out: Array<PieceKind | null> = [];
  for (let offset = 0; offset < maxSlots; offset += 1) {
    const sample = samples[startIndex + offset];
    out.push(sample?.action ?? null);
  }
  return out;
};

const buildActionMaskForChoices = (
  placements: ReturnType<typeof enumerateTrajectoryExecutorPlacements>,
  actionDim: number,
  actionSpaceKind: BotActionSpaceKind,
): number[] => {
  const actionMask = new Array<number>(actionDim).fill(0);
  let holdAvailable = false;
  for (const placement of placements) {
    if (actionSpaceKind === 'placement_hold_step_v2') {
      if (placement.holdUsed) {
        holdAvailable = true;
        continue;
      }
      const index = placementActionIndexFromNoHoldFields({
        lockRotation: normalizeRotation(placement.lockRotation),
        lockX: placement.lockX,
        lockY: placement.lockY,
      });
      if (index != null && index >= 0 && index < actionDim) {
        actionMask[index] = 1;
      }
      continue;
    }
    const index = placementActionIndexFromFields({
      holdUsed: placement.holdUsed,
      lockRotation: normalizeRotation(placement.lockRotation),
      lockX: placement.lockX,
      lockY: placement.lockY,
    });
    if (index != null && index >= 0 && index < actionDim) {
      actionMask[index] = 1;
    }
  }
  if (
    actionSpaceKind === 'placement_hold_step_v2' &&
    holdAvailable &&
    actionDim > PLACEMENT_ACTION_HOLD_STEP_INDEX
  ) {
    actionMask[PLACEMENT_ACTION_HOLD_STEP_INDEX] = 1;
  }
  return actionMask;
};

const findPlacementIndex = (
  actionDim: number,
  sample: TrajectorySessionV1['samples'][number],
  actionSpaceKind: BotActionSpaceKind,
): number => {
  const replay = sample.replay;
  if (!replay) return -1;
  const index =
    actionSpaceKind === 'placement_hold_step_v2'
      ? placementActionIndexFromNoHoldFields({
          lockRotation: normalizeRotation(replay.lockRotation),
          lockX: replay.lockX,
          lockY: replay.lockY,
        })
      : placementActionIndexFromFields({
          holdUsed: replay.holdUsed,
          lockRotation: normalizeRotation(replay.lockRotation),
          lockX: replay.lockX,
          lockY: replay.lockY,
        });
  if (index == null || index < 0 || index >= actionDim) return -1;
  return index;
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

type ProgressSnapshot = {
  filesDone: number;
  filesTotal: number;
  sessionsParsed: number;
  sessionsUsed: number;
  records: number;
};

const formatDuration = (elapsedMs: number): string => {
  const totalSec = Math.max(0, Math.floor(elapsedMs / 1000));
  const hours = Math.floor(totalSec / 3600);
  const minutes = Math.floor((totalSec % 3600) / 60);
  const seconds = totalSec % 60;
  if (hours > 0) {
    return `${hours}h${String(minutes).padStart(2, '0')}m${String(
      seconds,
    ).padStart(2, '0')}s`;
  }
  return `${minutes}m${String(seconds).padStart(2, '0')}s`;
};

const createProgressReporter = () => {
  const startedAt = Date.now();
  const tty = Boolean(process.stdout.isTTY);
  let lastRenderedAt = 0;
  let lastLoggedAt = 0;
  let lastLineLength = 0;

  const render = (snapshot: ProgressSnapshot, force = false): void => {
    const now = Date.now();
    const elapsedMs = now - startedAt;
    if (!force && now - lastRenderedAt < 200) return;
    lastRenderedAt = now;

    const fileRatio =
      snapshot.filesTotal > 0
        ? clamp(snapshot.filesDone / snapshot.filesTotal, 0, 1)
        : 1;
    const barWidth = 24;
    const filled = Math.max(
      0,
      Math.min(barWidth, Math.round(fileRatio * barWidth)),
    );
    const bar = `${'#'.repeat(filled)}${'-'.repeat(Math.max(0, barWidth - filled))}`;
    const recPerSec =
      elapsedMs > 0 ? snapshot.records / Math.max(1e-3, elapsedMs / 1000) : 0;

    const line =
      `[bc-dataset] [${bar}] ${snapshot.filesDone}/${snapshot.filesTotal} files ` +
      `| sessions ${snapshot.sessionsParsed} (${snapshot.sessionsUsed} used) ` +
      `| records ${snapshot.records} ` +
      `| ${recPerSec.toFixed(1)} rec/s ` +
      `| ${formatDuration(elapsedMs)}`;

    if (tty) {
      const padded = line.padEnd(lastLineLength, ' ');
      lastLineLength = Math.max(lastLineLength, line.length);
      process.stdout.write(`\r${padded}`);
      return;
    }

    if (force || now - lastLoggedAt >= 5_000) {
      lastLoggedAt = now;
      console.log(line);
    }
  };

  const close = (snapshot: ProgressSnapshot): void => {
    render(snapshot, true);
    if (tty) process.stdout.write('\n');
  };

  return {
    render,
    close,
  };
};

const writeDatasetStreaming = async (params: {
  outputPath: string;
  header: Omit<BcDataset, 'records' | 'summary'>;
  records: BcRecord[];
  summary: BcDataset['summary'];
}): Promise<void> => {
  const fh = await openFile(params.outputPath, 'w');
  try {
    await fh.write(
      `{"schema":${JSON.stringify(params.header.schema)},"createdAtMs":${
        params.header.createdAtMs
      },"modelPath":${JSON.stringify(
        params.header.modelPath,
      )},"observationSpace":${JSON.stringify(
        params.header.observationSpace,
      )},"phaseContextEnabled":${JSON.stringify(
        params.header.phaseContextEnabled,
      )},"actionSpaceKind":${JSON.stringify(
        params.header.actionSpaceKind ?? DEFAULT_ACTION_SPACE_KIND,
      )},"modeFilter":${JSON.stringify(
        params.header.modeFilter,
      )},"obsDim":${params.header.obsDim},"actionDim":${
        params.header.actionDim
      },"records":[`,
    );

    for (let i = 0; i < params.records.length; i += 1) {
      if (i > 0) await fh.write(',');
      await fh.write(JSON.stringify(params.records[i]));
    }

    await fh.write(`],"summary":${JSON.stringify(params.summary)}}\n`);
  } finally {
    await fh.close();
  }
};

const main = async (): Promise<void> => {
  const options = parseArgs(process.argv);
  if (!options) return;

  let model: LoadedModel | null = null;
  if (options.observationSpace === 'model_head_v1') {
    const modelText = await readFile(path.resolve(options.modelPath), 'utf8');
    model = parseWubModelFromJsonText(modelText);
  }

  const discovered = new Set<string>();
  for (const input of options.inputs) {
    for (const filePath of await collectInputFiles(input)) {
      discovered.add(filePath);
    }
  }
  const files = [...discovered].sort();
  const progress = createProgressReporter();

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
  let filesDone = 0;
  progress.render(
    {
      filesDone,
      filesTotal: files.length,
      sessionsParsed,
      sessionsUsed,
      records: records.length,
    },
    true,
  );

  for (let fileIndex = 0; fileIndex < files.length; fileIndex += 1) {
    const filePath = files[fileIndex];
    const parsed = await loadJsonUnknown(filePath).catch(() => null);
    if (!parsed) {
      skipped.parseFailed += 1;
      filesDone = fileIndex + 1;
      progress.render({
        filesDone,
        filesTotal: files.length,
        sessionsParsed,
        sessionsUsed,
        records: records.length,
      });
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
        const nextQueueBefore = buildVisibleQueueFromSamples(session.samples, i);
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
        const actionMask = buildActionMaskForChoices(
          placements,
          options.actionDim,
          options.actionSpaceKind,
        );

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
          observationSpace: options.observationSpace,
          phaseContextEnabled: options.phaseContextEnabled,
          board: boardBefore,
          hold: holdBefore,
          active: activeBefore,
          next: nextBefore,
          nextQueue: nextQueueBefore,
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

        const returnToGo = Number.isFinite(returnToGoByIndex[i])
          ? returnToGoByIndex[i]
          : null;
        const source = {
          sessionId: session.sessionId,
          sampleId: sample.id,
          sampleIndex: i,
          modeId: session.modeId,
          buildVersion: session.buildVersion,
        };

        if (
          options.actionSpaceKind === 'placement_hold_step_v2' &&
          replay.holdUsed
        ) {
          const holdActionIndex = PLACEMENT_ACTION_HOLD_STEP_INDEX;
          if (
            holdActionIndex < 0 ||
            holdActionIndex >= options.actionDim ||
            actionMask[holdActionIndex] <= 0
          ) {
            skipped.targetNotInActionSpace += 1;
            continue;
          }

          const activeAfterHold = holdBefore ?? nextBefore;
          if (activeAfterHold == null) {
            skipped.missingPreviousState += 1;
            continue;
          }
          const holdAfter = activeBefore;
          const nextAfterHold =
            holdBefore == null ? (session.samples[i + 1]?.action ?? null) : nextBefore;
          const nextQueueAfterHold =
            holdBefore == null
              ? buildVisibleQueueFromSamples(session.samples, i + 1)
              : nextQueueBefore;
          const activeAfterSpawn = spawnActiveForBoard(boardBefore, activeAfterHold);
          if (!activeAfterSpawn) {
            skipped.invalidActiveSpawn += 1;
            continue;
          }
          const postHoldPlacements = enumerateTrajectoryExecutorPlacements({
            board: boardBefore,
            active: activeAfterSpawn,
            hold: holdAfter,
            canHold: false,
            nextPieceOnFirstHold: nextAfterHold,
            maxNodesPerBranch: options.maxNodesPerBranch,
            allowSoftDrop: true,
          });
          if (postHoldPlacements.length === 0) {
            skipped.noPlacementChoices += 1;
            continue;
          }
          const postHoldActionMask = buildActionMaskForChoices(
            postHoldPlacements,
            options.actionDim,
            options.actionSpaceKind,
          );
          const postHoldActionIndex = findPlacementIndex(
            options.actionDim,
            sample,
            'placement_hold_step_v2',
          );
          if (
            postHoldActionIndex < 0 ||
            postHoldActionIndex >= options.actionDim ||
            postHoldActionMask[postHoldActionIndex] <= 0
          ) {
            skipped.targetNotInActionSpace += 1;
            continue;
          }
          const postHoldObs = buildObservation({
            model,
            observationSpace: options.observationSpace,
            phaseContextEnabled: options.phaseContextEnabled,
            board: boardBefore,
            hold: holdAfter,
            active: activeAfterHold,
            next: nextAfterHold,
            nextQueue: nextQueueAfterHold,
            canHold: false,
            totalLinesCleared,
            timeMs,
            level: 1,
            score,
            lineGoal: null,
          });
          if (postHoldObs.some((value) => !Number.isFinite(value))) {
            skipped.invalidRecord += 1;
            continue;
          }

          const holdRecord: BcRecord = {
            obs,
            actionMask,
            actionIndex: holdActionIndex,
            returnToGo,
            source,
          };
          const postHoldRecord: BcRecord = {
            obs: postHoldObs,
            actionMask: postHoldActionMask,
            actionIndex: postHoldActionIndex,
            returnToGo,
            source,
          };
          if (!pushIfRecord(records, holdRecord, options.maxRecords)) {
            skipped.maxRecordsReached += 1;
            break;
          }
          if (!pushIfRecord(records, postHoldRecord, options.maxRecords)) {
            records.pop();
            skipped.maxRecordsReached += 1;
            break;
          }
          sessionUsed = true;
          continue;
        }

        const actionIndex = findPlacementIndex(
          options.actionDim,
          sample,
          options.actionSpaceKind,
        );
        if (actionIndex < 0 || actionIndex >= options.actionDim) {
          skipped.targetNotInActionSpace += 1;
          continue;
        }
        if (actionMask[actionIndex] <= 0) {
          skipped.targetNotInActionSpace += 1;
          continue;
        }

        const record: BcRecord = {
          obs,
          actionMask,
          actionIndex,
          returnToGo,
          source,
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
      progress.render({
        filesDone,
        filesTotal: files.length,
        sessionsParsed,
        sessionsUsed,
        records: records.length,
      });
    }
    filesDone = fileIndex + 1;
    progress.render({
      filesDone,
      filesTotal: files.length,
      sessionsParsed,
      sessionsUsed,
      records: records.length,
    });
    if (options.maxRecords != null && records.length >= options.maxRecords) {
      break;
    }
  }
  progress.close({
    filesDone,
    filesTotal: files.length,
    sessionsParsed,
    sessionsUsed,
    records: records.length,
  });

  if (records.length === 0) {
    throw new Error('No compatible BC records were produced.');
  }

  const summary: BcDataset['summary'] = {
    filesScanned: files.length,
    sessionsParsed,
    sessionsUsed,
    records: records.length,
    skipped,
  };
  const header: Omit<BcDataset, 'records' | 'summary'> = {
    schema: 'wishuponablock.bot_bc_dataset.v1',
    createdAtMs: Date.now(),
    modelPath: options.modelPath,
    observationSpace: options.observationSpace,
    phaseContextEnabled: options.phaseContextEnabled,
    modeFilter: options.modeFilter,
    actionSpaceKind: options.actionSpaceKind,
    obsDim: records[0].obs.length,
    actionDim: options.actionDim,
  };

  const outputPath = path.resolve(options.outputPath);
  await mkdir(path.dirname(outputPath), { recursive: true });
  await writeDatasetStreaming({
    outputPath,
    header,
    records,
    summary,
  });
  console.log(
    `[bc-dataset] wrote ${records.length} records from ${sessionsUsed}/${sessionsParsed} sessions -> ${outputPath}`,
  );
};

void main().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`[bc-dataset] failed: ${message}`);
  process.exitCode = 1;
});
