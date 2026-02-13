import { GAME_PROTOCOL_VERSION } from './constants';
import {
  PIECES,
  type ActivePiece,
  type Board,
  type PieceKind,
  type PieceProbability,
} from './types';
import type { Settings } from './settings';

export interface SnapshotSessionMeta {
  id: string;
  createdAt: string;
  protocolVersion: number;
  buildVersion?: string;
  rows: number;
  cols: number;
  device_id?: string;
  user_id?: string;
  model_url?: string;
  pieceOrder?: readonly string[];
  settings: Pick<Settings, 'game' | 'generator'>;
  mode?: SnapshotModeInfo;
  comment?: string;
}

export interface SnapshotSample {
  index: number;
  timeMs: number;
  board: number[][];
  hold?: number;
  next?: number[];
  odds?: Array<{
    k: number;
    p: number;
  }>;
  active?: {
    k: number;
    r: number;
    x: number;
    y: number;
  };
  trigger?: SnapshotTrigger;
  linesLeft?: number;
  level?: number;
  score?: number;
}

export interface SnapshotSession {
  meta: SnapshotSessionMeta;
  samples: SnapshotSample[];
}

export interface SnapshotModeInfo {
  id: string;
  options?: Record<string, unknown>;
}

export type SnapshotTrigger = 'lock' | 'hold' | 'manual' | 'constructor';

const PIECE_TO_INDEX = new Map(PIECES.map((k, i) => [k, i + 1]));

function encodeBoard(board: Board): number[][] {
  return board.map((row) =>
    row.map((cell) => (cell ? (PIECE_TO_INDEX.get(cell) ?? 0) : 0)),
  );
}

function encodePiece(cell: PieceKind | null): number {
  return cell ? (PIECE_TO_INDEX.get(cell) ?? 0) : 0;
}

function encodeQueue(pieces: ReadonlyArray<PieceKind> | undefined): number[] {
  if (!pieces?.length) return [];
  return pieces
    .map((piece) => encodePiece(piece))
    .filter((value) => Number.isFinite(value) && value > 0);
}

function encodeOdds(
  odds: ReadonlyArray<PieceProbability> | undefined,
): Array<{ k: number; p: number }> {
  if (!odds?.length) return [];
  return odds
    .map((entry) => {
      const k = encodePiece(entry.piece);
      const p = Number.isFinite(entry.probability)
        ? Math.max(0, entry.probability)
        : 0;
      return {
        k,
        p: Math.round(p * 1_000_000) / 1_000_000,
      };
    })
    .filter((entry) => entry.k > 0 && entry.p > 0);
}

function encodeActivePiece(active: ActivePiece | null | undefined):
  | {
      k: number;
      r: number;
      x: number;
      y: number;
    }
  | undefined {
  if (!active) return undefined;
  return {
    k: encodePiece(active.k),
    r: Math.trunc(active.r),
    x: Math.trunc(active.x),
    y: Math.trunc(active.y),
  };
}

function makeSessionId(now = new Date()): string {
  const pad = (n: number) => String(n).padStart(2, '0');
  return [
    now.getFullYear(),
    pad(now.getMonth() + 1),
    pad(now.getDate()),
    pad(now.getHours()),
    pad(now.getMinutes()),
    pad(now.getSeconds()),
  ].join('');
}

export class SnapshotRecorder {
  private active: SnapshotSession | null = null;
  private startedAtMs = 0;
  private nextIndex = 0;

  get isRecording(): boolean {
    return this.active != null;
  }

  get sampleCount(): number {
    return this.active ? this.nextIndex : 0;
  }

  get sessionMeta(): SnapshotSessionMeta | null {
    return this.active?.meta ?? null;
  }

  start(
    settings: Settings,
    rows: number,
    cols: number,
    comment?: string,
    mode?: SnapshotModeInfo,
  ): SnapshotSession {
    const now = new Date();
    const trimmedComment = comment?.trim();
    const session: SnapshotSession = {
      meta: {
        id: makeSessionId(now),
        createdAt: now.toISOString(),
        protocolVersion: GAME_PROTOCOL_VERSION,
        rows,
        cols,
        settings: {
          game: { ...settings.game },
          generator: { ...settings.generator },
        },
        ...(mode ? { mode } : {}),
        ...(trimmedComment ? { comment: trimmedComment } : {}),
      },
      samples: [],
    };
    this.active = session;
    this.startedAtMs = performance.now();
    this.nextIndex = 0;
    return session;
  }

  record(
    board: Board,
    hold: PieceKind | null,
    options: {
      store?: boolean;
      active?: ActivePiece | null;
      next?: ReadonlyArray<PieceKind>;
      odds?: ReadonlyArray<PieceProbability>;
      trigger?: SnapshotTrigger;
      linesLeft?: number;
      level?: number;
      score?: number;
    } = {},
  ): SnapshotSample | null {
    if (!this.active) return null;
    const timeMs = performance.now() - this.startedAtMs;
    const linesLeft =
      options.linesLeft != null && Number.isFinite(options.linesLeft)
        ? Math.max(0, Math.trunc(options.linesLeft))
        : undefined;
    const level =
      options.level != null && Number.isFinite(options.level)
        ? Math.max(0, Math.trunc(options.level))
        : undefined;
    const score =
      options.score != null && Number.isFinite(options.score)
        ? Math.max(0, Math.trunc(options.score))
        : undefined;
    const encodedActive = encodeActivePiece(options.active);
    const encodedQueue = encodeQueue(options.next);
    const encodedOdds = encodeOdds(options.odds);
    const sample: SnapshotSample = {
      index: this.nextIndex,
      timeMs: Math.round(timeMs),
      board: encodeBoard(board),
      hold: encodePiece(hold),
      ...(encodedQueue.length > 0 ? { next: encodedQueue } : {}),
      ...(encodedOdds.length > 0 ? { odds: encodedOdds } : {}),
      ...(encodedActive ? { active: encodedActive } : {}),
      ...(options.trigger ? { trigger: options.trigger } : {}),
      ...(linesLeft != null ? { linesLeft } : {}),
      ...(level != null ? { level } : {}),
      ...(score != null ? { score } : {}),
    };
    this.nextIndex += 1;
    if (options.store ?? true) {
      this.active.samples.push(sample);
    }
    return sample;
  }

  setComment(comment: string): void {
    if (!this.active) return;
    const trimmed = comment.trim();
    if (trimmed) {
      this.active.meta.comment = trimmed;
    } else {
      delete this.active.meta.comment;
    }
  }

  stop(): SnapshotSession | null {
    const session = this.active;
    this.active = null;
    this.nextIndex = 0;
    return session;
  }

  discard(): void {
    this.active = null;
    this.nextIndex = 0;
  }
}

export function buildSnapshotFilename(session: SnapshotSession): string {
  return `snapshots_${session.meta.id}.json`;
}

export function serializeSnapshotSession(session: SnapshotSession): string {
  return JSON.stringify(session);
}

export async function saveSnapshotSessionToDirectory(
  session: SnapshotSession,
  directory: FileSystemDirectoryHandle,
): Promise<void> {
  const handle = await directory.getFileHandle(buildSnapshotFilename(session), {
    create: true,
  });
  const writable = await handle.createWritable();
  await writable.write(serializeSnapshotSession(session));
  await writable.close();
}

export function downloadSnapshotSession(session: SnapshotSession): void {
  const payload = serializeSnapshotSession(session);
  const blob = new Blob([payload], { type: 'application/json' });
  const url = URL.createObjectURL(blob);

  const a = document.createElement('a');
  a.href = url;
  a.download = buildSnapshotFilename(session);
  document.body.appendChild(a);
  a.click();
  a.remove();

  URL.revokeObjectURL(url);
}
