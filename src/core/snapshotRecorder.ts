import { GAME_PROTOCOL_VERSION } from './constants';
import { PIECES, type Board, type PieceKind } from './types';
import type { Settings } from './settings';

export interface SnapshotSessionMeta {
  id: string;
  createdAt: string;
  protocolVersion: number;
  rows: number;
  cols: number;
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
}

export interface SnapshotSession {
  meta: SnapshotSessionMeta;
  samples: SnapshotSample[];
}

export interface SnapshotModeInfo {
  id: string;
  options?: Record<string, unknown>;
}

const PIECE_TO_INDEX = new Map(PIECES.map((k, i) => [k, i + 1]));

function encodeBoard(board: Board): number[][] {
  return board.map((row) =>
    row.map((cell) => (cell ? (PIECE_TO_INDEX.get(cell) ?? 0) : 0)),
  );
}

function encodePiece(cell: PieceKind | null): number {
  return cell ? (PIECE_TO_INDEX.get(cell) ?? 0) : 0;
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
    options: { store?: boolean } = {},
  ): SnapshotSample | null {
    if (!this.active) return null;
    const timeMs = performance.now() - this.startedAtMs;
    const sample: SnapshotSample = {
      index: this.nextIndex,
      timeMs: Math.round(timeMs),
      board: encodeBoard(board),
      hold: encodePiece(hold),
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
