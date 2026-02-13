import { getMode, normalizeModeId } from '../../core/modes';
import { SPAWN_X, SPAWN_Y } from '../../core/constants';
import { collides } from '../../core/piece';
import type {
  SnapshotSession,
  SnapshotSample,
  SnapshotTrigger,
} from '../../core/snapshotRecorder';
import type { UploadClient } from '../../core/uploadClient';
import {
  PIECES,
  type ActivePiece,
  type Board,
  type PieceKind,
  type PieceProbability,
} from '../../core/types';
import { createToolScreen } from '../screens/toolScreen';
import type { ToolController } from './toolHost';
import type { ToolCanvas } from './toolCanvas';

type LabelingToolOptions = {
  toolUsesRemote: boolean;
  uploadClient: UploadClient;
  uploadBaseUrl: string;
  buildVersion?: string;
  canvas: ToolCanvas;
  onBack: () => void;
};

type Playstyle = 'beginner' | 'advanced';

const PLAYSTYLE_STORAGE_KEY = 'wub.labeler.playstyle';
const VIRTUAL_SESSION_SIZE = 100;
const REMOTE_RECENT_SESSION_WINDOW = 4;
const REMOTE_RECENT_SNAPSHOT_WINDOW = 6;
const REMOTE_LABELED_SNAPSHOT_WINDOW = 128;
const REMOTE_PREFETCH_BATCH_SIZE = 20;
const REMOTE_PREFETCH_MIN_QUEUE = 6;
const REMOTE_PREFETCH_MAX_QUEUE = 60;
const LOCAL_RECENT_SAMPLE_WINDOW = 6;
const LOCAL_RECENT_SESSION_WINDOW = 3;
const MAX_PICK_ATTEMPTS = 24;

export function createLabelingTool(
  options: LabelingToolOptions,
): ToolController {
  const {
    toolUsesRemote,
    uploadClient,
    uploadBaseUrl,
    buildVersion,
    canvas,
    onBack,
  } = options;

  type TriggerFilter =
    | 'all'
    | 'auto'
    | 'manual'
    | 'lock'
    | 'hold'
    | 'constructor';

  const toolUi = createToolScreen({ toolUsesRemote });
  const toolInputButton = toolUi.inputButton;
  const toolInputStatus = toolUi.inputStatus;
  const toolPlaystyleSelect = toolUi.playstyleSelect;
  const toolModeSelect = toolUi.modeSelect;
  const toolTriggerSelect = toolUi.triggerSelect;
  const toolBuildSelect = toolUi.buildSelect;
  const toolOutputButton = toolUi.outputButton;
  const toolOutputStatus = toolUi.outputStatus;
  const toolSampleStatus = toolUi.sampleStatus;
  const toolActionStatus = toolUi.actionStatus;
  const toolBackButton = toolUi.backButton;
  const toolNextButton = toolUi.nextButton;
  const toolPieceButtons = toolUi.pieceButtons;

  let toolOutputDirHandle: FileSystemDirectoryHandle | null = null;
  let toolSnapshotsAll: Array<{ name: string; session: SnapshotSession }> = [];
  let toolSnapshots: Array<{ name: string; session: SnapshotSession }> = [];
  let toolTotalSamples = 0;
  let toolTotalSamplesAll = 0;
  let toolModeFilter = 'all';
  let toolTriggerFilter: TriggerFilter = 'all';
  const defaultBuildFilter = buildVersion?.trim() ? buildVersion : 'all';
  let toolBuildFilter = defaultBuildFilter;
  let toolActive = false;
  let playstyle: Playstyle = 'beginner';
  let currentSample: {
    file: { name: string; session: SnapshotSession };
    index: number;
    board: Board;
    raw: number[][];
    hold: PieceKind | null;
    active?: ActivePiece | null;
    queuePieces: PieceKind[];
    queueOdds: PieceProbability[];
    snapshotId?: number;
    timeMs?: number;
    trigger?: SnapshotSample['trigger'];
    linesLeft?: number;
    level?: number;
    score?: number;
  } | null = null;
  let selectedLabels: PieceKind[] = [];
  let toolBusy = false;
  let labelIndex: Record<string, number> = {};
  let toolSampleIndex: Array<{
    file: { name: string; session: SnapshotSession };
    index: number;
  }> = [];
  let remoteBuildCounts: Array<{ build: string; count: number }> | null = null;
  let remoteBuildLoading = false;
  let remoteSampleQueue: Array<{
    snapshotId: number | null;
    sessionId: string | null;
    batchId: number | null;
    sampleIndex: number;
    session: SnapshotSession;
  }> = [];
  let remotePrefetchPromise: Promise<void> | null = null;
  let remotePrefetchVersion = 0;
  let recentRemoteSnapshotIds: number[] = [];
  let recentRemoteSessionIds: string[] = [];
  let recentRemoteLabeledSnapshotIds: number[] = [];
  let recentLocalSampleKeys: string[] = [];
  let recentLocalSessionIds: string[] = [];
  let recentLocalBatchKeys: string[] = [];
  const triggerLabel = (trigger: TriggerFilter): string => {
    switch (trigger) {
      case 'manual':
        return 'Manual';
      case 'lock':
        return 'Lock';
      case 'hold':
        return 'Hold';
      case 'constructor':
        return 'Constructor';
      case 'auto':
        return 'Auto (lock/hold)';
      default:
        return 'All Triggers';
    }
  };

  const normalizeTrigger = (sample: SnapshotSample): TriggerFilter => {
    const trigger = sample.trigger;
    if (
      trigger === 'manual' ||
      trigger === 'lock' ||
      trigger === 'hold' ||
      trigger === 'constructor'
    ) {
      return trigger;
    }
    return 'auto';
  };

  const updateLabelButtons = () => {
    for (const [piece, btn] of toolPieceButtons) {
      const selected = selectedLabels.includes(piece);
      btn.style.borderColor = selected ? '#ffffff' : '#1f2a37';
      btn.style.boxShadow = selected
        ? '0 0 0 1px rgba(255,255,255,0.5)'
        : 'none';
    }
  };

  const clearLabelSelection = () => {
    selectedLabels = [];
    updateLabelButtons();
  };

  for (const [piece, btn] of toolPieceButtons) {
    btn.addEventListener('click', () => {
      const idx = selectedLabels.indexOf(piece);
      if (idx >= 0) {
        selectedLabels.splice(idx, 1);
      } else {
        selectedLabels.push(piece);
      }
      updateLabelButtons();
    });
  }

  const decodeBoard = (raw: number[][], order?: readonly string[]): Board => {
    const resolvedOrder = order ?? PIECES;
    return raw.map((row) =>
      row.map((value) => {
        if (value <= 0) return null;
        const piece = resolvedOrder[value - 1] ?? PIECES[value - 1];
        return piece as PieceKind;
      }),
    );
  };

  const decodePiece = (
    value: unknown,
    order?: readonly string[],
  ): PieceKind | null => {
    const resolvedOrder = order ?? PIECES;
    if (typeof value === 'number' && Number.isFinite(value)) {
      const idx = Math.trunc(value) - 1;
      if (idx < 0) return null;
      const piece = resolvedOrder[idx] ?? PIECES[idx];
      return (piece as PieceKind) ?? null;
    }
    if (typeof value === 'string' && value.trim()) {
      const normalized = value.trim().toUpperCase();
      if (PIECES.includes(normalized as PieceKind)) {
        return normalized as PieceKind;
      }
    }
    return null;
  };

  const decodeHold = (
    hold: number | undefined,
    order?: readonly string[],
  ): PieceKind | null => {
    if (hold == null) return null;
    return decodePiece(hold, order);
  };

  const decodeActive = (
    active: unknown,
    board: Board,
    order?: readonly string[],
  ): ActivePiece | null => {
    if (!active || typeof active !== 'object') return null;
    const source = active as {
      k?: unknown;
      r?: unknown;
      x?: unknown;
      y?: unknown;
    };
    const piece = decodePiece(source.k, order);
    if (!piece) return null;
    // Labeling view normalizes active display to spawn pose for consistency.
    const normalized: ActivePiece = {
      k: piece,
      r: 0,
      x: SPAWN_X,
      y: SPAWN_Y,
    };
    if (!collides(board, normalized, normalized.r, 0, 1)) {
      normalized.y += 1;
    }
    return normalized;
  };

  const decodeQueueOdds = (
    sample: SnapshotSample,
    order?: readonly string[],
  ): PieceProbability[] => {
    const oddsEntries = Array.isArray(sample.odds) ? sample.odds : [];
    return oddsEntries
      .map((entry) => {
        if (!entry || typeof entry !== 'object') return null;
        const source = entry as { k?: unknown; p?: unknown };
        const piece = decodePiece(source.k, order);
        if (!piece) return null;
        const probability =
          typeof source.p === 'number' && Number.isFinite(source.p)
            ? Math.max(0, source.p)
            : 0;
        if (probability <= 0) return null;
        return { piece, probability };
      })
      .filter((entry): entry is PieceProbability => entry != null)
      .sort((a, b) => b.probability - a.probability);
  };

  const decodeQueuePieces = (
    sample: SnapshotSample,
    order?: readonly string[],
  ): PieceKind[] => {
    const fromOdds = decodeQueueOdds(sample, order).map((entry) => entry.piece);
    if (fromOdds.length > 0) return fromOdds;

    const nextPieces = Array.isArray(sample.next) ? sample.next : [];
    return nextPieces
      .map((value) => decodePiece(value, order))
      .filter((value): value is PieceKind => value != null);
  };

  const encodeBoardString = (raw: number[][]): string =>
    raw.map((row) => row.join('')).join('/');

  const sampleBatchId = (index: number): number | null => {
    if (!Number.isFinite(index)) return null;
    if (index < 0) return null;
    return Math.floor(index / VIRTUAL_SESSION_SIZE);
  };

  const pushRecentUnique = <T>(
    list: T[],
    value: T | null | undefined,
    limit: number,
  ): void => {
    if (value == null) return;
    const next = [value, ...list.filter((entry) => entry !== value)];
    list.splice(0, list.length, ...next.slice(0, Math.max(1, limit)));
  };

  const requestDirectoryAccess = async (
    handle: FileSystemDirectoryHandle,
    mode: 'read' | 'readwrite',
  ): Promise<boolean> => {
    let permission: PermissionState = 'granted';
    if (handle.queryPermission) {
      permission = await handle.queryPermission({ mode });
    }
    if (permission !== 'granted' && handle.requestPermission) {
      permission = await handle.requestPermission({ mode });
    }
    return permission === 'granted';
  };

  const isFileHandle = (
    handle: FileSystemHandle,
  ): handle is FileSystemFileHandle => handle.kind === 'file';

  const writeFileInDir = async (
    dir: FileSystemDirectoryHandle,
    name: string,
    contents: string,
  ): Promise<void> => {
    const handle = await dir.getFileHandle(name, { create: true });
    const writable = await handle.createWritable();
    await writable.write(contents);
    await writable.close();
  };

  const appendJsonl = async (
    dir: FileSystemDirectoryHandle,
    name: string,
    line: string,
  ): Promise<void> => {
    const handle = await dir.getFileHandle(name, { create: true });
    const file = await handle.getFile();
    const existing = await file.text();
    const prefix = existing && !existing.endsWith('\n') ? '\n' : '';
    const next = `${existing}${prefix}${line}\n`;
    await writeFileInDir(dir, name, next);
  };

  const loadLabelIndex = async (): Promise<void> => {
    if (toolUsesRemote) {
      labelIndex = {};
      return;
    }
    if (!toolOutputDirHandle) return;
    try {
      const handle = await toolOutputDirHandle.getFileHandle(
        'labeling_index.json',
        { create: true },
      );
      const file = await handle.getFile();
      const text = await file.text();
      labelIndex = text ? (JSON.parse(text) as Record<string, number>) : {};
    } catch {
      labelIndex = {};
    }
  };

  const saveLabelIndex = async (): Promise<void> => {
    if (toolUsesRemote) return;
    if (!toolOutputDirHandle) return;
    await writeFileInDir(
      toolOutputDirHandle,
      'labeling_index.json',
      JSON.stringify(labelIndex),
    );
  };

  const updateToolSampleStatus = (text: string) => {
    toolSampleStatus.textContent = text;
  };

  const updateToolActionStatus = (text: string) => {
    toolActionStatus.textContent = text;
  };

  const loadPlaystyle = (): Playstyle => {
    try {
      const stored = localStorage.getItem(PLAYSTYLE_STORAGE_KEY);
      if (stored === 'advanced') return 'advanced';
      return 'beginner';
    } catch {
      return 'beginner';
    }
  };

  const savePlaystyle = (value: Playstyle) => {
    try {
      localStorage.setItem(PLAYSTYLE_STORAGE_KEY, value);
    } catch {
      // Ignore storage errors.
    }
  };

  const initPlaystyleSelect = () => {
    const addOption = (value: Playstyle, label: string) => {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = label;
      toolPlaystyleSelect.appendChild(option);
    };
    toolPlaystyleSelect.innerHTML = '';
    addOption('beginner', 'Beginner');
    addOption('advanced', 'Advanced');
    playstyle = loadPlaystyle();
    toolPlaystyleSelect.value = playstyle;
  };

  const getModeLabel = (id: string): string => {
    if (id === 'unknown') return 'Unknown';
    const normalized = normalizeModeId(id) ?? id;
    if (
      normalized === 'practice' ||
      normalized === 'sprint' ||
      normalized === 'classic' ||
      normalized === 'cheese' ||
      normalized === 'charcuterie'
    ) {
      return getMode(normalized).label.toUpperCase();
    }
    return id.toUpperCase();
  };

  const refreshToolModeOptions = () => {
    toolModeSelect.innerHTML = '';
    const addOption = (value: string, label: string) => {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = label;
      toolModeSelect.appendChild(option);
    };

    addOption('all', 'All Modes');
    if (toolUsesRemote) {
      const knownModes = [
        'practice',
        'sprint',
        'classic',
        'cheese',
        'charcuterie',
      ];
      for (const modeId of knownModes) {
        addOption(modeId, getModeLabel(modeId));
      }
      addOption('unknown', 'Unknown');
      toolModeSelect.value = toolModeFilter;
      toolModeSelect.disabled = false;
      return;
    }
    const counts = new Map<string, number>();
    for (const file of toolSnapshotsAll) {
      const modeId = file.session.meta.mode?.id ?? 'unknown';
      counts.set(
        modeId,
        (counts.get(modeId) ?? 0) + file.session.samples.length,
      );
    }

    const modeIds = Array.from(counts.keys()).sort((a, b) => {
      if (a === 'unknown') return 1;
      if (b === 'unknown') return -1;
      return a.localeCompare(b);
    });
    for (const modeId of modeIds) {
      const label = `${getModeLabel(modeId)} (${counts.get(modeId) ?? 0})`;
      addOption(modeId, label);
    }

    if (toolModeFilter !== 'all' && !counts.has(toolModeFilter)) {
      toolModeFilter = 'all';
    }
    toolModeSelect.value = toolModeFilter;
    toolModeSelect.disabled = toolSnapshotsAll.length === 0;
  };

  const refreshToolTriggerOptions = () => {
    toolTriggerSelect.innerHTML = '';
    const addOption = (value: TriggerFilter, label: string) => {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = label;
      toolTriggerSelect.appendChild(option);
    };

    addOption('all', 'All Triggers');
    addOption('auto', 'Auto (lock/hold)');
    addOption('manual', 'Manual');
    addOption('constructor', 'Constructor');
    addOption('lock', 'Lock');
    addOption('hold', 'Hold');
    toolTriggerSelect.value = toolTriggerFilter;
    toolTriggerSelect.disabled =
      toolSnapshotsAll.length === 0 && !toolUsesRemote;
  };

  const refreshToolBuildOptions = () => {
    toolBuildSelect.innerHTML = '';
    const addOption = (value: string, label: string) => {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = label;
      toolBuildSelect.appendChild(option);
    };
    const hasOption = (value: string): boolean =>
      Array.from(toolBuildSelect.options).some((opt) => opt.value === value);
    const ensureBuildOption = (value: string): void => {
      if (!value || value === 'all' || hasOption(value)) return;
      addOption(value, `${value} (0)`);
    };

    addOption('all', 'All Builds');
    if (toolUsesRemote) {
      if (!remoteBuildCounts) {
        addOption('unknown', 'Unknown');
        ensureBuildOption(defaultBuildFilter);
        if (!remoteBuildLoading) {
          void fetchRemoteBuildCounts();
        }
        toolBuildSelect.value = toolBuildFilter;
        toolBuildSelect.disabled = false;
        return;
      }
      const builds = remoteBuildCounts.slice().sort((a, b) => {
        if (a.build === 'unknown') return 1;
        if (b.build === 'unknown') return -1;
        return a.build.localeCompare(b.build);
      });
      for (const entry of builds) {
        const label = `${entry.build} (${entry.count})`;
        addOption(entry.build, label);
      }
      ensureBuildOption(defaultBuildFilter);
      if (toolBuildFilter !== 'all' && !hasOption(toolBuildFilter)) {
        toolBuildFilter =
          defaultBuildFilter !== 'all' && hasOption(defaultBuildFilter)
            ? defaultBuildFilter
            : 'all';
      }
      toolBuildSelect.value = toolBuildFilter;
      toolBuildSelect.disabled = false;
      return;
    }
    const counts = new Map<string, number>();
    for (const file of toolSnapshotsAll) {
      const build = file.session.meta.buildVersion ?? 'unknown';
      counts.set(build, (counts.get(build) ?? 0) + file.session.samples.length);
    }
    const builds = Array.from(counts.keys()).sort((a, b) => {
      if (a === 'unknown') return 1;
      if (b === 'unknown') return -1;
      return a.localeCompare(b);
    });
    for (const build of builds) {
      const label = `${build} (${counts.get(build) ?? 0})`;
      addOption(build, label);
    }
    ensureBuildOption(defaultBuildFilter);

    if (toolBuildFilter !== 'all' && !hasOption(toolBuildFilter)) {
      toolBuildFilter =
        defaultBuildFilter !== 'all' && hasOption(defaultBuildFilter)
          ? defaultBuildFilter
          : 'all';
    }
    toolBuildSelect.value = toolBuildFilter;
    toolBuildSelect.disabled = toolSnapshotsAll.length === 0;
  };

  const fetchRemoteBuildCounts = async () => {
    remoteBuildLoading = true;
    try {
      const url = buildToolApiUrl('/snapshots/builds');
      const res = await fetch(url.toString());
      if (!res.ok) {
        remoteBuildCounts = null;
        return;
      }
      const payload = (await res.json()) as {
        builds?: Array<{ build?: string; count?: number }>;
      };
      remoteBuildCounts = (payload.builds ?? []).map((entry) => ({
        build:
          typeof entry.build === 'string' && entry.build.trim()
            ? entry.build
            : 'unknown',
        count:
          typeof entry.count === 'number' && Number.isFinite(entry.count)
            ? entry.count
            : 0,
      }));
    } catch {
      remoteBuildCounts = null;
    } finally {
      remoteBuildLoading = false;
      refreshToolBuildOptions();
    }
  };

  const rebuildToolSampleIndex = () => {
    toolSampleIndex = [];
    for (const file of toolSnapshots) {
      const samples = file.session.samples;
      for (let i = 0; i < samples.length; i += 1) {
        const sample = samples[i];
        const trigger = normalizeTrigger(sample);
        const matchesTrigger =
          toolTriggerFilter === 'all'
            ? true
            : toolTriggerFilter === 'auto'
              ? trigger === 'lock' || trigger === 'hold'
              : trigger === toolTriggerFilter;
        if (!matchesTrigger) continue;
        toolSampleIndex.push({ file, index: i });
      }
    }
    toolTotalSamples = toolSampleIndex.length;
  };

  const buildToolApiUrl = (path: string): URL => {
    const base = uploadBaseUrl.replace(/\/+$/, '');
    return new URL(`${base}${path}`, window.location.origin);
  };

  type RemoteSnapshotRecord = {
    snapshotId: number | null;
    sessionId: string | null;
    batchId: number | null;
    sampleIndex: number;
    session: SnapshotSession;
  };

  const parseRemoteSnapshotRecord = (
    payload: unknown,
  ): RemoteSnapshotRecord | null => {
    if (!payload || typeof payload !== 'object') return null;
    const source = payload as {
      id?: number | string;
      sessionId?: string | null;
      batchId?: number | string | null;
      meta?: {
        session?: SnapshotSession['meta'];
        sample?: Record<string, unknown>;
        trigger?: string;
      };
      board?: number[][];
    };
    const sessionMeta = source.meta?.session;
    if (!sessionMeta || !source.board) return null;
    const sampleMeta = source.meta?.sample ?? {};
    const readNumber = (value: unknown, fallback = 0): number => {
      return typeof value === 'number' && Number.isFinite(value)
        ? value
        : fallback;
    };
    const readOptionalNumber = (value: unknown): number | undefined => {
      return typeof value === 'number' && Number.isFinite(value)
        ? value
        : undefined;
    };
    const readTrigger = (value: unknown): SnapshotTrigger | undefined => {
      if (
        value === 'lock' ||
        value === 'hold' ||
        value === 'manual' ||
        value === 'constructor'
      ) {
        return value;
      }
      return undefined;
    };
    const holdValue =
      typeof sampleMeta.hold === 'number' && Number.isFinite(sampleMeta.hold)
        ? sampleMeta.hold
        : undefined;
    const sample: SnapshotSample = {
      index: readNumber(sampleMeta.index),
      timeMs: readNumber(sampleMeta.timeMs),
      board: source.board,
      hold: holdValue,
      next: Array.isArray(sampleMeta.next)
        ? sampleMeta.next
            .map((value) =>
              typeof value === 'number' && Number.isFinite(value)
                ? Math.trunc(value)
                : null,
            )
            .filter((value): value is number => value != null)
        : undefined,
      odds: Array.isArray(sampleMeta.odds)
        ? sampleMeta.odds
            .map((value) => {
              if (!value || typeof value !== 'object') return null;
              const source = value as { k?: unknown; p?: unknown };
              if (
                typeof source.k !== 'number' ||
                !Number.isFinite(source.k) ||
                typeof source.p !== 'number' ||
                !Number.isFinite(source.p)
              ) {
                return null;
              }
              return {
                k: Math.trunc(source.k),
                p: source.p,
              };
            })
            .filter(
              (
                entry,
              ): entry is {
                k: number;
                p: number;
              } => entry != null,
            )
        : undefined,
      active: sampleMeta.active as SnapshotSample['active'] | undefined,
      trigger: readTrigger(source.meta?.trigger),
      linesLeft: readOptionalNumber(sampleMeta.linesLeft),
      level: readOptionalNumber(sampleMeta.level),
      score: readOptionalNumber(sampleMeta.score),
    };
    const session: SnapshotSession = {
      meta: sessionMeta,
      samples: [sample],
    };
    const snapshotId =
      typeof source.id === 'number' && Number.isFinite(source.id)
        ? source.id
        : typeof source.id === 'string' && source.id.trim()
          ? Number.parseInt(source.id, 10)
          : null;
    const sessionId =
      typeof source.sessionId === 'string' && source.sessionId.trim()
        ? source.sessionId
        : typeof sessionMeta.id === 'string' && sessionMeta.id.trim()
          ? sessionMeta.id
          : null;
    const batchIdRaw =
      typeof source.batchId === 'number' && Number.isFinite(source.batchId)
        ? Math.trunc(source.batchId)
        : typeof source.batchId === 'string' && source.batchId.trim()
          ? Number.parseInt(source.batchId, 10)
          : null;
    const batchId = Number.isFinite(batchIdRaw)
      ? (batchIdRaw as number)
      : sampleBatchId(sample.index);
    return {
      snapshotId: Number.isFinite(snapshotId) ? snapshotId : null,
      sessionId,
      batchId,
      sampleIndex: sample.index,
      session,
    };
  };

  const fetchRemoteSamples = async (
    count: number,
  ): Promise<RemoteSnapshotRecord[]> => {
    const requestCount = Math.max(
      1,
      Math.min(REMOTE_PREFETCH_BATCH_SIZE, count),
    );
    const url = buildToolApiUrl('/snapshots/random');
    if (toolModeFilter !== 'all') {
      url.searchParams.set('mode', toolModeFilter);
    }
    if (toolTriggerFilter !== 'all') {
      url.searchParams.set('trigger', toolTriggerFilter);
    }
    if (toolBuildFilter !== 'all') {
      url.searchParams.set('build', toolBuildFilter);
    }
    if (requestCount > 1) {
      url.searchParams.set('count', String(requestCount));
    }

    const excludeSnapshotIds: number[] = [];
    const excludeSessionIds: string[] = [];
    const pushUniqueNumber = (list: number[], value: number | null): void => {
      if (value == null) return;
      if (!list.includes(value)) list.push(value);
    };
    const pushUniqueString = (list: string[], value: string | null): void => {
      if (!value) return;
      if (!list.includes(value)) list.push(value);
    };
    for (const value of recentRemoteSnapshotIds) {
      pushUniqueNumber(excludeSnapshotIds, value);
    }
    for (const value of recentRemoteLabeledSnapshotIds) {
      pushUniqueNumber(excludeSnapshotIds, value);
    }
    for (const queued of remoteSampleQueue) {
      pushUniqueNumber(excludeSnapshotIds, queued.snapshotId);
      pushUniqueString(excludeSessionIds, queued.sessionId);
    }
    for (const value of recentRemoteSessionIds) {
      pushUniqueString(excludeSessionIds, value);
    }
    if (excludeSnapshotIds.length > 0) {
      url.searchParams.set(
        'exclude_snapshot_ids',
        excludeSnapshotIds.join(','),
      );
    }
    if (excludeSessionIds.length > 0) {
      url.searchParams.set('exclude_session_ids', excludeSessionIds.join(','));
    }

    const res = await fetch(url.toString());
    if (!res.ok) return [];
    const payload = (await res.json()) as unknown;
    const items = (
      payload &&
      typeof payload === 'object' &&
      'items' in payload &&
      Array.isArray((payload as { items?: unknown }).items)
        ? ((payload as { items: unknown[] }).items ?? [])
        : [payload]
    )
      .map((entry) => parseRemoteSnapshotRecord(entry))
      .filter((entry): entry is RemoteSnapshotRecord => entry != null);
    return items;
  };

  const ensureRemotePrefetch = async (minRequired = 1): Promise<void> => {
    if (remoteSampleQueue.length >= minRequired) return;
    if (remotePrefetchPromise) {
      await remotePrefetchPromise;
      return;
    }
    const missing = Math.max(
      minRequired - remoteSampleQueue.length,
      REMOTE_PREFETCH_BATCH_SIZE,
    );
    const needed = Math.min(REMOTE_PREFETCH_MAX_QUEUE, missing);
    const version = remotePrefetchVersion;
    remotePrefetchPromise = (async () => {
      const fetched = await fetchRemoteSamples(needed);
      if (!fetched.length) return;
      if (version !== remotePrefetchVersion) return;
      const knownSnapshotIds = new Set<number>();
      for (const row of remoteSampleQueue) {
        if (row.snapshotId != null) knownSnapshotIds.add(row.snapshotId);
      }
      for (const row of fetched) {
        if (
          row.snapshotId != null &&
          (knownSnapshotIds.has(row.snapshotId) ||
            recentRemoteLabeledSnapshotIds.includes(row.snapshotId))
        ) {
          continue;
        }
        remoteSampleQueue.push(row);
        if (row.snapshotId != null) {
          knownSnapshotIds.add(row.snapshotId);
        }
        if (remoteSampleQueue.length >= REMOTE_PREFETCH_MAX_QUEUE) break;
      }
    })().finally(() => {
      remotePrefetchPromise = null;
    });
    await remotePrefetchPromise;
  };

  const applyToolFilters = async () => {
    remotePrefetchVersion += 1;
    remotePrefetchPromise = null;
    recentRemoteSnapshotIds = [];
    recentRemoteSessionIds = [];
    recentRemoteLabeledSnapshotIds = [];
    remoteSampleQueue = [];
    recentLocalSampleKeys = [];
    recentLocalSessionIds = [];
    recentLocalBatchKeys = [];

    if (toolUsesRemote) {
      const filterLabel =
        toolModeFilter === 'all' ? 'All Modes' : getModeLabel(toolModeFilter);
      const triggerText =
        toolTriggerFilter === 'all'
          ? 'All Triggers'
          : triggerLabel(toolTriggerFilter);
      const buildText =
        toolBuildFilter === 'all' ? 'All Builds' : toolBuildFilter;
      toolInputStatus.textContent =
        `Source: Online\n` +
        `Filter: ${filterLabel} / ${triggerText} / ${buildText}`;
      if (!toolActive) return;
      await ensureRemotePrefetch(1);
      await showNextToolSample();
      return;
    }
    if (toolModeFilter === 'all') {
      toolSnapshots = toolSnapshotsAll;
    } else {
      toolSnapshots = toolSnapshotsAll.filter((file) => {
        const modeId = file.session.meta.mode?.id ?? 'unknown';
        return modeId === toolModeFilter;
      });
    }
    if (toolBuildFilter !== 'all') {
      toolSnapshots = toolSnapshots.filter((file) => {
        const build = file.session.meta.buildVersion ?? 'unknown';
        return build === toolBuildFilter;
      });
    }
    rebuildToolSampleIndex();
    const filterLabel =
      toolModeFilter === 'all' ? 'All Modes' : getModeLabel(toolModeFilter);
    const triggerText =
      toolTriggerFilter === 'all'
        ? 'All Triggers'
        : triggerLabel(toolTriggerFilter);
    const buildText =
      toolBuildFilter === 'all' ? 'All Builds' : toolBuildFilter;
    toolInputStatus.textContent =
      `Source: Local\n` +
      `Loaded ${toolSnapshotsAll.length} files (${toolTotalSamplesAll} samples).\n` +
      `Filter: ${filterLabel} / ${triggerText} / ${buildText} (${toolTotalSamples} samples).`;
    await showNextToolSample();
  };

  const showNextToolSample = async (): Promise<void> => {
    if (toolUsesRemote) {
      await ensureRemotePrefetch(1);
      let remote: RemoteSnapshotRecord | null = null;
      while (remoteSampleQueue.length > 0) {
        const candidate = remoteSampleQueue.shift() ?? null;
        if (!candidate) continue;
        const seenSession =
          candidate.sessionId != null &&
          recentRemoteSessionIds.includes(candidate.sessionId);
        const seenSnapshot =
          candidate.snapshotId != null &&
          (recentRemoteSnapshotIds.includes(candidate.snapshotId) ||
            recentRemoteLabeledSnapshotIds.includes(candidate.snapshotId));
        if (seenSession || seenSnapshot) continue;
        remote = candidate;
        break;
      }
      if (!remote) {
        await ensureRemotePrefetch(1);
        while (remoteSampleQueue.length > 0) {
          const candidate = remoteSampleQueue.shift() ?? null;
          if (!candidate) continue;
          const seenSession =
            candidate.sessionId != null &&
            recentRemoteSessionIds.includes(candidate.sessionId);
          const seenSnapshot =
            candidate.snapshotId != null &&
            (recentRemoteSnapshotIds.includes(candidate.snapshotId) ||
              recentRemoteLabeledSnapshotIds.includes(candidate.snapshotId));
          if (seenSession || seenSnapshot) continue;
          remote = candidate;
          break;
        }
      }
      if (!remote || remote.session.samples.length === 0) {
        currentSample = null;
        updateToolSampleStatus('Sample: -');
        if (toolActive) {
          canvas.clear();
          toolUi.setQueueOddsMode(false);
          toolUi.setQueueProbabilities([]);
        }
        return;
      }
      const session = remote.session;
      const sample = session.samples[0];
      const rawBoard = sample.board;
      const board = decodeBoard(rawBoard, session.meta.pieceOrder);
      const hold = decodeHold(sample.hold, session.meta.pieceOrder);
      const active = decodeActive(
        sample.active,
        board,
        session.meta.pieceOrder,
      );
      const queuePieces = decodeQueuePieces(sample, session.meta.pieceOrder);
      const queueOdds = decodeQueueOdds(sample, session.meta.pieceOrder);
      const fileName = `remote:${session.meta.id ?? 'unknown'}`;
      const sessionId =
        remote.sessionId ??
        (typeof session.meta.id === 'string' ? session.meta.id : null);
      pushRecentUnique(
        recentRemoteSnapshotIds,
        remote.snapshotId,
        REMOTE_RECENT_SNAPSHOT_WINDOW,
      );
      pushRecentUnique(
        recentRemoteSessionIds,
        sessionId,
        REMOTE_RECENT_SESSION_WINDOW,
      );
      currentSample = {
        file: { name: fileName, session },
        index: sample.index,
        board,
        raw: rawBoard,
        hold,
        active,
        queuePieces,
        queueOdds,
        snapshotId: remote.snapshotId ?? undefined,
        timeMs: sample.timeMs,
        trigger: sample.trigger,
        linesLeft: sample.linesLeft,
        level: sample.level,
        score: sample.score,
      };
      const key = `${fileName}#${sample.index}`;
      labelIndex[key] = (labelIndex[key] ?? 0) + 1;
      updateToolSampleStatus(
        `Source: ${fileName} Sample: ${sample.index} Shown: ${labelIndex[key]}`,
      );
      clearLabelSelection();
      if (toolActive) {
        canvas.render(board, hold, undefined, active, queuePieces);
        toolUi.setQueueOddsMode(queueOdds.length > 0);
        toolUi.setQueueProbabilities(queueOdds);
      }
      if (remoteSampleQueue.length < REMOTE_PREFETCH_MIN_QUEUE) {
        void ensureRemotePrefetch(REMOTE_PREFETCH_MIN_QUEUE);
      }
      return;
    }
    if (toolSnapshots.length === 0 || toolTotalSamples === 0) {
      currentSample = null;
      updateToolSampleStatus('Sample: -');
      if (toolActive) {
        canvas.clear();
        toolUi.setQueueOddsMode(false);
        toolUi.setQueueProbabilities([]);
      }
      return;
    }
    let target: (typeof toolSampleIndex)[number] | undefined;
    for (let attempt = 0; attempt < MAX_PICK_ATTEMPTS; attempt += 1) {
      const pick = Math.floor(Math.random() * toolTotalSamples);
      const candidate = toolSampleIndex[pick];
      if (!candidate) continue;
      const file = candidate.file;
      const sessionId = file.session.meta.id ?? file.name;
      const sampleIndex = candidate.index;
      const key = `${file.name}#${sampleIndex}`;
      const batchId = sampleBatchId(sampleIndex);
      const batchKey =
        batchId == null ? null : `${sessionId}::b${Math.trunc(batchId)}`;
      const seenSample = recentLocalSampleKeys.includes(key);
      const seenSession = recentLocalSessionIds.includes(sessionId);
      const seenBatch =
        batchKey != null && recentLocalBatchKeys.includes(batchKey);
      if (!seenSample && !seenSession && !seenBatch) {
        target = candidate;
        break;
      }
      if (!target && !seenSample) {
        target = candidate;
      }
    }
    if (!target) {
      const pick = Math.floor(Math.random() * toolTotalSamples);
      target = toolSampleIndex[pick];
    }
    if (!target) {
      currentSample = null;
      updateToolSampleStatus('Sample: -');
      if (toolActive) {
        canvas.clear();
        toolUi.setQueueOddsMode(false);
        toolUi.setQueueProbabilities([]);
      }
      return;
    }
    const file = target.file;
    const indexInFile = target.index;
    const sample = file.session.samples[indexInFile];
    const rawBoard = sample.board;
    const board = decodeBoard(rawBoard, file.session.meta.pieceOrder);
    const hold = decodeHold(sample.hold, file.session.meta.pieceOrder);
    const active = decodeActive(
      sample.active,
      board,
      file.session.meta.pieceOrder,
    );
    const queuePieces = decodeQueuePieces(sample, file.session.meta.pieceOrder);
    const queueOdds = decodeQueueOdds(sample, file.session.meta.pieceOrder);
    const sessionId = file.session.meta.id ?? file.name;
    const batchId = sampleBatchId(indexInFile);
    pushRecentUnique(
      recentLocalSampleKeys,
      `${file.name}#${indexInFile}`,
      LOCAL_RECENT_SAMPLE_WINDOW,
    );
    pushRecentUnique(
      recentLocalSessionIds,
      sessionId,
      LOCAL_RECENT_SESSION_WINDOW,
    );
    if (batchId != null) {
      pushRecentUnique(
        recentLocalBatchKeys,
        `${sessionId}::b${batchId}`,
        LOCAL_RECENT_SAMPLE_WINDOW,
      );
    }
    currentSample = {
      file,
      index: indexInFile,
      board,
      raw: rawBoard,
      hold,
      active,
      queuePieces,
      queueOdds,
      timeMs: sample.timeMs,
      trigger: sample.trigger,
      linesLeft: sample.linesLeft,
      level: sample.level,
      score: sample.score,
    };
    const key = `${file.name}#${indexInFile}`;
    labelIndex[key] = (labelIndex[key] ?? 0) + 1;
    await saveLabelIndex();
    updateToolSampleStatus(
      `File: ${file.name} Sample: ${indexInFile} Shown: ${labelIndex[key]}`,
    );
    clearLabelSelection();
    if (toolActive) {
      canvas.render(board, hold, undefined, active, queuePieces);
      toolUi.setQueueOddsMode(queueOdds.length > 0);
      toolUi.setQueueProbabilities(queueOdds);
    }
  };

  toolInputButton.addEventListener('click', async () => {
    if (toolUsesRemote) return;
    const picker = (
      window as Window & {
        showDirectoryPicker?: () => Promise<FileSystemDirectoryHandle>;
      }
    ).showDirectoryPicker;
    if (!picker) {
      toolInputStatus.textContent = 'Folder access not supported.';
      return;
    }
    try {
      const root = await picker();
      const granted = await requestDirectoryAccess(root, 'read');
      if (!granted) {
        toolInputStatus.textContent = 'Folder access denied.';
        return;
      }
      const files: Array<{ name: string; session: SnapshotSession }> = [];
      for await (const entry of root.values()) {
        if (!isFileHandle(entry)) continue;
        if (!entry.name.endsWith('.json')) continue;
        const file = await entry.getFile();
        const text = await file.text();
        const session = JSON.parse(text) as SnapshotSession;
        if (!session?.samples?.length) continue;
        files.push({ name: entry.name, session });
      }
      files.sort((a, b) => a.name.localeCompare(b.name));
      toolSnapshotsAll = files;
      toolTotalSamplesAll = toolSnapshotsAll.reduce(
        (sum, file) => sum + file.session.samples.length,
        0,
      );
      refreshToolModeOptions();
      refreshToolTriggerOptions();
      refreshToolBuildOptions();
      await applyToolFilters();
      updateToolActionStatus(`Loaded ${toolSnapshotsAll.length} files.`);
    } catch {
      toolInputStatus.textContent = 'Folder selection cancelled.';
    }
  });

  toolModeSelect.addEventListener('change', async () => {
    toolModeFilter = toolModeSelect.value;
    await applyToolFilters();
  });

  toolTriggerSelect.addEventListener('change', async () => {
    toolTriggerFilter = toolTriggerSelect.value as TriggerFilter;
    await applyToolFilters();
  });

  toolBuildSelect.addEventListener('change', async () => {
    toolBuildFilter = toolBuildSelect.value;
    await applyToolFilters();
  });

  if (toolUsesRemote) {
    refreshToolModeOptions();
    refreshToolTriggerOptions();
    refreshToolBuildOptions();
  }

  initPlaystyleSelect();
  toolPlaystyleSelect.addEventListener('change', () => {
    playstyle =
      toolPlaystyleSelect.value === 'advanced' ? 'advanced' : 'beginner';
    savePlaystyle(playstyle);
  });

  toolOutputButton.addEventListener('click', async () => {
    if (toolUsesRemote) return;
    const picker = (
      window as Window & {
        showDirectoryPicker?: () => Promise<FileSystemDirectoryHandle>;
      }
    ).showDirectoryPicker;
    if (!picker) {
      toolOutputStatus.textContent = 'Folder access not supported.';
      return;
    }
    try {
      toolOutputDirHandle = await picker();
      const granted = await requestDirectoryAccess(
        toolOutputDirHandle,
        'readwrite',
      );
      if (!granted) {
        toolOutputStatus.textContent = 'Folder access denied.';
        toolOutputDirHandle = null;
        return;
      }
      await loadLabelIndex();
      toolOutputStatus.textContent = `Output: ${toolOutputDirHandle.name}`;
      updateToolActionStatus(`Output set: ${toolOutputDirHandle.name}`);
    } catch {
      toolOutputStatus.textContent = 'Folder selection cancelled.';
    }
  });

  toolNextButton.addEventListener('click', async () => {
    if (toolBusy) return;
    if (!currentSample) {
      updateToolSampleStatus('Sample: -');
      return;
    }
    if (!uploadClient.isRemote && !toolOutputDirHandle) {
      toolOutputStatus.textContent = 'Select output folder first.';
      return;
    }
    toolBusy = true;
    try {
      const key = `${currentSample.file.name}#${currentSample.index}`;
      const sessionMeta = currentSample.file.session.meta;
      const sampleMeta = {
        index: currentSample.index,
        timeMs: currentSample.timeMs,
        trigger: currentSample.trigger,
        ...(currentSample.active ? { active: currentSample.active } : {}),
        linesLeft: currentSample.linesLeft,
        level: currentSample.level,
        score: currentSample.score,
      };
      const record = {
        createdAt: new Date().toISOString(),
        source: {
          file: currentSample.file.name,
          sessionId: currentSample.file.session.meta.id,
          sampleIndex: currentSample.index,
          snapshotId: currentSample.snapshotId,
          shownCount: labelIndex[key] ?? 1,
        },
        snapshot_meta: {
          session: sessionMeta,
          sample: sampleMeta,
        },
        board: encodeBoardString(currentSample.raw),
        hold: currentSample.hold,
        labels: [...selectedLabels],
        label_context: {
          playstyle,
          mode_filter: toolModeFilter,
          trigger_filter: toolTriggerFilter,
          build_filter: toolBuildFilter,
        },
      };
      if (uploadClient.isRemote) {
        await uploadClient.enqueueLabel({
          createdAt: record.createdAt,
          data: record,
        });
        pushRecentUnique(
          recentRemoteLabeledSnapshotIds,
          currentSample.snapshotId ?? null,
          REMOTE_LABELED_SNAPSHOT_WINDOW,
        );
        await uploadClient.flush();
        updateToolActionStatus(
          `Uploaded label for ${currentSample.file.name} #${currentSample.index}`,
        );
      } else {
        await appendJsonl(
          toolOutputDirHandle!,
          'labels.jsonl',
          JSON.stringify(record),
        );
        updateToolActionStatus(
          `Saved label for ${currentSample.file.name} #${currentSample.index}`,
        );
      }
      await showNextToolSample();
    } finally {
      toolBusy = false;
    }
  });

  toolBackButton.addEventListener('click', () => onBack());

  const renderToolSample = async () => {
    if (!currentSample) {
      if (toolUsesRemote || toolSnapshots.length > 0) {
        await showNextToolSample();
      } else {
        canvas.clear();
        toolUi.setQueueOddsMode(false);
        toolUi.setQueueProbabilities([]);
      }
      return;
    }
    canvas.render(
      currentSample.board,
      currentSample.hold,
      undefined,
      currentSample.active ?? null,
      currentSample.queuePieces,
    );
    toolUi.setQueueOddsMode(currentSample.queueOdds.length > 0);
    toolUi.setQueueProbabilities(currentSample.queueOdds);
  };

  return {
    id: 'labeling',
    label: 'Wish Upon a Block',
    root: toolUi.root,
    setPiecePalette: toolUi.setPiecePalette,
    enter: async () => {
      toolActive = true;
      if (toolUsesRemote) {
        refreshToolModeOptions();
        refreshToolTriggerOptions();
        await applyToolFilters();
      } else {
        await renderToolSample();
      }
    },
    leave: () => {
      toolActive = false;
    },
  };
}
