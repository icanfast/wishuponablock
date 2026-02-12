import { getMode, normalizeModeId } from '../../core/modes';
import type {
  SnapshotSession,
  SnapshotSample,
  SnapshotTrigger,
} from '../../core/snapshotRecorder';
import type { UploadClient } from '../../core/uploadClient';
import { PIECES, type Board, type PieceKind } from '../../core/types';
import { createToolScreen } from '../screens/toolScreen';
import type { ToolController } from './toolHost';
import type { ToolCanvas } from './toolCanvas';

type LabelingToolOptions = {
  toolUsesRemote: boolean;
  uploadClient: UploadClient;
  uploadBaseUrl: string;
  canvas: ToolCanvas;
  onBack: () => void;
};

type Playstyle = 'beginner' | 'advanced';

const PLAYSTYLE_STORAGE_KEY = 'wub.labeler.playstyle';

export function createLabelingTool(
  options: LabelingToolOptions,
): ToolController {
  const { toolUsesRemote, uploadClient, uploadBaseUrl, canvas, onBack } =
    options;

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
  let toolBuildFilter = 'all';
  let toolActive = false;
  let playstyle: Playstyle = 'beginner';
  let currentSample: {
    file: { name: string; session: SnapshotSession };
    index: number;
    board: Board;
    raw: number[][];
    hold: PieceKind | null;
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

  const decodeHold = (
    hold: number | undefined,
    order?: readonly string[],
  ): PieceKind | null => {
    if (!hold || hold <= 0) return null;
    const resolvedOrder = order ?? PIECES;
    const piece = resolvedOrder[hold - 1] ?? PIECES[hold - 1];
    return (piece as PieceKind) ?? null;
  };

  const encodeBoardString = (raw: number[][]): string =>
    raw.map((row) => row.join('')).join('/');

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

    addOption('all', 'All Builds');
    if (toolUsesRemote) {
      if (!remoteBuildCounts) {
        addOption('unknown', 'Unknown');
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
      if (
        toolBuildFilter !== 'all' &&
        !builds.some((entry) => entry.build === toolBuildFilter)
      ) {
        toolBuildFilter = 'all';
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

    if (toolBuildFilter !== 'all' && !counts.has(toolBuildFilter)) {
      toolBuildFilter = 'all';
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

  const fetchRemoteSample = async (): Promise<{
    snapshotId: number | null;
    session: SnapshotSession;
  } | null> => {
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
    const res = await fetch(url.toString());
    if (!res.ok) {
      return null;
    }
    const payload = (await res.json()) as {
      id?: number | string;
      createdAt?: string;
      meta?: {
        session?: SnapshotSession['meta'];
        sample?: Record<string, unknown>;
        trigger?: string;
      };
      board?: number[][];
    };
    const sessionMeta = payload.meta?.session;
    if (!sessionMeta || !payload.board) return null;
    const sampleMeta = payload.meta?.sample ?? {};
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
    const sample = {
      index: readNumber(sampleMeta.index),
      timeMs: readNumber(sampleMeta.timeMs),
      board: payload.board,
      hold: holdValue,
      trigger: readTrigger(payload.meta?.trigger),
      linesLeft: readOptionalNumber(sampleMeta.linesLeft),
      level: readOptionalNumber(sampleMeta.level),
      score: readOptionalNumber(sampleMeta.score),
    };
    const session: SnapshotSession = {
      meta: sessionMeta,
      samples: [sample],
    };
    const snapshotId =
      typeof payload.id === 'number' && Number.isFinite(payload.id)
        ? payload.id
        : typeof payload.id === 'string' && payload.id.trim()
          ? Number.parseInt(payload.id, 10)
          : null;
    return {
      snapshotId: Number.isFinite(snapshotId) ? snapshotId : null,
      session,
    };
  };

  const applyToolFilters = async () => {
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
      const remote = await fetchRemoteSample();
      if (!remote || remote.session.samples.length === 0) {
        currentSample = null;
        updateToolSampleStatus('Sample: -');
        if (toolActive) {
          canvas.clear();
        }
        return;
      }
      const session = remote.session;
      const sample = session.samples[0];
      const rawBoard = sample.board;
      const board = decodeBoard(rawBoard, session.meta.pieceOrder);
      const hold = decodeHold(sample.hold, session.meta.pieceOrder);
      const fileName = `remote:${session.meta.id ?? 'unknown'}`;
      currentSample = {
        file: { name: fileName, session },
        index: sample.index,
        board,
        raw: rawBoard,
        hold,
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
        canvas.render(board, hold);
      }
      return;
    }
    if (toolSnapshots.length === 0 || toolTotalSamples === 0) {
      currentSample = null;
      updateToolSampleStatus('Sample: -');
      if (toolActive) {
        canvas.clear();
      }
      return;
    }
    const pick = Math.floor(Math.random() * toolTotalSamples);
    const target = toolSampleIndex[pick];
    if (!target) {
      updateToolSampleStatus('Sample: -');
      return;
    }
    const file = target.file;
    const indexInFile = target.index;
    const sample = file.session.samples[indexInFile];
    const rawBoard = sample.board;
    const board = decodeBoard(rawBoard, file.session.meta.pieceOrder);
    const hold = decodeHold(sample.hold, file.session.meta.pieceOrder);
    currentSample = {
      file,
      index: indexInFile,
      board,
      raw: rawBoard,
      hold,
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
      canvas.render(board, hold);
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
      }
      return;
    }
    canvas.render(currentSample.board, currentSample.hold);
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
