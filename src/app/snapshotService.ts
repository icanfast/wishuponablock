import type { SettingsStore } from '../core/settingsStore';
import { ML_MODEL_URL } from '../core/constants';
import {
  SnapshotRecorder,
  downloadSnapshotSession,
  saveSnapshotSessionToDirectory,
  type SnapshotTrigger,
  type SnapshotModeInfo,
  type SnapshotSession,
} from '../core/snapshotRecorder';
import type { UploadClient, SnapshotUploadRecord } from '../core/uploadClient';
import type { Board, PieceKind } from '../core/types';
import type { IdentityService } from './identityService';
import { usesModelGenerator } from '../core/generators';

export type SnapshotUiState = {
  folderStatus: string;
  recordButtonLabel: string;
  discardVisible: boolean;
  recordStatus: string;
  isRecording: boolean;
  sampleCount: number;
  enabled: boolean;
};

type SnapshotServiceOptions = {
  settingsStore: SettingsStore;
  rows: number;
  cols: number;
  uploadClient: UploadClient;
  useRemoteUpload: boolean;
  identityService: IdentityService;
  buildVersion?: string;
  onStateChange?: (state: SnapshotUiState) => void;
};

export type SnapshotService = {
  getState: () => SnapshotUiState;
  isRecording: () => boolean;
  setModeInfo: (mode: SnapshotModeInfo) => void;
  setComment: (comment: string) => void;
  ensureDirectory: () => Promise<boolean>;
  start: () => void;
  stop: (options?: { promptForFolder?: boolean }) => Promise<void>;
  restart: () => void;
  discard: () => void;
  handleLock: (
    board: Board,
    hold: PieceKind | null,
    meta?: { linesLeft?: number; level?: number; score?: number },
  ) => void;
  handleHold: (
    board: Board,
    hold: PieceKind | null,
    meta?: { linesLeft?: number; level?: number; score?: number },
  ) => void;
  handleManual: (
    board: Board,
    hold: PieceKind | null,
    meta?: { linesLeft?: number; level?: number; score?: number },
  ) => void;
  setRemoteSnapshotBankCount: (count: number | null) => void;
  flushRemoteUploads: () => Promise<void>;
  dispose: () => void;
};

const REMOTE_SNAPSHOT_BATCH_SIZE = 20;
const FULL_CAPTURE_THRESHOLD = 15_000;
const HALF_CAPTURE_THRESHOLD = 25_000;

const getDirectoryPicker = () =>
  (
    window as Window & {
      showDirectoryPicker?: () => Promise<FileSystemDirectoryHandle>;
    }
  ).showDirectoryPicker;

export function createSnapshotService(
  options: SnapshotServiceOptions,
): SnapshotService {
  const {
    settingsStore,
    rows,
    cols,
    uploadClient,
    useRemoteUpload,
    identityService,
    buildVersion,
  } = options;
  const recorder = new SnapshotRecorder();
  let snapshotDirHandle: FileSystemDirectoryHandle | null = null;
  let folderStatus = 'No folder selected.';
  let currentComment = '';
  let currentMode: SnapshotModeInfo = { id: 'practice', options: {} };
  let lastSnapshotKey: string | null = null;
  let recordStatusOverride: { message: string; timeoutId: number } | null =
    null;
  let snapshotsEnabled = settingsStore.get().privacy.shareSnapshots;
  const pendingRemoteSnapshots: SnapshotUploadRecord[] = [];
  let remoteFlushInProgress = false;
  let remoteSnapshotBankCount: number | null = null;
  let activeRemoteSamplingStride = 1;

  const notify = () => {
    options.onStateChange?.(getState());
  };

  const applyIdentityMeta = () => {
    const meta = recorder.sessionMeta;
    if (!meta) return;
    meta.device_id = identityService.getDeviceId();
    const userId = identityService.getUserId();
    if (userId) {
      meta.user_id = userId;
    } else {
      delete meta.user_id;
    }
  };

  const setFolderStatus = (text: string) => {
    folderStatus = text;
    notify();
  };

  const setRecordStatusTransient = (message: string, durationMs = 2000) => {
    if (recordStatusOverride) {
      window.clearTimeout(recordStatusOverride.timeoutId);
    }
    recordStatusOverride = {
      message,
      timeoutId: window.setTimeout(() => {
        recordStatusOverride = null;
        notify();
      }, durationMs),
    };
    notify();
  };

  const getState = (): SnapshotUiState => {
    const isRecording = snapshotsEnabled && recorder.isRecording;
    const recordButtonLabel = snapshotsEnabled
      ? isRecording
        ? snapshotDirHandle
          ? 'Stop & Save'
          : 'Stop & Download'
        : 'Start Recording'
      : 'Snapshots Disabled';
    return {
      folderStatus: snapshotsEnabled
        ? folderStatus
        : 'Snapshots disabled in Options.',
      recordButtonLabel,
      discardVisible: isRecording,
      recordStatus:
        recordStatusOverride?.message ??
        (snapshotsEnabled
          ? isRecording
            ? `Samples: ${recorder.sampleCount}`
            : 'Idle'
          : 'Snapshots disabled in Options.'),
      isRecording,
      sampleCount: snapshotsEnabled ? recorder.sampleCount : 0,
      enabled: snapshotsEnabled,
    };
  };

  const requestDirectoryAccess = async (
    handle: FileSystemDirectoryHandle,
  ): Promise<boolean> => {
    let permission: PermissionState = 'granted';
    if (handle.queryPermission) {
      permission = await handle.queryPermission({ mode: 'readwrite' });
    }
    if (permission !== 'granted' && handle.requestPermission) {
      permission = await handle.requestPermission({ mode: 'readwrite' });
    }
    return permission === 'granted';
  };

  const ensureDirectory = async (): Promise<boolean> => {
    if (!snapshotsEnabled) {
      setFolderStatus('Snapshots disabled in Options.');
      return false;
    }
    const picker = getDirectoryPicker();
    if (!picker) {
      setFolderStatus('Folder access not supported in this browser.');
      snapshotDirHandle = null;
      return false;
    }

    try {
      if (!snapshotDirHandle) {
        snapshotDirHandle = await picker();
      }

      const permission = await requestDirectoryAccess(snapshotDirHandle);
      if (!permission) {
        setFolderStatus('Folder access denied.');
        snapshotDirHandle = null;
        return false;
      }

      setFolderStatus(`Folder: ${snapshotDirHandle.name}`);
      notify();
      return true;
    } catch {
      setFolderStatus('Folder selection cancelled.');
      return false;
    }
  };

  const stopAndPersist = async (
    session: SnapshotSession,
    options: { promptForFolder?: boolean } = {},
  ) => {
    if (useRemoteUpload) return;
    if (options.promptForFolder && !snapshotDirHandle) {
      const ready = await ensureDirectory();
      if (!ready) {
        setFolderStatus('Auto-save unavailable. Downloading instead.');
      }
    }

    if (snapshotDirHandle) {
      void saveSnapshotSessionToDirectory(session, snapshotDirHandle)
        .then(() => setFolderStatus(`Saved: ${session.meta.id}`))
        .catch(() => {
          setFolderStatus('Save failed. Downloading instead.');
          downloadSnapshotSession(session);
        });
      return;
    }

    downloadSnapshotSession(session);
    setFolderStatus(`Downloaded: ${session.meta.id}`);
  };

  const normalizeCount = (value: number | null): number | null => {
    if (value == null || !Number.isFinite(value)) return null;
    return Math.max(0, Math.trunc(value));
  };

  const samplingStrideForCount = (count: number | null): number => {
    if (count == null) return 1;
    if (count < FULL_CAPTURE_THRESHOLD) return 1;
    if (count < HALF_CAPTURE_THRESHOLD) return 2;
    return 4;
  };

  const setRemoteSnapshotBankCount = (count: number | null): void => {
    remoteSnapshotBankCount = normalizeCount(count);
  };

  const flushRemoteUploads = async (): Promise<void> => {
    if (!useRemoteUpload) return;
    if (remoteFlushInProgress) return;
    remoteFlushInProgress = true;
    try {
      while (pendingRemoteSnapshots.length > 0) {
        const batch = pendingRemoteSnapshots.splice(
          0,
          REMOTE_SNAPSHOT_BATCH_SIZE,
        );
        await uploadClient.enqueueSnapshotBatch(batch);
      }
      await uploadClient.flush();
    } finally {
      remoteFlushInProgress = false;
      if (pendingRemoteSnapshots.length > 0) {
        void flushRemoteUploads();
      }
    }
  };

  const start = () => {
    if (!snapshotsEnabled) {
      notify();
      return;
    }
    if (recorder.isRecording) return;
    lastSnapshotKey = null;
    const session = recorder.start(
      settingsStore.get(),
      rows,
      cols,
      currentComment,
      currentMode,
    );
    if (buildVersion) {
      session.meta.buildVersion = buildVersion;
    }
    if (usesModelGenerator(session.meta.settings.generator.type)) {
      session.meta.model_url = ML_MODEL_URL;
    } else {
      delete session.meta.model_url;
    }
    activeRemoteSamplingStride = samplingStrideForCount(
      remoteSnapshotBankCount,
    );
    if (useRemoteUpload) {
      console.info(
        `[Snapshot] build_count=${remoteSnapshotBankCount ?? 'unknown'} stride=${activeRemoteSamplingStride}`,
      );
    }
    applyIdentityMeta();
    notify();
  };

  const stop = async (options: { promptForFolder?: boolean } = {}) => {
    if (!recorder.isRecording) {
      if (useRemoteUpload) {
        await flushRemoteUploads();
      }
      return;
    }
    const session = recorder.stop();
    notify();
    lastSnapshotKey = null;
    if (useRemoteUpload) {
      await flushRemoteUploads();
      return;
    }
    if (!session || session.samples.length === 0) return;
    await stopAndPersist(session, options);
  };

  const restart = () => {
    void stop();
    start();
  };

  const discard = () => {
    recorder.discard();
    lastSnapshotKey = null;
    notify();
  };

  const buildSnapshotKey = (board: Board, hold: PieceKind | null): string => {
    const rowsKey = board
      .map((row) => row.map((cell) => (cell ? cell : '.')).join(''))
      .join('/');
    return `${rowsKey}:${hold ?? 'none'}`;
  };

  const enqueueSnapshotSample = (
    board: Board,
    hold: PieceKind | null,
    reason: SnapshotTrigger,
    meta?: { linesLeft?: number; level?: number; score?: number },
  ) => {
    if (!snapshotsEnabled) return;
    if (!recorder.isRecording) {
      if (reason === 'manual') {
        console.warn('[Snapshot] Manual capture ignored (not recording).');
      }
      return;
    }
    const key = buildSnapshotKey(board, hold);
    if (reason === 'lock' && key === lastSnapshotKey) {
      console.warn(`[Snapshot] Skipping duplicate (${reason}).`);
      return;
    }
    lastSnapshotKey = key;
    const sample = recorder.record(board, hold, {
      store: !useRemoteUpload,
      trigger: reason,
      linesLeft: meta?.linesLeft,
      level: meta?.level,
      score: meta?.score,
    });
    notify();
    if (reason === 'manual' && !useRemoteUpload) {
      setRecordStatusTransient('Board saved manually.');
    }
    if (!sample || !useRemoteUpload) return;
    const isAutomaticTrigger = reason === 'lock';
    const shouldUploadSample =
      !isAutomaticTrigger ||
      activeRemoteSamplingStride <= 1 ||
      sample.index % activeRemoteSamplingStride === 0;
    if (!shouldUploadSample) return;
    const session = recorder.sessionMeta;
    if (!session) return;
    const payload: SnapshotUploadRecord = {
      createdAt: new Date().toISOString(),
      meta: {
        session,
        sample: {
          index: sample.index,
          timeMs: sample.timeMs,
          hold: sample.hold,
          ...(sample.linesLeft != null ? { linesLeft: sample.linesLeft } : {}),
          ...(sample.level != null ? { level: sample.level } : {}),
          ...(sample.score != null ? { score: sample.score } : {}),
        },
        trigger: reason,
      },
      board: sample.board,
    };
    console.info(
      `[Snapshot] ${reason} session=${session.id} index=${sample.index}`,
    );
    pendingRemoteSnapshots.push(payload);
    const shouldFlushNow =
      reason === 'manual' ||
      pendingRemoteSnapshots.length >= REMOTE_SNAPSHOT_BATCH_SIZE;
    if (shouldFlushNow) {
      void flushRemoteUploads();
    }
  };

  const beforeUnload = () => {
    if (recorder.isRecording) {
      recorder.stop();
    }
    if (useRemoteUpload) {
      void flushRemoteUploads();
    }
  };

  window.addEventListener('beforeunload', beforeUnload);
  uploadClient.setOnSnapshotUploaded((record) => {
    const meta = record.meta as { trigger?: string } | undefined;
    if (meta?.trigger === 'manual') {
      setRecordStatusTransient('Board saved manually.');
    }
  });
  const unsubscribeIdentity = identityService.subscribe(() => {
    applyIdentityMeta();
  });
  const unsubscribeSettings = settingsStore.subscribe((settings) => {
    const nextEnabled = settings.privacy.shareSnapshots;
    if (nextEnabled === snapshotsEnabled) return;
    snapshotsEnabled = nextEnabled;
    if (!snapshotsEnabled && recorder.isRecording) {
      recorder.stop();
      lastSnapshotKey = null;
      if (useRemoteUpload) {
        void flushRemoteUploads();
      }
    }
    notify();
  });

  notify();

  return {
    getState,
    isRecording: () => recorder.isRecording,
    setModeInfo: (mode) => {
      currentMode = { ...mode, options: { ...(mode.options ?? {}) } };
    },
    setComment: (comment) => {
      currentComment = comment;
      if (recorder.isRecording) {
        recorder.setComment(comment);
      }
    },
    ensureDirectory,
    start,
    stop,
    restart,
    discard,
    handleLock: (board, hold, meta) =>
      enqueueSnapshotSample(board, hold, 'lock', meta),
    handleHold: (board, hold, meta) =>
      enqueueSnapshotSample(board, hold, 'hold', meta),
    handleManual: (board, hold, meta) =>
      enqueueSnapshotSample(board, hold, 'manual', meta),
    setRemoteSnapshotBankCount,
    flushRemoteUploads,
    dispose: () => {
      if (useRemoteUpload) {
        void flushRemoteUploads();
      }
      window.removeEventListener('beforeunload', beforeUnload);
      uploadClient.setOnSnapshotUploaded(null);
      unsubscribeIdentity();
      unsubscribeSettings();
    },
  };
}
