import type { SettingsStore } from '../core/settingsStore';
import {
  SnapshotRecorder,
  downloadSnapshotSession,
  saveSnapshotSessionToDirectory,
  type SnapshotTrigger,
  type SnapshotModeInfo,
  type SnapshotSession,
} from '../core/snapshotRecorder';
import type { UploadClient } from '../core/uploadClient';
import type { Board, PieceKind } from '../core/types';
import type { IdentityService } from './identityService';

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
  handleLock: (board: Board, hold: PieceKind | null) => void;
  handleHold: (board: Board, hold: PieceKind | null) => void;
  handleManual: (board: Board, hold: PieceKind | null) => void;
  dispose: () => void;
};

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
  } = options;
  const recorder = new SnapshotRecorder();
  let snapshotDirHandle: FileSystemDirectoryHandle | null = null;
  let folderStatus = 'No folder selected.';
  let currentComment = '';
  let currentMode: SnapshotModeInfo = { id: 'default', options: {} };
  let lastSnapshotKey: string | null = null;
  let recordStatusOverride: { message: string; timeoutId: number } | null =
    null;
  let snapshotsEnabled = settingsStore.get().privacy.shareSnapshots;

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

  const start = () => {
    if (!snapshotsEnabled) {
      notify();
      return;
    }
    if (recorder.isRecording) return;
    lastSnapshotKey = null;
    recorder.start(
      settingsStore.get(),
      rows,
      cols,
      currentComment,
      currentMode,
    );
    applyIdentityMeta();
    notify();
  };

  const stop = async (options: { promptForFolder?: boolean } = {}) => {
    if (!recorder.isRecording) return;
    const session = recorder.stop();
    notify();
    lastSnapshotKey = null;
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
  ) => {
    if (!snapshotsEnabled) return;
    if (!recorder.isRecording) {
      if (reason === 'manual') {
        console.warn('[Snapshot] Manual capture ignored (not recording).');
      }
      return;
    }
    const key = buildSnapshotKey(board, hold);
    if (reason !== 'manual' && key === lastSnapshotKey) {
      console.warn(`[Snapshot] Skipping duplicate (${reason}).`);
      return;
    }
    lastSnapshotKey = key;
    const sample = recorder.record(board, hold, {
      store: !useRemoteUpload,
      trigger: reason,
    });
    notify();
    if (reason === 'manual' && !useRemoteUpload) {
      setRecordStatusTransient('Board saved manually.');
    }
    if (!sample || !useRemoteUpload) return;
    const session = recorder.sessionMeta;
    if (!session) return;
    const payload = {
      createdAt: new Date().toISOString(),
      meta: {
        session,
        sample: {
          index: sample.index,
          timeMs: sample.timeMs,
          hold: sample.hold,
        },
        trigger: reason,
      },
      board: sample.board,
    };
    console.info(
      `[Snapshot] ${reason} session=${session.id} index=${sample.index}`,
    );
    void uploadClient.enqueueSnapshot(payload);
  };

  const beforeUnload = () => {
    if (recorder.isRecording) {
      recorder.stop();
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
    handleLock: (board, hold) => enqueueSnapshotSample(board, hold, 'lock'),
    handleHold: (board, hold) => enqueueSnapshotSample(board, hold, 'hold'),
    handleManual: (board, hold) => enqueueSnapshotSample(board, hold, 'manual'),
    dispose: () => {
      window.removeEventListener('beforeunload', beforeUnload);
      uploadClient.setOnSnapshotUploaded(null);
      unsubscribeIdentity();
      unsubscribeSettings();
    },
  };
}
