import type { SettingsStore } from '../core/settingsStore';
import {
  SnapshotRecorder,
  downloadSnapshotSession,
  saveSnapshotSessionToDirectory,
  type SnapshotModeInfo,
  type SnapshotSession,
} from '../core/snapshotRecorder';
import type { UploadClient } from '../core/uploadClient';
import type { Board, PieceKind } from '../core/types';

export type SnapshotUiState = {
  folderStatus: string;
  recordButtonLabel: string;
  discardVisible: boolean;
  recordStatus: string;
  isRecording: boolean;
  sampleCount: number;
};

type SnapshotServiceOptions = {
  settingsStore: SettingsStore;
  rows: number;
  cols: number;
  uploadClient: UploadClient;
  useRemoteUpload: boolean;
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
  const { settingsStore, rows, cols, uploadClient, useRemoteUpload } = options;
  const recorder = new SnapshotRecorder();
  let snapshotDirHandle: FileSystemDirectoryHandle | null = null;
  let folderStatus = 'No folder selected.';
  let currentComment = '';
  let currentMode: SnapshotModeInfo = { id: 'default', options: {} };
  let lastSnapshotKey: string | null = null;

  const notify = () => {
    options.onStateChange?.(getState());
  };

  const setFolderStatus = (text: string) => {
    folderStatus = text;
    notify();
  };

  const getState = (): SnapshotUiState => {
    const isRecording = recorder.isRecording;
    const recordButtonLabel = isRecording
      ? snapshotDirHandle
        ? 'Stop & Save'
        : 'Stop & Download'
      : 'Start Recording';
    return {
      folderStatus,
      recordButtonLabel,
      discardVisible: isRecording,
      recordStatus: isRecording ? `Samples: ${recorder.sampleCount}` : 'Idle',
      isRecording,
      sampleCount: recorder.sampleCount,
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
    if (recorder.isRecording) return;
    lastSnapshotKey = null;
    recorder.start(
      settingsStore.get(),
      rows,
      cols,
      currentComment,
      currentMode,
    );
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
    reason: string,
  ) => {
    if (!recorder.isRecording) return;
    const key = buildSnapshotKey(board, hold);
    if (key === lastSnapshotKey) {
      console.warn(`[Snapshot] Skipping duplicate (${reason}).`);
      return;
    }
    lastSnapshotKey = key;
    const sample = recorder.record(board, hold, { store: !useRemoteUpload });
    notify();
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
    dispose: () => {
      window.removeEventListener('beforeunload', beforeUnload);
    },
  };
}
