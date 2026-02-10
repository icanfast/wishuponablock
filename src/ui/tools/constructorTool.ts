import { makeBoard } from '../../core/board';
import {
  BOARD_CELL_PX,
  BOARD_HEIGHT,
  BOARD_WIDTH,
  BOARD_X,
  BOARD_Y,
  COLS,
  ROWS,
} from '../../core/constants';
import { cellsOf } from '../../core/piece';
import { SnapshotRecorder } from '../../core/snapshotRecorder';
import type { SettingsStore } from '../../core/settingsStore';
import {
  PIECES,
  type ActivePiece,
  type Board,
  type PieceKind,
  type Rotation,
} from '../../core/types';
import type { UploadClient } from '../../core/uploadClient';
import type { IdentityService } from '../../app/identityService';
import { TETROMINOES } from '../../core/tetromino';
import { createToolScreen } from '../screens/toolScreen';
import type { ToolController } from './toolHost';
import type { ToolCanvas } from './toolCanvas';

type ConstructorToolOptions = {
  canvas: ToolCanvas;
  canvasElement: HTMLCanvasElement;
  uploadClient: UploadClient;
  settingsStore: SettingsStore;
  identityService: IdentityService;
  buildVersion?: string;
  onBack: () => void;
};

export function createConstructorTool(
  options: ConstructorToolOptions,
): ToolController {
  const {
    canvas,
    canvasElement,
    uploadClient,
    settingsStore,
    identityService,
    buildVersion,
    onBack,
  } = options;

  const toolUi = createToolScreen({ toolUsesRemote: uploadClient.isRemote });
  toolUi.title.textContent = 'GRID CONSTRUCTOR';
  toolUi.info.textContent =
    'How to use:\nSelect a piece, scroll to rotate, click to place.\n\n' +
    "Click 'Submit' to upload the snapshot.";

  toolUi.inputButton.style.display = 'none';
  toolUi.inputStatus.style.display = 'none';
  toolUi.playstyleLabel.style.display = 'none';
  toolUi.playstyleSelect.style.display = 'none';
  toolUi.modeLabel.style.display = 'none';
  toolUi.modeSelect.style.display = 'none';
  toolUi.triggerLabel.style.display = 'none';
  toolUi.triggerSelect.style.display = 'none';
  toolUi.buildLabel.style.display = 'none';
  toolUi.buildSelect.style.display = 'none';
  toolUi.outputButton.style.display = 'none';
  toolUi.outputStatus.style.display = 'none';
  toolUi.sampleStatus.style.display = 'none';

  toolUi.nextButton.textContent = 'SUBMIT';

  let board: Board = makeBoard();
  let selectedPiece: PieceKind | null = null;
  let rotation: Rotation = 0;
  let ghost: ActivePiece | null = null;
  let lastMouseCell: { x: number; y: number } | null = null;
  let toolActive = false;
  const recorder = new SnapshotRecorder();

  const updatePieceButtons = () => {
    for (const [piece, btn] of toolUi.pieceButtons) {
      const selected = selectedPiece === piece;
      btn.style.borderColor = selected ? '#ffffff' : '#1f2a37';
      btn.style.boxShadow = selected
        ? '0 0 0 1px rgba(255,255,255,0.5)'
        : 'none';
    }
  };

  const renderBoard = () => {
    if (!toolActive) return;
    canvas.render(board, null, ghost);
  };

  const clamp = (value: number, min: number, max: number): number =>
    Math.min(max, Math.max(min, value));

  const getShapeBounds = (
    shape: ReadonlyArray<readonly [number, number]>,
  ): { minX: number; maxX: number; minY: number; maxY: number } => {
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const [x, y] of shape) {
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }
    return { minX, maxX, minY, maxY };
  };

  const buildGhostAt = (cellX: number, cellY: number): ActivePiece | null => {
    if (!selectedPiece) return null;
    const shape = TETROMINOES[selectedPiece][rotation];
    const bounds = getShapeBounds(shape);
    const centerX = (bounds.minX + bounds.maxX) / 2;
    const centerY = (bounds.minY + bounds.maxY) / 2;
    let x = Math.round(cellX - centerX);
    let y = Math.round(cellY - centerY);
    const minX = -bounds.minX;
    const maxX = COLS - 1 - bounds.maxX;
    const minY = -bounds.minY;
    const maxY = ROWS - 1 - bounds.maxY;
    x = clamp(x, minX, maxX);
    y = clamp(y, minY, maxY);
    return { k: selectedPiece, r: rotation, x, y };
  };

  const startSession = () => {
    const settings = settingsStore.get();
    const session = recorder.start(settings, ROWS, COLS, undefined, {
      id: 'constructor',
      options: {},
    });
    if (buildVersion) {
      session.meta.buildVersion = buildVersion;
    }
    session.meta.device_id = identityService.getDeviceId();
    const userId = identityService.getUserId();
    if (userId) {
      session.meta.user_id = userId;
    } else {
      delete session.meta.user_id;
    }
  };

  const ensureSession = () => {
    if (!recorder.isRecording) {
      startSession();
    }
  };

  const updateGhostFromEvent = (event: MouseEvent) => {
    if (!selectedPiece) {
      ghost = null;
      lastMouseCell = null;
      renderBoard();
      return;
    }
    const rect = canvasElement.getBoundingClientRect();
    const localX = event.clientX - rect.left;
    const localY = event.clientY - rect.top;
    const insideBoard =
      localX >= BOARD_X &&
      localX <= BOARD_X + BOARD_WIDTH &&
      localY >= BOARD_Y &&
      localY <= BOARD_Y + BOARD_HEIGHT;
    if (!insideBoard) {
      ghost = null;
      lastMouseCell = null;
      renderBoard();
      return;
    }
    const gridX = Math.floor((localX - BOARD_X) / BOARD_CELL_PX);
    const gridY = Math.floor((localY - BOARD_Y) / BOARD_CELL_PX);
    const clampedX = clamp(gridX, 0, COLS - 1);
    const clampedY = clamp(gridY, 0, ROWS - 1);
    lastMouseCell = { x: clampedX, y: clampedY };
    ghost = buildGhostAt(clampedX, clampedY);
    renderBoard();
  };

  const clearGhost = () => {
    ghost = null;
    lastMouseCell = null;
    renderBoard();
  };

  const placePiece = () => {
    if (!selectedPiece || !ghost) return;
    const piece: ActivePiece = {
      k: selectedPiece,
      r: rotation,
      x: ghost.x,
      y: ghost.y,
    };
    for (const [x, y] of cellsOf(piece)) {
      if (x < 0 || x >= COLS || y < 0 || y >= ROWS) continue;
      board[y][x] = piece.k;
    }
    ghost = null;
    lastMouseCell = null;
    renderBoard();
  };

  for (const [piece, btn] of toolUi.pieceButtons) {
    btn.addEventListener('click', () => {
      selectedPiece = piece;
      if (lastMouseCell) {
        ghost = buildGhostAt(lastMouseCell.x, lastMouseCell.y);
      }
      updatePieceButtons();
      renderBoard();
    });
  }

  const buttonGap = 10;
  const buttonTop =
    toolUi.nextButton.style.top || `${BOARD_Y + BOARD_HEIGHT + 12}px`;
  const buttonWidth = Math.floor((BOARD_WIDTH - buttonGap) / 2);

  Object.assign(toolUi.nextButton.style, {
    left: `${BOARD_X}px`,
    width: `${buttonWidth}px`,
  });

  const clearButton = document.createElement('button');
  clearButton.textContent = 'CLEAR';
  Object.assign(clearButton.style, {
    position: 'absolute',
    left: `${BOARD_X + buttonWidth + buttonGap}px`,
    top: buttonTop,
    width: `${buttonWidth}px`,
    background: '#0b0f14',
    color: '#b6c2d4',
    border: '1px solid #1f2a37',
    borderRadius: '6px',
    padding: '10px 12px',
    fontSize: '13px',
    cursor: 'pointer',
    pointerEvents: 'auto',
  });
  toolUi.root.appendChild(clearButton);

  clearButton.addEventListener('click', () => {
    board = makeBoard();
    renderBoard();
  });

  toolUi.nextButton.addEventListener('click', async () => {
    ensureSession();
    const sample = recorder.record(board, null, {
      store: !uploadClient.isRemote,
      trigger: 'constructor',
    });
    if (!sample) {
      toolUi.actionStatus.textContent = 'Unable to save snapshot.';
      return;
    }
    const session = recorder.sessionMeta;
    if (!session) {
      toolUi.actionStatus.textContent = 'Unable to save snapshot.';
      return;
    }
    const payload = {
      createdAt: new Date().toISOString(),
      meta: {
        session,
        sample: {
          index: sample.index,
          timeMs: sample.timeMs,
          hold: sample.hold,
        },
        trigger: 'constructor',
      },
      board: sample.board,
    };
    await uploadClient.enqueueSnapshot(payload);
    toolUi.actionStatus.textContent = 'Board submitted.';
  });

  toolUi.backButton.addEventListener('click', () => onBack());

  const onWheel = (event: WheelEvent) => {
    if (!toolActive || !selectedPiece) return;
    event.preventDefault();
    const dir = event.deltaY > 0 ? 1 : -1;
    applyRotation(dir === 1 ? 1 : -1);
  };

  const applyRotation = (delta: -1 | 1 | 2) => {
    if (!selectedPiece) return;
    const next =
      delta === 2 ? (rotation + 2) % 4 : (((rotation + delta) % 4) + 4) % 4;
    rotation = next as Rotation;
    if (lastMouseCell) {
      ghost = buildGhostAt(lastMouseCell.x, lastMouseCell.y);
    }
    renderBoard();
  };

  const onKeyDown = (event: KeyboardEvent) => {
    if (!toolActive || !selectedPiece) return;
    const bindings = settingsStore.get().input.bindings;
    if (event.code === bindings.rotateCW || event.code === 'ArrowUp') {
      event.preventDefault();
      applyRotation(1);
      return;
    }
    if (event.code === bindings.rotateCCW) {
      event.preventDefault();
      applyRotation(-1);
      return;
    }
    if (event.code === bindings.rotate180) {
      event.preventDefault();
      applyRotation(2);
    }
  };

  const onMouseMove = (event: MouseEvent) => {
    if (!toolActive) return;
    updateGhostFromEvent(event);
  };

  const onMouseLeave = () => {
    if (!toolActive) return;
    clearGhost();
  };

  const onClick = (event: MouseEvent) => {
    if (!toolActive) return;
    updateGhostFromEvent(event);
    placePiece();
  };

  return {
    id: 'constructor',
    label: 'Grid Constructor',
    root: toolUi.root,
    setPiecePalette: toolUi.setPiecePalette,
    enter: () => {
      toolActive = true;
      board = makeBoard();
      selectedPiece = PIECES[0] ?? null;
      rotation = 0;
      ghost = null;
      updatePieceButtons();
      startSession();
      renderBoard();
      clearButton.style.display = 'block';
      canvasElement.addEventListener('wheel', onWheel, { passive: false });
      window.addEventListener('keydown', onKeyDown);
      canvasElement.addEventListener('mousemove', onMouseMove);
      canvasElement.addEventListener('mouseleave', onMouseLeave);
      canvasElement.addEventListener('click', onClick);
    },
    leave: () => {
      toolActive = false;
      clearButton.style.display = 'none';
      canvasElement.removeEventListener('wheel', onWheel);
      window.removeEventListener('keydown', onKeyDown);
      canvasElement.removeEventListener('mousemove', onMouseMove);
      canvasElement.removeEventListener('mouseleave', onMouseLeave);
      canvasElement.removeEventListener('click', onClick);
    },
  };
}
