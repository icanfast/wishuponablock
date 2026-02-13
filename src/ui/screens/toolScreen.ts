import {
  BOARD_CELL_PX,
  BOARD_WIDTH,
  BOARD_X,
  BOARD_Y,
  HOLD_WIDTH,
  HOLD_X,
  HOLD_Y,
  NEXT_COUNT,
  OUTER_MARGIN,
  PANEL_GAP,
  QUEUE_GAP_PX,
  QUEUE_LABEL_HEIGHT,
  QUEUE_PREVIEW_HEIGHT,
  PLAY_HEIGHT,
  QUEUE_WIDTH,
  QUEUE_X,
  QUEUE_Y,
  ROWS,
  SETTINGS_PANEL_WIDTH,
  SETTINGS_X,
  SETTINGS_Y,
} from '../../core/constants';
import {
  PIECES,
  type PieceKind,
  type PieceProbability,
} from '../../core/types';
import { TETROMINOES } from '../../core/tetromino';
import { PIECE_COLORS, type PiecePalette } from '../../core/palette';

export type ToolScreen = {
  root: HTMLDivElement;
  panel: HTMLDivElement;
  title: HTMLDivElement;
  info: HTMLDivElement;
  inputButton: HTMLButtonElement;
  inputStatus: HTMLDivElement;
  playstyleLabel: HTMLDivElement;
  playstyleSelect: HTMLSelectElement;
  modeLabel: HTMLDivElement;
  modeSelect: HTMLSelectElement;
  triggerLabel: HTMLDivElement;
  triggerSelect: HTMLSelectElement;
  buildLabel: HTMLDivElement;
  buildSelect: HTMLSelectElement;
  outputButton: HTMLButtonElement;
  outputStatus: HTMLDivElement;
  sampleStatus: HTMLDivElement;
  actionStatus: HTMLDivElement;
  backButton: HTMLButtonElement;
  nextButton: HTMLButtonElement;
  pieceButtons: Map<PieceKind, HTMLButtonElement>;
  setPiecePalette: (palette: PiecePalette) => void;
  setQueueOddsMode: (enabled: boolean) => void;
  setQueueProbabilities: (values: PieceProbability[]) => void;
};

type ToolScreenOptions = {
  toolUsesRemote: boolean;
};

const toHex = (color: number): string =>
  `#${color.toString(16).padStart(6, '0')}`;

export function createToolScreen(options: ToolScreenOptions): ToolScreen {
  const { toolUsesRemote } = options;

  let piecePalette: PiecePalette = PIECE_COLORS;

  const toolLayer = document.createElement('div');
  Object.assign(toolLayer.style, {
    position: 'absolute',
    inset: '0',
    pointerEvents: 'none',
    display: 'block',
  });

  const toolColumn = document.createElement('div');
  Object.assign(toolColumn.style, {
    position: 'absolute',
    left: `${SETTINGS_X}px`,
    top: `${SETTINGS_Y}px`,
    width: `${SETTINGS_PANEL_WIDTH}px`,
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  });
  toolLayer.appendChild(toolColumn);

  const holdLabel = document.createElement('div');
  holdLabel.textContent = 'HOLD';
  Object.assign(holdLabel.style, {
    position: 'absolute',
    left: `${HOLD_X}px`,
    top: `${HOLD_Y + 6}px`,
    width: `${HOLD_WIDTH}px`,
    color: '#b6c2d4',
    fontFamily: 'system-ui, -apple-system, Segoe UI, sans-serif',
    fontSize: '13px',
    letterSpacing: '0.5px',
    textAlign: 'center',
    pointerEvents: 'none',
  });
  toolLayer.appendChild(holdLabel);

  const nextLabel = document.createElement('div');
  nextLabel.textContent = 'NEXT';
  Object.assign(nextLabel.style, {
    position: 'absolute',
    left: `${QUEUE_X}px`,
    top: `${QUEUE_Y + 6}px`,
    width: `${QUEUE_WIDTH}px`,
    color: '#b6c2d4',
    fontFamily: 'system-ui, -apple-system, Segoe UI, sans-serif',
    fontSize: '13px',
    letterSpacing: '0.5px',
    textAlign: 'center',
    pointerEvents: 'none',
  });
  toolLayer.appendChild(nextLabel);

  const queueProbabilityLabels = Array.from({ length: NEXT_COUNT }, (_, i) => {
    const label = document.createElement('div');
    Object.assign(label.style, {
      position: 'absolute',
      left: `${QUEUE_X}px`,
      top: `${QUEUE_Y + QUEUE_LABEL_HEIGHT + i * (QUEUE_PREVIEW_HEIGHT + QUEUE_GAP_PX) + QUEUE_PREVIEW_HEIGHT - 16}px`,
      width: `${QUEUE_WIDTH}px`,
      color: '#8fa0b8',
      fontFamily: 'system-ui, -apple-system, Segoe UI, sans-serif',
      fontSize: '11px',
      letterSpacing: '0.2px',
      textAlign: 'center',
      pointerEvents: 'none',
      display: 'none',
    });
    toolLayer.appendChild(label);
    return label;
  });

  const toolPanelMaxHeight = Math.max(
    0,
    PLAY_HEIGHT - SETTINGS_Y - OUTER_MARGIN - 48,
  );
  const toolPanel = document.createElement('div');
  Object.assign(toolPanel.style, {
    width: '100%',
    maxHeight: `${toolPanelMaxHeight}px`,
    overflowY: 'auto',
    padding: '8px',
    background: '#121a24',
    color: '#e2e8f0',
    border: '2px solid #0b0f14',
    borderRadius: '6px',
    fontFamily: 'system-ui, -apple-system, Segoe UI, sans-serif',
    fontSize: '13px',
    pointerEvents: 'auto',
  });
  toolColumn.appendChild(toolPanel);

  const toolInfo = document.createElement('div');
  toolInfo.textContent =
    "How to use:\nSelect all the pieces you would want to fall in this situation, then click 'Next'\n\n" +
    'Your answers will be saved and used to train ML piece generator models.';
  Object.assign(toolInfo.style, {
    color: '#b6c2d4',
    fontSize: '13px',
    lineHeight: '1.4',
    textAlign: 'center',
    whiteSpace: 'pre-line',
    pointerEvents: 'none',
  });
  toolColumn.appendChild(toolInfo);

  const toolTitle = document.createElement('div');
  toolTitle.textContent = 'WISH UPON A BLOCK';
  Object.assign(toolTitle.style, {
    marginBottom: '8px',
    color: '#b6c2d4',
    letterSpacing: '0.5px',
  });
  toolPanel.appendChild(toolTitle);

  const toolInputButton = document.createElement('button');
  toolInputButton.textContent = 'Select Snapshot Folder';
  Object.assign(toolInputButton.style, {
    width: '100%',
    background: '#0b0f14',
    color: '#e2e8f0',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    padding: '8px',
    fontSize: '12px',
    cursor: 'pointer',
  });
  toolPanel.appendChild(toolInputButton);

  const toolInputStatus = document.createElement('div');
  Object.assign(toolInputStatus.style, {
    marginTop: '6px',
    fontSize: '12px',
    color: '#8fa0b8',
    whiteSpace: 'pre-line',
  });
  toolInputStatus.textContent = 'Source: Local (select folder).';
  toolPanel.appendChild(toolInputStatus);

  const toolPlaystyleLabel = document.createElement('div');
  toolPlaystyleLabel.textContent = 'Playstyle';
  Object.assign(toolPlaystyleLabel.style, {
    marginTop: '10px',
    fontSize: '12px',
    color: '#b6c2d4',
  });
  toolPanel.appendChild(toolPlaystyleLabel);

  const toolPlaystyleSelect = document.createElement('select');
  Object.assign(toolPlaystyleSelect.style, {
    width: '100%',
    marginTop: '6px',
    background: '#0b0f14',
    color: '#e2e8f0',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    padding: '6px 8px',
    fontSize: '12px',
  });
  toolPanel.appendChild(toolPlaystyleSelect);

  const toolModeLabel = document.createElement('div');
  toolModeLabel.textContent = 'Game Mode';
  Object.assign(toolModeLabel.style, {
    marginTop: '10px',
    fontSize: '12px',
    color: '#b6c2d4',
  });
  toolPanel.appendChild(toolModeLabel);

  const toolModeSelect = document.createElement('select');
  Object.assign(toolModeSelect.style, {
    width: '100%',
    marginTop: '6px',
    background: '#0b0f14',
    color: '#e2e8f0',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    padding: '6px 8px',
    fontSize: '12px',
  });
  toolPanel.appendChild(toolModeSelect);

  const toolTriggerLabel = document.createElement('div');
  toolTriggerLabel.textContent = 'Capture Trigger';
  Object.assign(toolTriggerLabel.style, {
    marginTop: '10px',
    fontSize: '12px',
    color: '#b6c2d4',
  });
  toolPanel.appendChild(toolTriggerLabel);

  const toolTriggerSelect = document.createElement('select');
  Object.assign(toolTriggerSelect.style, {
    width: '100%',
    marginTop: '6px',
    background: '#0b0f14',
    color: '#e2e8f0',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    padding: '6px 8px',
    fontSize: '12px',
  });
  toolPanel.appendChild(toolTriggerSelect);

  const toolBuildLabel = document.createElement('div');
  toolBuildLabel.textContent = 'Build Version';
  Object.assign(toolBuildLabel.style, {
    marginTop: '10px',
    fontSize: '12px',
    color: '#b6c2d4',
  });
  toolPanel.appendChild(toolBuildLabel);

  const toolBuildSelect = document.createElement('select');
  Object.assign(toolBuildSelect.style, {
    width: '100%',
    marginTop: '6px',
    background: '#0b0f14',
    color: '#e2e8f0',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    padding: '6px 8px',
    fontSize: '12px',
  });
  toolPanel.appendChild(toolBuildSelect);

  const toolOutputButton = document.createElement('button');
  toolOutputButton.textContent = 'Select Output Folder';
  Object.assign(toolOutputButton.style, {
    width: '100%',
    marginTop: '10px',
    background: '#0b0f14',
    color: '#e2e8f0',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    padding: '8px',
    fontSize: '12px',
    cursor: 'pointer',
  });
  toolPanel.appendChild(toolOutputButton);

  const toolOutputStatus = document.createElement('div');
  Object.assign(toolOutputStatus.style, {
    marginTop: '6px',
    fontSize: '12px',
    color: '#8fa0b8',
  });
  toolOutputStatus.textContent = 'No output folder.';
  toolPanel.appendChild(toolOutputStatus);

  if (toolUsesRemote) {
    toolInputButton.style.display = 'none';
    toolOutputButton.style.display = 'none';
    toolOutputStatus.style.display = 'none';
    toolInputStatus.textContent = 'Source: Online';
  }

  const toolSampleStatus = document.createElement('div');
  Object.assign(toolSampleStatus.style, {
    marginTop: '10px',
    fontSize: '12px',
    color: '#8fa0b8',
  });
  toolSampleStatus.textContent = 'Sample: -';
  toolPanel.appendChild(toolSampleStatus);

  const toolActionStatus = document.createElement('div');
  Object.assign(toolActionStatus.style, {
    marginTop: '6px',
    fontSize: '12px',
    color: '#8fa0b8',
  });
  toolActionStatus.textContent = '';
  toolPanel.appendChild(toolActionStatus);

  const toolBackButton = document.createElement('button');
  toolBackButton.textContent = 'BACK TO MENU';
  Object.assign(toolBackButton.style, {
    width: '100%',
    marginTop: '12px',
    background: '#0b0f14',
    color: '#b6c2d4',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    padding: '8px',
    fontSize: '12px',
    cursor: 'pointer',
  });
  toolPanel.appendChild(toolBackButton);

  const toolPieceCellPx = Math.max(10, Math.round(BOARD_CELL_PX / 2));
  const toolPiecePreviewPx = toolPieceCellPx * 4;
  const toolPieceButtonHeight = toolPiecePreviewPx + 10;
  const toolPieceButtonWidth = toolPiecePreviewPx + 8;
  const toolPieceGap = 6;
  const toolPieceRowWidth =
    PIECES.length * toolPieceButtonWidth + (PIECES.length - 1) * toolPieceGap;
  const toolPieceRowLeft = BOARD_X + (BOARD_WIDTH - toolPieceRowWidth) / 2;
  const toolPieceRowTop = BOARD_Y + ROWS * BOARD_CELL_PX + PANEL_GAP;

  const toolPieceRow = document.createElement('div');
  Object.assign(toolPieceRow.style, {
    position: 'absolute',
    left: `${toolPieceRowLeft}px`,
    top: `${toolPieceRowTop}px`,
    width: `${toolPieceRowWidth}px`,
    display: 'flex',
    gap: `${toolPieceGap}px`,
    alignItems: 'center',
    pointerEvents: 'auto',
  });
  toolLayer.appendChild(toolPieceRow);

  const toolNextButton = document.createElement('button');
  toolNextButton.textContent = 'NEXT';
  Object.assign(toolNextButton.style, {
    position: 'absolute',
    left: `${BOARD_X}px`,
    top: `${toolPieceRowTop + toolPieceButtonHeight + 8}px`,
    width: `${BOARD_WIDTH}px`,
    background: '#0b0f14',
    color: '#e2e8f0',
    border: '1px solid #1f2a37',
    borderRadius: '6px',
    padding: '10px 12px',
    fontSize: '13px',
    cursor: 'pointer',
    pointerEvents: 'auto',
  });
  toolLayer.appendChild(toolNextButton);

  const toolPieceButtons = new Map<PieceKind, HTMLButtonElement>();
  const toolPieceCanvases = new Map<PieceKind, HTMLCanvasElement>();

  const drawPiecePreview = (
    canvas: HTMLCanvasElement,
    piece: PieceKind,
  ): void => {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const color = toHex(piecePalette[piece]);
    const pad = Math.max(1, Math.floor(toolPieceCellPx / 6));
    const shape = TETROMINOES[piece][0];
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
    const width = maxX - minX + 1;
    const height = maxY - minY + 1;
    const dx = (4 - width) / 2 - minX;
    const dy = (4 - height) / 2 - minY;

    for (const [x, y] of shape) {
      const px = (x + dx) * toolPieceCellPx;
      const py = (y + dy) * toolPieceCellPx;
      ctx.fillStyle = color;
      ctx.fillRect(
        px + pad,
        py + pad,
        toolPieceCellPx - pad * 2,
        toolPieceCellPx - pad * 2,
      );
    }
  };

  for (const piece of PIECES) {
    const btn = document.createElement('button');
    Object.assign(btn.style, {
      flex: '1',
      minWidth: `${toolPieceButtonWidth}px`,
      height: `${toolPieceButtonHeight}px`,
      background: '#0b0f14',
      color: '#0b0f14',
      border: '2px solid #1f2a37',
      borderRadius: '6px',
      padding: '4px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      cursor: 'pointer',
    });
    const canvas = document.createElement('canvas');
    canvas.width = toolPiecePreviewPx;
    canvas.height = toolPiecePreviewPx;
    canvas.style.display = 'block';
    drawPiecePreview(canvas, piece);
    btn.appendChild(canvas);
    toolPieceRow.appendChild(btn);
    toolPieceButtons.set(piece, btn);
    toolPieceCanvases.set(piece, canvas);
  }

  const setPiecePalette = (palette: PiecePalette) => {
    piecePalette = palette;
    for (const [piece, canvas] of toolPieceCanvases) {
      drawPiecePreview(canvas, piece);
    }
  };

  const formatProbabilityPercent = (probability: number): string => {
    const value = Number.isFinite(probability) ? Math.max(0, probability) : 0;
    const percent = value * 100;
    const withOneDecimal = percent.toFixed(1);
    const compact = withOneDecimal.replace(/\.0$/, '');
    return `${compact}%`;
  };

  const setQueueProbabilities = (values: PieceProbability[]) => {
    if (!values.length) {
      for (const label of queueProbabilityLabels) {
        label.style.display = 'none';
        label.textContent = '';
      }
      return;
    }
    const sorted = values
      .slice()
      .sort((a, b) => b.probability - a.probability)
      .slice(0, NEXT_COUNT);
    const queueFixedSlotCount = 4;
    const fixedContentHeight =
      queueFixedSlotCount * QUEUE_PREVIEW_HEIGHT +
      (queueFixedSlotCount - 1) * QUEUE_GAP_PX;
    const defaultStep = QUEUE_PREVIEW_HEIGHT + QUEUE_GAP_PX;
    const compressedStep =
      (fixedContentHeight - QUEUE_PREVIEW_HEIGHT) / Math.max(1, NEXT_COUNT - 1);
    const itemStep = Math.min(defaultStep, compressedStep);
    for (let i = 0; i < queueProbabilityLabels.length; i += 1) {
      const label = queueProbabilityLabels[i];
      const entry = sorted[i];
      if (!entry) {
        label.style.display = 'none';
        label.textContent = '';
        continue;
      }
      label.style.top = `${QUEUE_Y + QUEUE_LABEL_HEIGHT + i * itemStep + QUEUE_PREVIEW_HEIGHT - 16}px`;
      label.textContent = formatProbabilityPercent(entry.probability);
      label.style.display = 'block';
    }
  };

  const setQueueOddsMode = (enabled: boolean) => {
    nextLabel.textContent = enabled ? 'PIECE ODDS' : 'NEXT';
  };

  return {
    root: toolLayer,
    panel: toolPanel,
    title: toolTitle,
    info: toolInfo,
    inputButton: toolInputButton,
    inputStatus: toolInputStatus,
    playstyleLabel: toolPlaystyleLabel,
    playstyleSelect: toolPlaystyleSelect,
    modeLabel: toolModeLabel,
    modeSelect: toolModeSelect,
    triggerLabel: toolTriggerLabel,
    triggerSelect: toolTriggerSelect,
    buildLabel: toolBuildLabel,
    buildSelect: toolBuildSelect,
    outputButton: toolOutputButton,
    outputStatus: toolOutputStatus,
    sampleStatus: toolSampleStatus,
    actionStatus: toolActionStatus,
    backButton: toolBackButton,
    nextButton: toolNextButton,
    pieceButtons: toolPieceButtons,
    setPiecePalette,
    setQueueOddsMode,
    setQueueProbabilities,
  };
}
