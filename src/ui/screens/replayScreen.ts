import {
  BOARD_CELL_PX,
  BOARD_Y,
  COLS,
  PANEL_GAP,
  ROWS,
} from '../../core/constants';
import type { PieceKind } from '../../core/types';

export type ReplayScreen = {
  root: HTMLDivElement;
  backButton: HTMLButtonElement;
  setStatus: (value: string) => void;
  setDetails: (value: string) => void;
  appendLog: (line: string) => void;
  clearLog: () => void;
  setHoldValue: (value: PieceKind | null) => void;
};

const makeButton = (labelText: string): HTMLButtonElement => {
  const button = document.createElement('button');
  button.textContent = labelText;
  Object.assign(button.style, {
    background: '#0b0f14',
    color: '#e2e8f0',
    border: '1px solid #1f2a37',
    borderRadius: '6px',
    padding: '10px 12px',
    fontSize: '13px',
    cursor: 'pointer',
  });
  return button;
};

const REPLAY_VIEW_Y_OFFSET = 20;

export const createReplayScreen = (): ReplayScreen => {
  const root = document.createElement('div');
  Object.assign(root.style, {
    position: 'absolute',
    inset: '0',
    pointerEvents: 'none',
  });

  const backButton = makeButton('BACK TO MENU');
  Object.assign(backButton.style, {
    position: 'absolute',
    left: '50%',
    transform: 'translateX(-50%)',
    top: `${BOARD_Y + ROWS * BOARD_CELL_PX + PANEL_GAP + REPLAY_VIEW_Y_OFFSET}px`,
    width: `${COLS * BOARD_CELL_PX}px`,
    pointerEvents: 'auto',
    zIndex: '3',
  });
  root.appendChild(backButton);

  const holdValue = document.createElement('div');
  holdValue.textContent = 'HOLD: -';
  Object.assign(holdValue.style, {
    position: 'absolute',
    left: '16px',
    top: '16px',
    padding: '8px 10px',
    background: 'rgba(11, 15, 20, 0.92)',
    color: '#e2e8f0',
    border: '1px solid #1f2a37',
    borderRadius: '6px',
    fontFamily: 'system-ui, -apple-system, Segoe UI, sans-serif',
    fontSize: '13px',
    letterSpacing: '0.2px',
    zIndex: '3',
  });
  root.appendChild(holdValue);

  const setHoldValue = (value: PieceKind | null): void => {
    holdValue.textContent = `HOLD: ${value ?? '-'}`;
  };

  return {
    root,
    backButton,
    setStatus: () => {},
    setDetails: () => {},
    appendLog: () => {},
    clearLog: () => {},
    setHoldValue,
  };
};
