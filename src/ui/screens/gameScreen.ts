import {
  BOARD_CELL_PX,
  BOARD_X,
  BOARD_Y,
  COLS,
  GAME_OVER_Y,
  HOLD_WIDTH,
  HOLD_X,
  HOLD_Y,
  OUTER_MARGIN,
  PANEL_GAP,
  PLAY_HEIGHT,
  QUEUE_WIDTH,
  QUEUE_X,
  QUEUE_Y,
  ROWS,
  SETTINGS_PANEL_WIDTH,
  SETTINGS_X,
  SETTINGS_Y,
} from '../../core/constants';
import type { GeneratorType } from '../../core/generators';

export type GameScreen = {
  root: HTMLDivElement;
  generatorSelect: HTMLSelectElement;
  modelStatusLabel: HTMLDivElement;
  sprintPanel: HTMLDivElement;
  sprintTimerValue: HTMLDivElement;
  sprintLinesValue: HTMLDivElement;
  classicPanel: HTMLDivElement;
  classicLevelValue: HTMLDivElement;
  classicScoreValue: HTMLDivElement;
  gameOverLabel: HTMLDivElement;
  recordRow: HTMLDivElement;
  commentInput: HTMLInputElement;
  folderButton: HTMLButtonElement;
  folderStatus: HTMLDivElement;
  recordButton: HTMLButtonElement;
  discardButton: HTMLButtonElement;
  recordStatus: HTMLDivElement;
  manualButton: HTMLButtonElement;
  menuButton: HTMLButtonElement;
};

type GameScreenOptions = {
  generatorTypes: readonly GeneratorType[];
  initialGeneratorType: GeneratorType;
};

const makeOverlayButton = (labelText: string) => {
  const btn = document.createElement('button');
  btn.textContent = labelText;
  Object.assign(btn.style, {
    background: '#0b0f14',
    color: '#e2e8f0',
    border: '1px solid #1f2a37',
    borderRadius: '6px',
    padding: '10px 12px',
    fontSize: '13px',
    cursor: 'pointer',
  });
  return btn;
};

const labelForGenerator = (type: GeneratorType): string => {
  switch (type) {
    case 'bag7':
      return 'Bag 7';
    case 'bag8i':
      return 'I-Plus Bag';
    case 'inconvenient':
      return 'Inconvenient Bag';
    case 'random':
      return 'Random';
    case 'nes':
      return 'NES';
    case 'ml':
      return 'Wish Upon a Block';
    default:
      return type;
  }
};

export function createGameScreen(options: GameScreenOptions): GameScreen {
  const { generatorTypes, initialGeneratorType } = options;

  const root = document.createElement('div');
  Object.assign(root.style, {
    position: 'absolute',
    inset: '0',
    pointerEvents: 'none',
  });

  const createHoldLabel = () => {
    const label = document.createElement('div');
    label.textContent = 'HOLD';
    Object.assign(label.style, {
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
    return label;
  };
  root.appendChild(createHoldLabel());

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
  root.appendChild(nextLabel);

  const gameOverLabel = document.createElement('div');
  gameOverLabel.textContent = 'GAME OVER';
  Object.assign(gameOverLabel.style, {
    position: 'absolute',
    left: `${HOLD_X}px`,
    top: `${GAME_OVER_Y}px`,
    width: `${HOLD_WIDTH}px`,
    color: '#7f8a9a',
    fontFamily: 'system-ui, -apple-system, Segoe UI, sans-serif',
    fontSize: '13px',
    letterSpacing: '0.5px',
    textAlign: 'center',
    pointerEvents: 'none',
    display: 'none',
  });
  root.appendChild(gameOverLabel);

  const settingsPanel = document.createElement('div');
  Object.assign(settingsPanel.style, {
    position: 'absolute',
    left: `${SETTINGS_X}px`,
    top: `${SETTINGS_Y}px`,
    width: `${SETTINGS_PANEL_WIDTH}px`,
    maxHeight: `${PLAY_HEIGHT - SETTINGS_Y - OUTER_MARGIN}px`,
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
  root.appendChild(settingsPanel);

  const label = document.createElement('div');
  label.textContent = 'Generator';
  Object.assign(label.style, {
    marginBottom: '6px',
    color: '#b6c2d4',
  });

  const select = document.createElement('select');
  Object.assign(select.style, {
    width: '100%',
    background: '#0b0f14',
    color: '#e2e8f0',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    padding: '6px 8px',
    fontSize: '13px',
  });

  for (const type of generatorTypes) {
    const option = document.createElement('option');
    option.value = type;
    option.textContent = labelForGenerator(type);
    select.appendChild(option);
  }
  select.value = initialGeneratorType;

  settingsPanel.appendChild(label);
  settingsPanel.appendChild(select);

  const modelStatusLabel = document.createElement('div');
  Object.assign(modelStatusLabel.style, {
    marginTop: '6px',
    color: '#8fa0b8',
    fontSize: '12px',
  });
  settingsPanel.appendChild(modelStatusLabel);

  const sprintPanel = document.createElement('div');
  Object.assign(sprintPanel.style, {
    marginTop: '10px',
    padding: '8px',
    background: '#0b0f14',
    border: '1px solid #1f2a37',
    borderRadius: '6px',
    display: 'none',
    fontSize: '12px',
    color: '#b6c2d4',
  });

  const sprintTitle = document.createElement('div');
  sprintTitle.textContent = 'SPRINT';
  Object.assign(sprintTitle.style, {
    color: '#8fa0b8',
    fontSize: '11px',
    letterSpacing: '0.5px',
    marginBottom: '6px',
  });
  sprintPanel.appendChild(sprintTitle);

  const makeSprintRow = (labelText: string) => {
    const row = document.createElement('div');
    Object.assign(row.style, {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: '4px',
    });
    const label = document.createElement('div');
    label.textContent = labelText;
    Object.assign(label.style, {
      color: '#b6c2d4',
    });
    const value = document.createElement('div');
    Object.assign(value.style, {
      color: '#e2e8f0',
      fontFamily: 'system-ui, -apple-system, Segoe UI, sans-serif',
    });
    row.appendChild(label);
    row.appendChild(value);
    return { row, value };
  };

  const sprintTimerRow = makeSprintRow('Time');
  const sprintLinesRow = makeSprintRow('Lines Left');
  sprintPanel.appendChild(sprintTimerRow.row);
  sprintPanel.appendChild(sprintLinesRow.row);

  settingsPanel.appendChild(sprintPanel);

  const classicPanel = document.createElement('div');
  Object.assign(classicPanel.style, {
    marginTop: '10px',
    padding: '8px',
    background: '#0b0f14',
    border: '1px solid #1f2a37',
    borderRadius: '6px',
    display: 'none',
    fontSize: '12px',
    color: '#b6c2d4',
  });

  const classicTitle = document.createElement('div');
  classicTitle.textContent = 'CLASSIC';
  Object.assign(classicTitle.style, {
    color: '#8fa0b8',
    fontSize: '11px',
    letterSpacing: '0.5px',
    marginBottom: '6px',
  });
  classicPanel.appendChild(classicTitle);

  const classicLevelRow = makeSprintRow('Level');
  const classicScoreRow = makeSprintRow('Score');
  classicPanel.appendChild(classicLevelRow.row);
  classicPanel.appendChild(classicScoreRow.row);

  settingsPanel.appendChild(classicPanel);

  const recordLabel = document.createElement('div');
  recordLabel.textContent = 'Snapshots';
  Object.assign(recordLabel.style, {
    marginTop: '12px',
    marginBottom: '6px',
    color: '#b6c2d4',
  });
  settingsPanel.appendChild(recordLabel);

  const folderButton = document.createElement('button');
  folderButton.textContent = 'Select Folder';
  Object.assign(folderButton.style, {
    background: '#0b0f14',
    color: '#b6c2d4',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    padding: '6px 8px',
    fontSize: '12px',
    cursor: 'pointer',
  });
  settingsPanel.appendChild(folderButton);

  const folderStatus = document.createElement('div');
  Object.assign(folderStatus.style, {
    marginTop: '6px',
    fontSize: '12px',
    color: '#8fa0b8',
  });
  settingsPanel.appendChild(folderStatus);

  const recordRow = document.createElement('div');
  Object.assign(recordRow.style, {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  });

  const commentInput = document.createElement('input');
  commentInput.type = 'text';
  commentInput.placeholder = 'Comment';
  Object.assign(commentInput.style, {
    flex: '1',
    minWidth: '0',
    background: '#0b0f14',
    color: '#e2e8f0',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    padding: '6px 8px',
    fontSize: '12px',
  });

  const recordButton = document.createElement('button');
  Object.assign(recordButton.style, {
    flex: '1',
    background: '#0b0f14',
    color: '#e2e8f0',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    padding: '6px 8px',
    fontSize: '12px',
    cursor: 'pointer',
  });

  const discardButton = document.createElement('button');
  discardButton.textContent = 'Discard';
  Object.assign(discardButton.style, {
    background: '#0b0f14',
    color: '#b6c2d4',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    padding: '6px 8px',
    fontSize: '12px',
    cursor: 'pointer',
    display: 'none',
  });

  recordRow.appendChild(commentInput);
  recordRow.appendChild(recordButton);
  recordRow.appendChild(discardButton);
  settingsPanel.appendChild(recordRow);

  const recordStatus = document.createElement('div');
  Object.assign(recordStatus.style, {
    marginTop: '6px',
    fontSize: '12px',
    color: '#8fa0b8',
  });
  settingsPanel.appendChild(recordStatus);

  const manualButton = document.createElement('button');
  manualButton.textContent = 'Capture Now';
  Object.assign(manualButton.style, {
    marginTop: '6px',
    width: '100%',
    background: '#0b0f14',
    color: '#e2e8f0',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    padding: '6px 8px',
    fontSize: '12px',
    cursor: 'pointer',
    pointerEvents: 'auto',
  });
  settingsPanel.appendChild(manualButton);

  const menuButton = makeOverlayButton('MENU');
  Object.assign(menuButton.style, {
    position: 'absolute',
    left: `${BOARD_X}px`,
    top: `${BOARD_Y + ROWS * BOARD_CELL_PX + PANEL_GAP}px`,
    width: `${COLS * BOARD_CELL_PX}px`,
    pointerEvents: 'auto',
    zIndex: '2',
  });
  root.appendChild(menuButton);

  return {
    root,
    generatorSelect: select,
    modelStatusLabel,
    sprintPanel,
    sprintTimerValue: sprintTimerRow.value,
    sprintLinesValue: sprintLinesRow.value,
    classicPanel,
    classicLevelValue: classicLevelRow.value,
    classicScoreValue: classicScoreRow.value,
    gameOverLabel,
    recordRow,
    commentInput,
    folderButton,
    folderStatus,
    recordButton,
    discardButton,
    recordStatus,
    manualButton,
    menuButton,
  };
}
