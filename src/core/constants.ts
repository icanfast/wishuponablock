export const COLS = 10;
export const ROWS = 20;
export const NEXT_COUNT = 5;

// Layout
export const OUTER_MARGIN = 40;
export const BOARD_CELL_PX = 28;
export const PANEL_GAP = 16;

export const QUEUE_COLS = 6;
export const QUEUE_PREVIEW_ROWS = 4;
export const QUEUE_GAP_ROWS = 1;

export const HOLD_COLS = 6;
export const HOLD_ROWS = 4;
export const HOLD_LABEL_HEIGHT = BOARD_CELL_PX;
export const QUEUE_LABEL_HEIGHT = HOLD_LABEL_HEIGHT;

export const SETTINGS_PANEL_WIDTH = BOARD_CELL_PX * 5 + 40;

export const BOARD_WIDTH = COLS * BOARD_CELL_PX;
export const BOARD_HEIGHT = ROWS * BOARD_CELL_PX;

export const QUEUE_WIDTH = QUEUE_COLS * BOARD_CELL_PX;
export const QUEUE_PREVIEW_HEIGHT = QUEUE_PREVIEW_ROWS * BOARD_CELL_PX;
export const QUEUE_GAP_PX = QUEUE_GAP_ROWS * BOARD_CELL_PX;
export const QUEUE_PANEL_HEIGHT =
  QUEUE_LABEL_HEIGHT +
  NEXT_COUNT * QUEUE_PREVIEW_HEIGHT +
  (NEXT_COUNT - 1) * QUEUE_GAP_PX;

export const HOLD_WIDTH = HOLD_COLS * BOARD_CELL_PX;
export const HOLD_HEIGHT = HOLD_ROWS * BOARD_CELL_PX;
export const HOLD_PANEL_HEIGHT = HOLD_LABEL_HEIGHT + HOLD_HEIGHT;

export const HOLD_X = OUTER_MARGIN;
export const HOLD_Y = OUTER_MARGIN;
export const HOLD_INNER_Y = HOLD_Y + HOLD_LABEL_HEIGHT;
export const GAME_OVER_Y = HOLD_Y + HOLD_PANEL_HEIGHT + PANEL_GAP;

export const BOARD_X = HOLD_X + HOLD_WIDTH + PANEL_GAP;
export const BOARD_Y = OUTER_MARGIN;

export const QUEUE_X = BOARD_X + BOARD_WIDTH + PANEL_GAP;
export const QUEUE_Y = BOARD_Y;

export const SETTINGS_X = QUEUE_X + QUEUE_WIDTH + PANEL_GAP;
export const SETTINGS_Y = BOARD_Y;

export const PLAY_WIDTH = SETTINGS_X + SETTINGS_PANEL_WIDTH + OUTER_MARGIN;
export const PLAY_HEIGHT =
  OUTER_MARGIN +
  Math.max(BOARD_HEIGHT, QUEUE_PANEL_HEIGHT, HOLD_PANEL_HEIGHT) +
  OUTER_MARGIN;

export const DEFAULT_GRAVITY_MS = 800;
export const DEFAULT_SOFT_DROP_MS = 80;
export const DEFAULT_LOCK_DELAY_MS = 500;
export const DEFAULT_HARD_LOCK_DELAY_MS = 2000;
export const CLASSIC_LOCK_DELAY_MS = 500;
export const CLASSIC_HARD_LOCK_DELAY_MS = 500;
export const DEFAULT_DAS_MS = 200;
export const DEFAULT_ARR_MS = 50;
export const DEFAULT_MASTER_VOLUME = 0.5;
export const DEFAULT_SHARE_SNAPSHOTS = true;
export const DEFAULT_BUTTERFINGER_ENABLED = false;
export const DEFAULT_BUTTERFINGER_MISS_RATE = 0;
export const DEFAULT_BUTTERFINGER_WRONG_DIR_RATE = 0;
export const DEFAULT_BUTTERFINGER_EXTRA_TAP_RATE = 0;
export const DEFAULT_BUTTERFINGER_LOCK_NUDGE_RATE = 0;
export const DEFAULT_BUTTERFINGER_GRAVITY_DROP_RATE = 0;
export const DEFAULT_BUTTERFINGER_LOCK_ROTATE_RATE = 0.5;
export const DEFAULT_GRIDLINE_OPACITY = 0;
export const DEFAULT_GHOST_OPACITY = 0.35;
export const DEFAULT_HIGH_CONTRAST = false;
export const DEFAULT_COLORBLIND_MODE = false;
export const DEFAULT_ML_INFERENCE = {
  strategy: 'threshold',
  temperature: 1.25,
  threshold: 0.1,
  postSharpness: 1.25,
} as const;

export const DEFAULT_KEY_BINDINGS = {
  moveLeft: 'ArrowLeft',
  moveRight: 'ArrowRight',
  softDrop: 'ArrowDown',
  hardDrop: 'Space',
  rotateCW: 'KeyE',
  rotateCCW: 'KeyW',
  rotate180: 'KeyA',
  hold: 'KeyQ',
  restart: 'KeyR',
};

export const SETTINGS_STORAGE_KEY = 'wishuponablock.settings';

export const GAME_PROTOCOL_VERSION = 2;

const BASE_URL = import.meta.env.BASE_URL ?? '/';
export const ML_MODEL_URL = `${BASE_URL}models/model_v3.json`;

export const SPAWN_X = 3;
export const SPAWN_Y = -1;
