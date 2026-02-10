import {
  DEFAULT_ARR_MS,
  DEFAULT_BUTTERFINGER_ENABLED,
  DEFAULT_BUTTERFINGER_EXTRA_TAP_RATE,
  DEFAULT_BUTTERFINGER_LOCK_ROTATE_RATE,
  DEFAULT_BUTTERFINGER_GRAVITY_DROP_RATE,
  DEFAULT_BUTTERFINGER_LOCK_NUDGE_RATE,
  DEFAULT_BUTTERFINGER_MISS_RATE,
  DEFAULT_BUTTERFINGER_WRONG_DIR_RATE,
  DEFAULT_DAS_MS,
  DEFAULT_GRAVITY_MS,
  DEFAULT_GRIDLINE_OPACITY,
  DEFAULT_HIGH_CONTRAST,
  DEFAULT_ML_INFERENCE,
  DEFAULT_COLORBLIND_MODE,
  DEFAULT_HARD_LOCK_DELAY_MS,
  DEFAULT_KEY_BINDINGS,
  DEFAULT_LOCK_DELAY_MS,
  DEFAULT_MASTER_VOLUME,
  DEFAULT_SHARE_SNAPSHOTS,
  DEFAULT_SOFT_DROP_MS,
  SETTINGS_STORAGE_KEY,
} from './constants';
import {
  type GeneratorSettings,
  isGeneratorType,
  isMlInferenceStrategy,
} from './generators';
import type { GameConfig } from './game';
import type { InputConfig, KeyBindings } from '../input/controller';

export type GameSettings = Required<
  Pick<
    GameConfig,
    'gravityMs' | 'softDropMs' | 'lockDelayMs' | 'hardLockDelayMs'
  >
>;

export interface Settings {
  game: GameSettings;
  input: InputConfig;
  generator: GeneratorSettings;
  audio: AudioSettings;
  privacy: PrivacySettings;
  graphics: GraphicsSettings;
  butterfinger: ButterfingerSettings;
}

export interface AudioSettings {
  masterVolume: number;
}

export interface PrivacySettings {
  shareSnapshots: boolean;
}

export interface GraphicsSettings {
  gridlineOpacity: number;
  highContrast: boolean;
  colorblindMode: boolean;
}

export interface ButterfingerSettings {
  enabled: boolean;
  missRate: number;
  wrongDirRate: number;
  extraTapRate: number;
  lockNudgeRate: number;
  gravityDropRate: number;
  lockRotateRate: number;
}

export const DEFAULT_SETTINGS: Settings = {
  game: {
    gravityMs: DEFAULT_GRAVITY_MS,
    softDropMs: DEFAULT_SOFT_DROP_MS,
    lockDelayMs: DEFAULT_LOCK_DELAY_MS,
    hardLockDelayMs: DEFAULT_HARD_LOCK_DELAY_MS,
  },
  input: {
    dasMs: DEFAULT_DAS_MS,
    arrMs: DEFAULT_ARR_MS,
    bindings: { ...DEFAULT_KEY_BINDINGS },
  },
  generator: {
    type: 'ml',
    ml: { ...DEFAULT_ML_INFERENCE },
  },
  audio: {
    masterVolume: DEFAULT_MASTER_VOLUME,
  },
  privacy: {
    shareSnapshots: DEFAULT_SHARE_SNAPSHOTS,
  },
  graphics: {
    gridlineOpacity: DEFAULT_GRIDLINE_OPACITY,
    highContrast: DEFAULT_HIGH_CONTRAST,
    colorblindMode: DEFAULT_COLORBLIND_MODE,
  },
  butterfinger: {
    enabled: DEFAULT_BUTTERFINGER_ENABLED,
    missRate: DEFAULT_BUTTERFINGER_MISS_RATE,
    wrongDirRate: DEFAULT_BUTTERFINGER_WRONG_DIR_RATE,
    extraTapRate: DEFAULT_BUTTERFINGER_EXTRA_TAP_RATE,
    lockNudgeRate: DEFAULT_BUTTERFINGER_LOCK_NUDGE_RATE,
    gravityDropRate: DEFAULT_BUTTERFINGER_GRAVITY_DROP_RATE,
    lockRotateRate: DEFAULT_BUTTERFINGER_LOCK_ROTATE_RATE,
  },
};

function num(v: unknown): number | undefined {
  return typeof v === 'number' && Number.isFinite(v) ? v : undefined;
}

function bool(v: unknown): boolean | undefined {
  return typeof v === 'boolean' ? v : undefined;
}

function mergeBindings(
  base: KeyBindings,
  patch?: Partial<KeyBindings>,
): KeyBindings {
  const bind = (value: unknown, fallback: string): string => {
    if (typeof value !== 'string') return fallback;
    return value.trim();
  };
  return {
    moveLeft: bind(patch?.moveLeft, base.moveLeft),
    moveRight: bind(patch?.moveRight, base.moveRight),
    softDrop: bind(patch?.softDrop, base.softDrop),
    hardDrop: bind(patch?.hardDrop, base.hardDrop),
    rotateCW: bind(patch?.rotateCW, base.rotateCW),
    rotateCCW: bind(patch?.rotateCCW, base.rotateCCW),
    rotate180: bind(patch?.rotate180, base.rotate180),
    hold: bind(patch?.hold, base.hold),
    restart: bind(patch?.restart, base.restart),
  };
}

function mergeGame(
  base: GameSettings,
  patch?: Partial<GameSettings>,
): GameSettings {
  return {
    gravityMs: num(patch?.gravityMs) ?? base.gravityMs,
    softDropMs: num(patch?.softDropMs) ?? base.softDropMs,
    lockDelayMs: num(patch?.lockDelayMs) ?? base.lockDelayMs,
    hardLockDelayMs: num(patch?.hardLockDelayMs) ?? base.hardLockDelayMs,
  };
}

function mergeInput(
  base: InputConfig,
  patch?: Partial<InputConfig>,
): InputConfig {
  return {
    dasMs: num(patch?.dasMs) ?? base.dasMs,
    arrMs: num(patch?.arrMs) ?? base.arrMs,
    bindings: mergeBindings(base.bindings, patch?.bindings),
  };
}

function mergeGenerator(
  base: GeneratorSettings,
  patch?: Partial<GeneratorSettings>,
): GeneratorSettings {
  return {
    type: isGeneratorType(patch?.type) ? patch.type : base.type,
    ml: {
      strategy: isMlInferenceStrategy(patch?.ml?.strategy)
        ? patch.ml!.strategy
        : base.ml.strategy,
      temperature: num(patch?.ml?.temperature) ?? base.ml.temperature,
      threshold: num(patch?.ml?.threshold) ?? base.ml.threshold,
    },
  };
}

function mergeAudio(
  base: AudioSettings,
  patch?: Partial<AudioSettings>,
): AudioSettings {
  return {
    masterVolume: num(patch?.masterVolume) ?? base.masterVolume,
  };
}

function mergePrivacy(
  base: PrivacySettings,
  patch?: Partial<PrivacySettings>,
): PrivacySettings {
  return {
    shareSnapshots: bool(patch?.shareSnapshots) ?? base.shareSnapshots,
  };
}

function mergeGraphics(
  base: GraphicsSettings,
  patch?: Partial<GraphicsSettings>,
): GraphicsSettings {
  return {
    gridlineOpacity: num(patch?.gridlineOpacity) ?? base.gridlineOpacity,
    highContrast: bool(patch?.highContrast) ?? base.highContrast,
    colorblindMode: bool(patch?.colorblindMode) ?? base.colorblindMode,
  };
}

function mergeButterfinger(
  base: ButterfingerSettings,
  patch?: Partial<ButterfingerSettings>,
): ButterfingerSettings {
  return {
    enabled: bool(patch?.enabled) ?? base.enabled,
    missRate: num(patch?.missRate) ?? base.missRate,
    wrongDirRate: num(patch?.wrongDirRate) ?? base.wrongDirRate,
    extraTapRate: num(patch?.extraTapRate) ?? base.extraTapRate,
    lockNudgeRate: num(patch?.lockNudgeRate) ?? base.lockNudgeRate,
    gravityDropRate: num(patch?.gravityDropRate) ?? base.gravityDropRate,
    lockRotateRate: num(patch?.lockRotateRate) ?? base.lockRotateRate,
  };
}

export function mergeSettings(
  base: Settings,
  patch: Partial<Settings>,
): Settings {
  return {
    game: mergeGame(base.game, patch.game),
    input: mergeInput(base.input, patch.input),
    generator: mergeGenerator(base.generator, patch.generator),
    audio: mergeAudio(base.audio, patch.audio),
    privacy: mergePrivacy(base.privacy, patch.privacy),
    graphics: mergeGraphics(base.graphics, patch.graphics),
    butterfinger: mergeButterfinger(base.butterfinger, patch.butterfinger),
  };
}

export function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (!raw) return DEFAULT_SETTINGS;
    const parsed = JSON.parse(raw) as Partial<Settings>;
    return mergeSettings(DEFAULT_SETTINGS, parsed);
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export function saveSettings(settings: Settings): void {
  try {
    localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings));
  } catch {
    // Non-fatal; ignore persistence errors.
  }
}
