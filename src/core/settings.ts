import {
  DEFAULT_ARR_MS,
  DEFAULT_DAS_MS,
  DEFAULT_GRAVITY_MS,
  DEFAULT_HARD_LOCK_DELAY_MS,
  DEFAULT_LOCK_DELAY_MS,
  DEFAULT_MASTER_VOLUME,
  DEFAULT_SOFT_DROP_MS,
  SETTINGS_STORAGE_KEY,
} from './constants';
import { type GeneratorSettings, isGeneratorType } from './generators';
import type { GameConfig } from './game';
import type { InputConfig } from '../input/controller';

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
}

export interface AudioSettings {
  masterVolume: number;
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
  },
  generator: {
    type: 'bag7',
  },
  audio: {
    masterVolume: DEFAULT_MASTER_VOLUME,
  },
};

function num(v: unknown): number | undefined {
  return typeof v === 'number' && Number.isFinite(v) ? v : undefined;
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
  };
}

function mergeGenerator(
  base: GeneratorSettings,
  patch?: Partial<GeneratorSettings>,
): GeneratorSettings {
  return {
    type: isGeneratorType(patch?.type) ? patch.type : base.type,
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

export function mergeSettings(
  base: Settings,
  patch: Partial<Settings>,
): Settings {
  return {
    game: mergeGame(base.game, patch.game),
    input: mergeInput(base.input, patch.input),
    generator: mergeGenerator(base.generator, patch.generator),
    audio: mergeAudio(base.audio, patch.audio),
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
