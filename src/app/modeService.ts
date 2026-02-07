import type { GameMode, ModeOptions } from '../core/modes';
import type { Settings } from '../core/settings';
import type { Game } from '../core/game';

export function applyModeSettings(base: Settings, mode: GameMode): Settings {
  if (!mode.settingsPatch) return base;
  return {
    ...base,
    ...mode.settingsPatch,
    game: {
      ...base.game,
      ...mode.settingsPatch.game,
    },
    input: {
      ...base.input,
      ...mode.settingsPatch.input,
    },
    generator: {
      ...base.generator,
      ...mode.settingsPatch.generator,
    },
    audio: {
      ...base.audio,
      ...mode.settingsPatch.audio,
    },
  };
}

export function runModeStart(
  game: Game,
  mode: GameMode,
  options: ModeOptions,
): void {
  mode.onStart?.(game, options);
}
