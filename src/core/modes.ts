import type { Game } from './game';
import type { Settings } from './settings';

export interface ModeOptions {
  cheeseLines?: number;
}

export interface GameMode {
  id: 'default' | 'cheese';
  label: string;
  settingsPatch?: Partial<Settings>;
  onStart?: (game: Game, options: ModeOptions) => void;
}

export const MODES: GameMode[] = [
  {
    id: 'default',
    label: 'Default',
  },
  {
    id: 'cheese',
    label: 'Cheese',
    onStart: (game, options) => {
      if (options.cheeseLines) {
        game.applyCheese(options.cheeseLines);
      }
    },
  },
];

export function getMode(id: GameMode['id']): GameMode {
  return MODES.find((mode) => mode.id === id) ?? MODES[0];
}
