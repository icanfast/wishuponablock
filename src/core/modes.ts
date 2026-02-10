import type { Game } from './game';
import type { Settings } from './settings';

export interface ModeOptions {
  cheeseLines?: number;
  pieces?: number;
  simCount?: number;
  seed?: number;
}

export interface GameMode {
  id: 'practice' | 'sprint' | 'classic' | 'cheese' | 'charcuterie';
  label: string;
  lineGoal?: number;
  classicStartLevel?: number;
  scoringEnabled?: boolean;
  settingsPatch?: Partial<Settings>;
  onStart?: (game: Game, options: ModeOptions) => void;
}

export const MODES: GameMode[] = [
  {
    id: 'practice',
    label: 'Practice',
  },
  {
    id: 'sprint',
    label: 'Sprint',
    lineGoal: 40,
  },
  {
    id: 'classic',
    label: 'Classic',
    classicStartLevel: 0,
    scoringEnabled: true,
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
  {
    id: 'charcuterie',
    label: 'Charcuterie',
  },
];

export function normalizeModeId(id: string): GameMode['id'] | null {
  if (id === 'default') return 'practice';
  const match = MODES.find((mode) => mode.id === id);
  return match ? match.id : null;
}

export function getMode(id: string): GameMode {
  const normalized = normalizeModeId(id);
  return MODES.find((mode) => mode.id === normalized) ?? MODES[0];
}
