import { getMode, type GameMode, type ModeOptions } from '../core/modes';

export type ModeState = {
  mode: GameMode;
  options: ModeOptions;
};

export type ModeController = {
  getState: () => ModeState;
  setOnModeChange: (
    handler: ((mode: GameMode, options: ModeOptions) => void) | null,
  ) => void;
  setMode: (id: GameMode['id'], options: ModeOptions) => ModeState;
  startPractice: () => ModeState;
  startSprint: () => ModeState;
  startClassic: () => ModeState;
  startCheese: (lines: number) => ModeState;
  startCharcuterie: (
    pieces: number,
    options: { simCount: number; seed?: number },
  ) => ModeState;
};

type ModeControllerOptions = {
  initialModeId?: GameMode['id'];
  initialOptions?: ModeOptions;
};

export function createModeController(
  options: ModeControllerOptions = {},
): ModeController {
  let mode = getMode(options.initialModeId ?? 'practice');
  let modeOptions: ModeOptions = { ...(options.initialOptions ?? {}) };
  let onModeChange: ((mode: GameMode, options: ModeOptions) => void) | null =
    null;

  const apply = (id: GameMode['id'], nextOptions: ModeOptions): ModeState => {
    mode = getMode(id);
    modeOptions = { ...nextOptions };
    onModeChange?.(mode, modeOptions);
    return { mode, options: modeOptions };
  };

  return {
    getState: () => ({ mode, options: modeOptions }),
    setOnModeChange: (handler) => {
      onModeChange = handler;
    },
    setMode: (id, nextOptions) => apply(id, nextOptions),
    startPractice: () => apply('practice', {}),
    startSprint: () => apply('sprint', {}),
    startClassic: () => apply('classic', {}),
    startCheese: (lines) => apply('cheese', { cheeseLines: lines }),
    startCharcuterie: (pieces, next) =>
      apply('charcuterie', {
        pieces,
        simCount: next.simCount,
        ...(next.seed !== undefined ? { seed: next.seed } : {}),
      }),
  };
}
