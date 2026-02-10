import { isGeneratorType } from '../core/generators';
import type { Settings } from '../core/settings';
import type { SettingsStore } from '../core/settingsStore';
import type { Board, PieceKind } from '../core/types';
import type { GameScreen } from '../ui/screens/gameScreen';
import type { MenuScreen } from '../ui/screens/menuScreen';
import type { SnapshotService } from './snapshotService';
import type { SnapshotUiState } from './snapshotService';
import type { ModeController } from './modeController';

export type UiController = {
  bindGameUi: () => void;
  attachSnapshotService: (service: SnapshotService) => void;
  attachMenu: (menu: MenuScreen) => void;
  getMenuHandlers: () => {
    onStartPractice: () => void;
    onStartSprint: () => void;
    onStartClassic: () => void;
    onStartCheese: (lines: number) => void;
    onStartCharcuterie: (
      pieces: number,
      options: { simCount: number; seed?: number },
    ) => void;
    onOpenTool: (id: string) => void;
    onSendFeedback: (feedback: string, contact: string | null) => Promise<void>;
  };
  syncMenuSettings: (settings: Settings) => void;
  setMenuTools: (tools: Array<{ id: string; label: string }>) => void;
  showMenuMain: () => void;
  syncSnapshotUi: (state: SnapshotUiState) => void;
  syncGeneratorSelection: (value: string) => void;
};

type UiControllerOptions = {
  game: GameScreen;
  settingsStore: SettingsStore;
  modeController: ModeController;
  useRemoteUpload: boolean;
  getSnapshotState: () => {
    board: Board;
    hold: PieceKind | null;
    linesLeft?: number;
    level?: number;
    score?: number;
  };
  onPauseInputChange: (paused: boolean) => void;
  onMenuClick: () => void;
  onStartGame: () => void;
  onOpenTool: (id: string) => void;
  onSendFeedback: (feedback: string, contact: string | null) => Promise<void>;
};

export function createUiController(options: UiControllerOptions): UiController {
  const {
    game,
    settingsStore,
    modeController,
    useRemoteUpload,
    getSnapshotState,
    onPauseInputChange,
    onMenuClick,
    onStartGame,
    onOpenTool,
    onSendFeedback,
  } = options;

  let snapshotService: SnapshotService | null = null;
  let menuScreen: MenuScreen | null = null;

  const bindGameUi = () => {
    const select = game.generatorSelect;
    select.addEventListener('change', () => {
      const value = select.value;
      if (isGeneratorType(value)) {
        const current = settingsStore.get().generator;
        settingsStore.apply({ generator: { ...current, type: value } });
      }
      select.blur();
    });
    select.addEventListener('keydown', (event) => {
      event.preventDefault();
      select.blur();
    });

    game.menuButton.addEventListener('click', onMenuClick);
    game.commentInput.addEventListener('focus', () => {
      onPauseInputChange(true);
    });
    game.commentInput.addEventListener('blur', () => {
      onPauseInputChange(false);
    });
  };

  const attachSnapshotService = (service: SnapshotService) => {
    snapshotService = service;
    snapshotService.setComment(game.commentInput.value);
    syncSnapshotUi(snapshotService.getState());

    game.commentInput.addEventListener('input', () => {
      snapshotService?.setComment(game.commentInput.value);
    });

    if (useRemoteUpload) {
      game.folderButton.style.display = 'none';
      game.folderStatus.style.display = 'none';
      game.recordRow.style.display = 'none';
      game.recordStatus.textContent = 'Auto upload enabled.';
    }

    game.folderButton.addEventListener('click', async () => {
      await snapshotService?.ensureDirectory();
    });
    game.recordButton.addEventListener('click', async () => {
      if (!snapshotService?.isRecording()) {
        snapshotService?.start();
        return;
      }
      await snapshotService?.stop({ promptForFolder: true });
    });
    game.discardButton.addEventListener('click', () => {
      snapshotService?.discard();
    });

    game.manualButton.addEventListener('click', () => {
      if (!snapshotService) return;
      const { board, hold, linesLeft } = getSnapshotState();
      snapshotService.handleManual(board, hold, { linesLeft });
    });
  };

  const attachMenu = (menu: MenuScreen) => {
    menuScreen = menu;
    menuScreen.showMain();
  };

  const getMenuHandlers = () => ({
    onStartPractice: () => {
      modeController.startPractice();
      onStartGame();
    },
    onStartSprint: () => {
      modeController.startSprint();
      onStartGame();
    },
    onStartClassic: () => {
      modeController.startClassic();
      onStartGame();
    },
    onStartCheese: (lines: number) => {
      modeController.startCheese(lines);
      onStartGame();
    },
    onStartCharcuterie: (
      pieces: number,
      options: { simCount: number; seed?: number },
    ) => {
      modeController.startCharcuterie(pieces, options);
      menuScreen?.setCharcuterieSpinnerVisible(true);
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          onStartGame();
          menuScreen?.setCharcuterieSpinnerVisible(false);
        });
      });
    },
    onOpenTool,
    onSendFeedback,
  });

  const syncMenuSettings = (settings: Settings) => {
    menuScreen?.syncSettings(settings);
  };

  const setMenuTools = (tools: Array<{ id: string; label: string }>) => {
    menuScreen?.setTools(tools);
  };

  const showMenuMain = () => {
    menuScreen?.showMain();
  };

  const syncSnapshotUi = (state: SnapshotUiState) => {
    game.folderStatus.textContent = state.folderStatus;
    game.recordButton.textContent = state.recordButtonLabel;
    game.discardButton.style.display = state.discardVisible
      ? 'inline-block'
      : 'none';
    game.recordStatus.textContent = state.recordStatus;
    game.recordButton.disabled = !state.enabled;
    game.folderButton.disabled = !state.enabled;
    game.manualButton.disabled = !state.enabled;
    if (!state.enabled) {
      game.discardButton.disabled = true;
    } else {
      game.discardButton.disabled = false;
    }
  };

  const syncGeneratorSelection = (value: string) => {
    if (game.generatorSelect.value !== value) {
      game.generatorSelect.value = value;
    }
  };

  return {
    bindGameUi,
    attachSnapshotService,
    attachMenu,
    getMenuHandlers,
    syncMenuSettings,
    setMenuTools,
    showMenuMain,
    syncSnapshotUi,
    syncGeneratorSelection,
  };
}
