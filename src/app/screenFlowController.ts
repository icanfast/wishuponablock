import type { GameRuntime } from './runtime';
import type { SessionController } from './sessionController';
import type { InputService } from './inputService';
import type { ToolHost } from '../ui/tools/toolHost';
import type { GameScreen } from '../ui/screens/gameScreen';
import type { MenuScreen } from '../ui/screens/menuScreen';
import type { SettingsStore } from '../core/settingsStore';
import type { ScreenHandle } from './screenManager';

export type ScreenFlowController = {
  makeMenuScreen: (root: HTMLElement) => ScreenHandle;
  makeGameScreen: (root: HTMLElement) => ScreenHandle;
  makeToolScreen: (
    root: HTMLElement,
    getActiveToolId: () => string,
  ) => ScreenHandle;
};

type ScreenFlowOptions = {
  settingsStore: SettingsStore;
  runtime: GameRuntime;
  gameScreen: GameScreen;
  menuScreen: MenuScreen;
  inputService: InputService;
  sessionController: SessionController;
  toolHost: ToolHost;
  gameGfx: { visible: boolean };
  toolGfx: { visible: boolean };
  stopRecording: () => void;
  startRecording: () => void;
};

export function createScreenFlowController(
  options: ScreenFlowOptions,
): ScreenFlowController {
  const {
    settingsStore,
    runtime,
    gameScreen,
    menuScreen,
    inputService,
    sessionController,
    toolHost,
    gameGfx,
    toolGfx,
    stopRecording,
    startRecording,
  } = options;

  return {
    makeMenuScreen: (root) => ({
      root,
      enter: () => {
        menuScreen.showMain();
        runtime.setPausedByMenu(true);
        gameGfx.visible = false;
        toolGfx.visible = false;
        stopRecording();
        gameScreen.gameOverLabel.style.display = 'none';
        toolHost.deactivate();
      },
    }),
    makeGameScreen: (root) => ({
      root,
      enter: () => {
        runtime.setPausedByMenu(false);
        gameGfx.visible = true;
        toolGfx.visible = false;
        const nextSettings = settingsStore.get();
        inputService.applySettings(nextSettings);
        sessionController.setSettings(nextSettings);
        sessionController.rebuildSession();
        gameScreen.gameOverLabel.style.display = 'none';
        startRecording();
        toolHost.deactivate();
      },
      leave: () => {
        stopRecording();
        gameScreen.gameOverLabel.style.display = 'none';
      },
    }),
    makeToolScreen: (root, getActiveToolId) => ({
      root,
      enter: async () => {
        runtime.setPausedByMenu(true);
        gameGfx.visible = false;
        toolGfx.visible = true;
        stopRecording();
        gameScreen.gameOverLabel.style.display = 'none';
        await toolHost.setActive(getActiveToolId());
      },
      leave: () => {
        toolHost.deactivate();
      },
    }),
  };
}
