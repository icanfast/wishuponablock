import type { Settings } from '../core/settings';
import type { SettingsStore } from '../core/settingsStore';
import type { InputService } from './inputService';
import type { SoundService } from './soundService';
import type { UiController } from './uiController';
import type { SessionController } from './sessionController';
import type { ModelService, ModelStatus } from './modelService';

export type SettingsController = {
  start: () => () => void;
};

type SettingsControllerOptions = {
  settingsStore: SettingsStore;
  inputService: InputService;
  soundService: SoundService;
  uiController: UiController;
  sessionController: SessionController;
  modelService: ModelService;
  onModelStatus: (status: ModelStatus) => void;
};

export function createSettingsController(
  options: SettingsControllerOptions,
): SettingsController {
  const {
    settingsStore,
    inputService,
    soundService,
    uiController,
    sessionController,
    modelService,
    onModelStatus,
  } = options;
  let generatorType = settingsStore.get().generator.type;

  const handleSettingsChange = (next: Settings) => {
    soundService.applySettings(next);
    inputService.applySettings(next);
    uiController.syncMenuSettings(next);

    if (next.generator.type !== generatorType) {
      generatorType = next.generator.type;
      if (next.generator.type === 'ml') {
        void modelService.ensureLoaded();
      }
      onModelStatus(modelService.getStatus());
      sessionController.handleGeneratorChange(next);
      uiController.syncGeneratorSelection(next.generator.type);
      return;
    }

    sessionController.applyConfig({
      ...next,
      game: {
        ...next.game,
        lockNudgeRate: next.butterfinger.enabled
          ? next.butterfinger.lockNudgeRate
          : 0,
        gravityDropRate: next.butterfinger.enabled
          ? next.butterfinger.gravityDropRate
          : 0,
        lockRotateRate: next.butterfinger.enabled
          ? next.butterfinger.lockRotateRate
          : 0,
      },
    });
    uiController.syncGeneratorSelection(next.generator.type);
  };

  return {
    start: () => settingsStore.subscribe(handleSettingsChange),
  };
}
