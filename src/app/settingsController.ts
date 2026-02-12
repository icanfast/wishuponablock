import type { Settings } from '../core/settings';
import type { SettingsStore } from '../core/settingsStore';
import type { InputService } from './inputService';
import type { SoundService } from './soundService';
import type { UiController } from './uiController';
import type { SessionController } from './sessionController';
import type { ModelService, ModelStatus } from './modelService';
import { usesModelGenerator } from '../core/generators';

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
  onGraphicsChange?: (settings: Settings) => void;
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
    onGraphicsChange,
  } = options;
  let generatorType = settingsStore.get().generator.type;

  const handleSettingsChange = (next: Settings) => {
    soundService.applySettings(next);
    inputService.applySettings(next);
    uiController.syncMenuSettings(next);
    onGraphicsChange?.(next);

    if (next.generator.type !== generatorType) {
      generatorType = next.generator.type;
      if (usesModelGenerator(next.generator.type)) {
        void modelService.ensureLoaded();
      }
      onModelStatus(modelService.getStatus());
      sessionController.handleGeneratorChange(next);
      uiController.syncGeneratorSelection(next.generator.type);
      return;
    }

    sessionController.applyConfig(next);
    uiController.syncGeneratorSelection(next.generator.type);
  };

  return {
    start: () => settingsStore.subscribe(handleSettingsChange),
  };
}
