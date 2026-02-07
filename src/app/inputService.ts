import type { InputSource } from '../core/runner';
import type { Settings } from '../core/settings';
import { InputController } from '../input/controller';
import { Keyboard } from '../input/keyboard';
import { KeyboardInputSource } from '../input/keyboardInputSource';
import { ButterfingerInputSource } from '../input/butterfingerInputSource';

export type InputService = {
  getInputSource: () => InputSource;
  applySettings: (settings: Settings) => void;
  setOnInputSourceChange: (
    handler: ((source: InputSource) => void) | null,
  ) => void;
};

type InputServiceOptions = {
  settings: Settings;
};

export function createInputService(options: InputServiceOptions): InputService {
  const kb = new Keyboard();
  const controller = new InputController(kb, options.settings.input);
  const baseSource = new KeyboardInputSource(controller);
  const butterfingerSource = new ButterfingerInputSource(
    baseSource,
    options.settings.butterfinger,
  );
  let inputSource: InputSource = options.settings.butterfinger.enabled
    ? butterfingerSource
    : baseSource;
  let onChange: ((source: InputSource) => void) | null = null;

  const updateInputSource = (settings: Settings) => {
    controller.setConfig(settings.input);
    butterfingerSource.setConfig(settings.butterfinger);
    inputSource = settings.butterfinger.enabled
      ? butterfingerSource
      : baseSource;
    onChange?.(inputSource);
  };

  return {
    getInputSource: () => inputSource,
    applySettings: updateInputSource,
    setOnInputSourceChange: (handler) => {
      onChange = handler;
    },
  };
}
