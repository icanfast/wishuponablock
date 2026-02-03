import {
  loadSettings,
  mergeSettings,
  saveSettings,
  type Settings,
} from './settings';

export interface SettingsStore {
  get(): Settings;
  apply(patch: Partial<Settings>): Settings;
  subscribe(listener: (settings: Settings) => void): () => void;
}

export function createSettingsStore(
  initial?: Settings,
): SettingsStore {
  let settings = initial ?? loadSettings();
  const listeners = new Set<(s: Settings) => void>();

  const notify = () => {
    for (const listener of listeners) listener(settings);
  };

  return {
    get() {
      return settings;
    },
    apply(patch) {
      settings = mergeSettings(settings, patch);
      saveSettings(settings);
      notify();
      return settings;
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}
