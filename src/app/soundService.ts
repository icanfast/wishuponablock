import type { Settings } from '../core/settings';
import lockSfxUrl from '../assets/sfx/lock.ogg';

export type SoundService = {
  playLock: () => void;
  playCombo: (combo: number) => void;
  applySettings: (settings: Settings) => void;
};

type SoundServiceOptions = {
  settings: Settings;
  lockUrl?: string;
  lockUrls?: string[];
};

export function createSoundService(options: SoundServiceOptions): SoundService {
  const lockSound = new Audio(options.lockUrl ?? lockSfxUrl);
  lockSound.preload = 'auto';
  lockSound.volume = options.settings.audio.masterVolume;

  const baseUrl = import.meta.env.BASE_URL ?? '/';
  const comboNotes = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C2'] as const;
  const comboSounds = comboNotes.map((note) => {
    const audio = new Audio(`${baseUrl}sfx/${note}.ogg`);
    audio.preload = 'auto';
    audio.volume = options.settings.audio.masterVolume;
    return audio;
  });
  const comboIndex = (combo: number): number => {
    if (!Number.isFinite(combo) || combo <= 0) return -1;
    return Math.min(comboNotes.length - 1, Math.max(1, combo) - 1);
  };

  return {
    playLock: () => {
      lockSound.currentTime = 0;
      void lockSound.play().catch(() => {
        // Ignore autoplay restrictions and playback errors.
      });
    },
    playCombo: (combo) => {
      const index = comboIndex(combo);
      if (index < 0) return;
      const sound = comboSounds[index];
      sound.currentTime = 0;
      void sound.play().catch(() => {
        // Ignore autoplay restrictions and playback errors.
      });
    },
    applySettings: (settings) => {
      lockSound.volume = settings.audio.masterVolume;
      for (const sound of comboSounds) {
        sound.volume = settings.audio.masterVolume;
      }
    },
  };
}
