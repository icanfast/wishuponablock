import type { Settings } from '../core/settings';
import lockSfxUrl from '../assets/sfx/lock.ogg';

export type SoundService = {
  playLock: () => void;
  applySettings: (settings: Settings) => void;
};

type SoundServiceOptions = {
  settings: Settings;
  lockUrl?: string;
};

export function createSoundService(options: SoundServiceOptions): SoundService {
  const lockSound = new Audio(options.lockUrl ?? lockSfxUrl);
  lockSound.preload = 'auto';
  lockSound.volume = options.settings.audio.masterVolume;

  return {
    playLock: () => {
      lockSound.currentTime = 0;
      void lockSound.play().catch(() => {
        // Ignore autoplay restrictions and playback errors.
      });
    },
    applySettings: (settings) => {
      lockSound.volume = settings.audio.masterVolume;
    },
  };
}
