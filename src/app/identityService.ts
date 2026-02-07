export type IdentityService = {
  getDeviceId: () => string;
  getUserId: () => string | null;
  setUserId: (userId: string | null) => void;
  subscribe: (listener: (userId: string | null) => void) => () => void;
};

const DEVICE_ID_KEY = 'wub_device_id';
const USER_ID_KEY = 'wub_user_id';

const generateDeviceId = (): string => {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID();
  }
  const rand = Math.random().toString(36).slice(2, 10);
  const time = Date.now().toString(36);
  return `wub_${time}_${rand}`;
};

const readStorage = (key: string): string | null => {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
};

const writeStorage = (key: string, value: string | null) => {
  try {
    if (value == null) {
      localStorage.removeItem(key);
    } else {
      localStorage.setItem(key, value);
    }
  } catch {
    // Ignore storage failures.
  }
};

export function createIdentityService(): IdentityService {
  let deviceId = readStorage(DEVICE_ID_KEY);
  if (!deviceId) {
    deviceId = generateDeviceId();
    writeStorage(DEVICE_ID_KEY, deviceId);
  }

  let userId = readStorage(USER_ID_KEY);
  if (userId != null && !userId.trim()) {
    userId = null;
    writeStorage(USER_ID_KEY, null);
  }

  const listeners = new Set<(next: string | null) => void>();

  const notify = () => {
    for (const listener of listeners) {
      listener(userId);
    }
  };

  return {
    getDeviceId: () => deviceId!,
    getUserId: () => userId,
    setUserId: (next) => {
      const trimmed = (next ?? '').trim();
      const normalized = trimmed.length > 0 ? trimmed : null;
      if (normalized === userId) return;
      userId = normalized;
      writeStorage(USER_ID_KEY, userId);
      notify();
    },
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}
