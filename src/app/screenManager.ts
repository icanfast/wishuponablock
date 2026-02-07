export type ScreenId = 'menu' | 'game' | 'tool';

export type ScreenHandle = {
  root: HTMLElement;
  enter?: () => void | Promise<void>;
  leave?: () => void;
};

export type ScreenManager = {
  register: (id: ScreenId, screen: ScreenHandle) => void;
  setActive: (id: ScreenId) => Promise<void>;
  getActive: () => ScreenId | null;
};

export function createScreenManager(): ScreenManager {
  const screens = new Map<ScreenId, ScreenHandle>();
  let active: ScreenId | null = null;

  const register = (id: ScreenId, screen: ScreenHandle) => {
    if (screens.has(id)) {
      throw new Error(`Screen already registered: ${id}`);
    }
    screens.set(id, screen);
    screen.root.style.display = 'none';
  };

  const setActive = async (id: ScreenId) => {
    if (active === id) return;
    if (active) {
      const current = screens.get(active);
      if (current) {
        current.leave?.();
        current.root.style.display = 'none';
      }
    }
    const next = screens.get(id);
    if (!next) {
      active = null;
      return;
    }
    active = id;
    next.root.style.display = 'block';
    await next.enter?.();
  };

  return {
    register,
    setActive,
    getActive: () => active,
  };
}
