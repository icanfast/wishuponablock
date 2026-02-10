import type { PiecePalette } from '../../core/palette';

export type ToolController = {
  id: string;
  label: string;
  root: HTMLElement;
  enter: () => void | Promise<void>;
  leave: () => void;
  setPiecePalette?: (palette: PiecePalette) => void;
};

export type ToolRegistryEntry = {
  id: string;
  label: string;
};

export type ToolHost = {
  register: (tool: ToolController) => void;
  setActive: (id: string) => Promise<void>;
  deactivate: () => void;
  getActiveId: () => string | null;
  list: () => ToolRegistryEntry[];
  get: (id: string) => ToolController | undefined;
};

export function createToolHost(container: HTMLElement): ToolHost {
  const tools = new Map<string, ToolController>();
  let activeId: string | null = null;

  const register = (tool: ToolController) => {
    if (tools.has(tool.id)) {
      throw new Error(`Tool already registered: ${tool.id}`);
    }
    tools.set(tool.id, tool);
    tool.root.style.display = 'none';
    container.appendChild(tool.root);
  };

  const setActive = async (id: string) => {
    if (activeId === id) return;
    if (activeId) {
      const current = tools.get(activeId);
      if (current) {
        current.leave();
        current.root.style.display = 'none';
      }
    }
    const next = tools.get(id);
    if (!next) {
      activeId = null;
      return;
    }
    activeId = id;
    next.root.style.display = 'block';
    await next.enter();
  };

  const deactivate = () => {
    if (!activeId) return;
    const current = tools.get(activeId);
    if (current) {
      current.leave();
      current.root.style.display = 'none';
    }
    activeId = null;
  };

  return {
    register,
    setActive,
    deactivate,
    getActiveId: () => activeId,
    list: () =>
      Array.from(tools.values()).map((tool) => ({
        id: tool.id,
        label: tool.label,
      })),
    get: (id: string) => tools.get(id),
  };
}
