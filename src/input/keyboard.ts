export class Keyboard {
  private held = new Set<string>();
  private pressed = new Set<string>(); // “went down since last consume”

  constructor() {
    window.addEventListener('keydown', (e) => {
      if (isTypingTarget(e)) return;
      const code = e.code;
      if (!this.held.has(code)) {
        this.pressed.add(code);
      }
      this.held.add(code);

      if (
        ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space'].includes(
          code,
        )
      ) {
        e.preventDefault();
      }
    });

    window.addEventListener('keyup', (e) => {
      this.held.delete(e.code);
    });

    window.addEventListener('blur', () => {
      this.held.clear();
      this.pressed.clear();
    });
  }

  isHeld(code: string): boolean {
    return this.held.has(code);
  }

  consumePressed(code: string): boolean {
    if (this.pressed.has(code)) {
      this.pressed.delete(code);
      return true;
    }
    return false;
  }
}

function isTypingTarget(e: KeyboardEvent): boolean {
  const target = e.target;
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  const tag = target.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';
}
