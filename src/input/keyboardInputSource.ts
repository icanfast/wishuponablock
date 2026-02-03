import type { InputSource } from '../core/runner';
import type { GameState, InputFrame } from '../core/types';
import { InputController } from './controller';

export class KeyboardInputSource implements InputSource {
  constructor(private controller: InputController) {}

  sample(_state: GameState, dtMs: number): InputFrame {
    return this.controller.sample(dtMs);
  }
}
