import type { InputFrame } from './types';
import type { Settings } from './settings';

export interface ReplayHeader {
  protocolVersion: number;
  buildVersion?: string;
  seed: number;
  generator: {
    type: string;
    params?: Record<string, unknown>;
  };
  settings: Settings;
  createdAt: string;
}

export interface Replay {
  header: ReplayHeader;
  inputs: InputFrame[];
}
