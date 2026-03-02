export type PieceSourceProfile = 'bag7' | 'active_generator';

export type JsonValue =
  | null
  | boolean
  | number
  | string
  | JsonValue[]
  | { [key: string]: JsonValue };

export type JsonObject = { [key: string]: JsonValue };

export type BridgeRequest = {
  id: number;
  cmd: 'init' | 'reset_many' | 'step_many' | 'close';
  payload?: JsonValue;
};

export type BridgeResponse = {
  id: number;
  ok: boolean;
  result?: JsonValue;
  error?: string;
};

export type InitPayload = {
  modeId?: string;
  numEnvs?: number;
  modelPath?: string;
  pieceSourceProfile?: PieceSourceProfile;
  queuePolicyId?: string;
  maxPiecesPerEpisode?: number;
  seed?: number;
};

export type ResetManyPayload = {
  envIds?: number[];
  seeds?: number[];
};

export type StepManyPayload = {
  envIds?: number[];
  actions?: number[];
};

export type StepBatchResult = {
  obs: number[][];
  action_masks: number[][];
  rewards: number[];
  dones: boolean[];
  infos: JsonObject[];
};
