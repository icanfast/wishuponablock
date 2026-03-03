export type PieceSourceProfile = 'bag7' | 'active_generator';
export type BotObservationSpace = 'model_head_v1' | 'raw_v1';

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
  cmd:
    | 'init'
    | 'set_piece_source'
    | 'reset_many'
    | 'step_many'
    | 'pop_trajectory'
    | 'close';
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
  observationSpace?: BotObservationSpace;
  pieceSourceProfile?: PieceSourceProfile;
  queuePolicyId?: string;
  maxPiecesPerEpisode?: number;
  seed?: number;
};

export type ResetManyPayload = {
  envIds?: number[];
  seeds?: number[];
};

export type SetPieceSourcePayload = {
  pieceSourceProfile?: PieceSourceProfile;
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
  profile?: {
    batch_total_s: number;
    env_count: number;
    step_env_total_s?: number;
    step_choices_current_s?: number;
    step_runner_s?: number;
    step_reward_s?: number;
    step_obs_s?: number;
    step_choices_next_s?: number;
    reset_env_total_s?: number;
    reset_obs_s?: number;
    reset_choices_s?: number;
  };
};
