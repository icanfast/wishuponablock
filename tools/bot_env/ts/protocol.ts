export type PieceSourceProfile = 'bag7' | 'active_generator' | 'random';
export type BotObservationSpace = 'model_head_v1' | 'raw_v1';
export type PlacementExecutionMode = 'commands' | 'teleport';
export type RewardFunctionId = 'v1' | 'v2' | 'v3' | 'harddrop_v1';
export type BotActionSpaceKind = 'placement_full_v1' | 'placement_hold_step_v2';

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
    | 'set_piece_sources'
    | 'set_curriculum'
    | 'set_reward_blend_step'
    | 'evaluate_hold_candidates_many'
    | 'probe_actions_many'
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
  phaseContextEnabled?: boolean;
  placementExecutionMode?: PlacementExecutionMode;
  actionSpaceKind?: BotActionSpaceKind;
  pieceSourceProfile?: PieceSourceProfile;
  queuePolicyId?: string;
  maxPiecesPerEpisode?: number;
  rewardFunctionFrom?: RewardFunctionId;
  rewardFunctionTo?: RewardFunctionId;
  rewardBlendTimesteps?: number;
  rewardBlendUnit?: 'timesteps' | 'updates';
  rewardBlendStartStep?: number;
  seed?: number;
};

export type ResetManyPayload = {
  envIds?: number[];
  seeds?: number[];
  initialBoards?: Array<number[][] | null>;
};

export type SetPieceSourcePayload = {
  pieceSourceProfile?: PieceSourceProfile;
};

export type SetPieceSourcesPayload = {
  envIds?: number[];
  pieceSourceProfiles?: PieceSourceProfile[];
};

export type SetCurriculumPayload = {
  topK?: number;
  biasStrength?: number;
  dangerHeight?: number;
  computeScores?: boolean;
};

export type SetRewardBlendStepPayload = {
  transitionStep?: number;
};

export type StepManyPayload = {
  envIds?: number[];
  actions?: number[];
};

export type EvaluateHoldCandidatesManyPayload = {
  envIds?: number[];
};

export type ProbeActionsManyPayload = {
  envIds?: number[];
};

export type HoldCandidateEvaluation = {
  action_index: number;
  hold_used: boolean;
  immediate_reward_no_hold_tax: number;
  done: boolean;
  obs: number[];
};

export type EvaluateHoldCandidatesBatchResult = {
  candidates: HoldCandidateEvaluation[][];
};

export type PlacementActionProbe = {
  action_index: number;
  planning_score: number;
  immediate_score: number;
  continuation_best_score: number;
  hold_step: boolean;
  hold_used: boolean;
  done: boolean;
  top_out: boolean;
  lines_cleared: number;
  height_delta: number;
  holes_delta: number;
  hold_term: number;
  kick_term: number;
  soft_drop_term: number;
  hole_term: number;
  hole_extended_term: number;
  board_quality_delta_term: number;
  board_quality_absolute_term: number;
  commands: string[];
  lock_piece?: string;
  lock_rotation?: number;
  lock_x?: number;
  lock_y?: number;
};

export type ProbeActionsBatchResult = {
  obs: number[][];
  action_masks: number[][];
  action_biases: number[][];
  action_scores: number[][];
  probes: PlacementActionProbe[][];
};

export type StepBatchResult = {
  obs: number[][];
  action_masks: number[][];
  action_biases: number[][];
  action_scores: number[][];
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
