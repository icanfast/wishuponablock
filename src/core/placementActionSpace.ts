import { COLS, ROWS } from './constants';
import type { TrajectoryExecutorReachablePlacement } from './trajectoryExecutor';

export const PLACEMENT_ACTION_X_MIN = -4;
export const PLACEMENT_ACTION_X_MAX = COLS + 3;
export const PLACEMENT_ACTION_Y_MIN = -4;
export const PLACEMENT_ACTION_Y_MAX = ROWS + 3;
export const PLACEMENT_ACTION_ROTATIONS = 4;
export const PLACEMENT_ACTION_HOLD_VARIANTS = 2;

export const PLACEMENT_ACTION_X_COUNT =
  PLACEMENT_ACTION_X_MAX - PLACEMENT_ACTION_X_MIN + 1;
export const PLACEMENT_ACTION_Y_COUNT =
  PLACEMENT_ACTION_Y_MAX - PLACEMENT_ACTION_Y_MIN + 1;
export const PLACEMENT_ACTION_DIM =
  PLACEMENT_ACTION_HOLD_VARIANTS *
  PLACEMENT_ACTION_ROTATIONS *
  PLACEMENT_ACTION_X_COUNT *
  PLACEMENT_ACTION_Y_COUNT;
export const PLACEMENT_ACTION_NO_HOLD_DIM =
  PLACEMENT_ACTION_ROTATIONS *
  PLACEMENT_ACTION_X_COUNT *
  PLACEMENT_ACTION_Y_COUNT;
export const PLACEMENT_ACTION_HOLD_STEP_INDEX = PLACEMENT_ACTION_NO_HOLD_DIM;
export const PLACEMENT_ACTION_HOLD_STEP_DIM = PLACEMENT_ACTION_NO_HOLD_DIM + 1;

const normalizeRotation = (value: number): number => {
  const normalized = Math.trunc(value) % PLACEMENT_ACTION_ROTATIONS;
  return normalized < 0 ? normalized + PLACEMENT_ACTION_ROTATIONS : normalized;
};

type PlacementActionLike = {
  holdUsed: boolean;
  lockRotation: number;
  lockX: number;
  lockY: number;
};

export const placementActionIndexFromFields = (
  placement: PlacementActionLike,
): number | null => {
  const rot = normalizeRotation(placement.lockRotation);
  const x = Math.trunc(placement.lockX);
  const y = Math.trunc(placement.lockY);
  if (x < PLACEMENT_ACTION_X_MIN || x > PLACEMENT_ACTION_X_MAX) return null;
  if (y < PLACEMENT_ACTION_Y_MIN || y > PLACEMENT_ACTION_Y_MAX) return null;
  const hold = placement.holdUsed ? 1 : 0;
  const xIndex = x - PLACEMENT_ACTION_X_MIN;
  const yIndex = y - PLACEMENT_ACTION_Y_MIN;
  return (
    (((hold * PLACEMENT_ACTION_ROTATIONS + rot) * PLACEMENT_ACTION_X_COUNT +
      xIndex) *
      PLACEMENT_ACTION_Y_COUNT +
      yIndex) >>>
    0
  );
};

export const placementActionIndexFromPlacement = (
  placement: TrajectoryExecutorReachablePlacement,
): number | null =>
  placementActionIndexFromFields({
    holdUsed: placement.holdUsed,
    lockRotation: placement.lockRotation,
    lockX: placement.lockX,
    lockY: placement.lockY,
  });

export const placementActionIndexFromNoHoldFields = (
  placement: Omit<PlacementActionLike, 'holdUsed'>,
): number | null =>
  placementActionIndexFromFields({
    holdUsed: false,
    lockRotation: placement.lockRotation,
    lockX: placement.lockX,
    lockY: placement.lockY,
  });

export const placementActionIndexFromNoHoldPlacement = (
  placement: TrajectoryExecutorReachablePlacement,
): number | null =>
  placement.holdUsed
    ? null
    : placementActionIndexFromNoHoldFields({
        lockRotation: placement.lockRotation,
        lockX: placement.lockX,
        lockY: placement.lockY,
      });

export const isPlacementHoldStepActionIndex = (actionIndex: number): boolean =>
  Math.trunc(actionIndex) === PLACEMENT_ACTION_HOLD_STEP_INDEX;
