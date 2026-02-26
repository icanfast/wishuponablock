import { describe, expect, it } from 'vitest';
import {
  DEFAULT_PERSONAL_TRAINING_PIPELINE_ID,
  getPersonalTrainingPipeline,
  resolvePersonalTrainingPipelineId,
} from '../app/trainingPipelines';

describe('training pipelines', () => {
  it('resolves unknown ids to the default pipeline', () => {
    expect(resolvePersonalTrainingPipelineId('unknown_pipeline')).toBe(
      DEFAULT_PERSONAL_TRAINING_PIPELINE_ID,
    );
  });

  it('returns the default pipeline contract', () => {
    const pipeline = getPersonalTrainingPipeline(null);
    expect(pipeline.id).toBe(DEFAULT_PERSONAL_TRAINING_PIPELINE_ID);
    expect(pipeline.rewardPolicyId).toBe('comfort_v1');
    expect(pipeline.strategyId).toBe('mlp_head_policy_v1');
    expect(pipeline.trainableLayers).toEqual(['mlp.0', 'mlp.2']);
    expect(pipeline.minSamples).toBeGreaterThanOrEqual(1);
    expect(pipeline.evalGate.holdoutRatio).toBeGreaterThan(0);
    expect(pipeline.evalGate.holdoutRatio).toBeLessThan(1);
    expect(pipeline.evalGate.minHoldoutSamples).toBeGreaterThanOrEqual(1);
    expect(pipeline.evalGate.minTrainSamples).toBeGreaterThanOrEqual(1);
  });
});
