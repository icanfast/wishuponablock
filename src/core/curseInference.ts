import { softmax } from './wubModel';

export function inferCurseDistribution(logits: Float32Array): Float32Array {
  const base = softmax(logits);
  return flipDistribution(base);
}

function flipDistribution(probs: Float32Array): Float32Array {
  const out = new Float32Array(probs.length);
  if (out.length === 0) return out;

  let sum = 0;
  for (let i = 0; i < probs.length; i++) {
    const value = Math.max(0, 1 - probs[i]);
    out[i] = value;
    sum += value;
  }

  if (sum <= 0) {
    const uniform = 1 / out.length;
    out.fill(uniform);
    return out;
  }

  for (let i = 0; i < out.length; i++) {
    out[i] /= sum;
  }
  return out;
}
