import type { PieceGenerator } from './generator';
import { Bag7 } from './bag7';
import { RandomGenerator } from './randomGenerator';

export const GENERATOR_TYPES = ['bag7', 'random'] as const;
export type GeneratorType = (typeof GENERATOR_TYPES)[number];

export interface GeneratorSettings {
  type: GeneratorType;
}

export function isGeneratorType(value: unknown): value is GeneratorType {
  return (
    typeof value === 'string' &&
    (GENERATOR_TYPES as readonly string[]).includes(value)
  );
}

export function createGeneratorFactory(
  settings: GeneratorSettings,
): (seed: number) => PieceGenerator {
  switch (settings.type) {
    case 'random':
      return (seed) => new RandomGenerator(seed);
    case 'bag7':
    default:
      return (seed) => new Bag7(seed);
  }
}
