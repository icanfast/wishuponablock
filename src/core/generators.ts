import type { PieceGenerator } from './generator';
import { Bag7 } from './bag7';
import { Bag8I } from './bag8I';
import { BagInconvenient } from './bagInconvenient';
import { RandomGenerator } from './randomGenerator';
import { NesGenerator } from './nesGenerator';

export const GENERATOR_TYPES = [
  'bag7',
  'bag8i',
  'inconvenient',
  'random',
  'nes',
] as const;
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
    case 'bag8i':
      return (seed) => new Bag8I(seed);
    case 'inconvenient':
      return (seed) => new BagInconvenient(seed);
    case 'nes':
      return (seed) => new NesGenerator(seed);
    case 'random':
      return (seed) => new RandomGenerator(seed);
    case 'bag7':
    default:
      return (seed) => new Bag7(seed);
  }
}
