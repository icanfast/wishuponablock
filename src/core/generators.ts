import type { PieceGenerator } from './generator';
import { Bag7 } from './bag7';
import { Bag8I } from './bag8I';
import { BagInconvenient } from './bagInconvenient';
import { RandomGenerator } from './randomGenerator';
import { NesGenerator } from './nesGenerator';
import { ModelGenerator } from './modelGenerator';
import type { LoadedModel } from './wubModel';

export const GENERATOR_TYPES = [
  'bag7',
  'bag8i',
  'inconvenient',
  'random',
  'nes',
  'ml',
] as const;
export type GeneratorType = (typeof GENERATOR_TYPES)[number];

export interface GeneratorSettings {
  type: GeneratorType;
}

export interface GeneratorFactoryOptions {
  mlModel?: LoadedModel | null;
  mlModelPromise?: Promise<LoadedModel | null>;
}

export function isGeneratorType(value: unknown): value is GeneratorType {
  return (
    typeof value === 'string' &&
    (GENERATOR_TYPES as readonly string[]).includes(value)
  );
}

export function createGeneratorFactory(
  settings: GeneratorSettings,
  options: GeneratorFactoryOptions = {},
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
    case 'ml':
      return (seed) =>
        new ModelGenerator(
          seed,
          options.mlModel ?? null,
          options.mlModelPromise,
        );
    case 'bag7':
    default:
      return (seed) => new Bag7(seed);
  }
}
