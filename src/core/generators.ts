import type { PieceGenerator } from './generator';
import { Bag7 } from './bag7';
import { Bag8I } from './bag8I';
import { BagInconvenient } from './bagInconvenient';
import { RandomGenerator } from './randomGenerator';
import { NesGenerator } from './nesGenerator';
import { ModelGenerator } from './modelGenerator';
import { CurseModelGenerator } from './curseModelGenerator';
import { DEFAULT_ML_INFERENCE } from './constants';
import type { LoadedModel } from './wubModel';

export const GENERATOR_TYPES = [
  'bag7',
  'bag8i',
  'inconvenient',
  'random',
  'nes',
  'ml',
  'curse',
] as const;
export type GeneratorType = (typeof GENERATOR_TYPES)[number];

export type MlInferenceStrategy = 'clean_uniform' | 'threshold';
export type MlInferenceConfig = {
  strategy: MlInferenceStrategy;
  temperature: number;
  threshold: number;
  postSharpness: number;
};

export interface GeneratorSettings {
  type: GeneratorType;
  ml: MlInferenceConfig;
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

export function isMlInferenceStrategy(
  value: unknown,
): value is MlInferenceStrategy {
  return value === 'clean_uniform' || value === 'threshold';
}

export function usesModelGenerator(type: GeneratorType): boolean {
  return type === 'ml' || type === 'curse';
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
          settings.ml ?? DEFAULT_ML_INFERENCE,
        );
    case 'curse':
      return (seed) =>
        new CurseModelGenerator(
          seed,
          options.mlModel ?? null,
          options.mlModelPromise,
        );
    case 'bag7':
    default:
      return (seed) => new Bag7(seed);
  }
}
