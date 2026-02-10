import type { PieceKind } from './types';

export type PiecePalette = Record<PieceKind, number>;

export const PIECE_COLORS: PiecePalette = {
  I: 0x4dd3ff,
  O: 0xffd84d,
  T: 0xc77dff,
  S: 0x6eea6e,
  Z: 0xff6b6b,
  J: 0x4d7cff,
  L: 0xffa94d,
};

export const PIECE_COLORS_HIGH_CONTRAST: PiecePalette = {
  I: 0x7fe6ff,
  O: 0xfff07a,
  T: 0xe3a6ff,
  S: 0x8bff8b,
  Z: 0xff9a9a,
  J: 0x7aa2ff,
  L: 0xffc07a,
};

export const PIECE_COLORS_COLORBLIND: PiecePalette = {
  I: 0x56b4e9,
  J: 0x0072b2,
  L: 0xe69f00,
  O: 0xf0e442,
  S: 0x009e73,
  T: 0xcc79a7,
  Z: 0xd55e00,
};

export const getPiecePalette = (options: {
  highContrast: boolean;
  colorblindMode: boolean;
}): PiecePalette => {
  if (options.colorblindMode) return PIECE_COLORS_COLORBLIND;
  if (options.highContrast) return PIECE_COLORS_HIGH_CONTRAST;
  return PIECE_COLORS;
};
