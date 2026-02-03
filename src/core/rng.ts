export class XorShift32 {
  private s: number;

  constructor(seed: number) {
    // avoid zero state
    this.s = seed | 0 || 0x12345678;
  }

  nextU32(): number {
    // xorshift32
    let x = this.s | 0;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.s = x | 0;
    return this.s >>> 0;
  }

  nextInt(maxExclusive: number): number {
    return this.nextU32() % maxExclusive;
  }
}

export function shuffleInPlace<T>(arr: T[], rng: XorShift32): void {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = rng.nextInt(i + 1);
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}
