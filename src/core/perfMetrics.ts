export type PerfMetricsSink = {
  recordDuration: (name: string, ms: number, nowMs?: number) => void;
  recordCount: (name: string, value: number, nowMs?: number) => void;
};

let sink: PerfMetricsSink | null = null;
const durationTotals = new Map<string, number>();
const countTotals = new Map<string, number>();

export function setPerfMetricsSink(next: PerfMetricsSink | null): void {
  sink = next;
  durationTotals.clear();
  countTotals.clear();
}

export function hasPerfMetricsSink(): boolean {
  return sink != null;
}

export function recordPerfDuration(
  name: string,
  ms: number,
  nowMs = performance.now(),
): void {
  if (!Number.isFinite(ms)) return;
  if (!sink) return;
  const clamped = Math.max(0, ms);
  durationTotals.set(name, (durationTotals.get(name) ?? 0) + clamped);
  sink.recordDuration(name, clamped, nowMs);
}

export function recordPerfCount(
  name: string,
  value: number,
  nowMs = performance.now(),
): void {
  if (!Number.isFinite(value)) return;
  if (!sink) return;
  countTotals.set(name, (countTotals.get(name) ?? 0) + value);
  sink.recordCount(name, value, nowMs);
}

export function readPerfDurationTotal(name: string): number {
  return durationTotals.get(name) ?? 0;
}
