import type { PerfMetricsSink } from '../core/perfMetrics';

type PerfOverlayOptions = {
  root: HTMLElement;
  windowMs?: number;
  targetFps?: number;
  refreshMs?: number;
  initialVisible?: boolean;
};

type Sample = {
  t: number;
  v: number;
};

type Stats = {
  current: number;
  avg: number;
  min: number;
  max: number;
};

type StageSpec = {
  key: string;
  label: string;
};

const STAGES: StageSpec[] = [
  { key: 'runtime.stage.input_ms', label: 'input' },
  { key: 'runtime.stage.ml_ms', label: 'ml' },
  { key: 'runtime.stage.simulation_ms', label: 'simulation' },
  { key: 'runtime.stage.render_ms', label: 'render' },
  { key: 'runtime.stage.ui_ms', label: 'ui' },
  { key: 'runtime.stage.other_ms', label: 'other' },
];

const ML_SUBSTAGES: StageSpec[] = [
  { key: 'runtime.stage.ml_input_encode_ms', label: 'input encode' },
  { key: 'runtime.stage.ml_conv_stack_ms', label: 'conv stack' },
  { key: 'runtime.stage.ml_pool_ms', label: 'pooling' },
  { key: 'runtime.stage.ml_head_ms', label: 'mlp head' },
  { key: 'runtime.stage.ml_post_ms', label: 'post/sample' },
  { key: 'runtime.stage.ml_other_ms', label: 'other' },
];

function pruneSamples(samples: Sample[], cutoff: number): void {
  let removeCount = 0;
  while (removeCount < samples.length && samples[removeCount].t < cutoff) {
    removeCount += 1;
  }
  if (removeCount > 0) {
    samples.splice(0, removeCount);
  }
}

function computeStats(samples: Sample[], cutoff: number): Stats | null {
  if (samples.length === 0) return null;
  pruneSamples(samples, cutoff);
  if (samples.length === 0) return null;

  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let sum = 0;
  for (const sample of samples) {
    if (sample.v < min) min = sample.v;
    if (sample.v > max) max = sample.v;
    sum += sample.v;
  }
  return {
    current: samples[samples.length - 1].v,
    avg: sum / samples.length,
    min,
    max,
  };
}

function formatFrameValue(valueMs: number, budgetMs: number): string {
  return `${(valueMs / budgetMs).toFixed(2).padStart(7, ' ')}f`;
}

function formatFrameLine(
  label: string,
  stats: Stats | null,
  budgetMs: number,
): string {
  if (!stats) {
    return `${label.padEnd(12, ' ')} cur    n/a   avg    n/a   min    n/a   max    n/a`;
  }
  return `${label.padEnd(12, ' ')} cur ${formatFrameValue(
    stats.current,
    budgetMs,
  )}  avg ${formatFrameValue(stats.avg, budgetMs)}  min ${formatFrameValue(
    stats.min,
    budgetMs,
  )}  max ${formatFrameValue(stats.max, budgetMs)}`;
}

export type PerfOverlay = PerfMetricsSink & {
  setVisible: (visible: boolean) => void;
  isVisible: () => boolean;
  destroy: () => void;
};

export function createPerfOverlay(options: PerfOverlayOptions): PerfOverlay {
  const {
    root,
    windowMs = 10_000,
    targetFps = 120,
    refreshMs = 250,
    initialVisible = true,
  } = options;

  const durationSamples = new Map<string, Sample[]>();

  const panel = document.createElement('div');
  Object.assign(panel.style, {
    position: 'absolute',
    left: '8px',
    top: '8px',
    zIndex: '9999',
    maxWidth: '760px',
    background: 'rgba(5,8,12,0.86)',
    color: '#d8e6ff',
    border: '1px solid rgba(110,168,255,0.35)',
    borderRadius: '8px',
    padding: '8px 10px',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    fontSize: '11px',
    lineHeight: '1.35',
    whiteSpace: 'pre',
    pointerEvents: 'none',
  });
  root.appendChild(panel);

  let visible = initialVisible;
  panel.style.display = visible ? 'block' : 'none';

  const pushSample = (key: string, value: number, nowMs: number) => {
    let samples = durationSamples.get(key);
    if (!samples) {
      samples = [];
      durationSamples.set(key, samples);
    }
    samples.push({ t: nowMs, v: value });
    pruneSamples(samples, nowMs - windowMs);
  };

  const render = () => {
    if (!visible) return;
    const nowMs = performance.now();
    const cutoff = nowMs - windowMs;
    const budgetMs = 1000 / Math.max(1, targetFps);

    const tickStats = computeStats(
      durationSamples.get('runtime.tick.total_ms') ?? [],
      cutoff,
    );
    const frameStats = computeStats(
      durationSamples.get('frame.interval_ms') ?? [],
      cutoff,
    );
    const headroomMs = tickStats ? budgetMs - tickStats.current : budgetMs;

    const lines: string[] = [];
    lines.push(
      `PERF OVERLAY  (${Math.round(windowMs / 1000)}s window)  toggle: F8`,
    );
    lines.push(
      `target ${targetFps.toFixed(0)}fps  (1.00f = ${budgetMs.toFixed(2)}ms)`,
    );
    lines.push(formatFrameLine('frame load', tickStats, budgetMs));
    lines.push(formatFrameLine('interval', frameStats, budgetMs));
    lines.push(
      `headroom     cur ${formatFrameValue(headroomMs, budgetMs)} (${headroomMs.toFixed(
        2,
      )}ms)`,
    );
    lines.push('');
    lines.push('stages (sum = frame load)');

    for (const stage of STAGES) {
      const stats = computeStats(durationSamples.get(stage.key) ?? [], cutoff);
      lines.push(formatFrameLine(stage.label, stats, budgetMs));
    }

    lines.push('');
    lines.push('ml substages (sum = ml)');
    for (const stage of ML_SUBSTAGES) {
      const stats = computeStats(durationSamples.get(stage.key) ?? [], cutoff);
      lines.push(formatFrameLine(stage.label, stats, budgetMs));
    }

    panel.textContent = lines.join('\n');
  };

  const timerId = window.setInterval(render, Math.max(50, refreshMs));
  render();

  const onKeyDown = (event: KeyboardEvent) => {
    if (event.code !== 'F8') return;
    visible = !visible;
    panel.style.display = visible ? 'block' : 'none';
    if (visible) render();
  };
  window.addEventListener('keydown', onKeyDown);

  return {
    recordDuration: (name, ms, nowMs = performance.now()) => {
      pushSample(name, ms, nowMs);
    },
    recordCount: () => {},
    setVisible: (next) => {
      visible = next;
      panel.style.display = visible ? 'block' : 'none';
      if (visible) render();
    },
    isVisible: () => visible,
    destroy: () => {
      window.clearInterval(timerId);
      window.removeEventListener('keydown', onKeyDown);
      panel.remove();
    },
  };
}
