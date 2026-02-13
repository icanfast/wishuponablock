import {
  DEFAULT_ARR_MS,
  DEFAULT_DAS_MS,
  DEFAULT_KEY_BINDINGS,
  DEFAULT_SOFT_DROP_MS,
  OUTER_MARGIN,
} from '../../core/constants';
import type { Settings } from '../../core/settings';
import type { SettingsStore } from '../../core/settingsStore';

type MenuPanel =
  | 'main'
  | 'play'
  | 'options'
  | 'about'
  | 'cheese'
  | 'charcuterie'
  | 'tools'
  | 'feedback';

export type LabelingProgressState = {
  buildVersion: string;
  labeledBoards: number | null;
  target: number;
};

export type MenuScreenOptions = {
  settingsStore: SettingsStore;
  showDevTools: boolean;
  version: string;
  charcuterieDefaultSimCount?: number;
  tools: Array<{ id: string; label: string }>;
  labelingProgress?: LabelingProgressState | null;
  onStartPractice: () => void;
  onStartSprint: () => void;
  onStartClassic: () => void;
  onStartCheese: (lines: number) => void;
  onStartCharcuterie: (
    pieces: number,
    options: { simCount: number; seed?: number },
  ) => void;
  onOpenTool: (id: string) => void;
  onSendFeedback: (feedback: string, contact: string | null) => Promise<void>;
};

export type MenuScreen = {
  root: HTMLDivElement;
  show: (panel: MenuPanel) => void;
  showMain: () => void;
  setTools: (tools: Array<{ id: string; label: string }>) => void;
  setLabelingProgress: (progress: LabelingProgressState | null) => void;
  setCharcuterieSpinnerVisible: (visible: boolean) => void;
  syncSettings: (settings: Settings) => void;
};

export function createMenuScreen(options: MenuScreenOptions): MenuScreen {
  const {
    settingsStore,
    showDevTools,
    version,
    charcuterieDefaultSimCount = 10000,
    tools,
    labelingProgress = null,
    onStartPractice,
    onStartSprint,
    onStartClassic,
    onStartCheese,
    onStartCharcuterie,
    onOpenTool,
    onSendFeedback,
  } = options;

  const ensureSpinnerStyle = () => {
    const styleId = 'wab-spin-style';
    if (document.getElementById(styleId)) return;
    const style = document.createElement('style');
    style.id = styleId;
    style.textContent = `
@keyframes wab-spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}
`;
    document.head.appendChild(style);
  };
  ensureSpinnerStyle();

  const ensureNumberInputStyle = () => {
    const styleId = 'wab-number-input-style';
    if (document.getElementById(styleId)) return;
    const style = document.createElement('style');
    style.id = styleId;
    style.textContent = `
input[type=number]::-webkit-outer-spin-button,
input[type=number]::-webkit-inner-spin-button {
  -webkit-appearance: none;
  margin: 0;
}
input[type=number] {
  appearance: textfield;
}
`;
    document.head.appendChild(style);
  };
  ensureNumberInputStyle();

  const root = document.createElement('div');
  Object.assign(root.style, {
    position: 'absolute',
    inset: '0',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: 'rgba(11, 15, 20, 0.7)',
    pointerEvents: 'auto',
  });

  const charcuterieSpinner = document.createElement('div');
  Object.assign(charcuterieSpinner.style, {
    position: 'absolute',
    inset: '0',
    display: 'none',
    alignItems: 'center',
    justifyContent: 'center',
    background: 'rgba(0, 0, 0, 0.35)',
    pointerEvents: 'auto',
    zIndex: '2',
  });
  root.appendChild(charcuterieSpinner);

  const spinnerPanel = document.createElement('div');
  Object.assign(spinnerPanel.style, {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    padding: '10px 12px',
    background: '#0b0f14',
    border: '1px solid #1f2a37',
    borderRadius: '8px',
    color: '#e2e8f0',
    fontFamily: 'system-ui, -apple-system, Segoe UI, sans-serif',
    fontSize: '12px',
  });
  charcuterieSpinner.appendChild(spinnerPanel);

  const spinnerRing = document.createElement('div');
  Object.assign(spinnerRing.style, {
    width: '18px',
    height: '18px',
    border: '2px solid #2c3a4a',
    borderTopColor: '#8fa0b8',
    borderRadius: '50%',
    animation: 'wab-spin 0.9s linear infinite',
  });
  spinnerPanel.appendChild(spinnerRing);

  const spinnerText = document.createElement('div');
  spinnerText.textContent = 'Generating board...';
  spinnerPanel.appendChild(spinnerText);

  const makeMenuPanel = () => {
    const panel = document.createElement('div');
    Object.assign(panel.style, {
      width: '240px',
      padding: '16px',
      background: '#121a24',
      color: '#e2e8f0',
      border: '2px solid #0b0f14',
      borderRadius: '8px',
      display: 'flex',
      flexDirection: 'column',
      gap: '10px',
      textAlign: 'center',
      fontFamily: 'system-ui, -apple-system, Segoe UI, sans-serif',
      fontSize: '14px',
    });
    return panel;
  };

  const makeMenuButton = (labelText: string) => {
    const btn = document.createElement('button');
    btn.textContent = labelText;
    Object.assign(btn.style, {
      background: '#0b0f14',
      color: '#e2e8f0',
      border: '1px solid #1f2a37',
      borderRadius: '6px',
      padding: '10px 12px',
      fontSize: '13px',
      cursor: 'pointer',
    });
    return btn;
  };

  const menuMainPanel = makeMenuPanel();
  const playPanel = makeMenuPanel();
  const optionsPanel = makeMenuPanel();
  const aboutPanel = makeMenuPanel();
  const cheesePanel = makeMenuPanel();
  const charcuteriePanel = makeMenuPanel();
  const toolsPanel = makeMenuPanel();
  const feedbackPanel = makeMenuPanel();
  const butterfingerPanel = makeMenuPanel();
  const playMenuRow = document.createElement('div');

  Object.assign(playPanel.style, { minHeight: '240px', display: 'flex' });
  Object.assign(optionsPanel.style, {
    minHeight: '240px',
    width: '700px',
    display: 'none',
  });
  Object.assign(aboutPanel.style, {
    minHeight: '260px',
    width: '300px',
    display: 'none',
    textAlign: 'left',
  });
  Object.assign(cheesePanel.style, { minHeight: '240px', display: 'none' });
  Object.assign(charcuteriePanel.style, {
    minHeight: '240px',
    display: 'none',
  });
  Object.assign(toolsPanel.style, { minHeight: '240px', display: 'none' });
  Object.assign(feedbackPanel.style, {
    minHeight: '260px',
    width: '320px',
    display: 'none',
    textAlign: 'left',
  });
  Object.assign(butterfingerPanel.style, {
    minHeight: '240px',
    width: '240px',
    display: showDevTools ? 'flex' : 'none',
  });
  Object.assign(playMenuRow.style, {
    display: 'none',
    gap: '16px',
    alignItems: 'flex-start',
  });

  const playButton = makeMenuButton('PLAY');
  const optionsButton = makeMenuButton('OPTIONS');
  const toolsButton = makeMenuButton('TOOLS');
  const aboutButton = makeMenuButton('ABOUT');

  menuMainPanel.appendChild(playButton);
  menuMainPanel.appendChild(optionsButton);
  menuMainPanel.appendChild(toolsButton);
  menuMainPanel.appendChild(aboutButton);

  const optionsTitle = document.createElement('div');
  optionsTitle.textContent = 'OPTIONS';
  Object.assign(optionsTitle.style, {
    color: '#8fa0b8',
    fontSize: '12px',
    letterSpacing: '0.5px',
    marginBottom: '4px',
  });

  const optionsContentRow = document.createElement('div');
  Object.assign(optionsContentRow.style, {
    display: 'flex',
    gap: '16px',
    alignItems: 'flex-start',
  });

  const optionsLeftColumn = document.createElement('div');
  Object.assign(optionsLeftColumn.style, {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
    flex: '1.3',
  });

  const optionsRightColumn = document.createElement('div');
  Object.assign(optionsRightColumn.style, {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
    flex: '1',
  });

  const optionsMiddleColumn = document.createElement('div');
  Object.assign(optionsMiddleColumn.style, {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
    flex: '1',
  });

  const controlsTitle = document.createElement('div');
  controlsTitle.textContent = 'CONTROLS';
  Object.assign(controlsTitle.style, {
    color: '#8fa0b8',
    fontSize: '12px',
    letterSpacing: '0.5px',
    marginTop: '12px',
    marginBottom: '4px',
  });

  const controlsList = document.createElement('div');
  Object.assign(controlsList.style, {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  });

  const audioTitle = document.createElement('div');
  audioTitle.textContent = 'AUDIO';
  Object.assign(audioTitle.style, {
    color: '#8fa0b8',
    fontSize: '12px',
    letterSpacing: '0.5px',
    marginTop: '10px',
    marginBottom: '4px',
  });

  const volumeLabel = document.createElement('div');
  volumeLabel.textContent = 'Master Volume';
  Object.assign(volumeLabel.style, {
    marginBottom: '6px',
    color: '#b6c2d4',
    fontSize: '11px',
  });

  const volumeRow = document.createElement('div');
  Object.assign(volumeRow.style, {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  });

  const volumeValue = document.createElement('input');
  volumeValue.type = 'number';
  volumeValue.min = '0';
  volumeValue.max = '100';
  volumeValue.step = '0.1';
  volumeValue.inputMode = 'decimal';
  Object.assign(volumeValue.style, {
    color: '#e2e8f0',
    background: '#0b0f14',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    fontSize: '12px',
    width: '52px',
    textAlign: 'right',
    padding: '4px 6px',
    MozAppearance: 'textfield',
  });
  volumeValue.addEventListener('wheel', (event) => {
    if (document.activeElement === volumeValue) {
      event.preventDefault();
    }
  });
  volumeValue.addEventListener('keydown', (event) => {
    if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
      event.preventDefault();
    }
  });

  const volumeSlider = document.createElement('input');
  volumeSlider.type = 'range';
  volumeSlider.min = '0';
  volumeSlider.max = '100';
  volumeSlider.step = '1';
  Object.assign(volumeSlider.style, {
    flex: '1',
    accentColor: '#6ea8ff',
  });

  const formatNumber = (value: number): string => {
    const rounded = Math.round(value * 10) / 10;
    return Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1);
  };

  const updateVolumeLabel = (value: number) => {
    volumeValue.value = formatNumber(value * 100);
  };

  volumeSlider.addEventListener('input', () => {
    const value = Math.max(0, Math.min(1, Number(volumeSlider.value) / 100));
    updateVolumeLabel(value);
    settingsStore.apply({ audio: { masterVolume: value } });
  });

  const applyVolumeInput = (commit: boolean) => {
    const raw = Number(volumeValue.value);
    if (!Number.isFinite(raw)) return;
    const clamped = Math.min(100, Math.max(0, raw));
    const value = clamped / 100;
    volumeSlider.value = String(Math.round(clamped));
    if (commit) {
      updateVolumeLabel(value);
    }
    settingsStore.apply({ audio: { masterVolume: value } });
  };

  volumeValue.addEventListener('input', () => applyVolumeInput(false));
  volumeValue.addEventListener('change', () => applyVolumeInput(true));
  volumeValue.addEventListener('blur', () => applyVolumeInput(true));

  volumeRow.appendChild(volumeSlider);
  volumeRow.appendChild(volumeValue);

  const gameplayTitle = document.createElement('div');
  gameplayTitle.textContent = 'GAMEPLAY';
  Object.assign(gameplayTitle.style, {
    color: '#8fa0b8',
    fontSize: '12px',
    letterSpacing: '0.5px',
    marginTop: '10px',
    marginBottom: '4px',
  });

  const makeMsSlider = (
    labelText: string,
    options: { min: number; max: number; step: number },
  ) => {
    const wrapper = document.createElement('div');
    Object.assign(wrapper.style, {
      display: 'flex',
      flexDirection: 'column',
      gap: '6px',
    });

    const label = document.createElement('div');
    label.textContent = labelText;
    Object.assign(label.style, {
      color: '#b6c2d4',
      fontSize: '11px',
      letterSpacing: '0.3px',
    });

    const row = document.createElement('div');
    Object.assign(row.style, {
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
    });

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = String(options.min);
    slider.max = String(options.max);
    slider.step = String(options.step);
    Object.assign(slider.style, {
      flex: '1',
      accentColor: '#6ea8ff',
    });

    const value = document.createElement('input');
    value.type = 'number';
    value.min = String(options.min);
    value.max = String(options.max);
    value.step = String(options.step);
    value.inputMode = 'numeric';
    Object.assign(value.style, {
      color: '#e2e8f0',
      background: '#0b0f14',
      border: '1px solid #1f2a37',
      borderRadius: '4px',
      fontSize: '11px',
      width: '52px',
      textAlign: 'right',
      padding: '4px 6px',
      MozAppearance: 'textfield',
    });
    value.addEventListener('wheel', (event) => {
      if (document.activeElement === value) {
        event.preventDefault();
      }
    });
    value.addEventListener('keydown', (event) => {
      if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
        event.preventDefault();
      }
    });

    row.appendChild(slider);
    row.appendChild(value);
    wrapper.appendChild(label);
    wrapper.appendChild(row);

    return { wrapper, slider, value, options };
  };

  const dasControl = makeMsSlider('DAS (ms)', {
    min: 0,
    max: 300,
    step: 1,
  });
  const arrControl = makeMsSlider('ARR (ms)', {
    min: 0,
    max: 100,
    step: 1,
  });
  const softDropControl = makeMsSlider('Soft Drop (ms, 0 = instant)', {
    min: 0,
    max: 200,
    step: 10,
  });

  const gameplayResetButton = makeMenuButton('RESET DEFAULTS');
  Object.assign(gameplayResetButton.style, {
    marginTop: '4px',
  });

  const clamp = (value: number, min: number, max: number): number =>
    Math.min(max, Math.max(min, value));

  const applyInputPatch = (patch: Partial<Settings['input']>) => {
    const current = settingsStore.get().input;
    settingsStore.apply({
      input: {
        ...current,
        ...patch,
      },
    });
  };

  const applyGamePatch = (patch: Partial<Settings['game']>) => {
    const current = settingsStore.get().game;
    settingsStore.apply({
      game: {
        ...current,
        ...patch,
      },
    });
  };

  const quantizeToStep = (value: number, step: number): number => {
    if (!Number.isFinite(step) || step <= 0) {
      return Math.round(value);
    }
    return Math.round(value / step) * step;
  };

  const updateMsControl = (
    control: {
      slider: HTMLInputElement;
      value: HTMLInputElement;
      options: { min: number; max: number; step: number };
    },
    nextValue: number,
  ) => {
    const clamped = clamp(
      quantizeToStep(nextValue, control.options.step),
      control.options.min,
      control.options.max,
    );
    control.slider.value = String(clamped);
    control.value.value = String(clamped);
  };

  dasControl.slider.addEventListener('input', () => {
    const value = Number(dasControl.slider.value);
    updateMsControl(dasControl, value);
    applyInputPatch({ dasMs: value });
  });

  arrControl.slider.addEventListener('input', () => {
    const value = Number(arrControl.slider.value);
    updateMsControl(arrControl, value);
    applyInputPatch({ arrMs: value });
  });

  softDropControl.slider.addEventListener('input', () => {
    const value = Number(softDropControl.slider.value);
    updateMsControl(softDropControl, value);
    applyGamePatch({ softDropMs: value });
  });

  gameplayResetButton.addEventListener('click', () => {
    applyInputPatch({
      dasMs: DEFAULT_DAS_MS,
      arrMs: DEFAULT_ARR_MS,
    });
    applyGamePatch({
      softDropMs: DEFAULT_SOFT_DROP_MS,
    });
  });

  const applyMsInput = (
    control: {
      slider: HTMLInputElement;
      value: HTMLInputElement;
      options: { min: number; max: number; step: number };
    },
    patch: (value: number) => void,
    commit: boolean,
  ) => {
    const raw = Number(control.value.value);
    if (!Number.isFinite(raw)) return;
    const clamped = clamp(
      quantizeToStep(raw, control.options.step),
      control.options.min,
      control.options.max,
    );
    control.slider.value = String(clamped);
    if (commit) {
      control.value.value = String(clamped);
    }
    patch(clamped);
  };

  dasControl.value.addEventListener('input', () => {
    applyMsInput(
      dasControl,
      (value) => applyInputPatch({ dasMs: value }),
      false,
    );
  });
  dasControl.value.addEventListener('change', () => {
    applyMsInput(
      dasControl,
      (value) => applyInputPatch({ dasMs: value }),
      true,
    );
  });
  dasControl.value.addEventListener('blur', () => {
    applyMsInput(
      dasControl,
      (value) => applyInputPatch({ dasMs: value }),
      true,
    );
  });

  arrControl.value.addEventListener('input', () => {
    applyMsInput(
      arrControl,
      (value) => applyInputPatch({ arrMs: value }),
      false,
    );
  });
  arrControl.value.addEventListener('change', () => {
    applyMsInput(
      arrControl,
      (value) => applyInputPatch({ arrMs: value }),
      true,
    );
  });
  arrControl.value.addEventListener('blur', () => {
    applyMsInput(
      arrControl,
      (value) => applyInputPatch({ arrMs: value }),
      true,
    );
  });

  softDropControl.value.addEventListener('input', () => {
    applyMsInput(
      softDropControl,
      (value) => applyGamePatch({ softDropMs: value }),
      false,
    );
  });
  softDropControl.value.addEventListener('change', () => {
    applyMsInput(
      softDropControl,
      (value) => applyGamePatch({ softDropMs: value }),
      true,
    );
  });
  softDropControl.value.addEventListener('blur', () => {
    applyMsInput(
      softDropControl,
      (value) => applyGamePatch({ softDropMs: value }),
      true,
    );
  });

  const graphicsTitle = document.createElement('div');
  graphicsTitle.textContent = 'GRAPHICS';
  Object.assign(graphicsTitle.style, {
    color: '#8fa0b8',
    fontSize: '12px',
    letterSpacing: '0.5px',
    marginTop: '10px',
    marginBottom: '4px',
  });

  const gridlineLabel = document.createElement('div');
  gridlineLabel.textContent = 'Gridlines Opacity';
  Object.assign(gridlineLabel.style, {
    marginBottom: '6px',
    color: '#b6c2d4',
    fontSize: '11px',
  });

  const gridlineRow = document.createElement('div');
  Object.assign(gridlineRow.style, {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  });

  const gridlineValue = document.createElement('input');
  gridlineValue.type = 'number';
  gridlineValue.min = '0';
  gridlineValue.max = '100';
  gridlineValue.step = '1';
  gridlineValue.inputMode = 'numeric';
  Object.assign(gridlineValue.style, {
    color: '#e2e8f0',
    background: '#0b0f14',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    fontSize: '12px',
    width: '52px',
    textAlign: 'right',
    padding: '4px 6px',
    MozAppearance: 'textfield',
  });
  gridlineValue.addEventListener('wheel', (event) => {
    if (document.activeElement === gridlineValue) {
      event.preventDefault();
    }
  });
  gridlineValue.addEventListener('keydown', (event) => {
    if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
      event.preventDefault();
    }
  });

  const gridlineSlider = document.createElement('input');
  gridlineSlider.type = 'range';
  gridlineSlider.min = '0';
  gridlineSlider.max = '100';
  gridlineSlider.step = '1';
  Object.assign(gridlineSlider.style, {
    flex: '1',
    accentColor: '#6ea8ff',
  });

  const updateGridlineLabel = (value: number) => {
    gridlineValue.value = String(Math.round(value * 100));
  };

  gridlineSlider.addEventListener('input', () => {
    const value = Math.max(0, Math.min(1, Number(gridlineSlider.value) / 100));
    updateGridlineLabel(value);
    const current = settingsStore.get().graphics;
    settingsStore.apply({
      graphics: { ...current, gridlineOpacity: value },
    });
  });

  const applyGridlineInput = (commit: boolean) => {
    const raw = Number(gridlineValue.value);
    if (!Number.isFinite(raw)) return;
    const clamped = Math.max(0, Math.min(100, raw));
    const value = clamped / 100;
    gridlineSlider.value = String(Math.round(clamped));
    if (commit) {
      updateGridlineLabel(value);
    }
    const current = settingsStore.get().graphics;
    settingsStore.apply({
      graphics: { ...current, gridlineOpacity: value },
    });
  };

  gridlineValue.addEventListener('input', () => applyGridlineInput(false));
  gridlineValue.addEventListener('change', () => applyGridlineInput(true));
  gridlineValue.addEventListener('blur', () => applyGridlineInput(true));

  gridlineRow.appendChild(gridlineSlider);
  gridlineRow.appendChild(gridlineValue);

  const ghostOpacityLabel = document.createElement('div');
  ghostOpacityLabel.textContent = 'Ghost Piece Opacity';
  Object.assign(ghostOpacityLabel.style, {
    marginTop: '6px',
    marginBottom: '6px',
    color: '#b6c2d4',
    fontSize: '11px',
  });

  const ghostOpacityRow = document.createElement('div');
  Object.assign(ghostOpacityRow.style, {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  });

  const ghostOpacityValue = document.createElement('input');
  ghostOpacityValue.type = 'number';
  ghostOpacityValue.min = '0';
  ghostOpacityValue.max = '100';
  ghostOpacityValue.step = '1';
  ghostOpacityValue.inputMode = 'numeric';
  Object.assign(ghostOpacityValue.style, {
    color: '#e2e8f0',
    background: '#0b0f14',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    fontSize: '12px',
    width: '52px',
    textAlign: 'right',
    padding: '4px 6px',
    MozAppearance: 'textfield',
  });
  ghostOpacityValue.addEventListener('wheel', (event) => {
    if (document.activeElement === ghostOpacityValue) {
      event.preventDefault();
    }
  });
  ghostOpacityValue.addEventListener('keydown', (event) => {
    if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
      event.preventDefault();
    }
  });

  const ghostOpacitySlider = document.createElement('input');
  ghostOpacitySlider.type = 'range';
  ghostOpacitySlider.min = '0';
  ghostOpacitySlider.max = '100';
  ghostOpacitySlider.step = '1';
  Object.assign(ghostOpacitySlider.style, {
    flex: '1',
    accentColor: '#6ea8ff',
  });

  const updateGhostOpacityLabel = (value: number) => {
    ghostOpacityValue.value = String(Math.round(value * 100));
  };

  ghostOpacitySlider.addEventListener('input', () => {
    const value = Math.max(
      0,
      Math.min(1, Number(ghostOpacitySlider.value) / 100),
    );
    updateGhostOpacityLabel(value);
    const current = settingsStore.get().graphics;
    settingsStore.apply({
      graphics: { ...current, ghostOpacity: value },
    });
  });

  const applyGhostOpacityInput = (commit: boolean) => {
    const raw = Number(ghostOpacityValue.value);
    if (!Number.isFinite(raw)) return;
    const clamped = Math.max(0, Math.min(100, raw));
    const value = clamped / 100;
    ghostOpacitySlider.value = String(Math.round(clamped));
    if (commit) {
      updateGhostOpacityLabel(value);
    }
    const current = settingsStore.get().graphics;
    settingsStore.apply({
      graphics: { ...current, ghostOpacity: value },
    });
  };

  ghostOpacityValue.addEventListener('input', () =>
    applyGhostOpacityInput(false),
  );
  ghostOpacityValue.addEventListener('change', () =>
    applyGhostOpacityInput(true),
  );
  ghostOpacityValue.addEventListener('blur', () =>
    applyGhostOpacityInput(true),
  );

  ghostOpacityRow.appendChild(ghostOpacitySlider);
  ghostOpacityRow.appendChild(ghostOpacityValue);

  const highContrastRow = document.createElement('label');
  Object.assign(highContrastRow.style, {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: '8px',
    color: '#b6c2d4',
    marginTop: '6px',
  });

  const highContrastLabel = document.createElement('span');
  highContrastLabel.textContent = 'High Contrast';

  const highContrastToggle = document.createElement('input');
  highContrastToggle.type = 'checkbox';
  Object.assign(highContrastToggle.style, {
    accentColor: '#6ea8ff',
  });

  highContrastToggle.addEventListener('change', () => {
    const current = settingsStore.get().graphics;
    settingsStore.apply({
      graphics: { ...current, highContrast: highContrastToggle.checked },
    });
  });

  highContrastRow.appendChild(highContrastLabel);
  highContrastRow.appendChild(highContrastToggle);

  const colorblindRow = document.createElement('label');
  Object.assign(colorblindRow.style, {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: '8px',
    color: '#b6c2d4',
    marginTop: '6px',
  });

  const colorblindLabel = document.createElement('span');
  colorblindLabel.textContent = 'Colorblind Mode';

  const colorblindToggle = document.createElement('input');
  colorblindToggle.type = 'checkbox';
  Object.assign(colorblindToggle.style, {
    accentColor: '#6ea8ff',
  });

  colorblindToggle.addEventListener('change', () => {
    const current = settingsStore.get().graphics;
    settingsStore.apply({
      graphics: { ...current, colorblindMode: colorblindToggle.checked },
    });
  });

  colorblindRow.appendChild(colorblindLabel);
  colorblindRow.appendChild(colorblindToggle);

  const dataTitle = document.createElement('div');
  dataTitle.textContent = 'DATA';
  Object.assign(dataTitle.style, {
    color: '#8fa0b8',
    fontSize: '12px',
    letterSpacing: '0.5px',
    marginTop: '10px',
    marginBottom: '4px',
  });

  const shareRow = document.createElement('label');
  Object.assign(shareRow.style, {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: '8px',
    color: '#b6c2d4',
  });

  const shareLabel = document.createElement('span');
  shareLabel.textContent = 'Share snapshots';

  const shareToggle = document.createElement('input');
  shareToggle.type = 'checkbox';
  Object.assign(shareToggle.style, {
    accentColor: '#6ea8ff',
  });

  shareToggle.addEventListener('change', () => {
    settingsStore.apply({ privacy: { shareSnapshots: shareToggle.checked } });
  });

  shareRow.appendChild(shareLabel);
  shareRow.appendChild(shareToggle);

  const controlsResetButton = makeMenuButton('RESET DEFAULTS');
  Object.assign(controlsResetButton.style, {
    marginTop: '6px',
  });

  const optionsBackButton = makeMenuButton('BACK');
  Object.assign(optionsBackButton.style, {
    marginTop: 'auto',
    width: '240px',
    alignSelf: 'center',
  });

  optionsPanel.appendChild(optionsTitle);
  optionsPanel.appendChild(optionsContentRow);
  optionsContentRow.appendChild(optionsLeftColumn);
  optionsContentRow.appendChild(optionsMiddleColumn);
  optionsContentRow.appendChild(optionsRightColumn);

  optionsLeftColumn.appendChild(controlsTitle);
  optionsLeftColumn.appendChild(controlsList);
  optionsLeftColumn.appendChild(controlsResetButton);

  const volumeWrapper = document.createElement('div');
  Object.assign(volumeWrapper.style, {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
  });
  volumeWrapper.appendChild(volumeLabel);
  volumeWrapper.appendChild(volumeRow);

  optionsMiddleColumn.appendChild(gameplayTitle);
  optionsMiddleColumn.appendChild(dasControl.wrapper);
  optionsMiddleColumn.appendChild(arrControl.wrapper);
  optionsMiddleColumn.appendChild(softDropControl.wrapper);
  optionsMiddleColumn.appendChild(gameplayResetButton);
  optionsMiddleColumn.appendChild(dataTitle);
  optionsMiddleColumn.appendChild(shareRow);

  optionsRightColumn.appendChild(audioTitle);
  optionsRightColumn.appendChild(volumeWrapper);
  optionsRightColumn.appendChild(graphicsTitle);
  optionsRightColumn.appendChild(gridlineLabel);
  optionsRightColumn.appendChild(gridlineRow);
  optionsRightColumn.appendChild(ghostOpacityLabel);
  optionsRightColumn.appendChild(ghostOpacityRow);
  optionsRightColumn.appendChild(highContrastRow);
  optionsRightColumn.appendChild(colorblindRow);

  optionsPanel.appendChild(optionsBackButton);

  const aboutTitle = document.createElement('div');
  aboutTitle.textContent = 'ABOUT';
  Object.assign(aboutTitle.style, {
    color: '#8fa0b8',
    fontSize: '12px',
    letterSpacing: '0.5px',
    marginBottom: '4px',
  });

  const aboutBody = document.createElement('div');
  aboutBody.textContent =
    'Wish Upon a Block is a lightweight low-latency guideline tetromino game. ' +
    'It is meant to recreate the feeling of "Tetris effect" when every next piece is "just right". ' +
    'Play and try out the ML powered "Wish Upon a Block" piece generator.';
  Object.assign(aboutBody.style, {
    color: '#b6c2d4',
    fontSize: '12px',
    lineHeight: '1.4',
    textAlign: 'center',
  });

  const creditsTitle = document.createElement('div');
  creditsTitle.textContent = 'CREDITS';
  Object.assign(creditsTitle.style, {
    color: '#8fa0b8',
    fontSize: '12px',
    letterSpacing: '0.5px',
    marginTop: '12px',
    marginBottom: '4px',
  });

  const creditsList = document.createElement('div');
  Object.assign(creditsList.style, {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
    color: '#b6c2d4',
    fontSize: '12px',
  });

  const creditsName = document.createElement('div');
  creditsName.textContent = 'Максим Никитин (Maksim Nikitin)';

  const makeCreditsLink = (href: string, text: string) => {
    const link = document.createElement('a');
    link.href = href;
    link.textContent = text;
    link.target = '_blank';
    link.rel = 'noreferrer';
    Object.assign(link.style, {
      color: '#b6c2d4',
      textDecoration: 'none',
      wordBreak: 'break-all',
    });
    return link;
  };

  creditsList.appendChild(creditsName);
  creditsList.appendChild(
    makeCreditsLink(
      'https://github.com/icanfast/wishuponablock',
      'https://github.com/icanfast/wishuponablock',
    ),
  );
  creditsList.appendChild(
    makeCreditsLink(
      'mailto:nikitin.maxim.94@gmail.com',
      'nikitin.maxim.94@gmail.com',
    ),
  );
  creditsList.appendChild(
    makeCreditsLink('https://t.me/icanfast', 't.me/icanfast'),
  );

  const aboutBackButton = makeMenuButton('BACK');
  Object.assign(aboutBackButton.style, {
    marginTop: 'auto',
  });

  aboutPanel.appendChild(aboutTitle);
  aboutPanel.appendChild(aboutBody);
  aboutPanel.appendChild(creditsTitle);
  aboutPanel.appendChild(creditsList);
  aboutPanel.appendChild(aboutBackButton);

  type KeyBindingKey = keyof Settings['input']['bindings'];
  const keybindButtons = new Map<KeyBindingKey, HTMLButtonElement>();
  let rebindingKey: KeyBindingKey | null = null;
  let rebindingButton: HTMLButtonElement | null = null;

  const formatKeyLabel = (code: string): string => {
    if (!code) return '';
    if (code === 'Space') return 'Space';
    if (code.startsWith('Key')) return code.slice(3);
    if (code.startsWith('Digit')) return code.slice(5);
    if (code.startsWith('Arrow')) return code.slice(5);
    if (code.startsWith('Numpad')) return `Num ${code.slice(6)}`;
    return code.replace(/([a-z])([A-Z])/g, '$1 $2');
  };

  const updateKeybindButtons = (bindings: Settings['input']['bindings']) => {
    for (const [key, button] of keybindButtons) {
      if (rebindingKey === key) continue;
      const label = formatKeyLabel(bindings[key]);
      button.textContent = label || '\u00a0';
    }
  };

  const cancelRebind = () => {
    if (rebindingKey && rebindingButton) {
      const current = settingsStore.get().input.bindings;
      rebindingButton.textContent = formatKeyLabel(current[rebindingKey]);
      rebindingButton.blur();
    }
    rebindingKey = null;
    rebindingButton = null;
  };

  const startRebind = (key: KeyBindingKey, button: HTMLButtonElement) => {
    cancelRebind();
    rebindingKey = key;
    rebindingButton = button;
    button.textContent = 'Press a key';
  };

  const reservedCodes = new Set([
    'ControlLeft',
    'ControlRight',
    'AltLeft',
    'AltRight',
    'MetaLeft',
    'MetaRight',
  ]);

  window.addEventListener(
    'keydown',
    (event) => {
      if (!rebindingKey || !rebindingButton) return;
      event.preventDefault();
      event.stopPropagation();
      if (event.code === 'Escape') {
        cancelRebind();
        return;
      }
      if (reservedCodes.has(event.code)) return;
      const currentInput = settingsStore.get().input;
      const nextBindings = { ...currentInput.bindings };
      for (const [action, code] of Object.entries(nextBindings)) {
        if (action === rebindingKey) continue;
        if (code === event.code) {
          nextBindings[action as KeyBindingKey] = '';
        }
      }
      updateKeybindButtons(nextBindings);
      settingsStore.apply({
        input: {
          ...currentInput,
          bindings: { ...nextBindings, [rebindingKey]: event.code },
        },
      });
      cancelRebind();
    },
    true,
  );

  const keybindConfig: Array<{ key: KeyBindingKey; label: string }> = [
    { key: 'moveLeft', label: 'Move Left' },
    { key: 'moveRight', label: 'Move Right' },
    { key: 'softDrop', label: 'Soft Drop' },
    { key: 'hardDrop', label: 'Hard Drop' },
    { key: 'rotateCW', label: 'Rotate CW' },
    { key: 'rotateCCW', label: 'Rotate CCW' },
    { key: 'rotate180', label: 'Rotate 180' },
    { key: 'hold', label: 'Hold' },
    { key: 'restart', label: 'Restart' },
  ];

  for (const item of keybindConfig) {
    const row = document.createElement('div');
    Object.assign(row.style, {
      display: 'grid',
      gridTemplateColumns: '1fr 110px',
      alignItems: 'center',
      columnGap: '8px',
    });

    const label = document.createElement('div');
    label.textContent = item.label;
    Object.assign(label.style, {
      color: '#b6c2d4',
      fontSize: '12px',
      flex: '1',
    });

    const button = document.createElement('button');
    Object.assign(button.style, {
      background: '#0b0f14',
      color: '#e2e8f0',
      border: '1px solid #1f2a37',
      borderRadius: '4px',
      padding: '4px 8px',
      fontSize: '12px',
      lineHeight: '1.2',
      minHeight: '24px',
      width: '110px',
      cursor: 'pointer',
      textAlign: 'center',
    });
    button.addEventListener('click', () => startRebind(item.key, button));

    row.appendChild(label);
    row.appendChild(button);
    controlsList.appendChild(row);
    keybindButtons.set(item.key, button);
  }

  updateKeybindButtons(settingsStore.get().input.bindings);

  controlsResetButton.addEventListener('click', () => {
    const currentInput = settingsStore.get().input;
    settingsStore.apply({
      input: {
        ...currentInput,
        bindings: { ...DEFAULT_KEY_BINDINGS },
      },
    });
  });

  const dataNotice = document.createElement('div');
  dataNotice.textContent =
    'This game collects anonymized board snapshots to train the piece generator.';
  Object.assign(dataNotice.style, {
    maxWidth: '320px',
    color: '#8fa0b8',
    fontSize: '12px',
    lineHeight: '1.4',
    textAlign: 'center',
    fontFamily: 'system-ui, -apple-system, Segoe UI, sans-serif',
    marginTop: '6px',
  });

  const menuMainWrapper = document.createElement('div');
  Object.assign(menuMainWrapper.style, {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '12px',
  });
  menuMainWrapper.appendChild(menuMainPanel);
  menuMainWrapper.appendChild(dataNotice);

  const menuTitle = document.createElement('div');
  menuTitle.textContent = 'WISH UPON A BLOCK';
  Object.assign(menuTitle.style, {
    position: 'absolute',
    left: '50%',
    top: '50%',
    transform: 'translate(-50%, -240px)',
    color: '#e2e8f0',
    fontSize: '30px',
    letterSpacing: '2px',
    textAlign: 'center',
    fontFamily: 'system-ui, -apple-system, Segoe UI, sans-serif',
    pointerEvents: 'none',
  });

  const playTitle = document.createElement('div');
  playTitle.textContent = 'PLAY';
  Object.assign(playTitle.style, {
    color: '#8fa0b8',
    fontSize: '12px',
    letterSpacing: '0.5px',
    marginBottom: '4px',
  });

  const practiceButton = makeMenuButton('PRACTICE');
  const sprintButton = makeMenuButton('SPRINT');
  const classicButton = makeMenuButton('CLASSIC');
  const cheeseModeButton = makeMenuButton('CHEESE');
  const charcuterieModeButton = makeMenuButton('CHARCUTERIE');
  const playBackButton = makeMenuButton('BACK');
  Object.assign(playBackButton.style, {
    marginTop: 'auto',
  });

  playPanel.appendChild(playTitle);
  playPanel.appendChild(practiceButton);
  playPanel.appendChild(sprintButton);
  playPanel.appendChild(classicButton);
  playPanel.appendChild(cheeseModeButton);
  playPanel.appendChild(charcuterieModeButton);
  playPanel.appendChild(playBackButton);

  let updateButterfingerUI: (cfg: Settings['butterfinger']) => void = () => {};
  if (showDevTools) {
    const butterfingerTitle = document.createElement('div');
    butterfingerTitle.textContent = 'BUTTERFINGER';
    Object.assign(butterfingerTitle.style, {
      color: '#8fa0b8',
      fontSize: '12px',
      letterSpacing: '0.5px',
      marginBottom: '4px',
    });
    butterfingerPanel.appendChild(butterfingerTitle);

    const butterfingerToggleRow = document.createElement('label');
    Object.assign(butterfingerToggleRow.style, {
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
      fontSize: '12px',
      color: '#e2e8f0',
      cursor: 'pointer',
    });
    const butterfingerToggle = document.createElement('input');
    butterfingerToggle.type = 'checkbox';
    butterfingerToggle.style.cursor = 'pointer';
    const butterfingerToggleText = document.createElement('span');
    butterfingerToggleText.textContent = 'Enable';
    butterfingerToggleRow.appendChild(butterfingerToggle);
    butterfingerToggleRow.appendChild(butterfingerToggleText);
    butterfingerPanel.appendChild(butterfingerToggleRow);

    const makeButterfingerSlider = (
      labelText: string,
      options: { max?: number; step?: number } = {},
    ) => {
      const row = document.createElement('div');
      Object.assign(row.style, {
        display: 'flex',
        flexDirection: 'column',
        gap: '6px',
        textAlign: 'left',
        marginTop: '8px',
      });
      const label = document.createElement('div');
      label.textContent = labelText;
      Object.assign(label.style, {
        color: '#b6c2d4',
        fontSize: '11px',
        letterSpacing: '0.3px',
      });
      const value = document.createElement('span');
      Object.assign(value.style, {
        color: '#8fa0b8',
        fontSize: '11px',
        marginLeft: '6px',
      });
      const labelRow = document.createElement('div');
      Object.assign(labelRow.style, {
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
      });
      labelRow.appendChild(label);
      labelRow.appendChild(value);

      const slider = document.createElement('input');
      slider.type = 'range';
      slider.min = '0';
      slider.max = String(options.max ?? 10);
      slider.step = String(options.step ?? 0.1);
      Object.assign(slider.style, {
        width: '100%',
        accentColor: '#6ea8ff',
      });

      row.appendChild(labelRow);
      row.appendChild(slider);
      butterfingerPanel.appendChild(row);
      return { slider, value };
    };

    const missSlider = makeButterfingerSlider('Miss Rate');
    const wrongDirSlider = makeButterfingerSlider('Wrong Direction');
    const extraTapSlider = makeButterfingerSlider('Extra Tap');
    const lockNudgeSlider = makeButterfingerSlider('Lock Nudge');
    const gravityDropSlider = makeButterfingerSlider('Gravity Drop');
    const lockRotateSlider = makeButterfingerSlider('Lock Rotate', {
      max: 100,
      step: 1,
    });

    const clampRate = (value: number): number =>
      Math.min(1, Math.max(0, value));
    const formatPercent = (value: number): string => {
      const rounded = Math.round(value * 10) / 10;
      return Number.isInteger(rounded)
        ? `${rounded}%`
        : `${rounded.toFixed(1)}%`;
    };
    const updateButterfingerControl = (
      control: { slider: HTMLInputElement; value: HTMLSpanElement },
      rate: number,
    ) => {
      const sliderMax = Number(control.slider.max) || 100;
      const percent = Math.min(sliderMax, clampRate(rate) * 100);
      control.slider.value = String(percent);
      control.value.textContent = formatPercent(percent);
    };

    updateButterfingerUI = (cfg: Settings['butterfinger']) => {
      butterfingerToggle.checked = cfg.enabled;
      updateButterfingerControl(missSlider, cfg.missRate);
      updateButterfingerControl(wrongDirSlider, cfg.wrongDirRate);
      updateButterfingerControl(extraTapSlider, cfg.extraTapRate);
      updateButterfingerControl(lockNudgeSlider, cfg.lockNudgeRate);
      updateButterfingerControl(gravityDropSlider, cfg.gravityDropRate);
      updateButterfingerControl(lockRotateSlider, cfg.lockRotateRate);
    };

    const readButterfingerRate = (control: {
      slider: HTMLInputElement;
      value: HTMLSpanElement;
    }): number => clampRate(Number(control.slider.value) / 100);

    const applyButterfinger = (patch: Partial<Settings['butterfinger']>) => {
      const current = settingsStore.get().butterfinger;
      settingsStore.apply({
        butterfinger: { ...current, ...patch },
      });
    };

    butterfingerToggle.addEventListener('change', () => {
      applyButterfinger({ enabled: butterfingerToggle.checked });
    });

    missSlider.slider.addEventListener('input', () => {
      const rate = readButterfingerRate(missSlider);
      missSlider.value.textContent = formatPercent(rate * 100);
      applyButterfinger({ missRate: rate });
    });

    wrongDirSlider.slider.addEventListener('input', () => {
      const rate = readButterfingerRate(wrongDirSlider);
      wrongDirSlider.value.textContent = formatPercent(rate * 100);
      applyButterfinger({ wrongDirRate: rate });
    });

    extraTapSlider.slider.addEventListener('input', () => {
      const rate = readButterfingerRate(extraTapSlider);
      extraTapSlider.value.textContent = formatPercent(rate * 100);
      applyButterfinger({ extraTapRate: rate });
    });

    lockNudgeSlider.slider.addEventListener('input', () => {
      const rate = readButterfingerRate(lockNudgeSlider);
      lockNudgeSlider.value.textContent = formatPercent(rate * 100);
      applyButterfinger({ lockNudgeRate: rate });
    });

    gravityDropSlider.slider.addEventListener('input', () => {
      const rate = readButterfingerRate(gravityDropSlider);
      gravityDropSlider.value.textContent = formatPercent(rate * 100);
      applyButterfinger({ gravityDropRate: rate });
    });

    lockRotateSlider.slider.addEventListener('input', () => {
      const rate = readButterfingerRate(lockRotateSlider);
      lockRotateSlider.value.textContent = formatPercent(rate * 100);
      applyButterfinger({ lockRotateRate: rate });
    });
  }

  const cheeseTitle = document.createElement('div');
  cheeseTitle.textContent = 'CHEESE';
  Object.assign(cheeseTitle.style, {
    color: '#8fa0b8',
    fontSize: '12px',
    letterSpacing: '0.5px',
    marginBottom: '4px',
  });

  const cheese4Button = makeMenuButton('4 LINES');
  const cheese8Button = makeMenuButton('8 LINES');
  const cheese12Button = makeMenuButton('12 LINES');
  const cheeseBackButton = makeMenuButton('BACK');
  Object.assign(cheeseBackButton.style, { marginTop: 'auto' });

  cheesePanel.appendChild(cheeseTitle);
  cheesePanel.appendChild(cheese4Button);
  cheesePanel.appendChild(cheese8Button);
  cheesePanel.appendChild(cheese12Button);
  cheesePanel.appendChild(cheeseBackButton);

  const charcuterieTitle = document.createElement('div');
  charcuterieTitle.textContent = 'CHARCUTERIE';
  Object.assign(charcuterieTitle.style, {
    color: '#8fa0b8',
    fontSize: '12px',
    letterSpacing: '0.5px',
    marginBottom: '4px',
  });

  const charcuterie8Button = makeMenuButton('8 PIECES');
  const charcuterie14Button = makeMenuButton('14 PIECES');
  const charcuterie20Button = makeMenuButton('20 PIECES');
  const charcuterieSimInput = document.createElement('input');
  charcuterieSimInput.type = 'number';
  charcuterieSimInput.min = '1';
  charcuterieSimInput.step = '1';
  charcuterieSimInput.value = String(charcuterieDefaultSimCount);
  Object.assign(charcuterieSimInput.style, {
    width: '100%',
    boxSizing: 'border-box',
    background: '#0b0f14',
    color: '#e2e8f0',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    padding: '6px 8px',
    fontSize: '12px',
  });

  const charcuterieSeedInput = document.createElement('input');
  charcuterieSeedInput.type = 'text';
  charcuterieSeedInput.placeholder = 'Random';
  Object.assign(charcuterieSeedInput.style, {
    width: '100%',
    boxSizing: 'border-box',
    background: '#0b0f14',
    color: '#e2e8f0',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    padding: '6px 8px',
    fontSize: '12px',
  });

  const makeMenuField = (labelText: string, input: HTMLInputElement) => {
    const field = document.createElement('div');
    Object.assign(field.style, {
      display: 'flex',
      flexDirection: 'column',
      gap: '6px',
      textAlign: 'left',
      marginTop: '4px',
    });
    const label = document.createElement('div');
    label.textContent = labelText;
    Object.assign(label.style, {
      color: '#b6c2d4',
      fontSize: '11px',
      letterSpacing: '0.3px',
    });
    field.appendChild(label);
    field.appendChild(input);
    return field;
  };

  const charcuterieSimField = makeMenuField('SIMULATIONS', charcuterieSimInput);
  const charcuterieSeedField = makeMenuField('SEED', charcuterieSeedInput);
  const charcuterieBackButton = makeMenuButton('BACK');
  Object.assign(charcuterieBackButton.style, { marginTop: 'auto' });

  charcuteriePanel.appendChild(charcuterieTitle);
  charcuteriePanel.appendChild(charcuterie8Button);
  charcuteriePanel.appendChild(charcuterie14Button);
  charcuteriePanel.appendChild(charcuterie20Button);
  if (showDevTools) {
    charcuteriePanel.appendChild(charcuterieSimField);
    charcuteriePanel.appendChild(charcuterieSeedField);
  }
  charcuteriePanel.appendChild(charcuterieBackButton);

  const toolsTitle = document.createElement('div');
  toolsTitle.textContent = 'TOOLS';
  Object.assign(toolsTitle.style, {
    color: '#8fa0b8',
    fontSize: '12px',
    letterSpacing: '0.5px',
    marginBottom: '4px',
  });

  const toolsBackButton = makeMenuButton('BACK');
  Object.assign(toolsBackButton.style, { marginTop: 'auto' });

  toolsPanel.appendChild(toolsTitle);
  const toolsProgressCard = document.createElement('div');
  Object.assign(toolsProgressCard.style, {
    background: '#0b0f14',
    border: '1px solid #1f2a37',
    borderRadius: '6px',
    padding: '10px',
    textAlign: 'left',
  });
  const toolsProgressLabel = document.createElement('div');
  Object.assign(toolsProgressLabel.style, {
    color: '#8fa0b8',
    fontSize: '11px',
    letterSpacing: '0.4px',
    marginBottom: '4px',
  });
  const toolsProgressValue = document.createElement('div');
  Object.assign(toolsProgressValue.style, {
    color: '#e2e8f0',
    fontSize: '12px',
    marginBottom: '8px',
  });
  const toolsProgressTrack = document.createElement('div');
  Object.assign(toolsProgressTrack.style, {
    width: '100%',
    height: '8px',
    background: '#121a24',
    border: '1px solid #1f2a37',
    borderRadius: '999px',
    overflow: 'hidden',
  });
  const toolsProgressFill = document.createElement('div');
  Object.assign(toolsProgressFill.style, {
    height: '100%',
    width: '0%',
    background: '#6ea8ff',
    transition: 'width 180ms ease-out',
  });
  toolsProgressTrack.appendChild(toolsProgressFill);
  toolsProgressCard.appendChild(toolsProgressLabel);
  toolsProgressCard.appendChild(toolsProgressValue);
  toolsProgressCard.appendChild(toolsProgressTrack);
  toolsPanel.appendChild(toolsProgressCard);

  const toolsList = document.createElement('div');
  Object.assign(toolsList.style, {
    display: 'flex',
    flexDirection: 'column',
    gap: '10px',
  });
  const toolsEmptyLabel = document.createElement('div');
  toolsEmptyLabel.textContent = 'No tools available.';
  Object.assign(toolsEmptyLabel.style, {
    color: '#8fa0b8',
    fontSize: '12px',
    padding: '6px 0',
  });
  toolsPanel.appendChild(toolsList);
  toolsPanel.appendChild(toolsBackButton);

  const setTools = (nextTools: Array<{ id: string; label: string }>) => {
    toolsList.innerHTML = '';
    if (nextTools.length === 0) {
      toolsList.appendChild(toolsEmptyLabel);
      return;
    }
    for (const tool of nextTools) {
      const btn = makeMenuButton(tool.label);
      btn.addEventListener('click', () => onOpenTool(tool.id));
      toolsList.appendChild(btn);
    }
  };

  const setLabelingProgress = (progress: LabelingProgressState | null) => {
    const target = Math.max(1, Math.trunc(progress?.target ?? 1000));
    const buildVersion =
      progress?.buildVersion && progress.buildVersion.trim()
        ? progress.buildVersion.trim()
        : 'unknown';
    if (
      progress == null ||
      progress.labeledBoards == null ||
      !Number.isFinite(progress.labeledBoards)
    ) {
      toolsProgressLabel.textContent = `LABELING PROGRESS · ${buildVersion}`;
      toolsProgressValue.textContent = `0 / ${target.toLocaleString()} (offline)`;
      toolsProgressFill.style.width = '0%';
      toolsProgressFill.style.background = '#6ea8ff';
      return;
    }
    const labeledBoards = Math.max(0, Math.trunc(progress.labeledBoards));
    const ratio = Math.min(1, labeledBoards / target);
    const percent = Math.round(ratio * 100);
    toolsProgressLabel.textContent = `LABELING PROGRESS · ${buildVersion}`;
    toolsProgressValue.textContent = `${labeledBoards.toLocaleString()} / ${target.toLocaleString()} (${percent}%)`;
    toolsProgressFill.style.width = `${ratio * 100}%`;
    toolsProgressFill.style.background = ratio >= 1 ? '#8fd19e' : '#6ea8ff';
  };

  setTools(tools);
  setLabelingProgress(labelingProgress);

  const feedbackTitle = document.createElement('div');
  feedbackTitle.textContent = 'FEEDBACK';
  Object.assign(feedbackTitle.style, {
    color: '#8fa0b8',
    fontSize: '12px',
    letterSpacing: '0.5px',
    marginBottom: '6px',
  });

  const feedbackBody = document.createElement('textarea');
  feedbackBody.placeholder = 'Your feedback and suggestions...';
  Object.assign(feedbackBody.style, {
    width: '100%',
    minHeight: '120px',
    resize: 'vertical',
    boxSizing: 'border-box',
    background: '#0b0f14',
    color: '#e2e8f0',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    padding: '8px',
    fontSize: '12px',
    fontFamily: 'system-ui, -apple-system, Segoe UI, sans-serif',
  });

  const feedbackContact = document.createElement('input');
  feedbackContact.type = 'text';
  feedbackContact.placeholder = 'Your contact (optional)';
  Object.assign(feedbackContact.style, {
    width: '100%',
    boxSizing: 'border-box',
    background: '#0b0f14',
    color: '#e2e8f0',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    padding: '8px',
    fontSize: '12px',
  });

  const feedbackButtons = document.createElement('div');
  Object.assign(feedbackButtons.style, {
    display: 'flex',
    justifyContent: 'space-between',
    gap: '8px',
    marginTop: '8px',
  });

  const feedbackBackButton = makeMenuButton('BACK');
  const feedbackSendButton = makeMenuButton('SEND');
  Object.assign(feedbackBackButton.style, { flex: '1' });
  Object.assign(feedbackSendButton.style, { flex: '1' });

  const updateFeedbackSendState = () => {
    const canSend = feedbackBody.value.trim().length > 0;
    feedbackSendButton.disabled = !canSend;
    feedbackSendButton.style.opacity = canSend ? '1' : '0.55';
    feedbackSendButton.style.cursor = canSend ? 'pointer' : 'default';
  };
  updateFeedbackSendState();

  feedbackBody.addEventListener('input', () => {
    updateFeedbackSendState();
  });

  feedbackButtons.appendChild(feedbackBackButton);
  feedbackButtons.appendChild(feedbackSendButton);

  const feedbackStatus = document.createElement('div');
  Object.assign(feedbackStatus.style, {
    marginTop: '6px',
    fontSize: '12px',
    color: '#8fa0b8',
    textAlign: 'center',
    minHeight: '16px',
  });

  feedbackPanel.appendChild(feedbackTitle);
  feedbackPanel.appendChild(feedbackBody);
  feedbackPanel.appendChild(feedbackContact);
  feedbackPanel.appendChild(feedbackButtons);
  feedbackPanel.appendChild(feedbackStatus);

  const menuLayer = document.createElement('div');
  Object.assign(menuLayer.style, {
    position: 'absolute',
    inset: '0',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    pointerEvents: 'auto',
  });
  menuLayer.appendChild(menuMainWrapper);
  menuLayer.appendChild(menuTitle);
  playMenuRow.appendChild(playPanel);
  if (showDevTools) {
    playMenuRow.appendChild(butterfingerPanel);
  }
  menuLayer.appendChild(playMenuRow);
  menuLayer.appendChild(optionsPanel);
  menuLayer.appendChild(aboutPanel);
  menuLayer.appendChild(cheesePanel);
  menuLayer.appendChild(charcuteriePanel);
  menuLayer.appendChild(toolsPanel);
  menuLayer.appendChild(feedbackPanel);
  root.appendChild(menuLayer);

  const feedbackMenuButton = makeMenuButton('LEAVE FEEDBACK');
  Object.assign(feedbackMenuButton.style, {
    position: 'absolute',
    left: '50%',
    bottom: `${OUTER_MARGIN + 84}px`,
    transform: 'translateX(-50%)',
    width: '200px',
    borderColor: '#bda56a',
    boxShadow: '0 0 0 1px rgba(189, 165, 106, 0.3)',
    pointerEvents: 'auto',
  });
  root.appendChild(feedbackMenuButton);

  const footer = document.createElement('div');
  Object.assign(footer.style, {
    position: 'absolute',
    left: '0',
    right: '0',
    bottom: `${OUTER_MARGIN}px`,
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    flexDirection: 'column',
    gap: '6px',
    pointerEvents: 'auto',
  });

  const githubLink = document.createElement('a');
  githubLink.href = 'https://github.com/icanfast/wishuponablock';
  githubLink.target = '_blank';
  githubLink.rel = 'noreferrer';
  Object.assign(githubLink.style, {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '6px',
    textDecoration: 'none',
    color: '#b6c2d4',
    opacity: '0.85',
  });

  const githubIcon = document.createElement('img');
  githubIcon.src = `${import.meta.env.BASE_URL}assets/GitHub_Invertocat_White_Clearspace.svg`;
  githubIcon.alt = 'GitHub';
  Object.assign(githubIcon.style, {
    width: '36px',
    height: '36px',
    display: 'block',
  });

  githubLink.appendChild(githubIcon);
  footer.appendChild(githubLink);

  const versionLabel = document.createElement('div');
  versionLabel.textContent = `v${version}`;
  Object.assign(versionLabel.style, {
    color: '#8fa0b8',
    fontSize: '12px',
    fontFamily: 'system-ui, -apple-system, Segoe UI, sans-serif',
    pointerEvents: 'none',
  });
  footer.appendChild(versionLabel);
  root.appendChild(footer);

  const readCharcuterieSimCount = (): number => {
    const raw = Number(charcuterieSimInput.value);
    if (!Number.isFinite(raw)) return charcuterieDefaultSimCount;
    return Math.max(1, Math.trunc(raw));
  };

  const readCharcuterieSeed = (): number | undefined => {
    const raw = charcuterieSeedInput.value.trim();
    if (!raw) return undefined;
    const value = Number(raw);
    if (!Number.isFinite(value)) return undefined;
    return Math.trunc(value);
  };

  let activePanel: MenuPanel = 'main';

  const show = (panel: MenuPanel) => {
    activePanel = panel;
    menuMainWrapper.style.display = panel === 'main' ? 'flex' : 'none';
    playMenuRow.style.display = panel === 'play' ? 'flex' : 'none';
    optionsPanel.style.display = panel === 'options' ? 'flex' : 'none';
    aboutPanel.style.display = panel === 'about' ? 'flex' : 'none';
    cheesePanel.style.display = panel === 'cheese' ? 'flex' : 'none';
    charcuteriePanel.style.display = panel === 'charcuterie' ? 'flex' : 'none';
    toolsPanel.style.display = panel === 'tools' ? 'flex' : 'none';
    feedbackPanel.style.display = panel === 'feedback' ? 'flex' : 'none';
    feedbackMenuButton.style.display = panel === 'main' ? 'block' : 'none';
    menuTitle.style.display = panel === 'options' ? 'none' : 'block';
  };

  const showMain = () => show('main');

  playButton.addEventListener('click', () => show('play'));
  optionsButton.addEventListener('click', () => show('options'));
  toolsButton.addEventListener('click', () => show('tools'));
  aboutButton.addEventListener('click', () => show('about'));
  feedbackMenuButton.addEventListener('click', () => show('feedback'));

  practiceButton.addEventListener('click', () => onStartPractice());
  sprintButton.addEventListener('click', () => onStartSprint());
  classicButton.addEventListener('click', () => onStartClassic());
  cheeseModeButton.addEventListener('click', () => show('cheese'));
  charcuterieModeButton.addEventListener('click', () => show('charcuterie'));

  cheese4Button.addEventListener('click', () => onStartCheese(4));
  cheese8Button.addEventListener('click', () => onStartCheese(8));
  cheese12Button.addEventListener('click', () => onStartCheese(12));

  charcuterie8Button.addEventListener('click', () =>
    onStartCharcuterie(8, {
      simCount: readCharcuterieSimCount(),
      ...(readCharcuterieSeed() !== undefined
        ? { seed: readCharcuterieSeed() }
        : {}),
    }),
  );
  charcuterie14Button.addEventListener('click', () =>
    onStartCharcuterie(14, {
      simCount: readCharcuterieSimCount(),
      ...(readCharcuterieSeed() !== undefined
        ? { seed: readCharcuterieSeed() }
        : {}),
    }),
  );
  charcuterie20Button.addEventListener('click', () =>
    onStartCharcuterie(20, {
      simCount: readCharcuterieSimCount(),
      ...(readCharcuterieSeed() !== undefined
        ? { seed: readCharcuterieSeed() }
        : {}),
    }),
  );

  playBackButton.addEventListener('click', showMain);
  optionsBackButton.addEventListener('click', showMain);
  aboutBackButton.addEventListener('click', showMain);
  cheeseBackButton.addEventListener('click', () => show('play'));
  charcuterieBackButton.addEventListener('click', () => show('play'));
  toolsBackButton.addEventListener('click', showMain);
  feedbackBackButton.addEventListener('click', showMain);

  window.addEventListener('keydown', (event) => {
    if (event.code !== 'Escape') return;
    if (root.style.display === 'none') return;
    if (rebindingKey) return;
    if (activePanel === 'main') return;
    event.preventDefault();
    event.stopPropagation();
    if (activePanel === 'cheese' || activePanel === 'charcuterie') {
      show('play');
      return;
    }
    showMain();
  });

  feedbackSendButton.addEventListener('click', async () => {
    const feedback = feedbackBody.value.trim();
    if (!feedback) {
      updateFeedbackSendState();
      return;
    }
    feedbackSendButton.disabled = true;
    feedbackSendButton.textContent = 'SENDING...';
    feedbackStatus.textContent = '';
    try {
      await onSendFeedback(feedback, feedbackContact.value.trim() || null);
      feedbackBody.value = '';
      feedbackContact.value = '';
      updateFeedbackSendState();
      feedbackStatus.textContent = 'Thank you for your feedback!';
    } catch {
      updateFeedbackSendState();
      feedbackStatus.textContent = 'Could not send feedback.';
    } finally {
      feedbackSendButton.textContent = 'SEND';
    }
  });

  const syncSettings = (settings: Settings) => {
    const nextVolume = Math.round(settings.audio.masterVolume * 100);
    if (Number(volumeSlider.value) !== nextVolume) {
      volumeSlider.value = String(nextVolume);
    }
    updateVolumeLabel(settings.audio.masterVolume);
    updateMsControl(dasControl, settings.input.dasMs);
    updateMsControl(arrControl, settings.input.arrMs);
    updateMsControl(softDropControl, settings.game.softDropMs);
    const nextGridline = Math.round(settings.graphics.gridlineOpacity * 100);
    if (Number(gridlineSlider.value) !== nextGridline) {
      gridlineSlider.value = String(nextGridline);
    }
    updateGridlineLabel(settings.graphics.gridlineOpacity);
    const nextGhostOpacity = Math.round(settings.graphics.ghostOpacity * 100);
    if (Number(ghostOpacitySlider.value) !== nextGhostOpacity) {
      ghostOpacitySlider.value = String(nextGhostOpacity);
    }
    updateGhostOpacityLabel(settings.graphics.ghostOpacity);
    highContrastToggle.checked = settings.graphics.highContrast;
    colorblindToggle.checked = settings.graphics.colorblindMode;
    shareToggle.checked = settings.privacy.shareSnapshots;
    updateKeybindButtons(settings.input.bindings);
    updateButterfingerUI(settings.butterfinger);
  };

  syncSettings(settingsStore.get());
  showMain();

  return {
    root,
    show,
    showMain,
    setTools,
    setLabelingProgress,
    setCharcuterieSpinnerVisible: (visible) => {
      charcuterieSpinner.style.display = visible ? 'flex' : 'none';
    },
    syncSettings,
  };
}
