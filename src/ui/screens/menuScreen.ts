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
  | 'feedback'
  | 'account'
  | 'my_models'
  | 'admin'
  | 'bot_lab';

export type MenuAuthUser = {
  id: string;
  username: string;
  email: string | null;
  emailVerifiedAtMs: number | null;
  isAdmin: boolean;
};

export type MenuAuthState = {
  loading: boolean;
  authenticated: boolean;
  user: MenuAuthUser | null;
};

export type MenuAuthStatusTone = 'neutral' | 'success' | 'error';
export type MenuMlBackendPreference =
  | 'native'
  | 'tfjs_auto'
  | 'tfjs_webgl'
  | 'tfjs_cpu';
export type MenuMlParityResult = {
  ok: boolean;
  message: string;
};
export type MenuLocalTrainingStats = {
  currentModeId: string;
  currentModeAxesSamples: number;
  currentModeSamples: number;
  totalSamples: number;
  lastSampleAtMs: number | null;
};
export type MenuLocalTrainingResult = {
  ok: boolean;
  message: string;
  samplesUsed: number;
  finalLoss: number | null;
};
export type MenuLocalTrainingBackendPreference = 'auto' | 'webgl' | 'cpu';
export type MenuLocalTrainingRequest = {
  sampleLimit?: number;
  epochs?: number;
  learningRate?: number;
  l2?: number;
  backendPreference?: MenuLocalTrainingBackendPreference;
};
export type MenuLocalTrainingPreset = {
  pipelineId: string;
  modeId: string;
  minSamples: number;
  trainDefaults: {
    epochs: number;
    learningRate: number;
    l2: number;
    sampleLimit: number;
    backendPreference: MenuLocalTrainingBackendPreference;
  };
  evalGate: {
    holdoutRatio: number;
    minHoldoutSamples: number;
    minTrainSamples: number;
    minObjectiveGain: number;
  };
};

export type MenuGlobalModelSummary = {
  id: string;
  mode: string;
  arch: string | null;
  rewardProfileId: string | null;
  queuePolicyId: string | null;
  pipelineId: string | null;
  label: string | null;
  isDefault: boolean;
  sizeBytes: number | null;
  updatedAtMs: number | null;
};

export type MenuModelAxes = {
  arch: string;
  rewardProfileId: string;
  queuePolicyId: string;
};

export type MenuAdminGlobalBaselineSummary = {
  id: string;
  mode: string;
  arch: string | null;
  rewardProfileId: string | null;
  queuePolicyId: string | null;
  pipelineId: string | null;
  label: string | null;
  isDefault: boolean;
  sizeBytes: number | null;
  updatedAtMs: number | null;
  retiredAtMs: number | null;
};

export type MenuAdminRecordingSummary = {
  id: string;
  userId: string;
  mode: string;
  buildVersion: string;
  startedAtMs: number;
  durationMs: number;
  samples: number;
};

export type MenuAdminRecordingsPage = {
  recordings: MenuAdminRecordingSummary[];
  page: {
    limit: number;
    nextCursor: string | null;
    returned: number;
  };
};

export type MenuAdminRecordingsQuery = {
  mode?: string;
  build?: string;
  limit?: number;
  cursor?: string | null;
};

export type MenuAdminManifestQuery = {
  mode?: string;
  build?: string;
  limit?: number;
  cursor?: string | null;
  minSamples?: number;
  actorType?: 'human' | 'bot' | '';
};

export type MenuAdminRecordingPreview = {
  id: string;
  sessionId: string;
  modeId: string;
  buildVersion: string;
  samples: number;
  durationMs: number;
  startedAtMs: number;
  endedAtMs: number;
  avgReward: number | null;
  meanDeliberationMs: number | null;
  rewardPolicy: string | null;
  rewardPolicyId: string | null;
  rewardProfileId: string | null;
  queuePolicyId: string | null;
  rewardKind: string | null;
  rewardGamma: number | null;
  pipelineId: string | null;
  pipelineMode: string | null;
  modelArchId: string | null;
  modelArch: string | null;
  outcome: string | null;
};

export type MenuBotPolicySummary = {
  id: string;
  modeId: string;
  archId: string;
  queuePolicyId: string;
  pipelineId: string;
  pieceSourceProfile: string;
  version: number;
  isPinned: boolean;
  createdAtMs: number;
};

export type MenuBotPoliciesPage = {
  policies: MenuBotPolicySummary[];
  page: {
    returned: number;
    nextCursor: string | null;
  };
};

export type LabelingProgressState = {
  buildVersion: string;
  labeledBoards: number | null;
  target: number;
};

export type MenuScreenOptions = {
  settingsStore: SettingsStore;
  showExperimentalGameplayControls: boolean;
  showLegacyDataTools: boolean;
  version: string;
  charcuterieDefaultSimCount?: number;
  tools: Array<{ id: string; label: string }>;
  labelingProgress?: LabelingProgressState | null;
  authState: MenuAuthState;
  authInitialResetToken?: string | null;
  authInitialStatus?: { message: string; tone?: MenuAuthStatusTone } | null;
  mlBackendPreference: MenuMlBackendPreference;
  getMlRuntimeSummary: () => string;
  onMlBackendPreferenceChange: (next: MenuMlBackendPreference) => void;
  onMlRunParityCheck: (
    preference: MenuMlBackendPreference,
  ) => Promise<MenuMlParityResult>;
  getLocalTrainingStats: () => MenuLocalTrainingStats;
  getLocalTrainingPreset: () => MenuLocalTrainingPreset;
  onRunLocalBiasTraining: (
    request?: MenuLocalTrainingRequest,
  ) => Promise<MenuLocalTrainingResult>;
  getTrajectoryUploadSummary: () => string;
  onUploadLatestTrajectory: () => Promise<string>;
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
  onAuthRefresh: () => Promise<void>;
  onAuthStartOAuth: (provider: 'google' | 'discord') => void;
  onAuthLogout: () => Promise<void>;
  onAuthSendVerifyEmail: () => Promise<{ alreadyVerified: boolean }>;
  onAuthEmailSignup: (payload: {
    email: string;
    password: string;
    username?: string;
  }) => Promise<void>;
  onAuthEmailLogin: (payload: {
    email: string;
    password: string;
  }) => Promise<void>;
  onAuthPasswordForgot: (email: string) => Promise<void>;
  onAuthPasswordReset: (payload: {
    token: string;
    password: string;
  }) => Promise<void>;
  getModelAxes: () => MenuModelAxes;
  onModelAxesChange: (next: MenuModelAxes) => Promise<string>;
  modelAxesOptions: {
    arch: string[];
    rewardProfile: string[];
    queuePolicy: string[];
  };
  onAuthListGlobalModels: () => Promise<MenuGlobalModelSummary[]>;
  onAuthResetCurrentModelToGlobal: (
    globalModelId: string | null,
  ) => Promise<string>;
  onAuthLoadCurrentModel: () => Promise<string>;
  onAuthSaveCurrentModel: () => Promise<string>;
  onAdminListRecordings: (
    query: MenuAdminRecordingsQuery,
  ) => Promise<MenuAdminRecordingsPage>;
  onAdminLoadRecording: (id: string) => Promise<MenuAdminRecordingPreview>;
  onAdminPublishCurrentModelBaseline: (payload?: {
    label?: string;
    setDefault?: boolean;
  }) => Promise<string>;
  onAdminListGlobalBaselines: () => Promise<MenuAdminGlobalBaselineSummary[]>;
  onAdminSetGlobalBaselineDefault: (id: string) => Promise<string>;
  onAdminRetireGlobalBaseline: (id: string) => Promise<string>;
  onAdminPrepareManifest: (query: MenuAdminManifestQuery) => Promise<string>;
  onAdminTrainGlobalOneShot: () => Promise<string>;
  onAdminPublishGlobalCandidate: () => Promise<string>;
  onAdminTrainBotPolicyOneShot: (options: {
    episodes?: number;
    maxPiecesPerEpisode?: number;
  }) => Promise<string>;
  onAdminGenerateBotRecordings: (options: {
    sessions?: number;
    maxPiecesPerEpisode?: number;
  }) => Promise<string>;
  onAdminRunCapabilityBenchmark: (options?: {
    episodes?: number;
    maxPiecesPerEpisode?: number;
  }) => Promise<string>;
  onAdminApplyBenchmarkSuggestedArch: () => Promise<string>;
  onBotLabFetchCurrentPolicy: () => Promise<string>;
  onBotLabListPolicies: (options?: {
    limit?: number;
    cursor?: string | null;
  }) => Promise<MenuBotPoliciesPage>;
  onBotLabLoadPolicyById: (id: string) => Promise<string>;
  onBotLabPublishPolicy: (options?: {
    pin?: boolean;
    setCurrent?: boolean;
    pieceSourceProfile?: 'bag7' | 'active_generator';
  }) => Promise<string>;
  onBotLabSelectCurrentPolicy: (id: string) => Promise<string>;
  onBotLabPinPolicy: (id: string) => Promise<string>;
  onBotLabUnpinPolicy: (id: string) => Promise<string>;
  onBotLabTrainPolicyOneShot: (options: {
    episodes?: number;
    maxPiecesPerEpisode?: number;
    seed?: number;
    pieceSourceProfile?: 'bag7' | 'active_generator';
  }) => Promise<string>;
  onBotLabRunHeadlessValidate: (options?: {
    maxPieces?: number;
    seed?: number;
    pieceSourceProfile?: 'bag7' | 'active_generator';
  }) => Promise<string>;
  onBotLabStartGuiInspect: (options?: {
    apmInput?: number;
    seed?: number;
    pieceSourceProfile?: 'bag7' | 'active_generator';
    pieces?: number;
  }) => Promise<string>;
  onBotLabStopGuiInspect: () => Promise<string> | string;
  onBotLabGenerateRecordings: (options: {
    sessions?: number;
    maxPiecesPerEpisode?: number;
    pieceSourceProfile?: 'bag7' | 'active_generator';
  }) => Promise<string>;
  onBotLabRunBenchmark: (options?: {
    episodes?: number;
    maxPiecesPerEpisode?: number;
    pieceSourceProfile?: 'bag7' | 'active_generator';
  }) => Promise<string>;
};

export type MenuScreen = {
  root: HTMLDivElement;
  show: (panel: MenuPanel) => void;
  showMain: () => void;
  setTools: (tools: Array<{ id: string; label: string }>) => void;
  setLabelingProgress: (progress: LabelingProgressState | null) => void;
  setAuthState: (state: MenuAuthState) => void;
  setCharcuterieSpinnerVisible: (visible: boolean) => void;
  syncSettings: (settings: Settings) => void;
};

export function createMenuScreen(options: MenuScreenOptions): MenuScreen {
  const {
    settingsStore,
    showExperimentalGameplayControls,
    showLegacyDataTools,
    version,
    charcuterieDefaultSimCount = 10000,
    tools,
    labelingProgress = null,
    authState,
    authInitialResetToken = null,
    authInitialStatus = null,
    mlBackendPreference,
    getMlRuntimeSummary,
    onMlBackendPreferenceChange,
    onMlRunParityCheck,
    getLocalTrainingStats,
    getLocalTrainingPreset,
    onRunLocalBiasTraining,
    getTrajectoryUploadSummary,
    onUploadLatestTrajectory,
    onStartPractice,
    onStartSprint,
    onStartClassic,
    onStartCheese,
    onStartCharcuterie,
    onOpenTool,
    onSendFeedback,
    onAuthRefresh,
    onAuthStartOAuth,
    onAuthLogout,
    onAuthSendVerifyEmail,
    onAuthEmailSignup,
    onAuthEmailLogin,
    onAuthPasswordForgot,
    onAuthPasswordReset,
    getModelAxes,
    onModelAxesChange,
    modelAxesOptions,
    onAuthListGlobalModels,
    onAuthResetCurrentModelToGlobal,
    onAuthLoadCurrentModel,
    onAuthSaveCurrentModel,
    onAdminListRecordings,
    onAdminLoadRecording,
    onAdminPublishCurrentModelBaseline,
    onAdminListGlobalBaselines,
    onAdminSetGlobalBaselineDefault,
    onAdminRetireGlobalBaseline,
    onAdminPrepareManifest,
    onAdminTrainGlobalOneShot,
    onAdminPublishGlobalCandidate,
    onAdminApplyBenchmarkSuggestedArch,
    onBotLabFetchCurrentPolicy,
    onBotLabListPolicies,
    onBotLabLoadPolicyById,
    onBotLabPublishPolicy,
    onBotLabSelectCurrentPolicy,
    onBotLabPinPolicy,
    onBotLabUnpinPolicy,
    onBotLabTrainPolicyOneShot,
    onBotLabRunHeadlessValidate,
    onBotLabStartGuiInspect,
    onBotLabStopGuiInspect,
    onBotLabGenerateRecordings,
    onBotLabRunBenchmark,
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
  const accountPanel = makeMenuPanel();
  const myModelsPanel = makeMenuPanel();
  const adminPanel = makeMenuPanel();
  const botLabPanel = makeMenuPanel();
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
  Object.assign(accountPanel.style, {
    minHeight: '260px',
    width: '320px',
    display: 'none',
    textAlign: 'left',
  });
  Object.assign(myModelsPanel.style, {
    minHeight: '320px',
    width: '320px',
    display: 'none',
    textAlign: 'left',
    maxHeight: '520px',
    overflowY: 'auto',
  });
  Object.assign(adminPanel.style, {
    minHeight: '260px',
    width: '360px',
    display: 'none',
    textAlign: 'left',
    maxHeight: '520px',
    overflowY: 'auto',
  });
  Object.assign(botLabPanel.style, {
    minHeight: '260px',
    width: '360px',
    display: 'none',
    textAlign: 'left',
    maxHeight: '520px',
    overflowY: 'auto',
  });
  Object.assign(butterfingerPanel.style, {
    minHeight: '240px',
    width: '240px',
    display: showExperimentalGameplayControls ? 'flex' : 'none',
  });
  Object.assign(playMenuRow.style, {
    display: 'none',
    gap: '16px',
    alignItems: 'flex-start',
  });

  const playButton = makeMenuButton('PLAY');
  const optionsButton = makeMenuButton('OPTIONS');
  const toolsButton = makeMenuButton('TOOLS');
  if (!showLegacyDataTools) {
    toolsButton.style.display = 'none';
  }
  const accountButton = makeMenuButton('ACCOUNT');
  const myModelsButton = makeMenuButton('MY MODELS');
  const adminButton = makeMenuButton('ADMIN');
  const botLabButton = makeMenuButton('BOT LAB');
  adminButton.style.display = 'none';
  botLabButton.style.display = 'none';
  const aboutButton = makeMenuButton('ABOUT');

  menuMainPanel.appendChild(playButton);
  menuMainPanel.appendChild(optionsButton);
  menuMainPanel.appendChild(toolsButton);
  menuMainPanel.appendChild(accountButton);
  menuMainPanel.appendChild(myModelsButton);
  menuMainPanel.appendChild(adminButton);
  menuMainPanel.appendChild(botLabButton);
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

  const mlTitle = document.createElement('div');
  mlTitle.textContent = 'ML';
  Object.assign(mlTitle.style, {
    color: '#8fa0b8',
    fontSize: '12px',
    letterSpacing: '0.5px',
    marginTop: '10px',
    marginBottom: '4px',
  });

  const mlRuntimeSummary = document.createElement('div');
  Object.assign(mlRuntimeSummary.style, {
    color: '#b6c2d4',
    fontSize: '11px',
    lineHeight: '1.35',
    marginBottom: '6px',
    textAlign: 'left',
    whiteSpace: 'pre-wrap',
  });

  const mlBackendRow = document.createElement('div');
  Object.assign(mlBackendRow.style, {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
    marginBottom: '6px',
  });

  const mlBackendLabel = document.createElement('div');
  mlBackendLabel.textContent = 'Backend';
  Object.assign(mlBackendLabel.style, {
    color: '#b6c2d4',
    fontSize: '11px',
  });

  const mlBackendSelect = document.createElement('select');
  Object.assign(mlBackendSelect.style, {
    width: '100%',
    boxSizing: 'border-box',
    background: '#0b0f14',
    color: '#e2e8f0',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    padding: '6px 8px',
    fontSize: '12px',
  });
  const mlBackendOptions: Array<{
    value: MenuMlBackendPreference;
    label: string;
  }> = [
    { value: 'tfjs_auto', label: 'TFJS (auto: WebGL->CPU)' },
    { value: 'tfjs_webgl', label: 'TFJS (WebGL)' },
    { value: 'tfjs_cpu', label: 'TFJS (CPU)' },
    { value: 'native', label: 'Native (legacy)' },
  ];
  for (const option of mlBackendOptions) {
    const el = document.createElement('option');
    el.value = option.value;
    el.textContent = option.label;
    mlBackendSelect.appendChild(el);
  }
  mlBackendSelect.value = mlBackendPreference;

  mlBackendRow.appendChild(mlBackendLabel);
  mlBackendRow.appendChild(mlBackendSelect);

  const mlActions = document.createElement('div');
  Object.assign(mlActions.style, {
    display: 'flex',
    gap: '8px',
  });

  const mlApplyButton = makeMenuButton('APPLY + RELOAD');
  const mlParityButton = makeMenuButton('RUN PARITY CHECK');
  Object.assign(mlApplyButton.style, { flex: '1', padding: '8px 10px' });
  Object.assign(mlParityButton.style, { flex: '1', padding: '8px 10px' });
  mlActions.appendChild(mlApplyButton);
  mlActions.appendChild(mlParityButton);

  const mlStatus = document.createElement('div');
  Object.assign(mlStatus.style, {
    marginTop: '4px',
    color: '#8fa0b8',
    fontSize: '11px',
    lineHeight: '1.35',
    textAlign: 'left',
    minHeight: '28px',
    whiteSpace: 'pre-wrap',
  });
  const setMlStatus = (
    message: string,
    tone: MenuAuthStatusTone = 'neutral',
  ) => {
    mlStatus.textContent = message;
    mlStatus.style.color =
      tone === 'success' ? '#8fd19e' : tone === 'error' ? '#f28b82' : '#8fa0b8';
  };
  const syncMlRuntimeSummary = () => {
    mlRuntimeSummary.textContent = getMlRuntimeSummary();
  };
  syncMlRuntimeSummary();
  let mlActionPending = false;
  const syncMlControls = () => {
    const disabled = mlActionPending;
    mlBackendSelect.disabled = disabled;
    mlApplyButton.disabled = disabled;
    mlParityButton.disabled = disabled;
    mlApplyButton.style.opacity = disabled ? '0.65' : '1';
    mlParityButton.style.opacity = disabled ? '0.65' : '1';
    mlApplyButton.style.cursor = disabled ? 'default' : 'pointer';
    mlParityButton.style.cursor = disabled ? 'default' : 'pointer';
  };
  syncMlControls();

  mlApplyButton.addEventListener('click', () => {
    if (mlActionPending) return;
    const next = mlBackendSelect.value as MenuMlBackendPreference;
    if (
      next !== 'native' &&
      next !== 'tfjs_auto' &&
      next !== 'tfjs_webgl' &&
      next !== 'tfjs_cpu'
    ) {
      setMlStatus('Invalid backend preference selected.', 'error');
      return;
    }
    setMlStatus('Saving preference and reloading...');
    onMlBackendPreferenceChange(next);
  });

  mlParityButton.addEventListener('click', async () => {
    if (mlActionPending) return;
    mlActionPending = true;
    syncMlControls();
    setMlStatus('Running parity check...');
    try {
      const preference = mlBackendSelect.value as MenuMlBackendPreference;
      const result = await onMlRunParityCheck(preference);
      setMlStatus(result.message, result.ok ? 'success' : 'error');
      syncMlRuntimeSummary();
    } catch {
      setMlStatus('Parity check failed.', 'error');
    } finally {
      mlActionPending = false;
      syncMlControls();
    }
  });

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
  optionsMiddleColumn.appendChild(mlTitle);
  optionsMiddleColumn.appendChild(mlRuntimeSummary);
  optionsMiddleColumn.appendChild(mlBackendRow);
  optionsMiddleColumn.appendChild(mlActions);
  optionsMiddleColumn.appendChild(mlStatus);
  if (showLegacyDataTools) {
    optionsMiddleColumn.appendChild(dataTitle);
    optionsMiddleColumn.appendChild(shareRow);
  }

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
  if (showExperimentalGameplayControls) {
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
  if (showExperimentalGameplayControls) {
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

  const accountTitle = document.createElement('div');
  accountTitle.textContent = 'ACCOUNT';
  Object.assign(accountTitle.style, {
    color: '#8fa0b8',
    fontSize: '12px',
    letterSpacing: '0.5px',
    marginBottom: '6px',
    textAlign: 'center',
  });

  const accountSummary = document.createElement('div');
  Object.assign(accountSummary.style, {
    color: '#b6c2d4',
    fontSize: '12px',
    lineHeight: '1.45',
    minHeight: '66px',
    background: '#0b0f14',
    border: '1px solid #1f2a37',
    borderRadius: '6px',
    padding: '8px',
    whiteSpace: 'pre-wrap',
  });

  const accountActionStatus = document.createElement('div');
  Object.assign(accountActionStatus.style, {
    marginTop: '4px',
    fontSize: '12px',
    color: '#8fa0b8',
    minHeight: '16px',
    textAlign: 'center',
  });

  const makeAccountInput = (
    placeholder: string,
    type = 'text',
  ): HTMLInputElement => {
    const input = document.createElement('input');
    input.type = type;
    input.placeholder = placeholder;
    Object.assign(input.style, {
      width: '100%',
      boxSizing: 'border-box',
      background: '#0b0f14',
      color: '#e2e8f0',
      border: '1px solid #1f2a37',
      borderRadius: '4px',
      padding: '8px',
      fontSize: '12px',
    });
    return input;
  };

  const makeSectionLabel = (text: string) => {
    const label = document.createElement('div');
    label.textContent = text;
    Object.assign(label.style, {
      color: '#8fa0b8',
      fontSize: '11px',
      letterSpacing: '0.3px',
      marginTop: '4px',
    });
    return label;
  };

  const makeTextAction = (text: string) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = text;
    Object.assign(button.style, {
      background: 'transparent',
      border: 'none',
      color: '#8fa0b8',
      fontSize: '12px',
      padding: '0',
      textDecoration: 'underline',
      textAlign: 'left',
      cursor: 'pointer',
    });
    return button;
  };

  const accountRefreshButton = makeMenuButton('REFRESH SESSION');
  const accountGoogleButton = makeMenuButton('SIGN IN WITH GOOGLE');
  const accountDiscordButton = makeMenuButton('SIGN IN WITH DISCORD');
  const accountVerifyButton = makeMenuButton('SEND VERIFICATION EMAIL');
  const accountLogoutButton = makeMenuButton('LOG OUT');
  const accountEmailContinueButton = makeMenuButton('CONTINUE WITH EMAIL');
  const accountEmailSubmitButton = makeMenuButton('LOG IN');
  const accountResetSubmitButton = makeMenuButton('SET NEW PASSWORD');
  const accountForgotLink = makeTextAction('Forgot password?');
  const accountSwitchModeLink = makeTextAction('Need an account? Sign up');
  const accountBackToEmailLink = makeTextAction('Use different email');
  const accountResetBackLink = makeTextAction('Back to sign in');
  const accountBackButton = makeMenuButton('BACK');
  Object.assign(accountBackButton.style, { marginTop: 'auto' });

  const accountEmailInput = makeAccountInput('Email', 'email');
  accountEmailInput.autocomplete = 'email';
  const accountUsernameInput = makeAccountInput('Username (optional)');
  accountUsernameInput.autocomplete = 'username';
  const accountPasswordInput = makeAccountInput('Password', 'password');
  accountPasswordInput.autocomplete = 'current-password';
  const accountResetPasswordInput = makeAccountInput(
    'New password',
    'password',
  );
  accountResetPasswordInput.autocomplete = 'new-password';
  let hiddenResetToken: string | null = authInitialResetToken;

  const accountSignedOutActions = document.createElement('div');
  Object.assign(accountSignedOutActions.style, {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  });
  const accountSignedInActions = document.createElement('div');
  Object.assign(accountSignedInActions.style, {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  });

  const accountEmailEntrySection = document.createElement('div');
  Object.assign(accountEmailEntrySection.style, {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  });
  const accountEmailAuthSection = document.createElement('div');
  Object.assign(accountEmailAuthSection.style, {
    display: 'none',
    flexDirection: 'column',
    gap: '8px',
  });
  const accountResetSection = document.createElement('div');
  Object.assign(accountResetSection.style, {
    display: 'none',
    flexDirection: 'column',
    gap: '8px',
  });

  const accountEmailModeLabel = makeSectionLabel('SIGN IN WITH EMAIL');
  Object.assign(accountEmailModeLabel.style, { marginTop: '0' });
  const accountEmailSelectedValue = document.createElement('div');
  Object.assign(accountEmailSelectedValue.style, {
    color: '#b6c2d4',
    fontSize: '12px',
  });

  const accountResetHelp = document.createElement('div');
  accountResetHelp.textContent =
    'Use the password reset link from your email to set a new password.';
  Object.assign(accountResetHelp.style, {
    color: '#b6c2d4',
    fontSize: '12px',
    lineHeight: '1.4',
  });
  const accountDivider = document.createElement('div');
  accountDivider.textContent = 'or';
  Object.assign(accountDivider.style, {
    color: '#8fa0b8',
    fontSize: '11px',
    textAlign: 'center',
    margin: '2px 0',
  });

  accountSignedOutActions.appendChild(makeSectionLabel('OAUTH'));
  accountSignedOutActions.appendChild(accountGoogleButton);
  accountSignedOutActions.appendChild(accountDiscordButton);
  accountSignedOutActions.appendChild(accountDivider);
  accountSignedOutActions.appendChild(makeSectionLabel('EMAIL'));
  accountEmailEntrySection.appendChild(accountEmailInput);
  accountEmailEntrySection.appendChild(accountEmailContinueButton);
  accountSignedOutActions.appendChild(accountEmailEntrySection);

  accountEmailAuthSection.appendChild(accountEmailModeLabel);
  accountEmailAuthSection.appendChild(accountEmailSelectedValue);
  accountEmailAuthSection.appendChild(accountUsernameInput);
  accountEmailAuthSection.appendChild(accountPasswordInput);
  accountEmailAuthSection.appendChild(accountEmailSubmitButton);
  accountEmailAuthSection.appendChild(accountForgotLink);
  accountEmailAuthSection.appendChild(accountSwitchModeLink);
  accountEmailAuthSection.appendChild(accountBackToEmailLink);
  accountSignedOutActions.appendChild(accountEmailAuthSection);

  accountResetSection.appendChild(makeSectionLabel('RESET PASSWORD'));
  accountResetSection.appendChild(accountResetHelp);
  accountResetSection.appendChild(accountResetPasswordInput);
  accountResetSection.appendChild(accountResetSubmitButton);
  accountResetSection.appendChild(accountResetBackLink);
  accountSignedOutActions.appendChild(accountResetSection);

  accountSignedInActions.appendChild(accountRefreshButton);
  accountSignedInActions.appendChild(accountVerifyButton);
  accountSignedInActions.appendChild(accountLogoutButton);

  const myModelsTitle = document.createElement('div');
  myModelsTitle.textContent = 'MY MODELS';
  Object.assign(myModelsTitle.style, {
    color: '#8fa0b8',
    fontSize: '12px',
    letterSpacing: '0.5px',
    marginBottom: '6px',
    textAlign: 'center',
  });
  const myModelsSummary = document.createElement('div');
  Object.assign(myModelsSummary.style, {
    color: '#b6c2d4',
    fontSize: '12px',
    lineHeight: '1.45',
    minHeight: '44px',
    background: '#0b0f14',
    border: '1px solid #1f2a37',
    borderRadius: '6px',
    padding: '8px',
    whiteSpace: 'pre-wrap',
  });
  const myModelsActionStatus = document.createElement('div');
  Object.assign(myModelsActionStatus.style, {
    marginTop: '4px',
    fontSize: '12px',
    color: '#8fa0b8',
    minHeight: '16px',
    textAlign: 'center',
  });
  const myModelsSignedOutHint = document.createElement('div');
  myModelsSignedOutHint.textContent = 'Sign in from ACCOUNT to manage models.';
  Object.assign(myModelsSignedOutHint.style, {
    color: '#8fa0b8',
    fontSize: '12px',
    lineHeight: '1.4',
    textAlign: 'center',
    marginTop: '2px',
  });
  const myModelsActions = document.createElement('div');
  Object.assign(myModelsActions.style, {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  });
  const myModelsCloudLabel = makeSectionLabel('CLOUD MODEL');
  Object.assign(myModelsCloudLabel.style, { marginTop: '4px' });
  const myModelsLoadButton = makeMenuButton('LOAD CLOUD MODEL');
  const myModelsSaveButton = makeMenuButton('SAVE CURRENT MODEL');
  const myModelsAxesLabel = makeSectionLabel('MODEL AXES');
  const myModelsAxesControls = document.createElement('div');
  Object.assign(myModelsAxesControls.style, {
    display: 'grid',
    gridTemplateColumns: '1fr',
    gap: '6px',
  });
  const makeAxisSelect = (
    options: string[],
    fallbackLabel: string,
  ): HTMLSelectElement => {
    const select = document.createElement('select');
    Object.assign(select.style, {
      color: '#e2e8f0',
      background: '#0b0f14',
      border: '1px solid #1f2a37',
      borderRadius: '4px',
      fontSize: '12px',
      padding: '6px 8px',
      width: '100%',
      boxSizing: 'border-box',
    });
    const values = options.length > 0 ? options : [fallbackLabel];
    for (const value of values) {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = value;
      select.appendChild(option);
    }
    return select;
  };
  const myModelsArchSelect = makeAxisSelect(modelAxesOptions.arch, 'full');
  const myModelsRewardProfileSelect = makeAxisSelect(
    modelAxesOptions.rewardProfile,
    'default',
  );
  const myModelsQueuePolicySelect = makeAxisSelect(
    modelAxesOptions.queuePolicy,
    'next_piece_v1',
  );
  const myModelsApplyAxesButton = makeMenuButton('APPLY AXES');
  myModelsAxesControls.appendChild(myModelsArchSelect);
  myModelsAxesControls.appendChild(myModelsRewardProfileSelect);
  myModelsAxesControls.appendChild(myModelsQueuePolicySelect);
  myModelsAxesControls.appendChild(myModelsApplyAxesButton);
  const myModelsBaselinesLabel = makeSectionLabel('GLOBAL BASELINES');
  const myModelsBaselinesSummary = document.createElement('div');
  Object.assign(myModelsBaselinesSummary.style, {
    color: '#b6c2d4',
    fontSize: '12px',
    lineHeight: '1.35',
    background: '#0b0f14',
    border: '1px solid #1f2a37',
    borderRadius: '6px',
    padding: '8px',
    whiteSpace: 'pre-wrap',
  });
  const myModelsBaselinesSelect = document.createElement('select');
  myModelsBaselinesSelect.size = 5;
  Object.assign(myModelsBaselinesSelect.style, {
    width: '100%',
    boxSizing: 'border-box',
    background: '#0b0f14',
    color: '#e2e8f0',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    padding: '6px 8px',
    fontSize: '12px',
  });
  const myModelsBaselinesButtons = document.createElement('div');
  Object.assign(myModelsBaselinesButtons.style, {
    display: 'flex',
    gap: '6px',
  });
  const myModelsBaselinesRefreshButton = makeMenuButton('REFRESH');
  const myModelsResetToBaselineButton = makeMenuButton('RESET TO SELECTED');
  Object.assign(myModelsBaselinesRefreshButton.style, { flex: '1' });
  Object.assign(myModelsResetToBaselineButton.style, { flex: '1' });
  myModelsBaselinesButtons.appendChild(myModelsBaselinesRefreshButton);
  myModelsBaselinesButtons.appendChild(myModelsResetToBaselineButton);
  const myModelsTrainingLabel = makeSectionLabel('LOCAL TRAINING');
  Object.assign(myModelsTrainingLabel.style, { marginTop: '4px' });
  const myModelsTrainingSummary = document.createElement('div');
  Object.assign(myModelsTrainingSummary.style, {
    color: '#b6c2d4',
    fontSize: '12px',
    lineHeight: '1.4',
    background: '#0b0f14',
    border: '1px solid #1f2a37',
    borderRadius: '6px',
    padding: '8px',
    whiteSpace: 'pre-wrap',
  });
  const myModelsAdvancedLabel = makeSectionLabel('TRAINING SETTINGS');
  const myModelsTrainingControls = document.createElement('div');
  Object.assign(myModelsTrainingControls.style, {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '6px',
  });
  const makeMyModelsTrainingInput = (
    placeholder: string,
    type: 'number' | 'text' = 'number',
  ): HTMLInputElement => {
    const input = document.createElement('input');
    input.type = type;
    input.placeholder = placeholder;
    Object.assign(input.style, {
      color: '#e2e8f0',
      background: '#0b0f14',
      border: '1px solid #1f2a37',
      borderRadius: '4px',
      fontSize: '12px',
      padding: '6px 8px',
      width: '100%',
      boxSizing: 'border-box',
    });
    return input;
  };
  const myModelsEpochsInput = makeMyModelsTrainingInput('epochs');
  myModelsEpochsInput.min = '1';
  myModelsEpochsInput.step = '1';
  const myModelsLearningRateInput = makeMyModelsTrainingInput('learning rate');
  myModelsLearningRateInput.min = '0.000001';
  myModelsLearningRateInput.step = '0.0001';
  const myModelsSampleLimitInput = makeMyModelsTrainingInput('sample limit');
  myModelsSampleLimitInput.min = '1';
  myModelsSampleLimitInput.step = '1';
  const myModelsBackendSelect = document.createElement('select');
  for (const [value, label] of [
    ['auto', 'backend: auto'],
    ['webgl', 'backend: webgl'],
    ['cpu', 'backend: cpu'],
  ] as const) {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = label;
    myModelsBackendSelect.appendChild(option);
  }
  Object.assign(myModelsBackendSelect.style, {
    color: '#e2e8f0',
    background: '#0b0f14',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    fontSize: '12px',
    padding: '6px 8px',
    width: '100%',
    boxSizing: 'border-box',
  });
  const myModelsTrainingPresetSummary = document.createElement('div');
  Object.assign(myModelsTrainingPresetSummary.style, {
    color: '#8fa0b8',
    fontSize: '11px',
    lineHeight: '1.35',
    whiteSpace: 'pre-wrap',
  });
  myModelsTrainingControls.appendChild(myModelsEpochsInput);
  myModelsTrainingControls.appendChild(myModelsLearningRateInput);
  myModelsTrainingControls.appendChild(myModelsSampleLimitInput);
  myModelsTrainingControls.appendChild(myModelsBackendSelect);
  const myModelsTrainButton = makeMenuButton('TRAIN ON CLIENT SAMPLES');
  const myModelsRecordingLabel = makeSectionLabel('TRAJECTORY RECORDINGS');
  Object.assign(myModelsRecordingLabel.style, { marginTop: '4px' });
  const myModelsRecordingSummary = document.createElement('div');
  Object.assign(myModelsRecordingSummary.style, {
    color: '#b6c2d4',
    fontSize: '12px',
    lineHeight: '1.4',
    background: '#0b0f14',
    border: '1px solid #1f2a37',
    borderRadius: '6px',
    padding: '8px',
    whiteSpace: 'pre-wrap',
  });
  const myModelsUploadTrajectoryButton = makeMenuButton('UPLOAD TRAJECTORY');
  const myModelsBackButton = makeMenuButton('BACK');
  Object.assign(myModelsBackButton.style, { marginTop: 'auto' });
  myModelsActions.appendChild(myModelsLoadButton);
  myModelsActions.appendChild(myModelsSaveButton);
  myModelsPanel.appendChild(myModelsTitle);
  myModelsPanel.appendChild(myModelsSummary);
  myModelsPanel.appendChild(myModelsActionStatus);
  myModelsPanel.appendChild(myModelsCloudLabel);
  myModelsPanel.appendChild(myModelsSignedOutHint);
  myModelsPanel.appendChild(myModelsActions);
  myModelsPanel.appendChild(myModelsAxesLabel);
  myModelsPanel.appendChild(myModelsAxesControls);
  myModelsPanel.appendChild(myModelsBaselinesLabel);
  myModelsPanel.appendChild(myModelsBaselinesSummary);
  myModelsPanel.appendChild(myModelsBaselinesSelect);
  myModelsPanel.appendChild(myModelsBaselinesButtons);
  myModelsPanel.appendChild(myModelsTrainingLabel);
  myModelsPanel.appendChild(myModelsTrainingSummary);
  myModelsPanel.appendChild(myModelsAdvancedLabel);
  myModelsPanel.appendChild(myModelsTrainingControls);
  myModelsPanel.appendChild(myModelsTrainingPresetSummary);
  myModelsPanel.appendChild(myModelsTrainButton);
  myModelsPanel.appendChild(myModelsRecordingLabel);
  myModelsPanel.appendChild(myModelsRecordingSummary);
  myModelsPanel.appendChild(myModelsUploadTrajectoryButton);
  myModelsPanel.appendChild(myModelsBackButton);

  const adminTitle = document.createElement('div');
  adminTitle.textContent = 'ADMIN';
  Object.assign(adminTitle.style, {
    color: '#8fa0b8',
    fontSize: '12px',
    letterSpacing: '0.5px',
    marginBottom: '4px',
  });
  const adminSummary = document.createElement('div');
  Object.assign(adminSummary.style, {
    color: '#b6c2d4',
    fontSize: '12px',
    lineHeight: '1.4',
    background: '#0b0f14',
    border: '1px solid #1f2a37',
    borderRadius: '6px',
    padding: '8px',
    whiteSpace: 'pre-wrap',
  });
  const adminStatus = document.createElement('div');
  Object.assign(adminStatus.style, {
    minHeight: '28px',
    color: '#8fa0b8',
    fontSize: '11px',
    lineHeight: '1.35',
    marginTop: '2px',
    whiteSpace: 'pre-wrap',
  });
  const adminActions = document.createElement('div');
  Object.assign(adminActions.style, {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  });
  const adminWhoAmIButton = makeMenuButton('CHECK ADMIN API');
  const adminOpenRouteButton = makeMenuButton('OPEN /admin');
  const adminPublishBaselineButton = makeMenuButton(
    'PUBLISH CURRENT MODEL AS BASELINE',
  );
  const adminPublishLabelInput = document.createElement('input');
  adminPublishLabelInput.type = 'text';
  adminPublishLabelInput.placeholder = 'baseline label (optional)';
  Object.assign(adminPublishLabelInput.style, {
    color: '#e2e8f0',
    background: '#0b0f14',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    fontSize: '12px',
    padding: '6px 8px',
    width: '100%',
    boxSizing: 'border-box',
  });
  const adminPublishDefaultRow = document.createElement('label');
  Object.assign(adminPublishDefaultRow.style, {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    color: '#b6c2d4',
    fontSize: '12px',
  });
  const adminPublishDefaultCheckbox = document.createElement('input');
  adminPublishDefaultCheckbox.type = 'checkbox';
  adminPublishDefaultCheckbox.checked = false;
  adminPublishDefaultRow.appendChild(adminPublishDefaultCheckbox);
  adminPublishDefaultRow.appendChild(
    document.createTextNode('Set as default baseline'),
  );
  adminActions.appendChild(adminWhoAmIButton);
  adminActions.appendChild(adminOpenRouteButton);
  adminActions.appendChild(adminPublishLabelInput);
  adminActions.appendChild(adminPublishDefaultRow);
  adminActions.appendChild(adminPublishBaselineButton);

  const adminRlLabel = makeSectionLabel('RL DATA LOOP');
  Object.assign(adminRlLabel.style, { marginTop: '4px' });
  const adminRlControls = document.createElement('div');
  Object.assign(adminRlControls.style, {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
  });
  const adminManifestMinSamplesInput = document.createElement('input');
  adminManifestMinSamplesInput.type = 'number';
  adminManifestMinSamplesInput.min = '1';
  adminManifestMinSamplesInput.step = '1';
  adminManifestMinSamplesInput.value = '8';
  adminManifestMinSamplesInput.placeholder = 'manifest min samples';
  Object.assign(adminManifestMinSamplesInput.style, {
    color: '#e2e8f0',
    background: '#0b0f14',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    fontSize: '12px',
    padding: '6px 8px',
    width: '100%',
    boxSizing: 'border-box',
  });
  const adminManifestActorTypeSelect = document.createElement('select');
  for (const [value, label] of [
    ['', 'manifest actor: all'],
    ['human', 'manifest actor: human'],
    ['bot', 'manifest actor: bot'],
  ] as const) {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = label;
    adminManifestActorTypeSelect.appendChild(option);
  }
  Object.assign(adminManifestActorTypeSelect.style, {
    color: '#e2e8f0',
    background: '#0b0f14',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    fontSize: '12px',
    padding: '6px 8px',
    width: '100%',
    boxSizing: 'border-box',
  });
  const adminPrepareManifestButton = makeMenuButton('PREPARE MANIFEST');
  const adminTrainGlobalOneShotButton = makeMenuButton(
    'TRAIN GLOBAL (ONE-SHOT)',
  );
  const adminPublishGlobalCandidateButton = makeMenuButton(
    'PUBLISH GLOBAL CANDIDATE',
  );
  adminRlControls.appendChild(adminManifestMinSamplesInput);
  adminRlControls.appendChild(adminManifestActorTypeSelect);
  adminRlControls.appendChild(adminPrepareManifestButton);
  adminRlControls.appendChild(adminTrainGlobalOneShotButton);
  adminRlControls.appendChild(adminPublishGlobalCandidateButton);

  const adminBaselinesLabel = makeSectionLabel('GLOBAL BASELINES');
  Object.assign(adminBaselinesLabel.style, { marginTop: '4px' });
  const adminBaselinesControls = document.createElement('div');
  Object.assign(adminBaselinesControls.style, {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
  });
  const adminBaselinesButtons = document.createElement('div');
  Object.assign(adminBaselinesButtons.style, {
    display: 'flex',
    gap: '6px',
  });
  const adminBaselinesRefreshButton = makeMenuButton('REFRESH');
  const adminBaselinesSetDefaultButton = makeMenuButton('SET DEFAULT');
  const adminBaselinesRetireButton = makeMenuButton('RETIRE');
  Object.assign(adminBaselinesRefreshButton.style, { flex: '1' });
  Object.assign(adminBaselinesSetDefaultButton.style, { flex: '1' });
  Object.assign(adminBaselinesRetireButton.style, { flex: '1' });
  adminBaselinesButtons.appendChild(adminBaselinesRefreshButton);
  adminBaselinesButtons.appendChild(adminBaselinesSetDefaultButton);
  adminBaselinesButtons.appendChild(adminBaselinesRetireButton);
  const adminBaselinesSummary = document.createElement('div');
  Object.assign(adminBaselinesSummary.style, {
    color: '#b6c2d4',
    fontSize: '12px',
    lineHeight: '1.35',
    background: '#0b0f14',
    border: '1px solid #1f2a37',
    borderRadius: '6px',
    padding: '8px',
    whiteSpace: 'pre-wrap',
  });
  const adminBaselinesSelect = document.createElement('select');
  adminBaselinesSelect.size = 6;
  Object.assign(adminBaselinesSelect.style, {
    width: '100%',
    boxSizing: 'border-box',
    background: '#0b0f14',
    color: '#e2e8f0',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    padding: '6px 8px',
    fontSize: '12px',
  });
  adminBaselinesControls.appendChild(adminBaselinesButtons);
  adminBaselinesControls.appendChild(adminBaselinesSummary);
  adminBaselinesControls.appendChild(adminBaselinesSelect);

  const adminDatasetLabel = makeSectionLabel('RECORDINGS DATASET');
  Object.assign(adminDatasetLabel.style, { marginTop: '4px' });
  const adminDatasetControls = document.createElement('div');
  Object.assign(adminDatasetControls.style, {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
  });
  const adminModeFilterInput = document.createElement('input');
  adminModeFilterInput.type = 'text';
  adminModeFilterInput.placeholder = 'mode (optional)';
  Object.assign(adminModeFilterInput.style, {
    color: '#e2e8f0',
    background: '#0b0f14',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    fontSize: '12px',
    padding: '6px 8px',
    width: '100%',
    boxSizing: 'border-box',
  });
  const adminBuildFilterInput = document.createElement('input');
  adminBuildFilterInput.type = 'text';
  adminBuildFilterInput.placeholder = 'build version (optional)';
  Object.assign(adminBuildFilterInput.style, {
    color: '#e2e8f0',
    background: '#0b0f14',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    fontSize: '12px',
    padding: '6px 8px',
    width: '100%',
    boxSizing: 'border-box',
  });
  const adminLimitInput = document.createElement('input');
  adminLimitInput.type = 'number';
  adminLimitInput.min = '1';
  adminLimitInput.max = '200';
  adminLimitInput.step = '1';
  adminLimitInput.value = '25';
  adminLimitInput.placeholder = 'limit';
  Object.assign(adminLimitInput.style, {
    color: '#e2e8f0',
    background: '#0b0f14',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    fontSize: '12px',
    padding: '6px 8px',
    width: '100%',
    boxSizing: 'border-box',
  });
  adminLimitInput.addEventListener('wheel', (event) => {
    if (document.activeElement === adminLimitInput) {
      event.preventDefault();
    }
  });
  adminLimitInput.addEventListener('keydown', (event) => {
    if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
      event.preventDefault();
    }
  });
  const adminDatasetButtons = document.createElement('div');
  Object.assign(adminDatasetButtons.style, {
    display: 'flex',
    gap: '6px',
  });
  const adminQueryButton = makeMenuButton('QUERY');
  const adminNextPageButton = makeMenuButton('NEXT PAGE');
  Object.assign(adminQueryButton.style, { flex: '1' });
  Object.assign(adminNextPageButton.style, { flex: '1' });
  adminDatasetButtons.appendChild(adminQueryButton);
  adminDatasetButtons.appendChild(adminNextPageButton);
  const adminDatasetSummary = document.createElement('div');
  Object.assign(adminDatasetSummary.style, {
    color: '#b6c2d4',
    fontSize: '12px',
    lineHeight: '1.35',
    background: '#0b0f14',
    border: '1px solid #1f2a37',
    borderRadius: '6px',
    padding: '8px',
    whiteSpace: 'pre-wrap',
  });
  const adminRecordingsSelect = document.createElement('select');
  adminRecordingsSelect.size = 8;
  Object.assign(adminRecordingsSelect.style, {
    width: '100%',
    boxSizing: 'border-box',
    background: '#0b0f14',
    color: '#e2e8f0',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    padding: '6px 8px',
    fontSize: '12px',
  });
  const adminLoadSelectedButton = makeMenuButton('LOAD SELECTED RECORDING');
  const adminLoadedSummary = document.createElement('div');
  Object.assign(adminLoadedSummary.style, {
    color: '#b6c2d4',
    fontSize: '12px',
    lineHeight: '1.35',
    background: '#0b0f14',
    border: '1px solid #1f2a37',
    borderRadius: '6px',
    padding: '8px',
    whiteSpace: 'pre-wrap',
  });
  adminDatasetControls.appendChild(adminModeFilterInput);
  adminDatasetControls.appendChild(adminBuildFilterInput);
  adminDatasetControls.appendChild(adminLimitInput);
  adminDatasetControls.appendChild(adminDatasetButtons);
  adminDatasetControls.appendChild(adminDatasetSummary);
  adminDatasetControls.appendChild(adminRecordingsSelect);
  adminDatasetControls.appendChild(adminLoadSelectedButton);
  adminDatasetControls.appendChild(adminLoadedSummary);
  const adminBackButton = makeMenuButton('BACK');
  Object.assign(adminBackButton.style, { marginTop: 'auto' });
  adminPanel.appendChild(adminTitle);
  adminPanel.appendChild(adminSummary);
  adminPanel.appendChild(adminStatus);
  adminPanel.appendChild(adminActions);
  adminPanel.appendChild(adminRlLabel);
  adminPanel.appendChild(adminRlControls);
  adminPanel.appendChild(adminBaselinesLabel);
  adminPanel.appendChild(adminBaselinesControls);
  adminPanel.appendChild(adminDatasetLabel);
  adminPanel.appendChild(adminDatasetControls);
  adminPanel.appendChild(adminBackButton);

  const botLabTitle = document.createElement('div');
  botLabTitle.textContent = 'BOT LAB';
  Object.assign(botLabTitle.style, {
    color: '#8fa0b8',
    fontSize: '12px',
    letterSpacing: '0.5px',
    marginBottom: '4px',
  });
  const botLabSummary = document.createElement('div');
  Object.assign(botLabSummary.style, {
    color: '#b6c2d4',
    fontSize: '12px',
    lineHeight: '1.35',
    background: '#0b0f14',
    border: '1px solid #1f2a37',
    borderRadius: '6px',
    padding: '8px',
    whiteSpace: 'pre-wrap',
  });
  const botLabStatus = document.createElement('div');
  Object.assign(botLabStatus.style, {
    minHeight: '24px',
    color: '#8fa0b8',
    fontSize: '11px',
    lineHeight: '1.35',
    marginTop: '2px',
    whiteSpace: 'pre-wrap',
  });

  const botLabPolicyLabel = makeSectionLabel('POLICY REGISTRY');
  Object.assign(botLabPolicyLabel.style, { marginTop: '4px' });
  const botLabPolicyButtonsTop = document.createElement('div');
  Object.assign(botLabPolicyButtonsTop.style, {
    display: 'flex',
    gap: '6px',
  });
  const botLabFetchCurrentButton = makeMenuButton('LOAD CURRENT');
  const botLabRefreshPoliciesButton = makeMenuButton('REFRESH LIST');
  Object.assign(botLabFetchCurrentButton.style, { flex: '1' });
  Object.assign(botLabRefreshPoliciesButton.style, { flex: '1' });
  botLabPolicyButtonsTop.appendChild(botLabFetchCurrentButton);
  botLabPolicyButtonsTop.appendChild(botLabRefreshPoliciesButton);
  const botLabPoliciesSelect = document.createElement('select');
  botLabPoliciesSelect.size = 7;
  Object.assign(botLabPoliciesSelect.style, {
    width: '100%',
    boxSizing: 'border-box',
    background: '#0b0f14',
    color: '#e2e8f0',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    padding: '6px 8px',
    fontSize: '12px',
  });
  const botLabPolicyButtonsRow = document.createElement('div');
  Object.assign(botLabPolicyButtonsRow.style, {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '6px',
  });
  const botLabPolicyNextButton = makeMenuButton('NEXT');
  const botLabPolicyLoadButton = makeMenuButton('LOAD');
  const botLabPolicySelectCurrentButton = makeMenuButton('SET CURRENT');
  const botLabPolicyPinButton = makeMenuButton('PIN');
  const botLabPolicyUnpinButton = makeMenuButton('UNPIN');
  Object.assign(botLabPolicyNextButton.style, { flex: '1' });
  Object.assign(botLabPolicyLoadButton.style, { flex: '1' });
  Object.assign(botLabPolicySelectCurrentButton.style, { flex: '1' });
  Object.assign(botLabPolicyPinButton.style, { flex: '1' });
  Object.assign(botLabPolicyUnpinButton.style, { flex: '1' });
  botLabPolicyButtonsRow.appendChild(botLabPolicyNextButton);
  botLabPolicyButtonsRow.appendChild(botLabPolicyLoadButton);
  botLabPolicyButtonsRow.appendChild(botLabPolicySelectCurrentButton);
  botLabPolicyButtonsRow.appendChild(botLabPolicyPinButton);
  botLabPolicyButtonsRow.appendChild(botLabPolicyUnpinButton);
  const botLabPublishRow = document.createElement('div');
  Object.assign(botLabPublishRow.style, {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    flexWrap: 'wrap',
  });
  const botLabPublishSourceSelect = document.createElement('select');
  for (const [value, label] of [
    ['bag7', 'publish source: bag7'],
    ['active_generator', 'publish source: active'],
  ] as const) {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = label;
    botLabPublishSourceSelect.appendChild(option);
  }
  Object.assign(botLabPublishSourceSelect.style, {
    flex: '1',
    color: '#e2e8f0',
    background: '#0b0f14',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    fontSize: '12px',
    padding: '6px 8px',
  });
  const botLabPublishPinToggle = document.createElement('input');
  botLabPublishPinToggle.type = 'checkbox';
  const botLabPublishCurrentToggle = document.createElement('input');
  botLabPublishCurrentToggle.type = 'checkbox';
  botLabPublishCurrentToggle.checked = true;
  const makeInlineToggle = (text: string, input: HTMLInputElement) => {
    const label = document.createElement('label');
    Object.assign(label.style, {
      display: 'inline-flex',
      alignItems: 'center',
      gap: '4px',
      color: '#b6c2d4',
      fontSize: '11px',
    });
    label.appendChild(input);
    label.appendChild(document.createTextNode(text));
    return label;
  };
  botLabPublishRow.appendChild(botLabPublishSourceSelect);
  botLabPublishRow.appendChild(makeInlineToggle('pin', botLabPublishPinToggle));
  botLabPublishRow.appendChild(
    makeInlineToggle('set current', botLabPublishCurrentToggle),
  );
  const botLabPublishPolicyButton = makeMenuButton('PUBLISH LOADED POLICY');

  const botLabTrainLabel = makeSectionLabel('TRAIN / VALIDATE');
  Object.assign(botLabTrainLabel.style, { marginTop: '4px' });
  const botLabTrainControls = document.createElement('div');
  Object.assign(botLabTrainControls.style, {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '6px',
  });
  const botLabEpisodesInput = document.createElement('input');
  botLabEpisodesInput.type = 'number';
  botLabEpisodesInput.min = '1';
  botLabEpisodesInput.step = '1';
  botLabEpisodesInput.value = '24';
  botLabEpisodesInput.placeholder = 'episodes';
  const botLabMaxPiecesInput = document.createElement('input');
  botLabMaxPiecesInput.type = 'number';
  botLabMaxPiecesInput.min = '8';
  botLabMaxPiecesInput.step = '1';
  botLabMaxPiecesInput.value = '120';
  botLabMaxPiecesInput.placeholder = 'max pieces';
  const botLabSeedInput = document.createElement('input');
  botLabSeedInput.type = 'number';
  botLabSeedInput.step = '1';
  botLabSeedInput.placeholder = 'seed';
  botLabSeedInput.value = '42030';
  const botLabTrainSourceSelect = document.createElement('select');
  for (const [value, label] of [
    ['bag7', 'train source: bag7'],
    ['active_generator', 'train source: active'],
  ] as const) {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = label;
    botLabTrainSourceSelect.appendChild(option);
  }
  botLabTrainSourceSelect.value = 'bag7';
  for (const input of [
    botLabEpisodesInput,
    botLabMaxPiecesInput,
    botLabSeedInput,
  ]) {
    Object.assign(input.style, {
      color: '#e2e8f0',
      background: '#0b0f14',
      border: '1px solid #1f2a37',
      borderRadius: '4px',
      fontSize: '12px',
      padding: '6px 8px',
      width: '100%',
      boxSizing: 'border-box',
    });
  }
  Object.assign(botLabTrainSourceSelect.style, {
    color: '#e2e8f0',
    background: '#0b0f14',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    fontSize: '12px',
    padding: '6px 8px',
    width: '100%',
    boxSizing: 'border-box',
  });
  botLabTrainControls.appendChild(botLabEpisodesInput);
  botLabTrainControls.appendChild(botLabMaxPiecesInput);
  botLabTrainControls.appendChild(botLabSeedInput);
  botLabTrainControls.appendChild(botLabTrainSourceSelect);
  const botLabTrainButton = makeMenuButton('TRAIN POLICY (ONE-SHOT)');
  const botLabHeadlessValidateButton = makeMenuButton('HEADLESS VALIDATE 10K');

  const botLabGuiLabel = makeSectionLabel('GUI INSPECT');
  Object.assign(botLabGuiLabel.style, { marginTop: '4px' });
  const botLabGuiControls = document.createElement('div');
  Object.assign(botLabGuiControls.style, {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '6px',
  });
  const botLabGuiApmInput = document.createElement('input');
  botLabGuiApmInput.type = 'number';
  botLabGuiApmInput.min = '20';
  botLabGuiApmInput.max = '1200';
  botLabGuiApmInput.step = '1';
  botLabGuiApmInput.value = '240';
  botLabGuiApmInput.placeholder = 'APM';
  const botLabGuiSeedInput = document.createElement('input');
  botLabGuiSeedInput.type = 'number';
  botLabGuiSeedInput.step = '1';
  botLabGuiSeedInput.value = '42030';
  botLabGuiSeedInput.placeholder = 'seed';
  const botLabGuiPiecesInput = document.createElement('input');
  botLabGuiPiecesInput.type = 'number';
  botLabGuiPiecesInput.min = '1';
  botLabGuiPiecesInput.step = '1';
  botLabGuiPiecesInput.value = '20';
  botLabGuiPiecesInput.placeholder = 'pieces';
  const botLabGuiSourceSelect = document.createElement('select');
  for (const [value, label] of [
    ['bag7', 'GUI source: bag7'],
    ['active_generator', 'GUI source: active'],
  ] as const) {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = label;
    botLabGuiSourceSelect.appendChild(option);
  }
  botLabGuiSourceSelect.value = 'bag7';
  for (const input of [
    botLabGuiApmInput,
    botLabGuiSeedInput,
    botLabGuiPiecesInput,
  ]) {
    Object.assign(input.style, {
      color: '#e2e8f0',
      background: '#0b0f14',
      border: '1px solid #1f2a37',
      borderRadius: '4px',
      fontSize: '12px',
      padding: '6px 8px',
      width: '100%',
      boxSizing: 'border-box',
    });
  }
  Object.assign(botLabGuiSourceSelect.style, {
    color: '#e2e8f0',
    background: '#0b0f14',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    fontSize: '12px',
    padding: '6px 8px',
    width: '100%',
    boxSizing: 'border-box',
  });
  botLabGuiControls.appendChild(botLabGuiApmInput);
  botLabGuiControls.appendChild(botLabGuiSeedInput);
  botLabGuiControls.appendChild(botLabGuiPiecesInput);
  botLabGuiControls.appendChild(botLabGuiSourceSelect);
  const botLabGuiButtons = document.createElement('div');
  Object.assign(botLabGuiButtons.style, {
    display: 'flex',
    gap: '6px',
  });
  const botLabStartGuiButton = makeMenuButton('START GUI INSPECT');
  const botLabStopGuiButton = makeMenuButton('STOP GUI INSPECT');
  Object.assign(botLabStartGuiButton.style, { flex: '1' });
  Object.assign(botLabStopGuiButton.style, { flex: '1' });
  botLabGuiButtons.appendChild(botLabStartGuiButton);
  botLabGuiButtons.appendChild(botLabStopGuiButton);

  const botLabDataLabel = makeSectionLabel('GENERATE / BENCHMARK');
  Object.assign(botLabDataLabel.style, { marginTop: '4px' });
  const botLabDataControls = document.createElement('div');
  Object.assign(botLabDataControls.style, {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '6px',
  });
  const botLabGenerateSessionsInput = document.createElement('input');
  botLabGenerateSessionsInput.type = 'number';
  botLabGenerateSessionsInput.min = '1';
  botLabGenerateSessionsInput.step = '1';
  botLabGenerateSessionsInput.value = '3';
  botLabGenerateSessionsInput.placeholder = 'sessions';
  const botLabGenerateSourceSelect = document.createElement('select');
  for (const [value, label] of [
    ['active_generator', 'gen source: active'],
    ['bag7', 'gen source: bag7'],
  ] as const) {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = label;
    botLabGenerateSourceSelect.appendChild(option);
  }
  botLabGenerateSourceSelect.value = 'active_generator';
  for (const input of [botLabGenerateSessionsInput]) {
    Object.assign(input.style, {
      color: '#e2e8f0',
      background: '#0b0f14',
      border: '1px solid #1f2a37',
      borderRadius: '4px',
      fontSize: '12px',
      padding: '6px 8px',
      width: '100%',
      boxSizing: 'border-box',
    });
  }
  Object.assign(botLabGenerateSourceSelect.style, {
    color: '#e2e8f0',
    background: '#0b0f14',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    fontSize: '12px',
    padding: '6px 8px',
    width: '100%',
    boxSizing: 'border-box',
  });
  botLabDataControls.appendChild(botLabGenerateSessionsInput);
  botLabDataControls.appendChild(botLabGenerateSourceSelect);
  const botLabGenerateButton = makeMenuButton('GENERATE RECORDINGS');
  const botLabBenchmarkButton = makeMenuButton('RUN BENCHMARK');
  const botLabApplyBenchmarkArchButton = makeMenuButton('APPLY BENCHMARK ARCH');
  const botLabBackButton = makeMenuButton('BACK');
  Object.assign(botLabBackButton.style, { marginTop: 'auto' });

  botLabPanel.appendChild(botLabTitle);
  botLabPanel.appendChild(botLabSummary);
  botLabPanel.appendChild(botLabStatus);
  botLabPanel.appendChild(botLabPolicyLabel);
  botLabPanel.appendChild(botLabPolicyButtonsTop);
  botLabPanel.appendChild(botLabPoliciesSelect);
  botLabPanel.appendChild(botLabPolicyButtonsRow);
  botLabPanel.appendChild(botLabPublishRow);
  botLabPanel.appendChild(botLabPublishPolicyButton);
  botLabPanel.appendChild(botLabTrainLabel);
  botLabPanel.appendChild(botLabTrainControls);
  botLabPanel.appendChild(botLabTrainButton);
  botLabPanel.appendChild(botLabHeadlessValidateButton);
  botLabPanel.appendChild(botLabGuiLabel);
  botLabPanel.appendChild(botLabGuiControls);
  botLabPanel.appendChild(botLabGuiButtons);
  botLabPanel.appendChild(botLabDataLabel);
  botLabPanel.appendChild(botLabDataControls);
  botLabPanel.appendChild(botLabGenerateButton);
  botLabPanel.appendChild(botLabBenchmarkButton);
  botLabPanel.appendChild(botLabApplyBenchmarkArchButton);
  botLabPanel.appendChild(botLabBackButton);

  type SignedOutStage = 'email' | 'login' | 'signup' | 'reset';
  let signedOutStage: SignedOutStage = hiddenResetToken ? 'reset' : 'email';
  let currentAuthState: MenuAuthState = authState;
  let authActionPending = false;
  let modelActionPending = false;
  let adminActionPending = false;
  let adminDatasetPending = false;
  let adminCurrentCursor: string | null = null;
  let adminCurrentRecordings: MenuAdminRecordingSummary[] = [];
  let adminSelectedRecordingId: string | null = null;
  let adminLastPage: MenuAdminRecordingsPage | null = null;
  let adminGlobalBaselines: MenuAdminGlobalBaselineSummary[] = [];
  let adminSelectedBaselineId: string | null = null;
  let adminBaselinesPending = false;
  let adminBaselinesSelectorKey: string | null = null;
  let currentModelAxes = getModelAxes();
  let myModelsGlobalBaselines: MenuGlobalModelSummary[] = [];
  let myModelsSelectedBaselineId: string | null = null;
  let myModelsBaselinesLoading = false;
  let myModelsBaselinesSelectorKey: string | null = null;
  let botLabActionPending = false;
  let botLabListPending = false;
  let botLabPolicies: MenuBotPolicySummary[] = [];
  let botLabSelectedPolicyId: string | null = null;
  let botLabCurrentPolicyId: string | null = null;
  let botLabPolicyCursor: string | null = null;
  let botLabGuiInspectRunning = false;
  let lastTrainingPresetKey = '';
  const statusColor = (tone: MenuAuthStatusTone): string => {
    if (tone === 'success') return '#8fd19e';
    if (tone === 'error') return '#f28b82';
    return '#8fa0b8';
  };
  const setAccountActionStatus = (
    message: string,
    tone: MenuAuthStatusTone = 'neutral',
  ) => {
    accountActionStatus.textContent = message;
    accountActionStatus.style.color = statusColor(tone);
  };
  const setMyModelsActionStatus = (
    message: string,
    tone: MenuAuthStatusTone = 'neutral',
  ) => {
    myModelsActionStatus.textContent = message;
    myModelsActionStatus.style.color = statusColor(tone);
  };
  const setAdminActionStatus = (
    message: string,
    tone: MenuAuthStatusTone = 'neutral',
  ) => {
    adminStatus.textContent = message;
    adminStatus.style.color = statusColor(tone);
  };
  const setBotLabActionStatus = (
    message: string,
    tone: MenuAuthStatusTone = 'neutral',
  ) => {
    botLabStatus.textContent = message;
    botLabStatus.style.color = statusColor(tone);
  };
  const readField = (input: HTMLInputElement): string => input.value.trim();
  const toErrorMessage = (error: unknown, fallback: string): string => {
    if (error instanceof Error && error.message.trim()) return error.message;
    return fallback;
  };
  const parsePositiveIntInput = (
    input: HTMLInputElement,
  ): number | undefined => {
    const value = Number(input.value.trim());
    if (!Number.isFinite(value)) return undefined;
    const normalized = Math.trunc(value);
    if (normalized <= 0) return undefined;
    return normalized;
  };
  const parsePositiveFloatInput = (
    input: HTMLInputElement,
  ): number | undefined => {
    const value = Number(input.value.trim());
    if (!Number.isFinite(value)) return undefined;
    if (value <= 0) return undefined;
    return value;
  };
  const formatLearningRate = (value: number): string =>
    value >= 0.001 ? value.toFixed(4) : value.toExponential(2);
  const setSignedOutStage = (next: SignedOutStage) => {
    signedOutStage = next;
  };
  const syncSignedOutStageUi = () => {
    const isLogin = signedOutStage === 'login';
    const isSignup = signedOutStage === 'signup';
    const isEmailAuth = isLogin || isSignup;
    const isReset = signedOutStage === 'reset';
    accountEmailEntrySection.style.display =
      signedOutStage === 'email' ? 'flex' : 'none';
    accountEmailAuthSection.style.display = isEmailAuth ? 'flex' : 'none';
    accountResetSection.style.display = isReset ? 'flex' : 'none';
    accountDivider.style.display =
      signedOutStage === 'email' ? 'block' : 'none';
    accountUsernameInput.style.display = isSignup ? 'block' : 'none';
    accountForgotLink.style.display = isLogin ? 'inline-flex' : 'none';
    accountSwitchModeLink.textContent = isSignup
      ? 'Already have an account? Log in'
      : 'Need an account? Sign up';
    accountEmailSubmitButton.textContent = isSignup
      ? 'CREATE ACCOUNT'
      : 'LOG IN';
    accountEmailModeLabel.textContent = isSignup
      ? 'SIGN UP WITH EMAIL'
      : 'SIGN IN WITH EMAIL';
    const email = readField(accountEmailInput);
    accountEmailSelectedValue.textContent = email
      ? `Email: ${email}`
      : 'Email: (not set)';
  };

  const formatAccountSummary = (state: MenuAuthState): string => {
    if (state.loading) {
      return 'Checking session...';
    }
    if (!state.authenticated || !state.user) {
      return 'Not signed in.';
    }
    const emailLine = state.user.email ? `Email: ${state.user.email}` : '';
    const verificationLine =
      state.user.emailVerifiedAtMs == null
        ? 'Email verification: pending'
        : 'Email verification: complete';
    const roleLine = state.user.isAdmin ? 'Role: admin' : 'Role: user';
    return [
      `Signed in as ${state.user.username}`,
      emailLine,
      verificationLine,
      roleLine,
    ]
      .filter((line) => line.length > 0)
      .join('\n');
  };

  const syncModelAxesControls = (): void => {
    const ensureValue = (
      select: HTMLSelectElement,
      value: string,
      fallback: string,
    ): string => {
      if (
        Array.from(select.options).some(
          (option) => option.value.trim() === value.trim(),
        )
      ) {
        select.value = value;
        return value;
      }
      select.value = fallback;
      return fallback;
    };
    currentModelAxes = {
      arch: ensureValue(
        myModelsArchSelect,
        currentModelAxes.arch,
        myModelsArchSelect.options[0]?.value ?? 'full',
      ),
      rewardProfileId: ensureValue(
        myModelsRewardProfileSelect,
        currentModelAxes.rewardProfileId,
        myModelsRewardProfileSelect.options[0]?.value ?? 'default',
      ),
      queuePolicyId: ensureValue(
        myModelsQueuePolicySelect,
        currentModelAxes.queuePolicyId,
        myModelsQueuePolicySelect.options[0]?.value ?? 'next_piece_v1',
      ),
    };
  };

  const readModelAxesFromControls = (): MenuModelAxes => ({
    arch: myModelsArchSelect.value.trim().toLowerCase(),
    rewardProfileId: myModelsRewardProfileSelect.value.trim().toLowerCase(),
    queuePolicyId: myModelsQueuePolicySelect.value.trim().toLowerCase(),
  });

  const modelAxesEqual = (a: MenuModelAxes, b: MenuModelAxes): boolean =>
    a.arch === b.arch &&
    a.rewardProfileId === b.rewardProfileId &&
    a.queuePolicyId === b.queuePolicyId;

  const formatModelAxesCompact = (axes: MenuModelAxes): string =>
    `${axes.arch}/${axes.rewardProfileId}/${axes.queuePolicyId}`;

  const buildModelAxesSelectorKey = (
    modeId: string,
    axes: MenuModelAxes,
  ): string => `${modeId}:${formatModelAxesCompact(axes)}`;

  const formatMyModelsSummary = (state: MenuAuthState): string => {
    if (state.loading) {
      return 'Checking session...';
    }
    if (!state.authenticated || !state.user) {
      return 'Not signed in.';
    }
    return `Signed in as ${state.user.username}\nManage your cloud model for the current game mode.`;
  };

  const formatApproxBytes = (value: number | null): string => {
    if (value == null || !Number.isFinite(value) || value <= 0) return 'n/a';
    if (value < 1024) return `${Math.trunc(value)} B`;
    if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
    return `${(value / (1024 * 1024)).toFixed(2)} MB`;
  };

  const formatMyModelsBaselineOption = (
    baseline: MenuGlobalModelSummary,
  ): string => {
    const head = baseline.label || baseline.pipelineId || baseline.id;
    const prefix = baseline.isDefault ? '[default] ' : '';
    return `${prefix}${head}`;
  };

  const renderMyModelsBaselinesSelect = (): void => {
    myModelsBaselinesSelect.innerHTML = '';
    for (const baseline of myModelsGlobalBaselines) {
      const option = document.createElement('option');
      option.value = baseline.id;
      option.textContent = formatMyModelsBaselineOption(baseline);
      myModelsBaselinesSelect.appendChild(option);
    }
    if (myModelsGlobalBaselines.length > 0) {
      const activeId =
        myModelsSelectedBaselineId &&
        myModelsGlobalBaselines.some(
          (baseline) => baseline.id === myModelsSelectedBaselineId,
        )
          ? myModelsSelectedBaselineId
          : myModelsGlobalBaselines[0].id;
      myModelsBaselinesSelect.value = activeId;
      myModelsSelectedBaselineId = activeId;
    } else {
      myModelsSelectedBaselineId = null;
    }
  };

  const getSelectedMyModelsBaseline = (): MenuGlobalModelSummary | null => {
    if (!myModelsSelectedBaselineId) return null;
    return (
      myModelsGlobalBaselines.find(
        (baseline) => baseline.id === myModelsSelectedBaselineId,
      ) ?? null
    );
  };

  const formatMyModelsBaselinesSummary = (
    authenticated: boolean,
    modeId: string,
    axes: MenuModelAxes,
  ): string => {
    if (!authenticated) {
      return `Sign in to browse and reset to published global baselines.\nAxes: ${formatModelAxesCompact(axes)}`;
    }
    if (myModelsBaselinesLoading) {
      return `Loading baselines for mode "${modeId}"...\nAxes: ${formatModelAxesCompact(axes)}`;
    }
    if (myModelsGlobalBaselines.length === 0) {
      return `No global baselines published for mode "${modeId}".\nAxes: ${formatModelAxesCompact(axes)}`;
    }
    const selected = getSelectedMyModelsBaseline();
    if (!selected) {
      return [
        `Published baselines: ${myModelsGlobalBaselines.length}`,
        `Axes: ${formatModelAxesCompact(axes)}`,
      ].join('\n');
    }
    const lastUpdated =
      selected.updatedAtMs != null
        ? new Date(selected.updatedAtMs).toLocaleString()
        : 'n/a';
    return [
      `Published baselines: ${myModelsGlobalBaselines.length}`,
      `Selected: ${selected.label || selected.id}`,
      selected.pipelineId
        ? `Pipeline: ${selected.pipelineId}`
        : 'Pipeline: (none)',
      `Axes: ${formatModelAxesCompact(axes)}`,
      `Size: ${formatApproxBytes(selected.sizeBytes)}`,
      `Updated: ${lastUpdated}`,
    ].join('\n');
  };

  const refreshMyModelsBaselines = async (options?: {
    silent?: boolean;
  }): Promise<void> => {
    const authenticated =
      currentAuthState.authenticated && currentAuthState.user != null;
    if (!authenticated) {
      myModelsGlobalBaselines = [];
      myModelsSelectedBaselineId = null;
      myModelsBaselinesSelectorKey = null;
      renderMyModelsBaselinesSelect();
      updateMyModelsControls();
      return;
    }
    const modeId = getLocalTrainingStats().currentModeId;
    const selectorKey = buildModelAxesSelectorKey(modeId, currentModelAxes);
    myModelsBaselinesSelectorKey = selectorKey;
    myModelsBaselinesLoading = true;
    updateMyModelsControls();
    try {
      const baselines = await onAuthListGlobalModels();
      myModelsGlobalBaselines = baselines;
      renderMyModelsBaselinesSelect();
      if (!options?.silent) {
        if (baselines.length === 0) {
          setMyModelsActionStatus(
            `No global baselines found for mode "${modeId}".`,
            'neutral',
          );
        } else {
          setMyModelsActionStatus(
            `Loaded ${baselines.length} global baseline${baselines.length === 1 ? '' : 's'}.`,
            'success',
          );
        }
      }
    } catch (error) {
      if (!options?.silent) {
        setMyModelsActionStatus(
          toErrorMessage(error, 'Could not refresh global baselines.'),
          'error',
        );
      }
    } finally {
      myModelsBaselinesLoading = false;
      updateMyModelsControls();
    }
  };

  const formatAdminSummary = (state: MenuAuthState): string => {
    if (state.loading) {
      return 'Checking session...';
    }
    if (!state.authenticated || !state.user) {
      return 'Admin tools are locked.\nSign in with an admin account.';
    }
    if (!state.user.isAdmin) {
      return 'Admin tools are locked.\nThis account does not have admin access.';
    }
    return `Signed in as ${state.user.username}\nAdmin access: granted\nUse this panel for privileged training and data operations.`;
  };

  const formatAdminBaselineOption = (
    baseline: MenuAdminGlobalBaselineSummary,
  ): string => {
    const prefix = baseline.isDefault ? '[default] ' : '';
    const head = baseline.label || baseline.pipelineId || baseline.id;
    return `${prefix}${head}`;
  };

  const renderAdminBaselinesSelect = (): void => {
    adminBaselinesSelect.innerHTML = '';
    for (const baseline of adminGlobalBaselines) {
      const option = document.createElement('option');
      option.value = baseline.id;
      option.textContent = formatAdminBaselineOption(baseline);
      adminBaselinesSelect.appendChild(option);
    }
    if (adminGlobalBaselines.length > 0) {
      const activeId =
        adminSelectedBaselineId &&
        adminGlobalBaselines.some(
          (baseline) => baseline.id === adminSelectedBaselineId,
        )
          ? adminSelectedBaselineId
          : adminGlobalBaselines[0].id;
      adminBaselinesSelect.value = activeId;
      adminSelectedBaselineId = activeId;
    } else {
      adminSelectedBaselineId = null;
    }
  };

  const getSelectedAdminBaseline =
    (): MenuAdminGlobalBaselineSummary | null => {
      if (!adminSelectedBaselineId) return null;
      return (
        adminGlobalBaselines.find(
          (baseline) => baseline.id === adminSelectedBaselineId,
        ) ?? null
      );
    };

  const formatAdminBaselinesSummary = (isAdmin: boolean): string => {
    if (!isAdmin) {
      return `Admin account required.\nAxes: ${formatModelAxesCompact(currentModelAxes)}`;
    }
    const modeId = getLocalTrainingStats().currentModeId;
    if (adminBaselinesPending) {
      return `Loading baselines for mode "${modeId}"...\nAxes: ${formatModelAxesCompact(currentModelAxes)}`;
    }
    if (adminGlobalBaselines.length === 0) {
      return `No published baselines for mode "${modeId}".\nAxes: ${formatModelAxesCompact(currentModelAxes)}`;
    }
    const selected = getSelectedAdminBaseline();
    if (!selected) {
      return [
        `Published baselines: ${adminGlobalBaselines.length}`,
        `Axes: ${formatModelAxesCompact(currentModelAxes)}`,
      ].join('\n');
    }
    const updated =
      selected.updatedAtMs != null
        ? new Date(selected.updatedAtMs).toLocaleString()
        : 'n/a';
    return [
      `Published baselines: ${adminGlobalBaselines.length}`,
      `Selected: ${selected.label || selected.id}`,
      selected.pipelineId
        ? `Pipeline: ${selected.pipelineId}`
        : 'Pipeline: (none)',
      `Axes: ${formatModelAxesCompact(currentModelAxes)}`,
      `Size: ${formatApproxBytes(selected.sizeBytes)}`,
      `Updated: ${updated}`,
    ].join('\n');
  };

  const refreshAdminBaselines = async (options?: {
    silent?: boolean;
  }): Promise<void> => {
    const isAdmin =
      currentAuthState.authenticated &&
      currentAuthState.user != null &&
      currentAuthState.user.isAdmin;
    if (!isAdmin) {
      adminGlobalBaselines = [];
      adminSelectedBaselineId = null;
      adminBaselinesSelectorKey = null;
      renderAdminBaselinesSelect();
      updateAdminControls();
      return;
    }
    const modeId = getLocalTrainingStats().currentModeId;
    const selectorKey = buildModelAxesSelectorKey(modeId, currentModelAxes);
    adminBaselinesSelectorKey = selectorKey;
    adminBaselinesPending = true;
    updateAdminControls();
    try {
      const baselines = await onAdminListGlobalBaselines();
      adminGlobalBaselines = baselines;
      renderAdminBaselinesSelect();
      if (!options?.silent) {
        setAdminActionStatus(
          baselines.length === 0
            ? `No baselines found for mode "${modeId}".`
            : `Loaded ${baselines.length} baseline${baselines.length === 1 ? '' : 's'}.`,
          baselines.length === 0 ? 'neutral' : 'success',
        );
      }
    } catch (error) {
      if (!options?.silent) {
        setAdminActionStatus(
          toErrorMessage(error, 'Could not load global baselines.'),
          'error',
        );
      }
    } finally {
      adminBaselinesPending = false;
      updateAdminControls();
    }
  };

  const getAdminQueryLimit = (): number => {
    const raw = Number(adminLimitInput.value);
    if (!Number.isFinite(raw)) return 25;
    return Math.max(1, Math.min(200, Math.trunc(raw)));
  };

  const formatAdminDatasetSummary = (
    page: MenuAdminRecordingsPage | null,
  ): string => {
    if (!page) {
      return 'No query run yet.';
    }
    const next =
      page.page.nextCursor != null && page.page.nextCursor.length > 0
        ? 'available'
        : 'none';
    return [
      `Returned: ${page.page.returned}`,
      `Limit: ${page.page.limit}`,
      `Next page: ${next}`,
    ].join('\n');
  };

  const formatAdminRecordingOption = (
    recording: MenuAdminRecordingSummary,
  ): string => {
    const time = new Date(recording.startedAtMs).toLocaleTimeString();
    return `${recording.mode} · ${recording.samples} samples · ${time}`;
  };

  const renderAdminRecordingsSelect = (): void => {
    adminRecordingsSelect.innerHTML = '';
    for (const recording of adminCurrentRecordings) {
      const option = document.createElement('option');
      option.value = recording.id;
      option.textContent = formatAdminRecordingOption(recording);
      adminRecordingsSelect.appendChild(option);
    }
    if (adminCurrentRecordings.length > 0) {
      const activeId =
        adminSelectedRecordingId &&
        adminCurrentRecordings.some(
          (recording) => recording.id === adminSelectedRecordingId,
        )
          ? adminSelectedRecordingId
          : adminCurrentRecordings[0].id;
      adminRecordingsSelect.value = activeId;
      adminSelectedRecordingId = activeId;
    } else {
      adminSelectedRecordingId = null;
    }
  };

  const formatAdminRecordingPreview = (
    preview: MenuAdminRecordingPreview,
  ): string => {
    const lines = [
      `Loaded id: ${preview.id}`,
      `Session: ${preview.sessionId}`,
      `Mode: ${preview.modeId}`,
      `Build: ${preview.buildVersion}`,
      `Samples: ${preview.samples}`,
      `Duration: ${Math.max(0, Math.trunc(preview.durationMs / 1000))}s`,
      `Started: ${new Date(preview.startedAtMs).toLocaleTimeString()}`,
      preview.rewardPolicy
        ? `Reward policy: ${preview.rewardPolicy}`
        : 'Reward policy: (none)',
      preview.rewardPolicyId
        ? `Reward policy id: ${preview.rewardPolicyId}`
        : 'Reward policy id: (none)',
      preview.rewardProfileId
        ? `Reward profile id: ${preview.rewardProfileId}`
        : 'Reward profile id: (none)',
      preview.queuePolicyId
        ? `Queue policy id: ${preview.queuePolicyId}`
        : 'Queue policy id: (none)',
      preview.rewardKind
        ? `Reward kind: ${preview.rewardKind}`
        : 'Reward kind: (none)',
      preview.rewardGamma != null
        ? `Reward gamma: ${preview.rewardGamma.toFixed(3)}`
        : 'Reward gamma: (none)',
      preview.pipelineId
        ? `Pipeline id: ${preview.pipelineId}`
        : 'Pipeline id: (none)',
      preview.pipelineMode
        ? `Pipeline mode: ${preview.pipelineMode}`
        : 'Pipeline mode: (none)',
      preview.modelArchId
        ? `Model arch id: ${preview.modelArchId}`
        : 'Model arch id: (none)',
      preview.modelArch
        ? `Model arch: ${preview.modelArch}`
        : 'Model arch: (none)',
      preview.avgReward != null
        ? `Avg reward: ${preview.avgReward.toFixed(4)}`
        : 'Avg reward: (n/a)',
      preview.meanDeliberationMs != null
        ? `Mean deliberation: ${preview.meanDeliberationMs.toFixed(1)} ms`
        : 'Mean deliberation: (n/a)',
      preview.outcome ? `Outcome: ${preview.outcome}` : 'Outcome: (n/a)',
    ];
    return lines.join('\n');
  };

  const parseBotPieceSourceProfile = (
    value: string,
    fallback: 'bag7' | 'active_generator' = 'bag7',
  ): 'bag7' | 'active_generator' =>
    value === 'active_generator' ? 'active_generator' : fallback;

  const readOptionalSeedInput = (
    input: HTMLInputElement,
  ): number | undefined => {
    const raw = input.value.trim();
    if (!raw) return undefined;
    const value = Number(raw);
    if (!Number.isFinite(value)) return undefined;
    return Math.trunc(value);
  };

  const formatBotPolicyOption = (policy: MenuBotPolicySummary): string => {
    const pinPrefix = policy.isPinned ? '[pin] ' : '';
    const currentPrefix =
      policy.id === botLabCurrentPolicyId ? '[current] ' : '';
    return `${currentPrefix}${pinPrefix}v${policy.version} · ${policy.pipelineId} · ${new Date(policy.createdAtMs).toLocaleTimeString()}`;
  };

  const renderBotLabPoliciesSelect = (): void => {
    botLabPoliciesSelect.innerHTML = '';
    for (const policy of botLabPolicies) {
      const option = document.createElement('option');
      option.value = policy.id;
      option.textContent = formatBotPolicyOption(policy);
      botLabPoliciesSelect.appendChild(option);
    }
    if (botLabPolicies.length === 0) {
      botLabSelectedPolicyId = null;
      return;
    }
    const activeId =
      botLabSelectedPolicyId &&
      botLabPolicies.some((policy) => policy.id === botLabSelectedPolicyId)
        ? botLabSelectedPolicyId
        : botLabPolicies[0].id;
    botLabPoliciesSelect.value = activeId;
    botLabSelectedPolicyId = activeId;
  };

  const getSelectedBotLabPolicy = (): MenuBotPolicySummary | null => {
    if (!botLabSelectedPolicyId) return null;
    return (
      botLabPolicies.find((policy) => policy.id === botLabSelectedPolicyId) ??
      null
    );
  };

  const formatBotLabSummary = (isAdmin: boolean): string => {
    if (!isAdmin) {
      return `Bot Lab locked.\nSign in with an admin account.\nAxes: ${formatModelAxesCompact(currentModelAxes)}`;
    }
    const modeId = getLocalTrainingStats().currentModeId;
    const selected = getSelectedBotLabPolicy();
    const selectedLine = selected
      ? `Selected: ${selected.id} (v${selected.version}${selected.isPinned ? ', pinned' : ''})`
      : botLabCurrentPolicyId
        ? `Current: ${botLabCurrentPolicyId}`
        : 'Current: (none)';
    return [
      `Mode: ${modeId}`,
      `Axes: ${formatModelAxesCompact(currentModelAxes)}`,
      `Listed policies: ${botLabPolicies.length}`,
      selectedLine,
      `Next page: ${botLabPolicyCursor ? 'available' : 'none'}`,
      `GUI inspect: ${botLabGuiInspectRunning ? 'running' : 'stopped'}`,
    ].join('\n');
  };

  const updateBotLabControls = () => {
    const isAdmin =
      currentAuthState.authenticated &&
      currentAuthState.user != null &&
      currentAuthState.user.isAdmin;
    const busy =
      currentAuthState.loading || botLabActionPending || botLabListPending;
    const hasPolicies = botLabPolicies.length > 0;
    const hasSelected = botLabSelectedPolicyId != null;

    botLabButton.style.display = isAdmin ? 'block' : 'none';
    botLabSummary.textContent = formatBotLabSummary(isAdmin);
    botLabPolicyLabel.style.display = isAdmin ? 'block' : 'none';
    botLabPolicyButtonsTop.style.display = isAdmin ? 'flex' : 'none';
    botLabPoliciesSelect.style.display = isAdmin ? 'block' : 'none';
    botLabPolicyButtonsRow.style.display = isAdmin ? 'flex' : 'none';
    botLabPublishRow.style.display = isAdmin ? 'flex' : 'none';
    botLabPublishPolicyButton.style.display = isAdmin ? 'block' : 'none';
    botLabTrainLabel.style.display = isAdmin ? 'block' : 'none';
    botLabTrainControls.style.display = isAdmin ? 'grid' : 'none';
    botLabTrainButton.style.display = isAdmin ? 'block' : 'none';
    botLabHeadlessValidateButton.style.display = isAdmin ? 'block' : 'none';
    botLabGuiLabel.style.display = isAdmin ? 'block' : 'none';
    botLabGuiControls.style.display = isAdmin ? 'grid' : 'none';
    botLabGuiButtons.style.display = isAdmin ? 'flex' : 'none';
    botLabDataLabel.style.display = isAdmin ? 'block' : 'none';
    botLabDataControls.style.display = isAdmin ? 'grid' : 'none';
    botLabGenerateButton.style.display = isAdmin ? 'block' : 'none';
    botLabBenchmarkButton.style.display = isAdmin ? 'block' : 'none';
    botLabApplyBenchmarkArchButton.style.display = isAdmin ? 'block' : 'none';

    botLabFetchCurrentButton.disabled = !isAdmin || busy;
    botLabRefreshPoliciesButton.disabled = !isAdmin || busy;
    botLabPoliciesSelect.disabled = !isAdmin || busy || !hasPolicies;
    botLabPolicyNextButton.disabled =
      !isAdmin || busy || botLabPolicyCursor == null;
    botLabPolicyLoadButton.disabled = !isAdmin || busy || !hasSelected;
    botLabPolicySelectCurrentButton.disabled = !isAdmin || busy || !hasSelected;
    botLabPolicyPinButton.disabled = !isAdmin || busy || !hasSelected;
    botLabPolicyUnpinButton.disabled = !isAdmin || busy || !hasSelected;
    botLabPublishSourceSelect.disabled = !isAdmin || busy;
    botLabPublishPinToggle.disabled = !isAdmin || busy;
    botLabPublishCurrentToggle.disabled = !isAdmin || busy;
    botLabPublishPolicyButton.disabled = !isAdmin || busy;
    botLabEpisodesInput.disabled = !isAdmin || busy;
    botLabMaxPiecesInput.disabled = !isAdmin || busy;
    botLabSeedInput.disabled = !isAdmin || busy;
    botLabTrainSourceSelect.disabled = !isAdmin || busy;
    botLabTrainButton.disabled = !isAdmin || busy;
    botLabHeadlessValidateButton.disabled = !isAdmin || busy;
    botLabGuiApmInput.disabled = !isAdmin || busy;
    botLabGuiSeedInput.disabled = !isAdmin || busy;
    botLabGuiPiecesInput.disabled = !isAdmin || busy;
    botLabGuiSourceSelect.disabled = !isAdmin || busy;
    botLabStartGuiButton.disabled = !isAdmin || busy;
    botLabStopGuiButton.disabled = !isAdmin || busy || !botLabGuiInspectRunning;
    botLabGenerateSessionsInput.disabled = !isAdmin || busy;
    botLabGenerateSourceSelect.disabled = !isAdmin || busy;
    botLabGenerateButton.disabled = !isAdmin || busy;
    botLabBenchmarkButton.disabled = !isAdmin || busy;
    botLabApplyBenchmarkArchButton.disabled = !isAdmin || busy;

    const setVisualState = (button: HTMLButtonElement, enabled: boolean) => {
      button.style.opacity = enabled ? '1' : '0.65';
      button.style.cursor = enabled ? 'pointer' : 'default';
    };
    setVisualState(
      botLabFetchCurrentButton,
      !botLabFetchCurrentButton.disabled,
    );
    setVisualState(
      botLabRefreshPoliciesButton,
      !botLabRefreshPoliciesButton.disabled,
    );
    setVisualState(botLabPolicyNextButton, !botLabPolicyNextButton.disabled);
    setVisualState(botLabPolicyLoadButton, !botLabPolicyLoadButton.disabled);
    setVisualState(
      botLabPolicySelectCurrentButton,
      !botLabPolicySelectCurrentButton.disabled,
    );
    setVisualState(botLabPolicyPinButton, !botLabPolicyPinButton.disabled);
    setVisualState(botLabPolicyUnpinButton, !botLabPolicyUnpinButton.disabled);
    setVisualState(
      botLabPublishPolicyButton,
      !botLabPublishPolicyButton.disabled,
    );
    setVisualState(botLabTrainButton, !botLabTrainButton.disabled);
    setVisualState(
      botLabHeadlessValidateButton,
      !botLabHeadlessValidateButton.disabled,
    );
    setVisualState(botLabStartGuiButton, !botLabStartGuiButton.disabled);
    setVisualState(botLabStopGuiButton, !botLabStopGuiButton.disabled);
    setVisualState(botLabGenerateButton, !botLabGenerateButton.disabled);
    setVisualState(botLabBenchmarkButton, !botLabBenchmarkButton.disabled);
    setVisualState(
      botLabApplyBenchmarkArchButton,
      !botLabApplyBenchmarkArchButton.disabled,
    );
  };

  const formatMyModelsTrainingSummary = (
    stats: MenuLocalTrainingStats,
    preset: MenuLocalTrainingPreset,
  ): string => {
    const requiredSamples = Math.max(1, Math.trunc(preset.minSamples));
    const eligibilityLine =
      stats.currentModeAxesSamples >= requiredSamples
        ? 'Ready to train'
        : `Need ${requiredSamples - stats.currentModeAxesSamples} more mode+axes samples`;
    const lastSampleLine =
      stats.lastSampleAtMs == null
        ? 'Last sample: none yet'
        : `Last sample: ${new Date(stats.lastSampleAtMs).toLocaleTimeString()}`;
    return [
      `Mode: ${stats.currentModeId}`,
      `Client samples (mode+axes): ${stats.currentModeAxesSamples}`,
      `Client samples (mode): ${stats.currentModeSamples}`,
      `Client samples (total): ${stats.totalSamples}`,
      `Pipeline: ${preset.pipelineId}`,
      eligibilityLine,
      lastSampleLine,
    ].join('\n');
  };

  const formatMyModelsTrainingPresetSummary = (
    preset: MenuLocalTrainingPreset,
  ): string => {
    const holdoutPercent = Math.round(
      Math.max(0, Math.min(1, preset.evalGate.holdoutRatio)) * 100,
    );
    return [
      `Defaults (${preset.modeId}): epochs ${preset.trainDefaults.epochs}, lr ${formatLearningRate(preset.trainDefaults.learningRate)}, samples ${preset.trainDefaults.sampleLimit}, backend ${preset.trainDefaults.backendPreference}`,
      `Eval gate: holdout ${holdoutPercent}%, train>=${preset.evalGate.minTrainSamples}, holdout>=${preset.evalGate.minHoldoutSamples}, Δ>=${preset.evalGate.minObjectiveGain.toFixed(4)}`,
    ].join('\n');
  };

  const updateMyModelsControls = () => {
    const authenticated =
      currentAuthState.authenticated && currentAuthState.user != null;
    const trainingPreset = getLocalTrainingPreset();
    const presetKey = `${trainingPreset.pipelineId}:${trainingPreset.modeId}`;
    if (presetKey !== lastTrainingPresetKey) {
      myModelsEpochsInput.value = String(trainingPreset.trainDefaults.epochs);
      myModelsLearningRateInput.value = String(
        trainingPreset.trainDefaults.learningRate,
      );
      myModelsSampleLimitInput.value = String(
        trainingPreset.trainDefaults.sampleLimit,
      );
      myModelsBackendSelect.value =
        trainingPreset.trainDefaults.backendPreference;
      lastTrainingPresetKey = presetKey;
    }
    const trainingStats = getLocalTrainingStats();
    const currentModeId = trainingStats.currentModeId;
    const selectorKey = buildModelAxesSelectorKey(
      currentModeId,
      currentModelAxes,
    );
    const pendingAxes = readModelAxesFromControls();
    const axesDirty = !modelAxesEqual(pendingAxes, currentModelAxes);
    if (
      authenticated &&
      !modelActionPending &&
      !myModelsBaselinesLoading &&
      myModelsBaselinesSelectorKey !== selectorKey
    ) {
      void refreshMyModelsBaselines({ silent: true });
    }
    const hasTrainSamples =
      trainingStats.currentModeAxesSamples >=
      Math.max(1, Math.trunc(trainingPreset.minSamples));
    myModelsSummary.textContent = formatMyModelsSummary(currentAuthState);
    myModelsBaselinesSummary.textContent = formatMyModelsBaselinesSummary(
      authenticated,
      currentModeId,
      currentModelAxes,
    );
    myModelsTrainingSummary.textContent = formatMyModelsTrainingSummary(
      trainingStats,
      trainingPreset,
    );
    myModelsTrainingPresetSummary.textContent =
      formatMyModelsTrainingPresetSummary(trainingPreset);
    myModelsRecordingSummary.textContent = getTrajectoryUploadSummary();
    myModelsSignedOutHint.style.display = authenticated ? 'none' : 'block';
    myModelsActions.style.display = authenticated ? 'flex' : 'none';
    const busy = modelActionPending || currentAuthState.loading;
    myModelsLoadButton.disabled = !authenticated || busy;
    myModelsSaveButton.disabled = !authenticated || busy;
    myModelsArchSelect.disabled = busy;
    myModelsRewardProfileSelect.disabled = busy;
    myModelsQueuePolicySelect.disabled = busy;
    myModelsApplyAxesButton.disabled = busy || !axesDirty;
    myModelsBaselinesSelect.disabled =
      !authenticated ||
      busy ||
      myModelsBaselinesLoading ||
      myModelsGlobalBaselines.length === 0;
    myModelsBaselinesRefreshButton.disabled =
      !authenticated || busy || myModelsBaselinesLoading;
    myModelsResetToBaselineButton.disabled =
      !authenticated ||
      busy ||
      myModelsBaselinesLoading ||
      myModelsSelectedBaselineId == null;
    myModelsEpochsInput.disabled = !authenticated || busy;
    myModelsLearningRateInput.disabled = !authenticated || busy;
    myModelsSampleLimitInput.disabled = !authenticated || busy;
    myModelsBackendSelect.disabled = !authenticated || busy;
    myModelsTrainButton.disabled = busy || !hasTrainSamples;
    myModelsTrainButton.style.opacity = busy || !hasTrainSamples ? '0.65' : '1';
    myModelsTrainButton.style.cursor =
      busy || !hasTrainSamples ? 'default' : 'pointer';
    myModelsUploadTrajectoryButton.disabled = !authenticated || busy;
    myModelsUploadTrajectoryButton.style.opacity =
      !authenticated || busy ? '0.65' : '1';
    myModelsUploadTrajectoryButton.style.cursor =
      !authenticated || busy ? 'default' : 'pointer';
    myModelsApplyAxesButton.style.opacity = busy || !axesDirty ? '0.65' : '1';
    myModelsApplyAxesButton.style.cursor =
      busy || !axesDirty ? 'default' : 'pointer';
    myModelsBaselinesRefreshButton.style.opacity =
      !authenticated || busy || myModelsBaselinesLoading ? '0.65' : '1';
    myModelsBaselinesRefreshButton.style.cursor =
      !authenticated || busy || myModelsBaselinesLoading
        ? 'default'
        : 'pointer';
    myModelsResetToBaselineButton.style.opacity =
      !authenticated ||
      busy ||
      myModelsBaselinesLoading ||
      myModelsSelectedBaselineId == null
        ? '0.65'
        : '1';
    myModelsResetToBaselineButton.style.cursor =
      !authenticated ||
      busy ||
      myModelsBaselinesLoading ||
      myModelsSelectedBaselineId == null
        ? 'default'
        : 'pointer';
  };

  const updateAdminControls = () => {
    const isAdmin =
      currentAuthState.authenticated &&
      currentAuthState.user != null &&
      currentAuthState.user.isAdmin;
    const currentModeId = getLocalTrainingStats().currentModeId;
    if (
      isAdmin &&
      !adminActionPending &&
      !adminBaselinesPending &&
      adminBaselinesSelectorKey !==
        buildModelAxesSelectorKey(currentModeId, currentModelAxes)
    ) {
      void refreshAdminBaselines({ silent: true });
    }
    adminSummary.textContent = formatAdminSummary(currentAuthState);
    adminBaselinesSummary.textContent = formatAdminBaselinesSummary(isAdmin);
    adminDatasetSummary.textContent = formatAdminDatasetSummary(adminLastPage);
    adminButton.style.display = isAdmin ? 'block' : 'none';
    adminActions.style.display = isAdmin ? 'flex' : 'none';
    adminBaselinesLabel.style.display = isAdmin ? 'block' : 'none';
    adminBaselinesControls.style.display = isAdmin ? 'flex' : 'none';
    adminDatasetLabel.style.display = isAdmin ? 'block' : 'none';
    adminDatasetControls.style.display = isAdmin ? 'flex' : 'none';
    const busy = adminActionPending || currentAuthState.loading;
    const baselinesBusy = adminBaselinesPending || busy;
    const datasetBusy = adminDatasetPending || busy;
    adminManifestMinSamplesInput.disabled = !isAdmin || busy;
    adminManifestActorTypeSelect.disabled = !isAdmin || busy;
    adminPrepareManifestButton.disabled = !isAdmin || busy;
    adminTrainGlobalOneShotButton.disabled = !isAdmin || busy;
    adminPublishGlobalCandidateButton.disabled = !isAdmin || busy;
    adminWhoAmIButton.disabled = !isAdmin || busy;
    adminOpenRouteButton.disabled = busy;
    adminPublishBaselineButton.disabled = !isAdmin || busy;
    adminPublishLabelInput.disabled = !isAdmin || busy;
    adminPublishDefaultCheckbox.disabled = !isAdmin || busy;
    adminBaselinesRefreshButton.disabled = !isAdmin || baselinesBusy;
    adminBaselinesSelect.disabled =
      !isAdmin || baselinesBusy || adminGlobalBaselines.length === 0;
    adminBaselinesSetDefaultButton.disabled =
      !isAdmin || baselinesBusy || adminSelectedBaselineId == null;
    adminBaselinesRetireButton.disabled =
      !isAdmin || baselinesBusy || adminSelectedBaselineId == null;
    adminModeFilterInput.disabled = datasetBusy || !isAdmin;
    adminBuildFilterInput.disabled = datasetBusy || !isAdmin;
    adminLimitInput.disabled = datasetBusy || !isAdmin;
    adminQueryButton.disabled = datasetBusy || !isAdmin;
    adminNextPageButton.disabled =
      datasetBusy ||
      !isAdmin ||
      adminCurrentCursor == null ||
      adminCurrentCursor.length === 0;
    adminRecordingsSelect.disabled =
      datasetBusy || !isAdmin || adminCurrentRecordings.length === 0;
    adminLoadSelectedButton.disabled =
      datasetBusy || !isAdmin || adminSelectedRecordingId == null;
    adminWhoAmIButton.style.opacity = !isAdmin || busy ? '0.65' : '1';
    adminOpenRouteButton.style.opacity = busy ? '0.65' : '1';
    adminPublishBaselineButton.style.opacity = !isAdmin || busy ? '0.65' : '1';
    adminPrepareManifestButton.style.opacity = !isAdmin || busy ? '0.65' : '1';
    adminTrainGlobalOneShotButton.style.opacity =
      !isAdmin || busy ? '0.65' : '1';
    adminPublishGlobalCandidateButton.style.opacity =
      !isAdmin || busy ? '0.65' : '1';
    adminBaselinesRefreshButton.style.opacity =
      !isAdmin || baselinesBusy ? '0.65' : '1';
    adminBaselinesSetDefaultButton.style.opacity =
      !isAdmin || baselinesBusy || adminSelectedBaselineId == null
        ? '0.65'
        : '1';
    adminBaselinesRetireButton.style.opacity =
      !isAdmin || baselinesBusy || adminSelectedBaselineId == null
        ? '0.65'
        : '1';
    adminQueryButton.style.opacity = !isAdmin || datasetBusy ? '0.65' : '1';
    adminNextPageButton.style.opacity =
      !isAdmin || datasetBusy || !adminCurrentCursor ? '0.65' : '1';
    adminLoadSelectedButton.style.opacity =
      !isAdmin || datasetBusy || adminSelectedRecordingId == null
        ? '0.65'
        : '1';
    adminWhoAmIButton.style.cursor = !isAdmin || busy ? 'default' : 'pointer';
    adminOpenRouteButton.style.cursor = busy ? 'default' : 'pointer';
    adminPublishBaselineButton.style.cursor =
      !isAdmin || busy ? 'default' : 'pointer';
    adminPrepareManifestButton.style.cursor =
      !isAdmin || busy ? 'default' : 'pointer';
    adminTrainGlobalOneShotButton.style.cursor =
      !isAdmin || busy ? 'default' : 'pointer';
    adminPublishGlobalCandidateButton.style.cursor =
      !isAdmin || busy ? 'default' : 'pointer';
    adminBaselinesRefreshButton.style.cursor =
      !isAdmin || baselinesBusy ? 'default' : 'pointer';
    adminBaselinesSetDefaultButton.style.cursor =
      !isAdmin || baselinesBusy || adminSelectedBaselineId == null
        ? 'default'
        : 'pointer';
    adminBaselinesRetireButton.style.cursor =
      !isAdmin || baselinesBusy || adminSelectedBaselineId == null
        ? 'default'
        : 'pointer';
    adminQueryButton.style.cursor =
      !isAdmin || datasetBusy ? 'default' : 'pointer';
    adminNextPageButton.style.cursor =
      !isAdmin || datasetBusy || !adminCurrentCursor ? 'default' : 'pointer';
    adminLoadSelectedButton.style.cursor =
      !isAdmin || datasetBusy || adminSelectedRecordingId == null
        ? 'default'
        : 'pointer';
  };

  const updateAccountControls = () => {
    accountSummary.textContent = formatAccountSummary(currentAuthState);
    const authenticated =
      currentAuthState.authenticated && currentAuthState.user != null;
    if (!authenticated && signedOutStage === 'reset' && !hiddenResetToken) {
      setSignedOutStage('email');
    }
    syncSignedOutStageUi();
    const busy = authActionPending || currentAuthState.loading;
    for (const input of [
      accountEmailInput,
      accountUsernameInput,
      accountPasswordInput,
      accountResetPasswordInput,
    ]) {
      input.disabled = busy;
    }
    for (const button of [
      accountRefreshButton,
      accountGoogleButton,
      accountDiscordButton,
      accountLogoutButton,
      accountVerifyButton,
      accountEmailContinueButton,
      accountEmailSubmitButton,
      accountResetSubmitButton,
      accountForgotLink,
      accountSwitchModeLink,
      accountBackToEmailLink,
      accountResetBackLink,
    ]) {
      button.disabled = busy;
    }

    accountSignedOutActions.style.display = authenticated ? 'none' : 'flex';
    accountSignedInActions.style.display = authenticated ? 'flex' : 'none';
    accountVerifyButton.style.display =
      authenticated &&
      currentAuthState.user?.email &&
      currentAuthState.user.emailVerifiedAtMs == null
        ? 'block'
        : 'none';
    updateMyModelsControls();
    updateAdminControls();
    updateBotLabControls();
  };

  const setAuthState = (state: MenuAuthState) => {
    currentAuthState = state;
    updateAccountControls();
  };

  accountRefreshButton.addEventListener('click', async () => {
    if (authActionPending) return;
    authActionPending = true;
    setAccountActionStatus('Refreshing session...');
    updateAccountControls();
    try {
      await onAuthRefresh();
      setAccountActionStatus('');
    } catch {
      setAccountActionStatus('Could not refresh session.', 'error');
    } finally {
      authActionPending = false;
      updateAccountControls();
    }
  });

  accountEmailContinueButton.addEventListener('click', () => {
    if (authActionPending) return;
    const email = readField(accountEmailInput);
    if (!email) {
      setAccountActionStatus('Enter your email first.', 'error');
      return;
    }
    setSignedOutStage('login');
    setAccountActionStatus('');
    updateAccountControls();
  });

  accountEmailInput.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter') return;
    event.preventDefault();
    accountEmailContinueButton.click();
  });

  accountSwitchModeLink.addEventListener('click', () => {
    if (authActionPending) return;
    setSignedOutStage(signedOutStage === 'signup' ? 'login' : 'signup');
    setAccountActionStatus('');
    updateAccountControls();
  });

  accountBackToEmailLink.addEventListener('click', () => {
    if (authActionPending) return;
    setSignedOutStage('email');
    accountPasswordInput.value = '';
    accountUsernameInput.value = '';
    setAccountActionStatus('');
    updateAccountControls();
  });

  accountResetBackLink.addEventListener('click', () => {
    if (authActionPending) return;
    setSignedOutStage('email');
    setAccountActionStatus('');
    updateAccountControls();
  });

  accountGoogleButton.addEventListener('click', () => {
    setAccountActionStatus('Redirecting to Google...');
    onAuthStartOAuth('google');
  });

  accountDiscordButton.addEventListener('click', () => {
    setAccountActionStatus('Redirecting to Discord...');
    onAuthStartOAuth('discord');
  });

  accountVerifyButton.addEventListener('click', async () => {
    if (authActionPending) return;
    authActionPending = true;
    setAccountActionStatus('Sending verification email...');
    updateAccountControls();
    try {
      const result = await onAuthSendVerifyEmail();
      setAccountActionStatus(
        result.alreadyVerified
          ? 'Email is already verified.'
          : 'Verification email sent.',
        'success',
      );
      await onAuthRefresh().catch(() => {});
    } catch {
      setAccountActionStatus('Could not send verification email.', 'error');
    } finally {
      authActionPending = false;
      updateAccountControls();
    }
  });

  myModelsLoadButton.addEventListener('click', async () => {
    if (modelActionPending) return;
    modelActionPending = true;
    setMyModelsActionStatus('Loading cloud model...');
    updateMyModelsControls();
    try {
      const message = await onAuthLoadCurrentModel();
      setMyModelsActionStatus(message || 'Cloud model loaded.', 'success');
    } catch (error) {
      setMyModelsActionStatus(
        toErrorMessage(error, 'Could not load cloud model.'),
        'error',
      );
    } finally {
      modelActionPending = false;
      updateMyModelsControls();
    }
  });

  myModelsSaveButton.addEventListener('click', async () => {
    if (modelActionPending) return;
    modelActionPending = true;
    setMyModelsActionStatus('Saving current model...');
    updateMyModelsControls();
    try {
      const message = await onAuthSaveCurrentModel();
      setMyModelsActionStatus(message || 'Current model saved.', 'success');
    } catch (error) {
      setMyModelsActionStatus(
        toErrorMessage(error, 'Could not save current model.'),
        'error',
      );
    } finally {
      modelActionPending = false;
      updateMyModelsControls();
    }
  });

  myModelsApplyAxesButton.addEventListener('click', async () => {
    if (modelActionPending) return;
    const nextAxes = readModelAxesFromControls();
    if (modelAxesEqual(nextAxes, currentModelAxes)) {
      setMyModelsActionStatus('Model axes unchanged.');
      updateMyModelsControls();
      return;
    }
    modelActionPending = true;
    setMyModelsActionStatus(
      `Applying model axes (${formatModelAxesCompact(nextAxes)})...`,
    );
    updateMyModelsControls();
    try {
      const message = await onModelAxesChange(nextAxes);
      currentModelAxes = getModelAxes();
      syncModelAxesControls();
      myModelsBaselinesSelectorKey = null;
      adminBaselinesSelectorKey = null;
      await refreshMyModelsBaselines({ silent: true });
      await refreshAdminBaselines({ silent: true });
      setMyModelsActionStatus(message, 'success');
    } catch (error) {
      currentModelAxes = getModelAxes();
      syncModelAxesControls();
      setMyModelsActionStatus(
        toErrorMessage(error, 'Could not apply model axes.'),
        'error',
      );
    } finally {
      modelActionPending = false;
      updateMyModelsControls();
    }
  });

  myModelsBaselinesSelect.addEventListener('change', () => {
    const value = myModelsBaselinesSelect.value.trim();
    myModelsSelectedBaselineId = value.length > 0 ? value : null;
    updateMyModelsControls();
  });

  myModelsBaselinesRefreshButton.addEventListener('click', async () => {
    if (modelActionPending) return;
    modelActionPending = true;
    setMyModelsActionStatus('Refreshing global baselines...');
    updateMyModelsControls();
    try {
      await refreshMyModelsBaselines();
    } finally {
      modelActionPending = false;
      updateMyModelsControls();
    }
  });

  myModelsResetToBaselineButton.addEventListener('click', async () => {
    if (modelActionPending) return;
    if (!myModelsSelectedBaselineId) {
      setMyModelsActionStatus('Select a global baseline first.', 'error');
      return;
    }
    modelActionPending = true;
    setMyModelsActionStatus('Resetting to selected baseline...');
    updateMyModelsControls();
    try {
      const message = await onAuthResetCurrentModelToGlobal(
        myModelsSelectedBaselineId,
      );
      setMyModelsActionStatus(message, 'success');
    } catch (error) {
      setMyModelsActionStatus(
        toErrorMessage(error, 'Could not reset to selected baseline.'),
        'error',
      );
    } finally {
      modelActionPending = false;
      updateMyModelsControls();
    }
  });

  myModelsTrainButton.addEventListener('click', async () => {
    if (modelActionPending) return;
    modelActionPending = true;
    setMyModelsActionStatus('Training on local client samples...');
    updateMyModelsControls();
    try {
      const backendPreference = myModelsBackendSelect.value
        .trim()
        .toLowerCase();
      const result = await onRunLocalBiasTraining({
        epochs: parsePositiveIntInput(myModelsEpochsInput),
        learningRate: parsePositiveFloatInput(myModelsLearningRateInput),
        sampleLimit: parsePositiveIntInput(myModelsSampleLimitInput),
        backendPreference:
          backendPreference === 'webgl' || backendPreference === 'cpu'
            ? backendPreference
            : 'auto',
      });
      const lossSuffix =
        result.finalLoss != null
          ? `\nFinal loss: ${result.finalLoss.toExponential(3)}`
          : '';
      setMyModelsActionStatus(
        `${result.message}\nSamples used: ${result.samplesUsed}${lossSuffix}`,
        result.ok ? 'success' : 'error',
      );
    } catch (error) {
      setMyModelsActionStatus(
        toErrorMessage(error, 'Could not run local training.'),
        'error',
      );
    } finally {
      modelActionPending = false;
      updateMyModelsControls();
    }
  });

  myModelsUploadTrajectoryButton.addEventListener('click', async () => {
    if (modelActionPending) return;
    modelActionPending = true;
    setMyModelsActionStatus('Uploading trajectory recording...');
    updateMyModelsControls();
    try {
      const message = await onUploadLatestTrajectory();
      setMyModelsActionStatus(message, 'success');
    } catch (error) {
      setMyModelsActionStatus(
        toErrorMessage(error, 'Could not upload trajectory recording.'),
        'error',
      );
    } finally {
      modelActionPending = false;
      updateMyModelsControls();
    }
  });

  adminWhoAmIButton.addEventListener('click', async () => {
    if (adminActionPending) return;
    adminActionPending = true;
    setAdminActionStatus('Checking admin API access...');
    updateAdminControls();
    try {
      const response = await fetch('/api/admin/whoami', {
        method: 'GET',
        credentials: 'include',
        cache: 'no-store',
      });
      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as {
          error?: unknown;
        } | null;
        const message =
          typeof payload?.error === 'string' && payload.error.trim()
            ? payload.error
            : `Request failed (${response.status}).`;
        throw new Error(message);
      }
      const payload = (await response.json().catch(() => null)) as {
        admin?: { username?: unknown; email?: unknown } | null;
      } | null;
      const username =
        typeof payload?.admin?.username === 'string'
          ? payload.admin.username
          : 'admin';
      const email =
        typeof payload?.admin?.email === 'string' ? payload.admin.email : '';
      const suffix = email ? ` (${email})` : '';
      setAdminActionStatus(
        `Admin API access confirmed: ${username}${suffix}.`,
        'success',
      );
    } catch (error) {
      setAdminActionStatus(
        toErrorMessage(error, 'Admin API access check failed.'),
        'error',
      );
    } finally {
      adminActionPending = false;
      updateAdminControls();
    }
  });

  adminPublishBaselineButton.addEventListener('click', async () => {
    if (adminActionPending) return;
    adminActionPending = true;
    setAdminActionStatus('Publishing current model as global baseline...');
    updateAdminControls();
    try {
      const message = await onAdminPublishCurrentModelBaseline({
        label: adminPublishLabelInput.value.trim(),
        setDefault: adminPublishDefaultCheckbox.checked,
      });
      setAdminActionStatus(message, 'success');
      await refreshAdminBaselines({ silent: true });
    } catch (error) {
      setAdminActionStatus(
        toErrorMessage(error, 'Global baseline publish failed.'),
        'error',
      );
    } finally {
      adminActionPending = false;
      updateAdminControls();
    }
  });

  adminPrepareManifestButton.addEventListener('click', async () => {
    if (adminActionPending) return;
    adminActionPending = true;
    setAdminActionStatus('Preparing training manifest...');
    updateAdminControls();
    try {
      const actorType = adminManifestActorTypeSelect.value.trim();
      const message = await onAdminPrepareManifest({
        ...(adminModeFilterInput.value.trim()
          ? { mode: adminModeFilterInput.value.trim() }
          : {}),
        ...(adminBuildFilterInput.value.trim()
          ? { build: adminBuildFilterInput.value.trim() }
          : {}),
        limit: getAdminQueryLimit(),
        minSamples: parsePositiveIntInput(adminManifestMinSamplesInput),
        actorType:
          actorType === 'human' || actorType === 'bot' ? actorType : '',
      });
      setAdminActionStatus(message, 'success');
    } catch (error) {
      setAdminActionStatus(
        toErrorMessage(error, 'Could not prepare training manifest.'),
        'error',
      );
    } finally {
      adminActionPending = false;
      updateAdminControls();
    }
  });

  adminTrainGlobalOneShotButton.addEventListener('click', async () => {
    if (adminActionPending) return;
    adminActionPending = true;
    setAdminActionStatus('Running global one-shot training...');
    updateAdminControls();
    try {
      const message = await onAdminTrainGlobalOneShot();
      setAdminActionStatus(message, 'success');
    } catch (error) {
      setAdminActionStatus(
        toErrorMessage(error, 'Could not run global training.'),
        'error',
      );
    } finally {
      adminActionPending = false;
      updateAdminControls();
    }
  });

  adminPublishGlobalCandidateButton.addEventListener('click', async () => {
    if (adminActionPending) return;
    adminActionPending = true;
    setAdminActionStatus('Publishing global candidate...');
    updateAdminControls();
    try {
      const message = await onAdminPublishGlobalCandidate();
      setAdminActionStatus(message, 'success');
      await refreshAdminBaselines({ silent: true });
    } catch (error) {
      setAdminActionStatus(
        toErrorMessage(error, 'Could not publish global candidate.'),
        'error',
      );
    } finally {
      adminActionPending = false;
      updateAdminControls();
    }
  });

  adminBaselinesSelect.addEventListener('change', () => {
    const value = adminBaselinesSelect.value.trim();
    adminSelectedBaselineId = value.length > 0 ? value : null;
    updateAdminControls();
  });

  adminBaselinesRefreshButton.addEventListener('click', async () => {
    if (adminBaselinesPending) return;
    setAdminActionStatus('Refreshing global baselines...');
    await refreshAdminBaselines();
  });

  adminBaselinesSetDefaultButton.addEventListener('click', async () => {
    if (adminBaselinesPending) return;
    if (!adminSelectedBaselineId) {
      setAdminActionStatus('Select a baseline first.', 'error');
      return;
    }
    adminBaselinesPending = true;
    setAdminActionStatus('Setting default baseline...');
    updateAdminControls();
    try {
      const message = await onAdminSetGlobalBaselineDefault(
        adminSelectedBaselineId,
      );
      setAdminActionStatus(message, 'success');
      await refreshAdminBaselines({ silent: true });
    } catch (error) {
      setAdminActionStatus(
        toErrorMessage(error, 'Could not set default baseline.'),
        'error',
      );
    } finally {
      adminBaselinesPending = false;
      updateAdminControls();
    }
  });

  adminBaselinesRetireButton.addEventListener('click', async () => {
    if (adminBaselinesPending) return;
    if (!adminSelectedBaselineId) {
      setAdminActionStatus('Select a baseline first.', 'error');
      return;
    }
    adminBaselinesPending = true;
    setAdminActionStatus('Retiring selected baseline...');
    updateAdminControls();
    try {
      const message = await onAdminRetireGlobalBaseline(
        adminSelectedBaselineId,
      );
      setAdminActionStatus(message, 'success');
      await refreshAdminBaselines({ silent: true });
    } catch (error) {
      setAdminActionStatus(
        toErrorMessage(error, 'Could not retire baseline.'),
        'error',
      );
    } finally {
      adminBaselinesPending = false;
      updateAdminControls();
    }
  });

  adminOpenRouteButton.addEventListener('click', () => {
    window.location.assign('/admin');
  });

  adminRecordingsSelect.addEventListener('change', () => {
    const value = adminRecordingsSelect.value.trim();
    adminSelectedRecordingId = value.length > 0 ? value : null;
    updateAdminControls();
  });

  const runAdminRecordingsQuery = async (options: {
    useNextCursor: boolean;
  }): Promise<void> => {
    if (adminDatasetPending) return;
    const isAdmin =
      currentAuthState.authenticated &&
      currentAuthState.user != null &&
      currentAuthState.user.isAdmin;
    if (!isAdmin) {
      setAdminActionStatus('Admin account required.', 'error');
      return;
    }
    const mode = adminModeFilterInput.value.trim();
    const build = adminBuildFilterInput.value.trim();
    const limit = getAdminQueryLimit();
    adminDatasetPending = true;
    setAdminActionStatus(
      options.useNextCursor
        ? 'Loading next recordings page...'
        : 'Loading recordings...',
    );
    updateAdminControls();
    try {
      const page = await onAdminListRecordings({
        ...(mode ? { mode } : {}),
        ...(build ? { build } : {}),
        limit,
        ...(options.useNextCursor && adminCurrentCursor
          ? { cursor: adminCurrentCursor }
          : {}),
      });
      adminLastPage = page;
      adminCurrentRecordings = page.recordings;
      adminCurrentCursor = page.page.nextCursor;
      renderAdminRecordingsSelect();
      adminDatasetSummary.textContent = formatAdminDatasetSummary(page);
      setAdminActionStatus(
        `Loaded ${page.page.returned} recordings.`,
        'success',
      );
    } catch (error) {
      setAdminActionStatus(
        toErrorMessage(error, 'Could not load recordings list.'),
        'error',
      );
    } finally {
      adminDatasetPending = false;
      updateAdminControls();
    }
  };

  adminQueryButton.addEventListener('click', () => {
    adminCurrentCursor = null;
    void runAdminRecordingsQuery({ useNextCursor: false });
  });

  adminNextPageButton.addEventListener('click', () => {
    void runAdminRecordingsQuery({ useNextCursor: true });
  });

  adminLoadSelectedButton.addEventListener('click', async () => {
    if (adminDatasetPending) return;
    if (!adminSelectedRecordingId) {
      setAdminActionStatus('Select a recording first.', 'error');
      return;
    }
    adminDatasetPending = true;
    setAdminActionStatus('Loading selected recording...');
    updateAdminControls();
    try {
      const preview = await onAdminLoadRecording(adminSelectedRecordingId);
      adminLoadedSummary.textContent = formatAdminRecordingPreview(preview);
      setAdminActionStatus('Recording loaded.', 'success');
    } catch (error) {
      setAdminActionStatus(
        toErrorMessage(error, 'Could not load selected recording.'),
        'error',
      );
    } finally {
      adminDatasetPending = false;
      updateAdminControls();
    }
  });

  botLabPoliciesSelect.addEventListener('change', () => {
    const value = botLabPoliciesSelect.value.trim();
    botLabSelectedPolicyId = value.length > 0 ? value : null;
    updateBotLabControls();
  });

  const loadBotLabPolicies = async (options?: {
    useNextCursor?: boolean;
    silent?: boolean;
  }): Promise<void> => {
    const isAdmin =
      currentAuthState.authenticated &&
      currentAuthState.user != null &&
      currentAuthState.user.isAdmin;
    if (!isAdmin) {
      setBotLabActionStatus('Admin account required.', 'error');
      return;
    }
    if (botLabListPending) return;
    botLabListPending = true;
    setBotLabActionStatus(
      options?.useNextCursor
        ? 'Loading next bot policy page...'
        : 'Loading bot policies...',
    );
    updateBotLabControls();
    try {
      const page = await onBotLabListPolicies({
        limit: 12,
        cursor: options?.useNextCursor ? botLabPolicyCursor : null,
      });
      botLabPolicies = page.policies;
      botLabPolicyCursor = page.page.nextCursor;
      renderBotLabPoliciesSelect();
      if (!options?.silent) {
        setBotLabActionStatus(
          `Loaded ${page.page.returned} bot polic${page.page.returned === 1 ? 'y' : 'ies'}.`,
          'success',
        );
      }
    } catch (error) {
      setBotLabActionStatus(
        toErrorMessage(error, 'Could not load bot policies.'),
        'error',
      );
    } finally {
      botLabListPending = false;
      updateBotLabControls();
    }
  };

  botLabFetchCurrentButton.addEventListener('click', async () => {
    if (botLabActionPending) return;
    botLabActionPending = true;
    setBotLabActionStatus('Loading current bot policy...');
    updateBotLabControls();
    try {
      const message = await onBotLabFetchCurrentPolicy();
      const currentMatch = /Loaded current bot policy:\s*([a-z0-9-]{36})/i.exec(
        message,
      );
      botLabCurrentPolicyId = currentMatch?.[1] ?? null;
      setBotLabActionStatus(message, 'success');
      await loadBotLabPolicies({ silent: true });
    } catch (error) {
      setBotLabActionStatus(
        toErrorMessage(error, 'Could not load current bot policy.'),
        'error',
      );
    } finally {
      botLabActionPending = false;
      updateBotLabControls();
    }
  });

  botLabRefreshPoliciesButton.addEventListener('click', () => {
    botLabPolicyCursor = null;
    void loadBotLabPolicies();
  });

  botLabPolicyNextButton.addEventListener('click', () => {
    void loadBotLabPolicies({ useNextCursor: true });
  });

  botLabPolicyLoadButton.addEventListener('click', async () => {
    if (botLabActionPending) return;
    if (!botLabSelectedPolicyId) {
      setBotLabActionStatus('Select a policy first.', 'error');
      return;
    }
    botLabActionPending = true;
    setBotLabActionStatus('Loading selected bot policy...');
    updateBotLabControls();
    try {
      const message = await onBotLabLoadPolicyById(botLabSelectedPolicyId);
      setBotLabActionStatus(message, 'success');
    } catch (error) {
      setBotLabActionStatus(
        toErrorMessage(error, 'Could not load selected policy.'),
        'error',
      );
    } finally {
      botLabActionPending = false;
      updateBotLabControls();
    }
  });

  botLabPolicySelectCurrentButton.addEventListener('click', async () => {
    if (botLabActionPending) return;
    if (!botLabSelectedPolicyId) {
      setBotLabActionStatus('Select a policy first.', 'error');
      return;
    }
    botLabActionPending = true;
    setBotLabActionStatus('Setting current bot policy...');
    updateBotLabControls();
    try {
      const message = await onBotLabSelectCurrentPolicy(botLabSelectedPolicyId);
      botLabCurrentPolicyId = botLabSelectedPolicyId;
      setBotLabActionStatus(message, 'success');
      renderBotLabPoliciesSelect();
    } catch (error) {
      setBotLabActionStatus(
        toErrorMessage(error, 'Could not set current policy.'),
        'error',
      );
    } finally {
      botLabActionPending = false;
      updateBotLabControls();
    }
  });

  botLabPolicyPinButton.addEventListener('click', async () => {
    if (botLabActionPending) return;
    if (!botLabSelectedPolicyId) {
      setBotLabActionStatus('Select a policy first.', 'error');
      return;
    }
    botLabActionPending = true;
    setBotLabActionStatus('Pinning policy...');
    updateBotLabControls();
    try {
      const message = await onBotLabPinPolicy(botLabSelectedPolicyId);
      setBotLabActionStatus(message, 'success');
      await loadBotLabPolicies({ silent: true });
    } catch (error) {
      setBotLabActionStatus(
        toErrorMessage(error, 'Could not pin policy.'),
        'error',
      );
    } finally {
      botLabActionPending = false;
      updateBotLabControls();
    }
  });

  botLabPolicyUnpinButton.addEventListener('click', async () => {
    if (botLabActionPending) return;
    if (!botLabSelectedPolicyId) {
      setBotLabActionStatus('Select a policy first.', 'error');
      return;
    }
    botLabActionPending = true;
    setBotLabActionStatus('Unpinning policy...');
    updateBotLabControls();
    try {
      const message = await onBotLabUnpinPolicy(botLabSelectedPolicyId);
      setBotLabActionStatus(message, 'success');
      await loadBotLabPolicies({ silent: true });
    } catch (error) {
      setBotLabActionStatus(
        toErrorMessage(error, 'Could not unpin policy.'),
        'error',
      );
    } finally {
      botLabActionPending = false;
      updateBotLabControls();
    }
  });

  botLabPublishPolicyButton.addEventListener('click', async () => {
    if (botLabActionPending) return;
    botLabActionPending = true;
    setBotLabActionStatus('Publishing loaded policy...');
    updateBotLabControls();
    try {
      const message = await onBotLabPublishPolicy({
        pin: botLabPublishPinToggle.checked,
        setCurrent: botLabPublishCurrentToggle.checked,
        pieceSourceProfile: parseBotPieceSourceProfile(
          botLabPublishSourceSelect.value,
          'bag7',
        ),
      });
      if (botLabPublishCurrentToggle.checked && botLabSelectedPolicyId) {
        botLabCurrentPolicyId = botLabSelectedPolicyId;
      }
      setBotLabActionStatus(message, 'success');
      await loadBotLabPolicies({ silent: true });
    } catch (error) {
      setBotLabActionStatus(
        toErrorMessage(error, 'Could not publish bot policy.'),
        'error',
      );
    } finally {
      botLabActionPending = false;
      updateBotLabControls();
    }
  });

  botLabTrainButton.addEventListener('click', async () => {
    if (botLabActionPending) return;
    botLabActionPending = true;
    setBotLabActionStatus('Training bot policy...');
    updateBotLabControls();
    try {
      const message = await onBotLabTrainPolicyOneShot({
        episodes: parsePositiveIntInput(botLabEpisodesInput),
        maxPiecesPerEpisode: parsePositiveIntInput(botLabMaxPiecesInput),
        seed: readOptionalSeedInput(botLabSeedInput),
        pieceSourceProfile: parseBotPieceSourceProfile(
          botLabTrainSourceSelect.value,
          'bag7',
        ),
      });
      setBotLabActionStatus(message, 'success');
    } catch (error) {
      setBotLabActionStatus(
        toErrorMessage(error, 'Could not train bot policy.'),
        'error',
      );
    } finally {
      botLabActionPending = false;
      updateBotLabControls();
    }
  });

  botLabHeadlessValidateButton.addEventListener('click', async () => {
    if (botLabActionPending) return;
    botLabActionPending = true;
    setBotLabActionStatus('Running headless 10k validation...');
    updateBotLabControls();
    try {
      const message = await onBotLabRunHeadlessValidate({
        maxPieces: 10_000,
        seed: readOptionalSeedInput(botLabSeedInput),
        pieceSourceProfile: parseBotPieceSourceProfile(
          botLabTrainSourceSelect.value,
          'bag7',
        ),
      });
      setBotLabActionStatus(message, 'success');
    } catch (error) {
      setBotLabActionStatus(
        toErrorMessage(error, 'Could not run headless validation.'),
        'error',
      );
    } finally {
      botLabActionPending = false;
      updateBotLabControls();
    }
  });

  botLabStartGuiButton.addEventListener('click', async () => {
    if (botLabActionPending) return;
    botLabActionPending = true;
    setBotLabActionStatus('Starting GUI inspect...');
    updateBotLabControls();
    try {
      const message = await onBotLabStartGuiInspect({
        apmInput: parsePositiveIntInput(botLabGuiApmInput),
        seed: readOptionalSeedInput(botLabGuiSeedInput),
        pieces: parsePositiveIntInput(botLabGuiPiecesInput),
        pieceSourceProfile: parseBotPieceSourceProfile(
          botLabGuiSourceSelect.value,
          'bag7',
        ),
      });
      botLabGuiInspectRunning = true;
      setBotLabActionStatus(message, 'success');
    } catch (error) {
      botLabGuiInspectRunning = false;
      setBotLabActionStatus(
        toErrorMessage(error, 'Could not start GUI inspect.'),
        'error',
      );
    } finally {
      botLabActionPending = false;
      updateBotLabControls();
    }
  });

  botLabStopGuiButton.addEventListener('click', async () => {
    if (botLabActionPending) return;
    botLabActionPending = true;
    setBotLabActionStatus('Stopping GUI inspect...');
    updateBotLabControls();
    try {
      const message = await onBotLabStopGuiInspect();
      botLabGuiInspectRunning = false;
      setBotLabActionStatus(message, 'success');
    } catch (error) {
      setBotLabActionStatus(
        toErrorMessage(error, 'Could not stop GUI inspect.'),
        'error',
      );
    } finally {
      botLabActionPending = false;
      updateBotLabControls();
    }
  });

  botLabGenerateButton.addEventListener('click', async () => {
    if (botLabActionPending) return;
    botLabActionPending = true;
    setBotLabActionStatus('Generating bot recordings...');
    updateBotLabControls();
    try {
      const message = await onBotLabGenerateRecordings({
        sessions: parsePositiveIntInput(botLabGenerateSessionsInput),
        maxPiecesPerEpisode: parsePositiveIntInput(botLabMaxPiecesInput),
        pieceSourceProfile: parseBotPieceSourceProfile(
          botLabGenerateSourceSelect.value,
          'active_generator',
        ),
      });
      setBotLabActionStatus(message, 'success');
    } catch (error) {
      setBotLabActionStatus(
        toErrorMessage(error, 'Could not generate bot recordings.'),
        'error',
      );
    } finally {
      botLabActionPending = false;
      updateBotLabControls();
    }
  });

  botLabBenchmarkButton.addEventListener('click', async () => {
    if (botLabActionPending) return;
    botLabActionPending = true;
    setBotLabActionStatus('Running bot benchmark...');
    updateBotLabControls();
    try {
      const message = await onBotLabRunBenchmark({
        episodes: parsePositiveIntInput(botLabEpisodesInput),
        maxPiecesPerEpisode: parsePositiveIntInput(botLabMaxPiecesInput),
        pieceSourceProfile: parseBotPieceSourceProfile(
          botLabGenerateSourceSelect.value,
          'bag7',
        ),
      });
      setBotLabActionStatus(message, 'success');
    } catch (error) {
      setBotLabActionStatus(
        toErrorMessage(error, 'Could not run bot benchmark.'),
        'error',
      );
    } finally {
      botLabActionPending = false;
      updateBotLabControls();
    }
  });

  botLabApplyBenchmarkArchButton.addEventListener('click', async () => {
    if (botLabActionPending) return;
    botLabActionPending = true;
    setBotLabActionStatus('Applying benchmark architecture recommendation...');
    updateBotLabControls();
    try {
      const message = await onAdminApplyBenchmarkSuggestedArch();
      setBotLabActionStatus(message, 'success');
    } catch (error) {
      setBotLabActionStatus(
        toErrorMessage(error, 'Could not apply benchmark recommendation.'),
        'error',
      );
    } finally {
      botLabActionPending = false;
      updateBotLabControls();
    }
  });

  accountLogoutButton.addEventListener('click', async () => {
    if (authActionPending) return;
    authActionPending = true;
    setAccountActionStatus('Signing out...');
    updateAccountControls();
    try {
      await onAuthLogout();
      setAccountActionStatus('Signed out.', 'success');
    } catch {
      setAccountActionStatus('Could not sign out.', 'error');
    } finally {
      authActionPending = false;
      updateAccountControls();
    }
  });

  accountEmailSubmitButton.addEventListener('click', async () => {
    if (authActionPending) return;
    const email = readField(accountEmailInput);
    const password = accountPasswordInput.value;
    const username = readField(accountUsernameInput);
    if (!email || !password) {
      setAccountActionStatus('Email and password are required.', 'error');
      return;
    }

    const isSignup = signedOutStage === 'signup';
    authActionPending = true;
    setAccountActionStatus(isSignup ? 'Creating account...' : 'Signing in...');
    updateAccountControls();
    try {
      if (isSignup) {
        await onAuthEmailSignup({
          email,
          password,
          ...(username ? { username } : {}),
        });
      } else {
        await onAuthEmailLogin({ email, password });
      }
      setAccountActionStatus('Signed in.', 'success');
      accountPasswordInput.value = '';
      accountUsernameInput.value = '';
    } catch (error) {
      setAccountActionStatus(
        toErrorMessage(
          error,
          isSignup ? 'Could not create account.' : 'Could not sign in.',
        ),
        'error',
      );
    } finally {
      authActionPending = false;
      updateAccountControls();
    }
  });

  accountPasswordInput.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter') return;
    event.preventDefault();
    accountEmailSubmitButton.click();
  });

  accountForgotLink.addEventListener('click', async () => {
    if (authActionPending) return;
    const email = readField(accountEmailInput);
    if (!email) {
      setAccountActionStatus('Enter your email first.', 'error');
      return;
    }
    authActionPending = true;
    setAccountActionStatus('Sending password reset email...');
    updateAccountControls();
    try {
      await onAuthPasswordForgot(email);
      setAccountActionStatus(
        'If account exists, reset email was sent.',
        'success',
      );
    } catch (error) {
      setAccountActionStatus(
        toErrorMessage(error, 'Could not send reset email.'),
        'error',
      );
    } finally {
      authActionPending = false;
      updateAccountControls();
    }
  });

  accountResetSubmitButton.addEventListener('click', async () => {
    if (authActionPending) return;
    const token = hiddenResetToken;
    const password = accountResetPasswordInput.value;
    if (!token || !password) {
      setAccountActionStatus(
        'Reset link is missing or password is empty.',
        'error',
      );
      return;
    }

    authActionPending = true;
    setAccountActionStatus('Resetting password...');
    updateAccountControls();
    try {
      await onAuthPasswordReset({ token, password });
      setAccountActionStatus('Password reset complete. Signed in.', 'success');
      hiddenResetToken = null;
      accountResetPasswordInput.value = '';
      accountPasswordInput.value = '';
      setSignedOutStage('login');
    } catch (error) {
      setAccountActionStatus(
        toErrorMessage(error, 'Could not reset password.'),
        'error',
      );
    } finally {
      authActionPending = false;
      updateAccountControls();
    }
  });

  accountResetPasswordInput.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter') return;
    event.preventDefault();
    accountResetSubmitButton.click();
  });

  accountPanel.appendChild(accountTitle);
  accountPanel.appendChild(accountSummary);
  accountPanel.appendChild(accountActionStatus);
  accountPanel.appendChild(accountSignedOutActions);
  accountPanel.appendChild(accountSignedInActions);
  accountPanel.appendChild(accountBackButton);
  if (authInitialStatus?.message) {
    setAccountActionStatus(
      authInitialStatus.message,
      authInitialStatus.tone ?? 'neutral',
    );
  } else {
    setAccountActionStatus('');
  }
  syncModelAxesControls();
  setMyModelsActionStatus('');
  setAdminActionStatus('');
  adminBaselinesSummary.textContent = formatAdminBaselinesSummary(false);
  renderAdminBaselinesSelect();
  adminDatasetSummary.textContent = formatAdminDatasetSummary(adminLastPage);
  adminLoadedSummary.textContent = 'No recording loaded.';
  setAuthState(currentAuthState);
  setBotLabActionStatus('');

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
  if (showExperimentalGameplayControls) {
    playMenuRow.appendChild(butterfingerPanel);
  }
  menuLayer.appendChild(playMenuRow);
  menuLayer.appendChild(optionsPanel);
  menuLayer.appendChild(aboutPanel);
  menuLayer.appendChild(cheesePanel);
  menuLayer.appendChild(charcuteriePanel);
  menuLayer.appendChild(toolsPanel);
  menuLayer.appendChild(feedbackPanel);
  menuLayer.appendChild(accountPanel);
  menuLayer.appendChild(myModelsPanel);
  menuLayer.appendChild(adminPanel);
  menuLayer.appendChild(botLabPanel);
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
  const footerVersion =
    import.meta.env.MODE === 'dev030' ? `${version}-dev` : version;
  versionLabel.textContent = `v${footerVersion}`;
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
    accountPanel.style.display = panel === 'account' ? 'flex' : 'none';
    myModelsPanel.style.display = panel === 'my_models' ? 'flex' : 'none';
    adminPanel.style.display = panel === 'admin' ? 'flex' : 'none';
    botLabPanel.style.display = panel === 'bot_lab' ? 'flex' : 'none';
    feedbackMenuButton.style.display = panel === 'main' ? 'block' : 'none';
    menuTitle.style.display =
      panel === 'options' ||
      panel === 'my_models' ||
      panel === 'admin' ||
      panel === 'bot_lab'
        ? 'none'
        : 'block';
    if (panel === 'options') {
      syncMlRuntimeSummary();
    }
    if (panel === 'my_models') {
      updateMyModelsControls();
    }
    if (panel === 'admin') {
      updateAdminControls();
    }
    if (panel === 'bot_lab') {
      updateBotLabControls();
    }
  };

  const showMain = () => show('main');

  playButton.addEventListener('click', () => show('play'));
  optionsButton.addEventListener('click', () => show('options'));
  if (showLegacyDataTools) {
    toolsButton.addEventListener('click', () => show('tools'));
  }
  accountButton.addEventListener('click', () => show('account'));
  myModelsButton.addEventListener('click', () => show('my_models'));
  adminButton.addEventListener('click', () => show('admin'));
  botLabButton.addEventListener('click', () => show('bot_lab'));
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
  accountBackButton.addEventListener('click', showMain);
  myModelsBackButton.addEventListener('click', showMain);
  adminBackButton.addEventListener('click', showMain);
  botLabBackButton.addEventListener('click', showMain);

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
    const nextModelAxes = getModelAxes();
    if (!modelAxesEqual(currentModelAxes, nextModelAxes)) {
      currentModelAxes = nextModelAxes;
      syncModelAxesControls();
      myModelsBaselinesSelectorKey = null;
      adminBaselinesSelectorKey = null;
    }
  };

  syncSettings(settingsStore.get());
  showMain();

  return {
    root,
    show,
    showMain,
    setTools,
    setLabelingProgress,
    setAuthState,
    setCharcuterieSpinnerVisible: (visible) => {
      charcuterieSpinner.style.display = visible ? 'flex' : 'none';
    },
    syncSettings,
  };
}
