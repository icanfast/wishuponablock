export type ReplayScreen = {
  root: HTMLDivElement;
  backButton: HTMLButtonElement;
  setStatus: (value: string) => void;
  setDetails: (value: string) => void;
  appendLog: (line: string) => void;
  clearLog: () => void;
};

const makeButton = (labelText: string): HTMLButtonElement => {
  const button = document.createElement('button');
  button.textContent = labelText;
  Object.assign(button.style, {
    background: '#0b0f14',
    color: '#e2e8f0',
    border: '1px solid #1f2a37',
    borderRadius: '6px',
    padding: '10px 12px',
    fontSize: '13px',
    cursor: 'pointer',
  });
  return button;
};

export const createReplayScreen = (): ReplayScreen => {
  const root = document.createElement('div');
  Object.assign(root.style, {
    position: 'absolute',
    inset: '0',
    pointerEvents: 'none',
  });

  const panel = document.createElement('div');
  Object.assign(panel.style, {
    position: 'absolute',
    left: '16px',
    top: '16px',
    width: '320px',
    maxHeight: 'calc(100% - 32px)',
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
    padding: '10px',
    background: '#121a24',
    border: '2px solid #0b0f14',
    borderRadius: '8px',
    color: '#e2e8f0',
    pointerEvents: 'auto',
    fontFamily: 'system-ui, -apple-system, Segoe UI, sans-serif',
    fontSize: '12px',
    boxSizing: 'border-box',
  });
  root.appendChild(panel);

  const title = document.createElement('div');
  title.textContent = 'REPLAY';
  Object.assign(title.style, {
    color: '#8fa0b8',
    fontSize: '12px',
    letterSpacing: '0.5px',
  });
  panel.appendChild(title);

  const status = document.createElement('div');
  Object.assign(status.style, {
    color: '#b6c2d4',
    lineHeight: '1.35',
    whiteSpace: 'pre-wrap',
    background: '#0b0f14',
    border: '1px solid #1f2a37',
    borderRadius: '6px',
    padding: '8px',
    minHeight: '48px',
    boxSizing: 'border-box',
  });
  panel.appendChild(status);

  const details = document.createElement('div');
  Object.assign(details.style, {
    color: '#8fa0b8',
    lineHeight: '1.35',
    whiteSpace: 'pre-wrap',
    background: '#0b0f14',
    border: '1px solid #1f2a37',
    borderRadius: '6px',
    padding: '8px',
    minHeight: '64px',
    boxSizing: 'border-box',
  });
  panel.appendChild(details);

  const logWrap = document.createElement('div');
  Object.assign(logWrap.style, {
    background: '#0b0f14',
    border: '1px solid #1f2a37',
    borderRadius: '6px',
    padding: '8px',
    minHeight: '120px',
    flex: '1',
    overflowY: 'auto',
    boxSizing: 'border-box',
  });
  panel.appendChild(logWrap);

  const log = document.createElement('pre');
  Object.assign(log.style, {
    margin: '0',
    color: '#b6c2d4',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    fontSize: '11px',
    lineHeight: '1.35',
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word',
  });
  logWrap.appendChild(log);

  const backButton = makeButton('BACK TO MENU');
  panel.appendChild(backButton);

  return {
    root,
    backButton,
    setStatus: (value) => {
      status.textContent = value;
    },
    setDetails: (value) => {
      details.textContent = value;
    },
    appendLog: (line) => {
      const next = log.textContent ? `${log.textContent}\n${line}` : line;
      log.textContent = next;
      logWrap.scrollTop = logWrap.scrollHeight;
    },
    clearLog: () => {
      log.textContent = '';
    },
  };
};
