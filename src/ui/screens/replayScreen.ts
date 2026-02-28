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

  const backButton = makeButton('BACK TO MENU');
  Object.assign(backButton.style, {
    position: 'absolute',
    right: '16px',
    top: '16px',
    pointerEvents: 'auto',
    zIndex: '3',
  });
  root.appendChild(backButton);

  return {
    root,
    backButton,
    setStatus: () => {},
    setDetails: () => {},
    appendLog: () => {},
    clearLog: () => {},
  };
};
