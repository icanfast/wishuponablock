import { UploadClient, type UploadMode } from '../core/uploadClient';
import { createHttpClient, type HttpClient } from './httpClient';

export type UploadService = {
  uploadClient: UploadClient;
  httpClient: HttpClient;
  mode: UploadMode;
  baseUrl: string;
  useRemote: boolean;
  toolUsesRemote: boolean;
  sendFeedback: (feedback: string, contact: string | null) => Promise<void>;
};

type UploadServiceOptions = {
  envMode?: string;
  envBaseUrl?: string;
};

const inferUploadMode = (): UploadMode => {
  const host = window.location.hostname;
  if (!host) return 'local';
  if (host === 'localhost' || host === '127.0.0.1') return 'local';
  return 'remote';
};

export function createUploadService(
  options: UploadServiceOptions,
): UploadService {
  const rawMode = options.envMode?.toLowerCase();
  const mode: UploadMode =
    rawMode === 'local' || rawMode === 'remote' || rawMode === 'auto'
      ? (rawMode as UploadMode)
      : inferUploadMode();
  const baseUrl = (options.envBaseUrl ?? '/api').replace(/\/+$/, '');
  const uploadClient = new UploadClient({ mode, baseUrl });
  const httpClient = createHttpClient({ baseUrl });
  const useRemote = uploadClient.isRemote;

  console.info(`[Upload] mode=${mode} baseUrl=${baseUrl}`);

  return {
    uploadClient,
    httpClient,
    mode,
    baseUrl,
    useRemote,
    toolUsesRemote: useRemote,
    sendFeedback: async (feedback, contact) => {
      if (!useRemote) {
        console.warn('[Feedback] Remote upload disabled.');
        throw new Error('Remote upload disabled');
      }
      await httpClient.requestJson('/feedback', {
        method: 'POST',
        body: JSON.stringify({
          createdAt: new Date().toISOString(),
          feedback,
          contact,
        }),
      });
    },
  };
}
