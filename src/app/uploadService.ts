import { UploadClient, type UploadMode } from '../core/uploadClient';
import { createHttpClient, type HttpClient } from './httpClient';

export type UploadService = {
  uploadClient: UploadClient;
  httpClient: HttpClient;
  mode: UploadMode;
  baseUrl: string;
  useRemote: boolean;
  toolUsesRemote: boolean;
  getSnapshotCountForBuild: (buildVersion: string) => Promise<number | null>;
  getLabeledBoardCountForBuild: (
    buildVersion: string,
  ) => Promise<number | null>;
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
    getSnapshotCountForBuild: async (buildVersion) => {
      if (!useRemote || !buildVersion.trim()) return null;
      const payload = await httpClient.requestJson<{
        builds?: Array<{ build?: string; count?: number }>;
      }>('/snapshots/builds');
      const builds = payload?.builds ?? [];
      const match = builds.find(
        (entry) =>
          typeof entry.build === 'string' &&
          entry.build.trim() === buildVersion,
      );
      if (!match) return 0;
      if (typeof match.count === 'number' && Number.isFinite(match.count)) {
        return Math.max(0, Math.trunc(match.count));
      }
      return 0;
    },
    getLabeledBoardCountForBuild: async (buildVersion) => {
      const normalizedBuild = buildVersion.trim();
      if (!useRemote || !normalizedBuild) return null;
      const payload = await httpClient.requestJson<{
        build?: string;
        labeledBoards?: number;
      }>(`/labels/progress?build=${encodeURIComponent(normalizedBuild)}`);
      if (
        typeof payload?.labeledBoards === 'number' &&
        Number.isFinite(payload.labeledBoards)
      ) {
        return Math.max(0, Math.trunc(payload.labeledBoards));
      }
      return 0;
    },
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
