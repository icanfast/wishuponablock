import type { SnapshotSample, SnapshotSessionMeta } from './snapshotRecorder';

export type UploadMode = 'local' | 'remote' | 'auto';

export type UploadClientOptions = {
  mode: UploadMode;
  baseUrl: string;
};

type QueueItem = {
  id?: number;
  type: 'snapshot' | 'label';
  payload: unknown;
  createdAt: string;
};

const DB_NAME = 'wishuponablock.uploads';
const STORE_NAME = 'queue';
const DB_VERSION = 1;

export class UploadClient {
  private mode: UploadMode;
  private baseUrl: string;
  private flushing = false;

  constructor(options: UploadClientOptions) {
    this.mode = options.mode;
    this.baseUrl = options.baseUrl.replace(/\/+$/, '');
  }

  get isRemote(): boolean {
    return this.mode !== 'local';
  }

  async enqueueSnapshot(
    session: SnapshotSessionMeta,
    sample: SnapshotSample,
  ): Promise<void> {
    await this.enqueue({
      type: 'snapshot',
      payload: { session, sample },
    });
  }

  async enqueueLabel(record: unknown): Promise<void> {
    await this.enqueue({ type: 'label', payload: record });
  }

  async flush(): Promise<void> {
    if (!this.isRemote || this.flushing) return;
    this.flushing = true;
    try {
      const items = await listQueueItems();
      for (const item of items) {
        await this.postItem(item);
        await deleteQueueItem(item.id!);
      }
    } catch (err) {
      console.warn('[Upload] Flush failed.', err);
      // Leave queued items for later retry.
    } finally {
      this.flushing = false;
    }
  }

  private async enqueue(item: Omit<QueueItem, 'createdAt'>): Promise<void> {
    await addQueueItem({
      ...item,
      createdAt: new Date().toISOString(),
    });
    if (this.isRemote) {
      void this.flush();
    }
  }

  private async postItem(item: QueueItem): Promise<void> {
    const url =
      item.type === 'snapshot'
        ? `${this.baseUrl}/snapshots`
        : `${this.baseUrl}/labels`;
    const summary = summarizePayload(item);
    const startedAt = performance.now();
    console.info(`[Upload] POST ${url}`, summary);
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(item.payload),
    });
    const elapsed = performance.now() - startedAt;
    if (!res.ok) {
      console.warn(
        `[Upload] Failed ${url} (${res.status}) in ${elapsed.toFixed(1)}ms`,
      );
      throw new Error(`Upload failed (${res.status})`);
    }
    console.info(
      `[Upload] OK ${url} (${res.status}) in ${elapsed.toFixed(1)}ms`,
    );
  }
}

function summarizePayload(item: QueueItem): Record<string, unknown> {
  if (item.type === 'snapshot') {
    const payload = item.payload as
      | { session?: { id?: string }; sample?: { index?: number } }
      | undefined;
    return {
      type: 'snapshot',
      sessionId: payload?.session?.id ?? null,
      sampleIndex: payload?.sample?.index ?? null,
    };
  }
  const payload = item.payload as
    | { source?: { sessionId?: string; sampleIndex?: number } }
    | undefined;
  return {
    type: 'label',
    sessionId: payload?.source?.sessionId ?? null,
    sampleIndex: payload?.source?.sampleIndex ?? null,
  };
}

async function openDb(): Promise<IDBDatabase> {
  return await new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onerror = () => reject(request.error);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, {
          keyPath: 'id',
          autoIncrement: true,
        });
      }
    };
    request.onsuccess = () => resolve(request.result);
  });
}

async function addQueueItem(item: QueueItem): Promise<void> {
  const db = await openDb();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite');
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.objectStore(STORE_NAME).add(item);
  });
}

async function listQueueItems(): Promise<QueueItem[]> {
  const db = await openDb();
  return await new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly');
    const request = tx.objectStore(STORE_NAME).getAll();
    request.onsuccess = () => {
      const items = (request.result as QueueItem[]) ?? [];
      items.sort((a, b) => (a.id ?? 0) - (b.id ?? 0));
      resolve(items);
    };
    request.onerror = () => reject(request.error);
  });
}

async function deleteQueueItem(id: number): Promise<void> {
  const db = await openDb();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite');
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.objectStore(STORE_NAME).delete(id);
  });
}
