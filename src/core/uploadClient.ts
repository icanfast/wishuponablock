export type SnapshotUploadRecord = {
  createdAt: string;
  meta: Record<string, unknown>;
  board: number[][];
};

export type SnapshotBatchUploadRecord = SnapshotUploadRecord[];

export type LabelUploadRecord = {
  createdAt: string;
  data: unknown;
};

export type UploadMode = 'local' | 'remote' | 'auto';

export type UploadClientOptions = {
  mode: UploadMode;
  baseUrl: string;
};

type QueueItem = {
  id?: number;
  type: 'snapshot' | 'snapshot_batch' | 'label';
  payload: unknown;
  createdAt: string;
};

const DB_NAME = 'wishuponablock.uploads.v2';
const STORE_NAME = 'queue';
const DB_VERSION = 1;

export class UploadClient {
  private mode: UploadMode;
  private baseUrl: string;
  private flushing = false;
  private onSnapshotUploaded: ((record: SnapshotUploadRecord) => void) | null =
    null;

  constructor(options: UploadClientOptions) {
    this.mode = options.mode;
    this.baseUrl = options.baseUrl.replace(/\/+$/, '');
  }

  get isRemote(): boolean {
    return this.mode !== 'local';
  }

  setOnSnapshotUploaded(
    listener: ((record: SnapshotUploadRecord) => void) | null,
  ): void {
    this.onSnapshotUploaded = listener;
  }

  async enqueueSnapshot(record: SnapshotUploadRecord): Promise<void> {
    await this.enqueue({ type: 'snapshot', payload: record });
  }

  async enqueueSnapshotBatch(
    records: SnapshotBatchUploadRecord,
  ): Promise<void> {
    if (!Array.isArray(records) || records.length === 0) return;
    await this.enqueue({ type: 'snapshot_batch', payload: records });
  }

  async enqueueLabel(record: LabelUploadRecord): Promise<void> {
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
      item.type === 'snapshot' || item.type === 'snapshot_batch'
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
    if (item.type === 'snapshot' && this.onSnapshotUploaded) {
      this.onSnapshotUploaded(item.payload as SnapshotUploadRecord);
    }
    if (item.type === 'snapshot_batch' && this.onSnapshotUploaded) {
      const records = item.payload as SnapshotBatchUploadRecord;
      for (const record of records) {
        this.onSnapshotUploaded(record);
      }
    }
  }
}

function summarizePayload(item: QueueItem): Record<string, unknown> {
  if (item.type === 'snapshot_batch') {
    const records = Array.isArray(item.payload)
      ? (item.payload as SnapshotBatchUploadRecord)
      : [];
    const count = records.length;
    const first = count > 0 ? records[0] : undefined;
    const last = count > 0 ? records[count - 1] : undefined;
    const readSessionId = (
      record: SnapshotUploadRecord | undefined,
    ): string | null => {
      if (!record || !record.meta || typeof record.meta !== 'object')
        return null;
      const meta = record.meta as {
        session?: { id?: unknown };
        sample?: { index?: unknown };
      };
      return typeof meta.session?.id === 'string' ? meta.session.id : null;
    };
    const readSampleIndex = (
      record: SnapshotUploadRecord | undefined,
    ): number | null => {
      if (!record || !record.meta || typeof record.meta !== 'object')
        return null;
      const meta = record.meta as {
        sample?: { index?: unknown };
      };
      const value = meta.sample?.index;
      if (typeof value === 'number' && Number.isFinite(value)) return value;
      return null;
    };
    return {
      type: 'snapshot_batch',
      count,
      firstSessionId: readSessionId(first),
      firstSampleIndex: readSampleIndex(first),
      lastSessionId: readSessionId(last),
      lastSampleIndex: readSampleIndex(last),
    };
  }
  if (item.type === 'snapshot') {
    const payload = item.payload as
      | {
          meta?: {
            session?: { id?: string };
            sample?: { index?: number };
            trigger?: string;
          };
        }
      | undefined;
    return {
      type: 'snapshot',
      sessionId: payload?.meta?.session?.id ?? null,
      sampleIndex: payload?.meta?.sample?.index ?? null,
      trigger: payload?.meta?.trigger ?? null,
    };
  }
  const payload = item.payload as
    | {
        data?: {
          source?: {
            sessionId?: string;
            sampleIndex?: number;
            snapshotId?: number;
          };
        };
      }
    | undefined;
  return {
    type: 'label',
    snapshotId: payload?.data?.source?.snapshotId ?? null,
    sessionId: payload?.data?.source?.sessionId ?? null,
    sampleIndex: payload?.data?.source?.sampleIndex ?? null,
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
