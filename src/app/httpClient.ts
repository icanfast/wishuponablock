export type HttpClient = {
  requestJson: <T>(path: string, init?: RequestInit) => Promise<T | null>;
};

type HttpClientOptions = {
  baseUrl: string;
};

export function createHttpClient(options: HttpClientOptions): HttpClient {
  const { baseUrl } = options;

  const requestJson = async <T>(
    path: string,
    init: RequestInit = {},
  ): Promise<T | null> => {
    const url = `${baseUrl}${path}`;
    const headers = new Headers(init.headers);
    if (!headers.has('Content-Type')) {
      headers.set('Content-Type', 'application/json');
    }
    const response = await fetch(url, { ...init, headers });
    if (!response.ok) {
      const message = await response.text().catch(() => '');
      const detail = message ? ` ${message}` : '';
      throw new Error(`Request failed (${response.status}).${detail}`);
    }
    if (response.status === 204) return null;
    return (await response.json()) as T;
  };

  return { requestJson };
}
