export type AuthUser = {
  id: string;
  username: string;
  email: string | null;
  emailVerifiedAtMs: number | null;
};

export type AuthState = {
  authenticated: boolean;
  user: AuthUser | null;
  sessionExpiresAtMs: number | null;
};

export type AuthService = {
  getSession: () => Promise<AuthState>;
  logout: () => Promise<void>;
  sendVerificationEmail: () => Promise<{ alreadyVerified: boolean }>;
  consumeEmailVerification: (token: string) => Promise<AuthState>;
  emailSignup: (payload: {
    email: string;
    password: string;
    username?: string;
  }) => Promise<AuthState>;
  emailLogin: (payload: {
    email: string;
    password: string;
  }) => Promise<AuthState>;
  passwordForgot: (email: string) => Promise<void>;
  passwordReset: (payload: {
    token: string;
    password: string;
  }) => Promise<AuthState>;
  startOAuth: (provider: 'google' | 'discord') => void;
};

type AuthServiceOptions = {
  baseUrl: string;
};

type AuthMePayload = {
  authenticated?: unknown;
  user?: {
    id?: unknown;
    username?: unknown;
    email?: unknown;
    emailVerifiedAtMs?: unknown;
  } | null;
  session?: {
    expiresAtMs?: unknown;
  } | null;
};

const asString = (value: unknown): string | null =>
  typeof value === 'string' && value.trim() ? value.trim() : null;

const asInt = (value: unknown): number | null => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return Math.trunc(value);
  }
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return Math.trunc(parsed);
  }
  return null;
};

const parseErrorMessage = async (response: Response): Promise<string> => {
  const fallback = `Request failed (${response.status}).`;
  try {
    const payload = (await response.json()) as { error?: unknown };
    const error = asString(payload?.error);
    return error ?? fallback;
  } catch {
    return fallback;
  }
};

const normalizeBaseUrl = (baseUrl: string): string =>
  baseUrl.trim().replace(/\/+$/, '');

const normalizeToken = (value: unknown): string | null => {
  if (typeof value !== 'string') return null;
  const token = value.trim();
  return token ? token : null;
};

const toAuthState = (payload: AuthMePayload | null | undefined): AuthState => {
  if (!payload || payload.authenticated !== true || !payload.user) {
    return {
      authenticated: false,
      user: null,
      sessionExpiresAtMs: null,
    };
  }
  const id = asString(payload.user.id);
  const username = asString(payload.user.username);
  if (!id || !username) {
    return {
      authenticated: false,
      user: null,
      sessionExpiresAtMs: null,
    };
  }
  return {
    authenticated: true,
    user: {
      id,
      username,
      email: asString(payload.user.email),
      emailVerifiedAtMs: asInt(payload.user.emailVerifiedAtMs),
    },
    sessionExpiresAtMs: asInt(payload.session?.expiresAtMs),
  };
};

export function createAuthService(options: AuthServiceOptions): AuthService {
  const baseUrl = normalizeBaseUrl(options.baseUrl || '/api');

  const request = async (path: string, init?: RequestInit): Promise<Response> =>
    await fetch(`${baseUrl}${path}`, {
      ...init,
      credentials: 'include',
      cache: 'no-store',
      headers: {
        ...(init?.headers ?? {}),
      },
    });

  return {
    getSession: async () => {
      const response = await request('/auth/me', { method: 'GET' });
      if (!response.ok) {
        throw new Error(await parseErrorMessage(response));
      }
      const payload = (await response.json()) as AuthMePayload;
      return toAuthState(payload);
    },
    logout: async () => {
      const response = await request('/auth/logout', { method: 'POST' });
      if (!response.ok && response.status !== 204) {
        throw new Error(await parseErrorMessage(response));
      }
    },
    sendVerificationEmail: async () => {
      const response = await request('/auth/email/verify/send', {
        method: 'POST',
      });
      if (!response.ok) {
        throw new Error(await parseErrorMessage(response));
      }
      const payload = (await response.json().catch(() => null)) as {
        alreadyVerified?: unknown;
      } | null;
      return { alreadyVerified: payload?.alreadyVerified === true };
    },
    consumeEmailVerification: async (token) => {
      const normalizedToken = normalizeToken(token);
      if (!normalizedToken) {
        throw new Error('Verification token is required.');
      }
      const response = await request('/auth/email/verify/consume', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token: normalizedToken }),
      });
      if (!response.ok) {
        throw new Error(await parseErrorMessage(response));
      }
      const payload = (await response.json()) as AuthMePayload;
      return toAuthState(payload);
    },
    emailSignup: async (payload) => {
      const response = await request('/auth/email/signup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        throw new Error(await parseErrorMessage(response));
      }
      const body = (await response.json()) as AuthMePayload;
      return toAuthState(body);
    },
    emailLogin: async (payload) => {
      const response = await request('/auth/email/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        throw new Error(await parseErrorMessage(response));
      }
      const body = (await response.json()) as AuthMePayload;
      return toAuthState(body);
    },
    passwordForgot: async (email) => {
      const response = await request('/auth/password/forgot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email }),
      });
      if (!response.ok) {
        throw new Error(await parseErrorMessage(response));
      }
    },
    passwordReset: async (payload) => {
      const normalizedToken = normalizeToken(payload.token);
      if (!normalizedToken) {
        throw new Error('Reset token is required.');
      }
      const response = await request('/auth/password/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          token: normalizedToken,
          password: payload.password,
        }),
      });
      if (!response.ok) {
        throw new Error(await parseErrorMessage(response));
      }
      const body = (await response.json()) as AuthMePayload;
      return toAuthState(body);
    },
    startOAuth: (provider) => {
      window.location.assign(`${baseUrl}/auth/oauth/${provider}/start`);
    },
  };
}
