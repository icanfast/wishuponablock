type D1PreparedStatement = {
  bind: (...values: unknown[]) => D1PreparedStatement;
  run: () => Promise<unknown>;
  first: <T = Record<string, unknown>>(
    columnName?: string,
  ) => Promise<T | null>;
};

type D1Database = {
  prepare: (query: string) => D1PreparedStatement;
};

type Env = {
  DB: D1Database;
  ASSETS: { fetch: (request: Request) => Promise<Response> };
  RELEASE_CHANNEL?: string;
  FEATURE_FLAGS?: string;
  AUTH_TOKEN_PEPPER?: string;
  RESEND_API_KEY?: string;
  AUTH_EMAIL_FROM?: string;
  AUTH_APP_BASE_URL?: string;
  MODELS_BUCKET?: unknown;
  RECORDINGS_BUCKET?: unknown;
};

const DAY_MS = 24 * 60 * 60 * 1000;
const SESSION_COOKIE_NAME = 'wub_session';
const SESSION_TOUCH_INTERVAL_MS = DAY_MS;
const SESSION_MAX_AGE_MS = 90 * DAY_MS;
const SESSION_MAX_AGE_SECONDS = Math.trunc(SESSION_MAX_AGE_MS / 1000);
const SESSION_SLIDING_WINDOW_DAYS = 90;
const SESSION_REVOKED_RETENTION_MS = 30 * DAY_MS;
const TOKEN_CONSUMED_RETENTION_MS = 7 * DAY_MS;
const LEGACY_API_ERROR =
  'Legacy snapshot/label APIs were removed on the 0.3.0 dev branch.';
const EMAIL_VERIFY_TOKEN_TTL_MS = DAY_MS;
const PASSWORD_RESET_TOKEN_TTL_MS = 30 * 60 * 1000;
const PASSWORD_SCHEME = 'pbkdf2_sha256';
// Cloudflare Workers currently rejects PBKDF2 iteration counts above 100000.
const PASSWORD_PBKDF2_ITERATIONS = 100_000;
const PASSWORD_PBKDF2_MIN_ITERATIONS = 50_000;
const PASSWORD_PBKDF2_MAX_ITERATIONS = 100_000;
const PASSWORD_SALT_BYTES = 16;
const PASSWORD_HASH_BYTES = 32;
const MIN_PASSWORD_LENGTH = 8;
const MAX_PASSWORD_LENGTH = 128;
const MAX_EMAIL_LENGTH = 320;
const MIN_USERNAME_LENGTH = 2;
const MAX_USERNAME_LENGTH = 32;
const RESEND_SEND_EMAIL_URL = 'https://api.resend.com/emails';
const CORS_BASE_HEADERS = {
  'access-control-allow-origin': '*',
  'access-control-allow-headers': 'content-type',
  'access-control-allow-methods': 'GET,POST,OPTIONS',
};

const withBaseHeaders = (extra?: HeadersInit): Headers => {
  const headers = new Headers(CORS_BASE_HEADERS);
  if (!extra) return headers;
  const merged = new Headers(extra);
  merged.forEach((value, key) => {
    if (key.toLowerCase() === 'set-cookie') {
      headers.append(key, value);
      return;
    }
    headers.set(key, value);
  });
  return headers;
};

const jsonResponse = (
  body: unknown,
  status = 200,
  headers?: HeadersInit,
): Response => {
  const responseHeaders = withBaseHeaders(headers);
  responseHeaders.set('content-type', 'application/json');
  return new Response(JSON.stringify(body), {
    status,
    headers: responseHeaders,
  });
};

const emptyResponse = (status = 204, headers?: HeadersInit): Response =>
  new Response(null, {
    status,
    headers: withBaseHeaders(headers),
  });

const asString = (value: unknown): string | null =>
  typeof value === 'string' && value.trim() ? value.trim() : null;

const asInt = (value: unknown): number | null => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return Math.trunc(value);
  }
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return Math.trunc(parsed);
    }
  }
  return null;
};

const isSecureRequest = (request: Request): boolean =>
  new URL(request.url).protocol === 'https:';

const parseCookies = (header: string | null): Record<string, string> => {
  if (!header) return {};
  const out: Record<string, string> = {};
  for (const part of header.split(';')) {
    const segment = part.trim();
    if (!segment) continue;
    const idx = segment.indexOf('=');
    if (idx <= 0) continue;
    const key = segment.slice(0, idx).trim();
    const value = segment.slice(idx + 1).trim();
    if (!key || !value) continue;
    try {
      out[key] = decodeURIComponent(value);
    } catch {
      out[key] = value;
    }
  }
  return out;
};

const isLikelySessionToken = (value: string): boolean =>
  value.length >= 24 && value.length <= 512 && /^[A-Za-z0-9_-]+$/.test(value);

const getSessionTokenFromRequest = (request: Request): string | null => {
  const cookies = parseCookies(request.headers.get('cookie'));
  const token = cookies[SESSION_COOKIE_NAME];
  if (!token || !isLikelySessionToken(token)) return null;
  return token;
};

const getTokenPepper = (env: Env): string =>
  asString(env.AUTH_TOKEN_PEPPER) ?? '';

const toHex = (bytes: Uint8Array): string => {
  let out = '';
  for (const byte of bytes) {
    out += byte.toString(16).padStart(2, '0');
  }
  return out;
};

const bytesToBase64Url = (bytes: Uint8Array): string => {
  let binary = '';
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }
  return btoa(binary)
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/g, '');
};

const base64UrlToBytes = (value: string): Uint8Array | null => {
  if (!value || !/^[A-Za-z0-9_-]+$/.test(value)) return null;
  const padding =
    value.length % 4 === 0 ? '' : '='.repeat(4 - (value.length % 4));
  const base64 = value.replace(/-/g, '+').replace(/_/g, '/') + padding;
  try {
    const binary = atob(base64);
    const out = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) {
      out[i] = binary.charCodeAt(i);
    }
    return out;
  } catch {
    return null;
  }
};

const generateOpaqueToken = (bytes = 32): string => {
  const out = new Uint8Array(bytes);
  crypto.getRandomValues(out);
  return bytesToBase64Url(out);
};

const derivePasswordKey = async (
  password: string,
  salt: Uint8Array,
  iterations: number,
): Promise<Uint8Array> => {
  const key = await crypto.subtle.importKey(
    'raw',
    new TextEncoder().encode(password),
    'PBKDF2',
    false,
    ['deriveBits'],
  );
  const bits = await crypto.subtle.deriveBits(
    {
      name: 'PBKDF2',
      hash: 'SHA-256',
      salt,
      iterations,
    },
    key,
    PASSWORD_HASH_BYTES * 8,
  );
  return new Uint8Array(bits);
};

const hashPassword = async (password: string): Promise<string> => {
  const salt = new Uint8Array(PASSWORD_SALT_BYTES);
  crypto.getRandomValues(salt);
  const derived = await derivePasswordKey(
    password,
    salt,
    PASSWORD_PBKDF2_ITERATIONS,
  );
  return [
    PASSWORD_SCHEME,
    String(PASSWORD_PBKDF2_ITERATIONS),
    bytesToBase64Url(salt),
    bytesToBase64Url(derived),
  ].join('$');
};

const timingSafeEqual = (a: Uint8Array, b: Uint8Array): boolean => {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i += 1) {
    diff |= a[i] ^ b[i];
  }
  return diff === 0;
};

const verifyPassword = async (
  password: string,
  storedHash: string,
): Promise<boolean> => {
  const parts = storedHash.split('$');
  if (parts.length !== 4) return false;
  const [scheme, iterationsRaw, saltRaw, hashRaw] = parts;
  if (scheme !== PASSWORD_SCHEME) return false;
  const iterations = Number(iterationsRaw);
  if (
    !Number.isFinite(iterations) ||
    !Number.isInteger(iterations) ||
    iterations < PASSWORD_PBKDF2_MIN_ITERATIONS ||
    iterations > PASSWORD_PBKDF2_MAX_ITERATIONS
  ) {
    return false;
  }
  const salt = base64UrlToBytes(saltRaw);
  const expected = base64UrlToBytes(hashRaw);
  if (!salt || !expected || expected.length !== PASSWORD_HASH_BYTES) {
    return false;
  }
  const derived = await derivePasswordKey(password, salt, iterations);
  return timingSafeEqual(derived, expected);
};

const hashOpaqueToken = async (
  token: string,
  pepper: string,
): Promise<string> => {
  const digest = await crypto.subtle.digest(
    'SHA-256',
    new TextEncoder().encode(`${pepper}:${token}`),
  );
  return toHex(new Uint8Array(digest));
};

const clearSessionCookieHeader = (secure: boolean): string => {
  const parts = [
    `${SESSION_COOKIE_NAME}=`,
    'Path=/',
    'HttpOnly',
    'SameSite=Lax',
    'Max-Age=0',
    'Expires=Thu, 01 Jan 1970 00:00:00 GMT',
  ];
  if (secure) parts.push('Secure');
  return parts.join('; ');
};

const sessionCookieHeader = (token: string, secure: boolean): string => {
  const parts = [
    `${SESSION_COOKIE_NAME}=${encodeURIComponent(token)}`,
    'Path=/',
    'HttpOnly',
    'SameSite=Lax',
    `Max-Age=${SESSION_MAX_AGE_SECONDS}`,
  ];
  if (secure) parts.push('Secure');
  return parts.join('; ');
};

const normalizeEmail = (value: unknown): string | null => {
  if (typeof value !== 'string') return null;
  const normalized = value.trim().toLowerCase();
  if (!normalized || normalized.length > MAX_EMAIL_LENGTH) return null;
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(normalized)) return null;
  return normalized;
};

const normalizeUsername = (requested: unknown, emailNorm: string): string => {
  if (typeof requested === 'string') {
    const collapsed = requested.trim().replace(/\s+/g, ' ');
    if (
      collapsed.length >= MIN_USERNAME_LENGTH &&
      collapsed.length <= MAX_USERNAME_LENGTH
    ) {
      return collapsed;
    }
  }

  const localPart = emailNorm.split('@', 1)[0] ?? '';
  const fallback = localPart.replace(/[^A-Za-z0-9._-]/g, '');
  const trimmed = fallback.slice(0, MAX_USERNAME_LENGTH);
  if (trimmed.length >= MIN_USERNAME_LENGTH) {
    return trimmed;
  }
  return 'player';
};

const isValidPasswordCandidate = (value: unknown): value is string =>
  typeof value === 'string' &&
  value.length >= MIN_PASSWORD_LENGTH &&
  value.length <= MAX_PASSWORD_LENGTH;

const asErrorMessage = (error: unknown): string => {
  if (error instanceof Error) return error.message;
  return String(error);
};

const isUniqueEmailConstraintError = (error: unknown): boolean => {
  const message = asErrorMessage(error);
  return (
    message.includes('UNIQUE constraint failed') &&
    message.includes('users.email_norm')
  );
};

type SessionContext = {
  sessionId: string;
  userId: string;
  username: string;
  emailNorm: string | null;
  emailVerifiedAtMs: number | null;
  expiresAtMs: number;
  lastSeenAtMs: number | null;
};

type AuthUser = {
  id: string;
  username: string;
  emailNorm: string | null;
  emailVerifiedAtMs: number | null;
};

type UserLookupRecord = AuthUser & {
  passwordHash: string | null;
};

type AuthTokenType = 'email_verify' | 'password_reset';

type AuthTokenRecord = {
  id: string;
  userId: string;
  emailNorm: string | null;
  expiresAtMs: number;
};

const readUserByEmail = async (
  env: Env,
  emailNorm: string,
): Promise<UserLookupRecord | null> => {
  const row = await env.DB.prepare(
    `SELECT
       id,
       username,
       email_norm,
       email_verified_at_ms,
       password_hash
     FROM users
     WHERE email_norm = ?
     LIMIT 1`,
  )
    .bind(emailNorm)
    .first<Record<string, unknown>>();

  if (!row) return null;
  const id = asString(row.id);
  const username = asString(row.username);
  if (!id || !username) return null;
  return {
    id,
    username,
    emailNorm: asString(row.email_norm),
    emailVerifiedAtMs: asInt(row.email_verified_at_ms),
    passwordHash: asString(row.password_hash),
  };
};

const readUserById = async (
  env: Env,
  userId: string,
): Promise<AuthUser | null> => {
  const row = await env.DB.prepare(
    `SELECT id, username, email_norm, email_verified_at_ms
     FROM users
     WHERE id = ?
     LIMIT 1`,
  )
    .bind(userId)
    .first<Record<string, unknown>>();

  if (!row) return null;
  const id = asString(row.id);
  const username = asString(row.username);
  if (!id || !username) return null;
  return {
    id,
    username,
    emailNorm: asString(row.email_norm),
    emailVerifiedAtMs: asInt(row.email_verified_at_ms),
  };
};

const createSessionForUser = async (
  env: Env,
  userId: string,
  nowMs: number,
): Promise<{ token: string; expiresAtMs: number }> => {
  const sessionId = crypto.randomUUID();
  const token = generateOpaqueToken(32);
  const tokenHash = await hashOpaqueToken(token, getTokenPepper(env));
  const expiresAtMs = nowMs + SESSION_MAX_AGE_MS;
  await env.DB.prepare(
    `INSERT INTO auth_sessions (
       id,
       user_id,
       token_hash,
       expires_at_ms,
       last_seen_at_ms,
       created_at_ms
     ) VALUES (?, ?, ?, ?, ?, ?)`,
  )
    .bind(sessionId, userId, tokenHash, expiresAtMs, nowMs, nowMs)
    .run();
  return { token, expiresAtMs };
};

const createAuthToken = async (
  env: Env,
  type: AuthTokenType,
  userId: string,
  emailNorm: string | null,
  ttlMs: number,
  nowMs: number,
): Promise<{ token: string; expiresAtMs: number }> => {
  const tokenId = crypto.randomUUID();
  const token = generateOpaqueToken(32);
  const tokenHash = await hashOpaqueToken(token, getTokenPepper(env));
  const expiresAtMs = nowMs + ttlMs;
  await env.DB.prepare(
    `INSERT INTO auth_tokens (
       id,
       user_id,
       type,
       token_hash,
       email_norm,
       expires_at_ms,
       consumed_at_ms,
       created_at_ms
     ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?)`,
  )
    .bind(tokenId, userId, type, tokenHash, emailNorm, expiresAtMs, nowMs)
    .run();
  return { token, expiresAtMs };
};

const readActiveAuthTokenByHash = async (
  env: Env,
  type: AuthTokenType,
  tokenHash: string,
  nowMs: number,
): Promise<AuthTokenRecord | null> => {
  const row = await env.DB.prepare(
    `SELECT
       id,
       user_id,
       email_norm,
       expires_at_ms
     FROM auth_tokens
     WHERE type = ?
       AND token_hash = ?
       AND consumed_at_ms IS NULL
       AND expires_at_ms > ?
     LIMIT 1`,
  )
    .bind(type, tokenHash, nowMs)
    .first<Record<string, unknown>>();

  if (!row) return null;
  const id = asString(row.id);
  const userId = asString(row.user_id);
  const expiresAtMs = asInt(row.expires_at_ms);
  if (!id || !userId || expiresAtMs == null) {
    return null;
  }
  return {
    id,
    userId,
    emailNorm: asString(row.email_norm),
    expiresAtMs,
  };
};

const consumeAuthToken = async (
  env: Env,
  tokenId: string,
  nowMs: number,
): Promise<void> => {
  await env.DB.prepare(
    `UPDATE auth_tokens
     SET consumed_at_ms = ?
     WHERE id = ?
       AND consumed_at_ms IS NULL
       AND expires_at_ms > ?`,
  )
    .bind(nowMs, tokenId, nowMs)
    .run();
};

const isLikelyOpaqueToken = (value: unknown): value is string =>
  typeof value === 'string' &&
  value.length >= 24 &&
  value.length <= 512 &&
  /^[A-Za-z0-9_-]+$/.test(value);

const normalizeAppBaseUrl = (value: string): string =>
  value.replace(/\/+$/g, '');

const resolveAuthAppBaseUrl = (request: Request, env: Env): string => {
  const configured = asString(env.AUTH_APP_BASE_URL);
  if (configured) return normalizeAppBaseUrl(configured);
  return normalizeAppBaseUrl(new URL(request.url).origin);
};

const sendEmailViaResend = async (
  env: Env,
  to: string,
  subject: string,
  html: string,
): Promise<boolean> => {
  const apiKey = asString(env.RESEND_API_KEY);
  const from = asString(env.AUTH_EMAIL_FROM);
  if (!apiKey || !from) {
    console.warn('[auth] email delivery skipped: resend not configured');
    return false;
  }

  const response = await fetch(RESEND_SEND_EMAIL_URL, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'content-type': 'application/json',
    },
    body: JSON.stringify({
      from,
      to: [to],
      subject,
      html,
    }),
  });
  if (!response.ok) {
    const body = await response.text().catch(() => '');
    console.warn(
      `[auth] resend delivery failed: ${response.status} ${body.slice(0, 200)}`,
    );
    return false;
  }
  return true;
};

const buildAuthenticatedPayload = (
  user: AuthUser,
  expiresAtMs: number,
): Record<string, unknown> => ({
  authenticated: true,
  user: {
    id: user.id,
    username: user.username,
    email: user.emailNorm,
    emailVerifiedAtMs: user.emailVerifiedAtMs,
  },
  session: {
    expiresAtMs,
  },
});

const readSessionContext = async (
  env: Env,
  tokenHash: string,
  nowMs: number,
): Promise<SessionContext | null> => {
  const row = await env.DB.prepare(
    `SELECT
       s.id AS session_id,
       s.user_id AS user_id,
       s.expires_at_ms AS expires_at_ms,
       s.last_seen_at_ms AS last_seen_at_ms,
       u.username AS username,
       u.email_norm AS email_norm,
       u.email_verified_at_ms AS email_verified_at_ms
     FROM auth_sessions s
     JOIN users u ON u.id = s.user_id
     WHERE s.token_hash = ?
       AND s.revoked_at_ms IS NULL
       AND s.expires_at_ms > ?
     LIMIT 1`,
  )
    .bind(tokenHash, nowMs)
    .first<Record<string, unknown>>();

  if (!row) return null;

  const sessionId = asString(row.session_id);
  const userId = asString(row.user_id);
  const username = asString(row.username);
  const expiresAtMs = asInt(row.expires_at_ms);
  if (!sessionId || !userId || !username || expiresAtMs == null) {
    return null;
  }

  return {
    sessionId,
    userId,
    username,
    emailNorm: asString(row.email_norm),
    emailVerifiedAtMs: asInt(row.email_verified_at_ms),
    expiresAtMs,
    lastSeenAtMs: asInt(row.last_seen_at_ms),
  };
};

const requireAuthenticatedSession = async (
  request: Request,
  env: Env,
  nowMs: number,
): Promise<{ session: SessionContext | null; response?: Response }> => {
  const token = getSessionTokenFromRequest(request);
  if (!token) {
    return {
      session: null,
      response: jsonResponse({ error: 'Unauthorized.' }, 401, {
        'cache-control': 'no-store',
      }),
    };
  }

  const secure = isSecureRequest(request);
  try {
    const tokenHash = await hashOpaqueToken(token, getTokenPepper(env));
    const session = await readSessionContext(env, tokenHash, nowMs);
    if (!session) {
      return {
        session: null,
        response: jsonResponse({ error: 'Unauthorized.' }, 401, {
          'cache-control': 'no-store',
          'set-cookie': clearSessionCookieHeader(secure),
        }),
      };
    }
    return { session };
  } catch (error) {
    console.error('[auth] session lookup failed', error);
    return {
      session: null,
      response: jsonResponse(
        { error: 'Auth storage is currently unavailable.' },
        503,
        { 'cache-control': 'no-store' },
      ),
    };
  }
};

const touchSessionIfStale = async (
  env: Env,
  session: SessionContext,
  nowMs: number,
): Promise<boolean> => {
  const shouldTouch =
    session.lastSeenAtMs == null ||
    nowMs - session.lastSeenAtMs >= SESSION_TOUCH_INTERVAL_MS;
  if (!shouldTouch) return false;

  await env.DB.prepare(
    `UPDATE auth_sessions
     SET last_seen_at_ms = ?, expires_at_ms = ?
     WHERE id = ? AND revoked_at_ms IS NULL`,
  )
    .bind(nowMs, nowMs + SESSION_MAX_AGE_MS, session.sessionId)
    .run();
  return true;
};

const handleEmailSignup = async (
  request: Request,
  env: Env,
): Promise<Response> => {
  if (request.method !== 'POST') {
    return jsonResponse({ error: 'Method not allowed.' }, 405);
  }
  const body = await request.json().catch(() => null);
  if (!body || typeof body !== 'object') {
    return jsonResponse({ error: 'Invalid payload.' }, 400);
  }

  const payload = body as {
    email?: unknown;
    password?: unknown;
    username?: unknown;
  };

  const emailNorm = normalizeEmail(payload.email);
  if (!emailNorm) {
    return jsonResponse({ error: 'Invalid email.' }, 400);
  }
  if (!isValidPasswordCandidate(payload.password)) {
    return jsonResponse(
      {
        error: `Password must be ${MIN_PASSWORD_LENGTH}-${MAX_PASSWORD_LENGTH} characters.`,
      },
      400,
    );
  }
  const username = normalizeUsername(payload.username, emailNorm);
  const secure = isSecureRequest(request);

  try {
    const existing = await readUserByEmail(env, emailNorm);
    if (existing) {
      return jsonResponse({ error: 'Account already exists.' }, 409, {
        'cache-control': 'no-store',
      });
    }

    const nowMs = Date.now();
    const passwordHash = await hashPassword(payload.password);
    const userId = crypto.randomUUID();
    await env.DB.prepare(
      `INSERT INTO users (
         id,
         username,
         email_norm,
         password_hash,
         created_at_ms,
         updated_at_ms
       ) VALUES (?, ?, ?, ?, ?, ?)`,
    )
      .bind(userId, username, emailNorm, passwordHash, nowMs, nowMs)
      .run();

    const session = await createSessionForUser(env, userId, nowMs);
    const user: AuthUser = {
      id: userId,
      username,
      emailNorm,
      emailVerifiedAtMs: null,
    };

    return jsonResponse(
      buildAuthenticatedPayload(user, session.expiresAtMs),
      201,
      {
        'cache-control': 'no-store',
        'set-cookie': sessionCookieHeader(session.token, secure),
      },
    );
  } catch (error) {
    if (isUniqueEmailConstraintError(error)) {
      return jsonResponse({ error: 'Account already exists.' }, 409, {
        'cache-control': 'no-store',
      });
    }
    console.error('[auth] email signup failed', error);
    return jsonResponse(
      { error: 'Auth storage is currently unavailable.' },
      503,
      { 'cache-control': 'no-store' },
    );
  }
};

const handleEmailLogin = async (
  request: Request,
  env: Env,
): Promise<Response> => {
  if (request.method !== 'POST') {
    return jsonResponse({ error: 'Method not allowed.' }, 405);
  }
  const body = await request.json().catch(() => null);
  if (!body || typeof body !== 'object') {
    return jsonResponse({ error: 'Invalid payload.' }, 400);
  }

  const payload = body as {
    email?: unknown;
    password?: unknown;
  };
  const emailNorm = normalizeEmail(payload.email);
  const password =
    typeof payload.password === 'string' ? payload.password : null;
  if (!emailNorm || !password) {
    return jsonResponse({ error: 'Invalid credentials.' }, 401, {
      'cache-control': 'no-store',
    });
  }
  const secure = isSecureRequest(request);

  try {
    const user = await readUserByEmail(env, emailNorm);
    if (!user || !user.passwordHash) {
      return jsonResponse({ error: 'Invalid credentials.' }, 401, {
        'cache-control': 'no-store',
      });
    }

    const passwordOk = await verifyPassword(password, user.passwordHash);
    if (!passwordOk) {
      return jsonResponse({ error: 'Invalid credentials.' }, 401, {
        'cache-control': 'no-store',
      });
    }

    const nowMs = Date.now();
    const session = await createSessionForUser(env, user.id, nowMs);
    const authUser: AuthUser = {
      id: user.id,
      username: user.username,
      emailNorm: user.emailNorm,
      emailVerifiedAtMs: user.emailVerifiedAtMs,
    };
    return jsonResponse(
      buildAuthenticatedPayload(authUser, session.expiresAtMs),
      200,
      {
        'cache-control': 'no-store',
        'set-cookie': sessionCookieHeader(session.token, secure),
      },
    );
  } catch (error) {
    console.error('[auth] email login failed', error);
    return jsonResponse(
      { error: 'Auth storage is currently unavailable.' },
      503,
      { 'cache-control': 'no-store' },
    );
  }
};

const handleEmailVerifySend = async (
  request: Request,
  env: Env,
): Promise<Response> => {
  if (request.method !== 'POST') {
    return jsonResponse({ error: 'Method not allowed.' }, 405);
  }
  const nowMs = Date.now();
  const auth = await requireAuthenticatedSession(request, env, nowMs);
  if (auth.response) return auth.response;
  const session = auth.session!;
  if (!session.emailNorm) {
    return jsonResponse(
      { error: 'Email is not available for this account.' },
      400,
      { 'cache-control': 'no-store' },
    );
  }
  if (session.emailVerifiedAtMs != null) {
    return jsonResponse({ ok: true, alreadyVerified: true }, 200, {
      'cache-control': 'no-store',
    });
  }

  try {
    const token = await createAuthToken(
      env,
      'email_verify',
      session.userId,
      session.emailNorm,
      EMAIL_VERIFY_TOKEN_TTL_MS,
      nowMs,
    );
    const appBaseUrl = resolveAuthAppBaseUrl(request, env);
    const verifyUrl = `${appBaseUrl}/auth/email/verify?token=${encodeURIComponent(token.token)}`;
    const html =
      `<p>Verify your Wish Upon a Block account email.</p>` +
      `<p><a href="${verifyUrl}">Verify email</a></p>` +
      `<p>If the button does not work, paste this URL into your browser:</p>` +
      `<p>${verifyUrl}</p>`;
    await sendEmailViaResend(
      env,
      session.emailNorm,
      'Verify your Wish Upon a Block email',
      html,
    );
  } catch (error) {
    console.error('[auth] email verify send failed', error);
    return jsonResponse(
      { error: 'Auth storage is currently unavailable.' },
      503,
      { 'cache-control': 'no-store' },
    );
  }

  return jsonResponse({ ok: true }, 202, { 'cache-control': 'no-store' });
};

const handleEmailVerifyConsume = async (
  request: Request,
  env: Env,
): Promise<Response> => {
  if (request.method !== 'POST') {
    return jsonResponse({ error: 'Method not allowed.' }, 405);
  }
  const body = await request.json().catch(() => null);
  if (!body || typeof body !== 'object') {
    return jsonResponse({ error: 'Invalid payload.' }, 400);
  }
  const payload = body as { token?: unknown };
  if (!isLikelyOpaqueToken(payload.token)) {
    return jsonResponse({ error: 'Invalid or expired token.' }, 400, {
      'cache-control': 'no-store',
    });
  }

  const nowMs = Date.now();
  const secure = isSecureRequest(request);
  try {
    const tokenHash = await hashOpaqueToken(payload.token, getTokenPepper(env));
    const token = await readActiveAuthTokenByHash(
      env,
      'email_verify',
      tokenHash,
      nowMs,
    );
    if (!token) {
      return jsonResponse({ error: 'Invalid or expired token.' }, 400, {
        'cache-control': 'no-store',
      });
    }

    if (token.emailNorm) {
      await env.DB.prepare(
        `UPDATE users
         SET email_verified_at_ms = COALESCE(email_verified_at_ms, ?),
             updated_at_ms = ?
         WHERE id = ? AND email_norm = ?`,
      )
        .bind(nowMs, nowMs, token.userId, token.emailNorm)
        .run();
    } else {
      await env.DB.prepare(
        `UPDATE users
         SET email_verified_at_ms = COALESCE(email_verified_at_ms, ?),
             updated_at_ms = ?
         WHERE id = ?`,
      )
        .bind(nowMs, nowMs, token.userId)
        .run();
    }

    await consumeAuthToken(env, token.id, nowMs);
    const user = await readUserById(env, token.userId);
    if (!user) {
      return jsonResponse(
        { error: 'Auth storage is currently unavailable.' },
        503,
        { 'cache-control': 'no-store' },
      );
    }
    const session = await createSessionForUser(env, user.id, nowMs);
    return jsonResponse(
      buildAuthenticatedPayload(
        {
          id: user.id,
          username: user.username,
          emailNorm: user.emailNorm,
          emailVerifiedAtMs: user.emailVerifiedAtMs ?? nowMs,
        },
        session.expiresAtMs,
      ),
      200,
      {
        'cache-control': 'no-store',
        'set-cookie': sessionCookieHeader(session.token, secure),
      },
    );
  } catch (error) {
    console.error('[auth] email verify consume failed', error);
    return jsonResponse(
      { error: 'Auth storage is currently unavailable.' },
      503,
      { 'cache-control': 'no-store' },
    );
  }
};

const handlePasswordForgot = async (
  request: Request,
  env: Env,
): Promise<Response> => {
  if (request.method !== 'POST') {
    return jsonResponse({ error: 'Method not allowed.' }, 405);
  }
  const body = await request.json().catch(() => null);
  const payload = body as { email?: unknown } | null;
  const emailNorm = normalizeEmail(payload?.email);
  const nowMs = Date.now();
  try {
    if (emailNorm) {
      const user = await readUserByEmail(env, emailNorm);
      if (user && user.passwordHash) {
        const token = await createAuthToken(
          env,
          'password_reset',
          user.id,
          user.emailNorm,
          PASSWORD_RESET_TOKEN_TTL_MS,
          nowMs,
        );
        const appBaseUrl = resolveAuthAppBaseUrl(request, env);
        const resetUrl = `${appBaseUrl}/auth/password/reset?token=${encodeURIComponent(token.token)}`;
        const html =
          `<p>Reset your Wish Upon a Block password.</p>` +
          `<p><a href="${resetUrl}">Reset password</a></p>` +
          `<p>If you did not request this, you can ignore this email.</p>`;
        await sendEmailViaResend(
          env,
          user.emailNorm ?? emailNorm,
          'Reset your Wish Upon a Block password',
          html,
        );
      }
    }
  } catch (error) {
    console.error('[auth] password forgot failed', error);
  }

  return jsonResponse({ ok: true }, 202, { 'cache-control': 'no-store' });
};

const handlePasswordReset = async (
  request: Request,
  env: Env,
): Promise<Response> => {
  if (request.method !== 'POST') {
    return jsonResponse({ error: 'Method not allowed.' }, 405);
  }
  const body = await request.json().catch(() => null);
  if (!body || typeof body !== 'object') {
    return jsonResponse({ error: 'Invalid payload.' }, 400);
  }

  const payload = body as {
    token?: unknown;
    password?: unknown;
  };
  if (!isLikelyOpaqueToken(payload.token)) {
    return jsonResponse({ error: 'Invalid or expired token.' }, 400, {
      'cache-control': 'no-store',
    });
  }
  if (!isValidPasswordCandidate(payload.password)) {
    return jsonResponse(
      {
        error: `Password must be ${MIN_PASSWORD_LENGTH}-${MAX_PASSWORD_LENGTH} characters.`,
      },
      400,
      { 'cache-control': 'no-store' },
    );
  }

  const nowMs = Date.now();
  const secure = isSecureRequest(request);
  try {
    const tokenHash = await hashOpaqueToken(payload.token, getTokenPepper(env));
    const token = await readActiveAuthTokenByHash(
      env,
      'password_reset',
      tokenHash,
      nowMs,
    );
    if (!token) {
      return jsonResponse({ error: 'Invalid or expired token.' }, 400, {
        'cache-control': 'no-store',
      });
    }

    const passwordHash = await hashPassword(payload.password);
    await env.DB.prepare(
      `UPDATE users
       SET password_hash = ?, updated_at_ms = ?
       WHERE id = ?`,
    )
      .bind(passwordHash, nowMs, token.userId)
      .run();
    await consumeAuthToken(env, token.id, nowMs);
    await env.DB.prepare(
      `UPDATE auth_sessions
       SET revoked_at_ms = ?
       WHERE user_id = ? AND revoked_at_ms IS NULL`,
    )
      .bind(nowMs, token.userId)
      .run();

    const user = await readUserById(env, token.userId);
    if (!user) {
      return jsonResponse(
        { error: 'Auth storage is currently unavailable.' },
        503,
        { 'cache-control': 'no-store' },
      );
    }
    const session = await createSessionForUser(env, user.id, nowMs);
    return jsonResponse(
      buildAuthenticatedPayload(user, session.expiresAtMs),
      200,
      {
        'cache-control': 'no-store',
        'set-cookie': sessionCookieHeader(session.token, secure),
      },
    );
  } catch (error) {
    console.error('[auth] password reset failed', error);
    return jsonResponse(
      { error: 'Auth storage is currently unavailable.' },
      503,
      { 'cache-control': 'no-store' },
    );
  }
};

const handleAuthMe = async (request: Request, env: Env): Promise<Response> => {
  if (request.method !== 'GET') {
    return jsonResponse({ error: 'Method not allowed.' }, 405);
  }

  const token = getSessionTokenFromRequest(request);
  if (!token) {
    return jsonResponse(
      {
        authenticated: false,
        user: null,
      },
      200,
      { 'cache-control': 'no-store' },
    );
  }

  const nowMs = Date.now();
  const secure = isSecureRequest(request);
  const clearCookie = clearSessionCookieHeader(secure);
  let session: SessionContext | null = null;

  try {
    const tokenHash = await hashOpaqueToken(token, getTokenPepper(env));
    session = await readSessionContext(env, tokenHash, nowMs);
  } catch (error) {
    console.error('[auth] /me lookup failed', error);
    return jsonResponse(
      { error: 'Auth storage is currently unavailable.' },
      503,
      { 'cache-control': 'no-store' },
    );
  }

  if (!session) {
    return jsonResponse(
      {
        authenticated: false,
        user: null,
      },
      200,
      {
        'cache-control': 'no-store',
        'set-cookie': clearCookie,
      },
    );
  }

  try {
    const touched = await touchSessionIfStale(env, session, nowMs);
    if (touched) {
      session.expiresAtMs = nowMs + SESSION_MAX_AGE_MS;
    }
  } catch (error) {
    console.error('[auth] /me touch failed', error);
  }

  return jsonResponse(
    buildAuthenticatedPayload(
      {
        id: session.userId,
        username: session.username,
        emailNorm: session.emailNorm,
        emailVerifiedAtMs: session.emailVerifiedAtMs,
      },
      session.expiresAtMs,
    ),
    200,
    { 'cache-control': 'no-store' },
  );
};

const handleAuthLogout = async (
  request: Request,
  env: Env,
): Promise<Response> => {
  if (request.method !== 'POST') {
    return jsonResponse({ error: 'Method not allowed.' }, 405);
  }

  const secure = isSecureRequest(request);
  const clearCookie = clearSessionCookieHeader(secure);
  const token = getSessionTokenFromRequest(request);
  if (!token) {
    return emptyResponse(204, {
      'cache-control': 'no-store',
      'set-cookie': clearCookie,
    });
  }

  try {
    const nowMs = Date.now();
    const tokenHash = await hashOpaqueToken(token, getTokenPepper(env));
    await env.DB.prepare(
      `UPDATE auth_sessions
       SET revoked_at_ms = ?
       WHERE token_hash = ? AND revoked_at_ms IS NULL`,
    )
      .bind(nowMs, tokenHash)
      .run();
  } catch (error) {
    console.error('[auth] /logout revoke failed', error);
    return jsonResponse(
      { error: 'Auth storage is currently unavailable.' },
      503,
      {
        'cache-control': 'no-store',
        'set-cookie': clearCookie,
      },
    );
  }

  return emptyResponse(204, {
    'cache-control': 'no-store',
    'set-cookie': clearCookie,
  });
};

const parseFeatureFlags = (raw: unknown): Record<string, unknown> => {
  if (typeof raw !== 'string' || !raw.trim()) {
    return {};
  }
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      return {};
    }
    const out: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(parsed)) {
      if (!key.trim()) continue;
      if (
        value == null ||
        typeof value === 'boolean' ||
        typeof value === 'number' ||
        typeof value === 'string'
      ) {
        out[key] = value;
      }
    }
    return out;
  } catch {
    return {};
  }
};

const cleanupAuthState = async (env: Env): Promise<void> => {
  const nowMs = Date.now();
  const consumedTokenCutoffMs = nowMs - TOKEN_CONSUMED_RETENTION_MS;
  const revokedSessionCutoffMs = nowMs - SESSION_REVOKED_RETENTION_MS;
  try {
    await env.DB.prepare(
      `DELETE FROM auth_tokens
       WHERE expires_at_ms < ?
          OR (consumed_at_ms IS NOT NULL AND consumed_at_ms < ?)`,
    )
      .bind(nowMs, consumedTokenCutoffMs)
      .run();
    await env.DB.prepare(
      `DELETE FROM auth_sessions
       WHERE expires_at_ms < ?
          OR (revoked_at_ms IS NOT NULL AND revoked_at_ms < ?)`,
    )
      .bind(nowMs, revokedSessionCutoffMs)
      .run();
  } catch (error) {
    console.error(
      `[cron] auth cleanup failed (session_window_days=${SESSION_SLIDING_WINDOW_DAYS})`,
      error,
    );
  }
};

const handleFeedback = async (
  env: Env,
  payload: unknown,
): Promise<Response> => {
  if (!payload || typeof payload !== 'object') {
    return jsonResponse({ error: 'Invalid payload.' }, 400);
  }
  const record = payload as {
    createdAt?: string;
    feedback?: string;
    contact?: string | null;
  };

  const feedback =
    typeof record.feedback === 'string' ? record.feedback.trim() : '';
  if (!feedback) {
    return jsonResponse({ error: 'Missing feedback.' }, 400);
  }
  const contact =
    typeof record.contact === 'string' ? record.contact.trim() : '';
  const createdAt =
    typeof record.createdAt === 'string'
      ? record.createdAt
      : new Date().toISOString();

  try {
    await env.DB.prepare(
      `INSERT INTO feedback (created_at, feedback, contact) VALUES (?, ?, ?)`,
    )
      .bind(createdAt, feedback, contact || null)
      .run();
  } catch {
    return jsonResponse(
      {
        error: 'Feedback storage is currently unavailable in this environment.',
      },
      503,
    );
  }

  return emptyResponse();
};

const isLegacyApiPath = (pathname: string): boolean =>
  pathname.startsWith('/api/snapshots') || pathname.startsWith('/api/labels');

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    if (request.method === 'OPTIONS') {
      return emptyResponse();
    }

    if (url.pathname.startsWith('/api/runtime/flags')) {
      if (request.method !== 'GET') {
        return jsonResponse({ error: 'Method not allowed.' }, 405);
      }
      const channel = asString(env.RELEASE_CHANNEL) ?? 'unknown';
      return jsonResponse({
        channel,
        flags: parseFeatureFlags(env.FEATURE_FLAGS),
      });
    }

    if (isLegacyApiPath(url.pathname)) {
      return jsonResponse({ error: LEGACY_API_ERROR }, 410);
    }

    if (url.pathname === '/api/auth/email/signup') {
      return handleEmailSignup(request, env);
    }

    if (url.pathname === '/api/auth/email/login') {
      return handleEmailLogin(request, env);
    }

    if (url.pathname === '/api/auth/email/verify/send') {
      return handleEmailVerifySend(request, env);
    }

    if (url.pathname === '/api/auth/email/verify/consume') {
      return handleEmailVerifyConsume(request, env);
    }

    if (url.pathname === '/api/auth/password/forgot') {
      return handlePasswordForgot(request, env);
    }

    if (url.pathname === '/api/auth/password/reset') {
      return handlePasswordReset(request, env);
    }

    if (url.pathname === '/api/auth/me') {
      return handleAuthMe(request, env);
    }

    if (url.pathname === '/api/auth/logout') {
      return handleAuthLogout(request, env);
    }

    if (url.pathname.startsWith('/api/feedback')) {
      if (request.method !== 'POST') {
        return jsonResponse({ error: 'Method not allowed.' }, 405);
      }
      const payload = await request.json().catch(() => null);
      return handleFeedback(env, payload);
    }

    if (url.pathname.startsWith('/api/')) {
      return jsonResponse({ error: 'Not found.' }, 404);
    }

    return env.ASSETS.fetch(request);
  },
  async scheduled(_controller: unknown, env: Env): Promise<void> {
    await cleanupAuthState(env);
  },
};
