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

type R2ObjectBody = {
  arrayBuffer: () => Promise<ArrayBuffer>;
};

type R2PutOptions = {
  httpMetadata?: {
    contentType?: string;
  };
};

type R2Bucket = {
  get: (key: string) => Promise<R2ObjectBody | null>;
  put: (
    key: string,
    value: ArrayBuffer | ArrayBufferView | string,
    options?: R2PutOptions,
  ) => Promise<void>;
  delete: (key: string) => Promise<void>;
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
  GOOGLE_OAUTH_CLIENT_ID?: string;
  GOOGLE_OAUTH_CLIENT_SECRET?: string;
  DISCORD_OAUTH_CLIENT_ID?: string;
  DISCORD_OAUTH_CLIENT_SECRET?: string;
  MODELS_BUCKET?: R2Bucket;
  RECORDINGS_BUCKET?: R2Bucket;
};

const DAY_MS = 24 * 60 * 60 * 1000;
const SESSION_COOKIE_NAME = 'wub_session';
const SESSION_TOUCH_INTERVAL_MS = DAY_MS;
const SESSION_MAX_AGE_MS = 90 * DAY_MS;
const SESSION_MAX_AGE_SECONDS = Math.trunc(SESSION_MAX_AGE_MS / 1000);
const SESSION_SLIDING_WINDOW_DAYS = 90;
const SESSION_REVOKED_RETENTION_MS = 30 * DAY_MS;
const TOKEN_CONSUMED_RETENTION_MS = 7 * DAY_MS;
const MAX_PERSONALIZED_MODEL_BYTES = 2 * 1024 * 1024;
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
const OAUTH_STATE_COOKIE_NAME = 'wub_oauth_state';
const OAUTH_STATE_MAX_AGE_MS = 10 * 60 * 1000;
const OAUTH_STATE_MAX_AGE_SECONDS = Math.trunc(OAUTH_STATE_MAX_AGE_MS / 1000);
const OAUTH_GOOGLE_AUTHORIZE_URL =
  'https://accounts.google.com/o/oauth2/v2/auth';
const OAUTH_GOOGLE_TOKEN_URL = 'https://oauth2.googleapis.com/token';
const OAUTH_GOOGLE_USERINFO_URL =
  'https://openidconnect.googleapis.com/v1/userinfo';
const OAUTH_DISCORD_AUTHORIZE_URL = 'https://discord.com/oauth2/authorize';
const OAUTH_DISCORD_TOKEN_URL = 'https://discord.com/api/oauth2/token';
const OAUTH_DISCORD_USERINFO_URL = 'https://discord.com/api/v10/users/@me';
const OAUTH_GOOGLE_SCOPES = 'openid email profile';
const OAUTH_DISCORD_SCOPES = 'identify email';
const CORS_BASE_HEADERS = {
  'access-control-allow-origin': '*',
  'access-control-allow-headers': 'content-type',
  'access-control-allow-methods': 'GET,POST,PUT,OPTIONS',
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

const redirectResponse = (
  location: string,
  status = 302,
  headers?: HeadersInit,
): Response => {
  const responseHeaders = withBaseHeaders(headers);
  responseHeaders.set('location', location);
  return new Response(null, {
    status,
    headers: responseHeaders,
  });
};

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

const asBoolean = (value: unknown): boolean | null => {
  if (typeof value === 'boolean') return value;
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (normalized === 'true') return true;
    if (normalized === 'false') return false;
  }
  return null;
};

const binaryResponse = (
  body: BodyInit,
  status = 200,
  headers?: HeadersInit,
): Response =>
  new Response(body, {
    status,
    headers: withBaseHeaders(headers),
  });

const normalizeGameMode = (value: unknown): string | null => {
  if (typeof value !== 'string') return null;
  const mode = value.trim().toLowerCase();
  if (!mode) return null;
  if (mode.length > 64) return null;
  if (!/^[a-z0-9_-]+$/.test(mode)) return null;
  return mode;
};

const readGameModeFromRequest = (request: Request): string | null => {
  const mode = new URL(request.url).searchParams.get('mode');
  return normalizeGameMode(mode);
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

const oauthStateCookieHeader = (
  provider: OAuthProvider,
  state: string,
  secure: boolean,
): string => {
  const parts = [
    `${OAUTH_STATE_COOKIE_NAME}=${encodeURIComponent(`${provider}:${state}`)}`,
    'Path=/',
    'HttpOnly',
    'SameSite=Lax',
    `Max-Age=${OAUTH_STATE_MAX_AGE_SECONDS}`,
  ];
  if (secure) parts.push('Secure');
  return parts.join('; ');
};

const clearOAuthStateCookieHeader = (secure: boolean): string => {
  const parts = [
    `${OAUTH_STATE_COOKIE_NAME}=`,
    'Path=/',
    'HttpOnly',
    'SameSite=Lax',
    'Max-Age=0',
    'Expires=Thu, 01 Jan 1970 00:00:00 GMT',
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

type OAuthProvider = 'google' | 'discord';

type OAuthResolvedProfile = {
  providerSub: string;
  emailNorm: string | null;
  emailVerified: boolean;
  usernameHint: string;
};

type OAuthProviderConfig = {
  clientId: string;
  clientSecret: string;
  authorizeUrl: string;
  tokenUrl: string;
  userInfoUrl: string;
  scope: string;
};

type UserLookupRecord = AuthUser & {
  passwordHash: string | null;
  googleSub: string | null;
  discordSub: string | null;
};

type AuthTokenType = 'email_verify' | 'password_reset';

type AuthTokenRecord = {
  id: string;
  userId: string;
  emailNorm: string | null;
  expiresAtMs: number;
};

type PersonalModelRecord = {
  userId: string;
  gameMode: string;
  r2Key: string;
  version: number;
  modelSizeBytes: number | null;
  modelSha256: string | null;
  updatedAtMs: number;
};

const readPersonalModelRecord = (
  row: Record<string, unknown>,
): PersonalModelRecord | null => {
  const userId = asString(row.user_id);
  const gameMode = asString(row.game_mode);
  const r2Key = asString(row.r2_key);
  const version = asInt(row.version);
  const updatedAtMs = asInt(row.updated_at_ms);
  if (
    !userId ||
    !gameMode ||
    !r2Key ||
    version == null ||
    updatedAtMs == null
  ) {
    return null;
  }
  return {
    userId,
    gameMode,
    r2Key,
    version,
    modelSizeBytes: asInt(row.model_size_bytes),
    modelSha256: asString(row.model_sha256),
    updatedAtMs,
  };
};

const readCurrentPersonalModel = async (
  env: Env,
  userId: string,
  gameMode: string,
): Promise<PersonalModelRecord | null> => {
  const row = await env.DB.prepare(
    `SELECT
       user_id,
       game_mode,
       r2_key,
       version,
       model_size_bytes,
       model_sha256,
       updated_at_ms
     FROM user_models
     WHERE user_id = ? AND game_mode = ?
     LIMIT 1`,
  )
    .bind(userId, gameMode)
    .first<Record<string, unknown>>();
  if (!row) return null;
  return readPersonalModelRecord(row);
};

const sha256HexFromBuffer = async (buffer: ArrayBuffer): Promise<string> => {
  const digest = await crypto.subtle.digest('SHA-256', buffer);
  return toHex(new Uint8Array(digest));
};

const getOAuthProviderColumn = (
  provider: OAuthProvider,
): 'google_sub' | 'discord_sub' =>
  provider === 'google' ? 'google_sub' : 'discord_sub';

const getOAuthProviderConfig = (
  env: Env,
  provider: OAuthProvider,
): OAuthProviderConfig | null => {
  if (provider === 'google') {
    const clientId = asString(env.GOOGLE_OAUTH_CLIENT_ID);
    const clientSecret = asString(env.GOOGLE_OAUTH_CLIENT_SECRET);
    if (!clientId || !clientSecret) return null;
    return {
      clientId,
      clientSecret,
      authorizeUrl: OAUTH_GOOGLE_AUTHORIZE_URL,
      tokenUrl: OAUTH_GOOGLE_TOKEN_URL,
      userInfoUrl: OAUTH_GOOGLE_USERINFO_URL,
      scope: OAUTH_GOOGLE_SCOPES,
    };
  }
  const clientId = asString(env.DISCORD_OAUTH_CLIENT_ID);
  const clientSecret = asString(env.DISCORD_OAUTH_CLIENT_SECRET);
  if (!clientId || !clientSecret) return null;
  return {
    clientId,
    clientSecret,
    authorizeUrl: OAUTH_DISCORD_AUTHORIZE_URL,
    tokenUrl: OAUTH_DISCORD_TOKEN_URL,
    userInfoUrl: OAUTH_DISCORD_USERINFO_URL,
    scope: OAUTH_DISCORD_SCOPES,
  };
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
       password_hash,
       google_sub,
       discord_sub
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
    googleSub: asString(row.google_sub),
    discordSub: asString(row.discord_sub),
  };
};

const readUserByOAuthSub = async (
  env: Env,
  provider: OAuthProvider,
  providerSub: string,
): Promise<UserLookupRecord | null> => {
  const providerColumn = getOAuthProviderColumn(provider);
  const row = await env.DB.prepare(
    `SELECT
       id,
       username,
       email_norm,
       email_verified_at_ms,
       password_hash,
       google_sub,
       discord_sub
     FROM users
     WHERE ${providerColumn} = ?
     LIMIT 1`,
  )
    .bind(providerSub)
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
    googleSub: asString(row.google_sub),
    discordSub: asString(row.discord_sub),
  };
};

const normalizeOAuthUsername = (
  usernameHint: string,
  emailNorm: string | null,
): string => {
  if (emailNorm) {
    return normalizeUsername(usernameHint, emailNorm);
  }
  const collapsed = usernameHint.trim().replace(/\s+/g, ' ');
  const safe = collapsed.replace(/[^A-Za-z0-9._ -]/g, '').trim();
  if (
    safe.length >= MIN_USERNAME_LENGTH &&
    safe.length <= MAX_USERNAME_LENGTH
  ) {
    return safe;
  }
  const trimmed = safe.slice(0, MAX_USERNAME_LENGTH).trim();
  if (trimmed.length >= MIN_USERNAME_LENGTH) {
    return trimmed;
  }
  return 'player';
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

const resolveOAuthCallbackUrl = (
  request: Request,
  provider: OAuthProvider,
): string => {
  const origin = new URL(request.url).origin;
  return `${origin}/api/auth/oauth/${provider}/callback`;
};

const resolveOAuthLandingUrl = (request: Request, env: Env): string =>
  new URL('/', `${resolveAuthAppBaseUrl(request, env)}/`).toString();

const resolveOAuthErrorUrl = (
  request: Request,
  env: Env,
  errorCode: string,
): string => {
  const url = new URL(resolveOAuthLandingUrl(request, env));
  url.searchParams.set('auth_error', errorCode);
  return url.toString();
};

const readOAuthStateFromRequest = (
  request: Request,
  provider: OAuthProvider,
): string | null => {
  const cookies = parseCookies(request.headers.get('cookie'));
  const cookieValue = cookies[OAUTH_STATE_COOKIE_NAME];
  if (!cookieValue) return null;
  const separatorIdx = cookieValue.indexOf(':');
  if (separatorIdx <= 0) return null;
  const cookieProvider = cookieValue.slice(0, separatorIdx);
  const state = cookieValue.slice(separatorIdx + 1);
  if (cookieProvider !== provider) return null;
  if (!isLikelyOpaqueToken(state)) return null;
  return state;
};

const buildOAuthAuthorizeUrl = (
  provider: OAuthProvider,
  config: OAuthProviderConfig,
  redirectUri: string,
  state: string,
): string => {
  const params = new URLSearchParams({
    client_id: config.clientId,
    redirect_uri: redirectUri,
    response_type: 'code',
    scope: config.scope,
    state,
  });
  if (provider === 'google') {
    params.set('include_granted_scopes', 'true');
    params.set('access_type', 'online');
  }
  return `${config.authorizeUrl}?${params.toString()}`;
};

const exchangeOAuthCodeForAccessToken = async (
  config: OAuthProviderConfig,
  code: string,
  redirectUri: string,
): Promise<string> => {
  const body = new URLSearchParams({
    grant_type: 'authorization_code',
    code,
    redirect_uri: redirectUri,
    client_id: config.clientId,
    client_secret: config.clientSecret,
  });
  const response = await fetch(config.tokenUrl, {
    method: 'POST',
    headers: {
      'content-type': 'application/x-www-form-urlencoded',
    },
    body: body.toString(),
  });
  const payload = (await response.json().catch(() => null)) as Record<
    string,
    unknown
  > | null;
  if (!response.ok || !payload || typeof payload !== 'object') {
    throw new Error('OAuth token exchange failed.');
  }
  const accessToken = asString(payload.access_token);
  if (!accessToken) {
    throw new Error('OAuth token exchange returned no access token.');
  }
  return accessToken;
};

const fetchOAuthProfile = async (
  provider: OAuthProvider,
  config: OAuthProviderConfig,
  accessToken: string,
): Promise<OAuthResolvedProfile> => {
  const response = await fetch(config.userInfoUrl, {
    method: 'GET',
    headers: {
      Authorization: `Bearer ${accessToken}`,
      Accept: 'application/json',
    },
  });
  const payload = (await response.json().catch(() => null)) as Record<
    string,
    unknown
  > | null;
  if (!response.ok || !payload || typeof payload !== 'object') {
    throw new Error('OAuth profile request failed.');
  }

  if (provider === 'google') {
    const providerSub = asString(payload.sub);
    if (!providerSub) {
      throw new Error('Google OAuth profile is missing subject identifier.');
    }
    const emailNorm = normalizeEmail(payload.email);
    const emailVerified = asBoolean(payload.email_verified) === true;
    const usernameHint =
      asString(payload.name) ??
      asString(payload.given_name) ??
      emailNorm ??
      'player';
    return {
      providerSub,
      emailNorm,
      emailVerified,
      usernameHint,
    };
  }

  const providerSub = asString(payload.id);
  if (!providerSub) {
    throw new Error('Discord OAuth profile is missing subject identifier.');
  }
  const emailNorm = normalizeEmail(payload.email);
  const emailVerified = asBoolean(payload.verified) === true;
  const usernameHint =
    asString(payload.global_name) ?? asString(payload.username) ?? 'player';
  return {
    providerSub,
    emailNorm,
    emailVerified,
    usernameHint,
  };
};

const maybeUpdateOAuthLinkedUser = async (
  env: Env,
  user: UserLookupRecord,
  provider: OAuthProvider,
  profile: OAuthResolvedProfile,
  nowMs: number,
): Promise<void> => {
  const providerColumn = getOAuthProviderColumn(provider);
  const existingProviderSub =
    provider === 'google' ? user.googleSub : user.discordSub;
  if (existingProviderSub && existingProviderSub !== profile.providerSub) {
    throw new Error(`OAuth ${provider} account is linked to a different user.`);
  }

  const updates: string[] = [];
  const values: unknown[] = [];
  if (!existingProviderSub) {
    updates.push(`${providerColumn} = ?`);
    values.push(profile.providerSub);
  }
  if (!user.emailNorm && profile.emailNorm) {
    updates.push('email_norm = ?');
    values.push(profile.emailNorm);
  }
  const canVerifyEmail =
    profile.emailNorm != null &&
    profile.emailVerified &&
    user.emailVerifiedAtMs == null &&
    (user.emailNorm == null || user.emailNorm === profile.emailNorm);
  if (canVerifyEmail) {
    updates.push('email_verified_at_ms = ?');
    values.push(nowMs);
  }
  if (updates.length === 0) return;

  updates.push('updated_at_ms = ?');
  values.push(nowMs);
  await env.DB.prepare(
    `UPDATE users
     SET ${updates.join(', ')}
     WHERE id = ?`,
  )
    .bind(...values, user.id)
    .run();
};

const resolveOAuthUser = async (
  env: Env,
  provider: OAuthProvider,
  profile: OAuthResolvedProfile,
  nowMs: number,
): Promise<AuthUser> => {
  const byProvider = await readUserByOAuthSub(
    env,
    provider,
    profile.providerSub,
  );
  if (byProvider) {
    await maybeUpdateOAuthLinkedUser(env, byProvider, provider, profile, nowMs);
    const user = await readUserById(env, byProvider.id);
    if (user) return user;
    throw new Error('Linked OAuth user could not be loaded.');
  }

  if (profile.emailNorm) {
    const byEmail = await readUserByEmail(env, profile.emailNorm);
    if (byEmail) {
      await maybeUpdateOAuthLinkedUser(env, byEmail, provider, profile, nowMs);
      const user = await readUserById(env, byEmail.id);
      if (user) return user;
      throw new Error('Email-linked OAuth user could not be loaded.');
    }
  }

  const userId = crypto.randomUUID();
  const providerColumn = getOAuthProviderColumn(provider);
  const emailVerifiedAtMs =
    profile.emailNorm && profile.emailVerified ? nowMs : null;
  const username = normalizeOAuthUsername(
    profile.usernameHint,
    profile.emailNorm,
  );

  try {
    await env.DB.prepare(
      `INSERT INTO users (
         id,
         username,
         email_norm,
         email_verified_at_ms,
         ${providerColumn},
         created_at_ms,
         updated_at_ms
       ) VALUES (?, ?, ?, ?, ?, ?, ?)`,
    )
      .bind(
        userId,
        username,
        profile.emailNorm,
        emailVerifiedAtMs,
        profile.providerSub,
        nowMs,
        nowMs,
      )
      .run();
  } catch (error) {
    const retriedByProvider = await readUserByOAuthSub(
      env,
      provider,
      profile.providerSub,
    );
    if (retriedByProvider) {
      await maybeUpdateOAuthLinkedUser(
        env,
        retriedByProvider,
        provider,
        profile,
        nowMs,
      );
      const user = await readUserById(env, retriedByProvider.id);
      if (user) return user;
    }
    if (profile.emailNorm) {
      const retriedByEmail = await readUserByEmail(env, profile.emailNorm);
      if (retriedByEmail) {
        await maybeUpdateOAuthLinkedUser(
          env,
          retriedByEmail,
          provider,
          profile,
          nowMs,
        );
        const user = await readUserById(env, retriedByEmail.id);
        if (user) return user;
      }
    }
    throw error;
  }

  const createdUser = await readUserById(env, userId);
  if (!createdUser) {
    throw new Error('Created OAuth user could not be loaded.');
  }
  return createdUser;
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

const handleOAuthStart = async (
  request: Request,
  env: Env,
  provider: OAuthProvider,
): Promise<Response> => {
  if (request.method !== 'GET') {
    return jsonResponse({ error: 'Method not allowed.' }, 405);
  }

  const config = getOAuthProviderConfig(env, provider);
  if (!config) {
    return jsonResponse({ error: 'OAuth provider is not configured.' }, 503, {
      'cache-control': 'no-store',
    });
  }

  const state = generateOpaqueToken(24);
  const secure = isSecureRequest(request);
  const redirectUri = resolveOAuthCallbackUrl(request, provider);
  const authorizeUrl = buildOAuthAuthorizeUrl(
    provider,
    config,
    redirectUri,
    state,
  );

  return redirectResponse(authorizeUrl, 302, {
    'cache-control': 'no-store',
    'set-cookie': oauthStateCookieHeader(provider, state, secure),
  });
};

const handleOAuthCallback = async (
  request: Request,
  env: Env,
  provider: OAuthProvider,
): Promise<Response> => {
  if (request.method !== 'GET') {
    return jsonResponse({ error: 'Method not allowed.' }, 405);
  }

  const secure = isSecureRequest(request);
  const clearStateCookie = clearOAuthStateCookieHeader(secure);
  const config = getOAuthProviderConfig(env, provider);
  if (!config) {
    return redirectResponse(
      resolveOAuthErrorUrl(request, env, 'oauth_not_configured'),
      302,
      {
        'cache-control': 'no-store',
        'set-cookie': clearStateCookie,
      },
    );
  }

  const url = new URL(request.url);
  if (url.searchParams.get('error')) {
    return redirectResponse(
      resolveOAuthErrorUrl(request, env, 'oauth_denied'),
      302,
      {
        'cache-control': 'no-store',
        'set-cookie': clearStateCookie,
      },
    );
  }

  const stateParam = asString(url.searchParams.get('state'));
  const code = asString(url.searchParams.get('code'));
  const stateCookie = readOAuthStateFromRequest(request, provider);
  if (!stateParam || !code || !stateCookie || stateParam !== stateCookie) {
    return redirectResponse(
      resolveOAuthErrorUrl(request, env, 'oauth_state_mismatch'),
      302,
      {
        'cache-control': 'no-store',
        'set-cookie': clearStateCookie,
      },
    );
  }

  try {
    const redirectUri = resolveOAuthCallbackUrl(request, provider);
    const accessToken = await exchangeOAuthCodeForAccessToken(
      config,
      code,
      redirectUri,
    );
    const profile = await fetchOAuthProfile(provider, config, accessToken);
    if (provider === 'google') {
      profile.emailVerified = true;
    }

    const nowMs = Date.now();
    const user = await resolveOAuthUser(env, provider, profile, nowMs);
    const session = await createSessionForUser(env, user.id, nowMs);

    const headers = withBaseHeaders({
      'cache-control': 'no-store',
      location: resolveOAuthLandingUrl(request, env),
    });
    headers.append('set-cookie', clearStateCookie);
    headers.append('set-cookie', sessionCookieHeader(session.token, secure));
    return new Response(null, {
      status: 302,
      headers,
    });
  } catch (error) {
    console.error(`[auth] oauth ${provider} callback failed`, error);
    return redirectResponse(
      resolveOAuthErrorUrl(request, env, 'oauth_failed'),
      302,
      {
        'cache-control': 'no-store',
        'set-cookie': clearStateCookie,
      },
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

const handleGetCurrentPersonalModel = async (
  request: Request,
  env: Env,
): Promise<Response> => {
  if (request.method !== 'GET') {
    return jsonResponse({ error: 'Method not allowed.' }, 405);
  }
  const gameMode = readGameModeFromRequest(request);
  if (!gameMode) {
    return jsonResponse({ error: 'Missing or invalid mode.' }, 400, {
      'cache-control': 'no-store',
    });
  }
  if (!env.MODELS_BUCKET) {
    return jsonResponse({ error: 'Model storage is not configured.' }, 503, {
      'cache-control': 'no-store',
    });
  }

  const nowMs = Date.now();
  const auth = await requireAuthenticatedSession(request, env, nowMs);
  if (auth.response) return auth.response;
  const session = auth.session!;
  try {
    await touchSessionIfStale(env, session, nowMs);
  } catch (error) {
    console.error('[models] touch session failed', error);
  }

  try {
    const record = await readCurrentPersonalModel(
      env,
      session.userId,
      gameMode,
    );
    if (!record) {
      return jsonResponse({ error: 'Model not found.' }, 404, {
        'cache-control': 'no-store',
      });
    }
    const object = await env.MODELS_BUCKET.get(record.r2Key);
    if (!object) {
      return jsonResponse({ error: 'Model blob is missing.' }, 404, {
        'cache-control': 'no-store',
      });
    }
    const bytes = await object.arrayBuffer();
    const responseHeaders = withBaseHeaders({
      'cache-control': 'no-store',
      'content-type': 'application/octet-stream',
      'x-wub-model-mode': record.gameMode,
      'x-wub-model-version': String(record.version),
      'x-wub-model-size': String(bytes.byteLength),
      'x-wub-model-updated-at-ms': String(record.updatedAtMs),
    });
    if (record.modelSha256) {
      responseHeaders.set('x-wub-model-sha256', record.modelSha256);
    }
    return binaryResponse(bytes, 200, responseHeaders);
  } catch (error) {
    console.error('[models] current model get failed', error);
    return jsonResponse(
      { error: 'Model storage is currently unavailable.' },
      503,
      { 'cache-control': 'no-store' },
    );
  }
};

const handlePutCurrentPersonalModel = async (
  request: Request,
  env: Env,
): Promise<Response> => {
  if (request.method !== 'PUT') {
    return jsonResponse({ error: 'Method not allowed.' }, 405);
  }
  const gameMode = readGameModeFromRequest(request);
  if (!gameMode) {
    return jsonResponse({ error: 'Missing or invalid mode.' }, 400, {
      'cache-control': 'no-store',
    });
  }
  if (!env.MODELS_BUCKET) {
    return jsonResponse({ error: 'Model storage is not configured.' }, 503, {
      'cache-control': 'no-store',
    });
  }

  const nowMs = Date.now();
  const auth = await requireAuthenticatedSession(request, env, nowMs);
  if (auth.response) return auth.response;
  const session = auth.session!;
  try {
    await touchSessionIfStale(env, session, nowMs);
  } catch (error) {
    console.error('[models] touch session failed', error);
  }

  let modelBuffer: ArrayBuffer;
  try {
    modelBuffer = await request.arrayBuffer();
  } catch {
    return jsonResponse({ error: 'Invalid model payload.' }, 400, {
      'cache-control': 'no-store',
    });
  }
  if (modelBuffer.byteLength <= 0) {
    return jsonResponse({ error: 'Model payload is empty.' }, 400, {
      'cache-control': 'no-store',
    });
  }
  if (modelBuffer.byteLength > MAX_PERSONALIZED_MODEL_BYTES) {
    return jsonResponse(
      {
        error: `Model payload is too large. Max bytes: ${MAX_PERSONALIZED_MODEL_BYTES}.`,
      },
      413,
      { 'cache-control': 'no-store' },
    );
  }

  try {
    const existing = await readCurrentPersonalModel(
      env,
      session.userId,
      gameMode,
    );
    const nextVersion = (existing?.version ?? 0) + 1;
    const nextR2Key = `models/${session.userId}/${gameMode}/v${nextVersion}-${nowMs}.bin`;
    const modelSha256 = await sha256HexFromBuffer(modelBuffer);
    const contentType =
      asString(request.headers.get('content-type')) ??
      'application/octet-stream';

    await env.MODELS_BUCKET.put(nextR2Key, modelBuffer, {
      httpMetadata: { contentType },
    });
    await env.DB.prepare(
      `INSERT INTO user_models (
         user_id,
         game_mode,
         r2_key,
         version,
         model_size_bytes,
         model_sha256,
         updated_at_ms
       ) VALUES (?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(user_id, game_mode) DO UPDATE SET
         r2_key = excluded.r2_key,
         version = excluded.version,
         model_size_bytes = excluded.model_size_bytes,
         model_sha256 = excluded.model_sha256,
         updated_at_ms = excluded.updated_at_ms`,
    )
      .bind(
        session.userId,
        gameMode,
        nextR2Key,
        nextVersion,
        modelBuffer.byteLength,
        modelSha256,
        nowMs,
      )
      .run();

    if (existing && existing.r2Key !== nextR2Key) {
      env.MODELS_BUCKET.delete(existing.r2Key).catch((error) => {
        console.warn(
          `[models] failed to delete previous model blob (${existing.r2Key})`,
          error,
        );
      });
    }

    return jsonResponse(
      {
        ok: true,
        model: {
          mode: gameMode,
          version: nextVersion,
          sizeBytes: modelBuffer.byteLength,
          sha256: modelSha256,
          updatedAtMs: nowMs,
        },
      },
      200,
      { 'cache-control': 'no-store' },
    );
  } catch (error) {
    console.error('[models] current model put failed', error);
    return jsonResponse(
      { error: 'Model storage is currently unavailable.' },
      503,
      { 'cache-control': 'no-store' },
    );
  }
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

    const oauthRoute = url.pathname.match(
      /^\/api\/auth\/oauth\/(google|discord)\/(start|callback)$/,
    );
    if (oauthRoute) {
      const provider = oauthRoute[1] as OAuthProvider;
      const action = oauthRoute[2];
      if (action === 'start') {
        return handleOAuthStart(request, env, provider);
      }
      return handleOAuthCallback(request, env, provider);
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

    if (url.pathname === '/api/models/me/current') {
      if (request.method === 'GET') {
        return handleGetCurrentPersonalModel(request, env);
      }
      if (request.method === 'PUT') {
        return handlePutCurrentPersonalModel(request, env);
      }
      return jsonResponse({ error: 'Method not allowed.' }, 405);
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
