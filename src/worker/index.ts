import {
  MIN_TRAJECTORY_SAMPLES_PER_SESSION,
  MAX_TRAJECTORY_SAMPLES_PER_SESSION,
  parseTrajectorySessionV1,
} from '../core/trajectoryProtocol';

type D1PreparedStatement = {
  bind: (...values: unknown[]) => D1PreparedStatement;
  run: () => Promise<unknown>;
  all: <T = Record<string, unknown>>() => Promise<{
    results?: T[];
  }>;
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
  AUTH_ADMIN_EMAILS?: string;
  AUTH_ADMIN_USER_IDS?: string;
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
const MAX_TRAJECTORY_UPLOAD_BYTES = 3 * 1024 * 1024;
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
const MAX_MODEL_AXIS_LENGTH = 64;
const RESEND_SEND_EMAIL_URL = 'https://api.resend.com/emails';
const DEFAULT_MODEL_ARCH = 'full';
const DEFAULT_REWARD_PROFILE_ID = 'default';
const DEFAULT_QUEUE_POLICY_ID = 'next_piece_v1';
const DEFAULT_PERSONAL_MODEL_SOURCE = 'upload';
const DEFAULT_BOT_ARCH_ID = 'full';
const DEFAULT_BOT_QUEUE_POLICY_ID = 'default';
const DEFAULT_BOT_PIPELINE_ID = 'bot_reinforce_v2';
const DEFAULT_BOT_PIECE_SOURCE_PROFILE = 'bag7';
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

const clampInt = (
  value: number | null,
  options: { min: number; max: number; fallback: number },
): number => {
  if (value == null || !Number.isFinite(value)) return options.fallback;
  return Math.max(options.min, Math.min(options.max, Math.trunc(value)));
};

const isDevReleaseChannel = (env: Env): boolean => {
  const channel = asString(env.RELEASE_CHANNEL)?.toLowerCase() ?? '';
  return (
    channel.includes('dev') ||
    channel.includes('staging') ||
    channel.includes('preview') ||
    channel === 'test'
  );
};

const errorDetailMessage = (error: unknown): string => {
  if (error instanceof Error) {
    const message = error.message?.trim();
    return message.length > 0 ? message : 'Unknown Error';
  }
  if (typeof error === 'string') {
    const message = error.trim();
    return message.length > 0 ? message : 'Unknown Error';
  }
  try {
    const serialized = JSON.stringify(error);
    return serialized.length > 0 ? serialized : 'Unknown Error';
  } catch {
    return 'Unknown Error';
  }
};

const isValidRecordingId = (value: string): boolean =>
  value.length >= 8 && value.length <= 256 && /^[A-Za-z0-9._:-]+$/.test(value);

const isValidGlobalModelId = (value: string): boolean =>
  value.length >= 8 && value.length <= 128 && /^[A-Za-z0-9._:-]+$/.test(value);

const isValidBotPolicyId = (value: string): boolean =>
  value.length >= 8 && value.length <= 128 && /^[A-Za-z0-9._:-]+$/.test(value);

type RecordingCursor = {
  startedAtMs: number;
  id: string;
};

type BotPolicyCursor = {
  createdAtMs: number;
  id: string;
};

const parseRecordingsCursor = (
  value: string | null,
): RecordingCursor | null => {
  if (!value) return null;
  const trimmed = value.trim();
  if (!trimmed) return null;
  const separator = trimmed.indexOf(':');
  if (separator <= 0 || separator >= trimmed.length - 1) return null;
  const startedAtMs = asInt(trimmed.slice(0, separator));
  const id = trimmed.slice(separator + 1);
  if (startedAtMs == null || startedAtMs < 0 || !isValidRecordingId(id)) {
    return null;
  }
  return { startedAtMs, id };
};

const formatRecordingsCursor = (cursor: RecordingCursor): string =>
  `${cursor.startedAtMs}:${cursor.id}`;

const parseBotPolicyCursor = (value: string | null): BotPolicyCursor | null => {
  if (!value) return null;
  const trimmed = value.trim();
  if (!trimmed) return null;
  const separator = trimmed.indexOf(':');
  if (separator <= 0 || separator >= trimmed.length - 1) return null;
  const createdAtMs = asInt(trimmed.slice(0, separator));
  const id = trimmed.slice(separator + 1);
  if (createdAtMs == null || createdAtMs < 0 || !isValidBotPolicyId(id)) {
    return null;
  }
  return { createdAtMs, id };
};

const formatBotPolicyCursor = (cursor: BotPolicyCursor): string =>
  `${cursor.createdAtMs}:${cursor.id}`;

const parseJsonObjectString = (
  value: string | null,
): Record<string, unknown> | null => {
  if (!value) return null;
  try {
    const parsed = JSON.parse(value) as unknown;
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      return null;
    }
    return parsed as Record<string, unknown>;
  } catch {
    return null;
  }
};

const normalizeModelAxis = (
  value: string | null | undefined,
  fallback: string,
): string => {
  if (typeof value !== 'string') return fallback;
  const normalized = value.trim().toLowerCase();
  if (!normalized) return fallback;
  if (normalized.length > MAX_MODEL_AXIS_LENGTH) return fallback;
  if (!/^[a-z0-9_-]+$/.test(normalized)) return fallback;
  return normalized;
};

const normalizeOptionalModelAxis = (
  value: string | null | undefined,
): string | null => {
  if (typeof value !== 'string') return null;
  const normalized = value.trim().toLowerCase();
  if (!normalized) return null;
  if (normalized.length > MAX_MODEL_AXIS_LENGTH) return null;
  if (!/^[a-z0-9_-]+$/.test(normalized)) return null;
  return normalized;
};

const parseBooleanFlag = (value: string | null): boolean => {
  if (!value) return false;
  const normalized = value.trim().toLowerCase();
  return (
    normalized === '1' ||
    normalized === 'true' ||
    normalized === 'yes' ||
    normalized === 'on'
  );
};

type PersonalModelSelector = {
  gameMode: string;
  modelArch: string;
  rewardProfileId: string;
  queuePolicyId: string;
};

type BotPolicySelector = {
  modeId: string;
  archId: string;
  queuePolicyId: string;
};

const readPersonalModelSelectorFromRequest = (
  request: Request,
): PersonalModelSelector | null => {
  const url = new URL(request.url);
  const gameMode = normalizeGameMode(url.searchParams.get('mode'));
  if (!gameMode) return null;
  return {
    gameMode,
    modelArch: normalizeModelAxis(
      url.searchParams.get('arch'),
      DEFAULT_MODEL_ARCH,
    ),
    rewardProfileId: normalizeModelAxis(
      url.searchParams.get('reward_profile'),
      DEFAULT_REWARD_PROFILE_ID,
    ),
    queuePolicyId: normalizeModelAxis(
      url.searchParams.get('queue_policy'),
      DEFAULT_QUEUE_POLICY_ID,
    ),
  };
};

const readBotPolicySelectorFromRequest = (
  request: Request,
): BotPolicySelector | null => {
  const url = new URL(request.url);
  const modeId = normalizeGameMode(url.searchParams.get('mode'));
  if (!modeId) return null;
  return {
    modeId,
    archId: normalizeModelAxis(
      url.searchParams.get('arch'),
      DEFAULT_BOT_ARCH_ID,
    ),
    queuePolicyId: normalizeModelAxis(
      url.searchParams.get('queue_policy'),
      DEFAULT_BOT_QUEUE_POLICY_ID,
    ),
  };
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

const splitTokenList = (value: string | null): string[] => {
  if (!value) return [];
  return value
    .split(/[,\n;\s]+/g)
    .map((entry) => entry.trim())
    .filter((entry) => entry.length > 0);
};

const isAdminUser = (
  env: Env,
  userId: string,
  emailNorm: string | null,
): boolean => {
  const adminUserIds = splitTokenList(asString(env.AUTH_ADMIN_USER_IDS));
  if (adminUserIds.includes(userId)) return true;
  if (!emailNorm) return false;
  const adminEmails = splitTokenList(asString(env.AUTH_ADMIN_EMAILS)).map(
    (entry) => entry.toLowerCase(),
  );
  return adminEmails.includes(emailNorm.toLowerCase());
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
  isAdmin: boolean;
  expiresAtMs: number;
  lastSeenAtMs: number | null;
};

type AuthUser = {
  id: string;
  username: string;
  emailNorm: string | null;
  emailVerifiedAtMs: number | null;
  isAdmin: boolean;
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
  modelArch: string;
  rewardProfileId: string;
  queuePolicyId: string;
  versionId: string;
  r2Key: string;
  version: number;
  modelSizeBytes: number | null;
  modelSha256: string | null;
  updatedAtMs: number;
  source: string | null;
  pipelineId: string | null;
  baseGlobalModelId: string | null;
};

type GlobalModelRecord = {
  id: string;
  modeId: string;
  modelArch: string;
  rewardProfileId: string;
  queuePolicyId: string;
  pipelineId: string;
  label: string | null;
  r2Key: string;
  sha256: string | null;
  sizeBytes: number | null;
  isDefault: boolean;
  createdAtMs: number;
  updatedAtMs: number;
  retiredAtMs: number | null;
};

type BotPolicyRecord = {
  id: string;
  modeId: string;
  archId: string;
  queuePolicyId: string;
  pipelineId: string;
  pieceSourceProfile: string;
  r2Key: string;
  version: number;
  isPinned: boolean;
  metrics: Record<string, unknown> | null;
  createdAtMs: number;
};

const readPersonalModelRecord = (
  row: Record<string, unknown>,
): PersonalModelRecord | null => {
  const userId = asString(row.user_id);
  const gameMode = asString(row.game_mode);
  const modelArch = asString(row.model_arch);
  const rewardProfileId = asString(row.reward_profile_id);
  const queuePolicyId = asString(row.queue_policy_id);
  const versionId = asString(row.version_id);
  const r2Key = asString(row.r2_key);
  const version = asInt(row.version);
  const updatedAtMs = asInt(row.updated_at_ms);
  if (
    !userId ||
    !gameMode ||
    !modelArch ||
    !rewardProfileId ||
    !queuePolicyId ||
    !versionId ||
    !r2Key ||
    version == null ||
    updatedAtMs == null
  ) {
    return null;
  }
  return {
    userId,
    gameMode,
    modelArch,
    rewardProfileId,
    queuePolicyId,
    versionId,
    r2Key,
    version,
    modelSizeBytes: asInt(row.model_size_bytes),
    modelSha256: asString(row.model_sha256),
    updatedAtMs,
    source: asString(row.source),
    pipelineId: asString(row.pipeline_id),
    baseGlobalModelId: asString(row.base_global_model_id),
  };
};

const readGlobalModelRecord = (
  row: Record<string, unknown>,
): GlobalModelRecord | null => {
  const id = asString(row.id);
  const modeId = normalizeGameMode(row.mode_id);
  const modelArch = asString(row.model_arch);
  const rewardProfileId = asString(row.reward_profile_id);
  const queuePolicyId = asString(row.queue_policy_id);
  const pipelineId = asString(row.pipeline_id);
  const r2Key = asString(row.r2_key);
  const createdAtMs = asInt(row.created_at_ms);
  const updatedAtMs = asInt(row.updated_at_ms);
  if (
    !id ||
    !modeId ||
    !modelArch ||
    !rewardProfileId ||
    !queuePolicyId ||
    !pipelineId ||
    !r2Key ||
    createdAtMs == null ||
    updatedAtMs == null
  ) {
    return null;
  }
  return {
    id,
    modeId,
    modelArch,
    rewardProfileId,
    queuePolicyId,
    pipelineId,
    label: asString(row.label),
    r2Key,
    sha256: asString(row.sha256),
    sizeBytes: asInt(row.size_bytes),
    isDefault: asInt(row.is_default) === 1,
    createdAtMs,
    updatedAtMs,
    retiredAtMs: asInt(row.retired_at_ms),
  };
};

const readBotPolicyRecord = (
  row: Record<string, unknown>,
): BotPolicyRecord | null => {
  const id = asString(row.id);
  const modeId = normalizeGameMode(row.mode_id);
  const archId = asString(row.arch_id);
  const queuePolicyId = asString(row.queue_policy_id);
  const pipelineId = asString(row.pipeline_id);
  const pieceSourceProfile = asString(row.piece_source_profile);
  const r2Key = asString(row.r2_key);
  const version = asInt(row.version);
  const createdAtMs = asInt(row.created_at_ms);
  if (
    !id ||
    !modeId ||
    !archId ||
    !queuePolicyId ||
    !pipelineId ||
    !pieceSourceProfile ||
    !r2Key ||
    version == null ||
    createdAtMs == null
  ) {
    return null;
  }
  return {
    id,
    modeId,
    archId,
    queuePolicyId,
    pipelineId,
    pieceSourceProfile,
    r2Key,
    version,
    isPinned: asInt(row.is_pinned) === 1,
    metrics: parseJsonObjectString(asString(row.metrics_json)),
    createdAtMs,
  };
};

const readCurrentPersonalModel = async (
  env: Env,
  userId: string,
  selector: PersonalModelSelector,
): Promise<PersonalModelRecord | null> => {
  const row = await env.DB.prepare(
    `SELECT
       s.user_id,
       s.game_mode,
       s.model_arch,
       s.reward_profile_id,
       s.queue_policy_id,
       v.id AS version_id,
       v.version_seq AS version,
       v.r2_key,
       v.size_bytes AS model_size_bytes,
       v.sha256 AS model_sha256,
       s.updated_at_ms,
       v.source,
       v.pipeline_id,
       v.base_global_model_id
     FROM personal_model_slots s
     JOIN personal_model_versions v ON v.id = s.active_version_id
     WHERE
       s.user_id = ?
       AND s.game_mode = ?
       AND s.model_arch = ?
       AND s.reward_profile_id = ?
       AND s.queue_policy_id = ?
     LIMIT 1`,
  )
    .bind(
      userId,
      selector.gameMode,
      selector.modelArch,
      selector.rewardProfileId,
      selector.queuePolicyId,
    )
    .first<Record<string, unknown>>();
  if (!row) return null;
  return readPersonalModelRecord(row);
};

const sha256HexFromBuffer = async (buffer: ArrayBuffer): Promise<string> => {
  const digest = await crypto.subtle.digest('SHA-256', buffer);
  return toHex(new Uint8Array(digest));
};

const listGlobalModelsForSelector = async (
  env: Env,
  selector: PersonalModelSelector,
): Promise<GlobalModelRecord[]> => {
  const result = await env.DB.prepare(
    `SELECT
       id,
       mode_id,
       model_arch,
       reward_profile_id,
       queue_policy_id,
       pipeline_id,
       label,
       r2_key,
       sha256,
       size_bytes,
       is_default,
       created_at_ms,
       updated_at_ms,
       retired_at_ms
     FROM global_models
     WHERE
       mode_id = ?
       AND model_arch = ?
       AND reward_profile_id = ?
       AND queue_policy_id = ?
       AND retired_at_ms IS NULL
     ORDER BY is_default DESC, created_at_ms DESC
     LIMIT 200`,
  )
    .bind(
      selector.gameMode,
      selector.modelArch,
      selector.rewardProfileId,
      selector.queuePolicyId,
    )
    .all<Record<string, unknown>>();
  const rows = Array.isArray(result.results) ? result.results : [];
  const out: GlobalModelRecord[] = [];
  for (const row of rows) {
    const model = readGlobalModelRecord(row);
    if (model) out.push(model);
  }
  return out;
};

const readGlobalModelById = async (
  env: Env,
  selector: PersonalModelSelector,
  id: string,
): Promise<GlobalModelRecord | null> => {
  const row = await env.DB.prepare(
    `SELECT
       id,
       mode_id,
       model_arch,
       reward_profile_id,
       queue_policy_id,
       pipeline_id,
       label,
       r2_key,
       sha256,
       size_bytes,
       is_default,
       created_at_ms,
       updated_at_ms,
       retired_at_ms
     FROM global_models
     WHERE
       id = ?
       AND mode_id = ?
       AND model_arch = ?
       AND reward_profile_id = ?
       AND queue_policy_id = ?
       AND retired_at_ms IS NULL
     LIMIT 1`,
  )
    .bind(
      id,
      selector.gameMode,
      selector.modelArch,
      selector.rewardProfileId,
      selector.queuePolicyId,
    )
    .first<Record<string, unknown>>();
  if (!row) return null;
  return readGlobalModelRecord(row);
};

const readGlobalModelByIdAnyAxes = async (
  env: Env,
  id: string,
): Promise<GlobalModelRecord | null> => {
  const row = await env.DB.prepare(
    `SELECT
       id,
       mode_id,
       model_arch,
       reward_profile_id,
       queue_policy_id,
       pipeline_id,
       label,
       r2_key,
       sha256,
       size_bytes,
       is_default,
       created_at_ms,
       updated_at_ms,
       retired_at_ms
     FROM global_models
     WHERE id = ?
     LIMIT 1`,
  )
    .bind(id)
    .first<Record<string, unknown>>();
  if (!row) return null;
  return readGlobalModelRecord(row);
};

const readDefaultGlobalModelId = async (
  env: Env,
  selector: PersonalModelSelector,
): Promise<string | null> => {
  const rows = await listGlobalModelsForSelector(env, selector);
  return rows[0]?.id ?? null;
};

const readBotPolicyById = async (
  env: Env,
  id: string,
): Promise<BotPolicyRecord | null> => {
  const row = await env.DB.prepare(
    `SELECT
       id,
       mode_id,
       arch_id,
       queue_policy_id,
       pipeline_id,
       piece_source_profile,
       r2_key,
       version,
       is_pinned,
       metrics_json,
       created_at_ms
     FROM bot_policies
     WHERE id = ?
     LIMIT 1`,
  )
    .bind(id)
    .first<Record<string, unknown>>();
  if (!row) return null;
  return readBotPolicyRecord(row);
};

const readCurrentBotPolicyForSelector = async (
  env: Env,
  selector: BotPolicySelector,
): Promise<BotPolicyRecord | null> => {
  const row = await env.DB.prepare(
    `SELECT
       p.id,
       p.mode_id,
       p.arch_id,
       p.queue_policy_id,
       p.pipeline_id,
       p.piece_source_profile,
       p.r2_key,
       p.version,
       p.is_pinned,
       p.metrics_json,
       p.created_at_ms
     FROM bot_policy_current c
     JOIN bot_policies p ON p.id = c.bot_policy_id
     WHERE
       c.mode_id = ?
       AND c.arch_id = ?
       AND c.queue_policy_id = ?
     LIMIT 1`,
  )
    .bind(selector.modeId, selector.archId, selector.queuePolicyId)
    .first<Record<string, unknown>>();
  if (!row) return null;
  return readBotPolicyRecord(row);
};

const listBotPoliciesForSelector = async (
  env: Env,
  selector: BotPolicySelector,
  options: {
    limit: number;
    cursor: BotPolicyCursor | null;
  },
): Promise<BotPolicyRecord[]> => {
  const whereParts: string[] = [
    'mode_id = ?',
    'arch_id = ?',
    'queue_policy_id = ?',
  ];
  const values: unknown[] = [
    selector.modeId,
    selector.archId,
    selector.queuePolicyId,
  ];
  if (options.cursor) {
    whereParts.push('(created_at_ms < ? OR (created_at_ms = ? AND id < ?))');
    values.push(
      options.cursor.createdAtMs,
      options.cursor.createdAtMs,
      options.cursor.id,
    );
  }
  const whereClause = `WHERE ${whereParts.join(' AND ')}`;
  const result = await env.DB.prepare(
    `SELECT
       id,
       mode_id,
       arch_id,
       queue_policy_id,
       pipeline_id,
       piece_source_profile,
       r2_key,
       version,
       is_pinned,
       metrics_json,
       created_at_ms
     FROM bot_policies
     ${whereClause}
     ORDER BY created_at_ms DESC, id DESC
     LIMIT ?`,
  )
    .bind(...values, options.limit)
    .all<Record<string, unknown>>();
  const rows = Array.isArray(result.results) ? result.results : [];
  const records: BotPolicyRecord[] = [];
  for (const row of rows) {
    const parsed = readBotPolicyRecord(row);
    if (parsed) records.push(parsed);
  }
  return records;
};

const readNextBotPolicyVersion = async (
  env: Env,
  selector: BotPolicySelector,
): Promise<number> => {
  const row = await env.DB.prepare(
    `SELECT MAX(version) AS max_version
     FROM bot_policies
     WHERE
       mode_id = ?
       AND arch_id = ?
       AND queue_policy_id = ?`,
  )
    .bind(selector.modeId, selector.archId, selector.queuePolicyId)
    .first<Record<string, unknown>>();
  const maxVersion = asInt(row?.max_version);
  return Math.max(0, maxVersion ?? 0) + 1;
};

const setCurrentBotPolicy = async (
  env: Env,
  policy: BotPolicyRecord,
  updatedAtMs: number,
): Promise<void> => {
  await env.DB.prepare(
    `INSERT INTO bot_policy_current (
       mode_id,
       arch_id,
       queue_policy_id,
       bot_policy_id,
       updated_at_ms
     ) VALUES (?, ?, ?, ?, ?)
     ON CONFLICT(mode_id, arch_id, queue_policy_id)
     DO UPDATE SET
       bot_policy_id = excluded.bot_policy_id,
       updated_at_ms = excluded.updated_at_ms`,
  )
    .bind(
      policy.modeId,
      policy.archId,
      policy.queuePolicyId,
      policy.id,
      updatedAtMs,
    )
    .run();
};

const writeCurrentPersonalModelVersion = async (
  env: Env,
  options: {
    existing: PersonalModelRecord | null;
    userId: string;
    selector: PersonalModelSelector;
    r2Key: string;
    version: number;
    modelSizeBytes: number;
    modelSha256: string;
    updatedAtMs: number;
    source?: string;
    pipelineId?: string | null;
    baseGlobalModelId?: string | null;
  },
): Promise<PersonalModelRecord> => {
  const {
    existing,
    userId,
    selector,
    r2Key,
    version,
    modelSizeBytes,
    modelSha256,
    updatedAtMs,
    source,
    pipelineId,
    baseGlobalModelId,
  } = options;
  const versionId = crypto.randomUUID();
  const parentVersionId = existing?.versionId ?? null;
  const resolvedBaseGlobalModelId =
    baseGlobalModelId ??
    existing?.baseGlobalModelId ??
    (await readDefaultGlobalModelId(env, selector));
  const sourceTag = normalizeModelAxis(source, DEFAULT_PERSONAL_MODEL_SOURCE);
  const normalizedPipelineId = asString(pipelineId);

  await env.DB.prepare(
    `INSERT INTO personal_model_versions (
       id,
       user_id,
       game_mode,
       model_arch,
       reward_profile_id,
       queue_policy_id,
       version_seq,
       parent_version_id,
       base_global_model_id,
       source,
       pipeline_id,
       r2_key,
       sha256,
       size_bytes,
       metrics_json,
       created_at_ms
     ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  )
    .bind(
      versionId,
      userId,
      selector.gameMode,
      selector.modelArch,
      selector.rewardProfileId,
      selector.queuePolicyId,
      version,
      parentVersionId,
      resolvedBaseGlobalModelId,
      sourceTag,
      normalizedPipelineId,
      r2Key,
      modelSha256,
      modelSizeBytes,
      null,
      updatedAtMs,
    )
    .run();

  await env.DB.prepare(
    `INSERT INTO personal_model_slots (
       user_id,
       game_mode,
       model_arch,
       reward_profile_id,
       queue_policy_id,
       active_version_id,
       updated_at_ms,
       created_at_ms
     ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT(user_id, game_mode, model_arch, reward_profile_id, queue_policy_id)
     DO UPDATE SET
       active_version_id = excluded.active_version_id,
       updated_at_ms = excluded.updated_at_ms`,
  )
    .bind(
      userId,
      selector.gameMode,
      selector.modelArch,
      selector.rewardProfileId,
      selector.queuePolicyId,
      versionId,
      updatedAtMs,
      updatedAtMs,
    )
    .run();

  return {
    userId,
    gameMode: selector.gameMode,
    modelArch: selector.modelArch,
    rewardProfileId: selector.rewardProfileId,
    queuePolicyId: selector.queuePolicyId,
    versionId,
    r2Key,
    version,
    modelSizeBytes,
    modelSha256,
    updatedAtMs,
    source: sourceTag,
    pipelineId: normalizedPipelineId,
    baseGlobalModelId: resolvedBaseGlobalModelId,
  };
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
  const resolvedEmailNorm = asString(row.email_norm);
  return {
    id,
    username,
    emailNorm: resolvedEmailNorm,
    emailVerifiedAtMs: asInt(row.email_verified_at_ms),
    isAdmin: isAdminUser(env, id, resolvedEmailNorm),
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
  const resolvedEmailNorm = asString(row.email_norm);
  return {
    id,
    username,
    emailNorm: resolvedEmailNorm,
    emailVerifiedAtMs: asInt(row.email_verified_at_ms),
    isAdmin: isAdminUser(env, id, resolvedEmailNorm),
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
  const emailNorm = asString(row.email_norm);
  return {
    id,
    username,
    emailNorm,
    emailVerifiedAtMs: asInt(row.email_verified_at_ms),
    isAdmin: isAdminUser(env, id, emailNorm),
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
    isAdmin: user.isAdmin,
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
  const emailNorm = asString(row.email_norm);

  return {
    sessionId,
    userId,
    username,
    emailNorm,
    emailVerifiedAtMs: asInt(row.email_verified_at_ms),
    isAdmin: isAdminUser(env, userId, emailNorm),
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

const requireAdminSession = async (
  request: Request,
  env: Env,
  nowMs: number,
): Promise<{ session: SessionContext | null; response?: Response }> => {
  const auth = await requireAuthenticatedSession(request, env, nowMs);
  if (auth.response) return auth;
  if (!auth.session || !auth.session.isAdmin) {
    return {
      session: null,
      response: jsonResponse({ error: 'Forbidden.' }, 403, {
        'cache-control': 'no-store',
      }),
    };
  }
  return auth;
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
      isAdmin: isAdminUser(env, userId, emailNorm),
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
      isAdmin: isAdminUser(env, user.id, user.emailNorm),
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
          isAdmin: user.isAdmin,
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
        isAdmin: session.isAdmin,
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
  const selector = readPersonalModelSelectorFromRequest(request);
  if (!selector) {
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
      selector,
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
      'x-wub-model-arch': record.modelArch,
      'x-wub-model-reward-profile': record.rewardProfileId,
      'x-wub-model-queue-policy': record.queuePolicyId,
      'x-wub-model-version': String(record.version),
      'x-wub-model-version-id': record.versionId,
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
  const selector = readPersonalModelSelectorFromRequest(request);
  if (!selector) {
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
      selector,
    );
    const nextVersion = (existing?.version ?? 0) + 1;
    const nextR2Key =
      `models/${session.userId}` +
      `/${selector.gameMode}` +
      `/${selector.modelArch}` +
      `/${selector.rewardProfileId}` +
      `/${selector.queuePolicyId}` +
      `/v${nextVersion}-${nowMs}.bin`;
    const modelSha256 = await sha256HexFromBuffer(modelBuffer);
    const contentType =
      asString(request.headers.get('content-type')) ??
      'application/octet-stream';

    await env.MODELS_BUCKET.put(nextR2Key, modelBuffer, {
      httpMetadata: { contentType },
    });
    const nextRecord = await writeCurrentPersonalModelVersion(env, {
      existing,
      userId: session.userId,
      selector,
      r2Key: nextR2Key,
      version: nextVersion,
      modelSizeBytes: modelBuffer.byteLength,
      modelSha256,
      updatedAtMs: nowMs,
      source: DEFAULT_PERSONAL_MODEL_SOURCE,
      pipelineId: null,
    });

    return jsonResponse(
      {
        ok: true,
        model: {
          mode: nextRecord.gameMode,
          arch: nextRecord.modelArch,
          rewardProfileId: nextRecord.rewardProfileId,
          queuePolicyId: nextRecord.queuePolicyId,
          versionId: nextRecord.versionId,
          version: nextRecord.version,
          sizeBytes: modelBuffer.byteLength,
          sha256: modelSha256,
          updatedAtMs: nextRecord.updatedAtMs,
          baseGlobalModelId: nextRecord.baseGlobalModelId,
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

const handleListGlobalModels = async (
  request: Request,
  env: Env,
): Promise<Response> => {
  if (request.method !== 'GET') {
    return jsonResponse({ error: 'Method not allowed.' }, 405);
  }
  const selector = readPersonalModelSelectorFromRequest(request);
  if (!selector) {
    return jsonResponse({ error: 'Missing or invalid mode.' }, 400, {
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
    const models = await listGlobalModelsForSelector(env, selector);
    return jsonResponse(
      {
        ok: true,
        selector: {
          mode: selector.gameMode,
          arch: selector.modelArch,
          rewardProfileId: selector.rewardProfileId,
          queuePolicyId: selector.queuePolicyId,
        },
        models: models.map((model) => ({
          id: model.id,
          mode: model.modeId,
          arch: model.modelArch,
          rewardProfileId: model.rewardProfileId,
          queuePolicyId: model.queuePolicyId,
          pipelineId: model.pipelineId,
          label: model.label,
          isDefault: model.isDefault,
          sha256: model.sha256,
          sizeBytes: model.sizeBytes,
          createdAtMs: model.createdAtMs,
          updatedAtMs: model.updatedAtMs,
        })),
      },
      200,
      { 'cache-control': 'no-store' },
    );
  } catch (error) {
    console.error('[models] global list failed', error);
    return jsonResponse(
      { error: 'Global model registry is currently unavailable.' },
      503,
      { 'cache-control': 'no-store' },
    );
  }
};

const handleGetCurrentGlobalModel = async (
  request: Request,
  env: Env,
): Promise<Response> => {
  if (request.method !== 'GET') {
    return jsonResponse({ error: 'Method not allowed.' }, 405);
  }
  const selector = readPersonalModelSelectorFromRequest(request);
  if (!selector) {
    return jsonResponse({ error: 'Missing or invalid mode.' }, 400, {
      'cache-control': 'no-store',
    });
  }
  if (!env.MODELS_BUCKET) {
    return jsonResponse({ error: 'Model storage is not configured.' }, 503, {
      'cache-control': 'no-store',
    });
  }
  const requestedGlobalModelId = asString(
    new URL(request.url).searchParams.get('global_model_id'),
  );
  if (requestedGlobalModelId && !isValidGlobalModelId(requestedGlobalModelId)) {
    return jsonResponse({ error: 'Missing or invalid global model id.' }, 400, {
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
    const globalModel = requestedGlobalModelId
      ? await readGlobalModelById(env, selector, requestedGlobalModelId)
      : ((await listGlobalModelsForSelector(env, selector))[0] ?? null);
    if (!globalModel) {
      return jsonResponse({ error: 'Global model not found.' }, 404, {
        'cache-control': 'no-store',
      });
    }

    const object = await env.MODELS_BUCKET.get(globalModel.r2Key);
    if (!object) {
      return jsonResponse({ error: 'Global model blob is missing.' }, 404, {
        'cache-control': 'no-store',
      });
    }
    const bytes = await object.arrayBuffer();
    if (bytes.byteLength <= 0) {
      return jsonResponse({ error: 'Global model payload is empty.' }, 503, {
        'cache-control': 'no-store',
      });
    }
    const responseHeaders = withBaseHeaders({
      'cache-control': 'no-store',
      'content-type': 'application/octet-stream',
      'x-wub-global-model-id': globalModel.id,
      'x-wub-global-model-mode': globalModel.modeId,
      'x-wub-global-model-arch': globalModel.modelArch,
      'x-wub-global-model-reward-profile': globalModel.rewardProfileId,
      'x-wub-global-model-queue-policy': globalModel.queuePolicyId,
      'x-wub-global-model-pipeline': globalModel.pipelineId,
      'x-wub-global-model-size': String(bytes.byteLength),
      'x-wub-global-model-created-at-ms': String(globalModel.createdAtMs),
      'x-wub-global-model-updated-at-ms': String(globalModel.updatedAtMs),
    });
    if (globalModel.label) {
      responseHeaders.set('x-wub-global-model-label', globalModel.label);
    }
    if (globalModel.sha256) {
      responseHeaders.set('x-wub-global-model-sha256', globalModel.sha256);
    }
    return binaryResponse(bytes, 200, responseHeaders);
  } catch (error) {
    console.error('[models] global current get failed', error);
    return jsonResponse(
      { error: 'Global model registry is currently unavailable.' },
      503,
      { 'cache-control': 'no-store' },
    );
  }
};

const handleResetCurrentPersonalModel = async (
  request: Request,
  env: Env,
): Promise<Response> => {
  if (request.method !== 'POST') {
    return jsonResponse({ error: 'Method not allowed.' }, 405);
  }
  const selector = readPersonalModelSelectorFromRequest(request);
  if (!selector) {
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

  let requestedGlobalModelId: string | null = null;
  try {
    const payload = (await request.json().catch(() => null)) as {
      globalModelId?: unknown;
    } | null;
    requestedGlobalModelId = asString(payload?.globalModelId);
  } catch {
    return jsonResponse({ error: 'Invalid JSON payload.' }, 400, {
      'cache-control': 'no-store',
    });
  }

  try {
    const globalModel = requestedGlobalModelId
      ? await readGlobalModelById(env, selector, requestedGlobalModelId)
      : ((await listGlobalModelsForSelector(env, selector))[0] ?? null);
    if (!globalModel) {
      return jsonResponse({ error: 'Global model not found.' }, 404, {
        'cache-control': 'no-store',
      });
    }

    const sourceObject = await env.MODELS_BUCKET.get(globalModel.r2Key);
    if (!sourceObject) {
      return jsonResponse({ error: 'Global model blob is missing.' }, 404, {
        'cache-control': 'no-store',
      });
    }
    const modelBuffer = await sourceObject.arrayBuffer();
    if (modelBuffer.byteLength <= 0) {
      return jsonResponse({ error: 'Global model payload is empty.' }, 503, {
        'cache-control': 'no-store',
      });
    }
    if (modelBuffer.byteLength > MAX_PERSONALIZED_MODEL_BYTES) {
      return jsonResponse(
        {
          error: `Global model is too large for personalized storage. Max bytes: ${MAX_PERSONALIZED_MODEL_BYTES}.`,
        },
        413,
        { 'cache-control': 'no-store' },
      );
    }

    const existing = await readCurrentPersonalModel(
      env,
      session.userId,
      selector,
    );
    const nextVersion = (existing?.version ?? 0) + 1;
    const nextR2Key =
      `models/${session.userId}` +
      `/${selector.gameMode}` +
      `/${selector.modelArch}` +
      `/${selector.rewardProfileId}` +
      `/${selector.queuePolicyId}` +
      `/v${nextVersion}-${nowMs}.bin`;
    const modelSha256 = await sha256HexFromBuffer(modelBuffer);
    await env.MODELS_BUCKET.put(nextR2Key, modelBuffer, {
      httpMetadata: { contentType: 'application/octet-stream' },
    });
    const nextRecord = await writeCurrentPersonalModelVersion(env, {
      existing,
      userId: session.userId,
      selector,
      r2Key: nextR2Key,
      version: nextVersion,
      modelSizeBytes: modelBuffer.byteLength,
      modelSha256,
      updatedAtMs: nowMs,
      source: 'reset',
      pipelineId: globalModel.pipelineId,
      baseGlobalModelId: globalModel.id,
    });

    return jsonResponse(
      {
        ok: true,
        model: {
          mode: nextRecord.gameMode,
          arch: nextRecord.modelArch,
          rewardProfileId: nextRecord.rewardProfileId,
          queuePolicyId: nextRecord.queuePolicyId,
          versionId: nextRecord.versionId,
          version: nextRecord.version,
          sizeBytes: modelBuffer.byteLength,
          sha256: modelSha256,
          updatedAtMs: nextRecord.updatedAtMs,
          baseGlobalModelId: nextRecord.baseGlobalModelId,
          source: nextRecord.source,
          pipelineId: nextRecord.pipelineId,
        },
        globalModel: {
          id: globalModel.id,
          label: globalModel.label,
          pipelineId: globalModel.pipelineId,
          isDefault: globalModel.isDefault,
        },
      },
      200,
      { 'cache-control': 'no-store' },
    );
  } catch (error) {
    console.error('[models] current model reset failed', error);
    return jsonResponse(
      { error: 'Model storage is currently unavailable.' },
      503,
      { 'cache-control': 'no-store' },
    );
  }
};

const handlePostTrajectoryRecording = async (
  request: Request,
  env: Env,
): Promise<Response> => {
  if (request.method !== 'POST') {
    return jsonResponse({ error: 'Method not allowed.' }, 405);
  }
  if (!env.RECORDINGS_BUCKET) {
    return jsonResponse(
      { error: 'Recording storage is not configured.' },
      503,
      {
        'cache-control': 'no-store',
      },
    );
  }

  const contentLengthRaw = asInt(request.headers.get('content-length'));
  const contentLength =
    contentLengthRaw != null && contentLengthRaw >= 0 ? contentLengthRaw : null;
  if (contentLength != null && contentLength > MAX_TRAJECTORY_UPLOAD_BYTES) {
    return jsonResponse(
      {
        error: `Recording payload is too large. Max bytes: ${MAX_TRAJECTORY_UPLOAD_BYTES}.`,
      },
      413,
      { 'cache-control': 'no-store' },
    );
  }

  const nowMs = Date.now();
  const auth = await requireAuthenticatedSession(request, env, nowMs);
  if (auth.response) return auth.response;
  const session = auth.session!;
  try {
    await touchSessionIfStale(env, session, nowMs);
  } catch (error) {
    console.error('[recordings] touch session failed', error);
  }

  let rawBuffer: ArrayBuffer;
  try {
    rawBuffer = await request.arrayBuffer();
  } catch {
    return jsonResponse({ error: 'Invalid payload.' }, 400, {
      'cache-control': 'no-store',
    });
  }
  if (rawBuffer.byteLength <= 0) {
    return jsonResponse({ error: 'Recording payload is empty.' }, 400, {
      'cache-control': 'no-store',
    });
  }
  if (rawBuffer.byteLength > MAX_TRAJECTORY_UPLOAD_BYTES) {
    return jsonResponse(
      {
        error: `Recording payload is too large. Max bytes: ${MAX_TRAJECTORY_UPLOAD_BYTES}.`,
      },
      413,
      { 'cache-control': 'no-store' },
    );
  }

  let payload: unknown;
  try {
    payload = JSON.parse(new TextDecoder().decode(rawBuffer));
  } catch {
    return jsonResponse({ error: 'Invalid JSON payload.' }, 400, {
      'cache-control': 'no-store',
    });
  }

  const parsed = parseTrajectorySessionV1(payload, {
    minSamples: MIN_TRAJECTORY_SAMPLES_PER_SESSION,
    maxSamples: MAX_TRAJECTORY_SAMPLES_PER_SESSION,
  });
  if (!parsed.ok) {
    return jsonResponse({ error: parsed.error }, 400, {
      'cache-control': 'no-store',
    });
  }
  const recording = parsed.value;
  const recordingId = `${session.userId}_${recording.sessionId}`;
  const monthPrefix = new Date(nowMs).toISOString().slice(0, 7);
  const r2Key = `recordings/${monthPrefix}/${session.userId}/${recording.modeId}/${recording.sessionId}.json`;
  const modelArch = normalizeOptionalModelAxis(
    recording.meta?.modelArchId ?? recording.meta?.modelArch,
  );
  const rewardProfileId = normalizeOptionalModelAxis(
    recording.meta?.rewardProfileId,
  );
  const queuePolicyId = normalizeOptionalModelAxis(
    recording.meta?.queuePolicyId,
  );
  const modelSource = normalizeOptionalModelAxis(recording.meta?.modelSource);
  const rewardPolicyId = normalizeOptionalModelAxis(
    recording.meta?.rewardPolicyId,
  );
  const pipelineId = normalizeOptionalModelAxis(recording.meta?.pipelineId);

  const metaJson = JSON.stringify({
    schema: recording.schema,
    sample_count: recording.samples.length,
    outcome: recording.meta?.outcome ?? null,
    model_source: recording.meta?.modelSource ?? null,
    generator_type: recording.meta?.generatorType ?? null,
    model_mode: recording.meta?.modelMode ?? null,
    model_version: recording.meta?.modelVersion ?? null,
    channel: recording.meta?.channel ?? null,
    reward_policy: recording.meta?.rewardPolicy ?? null,
    reward_policy_id: recording.meta?.rewardPolicyId ?? null,
    reward_profile_id: recording.meta?.rewardProfileId ?? null,
    queue_policy_id: recording.meta?.queuePolicyId ?? null,
    reward_kind: recording.meta?.rewardKind ?? null,
    reward_gamma: recording.meta?.rewardGamma ?? null,
    pipeline_id: recording.meta?.pipelineId ?? null,
    pipeline_mode: recording.meta?.pipelineMode ?? null,
    model_arch_id: recording.meta?.modelArchId ?? null,
    model_arch: recording.meta?.modelArch ?? null,
    actor_type: recording.meta?.actorType ?? null,
    actor_policy_id: recording.meta?.actorPolicyId ?? null,
    training_intent: recording.meta?.trainingIntent ?? null,
    piece_source_profile: recording.meta?.pieceSourceProfile ?? null,
  });

  try {
    await env.RECORDINGS_BUCKET.put(r2Key, rawBuffer, {
      httpMetadata: { contentType: 'application/json' },
    });
    await env.DB.prepare(
      `INSERT INTO recordings_index (
         id,
         user_id,
         game_mode,
         build_version,
         r2_key,
         started_at_ms,
         ended_at_ms,
         duration_ms,
         snapshots_total,
         model_arch,
         reward_profile_id,
         queue_policy_id,
         model_source,
         reward_policy_id,
         pipeline_id,
         meta_json,
         created_at_ms
       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET
         game_mode = excluded.game_mode,
         build_version = excluded.build_version,
         r2_key = excluded.r2_key,
         started_at_ms = excluded.started_at_ms,
         ended_at_ms = excluded.ended_at_ms,
         duration_ms = excluded.duration_ms,
         snapshots_total = excluded.snapshots_total,
         model_arch = excluded.model_arch,
         reward_profile_id = excluded.reward_profile_id,
         queue_policy_id = excluded.queue_policy_id,
         model_source = excluded.model_source,
         reward_policy_id = excluded.reward_policy_id,
         pipeline_id = excluded.pipeline_id,
         meta_json = excluded.meta_json`,
    )
      .bind(
        recordingId,
        session.userId,
        recording.modeId,
        recording.buildVersion,
        r2Key,
        recording.startedAtMs,
        recording.endedAtMs,
        recording.durationMs,
        recording.samples.length,
        modelArch,
        rewardProfileId,
        queuePolicyId,
        modelSource,
        rewardPolicyId,
        pipelineId,
        metaJson,
        nowMs,
      )
      .run();

    return jsonResponse(
      {
        ok: true,
        recording: {
          id: recordingId,
          mode: recording.modeId,
          r2Key,
          samples: recording.samples.length,
          createdAtMs: nowMs,
        },
      },
      200,
      { 'cache-control': 'no-store' },
    );
  } catch (error) {
    console.error('[recordings] trajectory upload failed', error);
    const body: Record<string, unknown> = {
      error: 'Recording storage is currently unavailable.',
    };
    if (isDevReleaseChannel(env)) {
      body.detail = errorDetailMessage(error);
    }
    return jsonResponse(body, 503, { 'cache-control': 'no-store' });
  }
};

const handleAdminWhoAmI = async (
  request: Request,
  env: Env,
): Promise<Response> => {
  if (request.method !== 'GET') {
    return jsonResponse({ error: 'Method not allowed.' }, 405, {
      'cache-control': 'no-store',
    });
  }

  const nowMs = Date.now();
  const auth = await requireAdminSession(request, env, nowMs);
  if (auth.response) return auth.response;
  const session = auth.session!;
  try {
    await touchSessionIfStale(env, session, nowMs);
  } catch (error) {
    console.error('[admin] touch session failed', error);
  }

  return jsonResponse(
    {
      ok: true,
      admin: {
        userId: session.userId,
        username: session.username,
        email: session.emailNorm,
      },
    },
    200,
    { 'cache-control': 'no-store' },
  );
};

type AdminRecordingSummary = {
  id: string;
  userId: string;
  mode: string;
  buildVersion: string;
  r2Key: string;
  startedAtMs: number;
  endedAtMs: number;
  durationMs: number;
  samples: number;
  createdAtMs: number;
  meta: Record<string, unknown> | null;
};

const toAdminRecordingSummary = (
  row: Record<string, unknown>,
): AdminRecordingSummary | null => {
  const id = asString(row.id);
  const userId = asString(row.user_id);
  const mode = normalizeGameMode(row.game_mode);
  const buildVersion = asString(row.build_version);
  const r2Key = asString(row.r2_key);
  const startedAtMs = asInt(row.started_at_ms);
  const endedAtMs = asInt(row.ended_at_ms);
  const durationMs = asInt(row.duration_ms);
  const samples = asInt(row.snapshots_total);
  const createdAtMs = asInt(row.created_at_ms);
  if (
    !id ||
    !userId ||
    !mode ||
    !buildVersion ||
    !r2Key ||
    startedAtMs == null ||
    endedAtMs == null ||
    durationMs == null ||
    samples == null ||
    createdAtMs == null
  ) {
    return null;
  }
  return {
    id,
    userId,
    mode,
    buildVersion,
    r2Key,
    startedAtMs,
    endedAtMs,
    durationMs,
    samples,
    createdAtMs,
    meta: parseJsonObjectString(asString(row.meta_json)),
  };
};

const handleAdminRecordingsIndex = async (
  request: Request,
  env: Env,
): Promise<Response> => {
  if (request.method !== 'GET') {
    return jsonResponse({ error: 'Method not allowed.' }, 405, {
      'cache-control': 'no-store',
    });
  }

  const nowMs = Date.now();
  const auth = await requireAdminSession(request, env, nowMs);
  if (auth.response) return auth.response;
  const session = auth.session!;
  try {
    await touchSessionIfStale(env, session, nowMs);
  } catch (error) {
    console.error('[admin] touch session failed', error);
  }

  const url = new URL(request.url);
  const requestedMode = url.searchParams.get('mode');
  const modeFilter = normalizeGameMode(requestedMode);
  if (requestedMode != null && !modeFilter) {
    return jsonResponse({ error: 'Invalid mode filter.' }, 400, {
      'cache-control': 'no-store',
    });
  }

  const requestedBuild = url.searchParams.get('build');
  const buildFilter = asString(requestedBuild);
  if (requestedBuild != null && !buildFilter) {
    return jsonResponse({ error: 'Invalid build filter.' }, 400, {
      'cache-control': 'no-store',
    });
  }

  const requestedUserId = asString(url.searchParams.get('user_id'));
  if (requestedUserId != null && !isValidRecordingId(requestedUserId)) {
    return jsonResponse({ error: 'Invalid user filter.' }, 400, {
      'cache-control': 'no-store',
    });
  }

  const startedFromMs = asInt(url.searchParams.get('started_from_ms'));
  const startedToMs = asInt(url.searchParams.get('started_to_ms'));
  if (
    (startedFromMs != null && startedFromMs < 0) ||
    (startedToMs != null && startedToMs < 0)
  ) {
    return jsonResponse({ error: 'Invalid time range filter.' }, 400, {
      'cache-control': 'no-store',
    });
  }
  if (
    startedFromMs != null &&
    startedToMs != null &&
    startedFromMs > startedToMs
  ) {
    return jsonResponse({ error: 'Invalid time range filter.' }, 400, {
      'cache-control': 'no-store',
    });
  }

  const requestedCursor = url.searchParams.get('cursor');
  const cursor = parseRecordingsCursor(requestedCursor);
  if (requestedCursor != null && !cursor) {
    return jsonResponse({ error: 'Invalid cursor.' }, 400, {
      'cache-control': 'no-store',
    });
  }

  const limit = clampInt(asInt(url.searchParams.get('limit')), {
    min: 1,
    max: 200,
    fallback: 50,
  });

  const whereParts: string[] = [];
  const values: unknown[] = [];
  if (modeFilter) {
    whereParts.push('game_mode = ?');
    values.push(modeFilter);
  }
  if (buildFilter) {
    whereParts.push('build_version = ?');
    values.push(buildFilter);
  }
  if (requestedUserId) {
    whereParts.push('user_id = ?');
    values.push(requestedUserId);
  }
  if (startedFromMs != null) {
    whereParts.push('started_at_ms >= ?');
    values.push(startedFromMs);
  }
  if (startedToMs != null) {
    whereParts.push('started_at_ms <= ?');
    values.push(startedToMs);
  }
  if (cursor) {
    whereParts.push('(started_at_ms < ? OR (started_at_ms = ? AND id < ?))');
    values.push(cursor.startedAtMs, cursor.startedAtMs, cursor.id);
  }

  const whereClause =
    whereParts.length > 0 ? `WHERE ${whereParts.join(' AND ')}` : '';

  try {
    const result = await env.DB.prepare(
      `SELECT
         id,
         user_id,
         game_mode,
         build_version,
         r2_key,
         started_at_ms,
         ended_at_ms,
         duration_ms,
         snapshots_total,
         meta_json,
         created_at_ms
       FROM recordings_index
       ${whereClause}
       ORDER BY started_at_ms DESC, id DESC
       LIMIT ?`,
    )
      .bind(...values, limit)
      .all<Record<string, unknown>>();
    const rows = Array.isArray(result.results) ? result.results : [];
    const recordings: AdminRecordingSummary[] = [];
    for (const row of rows) {
      const summary = toAdminRecordingSummary(row);
      if (summary) recordings.push(summary);
    }
    const nextCursor =
      rows.length >= limit && recordings.length > 0
        ? formatRecordingsCursor({
            startedAtMs: recordings[recordings.length - 1].startedAtMs,
            id: recordings[recordings.length - 1].id,
          })
        : null;
    return jsonResponse(
      {
        ok: true,
        recordings,
        page: {
          limit,
          nextCursor,
          returned: recordings.length,
        },
      },
      200,
      { 'cache-control': 'no-store' },
    );
  } catch (error) {
    console.error('[admin] recordings index failed', error);
    return jsonResponse(
      { error: 'Recording index is currently unavailable.' },
      503,
      { 'cache-control': 'no-store' },
    );
  }
};

const handleAdminRecordingsExportManifest = async (
  request: Request,
  env: Env,
): Promise<Response> => {
  if (request.method !== 'GET') {
    return jsonResponse({ error: 'Method not allowed.' }, 405, {
      'cache-control': 'no-store',
    });
  }

  const nowMs = Date.now();
  const auth = await requireAdminSession(request, env, nowMs);
  if (auth.response) return auth.response;
  const session = auth.session!;
  try {
    await touchSessionIfStale(env, session, nowMs);
  } catch (error) {
    console.error('[admin] touch session failed', error);
  }

  const url = new URL(request.url);
  const requestedMode = url.searchParams.get('mode');
  const modeFilter = normalizeGameMode(requestedMode);
  if (requestedMode != null && !modeFilter) {
    return jsonResponse({ error: 'Invalid mode filter.' }, 400, {
      'cache-control': 'no-store',
    });
  }

  const requestedBuild = url.searchParams.get('build');
  const buildFilter = asString(requestedBuild);
  if (requestedBuild != null && !buildFilter) {
    return jsonResponse({ error: 'Invalid build filter.' }, 400, {
      'cache-control': 'no-store',
    });
  }

  const requestedUserId = asString(url.searchParams.get('user_id'));
  if (requestedUserId != null && !isValidRecordingId(requestedUserId)) {
    return jsonResponse({ error: 'Invalid user filter.' }, 400, {
      'cache-control': 'no-store',
    });
  }

  const startedFromMs = asInt(url.searchParams.get('started_from_ms'));
  const startedToMs = asInt(url.searchParams.get('started_to_ms'));
  if (
    (startedFromMs != null && startedFromMs < 0) ||
    (startedToMs != null && startedToMs < 0)
  ) {
    return jsonResponse({ error: 'Invalid time range filter.' }, 400, {
      'cache-control': 'no-store',
    });
  }
  if (
    startedFromMs != null &&
    startedToMs != null &&
    startedFromMs > startedToMs
  ) {
    return jsonResponse({ error: 'Invalid time range filter.' }, 400, {
      'cache-control': 'no-store',
    });
  }

  const modelArch = normalizeOptionalModelAxis(url.searchParams.get('arch'));
  const rewardProfileId = normalizeOptionalModelAxis(
    url.searchParams.get('reward_profile'),
  );
  const queuePolicyId = normalizeOptionalModelAxis(
    url.searchParams.get('queue_policy'),
  );
  const pipelineId = normalizeOptionalModelAxis(
    url.searchParams.get('pipeline_id'),
  );
  const actorTypeRaw = asString(url.searchParams.get('actor_type'));
  const actorType =
    actorTypeRaw === 'human' || actorTypeRaw === 'bot' ? actorTypeRaw : null;
  if (actorTypeRaw != null && actorTypeRaw !== actorType) {
    return jsonResponse({ error: 'Invalid actor_type filter.' }, 400, {
      'cache-control': 'no-store',
    });
  }
  const pieceSourceProfile = normalizeOptionalModelAxis(
    url.searchParams.get('piece_source_profile'),
  );

  const minSamplesRaw = asInt(url.searchParams.get('min_samples'));
  const minSamples =
    minSamplesRaw == null
      ? null
      : clampInt(minSamplesRaw, {
          min: 1,
          max: MAX_TRAJECTORY_SAMPLES_PER_SESSION,
          fallback: 1,
        });
  if (minSamplesRaw != null && minSamplesRaw < 1) {
    return jsonResponse({ error: 'Invalid min_samples filter.' }, 400, {
      'cache-control': 'no-store',
    });
  }

  const requestedCursor = url.searchParams.get('cursor');
  const cursor = parseRecordingsCursor(requestedCursor);
  if (requestedCursor != null && !cursor) {
    return jsonResponse({ error: 'Invalid cursor.' }, 400, {
      'cache-control': 'no-store',
    });
  }

  const limit = clampInt(asInt(url.searchParams.get('limit')), {
    min: 1,
    max: 500,
    fallback: 100,
  });

  const whereParts: string[] = [];
  const values: unknown[] = [];
  if (modeFilter) {
    whereParts.push('game_mode = ?');
    values.push(modeFilter);
  }
  if (buildFilter) {
    whereParts.push('build_version = ?');
    values.push(buildFilter);
  }
  if (requestedUserId) {
    whereParts.push('user_id = ?');
    values.push(requestedUserId);
  }
  if (startedFromMs != null) {
    whereParts.push('started_at_ms >= ?');
    values.push(startedFromMs);
  }
  if (startedToMs != null) {
    whereParts.push('started_at_ms <= ?');
    values.push(startedToMs);
  }
  if (modelArch) {
    whereParts.push('model_arch = ?');
    values.push(modelArch);
  }
  if (rewardProfileId) {
    whereParts.push('reward_profile_id = ?');
    values.push(rewardProfileId);
  }
  if (queuePolicyId) {
    whereParts.push('queue_policy_id = ?');
    values.push(queuePolicyId);
  }
  if (pipelineId) {
    whereParts.push('pipeline_id = ?');
    values.push(pipelineId);
  }
  if (minSamples != null) {
    whereParts.push('snapshots_total >= ?');
    values.push(minSamples);
  }
  if (actorType) {
    whereParts.push(`json_extract(meta_json, '$.actor_type') = ?`);
    values.push(actorType);
  }
  if (pieceSourceProfile) {
    whereParts.push(`json_extract(meta_json, '$.piece_source_profile') = ?`);
    values.push(pieceSourceProfile);
  }
  if (cursor) {
    whereParts.push('(started_at_ms < ? OR (started_at_ms = ? AND id < ?))');
    values.push(cursor.startedAtMs, cursor.startedAtMs, cursor.id);
  }
  const whereClause =
    whereParts.length > 0 ? `WHERE ${whereParts.join(' AND ')}` : '';

  try {
    const result = await env.DB.prepare(
      `SELECT
         id,
         user_id,
         game_mode,
         build_version,
         r2_key,
         started_at_ms,
         ended_at_ms,
         duration_ms,
         snapshots_total,
         meta_json,
         created_at_ms
       FROM recordings_index
       ${whereClause}
       ORDER BY started_at_ms DESC, id DESC
       LIMIT ?`,
    )
      .bind(...values, limit)
      .all<Record<string, unknown>>();
    const rows = Array.isArray(result.results) ? result.results : [];
    const recordings: AdminRecordingSummary[] = [];
    for (const row of rows) {
      const summary = toAdminRecordingSummary(row);
      if (summary) recordings.push(summary);
    }
    const nextCursor =
      rows.length >= limit && recordings.length > 0
        ? formatRecordingsCursor({
            startedAtMs: recordings[recordings.length - 1].startedAtMs,
            id: recordings[recordings.length - 1].id,
          })
        : null;
    return jsonResponse(
      {
        ok: true,
        selector: {
          mode: modeFilter,
          build: buildFilter,
          userId: requestedUserId,
          arch: modelArch,
          rewardProfileId,
          queuePolicyId,
          pipelineId,
          actorType,
          pieceSourceProfile,
          minSamples,
          startedFromMs,
          startedToMs,
        },
        recordings,
        page: {
          limit,
          nextCursor,
          returned: recordings.length,
        },
      },
      200,
      { 'cache-control': 'no-store' },
    );
  } catch (error) {
    console.error('[admin] recordings export manifest failed', error);
    return jsonResponse(
      { error: 'Recording export manifest is currently unavailable.' },
      503,
      { 'cache-control': 'no-store' },
    );
  }
};

const handleAdminRecordingObject = async (
  request: Request,
  env: Env,
): Promise<Response> => {
  if (request.method !== 'GET') {
    return jsonResponse({ error: 'Method not allowed.' }, 405, {
      'cache-control': 'no-store',
    });
  }
  if (!env.RECORDINGS_BUCKET) {
    return jsonResponse(
      { error: 'Recording storage is not configured.' },
      503,
      {
        'cache-control': 'no-store',
      },
    );
  }

  const nowMs = Date.now();
  const auth = await requireAdminSession(request, env, nowMs);
  if (auth.response) return auth.response;
  const session = auth.session!;
  try {
    await touchSessionIfStale(env, session, nowMs);
  } catch (error) {
    console.error('[admin] touch session failed', error);
  }

  const url = new URL(request.url);
  const recordingId = asString(url.searchParams.get('id'));
  if (!recordingId || !isValidRecordingId(recordingId)) {
    return jsonResponse({ error: 'Missing or invalid recording id.' }, 400, {
      'cache-control': 'no-store',
    });
  }

  try {
    const row = await env.DB.prepare(
      `SELECT id, user_id, game_mode, build_version, r2_key
       FROM recordings_index
       WHERE id = ?
       LIMIT 1`,
    )
      .bind(recordingId)
      .first<Record<string, unknown>>();
    if (!row) {
      return jsonResponse({ error: 'Recording not found.' }, 404, {
        'cache-control': 'no-store',
      });
    }
    const r2Key = asString(row.r2_key);
    if (!r2Key) {
      return jsonResponse({ error: 'Recording metadata is invalid.' }, 500, {
        'cache-control': 'no-store',
      });
    }
    const object = await env.RECORDINGS_BUCKET.get(r2Key);
    if (!object) {
      return jsonResponse({ error: 'Recording object not found.' }, 404, {
        'cache-control': 'no-store',
      });
    }
    const body = await object.arrayBuffer();
    return binaryResponse(body, 200, {
      'cache-control': 'no-store',
      'content-type': 'application/json; charset=utf-8',
      'x-wub-recording-id': recordingId,
      'x-wub-recording-user-id': asString(row.user_id) ?? '',
      'x-wub-recording-mode': asString(row.game_mode) ?? '',
      'x-wub-recording-build': asString(row.build_version) ?? '',
      'x-wub-recording-r2-key': r2Key,
    });
  } catch (error) {
    console.error('[admin] recording object fetch failed', error);
    return jsonResponse(
      { error: 'Recording storage is currently unavailable.' },
      503,
      { 'cache-control': 'no-store' },
    );
  }
};

const handleAdminPublishGlobalModel = async (
  request: Request,
  env: Env,
): Promise<Response> => {
  if (request.method !== 'POST') {
    return jsonResponse({ error: 'Method not allowed.' }, 405, {
      'cache-control': 'no-store',
    });
  }
  if (!env.MODELS_BUCKET) {
    return jsonResponse({ error: 'Model storage is not configured.' }, 503, {
      'cache-control': 'no-store',
    });
  }

  const selector = readPersonalModelSelectorFromRequest(request);
  if (!selector) {
    return jsonResponse({ error: 'Missing or invalid mode.' }, 400, {
      'cache-control': 'no-store',
    });
  }

  const nowMs = Date.now();
  const auth = await requireAdminSession(request, env, nowMs);
  if (auth.response) return auth.response;
  const session = auth.session!;
  try {
    await touchSessionIfStale(env, session, nowMs);
  } catch (error) {
    console.error('[admin] touch session failed', error);
  }

  const url = new URL(request.url);
  const label = asString(url.searchParams.get('label'))?.slice(0, 120) ?? null;
  const pipelineId =
    normalizeOptionalModelAxis(url.searchParams.get('pipeline_id')) ??
    'manual_publish';
  const setDefault = parseBooleanFlag(url.searchParams.get('set_default'));

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

  const globalModelId = `global_${crypto.randomUUID()}`;
  const r2Key =
    `global/${selector.gameMode}` +
    `/${selector.modelArch}` +
    `/${selector.rewardProfileId}` +
    `/${selector.queuePolicyId}` +
    `/${nowMs}-${globalModelId}.json`;

  let stored = false;
  try {
    const modelSha256 = await sha256HexFromBuffer(modelBuffer);
    await env.MODELS_BUCKET.put(r2Key, modelBuffer, {
      httpMetadata: { contentType: 'application/json' },
    });
    stored = true;

    if (setDefault) {
      await env.DB.prepare(
        `UPDATE global_models
         SET is_default = 0, updated_at_ms = ?
         WHERE
           mode_id = ?
           AND model_arch = ?
           AND reward_profile_id = ?
           AND queue_policy_id = ?
           AND retired_at_ms IS NULL`,
      )
        .bind(
          nowMs,
          selector.gameMode,
          selector.modelArch,
          selector.rewardProfileId,
          selector.queuePolicyId,
        )
        .run();
    }

    await env.DB.prepare(
      `INSERT INTO global_models (
         id,
         mode_id,
         model_arch,
         reward_profile_id,
         queue_policy_id,
         pipeline_id,
         label,
         r2_key,
         sha256,
         size_bytes,
         metrics_json,
         is_default,
         created_at_ms,
         updated_at_ms,
         retired_at_ms
       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    )
      .bind(
        globalModelId,
        selector.gameMode,
        selector.modelArch,
        selector.rewardProfileId,
        selector.queuePolicyId,
        pipelineId,
        label,
        r2Key,
        modelSha256,
        modelBuffer.byteLength,
        null,
        setDefault ? 1 : 0,
        nowMs,
        nowMs,
        null,
      )
      .run();

    return jsonResponse(
      {
        ok: true,
        model: {
          id: globalModelId,
          mode: selector.gameMode,
          arch: selector.modelArch,
          rewardProfileId: selector.rewardProfileId,
          queuePolicyId: selector.queuePolicyId,
          pipelineId,
          label,
          r2Key,
          sha256: modelSha256,
          sizeBytes: modelBuffer.byteLength,
          isDefault: setDefault,
          createdAtMs: nowMs,
          updatedAtMs: nowMs,
        },
      },
      200,
      { 'cache-control': 'no-store' },
    );
  } catch (error) {
    if (stored) {
      await env.MODELS_BUCKET.delete(r2Key).catch((cleanupError) => {
        console.warn(
          '[admin] global model publish cleanup failed',
          cleanupError,
        );
      });
    }
    console.error('[admin] global model publish failed', error);
    return jsonResponse(
      { error: 'Global model publishing is currently unavailable.' },
      503,
      { 'cache-control': 'no-store' },
    );
  }
};

const handleAdminGlobalModelsIndex = async (
  request: Request,
  env: Env,
): Promise<Response> => {
  if (request.method !== 'GET') {
    return jsonResponse({ error: 'Method not allowed.' }, 405, {
      'cache-control': 'no-store',
    });
  }

  const selector = readPersonalModelSelectorFromRequest(request);
  if (!selector) {
    return jsonResponse({ error: 'Missing or invalid mode.' }, 400, {
      'cache-control': 'no-store',
    });
  }

  const nowMs = Date.now();
  const auth = await requireAdminSession(request, env, nowMs);
  if (auth.response) return auth.response;
  const session = auth.session!;
  try {
    await touchSessionIfStale(env, session, nowMs);
  } catch (error) {
    console.error('[admin] touch session failed', error);
  }

  const includeRetired = parseBooleanFlag(
    new URL(request.url).searchParams.get('include_retired'),
  );
  try {
    const rows = includeRetired
      ? await env.DB.prepare(
          `SELECT
             id,
             mode_id,
             model_arch,
             reward_profile_id,
             queue_policy_id,
             pipeline_id,
             label,
             r2_key,
             sha256,
             size_bytes,
             is_default,
             created_at_ms,
             updated_at_ms,
             retired_at_ms
           FROM global_models
           WHERE
             mode_id = ?
             AND model_arch = ?
             AND reward_profile_id = ?
             AND queue_policy_id = ?
           ORDER BY retired_at_ms IS NULL DESC, is_default DESC, created_at_ms DESC
           LIMIT 300`,
        )
          .bind(
            selector.gameMode,
            selector.modelArch,
            selector.rewardProfileId,
            selector.queuePolicyId,
          )
          .all<Record<string, unknown>>()
      : await env.DB.prepare(
          `SELECT
             id,
             mode_id,
             model_arch,
             reward_profile_id,
             queue_policy_id,
             pipeline_id,
             label,
             r2_key,
             sha256,
             size_bytes,
             is_default,
             created_at_ms,
             updated_at_ms,
             retired_at_ms
           FROM global_models
           WHERE
             mode_id = ?
             AND model_arch = ?
             AND reward_profile_id = ?
             AND queue_policy_id = ?
             AND retired_at_ms IS NULL
           ORDER BY is_default DESC, created_at_ms DESC
           LIMIT 300`,
        )
          .bind(
            selector.gameMode,
            selector.modelArch,
            selector.rewardProfileId,
            selector.queuePolicyId,
          )
          .all<Record<string, unknown>>();
    const results = Array.isArray(rows.results) ? rows.results : [];
    const models: GlobalModelRecord[] = [];
    for (const row of results) {
      const model = readGlobalModelRecord(row);
      if (model) models.push(model);
    }
    return jsonResponse(
      {
        ok: true,
        selector: {
          mode: selector.gameMode,
          arch: selector.modelArch,
          rewardProfileId: selector.rewardProfileId,
          queuePolicyId: selector.queuePolicyId,
        },
        includeRetired,
        models: models.map((model) => ({
          id: model.id,
          mode: model.modeId,
          arch: model.modelArch,
          rewardProfileId: model.rewardProfileId,
          queuePolicyId: model.queuePolicyId,
          pipelineId: model.pipelineId,
          label: model.label,
          r2Key: model.r2Key,
          sha256: model.sha256,
          sizeBytes: model.sizeBytes,
          isDefault: model.isDefault,
          createdAtMs: model.createdAtMs,
          updatedAtMs: model.updatedAtMs,
          retiredAtMs: model.retiredAtMs,
        })),
      },
      200,
      { 'cache-control': 'no-store' },
    );
  } catch (error) {
    console.error('[admin] global models index failed', error);
    return jsonResponse(
      { error: 'Global model registry is currently unavailable.' },
      503,
      { 'cache-control': 'no-store' },
    );
  }
};

const handleAdminSetGlobalModelDefault = async (
  request: Request,
  env: Env,
): Promise<Response> => {
  if (request.method !== 'POST') {
    return jsonResponse({ error: 'Method not allowed.' }, 405, {
      'cache-control': 'no-store',
    });
  }

  const nowMs = Date.now();
  const auth = await requireAdminSession(request, env, nowMs);
  if (auth.response) return auth.response;
  const session = auth.session!;
  try {
    await touchSessionIfStale(env, session, nowMs);
  } catch (error) {
    console.error('[admin] touch session failed', error);
  }

  const payload = (await request.json().catch(() => null)) as {
    id?: unknown;
  } | null;
  const modelId = asString(payload?.id);
  if (!modelId || !isValidGlobalModelId(modelId)) {
    return jsonResponse({ error: 'Missing or invalid model id.' }, 400, {
      'cache-control': 'no-store',
    });
  }

  try {
    const model = await readGlobalModelByIdAnyAxes(env, modelId);
    if (!model || model.retiredAtMs != null) {
      return jsonResponse({ error: 'Global model not found.' }, 404, {
        'cache-control': 'no-store',
      });
    }
    await env.DB.prepare(
      `UPDATE global_models
       SET is_default = 0, updated_at_ms = ?
       WHERE
         mode_id = ?
         AND model_arch = ?
         AND reward_profile_id = ?
         AND queue_policy_id = ?
         AND retired_at_ms IS NULL`,
    )
      .bind(
        nowMs,
        model.modeId,
        model.modelArch,
        model.rewardProfileId,
        model.queuePolicyId,
      )
      .run();
    await env.DB.prepare(
      `UPDATE global_models
       SET is_default = 1, updated_at_ms = ?
       WHERE id = ?`,
    )
      .bind(nowMs, model.id)
      .run();
    return jsonResponse(
      {
        ok: true,
        model: {
          id: model.id,
          mode: model.modeId,
          arch: model.modelArch,
          rewardProfileId: model.rewardProfileId,
          queuePolicyId: model.queuePolicyId,
        },
      },
      200,
      { 'cache-control': 'no-store' },
    );
  } catch (error) {
    console.error('[admin] global model set default failed', error);
    return jsonResponse(
      { error: 'Global model registry is currently unavailable.' },
      503,
      { 'cache-control': 'no-store' },
    );
  }
};

const handleAdminRetireGlobalModel = async (
  request: Request,
  env: Env,
): Promise<Response> => {
  if (request.method !== 'POST') {
    return jsonResponse({ error: 'Method not allowed.' }, 405, {
      'cache-control': 'no-store',
    });
  }

  const nowMs = Date.now();
  const auth = await requireAdminSession(request, env, nowMs);
  if (auth.response) return auth.response;
  const session = auth.session!;
  try {
    await touchSessionIfStale(env, session, nowMs);
  } catch (error) {
    console.error('[admin] touch session failed', error);
  }

  const payload = (await request.json().catch(() => null)) as {
    id?: unknown;
  } | null;
  const modelId = asString(payload?.id);
  if (!modelId || !isValidGlobalModelId(modelId)) {
    return jsonResponse({ error: 'Missing or invalid model id.' }, 400, {
      'cache-control': 'no-store',
    });
  }

  try {
    const model = await readGlobalModelByIdAnyAxes(env, modelId);
    if (!model || model.retiredAtMs != null) {
      return jsonResponse({ error: 'Global model not found.' }, 404, {
        'cache-control': 'no-store',
      });
    }
    await env.DB.prepare(
      `UPDATE global_models
       SET retired_at_ms = ?, is_default = 0, updated_at_ms = ?
       WHERE id = ?`,
    )
      .bind(nowMs, nowMs, model.id)
      .run();

    let replacementId: string | null = null;
    if (model.isDefault) {
      const replacement = await env.DB.prepare(
        `SELECT id
         FROM global_models
         WHERE
           mode_id = ?
           AND model_arch = ?
           AND reward_profile_id = ?
           AND queue_policy_id = ?
           AND retired_at_ms IS NULL
           AND id != ?
         ORDER BY created_at_ms DESC
         LIMIT 1`,
      )
        .bind(
          model.modeId,
          model.modelArch,
          model.rewardProfileId,
          model.queuePolicyId,
          model.id,
        )
        .first<Record<string, unknown>>();
      replacementId = asString(replacement?.id);
      if (replacementId) {
        await env.DB.prepare(
          `UPDATE global_models
           SET is_default = 1, updated_at_ms = ?
           WHERE id = ?`,
        )
          .bind(nowMs, replacementId)
          .run();
      }
    }

    return jsonResponse(
      {
        ok: true,
        retiredModelId: model.id,
        replacementDefaultId: replacementId,
      },
      200,
      { 'cache-control': 'no-store' },
    );
  } catch (error) {
    console.error('[admin] global model retire failed', error);
    return jsonResponse(
      { error: 'Global model registry is currently unavailable.' },
      503,
      { 'cache-control': 'no-store' },
    );
  }
};

const toBotPolicyResponse = (
  policy: BotPolicyRecord,
): Record<string, unknown> => ({
  id: policy.id,
  modeId: policy.modeId,
  archId: policy.archId,
  queuePolicyId: policy.queuePolicyId,
  pipelineId: policy.pipelineId,
  pieceSourceProfile: policy.pieceSourceProfile,
  r2Key: policy.r2Key,
  version: policy.version,
  isPinned: policy.isPinned,
  metrics: policy.metrics,
  createdAtMs: policy.createdAtMs,
});

const handleAdminBotPoliciesCurrent = async (
  request: Request,
  env: Env,
): Promise<Response> => {
  if (request.method !== 'GET') {
    return jsonResponse({ error: 'Method not allowed.' }, 405, {
      'cache-control': 'no-store',
    });
  }
  const selector = readBotPolicySelectorFromRequest(request);
  if (!selector) {
    return jsonResponse({ error: 'Missing or invalid mode.' }, 400, {
      'cache-control': 'no-store',
    });
  }

  const nowMs = Date.now();
  const auth = await requireAdminSession(request, env, nowMs);
  if (auth.response) return auth.response;
  const session = auth.session!;
  try {
    await touchSessionIfStale(env, session, nowMs);
  } catch (error) {
    console.error('[admin] touch session failed', error);
  }

  try {
    const current = await readCurrentBotPolicyForSelector(env, selector);
    return jsonResponse(
      {
        ok: true,
        selector,
        current: current ? toBotPolicyResponse(current) : null,
      },
      200,
      { 'cache-control': 'no-store' },
    );
  } catch (error) {
    console.error('[admin] bot policy current failed', error);
    return jsonResponse(
      { error: 'Bot policy registry is currently unavailable.' },
      503,
      { 'cache-control': 'no-store' },
    );
  }
};

const handleAdminBotPoliciesList = async (
  request: Request,
  env: Env,
): Promise<Response> => {
  if (request.method !== 'GET') {
    return jsonResponse({ error: 'Method not allowed.' }, 405, {
      'cache-control': 'no-store',
    });
  }
  const selector = readBotPolicySelectorFromRequest(request);
  if (!selector) {
    return jsonResponse({ error: 'Missing or invalid mode.' }, 400, {
      'cache-control': 'no-store',
    });
  }

  const url = new URL(request.url);
  const limit = clampInt(asInt(url.searchParams.get('limit')), {
    min: 1,
    max: 200,
    fallback: 40,
  });
  const requestedCursor = url.searchParams.get('cursor');
  const cursor = parseBotPolicyCursor(requestedCursor);
  if (requestedCursor != null && !cursor) {
    return jsonResponse({ error: 'Invalid cursor.' }, 400, {
      'cache-control': 'no-store',
    });
  }

  const nowMs = Date.now();
  const auth = await requireAdminSession(request, env, nowMs);
  if (auth.response) return auth.response;
  const session = auth.session!;
  try {
    await touchSessionIfStale(env, session, nowMs);
  } catch (error) {
    console.error('[admin] touch session failed', error);
  }

  try {
    const records = await listBotPoliciesForSelector(env, selector, {
      limit,
      cursor,
    });
    const nextCursor =
      records.length >= limit
        ? formatBotPolicyCursor({
            createdAtMs: records[records.length - 1].createdAtMs,
            id: records[records.length - 1].id,
          })
        : null;
    return jsonResponse(
      {
        ok: true,
        selector,
        policies: records.map(toBotPolicyResponse),
        page: {
          limit,
          returned: records.length,
          nextCursor,
        },
      },
      200,
      { 'cache-control': 'no-store' },
    );
  } catch (error) {
    console.error('[admin] bot policy list failed', error);
    return jsonResponse(
      { error: 'Bot policy registry is currently unavailable.' },
      503,
      { 'cache-control': 'no-store' },
    );
  }
};

const handleAdminBotPolicyObject = async (
  request: Request,
  env: Env,
): Promise<Response> => {
  if (request.method !== 'GET') {
    return jsonResponse({ error: 'Method not allowed.' }, 405, {
      'cache-control': 'no-store',
    });
  }
  if (!env.MODELS_BUCKET) {
    return jsonResponse({ error: 'Model storage is not configured.' }, 503, {
      'cache-control': 'no-store',
    });
  }

  const url = new URL(request.url);
  const policyId = asString(url.searchParams.get('id'));
  if (!policyId || !isValidBotPolicyId(policyId)) {
    return jsonResponse({ error: 'Missing or invalid policy id.' }, 400, {
      'cache-control': 'no-store',
    });
  }

  const nowMs = Date.now();
  const auth = await requireAdminSession(request, env, nowMs);
  if (auth.response) return auth.response;
  const session = auth.session!;
  try {
    await touchSessionIfStale(env, session, nowMs);
  } catch (error) {
    console.error('[admin] touch session failed', error);
  }

  try {
    const policy = await readBotPolicyById(env, policyId);
    if (!policy) {
      return jsonResponse({ error: 'Bot policy not found.' }, 404, {
        'cache-control': 'no-store',
      });
    }
    const object = await env.MODELS_BUCKET.get(policy.r2Key);
    if (!object) {
      return jsonResponse({ error: 'Bot policy object not found.' }, 404, {
        'cache-control': 'no-store',
      });
    }
    const bytes = await object.arrayBuffer();
    return binaryResponse(bytes, 200, {
      'cache-control': 'no-store',
      'content-type': 'application/json; charset=utf-8',
      'x-wub-bot-policy-id': policy.id,
      'x-wub-bot-mode': policy.modeId,
      'x-wub-bot-arch': policy.archId,
      'x-wub-bot-queue-policy': policy.queuePolicyId,
      'x-wub-bot-version': String(policy.version),
      'x-wub-bot-r2-key': policy.r2Key,
      'x-wub-bot-created-at-ms': String(policy.createdAtMs),
    });
  } catch (error) {
    console.error('[admin] bot policy object failed', error);
    return jsonResponse(
      { error: 'Bot policy storage is currently unavailable.' },
      503,
      { 'cache-control': 'no-store' },
    );
  }
};

const handleAdminBotPolicyPublish = async (
  request: Request,
  env: Env,
): Promise<Response> => {
  if (request.method !== 'POST') {
    return jsonResponse({ error: 'Method not allowed.' }, 405, {
      'cache-control': 'no-store',
    });
  }
  if (!env.MODELS_BUCKET) {
    return jsonResponse({ error: 'Model storage is not configured.' }, 503, {
      'cache-control': 'no-store',
    });
  }

  const nowMs = Date.now();
  const auth = await requireAdminSession(request, env, nowMs);
  if (auth.response) return auth.response;
  const session = auth.session!;
  try {
    await touchSessionIfStale(env, session, nowMs);
  } catch (error) {
    console.error('[admin] touch session failed', error);
  }

  const payload = (await request.json().catch(() => null)) as {
    modeId?: unknown;
    archId?: unknown;
    queuePolicyId?: unknown;
    pipelineId?: unknown;
    pieceSourceProfile?: unknown;
    metrics?: unknown;
    policyArtifact?: unknown;
    setCurrent?: unknown;
    pin?: unknown;
  } | null;
  const modeId = normalizeGameMode(payload?.modeId);
  if (!modeId) {
    return jsonResponse({ error: 'Missing or invalid mode id.' }, 400, {
      'cache-control': 'no-store',
    });
  }
  const selector: BotPolicySelector = {
    modeId,
    archId: normalizeModelAxis(asString(payload?.archId), DEFAULT_BOT_ARCH_ID),
    queuePolicyId: normalizeModelAxis(
      asString(payload?.queuePolicyId),
      DEFAULT_BOT_QUEUE_POLICY_ID,
    ),
  };
  const pipelineId = normalizeModelAxis(
    asString(payload?.pipelineId),
    DEFAULT_BOT_PIPELINE_ID,
  );
  const pieceSourceProfile = normalizeModelAxis(
    asString(payload?.pieceSourceProfile),
    DEFAULT_BOT_PIECE_SOURCE_PROFILE,
  );
  const metricsObject =
    payload?.metrics != null &&
    typeof payload.metrics === 'object' &&
    !Array.isArray(payload.metrics)
      ? (payload.metrics as Record<string, unknown>)
      : null;
  const artifactObject =
    payload?.policyArtifact != null &&
    typeof payload.policyArtifact === 'object' &&
    !Array.isArray(payload.policyArtifact)
      ? (payload.policyArtifact as Record<string, unknown>)
      : null;
  if (!artifactObject) {
    return jsonResponse({ error: 'Missing policyArtifact object.' }, 400, {
      'cache-control': 'no-store',
    });
  }
  const artifactJson = JSON.stringify(artifactObject);
  const artifactBytes = new TextEncoder().encode(artifactJson);
  if (artifactBytes.byteLength > MAX_PERSONALIZED_MODEL_BYTES) {
    return jsonResponse(
      {
        error: `Policy artifact is too large. Max bytes: ${MAX_PERSONALIZED_MODEL_BYTES}.`,
      },
      413,
      { 'cache-control': 'no-store' },
    );
  }

  const setCurrent = asBoolean(payload?.setCurrent) === true;
  const pin = asBoolean(payload?.pin) === true;

  const nextVersion = await readNextBotPolicyVersion(env, selector);
  const policyId = `bot_policy_${crypto.randomUUID()}`;
  const r2Key =
    `bot-policies/${selector.modeId}` +
    `/${selector.archId}` +
    `/${selector.queuePolicyId}` +
    `/v${nextVersion}-${nowMs}-${policyId}.json`;

  let stored = false;
  try {
    await env.MODELS_BUCKET.put(r2Key, artifactBytes, {
      httpMetadata: { contentType: 'application/json' },
    });
    stored = true;
    await env.DB.prepare(
      `INSERT INTO bot_policies (
         id,
         mode_id,
         arch_id,
         queue_policy_id,
         pipeline_id,
         piece_source_profile,
         r2_key,
         version,
         is_pinned,
         metrics_json,
         created_at_ms
       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    )
      .bind(
        policyId,
        selector.modeId,
        selector.archId,
        selector.queuePolicyId,
        pipelineId,
        pieceSourceProfile,
        r2Key,
        nextVersion,
        pin ? 1 : 0,
        metricsObject ? JSON.stringify(metricsObject) : null,
        nowMs,
      )
      .run();
    const created = await readBotPolicyById(env, policyId);
    if (!created) {
      throw new Error('Published policy could not be loaded.');
    }
    if (setCurrent) {
      await setCurrentBotPolicy(env, created, nowMs);
    }
    return jsonResponse(
      {
        ok: true,
        policy: toBotPolicyResponse(created),
        setCurrent,
      },
      200,
      { 'cache-control': 'no-store' },
    );
  } catch (error) {
    if (stored) {
      await env.MODELS_BUCKET.delete(r2Key).catch((cleanupError) => {
        console.warn('[admin] bot policy publish cleanup failed', cleanupError);
      });
    }
    console.error('[admin] bot policy publish failed', error);
    return jsonResponse(
      { error: 'Bot policy publish is currently unavailable.' },
      503,
      { 'cache-control': 'no-store' },
    );
  }
};

const handleAdminBotPolicySelectCurrent = async (
  request: Request,
  env: Env,
): Promise<Response> => {
  if (request.method !== 'POST') {
    return jsonResponse({ error: 'Method not allowed.' }, 405, {
      'cache-control': 'no-store',
    });
  }
  const payload = (await request.json().catch(() => null)) as {
    id?: unknown;
  } | null;
  const id = asString(payload?.id);
  if (!id || !isValidBotPolicyId(id)) {
    return jsonResponse({ error: 'Missing or invalid policy id.' }, 400, {
      'cache-control': 'no-store',
    });
  }

  const nowMs = Date.now();
  const auth = await requireAdminSession(request, env, nowMs);
  if (auth.response) return auth.response;
  const session = auth.session!;
  try {
    await touchSessionIfStale(env, session, nowMs);
  } catch (error) {
    console.error('[admin] touch session failed', error);
  }

  try {
    const policy = await readBotPolicyById(env, id);
    if (!policy) {
      return jsonResponse({ error: 'Bot policy not found.' }, 404, {
        'cache-control': 'no-store',
      });
    }
    await setCurrentBotPolicy(env, policy, nowMs);
    return jsonResponse(
      {
        ok: true,
        policy: toBotPolicyResponse(policy),
      },
      200,
      { 'cache-control': 'no-store' },
    );
  } catch (error) {
    console.error('[admin] bot policy select current failed', error);
    return jsonResponse(
      { error: 'Bot policy registry is currently unavailable.' },
      503,
      { 'cache-control': 'no-store' },
    );
  }
};

const handleAdminBotPolicyPinToggle = async (
  request: Request,
  env: Env,
  options: { pin: boolean },
): Promise<Response> => {
  if (request.method !== 'POST') {
    return jsonResponse({ error: 'Method not allowed.' }, 405, {
      'cache-control': 'no-store',
    });
  }
  const payload = (await request.json().catch(() => null)) as {
    id?: unknown;
  } | null;
  const id = asString(payload?.id);
  if (!id || !isValidBotPolicyId(id)) {
    return jsonResponse({ error: 'Missing or invalid policy id.' }, 400, {
      'cache-control': 'no-store',
    });
  }

  const nowMs = Date.now();
  const auth = await requireAdminSession(request, env, nowMs);
  if (auth.response) return auth.response;
  const session = auth.session!;
  try {
    await touchSessionIfStale(env, session, nowMs);
  } catch (error) {
    console.error('[admin] touch session failed', error);
  }

  try {
    await env.DB.prepare(
      `UPDATE bot_policies
       SET is_pinned = ?
       WHERE id = ?`,
    )
      .bind(options.pin ? 1 : 0, id)
      .run();
    const policy = await readBotPolicyById(env, id);
    if (!policy) {
      return jsonResponse({ error: 'Bot policy not found.' }, 404, {
        'cache-control': 'no-store',
      });
    }
    return jsonResponse(
      {
        ok: true,
        policy: toBotPolicyResponse(policy),
      },
      200,
      { 'cache-control': 'no-store' },
    );
  } catch (error) {
    console.error('[admin] bot policy pin toggle failed', error);
    return jsonResponse(
      { error: 'Bot policy registry is currently unavailable.' },
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

    if (url.pathname === '/api/models/global/list') {
      return handleListGlobalModels(request, env);
    }

    if (url.pathname === '/api/models/global/current') {
      return handleGetCurrentGlobalModel(request, env);
    }

    if (url.pathname === '/api/models/me/reset') {
      return handleResetCurrentPersonalModel(request, env);
    }

    if (url.pathname === '/api/recordings/me/trajectory') {
      return handlePostTrajectoryRecording(request, env);
    }

    if (url.pathname === '/api/admin/whoami') {
      return handleAdminWhoAmI(request, env);
    }

    if (url.pathname === '/api/admin/recordings/index') {
      return handleAdminRecordingsIndex(request, env);
    }

    if (url.pathname === '/api/admin/recordings/export-manifest') {
      return handleAdminRecordingsExportManifest(request, env);
    }

    if (url.pathname === '/api/admin/recordings/object') {
      return handleAdminRecordingObject(request, env);
    }

    if (url.pathname === '/api/admin/models/global/publish') {
      return handleAdminPublishGlobalModel(request, env);
    }

    if (url.pathname === '/api/admin/models/global/index') {
      return handleAdminGlobalModelsIndex(request, env);
    }

    if (url.pathname === '/api/admin/models/global/set-default') {
      return handleAdminSetGlobalModelDefault(request, env);
    }

    if (url.pathname === '/api/admin/models/global/retire') {
      return handleAdminRetireGlobalModel(request, env);
    }

    if (url.pathname === '/api/admin/bot/policies/current') {
      return handleAdminBotPoliciesCurrent(request, env);
    }

    if (url.pathname === '/api/admin/bot/policies/list') {
      return handleAdminBotPoliciesList(request, env);
    }

    if (url.pathname === '/api/admin/bot/policies/object') {
      return handleAdminBotPolicyObject(request, env);
    }

    if (url.pathname === '/api/admin/bot/policies/publish') {
      return handleAdminBotPolicyPublish(request, env);
    }

    if (url.pathname === '/api/admin/bot/policies/select-current') {
      return handleAdminBotPolicySelectCurrent(request, env);
    }

    if (url.pathname === '/api/admin/bot/policies/pin') {
      return handleAdminBotPolicyPinToggle(request, env, { pin: true });
    }

    if (url.pathname === '/api/admin/bot/policies/unpin') {
      return handleAdminBotPolicyPinToggle(request, env, { pin: false });
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
