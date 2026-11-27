import { randomBytes, randomInt } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

export interface WebConfig {
  host: string;
  port: number;
  publicUrl: string | null;
  adminPassword: string;
  adminPasswordFile: string | null;
  passwordGenerated: boolean;
  sessionSecret: Buffer;
  secureCookies: boolean;
  sessionTtlMs: number;
  uiDir: string;
  stateFile: string;
  aliceSkillId: string | null;
  aliceSkillName: string;
  aliceWebhookSecret: string | null;
  trustProxy: boolean;
}

const DEFAULT_PORT = 8787;
const SESSION_TTL_MS = 7 * 24 * 60 * 60_000;
const PASSWORD_ALPHABET = 'abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789';

function flag(name: string, fallback: boolean): boolean {
  const value = process.env[name];
  if (value === undefined || value === '') return fallback;
  return value === '1' || value.toLowerCase() === 'true';
}

function text(name: string): string | null {
  const value = process.env[name]?.trim();
  return value ? value : null;
}

function positiveInteger(name: string, fallback: number, min: number, max: number): number {
  const raw = process.env[name];
  if (!raw) return fallback;
  const value = Number(raw);
  return Number.isInteger(value) && value >= min && value <= max ? value : fallback;
}

async function readOrCreate(file: string, create: () => string): Promise<{ value: string; created: boolean }> {
  try {
    const value = (await readFile(file, 'utf8')).trim();
    if (value) return { value, created: false };
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
  }
  const value = create();
  await mkdir(path.dirname(file), { recursive: true });
  await writeFile(file, `${value}\n`, { encoding: 'utf8', mode: 0o600 });
  return { value, created: true };
}

function generatedPassword(): string {
  return Array.from({ length: 20 }, () => PASSWORD_ALPHABET[randomInt(PASSWORD_ALPHABET.length)]).join('');
}

function secretBuffer(value: string): Buffer {
  // An explicitly configured value is used verbatim; generated secrets are
  // written as 64 hex characters. Both are stable across restarts.
  return /^[0-9a-f]{64}$/i.test(value) ? Buffer.from(value, 'hex') : Buffer.from(value, 'utf8');
}

/**
 * The website is enabled unless WEB_ENABLED=0. It binds to localhost by
 * default; a public reverse proxy can forward the Alice webhook without
 * exposing the admin panel.
 */
export async function loadWebConfig(): Promise<WebConfig | null> {
  if (!flag('WEB_ENABLED', true)) return null;

  const runtimeDir = path.resolve('.runtime/web');
  const publicUrl = text('WEB_PUBLIC_URL')?.replace(/\/+$/u, '') ?? null;
  const passwordFromEnv = text('WEB_ADMIN_PASSWORD');
  const passwordFile = path.join(runtimeDir, 'admin-password.txt');
  const password = passwordFromEnv
    ? { value: passwordFromEnv, created: false }
    : await readOrCreate(passwordFile, generatedPassword);
  if (passwordFromEnv && passwordFromEnv.length < 12) {
    console.warn('[Web] WEB_ADMIN_PASSWORD короче 12 символов; используйте более длинный пароль.');
  }

  const secretFromEnv = text('WEB_SESSION_SECRET');
  const secretFile = path.join(runtimeDir, 'session-secret');
  const secret = secretFromEnv
    ? { value: secretFromEnv, created: false }
    : await readOrCreate(secretFile, () => randomBytes(32).toString('hex'));

  const skillId = text('ALICE_SKILL_ID');
  const webhookSecret = text('ALICE_WEBHOOK_SECRET');
  if (webhookSecret && webhookSecret.length < 16) {
    console.warn('[Web] ALICE_WEBHOOK_SECRET короче 16 символов; используйте длинную случайную строку.');
  }

  return {
    host: text('WEB_HOST') ?? '127.0.0.1',
    port: positiveInteger('WEB_PORT', DEFAULT_PORT, 1, 65535),
    publicUrl,
    adminPassword: password.value,
    adminPasswordFile: passwordFromEnv ? null : passwordFile,
    passwordGenerated: password.created,
    sessionSecret: secretBuffer(secret.value),
    secureCookies: flag('WEB_SECURE', publicUrl?.startsWith('https://') ?? false),
    sessionTtlMs: SESSION_TTL_MS,
    uiDir: path.resolve(text('WEB_UI_DIR') ?? 'web'),
    stateFile: path.resolve(text('WEB_STATE_FILE') ?? path.join(runtimeDir, 'state.json')),
    aliceSkillId: skillId,
    aliceSkillName: text('ALICE_SKILL_NAME') ?? 'Муза',
    aliceWebhookSecret: webhookSecret,
    trustProxy: flag('WEB_TRUST_PROXY', false),
  };
}
