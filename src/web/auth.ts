import { createHash, createHmac, randomUUID, timingSafeEqual } from 'node:crypto';
import type { IncomingMessage } from 'node:http';

const COOKIE_NAME = 'muz_session';
const LOGIN_WINDOW_MS = 10 * 60_000;
const LOGIN_LIMIT = 8;

export function readCookie(header: string | undefined, name: string): string | null {
  if (!header) return null;
  for (const part of header.split(';')) {
    const index = part.indexOf('=');
    if (index < 0) continue;
    if (part.slice(0, index).trim() !== name) continue;
    const value = part.slice(index + 1).trim();
    return value || null;
  }
  return null;
}

function digest(value: string): Buffer {
  return createHash('sha256').update(value, 'utf8').digest();
}

export class WebAuth {
  private readonly passwordHash: Buffer;
  private readonly attempts = new Map<string, { count: number; resetAt: number }>();

  constructor(password: string, private readonly secret: Buffer,
    private readonly ttlMs: number, private readonly secureCookies: boolean) {
    this.passwordHash = digest(password);
  }

  /** Returns a token when the password matches, a failure reason otherwise. */
  login(ip: string, value: string): { token: string } | 'invalid' | 'limited' {
    const now = Date.now();
    const current = this.attempts.get(ip);
    if (current && current.resetAt > now && current.count >= LOGIN_LIMIT) return 'limited';
    const candidate = digest(value);
    const matches = candidate.length === this.passwordHash.length && timingSafeEqual(candidate, this.passwordHash);
    if (!matches) {
      const bucket = current && current.resetAt > now ? current : { count: 0, resetAt: now + LOGIN_WINDOW_MS };
      bucket.count++;
      this.attempts.set(ip, bucket);
      if (this.attempts.size > 5000) {
        for (const [key, item] of this.attempts) if (item.resetAt <= now) this.attempts.delete(key);
      }
      return 'invalid';
    }
    this.attempts.delete(ip);
    return { token: this.issue() };
  }

  issue(): string {
    const payload = Buffer.from(JSON.stringify({
      v: 1,
      exp: Date.now() + this.ttlMs,
      jti: randomUUID(),
    }), 'utf8').toString('base64url');
    return `${payload}.${this.sign(payload)}`;
  }

  verify(token: string | null | undefined): boolean {
    if (!token) return false;
    const dot = token.indexOf('.');
    if (dot < 1 || dot === token.length - 1) return false;
    const payload = token.slice(0, dot);
    const signature = token.slice(dot + 1);
    if (!this.matches(signature, this.sign(payload))) return false;
    try {
      const data: unknown = JSON.parse(Buffer.from(payload, 'base64url').toString('utf8'));
      if (!data || typeof data !== 'object' || Array.isArray(data)) return false;
      const exp = (data as { exp?: unknown }).exp;
      return typeof exp === 'number' && Number.isFinite(exp) && exp > Date.now();
    } catch {
      return false;
    }
  }

  verifyRequest(req: IncomingMessage): boolean {
    return this.verify(readCookie(req.headers.cookie, COOKIE_NAME));
  }

  cookie(token?: string | null): string {
    const parts = [
      `${COOKIE_NAME}=${token ?? ''}`,
      'Path=/',
      'HttpOnly',
      'SameSite=Lax',
      `Max-Age=${token ? Math.floor(this.ttlMs / 1000) : 0}`,
    ];
    if (this.secureCookies) parts.push('Secure');
    return parts.join('; ');
  }

  private sign(payload: string): string {
    return createHmac('sha256', this.secret).update(payload, 'utf8').digest('base64url');
  }

  private matches(left: string, right: string): boolean {
    const a = Buffer.from(left, 'utf8');
    const b = Buffer.from(right, 'utf8');
    return a.length === b.length && timingSafeEqual(a, b);
  }
}
