import { readFile } from 'node:fs/promises';
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import path from 'node:path';
import type { MusicClient } from '../types.js';
import { loadWebConfig, type WebConfig } from './config.js';
import { WebAuth } from './auth.js';
import { WebStateStore } from './store.js';
import { PlayerService, radioStationById, type PlayerAction, type PlayerApi } from './player.js';
import { EventLog } from './log.js';
import type { StateView, WebTarget } from './types.js';

const BODY_LIMIT = 256 * 1024;
const CSP = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
  + "img-src 'self' https: data:; connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'";

class HttpError extends Error {
  constructor(readonly status: number, message: string) { super(message); this.name = 'HttpError'; }
}

export interface WebContext {
  client: MusicClient;
  config: WebConfig;
  auth: WebAuth;
  store: WebStateStore;
  player: PlayerApi;
  log: EventLog;
}

export interface ControlServer {
  host: string;
  port: number;
  url: string;
  close(): Promise<void>;
}

function isObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function isSnowflake(value: unknown): value is string {
  return typeof value === 'string' && /^\d{5,25}$/u.test(value);
}

function string(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value.trim() : null;
}

function clientIp(req: IncomingMessage, trustProxy: boolean): string {
  if (trustProxy) {
    const forwarded = req.headers['x-forwarded-for'];
    const first = (Array.isArray(forwarded) ? forwarded[0] : forwarded)?.split(',')[0]?.trim();
    if (first) return first;
  }
  return req.socket.remoteAddress ?? 'unknown';
}

function setCommonHeaders(res: ServerResponse): void {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('Referrer-Policy', 'no-referrer');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('Permissions-Policy', 'camera=(), microphone=(), geolocation=()');
}

function json(res: ServerResponse, status: number, payload: unknown): void {
  if (res.writableEnded) return;
  res.statusCode = status;
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.setHeader('Cache-Control', 'no-store');
  res.end(JSON.stringify(payload));
}

function text(res: ServerResponse, status: number, body: string): void {
  if (res.writableEnded) return;
  res.statusCode = status;
  res.setHeader('Content-Type', 'text/plain; charset=utf-8');
  res.setHeader('Cache-Control', 'no-store');
  res.end(body);
}

async function readJson(req: IncomingMessage, limit = BODY_LIMIT): Promise<Record<string, unknown>> {
  const type = (Array.isArray(req.headers['content-type']) ? req.headers['content-type'][0] : req.headers['content-type']) ?? '';
  if (!type.toLowerCase().includes('application/json')) throw new HttpError(415, 'Ожидается JSON.');
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk as string);
    size += buffer.length;
    if (size > limit) throw new HttpError(413, 'Слишком большой запрос.');
    chunks.push(buffer);
  }
  if (!size) return {};
  try {
    const data: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    if (!isObject(data)) throw new Error('not object');
    return data;
  } catch {
    throw new HttpError(400, 'Некорректный JSON.');
  }
}

function requireAdmin(req: IncomingMessage, res: ServerResponse, ctx: WebContext): boolean {
  if (ctx.auth.verifyRequest(req)) return true;
  json(res, 401, { error: 'Требуется вход в пульт.' });
  return false;
}

function requireCsrf(req: IncomingMessage, res: ServerResponse): boolean {
  if (req.headers['x-requested-with'] === 'muz-web') return true;
  json(res, 403, { error: 'Запрос отклонён.' });
  return false;
}

function stateOf(ctx: WebContext): StateView {
  const snapshot = ctx.player.snapshot();
  return {
    bot: snapshot.bot,
    guilds: snapshot.guilds,
    lastTarget: ctx.store.last(),
    radioStations: snapshot.radioStations,
    sampledAt: snapshot.sampledAt,
    log: ctx.log.list(),
  };
}

function playerFailure(error: unknown): string {
  const message = error instanceof Error ? error.message : '';
  if (/ничего не найдено/iu.test(message)) return 'Ничего не найдено.';
  if (/уже работает в другом/iu.test(message)) return 'Бот уже работает в другом голосовом канале.';
  if (/прав/iu.test(message)) return 'Нет прав на подключение и речь в этом канале.';
  if (/рекомендации|повтор недоступен/iu.test(message)) return 'Это недоступно во время радио.';
  if (/радио|эфир/iu.test(message)) return 'Радиостанция сейчас недоступна.';
  if (/голосовой канал не найден|сервер discord не найден/iu.test(message)) return 'Сервер или голосовой канал не найдены.';
  if (/громкость|запрос|название/iu.test(message)) return message;
  return 'Не удалось выполнить действие.';
}

async function serveStatic(req: IncomingMessage, res: ServerResponse, ctx: WebContext, pathname: string): Promise<boolean> {
  const files: Record<string, { name: string; type: string; html?: boolean }> = {
    '/': { name: 'index.html', type: 'text/html; charset=utf-8', html: true },
    '/index.html': { name: 'index.html', type: 'text/html; charset=utf-8', html: true },
    '/app.js': { name: 'app.js', type: 'text/javascript; charset=utf-8' },
    '/styles.css': { name: 'styles.css', type: 'text/css; charset=utf-8' },
    '/favicon.svg': { name: 'favicon.svg', type: 'image/svg+xml' },
  };
  const entry = files[pathname];
  if (!entry) return false;
  let body: Buffer;
  try {
    body = await readFile(path.join(ctx.config.uiDir, entry.name));
  } catch {
    text(res, 404, 'Интерфейс пульта не найден. Проверьте папку web/.');
    return true;
  }
  res.statusCode = 200;
  res.setHeader('Content-Type', entry.type);
  res.setHeader('Cache-Control', 'no-store');
  if (entry.html) res.setHeader('Content-Security-Policy', CSP);
  res.end(req.method === 'HEAD' ? undefined : body);
  return true;
}

async function route(req: IncomingMessage, res: ServerResponse, ctx: WebContext): Promise<void> {
  setCommonHeaders(res);
  const method = req.method ?? 'GET';
  const url = new URL(req.url ?? '/', `http://${req.headers.host ?? 'localhost'}`);
  const pathname = url.pathname;

  if ((method === 'GET' || method === 'HEAD') && await serveStatic(req, res, ctx, pathname)) return;

  if (method === 'GET' && pathname === '/healthz') {
    json(res, 200, { ok: true });
    return;
  }

  if (method === 'GET' && pathname === '/api/session') {
    json(res, 200, { authenticated: ctx.auth.verifyRequest(req) });
    return;
  }

  if (method === 'POST' && pathname === '/api/login') {
    if (!requireCsrf(req, res)) return;
    const body = await readJson(req, 16 * 1024);
    const password = string(body.password);
    if (!password) throw new HttpError(400, 'Укажите пароль.');
    const ip = clientIp(req, ctx.config.trustProxy);
    const result = ctx.auth.login(ip, password);
    if (result === 'limited') throw new HttpError(429, 'Слишком много попыток. Повторите через десять минут.');
    if (result === 'invalid') {
      ctx.log.add('auth', 'Неудачная попытка входа', false);
      throw new HttpError(401, 'Неверный пароль.');
    }
    res.setHeader('Set-Cookie', ctx.auth.cookie(result.token));
    ctx.log.add('auth', 'Вход в пульт', true);
    json(res, 200, { ok: true });
    return;
  }

  if (method === 'POST' && pathname === '/api/logout') {
    if (!requireCsrf(req, res)) return;
    res.setHeader('Set-Cookie', ctx.auth.cookie(null));
    json(res, 200, { ok: true });
    return;
  }

  if (method === 'GET' && pathname === '/api/state') {
    if (!requireAdmin(req, res, ctx)) return;
    json(res, 200, stateOf(ctx));
    return;
  }

  if (method === 'POST' && pathname === '/api/player') {
    if (!requireAdmin(req, res, ctx) || !requireCsrf(req, res)) return;
    const body = await readJson(req);
    const guildId = body.guildId;
    const channelId = body.channelId;
    const action = string(body.action);
    if (!isSnowflake(guildId) || !isSnowflake(channelId)) throw new HttpError(400, 'Выберите сервер и голосовой канал.');
    if (!action) throw new HttpError(400, 'Не указано действие.');
    const target: WebTarget = { guildId, channelId };
    try {
      if (action === 'play') {
        const query = string(body.query);
        if (!query) throw new HttpError(400, 'Укажите название трека или ссылку.');
        const source = body.source === 'soundcloud' ? 'soundcloud' : 'youtube';
        const result = await ctx.player.play(target, query, 'Сайт', { source, move: body.move !== false });
        await ctx.store.setLastTarget(target);
        ctx.log.add('web', `Включено: ${result.track.title}`, true);
        json(res, 200, { ok: true, ...result });
        return;
      }
      if (action === 'radio') {
        const stationId = string(body.stationId);
        const station = stationId ? radioStationById(stationId) : null;
        if (!station) throw new HttpError(400, 'Радиостанция не найдена.');
        const result = await ctx.player.radio(target, station, 'Сайт', { move: body.move !== false });
        await ctx.store.setLastTarget(target);
        ctx.log.add('web', `Радио: ${result.station}`, true);
        json(res, 200, { ok: true, station: result.station });
        return;
      }
      const actions: readonly PlayerAction[] = ['pause', 'resume', 'skip', 'stop', 'volume', 'autoplay', 'loop', 'clear'];
      if (!actions.includes(action as PlayerAction)) throw new HttpError(400, 'Неизвестное действие.');
      const value = action === 'volume' ? body.volume : body.value;
      if (value !== undefined && typeof value !== 'number' && typeof value !== 'boolean') {
        throw new HttpError(400, 'Некорректное значение.');
      }
      const result = await ctx.player.act(target, action as PlayerAction, value as number | boolean | undefined);
      if (action !== 'stop') await ctx.store.setLastTarget(target);
      ctx.log.add('web', `Действие: ${action}`, result.changed);
      json(res, 200, { ok: true, ...result });
      return;
    } catch (error) {
      if (error instanceof HttpError) throw error;
      ctx.log.add('web', `${action}: ${error instanceof Error ? error.message : 'ошибка'}`, false);
      throw new HttpError(400, playerFailure(error));
    }
  }

  throw new HttpError(404, 'Не найдено.');
}

export function createControlHandler(ctx: WebContext): (req: IncomingMessage, res: ServerResponse) => void {
  return (req, res) => {
    void route(req, res, ctx).catch((error: unknown) => {
      if (error instanceof HttpError) {
        json(res, error.status, { error: error.message });
        return;
      }
      console.error('[Web] request failed:', error);
      if (!res.headersSent) json(res, 500, { error: 'Внутренняя ошибка пульта.' });
      else if (!res.writableEnded) res.end();
    });
  };
}

export async function startControlServer(client: MusicClient): Promise<ControlServer | null> {
  const config = await loadWebConfig();
  if (!config) return null;
  const store = new WebStateStore(config.stateFile);
  try { await store.load(); }
  catch (error) {
    console.error('[Web] Не удалось прочитать настройки пульта; используется пустое состояние.', error);
  }
  const ctx: WebContext = {
    client,
    config,
    auth: new WebAuth(config.adminPassword, config.sessionSecret, config.sessionTtlMs, config.secureCookies),
    store,
    player: new PlayerService(client),
    log: new EventLog(),
  };
  const server = createServer(createControlHandler(ctx));
  server.on('error', (error) => console.error('[Web] server error:', error));
  await new Promise<void>((resolve, reject) => {
    const onError = (error: Error): void => { server.off('listening', onListening); reject(error); };
    const onListening = (): void => { server.off('error', onError); resolve(); };
    server.once('error', onError);
    server.once('listening', onListening);
    server.listen(config.port, config.host);
  });
  const address = server.address();
  const port = typeof address === 'object' && address ? address.port : config.port;
  const shownHost = config.host === '0.0.0.0' ? '127.0.0.1' : config.host;
  const url = `http://${shownHost.includes(':') ? `[${shownHost}]` : shownHost}:${port}`;
  console.log(`[Web] Пульт: ${url}`);
  if (config.adminPasswordFile) {
    console.log(`[Web] Пароль администратора${config.passwordGenerated ? ' создан' : ''}: ${config.adminPasswordFile}`);
  } else {
    console.log('[Web] Пароль администратора взят из WEB_ADMIN_PASSWORD.');
  }
  return {
    host: config.host,
    port,
    url,
    close: async () => {
      await new Promise<void>((resolve) => {
        server.close(() => resolve());
        server.closeAllConnections();
      });
    },
  };
}
