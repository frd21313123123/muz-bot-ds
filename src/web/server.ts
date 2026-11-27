import { randomBytes } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import path from 'node:path';
import type { MusicClient } from '../types.js';
import { loadWebConfig, type WebConfig } from './config.js';
import { WebAuth } from './auth.js';
import { SpeakerStore, validateSpeakerName, type Speaker } from './store.js';
import { PlayerService, radioStationById, type PlayerAction } from './player.js';
import { EventLog } from './log.js';
import { AliceAuthError, handleAliceRequest, quickAliceResponse } from './alice.js';
import { parsePairingCode, runVoiceCommand } from './commands.js';
import type { SpeakerView, StateView, WebTarget } from './types.js';

const BODY_LIMIT = 256 * 1024;
const ALICE_BODY_LIMIT = 128 * 1024;
const ALICE_DEADLINE_MS = 2_800;
const PAIR_ATTEMPT_LIMIT = 15;
const PAIR_WINDOW_MS = 10 * 60_000;
const CSP = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
  + "img-src 'self' https: data:; connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'";

class HttpError extends Error {
  constructor(readonly status: number, message: string) { super(message); this.name = 'HttpError'; }
}

export interface WebContext {
  client: MusicClient;
  config: WebConfig;
  auth: WebAuth;
  store: SpeakerStore;
  player: PlayerService;
  log: EventLog;
  pairAttempts: Map<string, { count: number; resetAt: number }>;
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

function noContent(res: ServerResponse): void {
  if (res.writableEnded) return;
  res.statusCode = 204;
  res.setHeader('Cache-Control', 'no-store');
  res.end();
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

function publicSpeaker(speaker: Speaker, online: boolean): SpeakerView {
  return {
    id: speaker.id, name: speaker.name, kind: speaker.kind, guildId: speaker.guildId,
    channelId: speaker.channelId, createdAt: speaker.createdAt, lastSeenAt: speaker.lastSeenAt, online,
  };
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

function bearer(req: IncomingMessage): string | null {
  const header = req.headers.authorization;
  if (!header || !header.toLowerCase().startsWith('bearer ')) return null;
  return header.slice(7).trim() || null;
}

function bridgeSpeaker(req: IncomingMessage, ctx: WebContext): Speaker | null {
  const token = bearer(req);
  return token ? ctx.store.findByToken(token) ?? null : null;
}

function stateOf(ctx: WebContext): StateView {
  const snapshot = ctx.player.snapshot();
  const now = Date.now();
  return {
    bot: snapshot.bot,
    guilds: snapshot.guilds,
    speakers: ctx.store.list().map(speaker => publicSpeaker(speaker, ctx.store.online(speaker, now))),
    lastTarget: ctx.store.last(),
    radioStations: snapshot.radioStations,
    skill: {
      name: ctx.config.aliceSkillName,
      publicUrl: ctx.config.publicUrl,
      configured: Boolean(ctx.config.aliceSkillId),
    },
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

function pairAllowed(ctx: WebContext, ip: string): boolean {
  const now = Date.now();
  if (ctx.pairAttempts.size > 2000) {
    for (const [key, bucket] of ctx.pairAttempts) if (bucket.resetAt <= now) ctx.pairAttempts.delete(key);
  }
  const bucket = ctx.pairAttempts.get(ip);
  if (!bucket || bucket.resetAt <= now) {
    ctx.pairAttempts.set(ip, { count: 1, resetAt: now + PAIR_WINDOW_MS });
    return true;
  }
  bucket.count++;
  return bucket.count <= PAIR_ATTEMPT_LIMIT;
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

  if (method === 'POST' && pathname === '/api/speakers/code') {
    if (!requireAdmin(req, res, ctx) || !requireCsrf(req, res)) return;
    const body = await readJson(req, 16 * 1024);
    const guildId = body.guildId;
    const channelId = body.channelId;
    if (!isSnowflake(guildId) || !isSnowflake(channelId)) throw new HttpError(400, 'Выберите сервер и голосовой канал.');
    const guild = ctx.player.snapshot().guilds.find(item => item.id === guildId);
    const channel = guild?.channels.find(item => item.id === channelId);
    if (!guild || !channel) throw new HttpError(400, 'Голосовой канал не найден.');
    if (!channel.canJoin) throw new HttpError(400, 'У бота нет прав на подключение и речь в этом канале.');
    let name: string;
    try { name = validateSpeakerName(string(body.name) ?? 'Колонка'); }
    catch (error) { throw new HttpError(400, error instanceof Error ? error.message : 'Некорректное название.'); }
    const created = ctx.store.createCode({ guildId, channelId }, name);
    ctx.log.add('web', `Создан код для «${created.name}»`, true);
    json(res, 200, { ok: true, code: created.code, expiresAt: created.expiresAt, skillName: ctx.config.aliceSkillName });
    return;
  }

  const speakerMatch = /^\/api\/speakers\/([0-9a-f-]{36})$/u.exec(pathname);
  if (speakerMatch) {
    if (!requireAdmin(req, res, ctx) || !requireCsrf(req, res)) return;
    const speakerId = speakerMatch[1]!;
    if (method === 'DELETE') {
      const removed = await ctx.store.remove(speakerId);
      if (!removed) throw new HttpError(404, 'Колонка не найдена.');
      ctx.log.add('web', 'Колонка отключена', true);
      json(res, 200, { ok: true });
      return;
    }
    if (method === 'PATCH') {
      if (!ctx.store.get(speakerId)) throw new HttpError(404, 'Колонка не найдена.');
      const body = await readJson(req, 16 * 1024);
      let name: string;
      try { name = validateSpeakerName(body.name); }
      catch (error) { throw new HttpError(400, error instanceof Error ? error.message : 'Некорректное название.'); }
      await ctx.store.rename(speakerId, name);
      json(res, 200, { ok: true });
      return;
    }
  }

  if (method === 'POST' && pathname === '/api/alice/pair') {
    const ip = clientIp(req, ctx.config.trustProxy);
    if (!pairAllowed(ctx, ip)) throw new HttpError(429, 'Слишком много попыток. Повторите позже.');
    const body = await readJson(req, 16 * 1024);
    const code = string(body.code);
    if (!code) throw new HttpError(400, 'Укажите код подключения.');
    let requestedName: string | undefined;
    if (string(body.name)) {
      try { requestedName = validateSpeakerName(body.name); }
      catch (error) { throw new HttpError(400, error instanceof Error ? error.message : 'Некорректное название.'); }
    }
    const pairing = ctx.store.takeCode(code);
    if (!pairing) throw new HttpError(400, 'Код неверный или истёк.');
    const token = randomBytes(32).toString('base64url');
    const speaker = await ctx.store.pairBridge(token, pairing, requestedName ?? pairing.name);
    ctx.log.add('bridge', `Подключён мост «${speaker.name}»`, true);
    json(res, 200, { ok: true, token, speaker: publicSpeaker(speaker, true) });
    return;
  }

  if (pathname === '/api/alice/state' && method === 'GET') {
    const speaker = bridgeSpeaker(req, ctx);
    if (!speaker) throw new HttpError(401, 'Неверный токен моста.');
    ctx.store.touch(speaker.id);
    const snapshot = ctx.player.snapshot();
    const guild = snapshot.guilds.find(item => item.id === speaker.guildId);
    const channel = guild?.channels.find(item => item.id === speaker.channelId);
    json(res, 200, {
      ok: true,
      speaker: publicSpeaker(speaker, true),
      guild: guild ? { id: guild.id, name: guild.name } : null,
      channel: channel ? { id: channel.id, name: channel.name } : null,
      queue: ctx.player.queue({ guildId: speaker.guildId, channelId: speaker.channelId }),
      serverTime: Date.now(),
    });
    return;
  }

  if (pathname === '/api/alice/heartbeat' && method === 'POST') {
    const speaker = bridgeSpeaker(req, ctx);
    if (!speaker) throw new HttpError(401, 'Неверный токен моста.');
    ctx.store.touch(speaker.id);
    noContent(res);
    return;
  }

  if (pathname === '/api/alice/command' && method === 'POST') {
    const speaker = bridgeSpeaker(req, ctx);
    if (!speaker) throw new HttpError(401, 'Неверный токен моста.');
    ctx.store.touch(speaker.id);
    const body = await readJson(req);
    const command = string(body.text);
    if (!command) throw new HttpError(400, 'Пустая команда.');
    const target: WebTarget = { guildId: speaker.guildId, channelId: speaker.channelId };
    const result = await runVoiceCommand(ctx.player, target, command, ctx.config.aliceSkillName);
    ctx.log.add('bridge', `${speaker.name}: ${command}`, result.ok);
    json(res, 200, {
      ok: result.ok, text: result.text, keepSession: result.keepSession,
      queue: ctx.player.queue(target),
    });
    return;
  }

  const alicePath = ctx.config.aliceWebhookSecret
    ? `/alice/webhook/${ctx.config.aliceWebhookSecret}`
    : '/alice/webhook';
  if (method === 'POST' && pathname === alicePath) {
    const body = await readJson(req, ALICE_BODY_LIMIT);
    const request = isObject(body.request) ? body.request : {};
    const command = string(request.command) ?? string(request.original_utterance) ?? '';
    if (command && parsePairingCode(command)
      && !pairAllowed(ctx, clientIp(req, ctx.config.trustProxy))) {
      json(res, 200, quickAliceResponse(body, 'Слишком много попыток подключения. Повторите позже.', true));
      return;
    }
    const deps = {
      store: ctx.store, player: ctx.player, skillId: ctx.config.aliceSkillId,
      skillName: ctx.config.aliceSkillName,
      log: (kind: string, message: string, ok: boolean) => ctx.log.add(kind, message, ok),
    };
    let timer: NodeJS.Timeout | undefined;
    const operation = handleAliceRequest(body, deps);
    const timeout = new Promise<'timeout'>((resolve) => {
      timer = setTimeout(() => resolve('timeout'), ALICE_DEADLINE_MS);
    });
    try {
      const result = await Promise.race([operation, timeout]);
      if (result === 'timeout') {
        operation.catch((error) => console.error('[Alice] slow command failed:', error));
        json(res, 200, quickAliceResponse(body, 'Секунду, ищу и включаю.', true));
        return;
      }
      json(res, 200, result);
    } catch (error) {
      if (error instanceof AliceAuthError) throw new HttpError(403, error.message);
      console.error('[Alice] webhook failed:', error);
      json(res, 200, quickAliceResponse(body, 'Не получилось выполнить команду.', false));
    } finally {
      if (timer) clearTimeout(timer);
    }
    return;
  }

  // Keep the public surface small: everything else is a 404, including a
  // webhook called without the configured secret path.
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
  const store = new SpeakerStore(config.stateFile);
  try { await store.load(); }
  catch (error) {
    console.error('[Web] Не удалось прочитать состояние колонок; начинаю с пустого списка.', error);
  }
  const ctx: WebContext = {
    client,
    config,
    auth: new WebAuth(config.adminPassword, config.sessionSecret, config.sessionTtlMs, config.secureCookies),
    store,
    player: new PlayerService(client),
    log: new EventLog(),
    pairAttempts: new Map(),
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
  if (!config.aliceSkillId) {
    console.log('[Web] ALICE_SKILL_ID не задан: вебхук принимает запросы без проверки навыка (задайте ID для защиты).');
  }
  if (!config.aliceWebhookSecret) {
    console.log('[Web] ALICE_WEBHOOK_SECRET не задан: публичный вебхук не защищён секретным путём.');
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
