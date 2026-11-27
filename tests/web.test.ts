import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { ChannelType, type Guild } from 'discord.js';
import type { MusicClient } from '../src/types.js';
import type { RadioStation } from '../src/utils/radio.js';
import { WebAuth } from '../src/web/auth.js';
import { EventLog } from '../src/web/log.js';
import { PlayerService, type PlayerApi, type PlayerSnapshot } from '../src/web/player.js';
import { createControlHandler, type WebContext } from '../src/web/server.js';
import { WebStateStore } from '../src/web/store.js';
import type { StateView, WebTarget } from '../src/web/types.js';
import type { WebConfig } from '../src/web/config.js';

class StubPlayer implements PlayerApi {
  calls: { action: string; target: WebTarget; query?: string; requestedBy?: string;
    options?: object; value?: number | boolean }[] = [];

  constructor(private readonly state: PlayerSnapshot) {}

  async play(target: WebTarget, query: string, requestedBy: string, options?: Parameters<PlayerApi['play']>[3]) {
    this.calls.push({ action: 'play', target, query, requestedBy, options });
    return {
      track: {
        title: query, url: 'https://www.youtube.com/watch?v=abcdefghijk', thumbnail: null,
        duration: '3:00', requestedBy, isAutoplay: false, isLive: false, source: 'youtube' as const,
      },
      started: true, playlist: false,
    };
  }
  async radio(target: WebTarget, station: RadioStation, requestedBy: string,
    options?: Parameters<PlayerApi['radio']>[3]) {
    this.calls.push({ action: 'radio', target, query: station.id, requestedBy, options });
    return { station: station.name };
  }
  async act(target: WebTarget, action: string, value?: number | boolean) {
    this.calls.push({ action, target, value });
    return { changed: true, level: typeof value === 'number' ? value : undefined };
  }
  snapshot() { return this.state; }
}

const target = { guildId: '123456789012345678', channelId: '223456789012345678' };

test('panel keeps its last playback destination across restarts', async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'muz-web-state-'));
  const file = path.join(dir, 'state.json');
  try {
    const store = new WebStateStore(file);
    await store.load();
    assert.equal(store.last(), null);
    await store.setLastTarget(target);
    await assert.rejects(store.setLastTarget({ ...target, channelId: 'invalid' }));
    const reloaded = new WebStateStore(file);
    await reloaded.load();
    assert.deepEqual(reloaded.last(), target);
    assert.deepEqual(JSON.parse(await readFile(file, 'utf8')), { version: 2, lastTarget: target });
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('panel upgrades stored preferences and discards obsolete device credentials', async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'muz-web-state-'));
  const file = path.join(dir, 'state.json');
  try {
    await writeFile(file, JSON.stringify({
      version: 1, lastTarget: target,
      speakers: [{ kind: 'bridge', deviceId: 'old-device', tokenHash: 'a'.repeat(64) }],
    }));
    const store = new WebStateStore(file);
    await store.load();
    assert.deepEqual(store.last(), target);
    assert.deepEqual(JSON.parse(await readFile(file, 'utf8')), { version: 2, lastTarget: target });

    await writeFile(file, JSON.stringify({ version: 1, lastTarget: { guildId: 'invalid', channelId: target.channelId } }));
    const invalid = new WebStateStore(file);
    await invalid.load();
    assert.equal(invalid.last(), null);
    assert.deepEqual(JSON.parse(await readFile(file, 'utf8')), { version: 2, lastTarget: null });
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('web auth rejects tampered and expired sessions', () => {
  const auth = new WebAuth('correct horse', Buffer.alloc(32, 7), 60_000, false);
  const token = auth.issue();
  assert.equal(auth.verify(token), true);
  assert.equal(auth.verify(`${token}x`), false);
  assert.equal(auth.login('1.2.3.4', 'wrong'), 'invalid');
  const short = new WebAuth('correct horse', Buffer.alloc(32, 7), -1, false);
  assert.equal(short.verify(short.issue()), false);
});

test('control HTTP API requires login and CSRF and retains panel playback controls', async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'muz-web-'));
  const store = new WebStateStore(path.join(dir, 'state.json'));
  const { guildId, channelId } = target;
  const channel = {
    id: channelId, name: 'General', type: ChannelType.GuildVoice,
    permissionsFor: () => ({ has: () => true }),
  };
  const guild = {
    id: guildId, name: 'Test Guild', iconURL: () => null,
    members: { me: { id: '333456789012345678' } },
    channels: { cache: new Map([[channelId, channel]]) },
  } as unknown as Guild;
  const client = {
    user: { id: '333456789012345678', tag: 'Bot#0001', displayAvatarURL: () => null },
    guilds: { cache: new Map([[guildId, guild]]) },
    queues: new Map(), ytdlp: {},
  } as unknown as MusicClient;
  const config: WebConfig = {
    host: '127.0.0.1', port: 0, publicUrl: null, adminPassword: 'secret',
    adminPasswordFile: null, passwordGenerated: false, sessionSecret: Buffer.alloc(32, 3),
    secureCookies: false, sessionTtlMs: 60_000, uiDir: path.resolve('web'), stateFile: path.join(dir, 'state.json'),
    trustProxy: false,
  };
  const player = new StubPlayer(new PlayerService(client).snapshot());
  const ctx: WebContext = {
    client, config, auth: new WebAuth(config.adminPassword, config.sessionSecret, config.sessionTtlMs, false),
    store, player, log: new EventLog(),
  };
  const server = createServer(createControlHandler(ctx));
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => resolve());
  });
  const address = server.address();
  const port = typeof address === 'object' && address ? address.port : 0;
  const base = `http://127.0.0.1:${port}`;
  const jsonHeaders = { 'content-type': 'application/json', 'x-requested-with': 'muz-web' };
  try {
    assert.equal((await fetch(`${base}/api/state`)).status, 401);
    assert.equal((await fetch(`${base}/api/state`, {
      headers: { authorization: 'Bearer old-device-token' },
    })).status, 401);
    const request = { ...target, action: 'play', query: 'Кино' };
    assert.equal((await fetch(`${base}/api/player`, {
      method: 'POST', headers: jsonHeaders, body: JSON.stringify(request),
    })).status, 401);
    assert.equal((await fetch(`${base}/api/player`, {
      method: 'POST', headers: { ...jsonHeaders, authorization: 'Bearer old-device-token' },
      body: JSON.stringify(request),
    })).status, 401);

    const noCsrf = await fetch(`${base}/api/login`, {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ password: 'secret' }),
    });
    assert.equal(noCsrf.status, 403);
    const login = await fetch(`${base}/api/login`, {
      method: 'POST', headers: jsonHeaders, body: JSON.stringify({ password: 'secret' }),
    });
    assert.equal(login.status, 200);
    const cookie = (login.headers.getSetCookie?.()[0] ?? login.headers.get('set-cookie') ?? '').split(';')[0] ?? '';
    assert.match(cookie, /^muz_session=/u);
    const headers = { ...jsonHeaders, cookie };

    const state = await fetch(`${base}/api/state`, { headers: { cookie } });
    assert.equal(state.status, 200);
    const stateBody = await state.json() as StateView;
    assert.equal(stateBody.guilds[0]?.id, guildId);
    assert.equal(stateBody.guilds[0]?.channels[0]?.id, channelId);
    assert.deepEqual(Object.keys(stateBody).sort(), ['bot', 'guilds', 'lastTarget', 'log', 'radioStations', 'sampledAt']);

    const page = await fetch(`${base}/`);
    assert.equal(page.status, 200);
    assert.match(await page.text(), /MUZ/u);

    assert.equal((await fetch(`${base}/api/player`, {
      method: 'POST', headers: { cookie, 'content-type': 'application/json' }, body: JSON.stringify(request),
    })).status, 403);
    assert.equal(player.calls.length, 0);

    const play = await fetch(`${base}/api/player`, { method: 'POST', headers, body: JSON.stringify(request) });
    assert.equal(play.status, 200);
    assert.equal((await play.json() as { track: { title: string } }).track.title, 'Кино');
    assert.deepEqual(player.calls.pop(), {
      action: 'play', target, query: 'Кино', requestedBy: 'Сайт', options: { source: 'youtube', move: true },
    });
    assert.deepEqual(store.last(), target);

    const soundcloud = await fetch(`${base}/api/player`, {
      method: 'POST', headers, body: JSON.stringify({ ...request, source: 'soundcloud', move: false }),
    });
    assert.equal(soundcloud.status, 200);
    assert.deepEqual(player.calls.pop()?.options, { source: 'soundcloud', move: false });

    const radio = await fetch(`${base}/api/player`, {
      method: 'POST', headers, body: JSON.stringify({ ...target, action: 'radio', stationId: 'retro-fm' }),
    });
    assert.equal(radio.status, 200);
    assert.deepEqual(player.calls.pop(), {
      action: 'radio', target, query: 'retro-fm', requestedBy: 'Сайт', options: { move: true },
    });

    for (const action of ['pause', 'resume', 'skip', 'stop', 'volume', 'autoplay', 'loop', 'clear']) {
      const value = action === 'volume' ? 70 : ['autoplay', 'loop'].includes(action) ? true : undefined;
      const response = await fetch(`${base}/api/player`, {
        method: 'POST', headers,
        body: JSON.stringify({ ...target, action, volume: value, value }),
      });
      assert.equal(response.status, 200, action);
      assert.deepEqual(player.calls.pop(), { action, target, value });
    }
    const reloaded = new WebStateStore(config.stateFile);
    await reloaded.load();
    assert.deepEqual(reloaded.last(), target);

    const invalid = await fetch(`${base}/api/player`, {
      method: 'POST', headers, body: JSON.stringify({ ...target, action: 'unknown' }),
    });
    assert.equal(invalid.status, 400);
    assert.equal(player.calls.length, 0);

    const removedRoutes = [
      ['POST', '/api/speakers/code'],
      ['PATCH', '/api/speakers/00000000-0000-0000-0000-000000000000'],
      ['DELETE', '/api/speakers/00000000-0000-0000-0000-000000000000'],
      ['POST', '/api/alice/pair'],
      ['GET', '/api/alice/state'],
      ['POST', '/api/alice/heartbeat'],
      ['POST', '/api/alice/command'],
      ['POST', '/alice/webhook'],
      ['POST', '/alice/webhook/old-secret'],
    ] as const;
    for (const authenticated of [false, true]) {
      for (const [method, pathname] of removedRoutes) {
        const removed = await fetch(`${base}${pathname}`, {
          method, headers: authenticated ? headers : jsonHeaders,
          body: method === 'GET' ? undefined : JSON.stringify({ ...target, code: '123456', text: 'включи Кино' }),
        });
        assert.equal(removed.status, 404, `${method} ${pathname}`);
      }
    }
    assert.equal(player.calls.length, 0);

    const logout = await fetch(`${base}/api/logout`, { method: 'POST', headers });
    assert.equal(logout.status, 200);
    assert.match(logout.headers.get('set-cookie') ?? '', /Max-Age=0/u);
    const session = await fetch(`${base}/api/session`);
    assert.deepEqual(await session.json(), { authenticated: false });
  } finally {
    await new Promise<void>((resolve) => {
      server.close(() => resolve());
      server.closeAllConnections();
    });
    await rm(dir, { recursive: true, force: true });
  }
});
