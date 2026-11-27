import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { createServer } from 'node:http';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { ChannelType, type Guild } from 'discord.js';
import type { MusicClient } from '../src/types.js';
import type { RadioStation } from '../src/utils/radio.js';
import { WebAuth } from '../src/web/auth.js';
import { parsePairingCode, runVoiceCommand } from '../src/web/commands.js';
import { handleAliceRequest } from '../src/web/alice.js';
import { EventLog } from '../src/web/log.js';
import { PlayerService, type PlayerApi } from '../src/web/player.js';
import { createControlHandler, type WebContext } from '../src/web/server.js';
import { SpeakerStore } from '../src/web/store.js';
import type { QueueView } from '../src/web/types.js';
import type { WebConfig } from '../src/web/config.js';

class StubPlayer implements PlayerApi {
  calls: string[] = [];
  view: QueueView | null = null;
  async play(_target: { guildId: string; channelId: string }, query: string, requestedBy: string) {
    this.calls.push(`play:${query}`);
    return {
      track: {
        title: query, url: 'https://www.youtube.com/watch?v=abcdefghijk', thumbnail: null,
        duration: '3:00', requestedBy, isAutoplay: false, isLive: false, source: 'youtube' as const,
      },
      started: true, playlist: false,
    };
  }
  async radio(_target: { guildId: string; channelId: string }, station: RadioStation) {
    this.calls.push(`radio:${station.id}`);
    return { station: station.name };
  }
  async act(_target: { guildId: string; channelId: string }, action: string, value?: number | boolean) {
    this.calls.push(`act:${action}:${String(value)}`);
    return { changed: true, level: typeof value === 'number' ? value : undefined };
  }
  queue() { return this.view; }
  snapshot() {
    return {
      bot: { id: 'bot', tag: 'Bot#0001', avatar: null, ready: true, uptimeMs: 1 },
      guilds: [], radioStations: [], sampledAt: Date.now(),
    };
  }
}

const target = { guildId: '123456789012345678', channelId: '223456789012345678' };

test('speaker pairing codes survive exactly one use and one target', () => {
  const store = new SpeakerStore(path.join(os.tmpdir(), `muz-store-${Date.now()}.json`));
  const code = store.createCode(target, 'Кухня', 1_000, 1_000);
  assert.match(code.code, /^\d{6}$/u);
  assert.equal(store.takeCode(code.code, 1_500)?.name, 'Кухня');
  assert.equal(store.takeCode(code.code, 1_500), null);
  const expired = store.createCode(target, 'Кухня', 10, 10);
  assert.equal(store.takeCode(expired.code, 30), null);
});

test('voice requests use the shared control vocabulary', async () => {
  const player = new StubPlayer();
  const play = await runVoiceCommand(player, target, 'включи Кино');
  assert.match(play.text, /Включаю: Кино/u);
  assert.deepEqual(player.calls, ['play:Кино']);

  player.calls = [];
  const skip = await runVoiceCommand(player, target, 'следующий трек');
  assert.match(skip.text, /Следующий/u);
  assert.deepEqual(player.calls, ['act:skip:undefined']);

  player.calls = [];
  const volume = await runVoiceCommand(player, target, 'громкость 70');
  assert.match(volume.text, /70/u);
  assert.deepEqual(player.calls, ['act:volume:70']);

  player.calls = [];
  const radio = await runVoiceCommand(player, target, 'включи радио Ретро FM');
  assert.match(radio.text, /Ретро FM/u);
  assert.deepEqual(player.calls, ['radio:retro-fm']);

  player.calls = [];
  const localBridge = await runVoiceCommand(player, target, 'включи Кино в дискорде');
  assert.match(localBridge.text, /Кино/u);
  assert.deepEqual(player.calls, ['play:Кино']);

  player.calls = [];
  const addressed = await runVoiceCommand(player, target, 'Алиса, включи Кино');
  assert.match(addressed.text, /Кино/u);
  assert.deepEqual(player.calls, ['play:Кино']);
});

test('voice status and exit are explicit', async () => {
  const player = new StubPlayer();
  assert.match((await runVoiceCommand(player, target, 'что играет')).text, /ничего не играет/iu);
  player.view = {
    guildId: target.guildId, channelId: target.channelId, channelName: 'General', playing: true,
    paused: false, radio: false, volume: 60, speed: 1, autoplay: false, loop: false, elapsed: 30,
    current: { title: 'Кино', url: 'https://example.com', thumbnail: null, duration: '3:00',
      requestedBy: 'Алиса', isAutoplay: false, isLive: false, source: 'youtube' },
    upcoming: [], upcomingCount: 0, sampledAt: Date.now(),
  };
  assert.match((await runVoiceCommand(player, target, 'что играет')).text, /Кино/u);
  const exit = await runVoiceCommand(player, target, 'хватит');
  assert.equal(exit.keepSession, false);
});

test('pairing code parser ignores ordinary six digit titles', () => {
  assert.equal(parsePairingCode('подключи колонку 123456'), '123456');
  assert.equal(parsePairingCode('подключи колонку, код 654321!'), '654321');
  assert.equal(parsePairingCode('подключи колонку с кодом 111222'), '111222');
  assert.equal(parsePairingCode('подключи колонку по коду 333444'), '333444');
  assert.equal(parsePairingCode('код 111222'), '111222');
  assert.equal(parsePairingCode('включи 123456'), null);
});

test('Alice skill pairs a device and then accepts commands', async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'muz-alice-'));
  try {
    const store = new SpeakerStore(path.join(dir, 'state.json'));
    const player = new StubPlayer();
    const deps = { store, player, skillId: 'skill-1', skillName: 'Муза', log: () => {} };
    const code = store.createCode(target, 'Кухня');
    const body = (command: string) => ({
      version: '1.0',
      session: {
        message_id: 0, session_id: 'session', skill_id: 'skill-1',
        application: { application_id: 'device-1' },
      },
      request: { command, original_utterance: command, type: 'SimpleUtterance' },
    });
    const paired = await handleAliceRequest(body(`подключи колонку ${code.code}`), deps);
    assert.match(paired.response.text, /подключена/iu);
    assert.equal(store.list().length, 1);
    assert.equal(store.list()[0]?.deviceId, 'device-1');
    const status = await handleAliceRequest(body('что играет'), deps);
    assert.match(status.response.text, /ничего не играет/iu);
    const unpaired = await handleAliceRequest(body('что играет'), {
      ...deps, store: new SpeakerStore(path.join(dir, 'empty.json')),
    });
    assert.match(unpaired.response.text, /не подключена/iu);
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

test('control HTTP API requires login and CSRF, then pairs an Alice device', async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'muz-web-'));
  const store = new SpeakerStore(path.join(dir, 'state.json'));
  const guildId = '123456789012345678';
  const channelId = '223456789012345678';
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
    queues: new Map(),
    ytdlp: {},
  } as unknown as MusicClient;
  const config: WebConfig = {
    host: '127.0.0.1', port: 0, publicUrl: null, adminPassword: 'secret',
    adminPasswordFile: null, passwordGenerated: false, sessionSecret: Buffer.alloc(32, 3),
    secureCookies: false, sessionTtlMs: 60_000, uiDir: path.resolve('web'), stateFile: path.join(dir, 'state.json'),
    aliceSkillId: 'skill-http', aliceSkillName: 'Муза', aliceWebhookSecret: null, trustProxy: false,
  };
  const ctx: WebContext = {
    client, config, auth: new WebAuth(config.adminPassword, config.sessionSecret, config.sessionTtlMs, false),
    store, player: new PlayerService(client), log: new EventLog(), pairAttempts: new Map(),
  };
  const server = createServer(createControlHandler(ctx));
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => resolve());
  });
  const address = server.address();
  const port = typeof address === 'object' && address ? address.port : 0;
  const base = `http://127.0.0.1:${port}`;
  try {
    const unauthorized = await fetch(`${base}/api/state`);
    assert.equal(unauthorized.status, 401);

    const noCsrf = await fetch(`${base}/api/login`, {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ password: 'secret' }),
    });
    assert.equal(noCsrf.status, 403);

    const login = await fetch(`${base}/api/login`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-requested-with': 'muz-web' },
      body: JSON.stringify({ password: 'secret' }),
    });
    assert.equal(login.status, 200);
    const cookie = (login.headers.getSetCookie?.()[0] ?? login.headers.get('set-cookie') ?? '').split(';')[0] ?? '';
    assert.ok(cookie);
    assert.match(cookie, /^muz_session=/u);

    const state = await fetch(`${base}/api/state`, { headers: { cookie } });
    assert.equal(state.status, 200);
    const stateBody = await state.json() as { guilds: { id: string; channels: { id: string }[] }[] };
    assert.equal(stateBody.guilds[0]?.id, guildId);

    const page = await fetch(`${base}/`);
    assert.equal(page.status, 200);
    assert.match(await page.text(), /MUZ/u);

    const codeResponse = await fetch(`${base}/api/speakers/code`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-requested-with': 'muz-web', cookie },
      body: JSON.stringify({ guildId, channelId, name: 'Кухня' }),
    });
    assert.equal(codeResponse.status, 200);
    const codeBody = await codeResponse.json() as { code: string };
    assert.match(codeBody.code, /^\d{6}$/u);

    const aliceBody = (command: string) => JSON.stringify({
      version: '1.0',
      session: {
        message_id: 0, session_id: 'session', skill_id: 'skill-http',
        application: { application_id: 'device-http' },
      },
      request: { command, original_utterance: command, type: 'SimpleUtterance' },
    });
    const pair = await fetch(`${base}/alice/webhook`, {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: aliceBody(`подключи колонку ${codeBody.code}`),
    });
    assert.equal(pair.status, 200);
    const pairBody = await pair.json() as { response: { text: string } };
    assert.match(pairBody.response.text, /подключена/iu);

    const status = await fetch(`${base}/alice/webhook`, {
      method: 'POST', headers: { 'content-type': 'application/json' }, body: aliceBody('что играет'),
    });
    assert.equal(status.status, 200);
    const statusBody = await status.json() as { response: { text: string } };
    assert.match(statusBody.response.text, /ничего не играет/iu);

    const bridgeCodeResponse = await fetch(`${base}/api/speakers/code`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-requested-with': 'muz-web', cookie },
      body: JSON.stringify({ guildId, channelId, name: 'Мост' }),
    });
    const bridgeCode = (await bridgeCodeResponse.json() as { code: string }).code;
    const bridgePair = await fetch(`${base}/api/alice/pair`, {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ code: bridgeCode, name: 'Мост' }),
    });
    assert.equal(bridgePair.status, 200);
    const bridgeToken = (await bridgePair.json() as { token: string }).token;
    assert.ok(bridgeToken);

    const bridgeState = await fetch(`${base}/api/alice/state`, {
      headers: { authorization: `Bearer ${bridgeToken}` },
    });
    assert.equal(bridgeState.status, 200);
    const bridgeStateBody = await bridgeState.json() as { speaker: { kind: string; name: string } };
    assert.equal(bridgeStateBody.speaker.kind, 'bridge');
    assert.equal(bridgeStateBody.speaker.name, 'Мост');

    const bridgeCommand = await fetch(`${base}/api/alice/command`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', authorization: `Bearer ${bridgeToken}` },
      body: JSON.stringify({ text: 'что играет' }),
    });
    assert.equal(bridgeCommand.status, 200);
    assert.match((await bridgeCommand.json() as { text: string }).text, /ничего не играет/iu);

    const anonymousBridge = await fetch(`${base}/api/alice/state`);
    assert.equal(anonymousBridge.status, 401);
  } finally {
    await new Promise<void>((resolve) => {
      server.close(() => resolve());
      server.closeAllConnections();
    });
    await rm(dir, { recursive: true, force: true });
  }
});
