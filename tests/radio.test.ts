import assert from 'node:assert/strict';
import test from 'node:test';
import { PassThrough } from 'node:stream';
import { StreamType } from '@discordjs/voice';
import type { MusicClient } from '../src/types.js';
import { GuildQueue } from '../src/utils/GuildQueue.js';
import { extractRadioRequest, findRadioStation, radioStations, radioSuggestions, radioTrack } from '../src/utils/radio.js';
import { waitForAudio, type ManagedAudioStream } from '../src/utils/stream.js';
import { toTrack } from '../src/utils/ytdlp.js';
import { playerEmbed, nowPlayingEmbed, queueEmbed, playerActionRow } from '../src/utils/embeds.js';
import { contextualCommand } from '../src/voice/intents.js';
import { MessageFlags, PermissionsBitField, type ChatInputCommandInteraction, type VoiceBasedChannel, type AutocompleteInteraction } from 'discord.js';
import { commands } from '../src/commands.js';
import { onInteraction } from '../src/events/interactionCreate.js';

const station = radioStations[0]!;
const song = toTrack({ id: 'aaaaaaaaaaa', title: 'Numb', duration: 180 }, 'Alice');
const nextSong = toTrack({ id: 'bbbbbbbbbbb', title: 'Sonne', duration: 200 }, 'Bob');
const tick = (): Promise<void> => new Promise(resolve => setImmediate(resolve));
async function until(predicate: () => boolean): Promise<void> {
  const end = Date.now() + 2000;
  while (!predicate()) { if (Date.now() >= end) throw new Error('State did not change'); await tick(); }
}
function media(ready = true): ManagedAudioStream & { stream: PassThrough } {
  const stream = new PassThrough(); stream.on('error', () => {});
  if (ready) stream.write(Buffer.alloc(3840 * 3));
  return { stream, type: StreamType.Raw, destroy: () => stream.destroy() };
}
function harness(factory?: (url: string) => ManagedAudioStream, retryDelays: readonly number[] = [0, 0, 0]) {
  const music: ManagedAudioStream[] = [], radio: ManagedAudioStream[] = [];
  const client = { queues: new Map(), ytdlp: { related: async () => { assert.fail('Radio must not request YouTube recommendations'); } } } as unknown as MusicClient;
  const queue = new GuildQueue('radio-test', client, () => {
    const value = media(false); music.push(value); return value;
  }, { factory: url => { const value = factory ? factory(url) : media(); radio.push(value); return value; },
    startupTimeoutMs: 30, retryDelays });
  client.queues.set(queue.guildId, queue);
  return { queue, music, radio };
}

test('every radio name, ID and alias resolves, with anchored Russian commands and Latin station names', () => {
  assert.equal(radioStations.length, 10);
  for (const item of radioStations) {
    for (const alias of [item.id, item.name, ...item.aliases]) {
      assert.equal(findRadioStation(alias)?.id, item.id, alias);
      for (const text of [`Включи ${alias}`, `Поставь радио ${alias}`, `Запусти, ${alias}`, alias]) {
        const request = extractRadioRequest(text, true);
        assert.equal(request?.kind, 'station', text);
        if (request?.kind === 'station') assert.equal(request.station.id, item.id, text);
      }
    }
  }
  assert.equal(extractRadioRequest('Европа Плюс'), null, 'bare names require an explicit command window');
  assert.equal(extractRadioRequest('Вот, включи Ретро FM, пожалуйста')?.kind, 'station');
  assert.equal(extractRadioRequest('включи неизвестную группу'), null);
  assert.equal(extractRadioRequest('Включи радио неизвестное')?.kind, 'unsupported');
  assert.equal(extractRadioRequest('Радио России', true)?.kind, 'unsupported');
  assert.equal(extractRadioRequest('Включи песню Radio Ga Ga'), null);
  assert.equal(extractRadioRequest('Включи песню Европа Плюс'), null);
  assert.equal(extractRadioRequest('Включи Numb и Encore'), null);
  assert.deepEqual(radioSuggestions('ретро'), [{ name: 'Ретро FM', value: 'retro-fm' }]);
  assert.equal(radioSuggestions('').length, 10);
});

test('negated, questioned and compound radio requests cannot become songs or controls', () => {
  for (const text of ['Не включай Европа Плюс', 'Включи Европа Плюс?', 'Как включить Retro FM?',
    'Включи Retro FM и сделай громче', 'Включи радио Европа Плюс или Ретро FM', 'Если включи Авторадио',
    'Поставь радио и затем паузу']) assert.equal(extractRadioRequest(text, true)?.kind, 'rejected', text);
  assert.equal(contextualCommand('Останови радио', false, true), 'Останови музыку');
  assert.equal(contextualCommand('Выключи радио', false, true), 'Останови музыку');
  assert.equal(contextualCommand('Не выключай радио', false, true), 'Не выключай радио');
});

test('audio preflight waits for real bytes, preserves them, and cleans listeners after errors, timeout and abort', async () => {
  const value = media(false);
  const signal = new AbortController();
  const opening = waitForAudio(value, signal.signal, 100);
  value.stream.write(Buffer.from([1, 2, 3, 4]));
  await opening;
  assert.equal(value.stream.readableLength, 4);
  assert.equal(value.stream.listenerCount('readable'), 0);
  value.destroy();
  for (const reason of ['error', 'timeout', 'abort', 'close']) {
    const source = media(false); const controller = new AbortController();
    const result = waitForAudio(source, controller.signal, 10);
    if (reason === 'error') source.stream.destroy(new Error('test failure'));
    if (reason === 'abort') controller.abort();
    if (reason === 'close') source.destroy();
    await assert.rejects(result);
    assert.equal(source.stream.listenerCount('readable'), 0);
    source.destroy();
  }
});

test('radio switches only after first audio, clears queued tracks and modes, preserves volume and shows the live station', async () => {
  const opening = media(false); const h = harness(() => opening);
  try {
    await h.queue.addTracks([song, nextSong]); h.queue.setLoop(true); h.queue.setSpeed(0.8);
    h.queue.setVolume(150); h.queue.setVoiceDucking(true);
    const oldTrack = h.queue.currentTrack;
    const pending = h.queue.playRadio(station, 'Bob');
    assert.equal(h.queue.currentTrack, oldTrack);
    assert.equal(h.queue.tracks.length, 1);
    opening.stream.write(Buffer.alloc(3840 * 3));
    assert.equal(await pending, true);
    assert.equal(h.queue.currentTrack?.source, 'radio');
    assert.equal(h.queue.currentTrack?.requestedBy, 'Bob');
    assert.equal(h.queue.tracks.length, 0);
    assert.equal(h.queue.autoplay, false); assert.equal(h.queue.loopCurrent, false); assert.equal(h.queue.speed, 1);
    assert.equal(h.queue.volume, 1.5); assert.equal(h.music.at(-1)!.stream.destroyed, true);
    assert.throws(() => h.queue.setSpeed(1.2), /эфира/);
    assert.throws(() => h.queue.setAutoplay(true), /радио/);
    assert.throws(() => h.queue.setLoop(true), /радио/);
    assert.match(playerEmbed(h.queue).toJSON().description!, /Прямой эфир/);
    assert.match(nowPlayingEmbed(h.queue.currentTrack!).toJSON().description!, /europaplus.ru/);
    assert.match(JSON.stringify(queueEmbed(h.queue).toJSON()), /Прямой эфир/);
    const buttons = playerActionRow(h.queue).toJSON().components;
    assert.equal(buttons[3]!.disabled, true); assert.equal(buttons[4]!.disabled, true);
  } finally { await h.queue.stop(); }
});

test('failed station startup preserves the original track, queue and modes', async () => {
  const h = harness(() => media(false));
  try {
    await h.queue.addTracks([song, nextSong]); h.queue.setAutoplay(true);
    const current = h.queue.currentTrack;
    await assert.rejects(h.queue.playRadio(station, 'Bob'), /недоступен/);
    assert.equal(h.queue.currentTrack, current); assert.deepEqual(h.queue.tracks, [nextSong]);
    assert.equal(h.queue.autoplay, true); assert.equal(h.music[0]!.stream.destroyed, false);
    assert.equal(h.radio[0]!.stream.destroyed, true);
  } finally { await h.queue.stop(); }
});

test('new radio requests supersede older ones; user departure and stop cancel only eligible pending work', async () => {
  const streams: ReturnType<typeof media>[] = [];
  const h = harness(() => { const value = media(false); streams.push(value); return value; });
  try {
    await h.queue.addTrack(song);
    const a = h.queue.playRadio(station, 'Alice', { userId: 'alice' });
    const b = h.queue.playRadio(radioStations[1]!, 'Bob', { userId: 'bob' });
    assert.equal(await a, false); assert.equal(streams[0]!.stream.destroyed, true);
    h.queue.cancelRadioRequest('alice');
    streams[1]!.stream.write(Buffer.alloc(3840));
    assert.equal(await b, true);
    const c = h.queue.playRadio(station, 'Alice', { userId: 'alice' });
    h.queue.cancelRadioRequest('alice'); assert.equal(await c, false);
    const d = h.queue.playRadio(station, 'Bob');
    await h.queue.stop(); assert.equal(await d, false);
    assert.ok(streams.every(value => value.stream.destroyed));
  } finally { await h.queue.stop(); }
});

test('aborted voice requests and changed presence never clear the queue or switch playback', async () => {
  for (const mode of ['abort', 'presence']) {
    const opening = media(false); const h = harness(() => opening);
    const controller = new AbortController(); let present = true;
    try {
      await h.queue.addTracks([song, nextSong]);
      const pending = h.queue.playRadio(station, 'Alice', { signal: controller.signal, valid: () => present });
      if (mode === 'abort') controller.abort(); else present = false;
      opening.stream.write(Buffer.alloc(3840));
      assert.equal(await pending, false); assert.equal(h.queue.currentTrack, song);
      assert.deepEqual(h.queue.tracks, [nextSong]); assert.equal(opening.stream.destroyed, true);
    } finally { await h.queue.stop(); }
  }
});

test('pause destroys the live stream; resume opens a fresh broadcast without seeking', async () => {
  const h = harness();
  try {
    await h.queue.playRadio(station, 'Alice');
    assert.equal(h.queue.pause(), true); assert.equal(h.queue.isPaused, true);
    assert.equal(h.radio[0]!.stream.destroyed, true);
    assert.equal(h.queue.pause(), false);
    assert.equal(h.queue.resume(), true); assert.equal(h.queue.resume(), false);
    await until(() => h.radio.length === 2 && h.queue.currentTrack?.source === 'radio');
    assert.equal(h.queue.isPaused, false);
    assert.equal(h.queue.currentTrack?.source, 'radio');
  } finally { await h.queue.stop(); }
});

test('music replaces radio immediately and skip ends radio without retrying it', async () => {
  for (const mode of ['music', 'skip']) {
    const h = harness();
    try {
      await h.queue.playRadio(station, 'Alice');
      if (mode === 'music') { await h.queue.addTrack(nextSong); assert.equal(h.queue.currentTrack, nextSong); }
      else { assert.equal(h.queue.skip(), true); await tick(); assert.equal(h.queue.currentTrack, null); }
      assert.equal(h.radio[0]!.stream.destroyed, true); assert.equal(h.radio.length, 1);
    } finally { await h.queue.stop(); }
  }
});

test('radio interruptions retry three times and fall back to queued music after exhaustion', async () => {
  let attempts = 0;
  const h = harness(() => { attempts++; if (attempts > 1) throw new Error('offline'); return media(); });
  try {
    await h.queue.playRadio(station, 'Alice');
    h.queue.pending.add(nextSong); // Simulate a pending track while a stream fails.
    h.queue.player.stop(true);
    await until(() => h.queue.currentTrack === nextSong);
    assert.equal(attempts, 4); assert.equal(h.radio[0]!.stream.destroyed, true);
    assert.equal(h.queue.autoplay, false);
  } finally { await h.queue.stop(); }
});

test('successful recovery keeps the live station and stop cancels a delayed reconnect', async () => {
  const h = harness(undefined, [5, 10, 20]);
  try {
    await h.queue.playRadio(station, 'Alice');
    h.queue.player.stop(true);
    await until(() => h.radio.length === 2 && h.queue.currentTrack?.source === 'radio');
    assert.equal(h.queue.isRadio, true);
    h.queue.player.stop(true); await h.queue.stop();
    await new Promise(resolve => setTimeout(resolve, 25));
    assert.equal(h.radio.length, 2);
    assert.ok(h.radio.every(value => value.stream.destroyed));
  } finally { await h.queue.stop(); }
});

test('a stale YouTube recommendation cannot replace a newly selected radio station', async () => {
  const h = harness();
  let finish!: (tracks: typeof song[]) => void;
  h.queue.client.ytdlp.related = () => new Promise(resolve => { finish = resolve; });
  try {
    await h.queue.addTrack(song); h.queue.setAutoplay(true);
    h.queue.player.stop(true); await until(() => Boolean(finish));
    await h.queue.playRadio(station, 'Bob'); finish([nextSong]); await tick();
    assert.equal(h.queue.isRadio, true); assert.equal(h.queue.tracks.length, 0);
  } finally { await h.queue.stop(); }
});

test('/radio and /play station requests use the same queue, ephemeral replies and current voice access', async () => {
  for (const name of ['radio', 'play']) {
    const h = harness();
    const replies: unknown[] = [];
    let target = name === 'radio' ? 'retro-fm' : 'Включи Retro FM';
    const channel = { id: 'voice', permissionsFor: () => new PermissionsBitField([
      PermissionsBitField.Flags.Connect, PermissionsBitField.Flags.Speak,
    ]) } as unknown as VoiceBasedChannel;
    const states = new Map([['alice', { channel }]]);
    const interaction = { guildId: h.queue.guildId, guild: { members: { me: {} }, voiceStates: { cache: states } },
      channelId: 'text', user: { id: 'alice', username: 'Alice' }, options: { getString: () => target },
      deferReply: async (payload: unknown) => { replies.push(payload); },
      editReply: async (payload: unknown) => { replies.push(payload); },
    } as unknown as ChatInputCommandInteraction;
    h.queue.voiceChannel = channel; h.queue.join = async () => {};
    h.queue.client.ytdlp.search = async () => { assert.fail('A station request must bypass YouTube'); };
    const command = commands.find(item => item.data.name === name)!;
    try {
      await command.execute(interaction, h.queue.client);
      assert.deepEqual(replies[0], { flags: MessageFlags.Ephemeral });
      assert.equal(h.queue.currentTrack?.source, 'radio');
      assert.match(JSON.stringify(replies.at(-1)), /Ретро FM/);
      states.set('alice', { channel: { ...channel, id: 'other' } as VoiceBasedChannel });
      target = 'Европа Плюс';
      await command.execute(interaction, h.queue.client);
      assert.match(JSON.stringify(replies.at(-1)), /тот же голосовой канал/);
      assert.equal(h.radio.length, 1);
      if (name === 'radio') {
        target = 'Unknown station'; await command.execute(interaction, h.queue.client);
        assert.match(JSON.stringify(replies.at(-1)), /не поддерживается/);
      }
    } finally { await h.queue.stop(); }
  }
});

test('/radio autocomplete returns only known station IDs and does not start playback', async () => {
  let suggestions: unknown;
  const interaction = { commandName: 'radio', isAutocomplete: () => true,
    options: { getFocused: () => 'retro' }, respond: async (value: unknown) => { suggestions = value; },
  } as unknown as AutocompleteInteraction;
  await onInteraction(interaction, {} as MusicClient);
  assert.deepEqual(suggestions, [{ name: 'Ретро FM', value: 'retro-fm' }]);
  const definition = commands.find(item => item.data.name === 'radio')!.data.toJSON();
  assert.equal(definition.options?.[0]?.type, 3);
  assert.equal(definition.options?.[0]?.autocomplete, true);
});

test('/radio verifies Connect/Speak permissions and rechecks membership after joining', async () => {
  for (const mode of ['permissions', 'depart']) {
    const h = harness(); let reply: unknown;
    const channel = { id: 'voice', permissionsFor: () => new PermissionsBitField(mode === 'permissions' ? [] : [
      PermissionsBitField.Flags.Connect, PermissionsBitField.Flags.Speak,
    ]) } as unknown as VoiceBasedChannel;
    const states = new Map([['alice', { channel }]]);
    const interaction = { guildId: h.queue.guildId, guild: { members: { me: {} }, voiceStates: { cache: states } },
      channelId: 'text', user: { id: 'alice', username: 'Alice' }, options: { getString: () => 'Европа Плюс' },
      deferReply: async () => {}, editReply: async (value: unknown) => { reply = value; },
    } as unknown as ChatInputCommandInteraction;
    h.queue.voiceChannel = channel;
    h.queue.join = async () => { states.clear(); };
    try {
      await commands.find(item => item.data.name === 'radio')!.execute(interaction, h.queue.client);
      assert.equal(h.queue.currentTrack, null); assert.equal(h.radio.length, 0);
      assert.match(JSON.stringify(reply), mode === 'permissions' ? /прав/ : /отменён/);
    } finally { await h.queue.stop(); }
  }
});
