import assert from 'node:assert/strict';
import test, { type TestContext } from 'node:test';
import { EventEmitter } from 'node:events';
import { readFileSync } from 'node:fs';
import { PassThrough } from 'node:stream';
import { gunzipSync } from 'node:zlib';
import { VoiceConnectionStatus, type VoiceConnection } from '@discordjs/voice';
import type { VoiceBasedChannel } from 'discord.js';
import OpusScript from 'opusscript';
import type { MusicClient } from '../src/types.js';
import { GuildQueue } from '../src/utils/GuildQueue.js';
import { GuildVoice } from '../src/voice/GuildVoice.js';
import { VoiceDecoder } from '../src/voice/decoder.js';
import type { VoiceRuntime } from '../src/voice/runtime.js';

const settle = async (): Promise<void> => { await new Promise<void>(resolve => setImmediate(resolve)); };
const confirmation = gunzipSync(readFileSync('assets/tts/play.pcm.gz'), { maxOutputLength: 48_000 * 4 * 12 });

/** A varying, nonperiodic signal: a fixed tone alone makes delay matching unrealistically easy. */
function music(sample: number): number {
  const t = sample / 48_000;
  return Math.round(3900 * Math.sin(2 * Math.PI * (173 * t + 18 * t * t))
    + 2400 * Math.sin(2 * Math.PI * (421 * t + 7 * Math.sin(t * 5)))
    + 1700 * Math.sin(2 * Math.PI * (937 * t + 31 * t * t)));
}

function foreground(sample: number): number {
  const t = sample / 16_000;
  const envelope = 0.65 + 0.35 * Math.sin(t * 2 * Math.PI * 7);
  return Math.round(envelope * (5600 * Math.sin(2 * Math.PI * (283 * t + 61 * t * t))
    + 3200 * Math.sin(2 * Math.PI * (1231 * t + 13 * Math.sin(t * 3)))));
}

function cue(sample: number): number {
  const ms = sample / 48;
  const start = ms < 120 ? 0 : 160;
  const length = start === 0 ? 120 : 160;
  if (ms < start || ms >= start + length) return 0;
  const position = sample - start * 48;
  const envelope = Math.min(1, position / 480, (length * 48 - 1 - position) / 480);
  return Math.round(6000 * envelope * Math.sin(2 * Math.PI * (start === 0 ? 660 : 990) * position / 48_000));
}

function receiverHarness(context: TestContext) {
  let now = 10_000;
  context.mock.method(performance, 'now', () => now);
  const streams = new Map<string, PassThrough>();
  const speaking = Object.assign(new EventEmitter(), { users: new Map<string, number>() });
  const states = new Map([['bot', { channelId: 'voice' }], ['listenerA', { channelId: 'voice' }]]);
  const connection = {
    state: { status: VoiceConnectionStatus.Ready }, joinConfig: {},
    receiver: { speaking, subscribe: (id: string) => {
      const stream = new PassThrough({ objectMode: true });
      streams.set(id, stream);
      return stream;
    } },
    rejoin: () => true,
    destroy: () => { connection.state.status = VoiceConnectionStatus.Destroyed; },
  } as unknown as VoiceConnection;
  const client = {
    queues: new Map(), user: { id: 'bot', username: 'Бот' },
    guilds: { cache: new Map([['echo-receive', { voiceStates: { cache: states },
      members: { me: { displayName: 'Бот' }, cache: new Map([['listenerA', { user: { bot: false } }]]) } }]]) },
    users: { cache: new Map([['listenerA', { bot: false, username: 'Listener A' }]]) },
    ytdlp: { related: async () => [] },
  } as unknown as MusicClient;
  const recognized: { command: boolean; pcm: Buffer }[] = [];
  let wakeChecks = 0;
  const runtime = Object.assign(new EventEmitter(), {
    ready: true,
    supportsWake: () => true,
    detectWake: async () => { wakeChecks++; return { wake: true, probability: 0.999 }; },
    transcribe: async (pcm: Buffer, _signal: AbortSignal, command: boolean) => {
      recognized.push({ command, pcm: Buffer.from(pcm) });
      return command ? 'Громкость семьдесят процентов' : 'Бот';
    },
    classify: async () => ({ action: 'volume_set', level: 70, confidence: 1 }),
  }) as unknown as VoiceRuntime;
  const queue = new GuildQueue('echo-receive', client);
  queue.voiceChannel = { id: 'voice' } as VoiceBasedChannel;
  queue.connection = connection;
  queue.playVoiceCue = async () => {};
  const voice = new GuildVoice(queue, runtime); queue.voice = voice;
  voice.attach(connection);

  // The outgoing stream goes through the actual 48 kHz stereo Opus encoder.
  // Then the simulated loudspeaker signal is delayed and encoded AGAIN by
  // the participant's microphone. This includes both codec delay and loss.
  const outgoing = new OpusScript(48_000, 2, OpusScript.Application.AUDIO);
  const referenceDecoder = new VoiceDecoder();
  const microphone = new OpusScript(16_000, 1, OpusScript.Application.VOIP);
  const played: number[] = [];
  let frame = 0;
  const open = (): PassThrough => {
    speaking.users.set('listenerA', Date.now());
    speaking.emit('start', 'listenerA');
    const stream = streams.get('listenerA');
    assert.ok(stream && !stream.destroyed, 'the eligible human must have a receiver');
    return stream;
  };
  const end = async (): Promise<void> => {
    streams.get('listenerA')?.end();
    speaking.users.delete('listenerA'); speaking.emit('end', 'listenerA');
    await settle();
  };
  const send = async (count: number, options: { output?: boolean; echoGain?: number; human?: boolean; roomLevel?: number;
    delaySamples?: number; sound?: 'music' | 'cue' | 'tts' } = {}): Promise<void> => {
    const { output = true, echoGain = 0.35, human = false, delaySamples = 2192, sound = 'music' } = options;
    for (let tick = 0; tick < count; tick++, frame++) {
      now += 20;
      const stereo = Buffer.alloc(3840);
      if (output && sound === 'tts') {
        const start = tick * stereo.length;
        if (start < confirmation.length) confirmation.copy(stereo, 0, start, Math.min(confirmation.length, start + stereo.length));
      } else if (output) for (let i = 0; i < 960; i++) {
        const value = sound === 'cue' ? cue(tick * 960 + i) : music(frame * 960 + i);
        stereo.writeInt16LE(value, i * 4); stereo.writeInt16LE(value, i * 4 + 2);
      }
      const packet = Buffer.from(outgoing.encode(stereo, 960));
      const reference = referenceDecoder.decode(packet)!;
      voice.observePlayback(reference);
      for (let i = 0; i < 320; i++) played.push(reference.readInt16LE(i * 2));
      const mic = Buffer.alloc(640);
      for (let i = 0; i < 320; i++) {
        const sample = frame * 320 + i;
        const reflected = (played[sample - delaySamples] ?? 0) * echoGain;
        const room = (options.roomLevel ?? 0) * Math.sin(sample * 2 * Math.PI * 347 / 16_000);
        const value = Math.round(reflected + room + (human ? foreground(sample) : 0));
        mic.writeInt16LE(Math.max(-32768, Math.min(32767, value)), i * 2);
      }
      const stream = streams.get('listenerA');
      if (stream && !stream.destroyed && !stream.writableEnded) stream.write(Buffer.from(microphone.encode(mic, 320)));
      else microphone.encode(mic, 320); // Keep the microphone encoder's history continuous.
      await settle();
    }
  };
  return { queue, voice, recognized, open, end, send,
    wakeChecks: () => wakeChecks,
    now: () => now,
    async close() {
      outgoing.delete(); microphone.delete(); referenceDecoder.delete();
      await queue.stop();
    },
  };
}

test('cold microphone replay of the bot music never reaches wake detection or transcription', async context => {
  const h = receiverHarness(context);
  try {
    h.open();
    await h.send(90); // A 137 ms acoustic delay, followed by a second Opus encoding.
    await h.send(60, { output: false });
    await h.end();
    assert.equal(h.wakeChecks(), 0, 'even the initial echo fragment must not reach the wake model');
    assert.deepEqual(h.recognized, [], 'the bot must not transcribe its own delayed playback');
    assert.equal(h.voice.session.phase, 'idle');
    assert.equal(h.queue.volume, 0.5);
  } finally { await h.close(); }
});

for (const [echoGain, humanFrames] of [[0, 25], [0.35, 25], [0.35, 8]] as const)
test(`${humanFrames * 20} ms human commands remain usable during playback with microphone echo gain ${echoGain}`, async context => {
  const h = receiverHarness(context);
  try {
    h.open();
    await h.send(35, { echoGain });
    assert.equal(h.recognized.length, 0);
    await h.send(humanFrames, { echoGain, human: true });
    await h.send(30, { echoGain });
    assert.equal(h.voice.session.awaits('listenerA'), true, 'human wake speech must survive ongoing playback');
    await h.send(humanFrames, { echoGain, human: true });
    await h.send(50, { echoGain });
    await h.end();
    assert.deepEqual(h.recognized.map(turn => turn.command), [false, true]);
    assert.equal(h.queue.volume, 0.7, 'the recognized human command must execute');
    assert.equal(h.voice.session.phase, 'idle');
    for (const turn of h.recognized) {
      let energy = 0;
      for (let i = 0; i < turn.pcm.length; i += 2) energy += turn.pcm.readInt16LE(i) ** 2;
      assert.ok(Math.sqrt(energy / (turn.pcm.length / 2)) > 500, 'foreground speech must retain useful energy');
    }
  } finally { await h.close(); }
});

for (const backgroundFrames of [45, 230]) test(`a wake finishes over music after ${backgroundFrames * 20} ms of steady microphone noise`, async context => {
  const h = receiverHarness(context);
  const room = { roomLevel: 2850, echoGain: 0.2 };
  try {
    h.open();
    await h.send(backgroundFrames, room);
    assert.equal(h.wakeChecks(), 0);
    await h.send(20, { ...room, human: true });
    await h.send(45, room);
    assert.ok(h.voice.session.awaits('listenerA'), 'wake must complete while the mic and player are still streaming');
    assert.equal(h.recognized.length, 1);
    assert.ok(h.recognized[0]!.pcm.length <= 1200 * 32);
  } finally { await h.close(); }
});

for (const sound of ['music', 'cue', 'tts'] as const) test(`${sound} confirmation echo cannot consume the awakened speaker command window`, async context => {
  const h = receiverHarness(context);
  let finishCue!: () => void;
  h.queue.playVoiceCue = () => {
    Reflect.set(h.queue, 'speakingVoiceAudioUntil', h.now() + 1000);
    return new Promise<void>(resolve => { finishCue = resolve; });
  };
  try {
    h.open();
    await h.send(25, { output: false, echoGain: 0, human: true });
    await h.send(30, { output: false, echoGain: 0 });
    assert.equal(h.voice.session.phase, 'signalling');
    await h.end();
    h.open();
    // Include one actual bundled spoken confirmation, not only synthetic tones.
    await h.send(sound === 'tts' ? Math.ceil(confirmation.length / 3840) : 16, { sound });
    finishCue(); await settle();
    await h.send(50, { output: false });
    await h.end();
    assert.deepEqual(h.recognized.map(turn => turn.command), [false], 'cue echo is not a command');
    assert.equal(h.voice.session.awaits('listenerA'), true, 'the original wake remains valid after cue echo');
    h.open();
    await h.send(25, { output: false, echoGain: 0, human: true });
    await h.send(50, { output: false, echoGain: 0 });
    await h.end();
    assert.deepEqual(h.recognized.map(turn => turn.command), [false, true]);
    assert.equal(h.queue.volume, 0.7);
  } finally { finishCue?.(); await h.close(); }
});
