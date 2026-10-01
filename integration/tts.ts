import 'dotenv/config';
import assert from 'node:assert/strict';
import { PassThrough } from 'node:stream';
import { AudioPlayerStatus, StreamType, VoiceConnectionStatus, type AudioPlayer, type VoiceConnection } from '@discordjs/voice';
import OpusScript from 'opusscript';
import { LocalTts, type Confirmation } from '../src/voice/tts.js';
import phrases from '../src/voice/confirmations.json' with { type: 'json' };
import { GuildQueue } from '../src/utils/GuildQueue.js';
import { toTrack } from '../src/utils/ytdlp.js';
import type { MusicClient } from '../src/types.js';

const waitFor = async (check: () => boolean): Promise<void> => {
  const deadline = Date.now() + 3000;
  while (!check()) {
    assert.ok(Date.now() < deadline, 'audio state did not settle');
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
};
const tts = new LocalTts();
assert.equal(await tts.load(), true, 'Run npm run setup:tts first');
const client = { queues: new Map(), ytdlp: { related: async () => [] } } as unknown as MusicClient;
const raw = new PassThrough();
const queue = new GuildQueue('tts-smoke', client, () => ({ stream: raw, type: StreamType.Raw, destroy: () => raw.destroy() }));
const decoder = new OpusScript(48000, 2, OpusScript.Application.AUDIO);
let subscription: { unsubscribe(): void } | undefined;
let audiblePackets = 0;
const connection = {
  state: { status: VoiceConnectionStatus.Ready },
  subscribe: (player: AudioPlayer) => {
    subscription?.unsubscribe();
    subscription = Reflect.apply(Reflect.get(player, 'subscribe'), player, [connection]) as { unsubscribe(): void };
    return subscription;
  },
  onSubscriptionRemoved: () => {}, setSpeaking: () => {},
  prepareAudioPacket: (packet: Buffer) => {
    const pcm = decoder.decode(packet);
    for (let i = 0; i < pcm.length; i += 2) if (Math.abs(pcm.readInt16LE(i)) > 80) { audiblePackets++; break; }
  },
  dispatchAudio: () => true,
  destroy: () => { subscription?.unsubscribe(); connection.state.status = VoiceConnectionStatus.Destroyed; },
} as unknown as VoiceConnection;
queue.connection = connection; connection.subscribe(queue.player);
const state = () => queue.player.state;
const resource = () => {
  const current = state();
  return current.status === AudioPlayerStatus.Idle ? null : current.resource;
};
try {
  for (const key of Object.keys(phrases) as Confirmation[]) {
    const audio = tts.audio(key)!;
    const before = Buffer.from(audio);
    const packetsBefore = audiblePackets;
    await queue.playVoiceAudio(audio, new AbortController().signal);
    assert.ok(audiblePackets > packetsBefore + 10, 'confirmation must produce audible Opus');
    assert.deepEqual(audio, before, 'cached PCM must remain reusable');
    assert.equal(state().status, AudioPlayerStatus.Idle);
    console.log(`Piper ${key}: PASS (${(audio.length / 192000).toFixed(2)}s)`);
  }
  await queue.addTrack(toTrack({ id: 'aaaaaaaaaaa', title: 'Music resource' }, 'Tester'));
  // A new song is usually still buffering when its confirmation starts.
  const starting = queue.playVoiceAudio(tts.audio('play')!, new AbortController().signal);
  raw.write(Buffer.alloc(48000 * 4 * 30));
  await starting;
  await waitFor(() => queue.player.state.status === AudioPlayerStatus.Playing);
  const musicResource = resource();
  assert.ok(musicResource);
  await queue.playVoiceAudio(tts.audio('queued')!, new AbortController().signal);
  assert.equal(state().status, AudioPlayerStatus.Playing);
  assert.equal(resource(), musicResource);
  assert.equal(queue.pause(), true);
  await queue.playVoiceAudio(tts.audio('pause')!, new AbortController().signal);
  assert.equal(state().status, AudioPlayerStatus.Paused);
  assert.equal(queue.resume(), true);
  await queue.playVoiceAudio(tts.audio('resume')!, new AbortController().signal);
  assert.equal(state().status, AudioPlayerStatus.Playing);
  assert.equal(resource(), musicResource);
  const controller = new AbortController();
  const aborted = queue.playVoiceAudio(tts.audio('play')!, controller.signal);
  const rejected = assert.rejects(aborted);
  controller.abort(); await rejected;
  assert.equal(state().status, AudioPlayerStatus.Playing);
  assert.equal(resource(), musicResource);
  const stopping = queue.playVoiceAudio(tts.audio('stop')!, new AbortController().signal);
  await queue.stop(); await stopping;
  assert.equal(queue.closed, true);
  assert.equal(queue.connection, null);
  console.log('Buffering/playing preservation, paused/resumed state, cancellation and stop: PASS');
} finally { await queue.stop(); decoder.delete(); }
