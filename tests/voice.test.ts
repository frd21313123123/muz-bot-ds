import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtemp, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { contextualCommand, isWakePhrase, parseVolume, validateIntent, validateWakeName, type VoiceIntent } from '../src/voice/intents.js';
import { VoiceSettings } from '../src/voice/settings.js';
import { VoiceSession, type VoiceBackend, type VoiceDiagnostic, type VoiceHost } from '../src/voice/session.js';

const audio = Buffer.from([0, 0]);
function harness(override: Partial<VoiceBackend> = {}, hostOverride: Partial<VoiceHost> = {}) {
  let text = 'Муза';
  let present = true;
  let name = 'Муза';
  let ducked = false;
  let cues = 0;
  const actions: VoiceIntent[] = [];
  const backend: VoiceBackend = {
    transcribe: async () => text,
    classify: async () => ({ action: 'skip', confidence: 0.99 }), ...override,
  };
  const host: VoiceHost = {
    wakeName: () => name, present: () => present,
    cue: async () => { cues++; }, duck: (value) => { ducked = value; },
    execute: async (intent) => { actions.push(intent); }, ...hostOverride,
  };
  const session = new VoiceSession(backend, host);
  return { session, actions, setText: (value: string) => { text = value; },
    setPresent: (value: boolean) => { present = value; }, setName: (value: string) => { name = value; },
    get ducked() { return ducked; }, get cues() { return cues; } };
}

test('wake name is exact after Unicode normalization; mentions in conversation do not wake', () => {
  assert.equal(isWakePhrase('  МУЗА! ', 'Муза'), true);
  assert.equal(isWakePhrase('Музочка', 'Муза'), false);
  assert.equal(isWakePhrase('Муза следующий', 'Муза'), false);
  assert.equal(isWakePhrase('Я говорил с Музой', 'Муза'), false);
  assert.equal(isWakePhrase('Бот енот', 'Бот Ёнот'), true);
  assert.equal(isWakePhrase('!!!', '!!!'), false);
  assert.throws(() => validateWakeName('🔊'));
});

test('volume numbers have a closed range and ambiguous values are rejected', () => {
  for (const [text, level] of [['Громкость 1', 1], ['Громкость 150 процентов', 150],
    ['Громкость пятьдесят пять', 55], ['Громкость сто двадцать три', 123]] as const) {
    assert.equal(parseVolume(text), level);
  }
  for (const text of ['Громкость 0', 'Громкость 151', 'Громкость 10.5', 'Громкость -10',
    'Громкость 50 или 60', 'Громкость пять пять', 'Громкость 100 50', 'Громкость двадцать десять', 'Громкость']) assert.equal(parseVolume(text), null);
  assert.deepEqual(validateIntent('Громкость сто пятьдесят', { action: 'volume_set', confidence: 0.99 }), { action: 'volume_set', level: 150 });
});

test('music requests, compound commands, negations and low confidence cannot control playback', () => {
  for (const text of ['Включи песню', 'Добавь в очередь', 'Поставь Metallica', 'Не останавливай',
    'Следующий трек и пауза', 'Сначала пауза потом продолжить', 'Найди следующий трек', 'Расскажи о паузе', 'Как сделать музыку громче?']) {
    assert.deepEqual(validateIntent(text, { action: 'resume', confidence: 0.999 }), { action: 'unknown' }, text);
  }
  assert.deepEqual(validateIntent('Следующий', { action: 'skip', confidence: 0.4 }), { action: 'unknown' });
  assert.deepEqual(validateIntent('Поставь на паузу', { action: 'pause', confidence: 0.99 }), { action: 'pause' });
  assert.deepEqual(validateIntent('Громче на 50', { action: 'volume_up', confidence: 0.99 }), { action: 'unknown' });
  assert.deepEqual(validateIntent('Громче на пятьдесят', { action: 'volume_up', confidence: 0.99 }), { action: 'unknown' });
  assert.deepEqual(validateIntent('Поставь на паузу', { action: 'resume', confidence: 0.99 }), { action: 'unknown' });
  assert.deepEqual(validateIntent('Приостанови музыку', { action: 'stop', confidence: 0.99 }), { action: 'unknown' });
  assert.deepEqual(validateIntent('Выключи свет', { action: 'stop', confidence: 0.99 }), { action: 'unknown' });
  assert.deepEqual(validateIntent('Стоп', { action: 'stop', confidence: 0.99 }), { action: 'stop' });
});

test('activation ducks only after cue, binds the speaker, executes once and restores gain', async () => {
  const h = harness();
  try {
    const wake = h.session.begin('alice')!;
    assert.equal(wake.silenceMs, 400);
    await wake.complete(audio);
    assert.equal(h.cues, 1); assert.equal(h.ducked, true); assert.equal(h.session.phase, 'awaiting');
    assert.equal(h.session.begin('bob'), null);
    h.setText('Следующий');
    const command = h.session.begin('alice')!;
    assert.equal(command.silenceMs, 800);
    await command.complete(audio); await command.complete(audio);
    assert.deepEqual(h.actions, [{ action: 'skip' }]);
    assert.equal(h.ducked, false); assert.equal(h.session.phase, 'idle');
  } finally { h.session.disable(); }
});

test('generic turn-on requests resume only a paused track, while song requests remain unsupported', async () => {
  let paused = true;
  const h = harness({ classify: async (text) => {
    assert.equal(text, 'Продолжи музыку');
    return { action: 'resume', confidence: 0.99 };
  } }, { paused: () => paused });
  try {
    for (const text of ['Включи', 'Включи музыку', 'Включи обратно', 'Включи музыку снова', 'Продолжай', 'Продолжай музыку', 'Возобнови']) {
      assert.equal(contextualCommand(text, true), 'Продолжи музыку');
      assert.equal(contextualCommand(text, false), text);
    }
    for (const text of ['Включи песню', 'Включи Metallica', 'Включи следующую музыку', 'Не включи', 'Включи и пауза', 'Включи музыку?']) {
      assert.equal(contextualCommand(text, true), text);
    }
    await h.session.begin('alice')!.complete(audio);
    h.setText('Продолжай');
    await h.session.begin('alice')!.complete(audio);
    assert.deepEqual(h.actions, [{ action: 'resume' }]);
    paused = false;
    h.setText('Муза'); await h.session.begin('alice')!.complete(audio);
    h.setText('Включи музыку'); await h.session.begin('alice')!.complete(audio);
    assert.equal(h.actions.length, 1);
    assert.equal(h.ducked, false);
  } finally { h.session.disable(); }
});

test('diagnostics distinguish model rejection and playback no-op without retaining speech', async () => {
  const events: VoiceDiagnostic[] = [];
  const h = harness({ classify: async () => ({ action: 'resume', confidence: 0.99 }) }, {
    paused: () => true, diagnostic: (event) => events.push(event), execute: async () => false,
  });
  try {
    await h.session.begin('alice')!.complete(audio);
    h.setText('Продолжай'); await h.session.begin('alice')!.complete(audio);
    assert.ok(events.some((event) => event.stage === 'recognition' && event.canonicalized));
    assert.ok(events.some((event) => event.stage === 'decision' && event.action === 'resume'));
    assert.ok(events.some((event) => event.stage === 'execution' && event.changed === false));
    assert.equal(JSON.stringify(events).includes('Продолжай'), false);
    assert.equal(JSON.stringify(events).includes('alice'), false);
  } finally { h.session.disable(); }

  const rejectedEvents: VoiceDiagnostic[] = [];
  const rejected = harness({ classify: async () => ({ action: 'private-content', confidence: 0.9 } as unknown as { action: 'skip'; confidence: number }) },
    { diagnostic: (event) => rejectedEvents.push(event) });
  try {
    await rejected.session.begin('alice')!.complete(audio);
    rejected.setText('Следующий'); await rejected.session.begin('alice')!.complete(audio);
    assert.equal(rejected.actions.length, 0);
    assert.ok(rejectedEvents.some((event) => event.stage === 'decision' && event.action === 'unknown'));
    assert.equal(JSON.stringify(rejectedEvents).includes('private-content'), false);
  } finally { rejected.session.disable(); }
});

test('ten second timeout and maximum command duration release the session', async (context) => {
  context.mock.timers.enable({ apis: ['setTimeout'] });
  const h = harness();
  await h.session.begin('alice')!.complete(audio);
  context.mock.timers.tick(9999); assert.equal(h.ducked, true);
  context.mock.timers.tick(1); assert.equal(h.ducked, false);
  await h.session.begin('alice')!.complete(audio);
  const command = h.session.begin('alice')!;
  context.mock.timers.tick(15_000);
  assert.equal(command.signal.aborted, true); assert.equal(h.ducked, false);
  await command.complete(audio); assert.equal(h.actions.length, 0);
  h.session.disable();
});

test('leaving during classification or disabling prevents stale execution', async () => {
  for (const cancel of ['leave', 'disable', 'reset'] as const) {
    let resolve!: (value: { action: 'skip'; confidence: number }) => void;
    const h = harness({ classify: () => new Promise((done) => { resolve = done; }) });
    await h.session.begin('alice')!.complete(audio);
    h.setText('Следующий');
    const processing = h.session.begin('alice')!.complete(audio);
    await Promise.resolve();
    if (cancel === 'leave') { h.setPresent(false); h.session.cancelUser('alice'); }
    else if (cancel === 'disable') h.session.disable(); else h.session.cancel();
    resolve({ action: 'skip', confidence: 0.99 });
    await processing;
    assert.equal(h.actions.length, 0); assert.equal(h.ducked, false);
    h.session.disable();
  }
});

test('model and cue errors restore state; unsupported requests never call classifier', async () => {
  const cueError = harness({}, { cue: async () => { throw new Error('test'); } });
  await cueError.session.begin('alice')!.complete(audio);
  assert.equal(cueError.session.phase, 'idle'); assert.equal(cueError.ducked, false);
  let calls = 0;
  const h = harness({ classify: async () => { calls++; throw new Error('test'); } });
  await h.session.begin('alice')!.complete(audio);
  h.setText('Включи музыку');
  await h.session.begin('alice')!.complete(audio);
  assert.equal(calls, 0); assert.equal(h.ducked, false);
  h.setText('Муза'); await h.session.begin('alice')!.complete(audio);
  h.setText('Следующий'); await h.session.begin('alice')!.complete(audio);
  assert.equal(calls, 1); assert.equal(h.ducked, false);
  h.session.disable(); cueError.session.disable();
});

test('concurrent wake results cannot steal activation; renamed wake key takes effect', async () => {
  const h = harness();
  const alice = h.session.begin('alice')!;
  const bob = h.session.begin('bob')!;
  await Promise.all([alice.complete(audio), bob.complete(audio)]);
  assert.equal(h.cues, 1); assert.equal(bob.signal.aborted, true);
  h.session.cancel(); h.setName('Джев');
  await h.session.begin('alice')!.complete(audio);
  assert.equal(h.session.phase, 'idle');
  h.setText('Джев'); await h.session.begin('alice')!.complete(audio);
  assert.equal(h.cues, 2); h.session.disable();
});

test('custom names persist per guild and reset independently, with serialized writes', async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'muz-voice-settings-'));
  try {
    const file = path.join(directory, 'names.json');
    const settings = new VoiceSettings(file);
    await settings.load();
    await Promise.all([settings.set('a', 'Джев'), settings.set('b', 'Муза')]);
    const loaded = new VoiceSettings(file); await loaded.load();
    assert.equal(loaded.get('a'), 'Джев'); assert.equal(loaded.get('b'), 'Муза');
    await loaded.set('a', null);
    const reset = new VoiceSettings(file); await reset.load();
    assert.equal(reset.get('a'), undefined); assert.equal(reset.get('b'), 'Муза');
    await assert.rejects(reset.set('a', '!!!'));
  } finally { await rm(directory, { recursive: true, force: true }); }
});
