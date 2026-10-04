import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtemp, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { contextualCommand, isWakePhrase, modeCommand, parseVolume, ruleIntent, validateIntent, validateWakeName, type VoiceIntent } from '../src/voice/intents.js';
import { extractMusicRequest } from '../src/voice/music.js';
import { VoiceSettings } from '../src/voice/settings.js';
import { VoiceSession, type VoiceBackend, type VoiceDiagnostic, type VoiceHost } from '../src/voice/session.js';
import type { TrainingExample } from '../src/voice/training.js';

const audio = Buffer.from([0, 0]);
test('light command rules cover controls and reject titles, negations and ambiguous requests', () => {
  const cases: [string, VoiceIntent][] = [
    ['next track', { action: 'skip' }], ['Вот, следующее пожалуйста', { action: 'skip' }],
    ['Пропусти текущий трек', { action: 'skip' }], ['Поставь на паузу', { action: 'pause' }],
    ['resume', { action: 'resume' }], ['Продолжи музыку', { action: 'resume' }],
    ['Останови музыку', { action: 'stop' }], ['Выйди из канала', { action: 'stop' }],
    ['Сделай громче', { action: 'volume_up' }], ['Уменьши громкость', { action: 'volume_down' }],
    ['Громкость сто пятьдесят', { action: 'volume_set', level: 150 }],
    ['Установи громкость 50 процентов', { action: 'volume_set', level: 50 }],
    ['Включи повтор трека', { action: 'loop_on' }], ['Очисти очередь', { action: 'queue_clear' }],
    ['Выключи голосовое управление', { action: 'voice_off' }],
  ];
  for (const [phrase, intent] of cases) assert.deepEqual(ruleIntent(phrase), intent, phrase);
  for (const phrase of ['Не ставь на паузу', 'Stop?', 'Сделай громче и пропусти трек',
    'Пропусти трек завтра', 'Включи песню Stop', 'Radio Ga Ga', 'Включи Европа Плюс',
    'Громкость -10', 'Громкость 10.5', 'Громкость 0', 'Громкость 151', 'Громкость 50 60',
    'Сделай громче на 20', 'Громкость пять пять', 'Останови вентилятор']) {
    assert.deepEqual(ruleIntent(phrase), { action: 'unknown' }, phrase);
  }
});
const modeCases: [string, VoiceIntent['action']][] = [
  ['Включи бесконечный режим', 'autoplay_on'], ['Выключи бесконечный режим', 'autoplay_off'],
  ['Включи автоплей', 'autoplay_on'], ['Отключи автоподбор песен', 'autoplay_off'],
  ['Включи рекомендации от YouTube', 'autoplay_on'], ['Выключи рекомендации музыки', 'autoplay_off'],
  ['Включи автоматический подбор песен', 'autoplay_on'], ['Останови автоматическое воспроизведение', 'autoplay_off'],
  ['Включи режим бесконечного воспроизведения', 'autoplay_on'],
  ['Включи режим автоподбора', 'autoplay_on'], ['Включи бесконечный режим на Ютубе', 'autoplay_on'],
  ['Поставь бесконечный режим', 'autoplay_on'], ['Включи музыку бесконечно', 'autoplay_on'],
  ['Включи режим повторения', 'loop_on'],
  ['Ключи повтор трека', 'loop_on'], ['Ключи автопли', 'autoplay_on'], ['Включи авто плей', 'autoplay_on'],
  ['Хлючи бесконечный режим', 'autoplay_on'], ['Хлючи, повтор трека', 'loop_on'], ['Хлючи автоплей', 'autoplay_on'],
  ['Включи повтор трека', 'loop_on'], ['Отключи повтор этой песни', 'loop_off'],
  ['Зацикли этот трек', 'loop_on'], ['Перестань повторять эту песню', 'loop_off'],
  ['Включи режим повтора', 'loop_on'], ['Выключи зацикливание трека', 'loop_off'],
  ['Очисти очередь', 'queue_clear'], ['Удали все треки из очереди', 'queue_clear'],
  ['Включи голосовое управление', 'voice_on'], ['Выключи голосовое управление', 'voice_off'],
  ['Включи паузу', 'pause'], ['Выключи режим паузы', 'resume'],
  ['Включи следующий трек', 'skip'], ['Ну, включи бесконечный режим, пожалуйста', 'autoplay_on'],
];
function harness(override: Partial<VoiceBackend> = {}, hostOverride: Partial<VoiceHost> = {}) {
  let text = 'Муза';
  let present = true;
  let name = 'Муза';
  let ducked = false;
  let cues = 0;
  const actions: VoiceIntent[] = [];
  const backend: VoiceBackend = {
    transcribe: async () => text,
    decideMusic: async () => ({ next_tool: 'unknown', query_source: 'message', search_result_policy: 'play_first_result', confidence: 1 }),
    rerankMusic: async () => ({ best_track: 0, confidence: 1 }),
    classify: async () => ({ action: 'skip', confidence: 0.99 }), ...override,
  };
  const host: VoiceHost = {
    wakeName: () => name, present: () => present,
    cue: async () => { cues++; }, duck: (value) => { ducked = value; },
    execute: async (intent) => { actions.push(intent); }, ...hostOverride,
    playMusic: hostOverride.playMusic ?? (async () => false),
  };
  const session = new VoiceSession(backend, host);
  return { session, actions, setText: (value: string) => { text = value; },
    setPresent: (value: boolean) => { present = value; }, setName: (value: string) => { name = value; },
    get ducked() { return ducked; }, get cues() { return cues; } };
}

test('wake name is exact after Unicode normalization; mentions in conversation do not wake', () => {
  assert.equal(isWakePhrase('  МУЗА! ', 'Муза'), true);
  assert.equal(isWakePhrase('Bot.', 'Бот'), true);
  assert.equal(isWakePhrase('Бот', 'Bot'), true);
  assert.equal(isWakePhrase('Muza', 'Муза'), true);
  assert.equal(isWakePhrase('Вот', 'Бот'), false);
  assert.equal(isWakePhrase('Кот', 'Бот'), false);
  assert.equal(isWakePhrase('Музочка', 'Муза'), false);
  assert.equal(isWakePhrase('Муза следующий', 'Муза'), false);
  assert.equal(isWakePhrase('Я говорил с Музой', 'Муза'), false);
  assert.equal(isWakePhrase('Бот енот', 'Бот Ёнот'), true);
  assert.equal(isWakePhrase('!!!', '!!!'), false);
  assert.throws(() => validateWakeName('🔊'));
});

test('every explicit mode command executes once without model routing or a YouTube search', async () => {
  for (const [phrase, action] of modeCases) {
    assert.deepEqual(modeCommand(phrase), { action }, phrase);
    assert.equal(extractMusicRequest(phrase, true), null, phrase);
    const h = harness({ classify: async () => { assert.fail('Explicit modes must not depend on old model labels'); } },
      { playMusic: async () => { assert.fail('A mode must not enqueue a song'); } });
    try {
      await h.session.begin('alice')!.complete(audio);
      h.setText(phrase); const capture = h.session.begin('alice')!;
      await capture.complete(audio); await capture.complete(audio);
      assert.deepEqual(h.actions, [{ action }], phrase);
    } finally { h.session.disable(); }
  }
});

test('unknown, negated, compound and incomplete modes cannot change state or search', async () => {
  for (const phrase of ['Не включай бесконечный режим', 'Включи бесконечный режим и паузу',
    'Включи бесконечный режим?', 'Не хлючи автоплей', 'Хлючи повтор и останови музыку',
    'Хлючи бесконечный режим?', 'Выключи повтор и останови музыку', 'Включи ночной режим',
    'Включи режим перемешивания', 'Автоплей', 'Повтор трека', 'Очисти очередь и включи Numb',
    'Включи бесконечный режим Metallica', 'Включи громкость']) {
    assert.equal(extractMusicRequest(phrase, true), null, phrase);
    const h = harness({ classify: async () => { assert.fail('Invalid modes must be rejected'); } },
      { playMusic: async () => { assert.fail('Invalid modes must not search'); } });
    try {
      await h.session.begin('alice')!.complete(audio); h.setText(phrase);
      await h.session.begin('alice')!.complete(audio);
      assert.deepEqual(h.actions, [], phrase);
    } finally { h.session.disable(); }
  }
  for (const phrase of ['Включи песню Бесконечный режим', 'Включи трек Повтор', 'Включи Повторяю']) {
    assert.equal(modeCommand(phrase), null, phrase);
    assert.equal(extractMusicRequest(phrase, true)?.kind, 'search', phrase);
  }
});

test('volume numbers have a closed range and ambiguous values are rejected', () => {
  assert.deepEqual(modeCommand('Поставь громкость пятьдесят процентов'), { action: 'volume_set', level: 50 });
  assert.equal(extractMusicRequest('Поставь громкость пятьдесят процентов', true), null);
  for (const phrase of ['Поставь громкость 0', 'Поставь громкость -10', 'Поставь громкость 10.5',
    'Поставь громкость 151', 'Поставь громкость 50 или 60']) assert.deepEqual(modeCommand(phrase), { action: 'unknown' }, phrase);
  for (const [text, level] of [['Громкость 1', 1], ['Громкость 150 процентов', 150],
    ['Громкость пятьдесят пять', 55], ['Громкость сто двадцать три', 123]] as const) {
    assert.equal(parseVolume(text), level);
  }
  for (const text of ['Громкость 0', 'Громкость 151', 'Громкость 10.5', 'Громкость -10',
    'Громкость 50 или 60', 'Громкость пять пять', 'Громкость 100 50', 'Громкость двадцать десять', 'Громкость']) assert.equal(parseVolume(text), null);
  assert.deepEqual(validateIntent('Громкость сто пятьдесят', { action: 'volume_set', confidence: 0.99 }), { action: 'volume_set', level: 150 });
});

test('short next/pause aliases classify as controls before search, while explicit titles remain music', async () => {
  for (const phrase of ['следующее', 'следующая', 'далее', 'дальше', 'next', 'next track', 'skip', 'переключи', 'Вот, следующее пожалуйста']) {
    const h = harness({ classify: async text => {
      assert.equal(text, 'Следующий трек'); return { action: 'skip', confidence: 0.99 };
    } }, { playMusic: async () => { assert.fail('A short control must never search'); } });
    try {
      await h.session.begin('alice')!.complete(audio);
      h.setText(phrase); const capture = h.session.begin('alice')!;
      await capture.complete(audio); await capture.complete(audio);
      assert.deepEqual(h.actions, [{ action: 'skip' }], phrase);
      assert.equal(h.ducked, false);
    } finally { h.session.disable(); }
  }
  let played = 0;
  const h = harness({ classify: async () => { assert.fail('An explicit title bypasses control classification'); } },
    { playMusic: async message => { assert.equal(message, 'Включи Next'); played++; return true; } });
  try {
    await h.session.begin('alice')!.complete(audio);
    h.setText('Включи Next'); await h.session.begin('alice')!.complete(audio);
    assert.equal(played, 1); assert.equal(h.actions.length, 0);
  } finally { h.session.disable(); }
});

test('bare artists receive a control preflight without letting a model hallucination skip playback', async () => {
  const calls: string[] = [];
  const h = harness({ classify: async () => { calls.push('classify'); return { action: 'skip', confidence: 0.999 }; } },
    { playMusic: async () => { calls.push('search'); return true; } });
  try {
    await h.session.begin('alice')!.complete(audio);
    h.setText('Монеточка'); await h.session.begin('alice')!.complete(audio);
    assert.deepEqual(calls, ['classify', 'search']); assert.equal(h.actions.length, 0);
  } finally { h.session.disable(); }
});

test('unknown short controls, negative and compound aliases never fall through to search', async () => {
  const h = harness({ classify: async () => ({ action: 'unknown', confidence: 1 }) },
    { playMusic: async () => { assert.fail('Must not search'); } });
  try {
    for (const phrase of ['Next', 'не следующее', 'next and pause', 'do not skip', 'переключи трек и пауза']) {
      h.setText('Муза'); await h.session.begin('alice')!.complete(audio);
      h.setText(phrase); await h.session.begin('alice')!.complete(audio);
      assert.equal(h.actions.length, 0); assert.equal(h.ducked, false);
    }
  } finally { h.session.disable(); }
});

test('the corpus captures commands once, preserves raw/canonical text and distinguishes cancelled decisions', async () => {
  const examples: TrainingExample[] = [];
  const h = harness({}, { training: row => examples.push(row) });
  try {
    await h.session.begin('alice')!.complete(audio); assert.equal(examples.length, 0);
    h.setText('Next'); const capture = h.session.begin('alice')!;
    await capture.complete(audio); await capture.complete(audio);
    assert.equal(examples.length, 1);
    assert.equal(examples[0]?.state.message, 'Next'); assert.equal(examples[0]?.state.canonical_message, 'Следующий трек');
    assert.deepEqual(examples[0]?.model_decision, { action: 'skip', confidence: 0.99 });
    assert.equal(examples[0]?.outcome, 'changed'); assert.equal(examples[0]?.route, 'control');
  } finally { h.session.disable(); }
  let release!: (value: { action: 'skip'; confidence: number }) => void;
  const cancelled = harness({ classify: async () => new Promise(resolve => { release = resolve; }) },
    { training: row => examples.push(row) });
  try {
    await cancelled.session.begin('alice')!.complete(audio);
    cancelled.setText('Next'); const processing = cancelled.session.begin('alice')!.complete(audio);
    await Promise.resolve(); cancelled.session.cancel(); release({ action: 'skip', confidence: 1 }); await processing;
    assert.equal(examples.at(-1)?.outcome, 'cancelled'); assert.equal(cancelled.actions.length, 0);
  } finally { cancelled.session.disable(); }
});

test('music corpus retains search stages and records late cancellation without a false success', async () => {
  for (const cancel of [false, true]) {
    const examples: TrainingExample[] = [];
    let release!: () => void;
    let entered!: () => void;
    const started = new Promise<void>(resolve => { entered = resolve; });
    const h = harness({}, { training: row => examples.push(row),
      playMusic: async (_message, _user, _signal, valid, report) => {
        report?.({ stage: 'search', candidates: 5, ms: 25 });
        entered(); await new Promise<void>(resolve => { release = resolve; });
        return valid();
      } });
    try {
      await h.session.begin('alice')!.complete(audio);
      h.setText('Включи Numb'); const capture = h.session.begin('alice')!;
      const processing = capture.complete(audio); await started;
      if (cancel) h.session.disable();
      release(); await processing; await capture.complete(audio);
      assert.equal(examples.length, 1);
      assert.equal(examples[0]?.outcome, cancel ? 'cancelled' : 'changed');
      assert.equal(examples[0]?.query, 'Numb');
      assert.ok(examples[0]?.diagnostics.some(event => event.stage === 'search' && event.candidates === 5));
      assert.equal(h.ducked, false);
    } finally { release?.(); h.session.disable(); }
  }
});

test('corpus callback failures do not change a successful command', async () => {
  const h = harness({}, { training: () => { throw new Error('write failed'); } });
  try {
    await h.session.begin('alice')!.complete(audio);
    h.setText('Next'); await h.session.begin('alice')!.complete(audio);
    assert.deepEqual(h.actions, [{ action: 'skip' }]); assert.equal(h.ducked, false);
  } finally { h.session.disable(); }
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

test('generic turn-on requests resume only a paused track and never canonicalize song requests', async () => {
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

test('a new wake utterance is captured while the previous one is being recognized', async () => {
  const pending: ((text: string) => void)[] = [];
  const h = harness({ transcribe: async (_pcm, signal) => new Promise<string>((resolve, reject) => {
    pending.push(resolve);
    signal.addEventListener('abort', () => reject(new Error('cancelled')), { once: true });
  }) });
  try {
    const first = h.session.begin('alice')!;
    assert.equal(h.session.begin('alice'), null, 'do not overlap audio captures');
    const firstResult = first.complete(audio);
    const second = h.session.begin('alice');
    assert.ok(second, 'recognition must not block receiving the next utterance');
    const secondResult = second.complete(audio);
    pending[1]!('Муза'); await secondResult;
    assert.equal(h.cues, 1); assert.equal(first.signal.aborted, true);
    pending[0]!('Муза'); await firstResult;
    assert.equal(h.cues, 1, 'late recognition must not activate twice');
    assert.equal(h.session.phase, 'awaiting');
  } finally { h.session.disable(); }
});

test('leaving cancels every queued wake recognition from the same speaker', async () => {
  const pending: ((text: string) => void)[] = [];
  const h = harness({ transcribe: async () => new Promise<string>((resolve) => pending.push(resolve)) });
  const captures = [];
  const results = [];
  for (let i = 0; i < 4; i++) {
    const capture = h.session.begin('alice')!;
    captures.push(capture); results.push(capture.complete(audio));
  }
  assert.equal(h.session.begin('alice'), null, 'bound pending audio');
  h.session.cancelUser('alice');
  for (const capture of captures) assert.equal(capture.signal.aborted, true);
  for (const resolve of pending) resolve('Муза');
  await Promise.all(results);
  assert.equal(h.cues, 0); assert.equal(h.session.phase, 'idle');
  h.session.disable();
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
