import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtemp, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { contextualCommand, isWakePhrase, modeCommand, parseVolume, playerControlRequest, ruleIntent, validateIntent, validateWakeName, type VoiceIntent } from '../src/voice/intents.js';
import { extractMusicRequest } from '../src/voice/music.js';
import { EXTRA_VOICE_PHRASES } from '../src/voice/phrases.js';
import { VoiceSettings } from '../src/voice/settings.js';
import { VoiceSession, type VoiceBackend, type VoiceDiagnostic, type VoiceHost } from '../src/voice/session.js';
import type { TrainingExample } from '../src/voice/training.js';

const audio = Buffer.from([0, 0]);
test('only activated STT input is archived, including compound requests and failed recognition', async () => {
  const archived: Buffer[] = [];
  let wake = false;
  let failStt = false;
  const h = harness({ detectWake: async () => ({ wake, probability: wake ? 1 : 0 }),
    transcribe: async () => { if (failStt) throw new Error('STT offline'); return 'Бот, следующий трек'; } },
    { archiveAudio: pcm => { archived.push(pcm); } });
  const short = Buffer.alloc(800 * 32);
  const command = Buffer.alloc(2000 * 32);
  try {
    await h.session.begin('alice')!.complete(short);
    assert.equal(archived.length, 0);
    wake = true;
    await h.session.begin('alice')!.complete(short);
    assert.equal(archived.length, 0);
    assert.equal(h.session.begin('bob'), null);
    await h.session.begin('alice')!.complete(command);
    assert.deepEqual(archived, [command]);
    await h.session.begin('alice')!.complete(command);
    assert.deepEqual(archived, [command, command]);
    await h.session.begin('alice')!.complete(short);
    failStt = true;
    await h.session.begin('alice')!.complete(command);
    assert.equal(archived.length, 3);
    h.setPresent(false);
    assert.equal(h.session.begin('alice'), null);
  } finally { h.session.disable(); }
});

test('archive errors do not stop a voice command', async () => {
  const h = harness({}, { archiveAudio: () => { throw new Error('disk full'); } });
  try {
    await h.session.begin('alice')!.complete(audio);
    h.setText('Следующий трек');
    await h.session.begin('alice')!.complete(audio);
    assert.deepEqual(h.actions, [{ action: 'skip' }]);
  } finally { h.session.disable(); }
});
test('extended phrase dictionary works through rules and voice sessions without changing music titles', async () => {
  for (const [canonical, phrases] of Object.entries(EXTRA_VOICE_PHRASES)) {
    const expected = ruleIntent(canonical);
    assert.notEqual(expected.action, 'unknown', canonical);
    const h = harness({ classify: async (text) => {
      assert.equal(text, canonical);
      return { action: expected.action, confidence: 0.99 };
    } }, { playMusic: async () => { assert.fail('Control must not search for music'); } });
    try {
      for (const phrase of phrases) {
        for (const text of [phrase, `Ну, ${phrase}, пожалуйста!`]) {
          assert.deepEqual(ruleIntent(text), expected, text);
          assert.equal(playerControlRequest(text), true, text);
          assert.equal(extractMusicRequest(text, true), null, text);
          h.setText('Муза'); await h.session.begin('alice')!.complete(audio);
          h.setText(text); await h.session.begin('alice')!.complete(audio);
        }
        for (const text of [`Не ${phrase}`, `${phrase}?`, `${phrase} и пауза`, `${phrase} завтра`]) {
          assert.deepEqual(ruleIntent(text), { action: 'unknown' }, text);
        }
        const title = `Включи песню ${phrase}`;
        assert.deepEqual(extractMusicRequest(title), { kind: 'search', query: phrase }, title);
      }
      assert.deepEqual(h.actions, phrases.flatMap(() => [expected, expected]), canonical);
    } finally { h.session.disable(); }
  }
});

test('spoken volume aliases preserve the range and reject malformed or relative values', () => {
  for (const text of ['Выставь громкость 50', 'Задай звук на пятьдесят процентов',
    'Поставь звук до 50', 'Установи громкость до 50', 'Звук на 50', 'Сделай громкость на 50']) {
    assert.equal(contextualCommand(text, false), 'Громкость 50', text);
    assert.deepEqual(ruleIntent(text), { action: 'volume_set', level: 50 }, text);
    assert.equal(extractMusicRequest(text, true), null, text);
  }
  for (const text of ['Задай звук 0', 'Выставь громкость 151', 'Звук -10', 'Звук 10.5',
    'Звук 10,5', 'Выставь громкость пять пять', 'Звук на 50 и пауза', 'Прибавь звук на 20']) {
    assert.deepEqual(ruleIntent(text), { action: 'unknown' }, text);
  }
});

const conversationalControls = [
  ['skip', 'Следующий трек', ['Следующее', 'Следующая', 'Следующий', 'Далее', 'Дальше',
    'Включи следующую', 'Включи следующее', 'Включить следующий трек', 'Поставь следующую песню',
    'Запусти следующий', 'Сыграй следующую композицию', 'Включи другую песню', 'Поставь другой трек',
    'Пропусти эту', 'Пропусти этот трек', 'Пропустить текущую песню', 'Пропускай', 'Скипнуть',
    'Поменяй песню', 'Смени трек', 'Переключи на следующую', 'Переключись на следующий трек',
    'Перейди к следующему треку', 'Переходи к следующей песне', 'Дальше по очереди']],
  ['pause', 'Поставь на паузу', ['Сделай паузу', 'Нажми паузу', 'Поставь музыку на паузу',
    'Поставь трек на паузу', 'Приостановить воспроизведение', 'Притормози', 'Притормози песню']],
  ['resume', 'Продолжи музыку', ['Сними паузу', 'Убери паузу', 'Убери с паузы', 'Сними трек с паузы',
    'Снять с паузы', 'Продолжить играть', 'Возобновить музыку', 'Возобнови проигрывание']],
  ['stop', 'Останови музыку', ['Стоп музыка', 'Остановить', 'Останови плеер', 'Выключить музыку',
    'Выключай музыку', 'Отключи музыку', 'Хватит музыки', 'Перестань играть', 'Прекрати воспроизведение',
    'Выйти из канала', 'Отключись от голосового канала']],
  ['volume_up', 'Сделай громче', ['Прибавь громкость', 'Добавь громкости', 'Добавь звука',
    'Погромче сделай', 'Сделай музыку громче', 'Сделай звук погромче', 'Увеличить громкость']],
  ['volume_down', 'Сделай тише', ['Убавь громкость', 'Убавь звука', 'Потише сделай',
    'Сделай музыку тише', 'Сделай трек потише', 'Уменьшить громкость']],
] as const;

test('conversational controls share canonical routing in both voice profiles', () => {
  for (const [action, canonical, phrases] of conversationalControls) {
    for (const phrase of phrases) {
      for (const text of [phrase, `Ну, ${phrase}, пожалуйста!`]) {
        const direct = modeCommand(text);
        const resolved = contextualCommand(text, false);
        if (direct) assert.deepEqual(direct, { action }, text);
        else assert.equal(resolved, canonical, text);
        assert.equal(contextualCommand(text, true), resolved, text);
        assert.equal(playerControlRequest(text), true, text);
        assert.equal(extractMusicRequest(text, true), null, text);
        assert.deepEqual(ruleIntent(text), { action }, text);
        assert.deepEqual(validateIntent(resolved, { action, confidence: 0.99 }), { action }, text);
      }
      for (const text of [`Не ${phrase}`, `${phrase}?`, `${phrase} и сделай паузу`, `${phrase} завтра`]) {
        assert.deepEqual(ruleIntent(text), { action: 'unknown' }, text);
      }
    }
  }
  for (const title of ['Следующее', 'Next', 'Сделай паузу', 'Поменяй песню', 'Хватит музыки']) {
    const text = `Включи песню ${title}`;
    assert.deepEqual(extractMusicRequest(text, true), { kind: 'search', query: title }, text);
    assert.deepEqual(ruleIntent(text), { action: 'unknown' }, text);
  }
});

test('voice session classifies canonical conversational requests and executes controls without music search', async () => {
  for (const [action, canonical, phrases] of conversationalControls) {
    const h = harness({ classify: async (text) => {
      assert.equal(text, canonical);
      return { action, confidence: 0.99 };
    } }, { playMusic: async () => { assert.fail('Control must not search for music'); } });
    try {
      for (const phrase of phrases) {
        h.setText('Муза'); await h.session.begin('alice')!.complete(audio);
        h.setText(phrase); await h.session.begin('alice')!.complete(audio);
      }
      assert.deepEqual(h.actions, phrases.map(() => ({ action })));
    } finally { h.session.disable(); }
  }
});

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

test('neural wake detector activates cue and awaiting mode on wake word without calling cloud STT', async () => {
  let cloudTranscribeCalls = 0;
  const h = harness({
    detectWake: async () => ({ wake: true, probability: 0.99 }),
    classify: async () => ({ action: 'pause', confidence: 0.99 }),
    transcribe: async (_pcm, _sig, command) => {
      cloudTranscribeCalls++;
      if (!command) assert.fail('Cloud transcribe must not be called during standalone wake detection');
      return 'Поставь на паузу';
    },
  });
  try {
    // 800ms audio (< 1300ms)
    const shortAudio = Buffer.alloc(800 * 32);
    await h.session.begin('alice')!.complete(shortAudio);
    assert.equal(cloudTranscribeCalls, 0, 'zero cloud STT calls for wake phrase');
    assert.equal(h.cues, 1, 'cue tone played');
    assert.equal(h.ducked, true, 'audio ducked');
    assert.equal(h.session.phase, 'awaiting');

    // Alice now speaks her command
    const commandCapture = h.session.begin('alice')!;
    await commandCapture.complete(shortAudio);
    assert.equal(cloudTranscribeCalls, 1, 'cloud STT called for spoken command');
    assert.deepEqual(h.actions, [{ action: 'pause' }]);
  } finally { h.session.disable(); }
});

test('neural wake detector rejects non-wake audio locally without invoking cloud STT', async () => {
  const h = harness({
    detectWake: async () => ({ wake: false, probability: 0.0001 }),
    transcribe: async () => {
      assert.fail('Cloud transcribe must not be called when wake detector rejects audio');
    },
  });
  try {
    const speechAudio = Buffer.alloc(1000 * 32);
    await h.session.begin('alice')!.complete(speechAudio);
    assert.equal(h.cues, 0);
    assert.equal(h.ducked, false);
    assert.equal(h.session.phase, 'idle');
    assert.deepEqual(h.actions, []);
  } finally { h.session.disable(); }
});

test('unavailable or incompatible wake detector falls back to STT and preserves custom names', async () => {
  let calls = 0;
  const h = harness({ supportsWake: () => false,
    detectWake: async () => { assert.fail('unsupported detector must not gate the session'); },
    transcribe: async () => { calls++; return 'Муза'; },
  });
  try {
    await h.session.begin('alice')!.complete(audio);
    assert.equal(calls, 1); assert.equal(h.cues, 1); assert.equal(h.session.phase, 'awaiting');
  } finally { h.session.disable(); }
});

test('neural wake detector on compound command activates cloud STT and executes immediately', async () => {
  let cloudTranscribed = false;
  const h = harness({
    detectWake: async () => ({ wake: true, probability: 0.99 }),
    transcribe: async () => {
      cloudTranscribed = true;
      return 'Бот, следующий трек';
    },
  });
  try {
    // 2000ms audio (> 1300ms, long utterance containing wake + command)
    const longAudio = Buffer.alloc(2000 * 32);
    await h.session.begin('alice')!.complete(longAudio);
    assert.equal(cloudTranscribed, true, 'cloud STT called for compound command');
    assert.deepEqual(h.actions, [{ action: 'skip' }]);
  } finally { h.session.disable(); }
});

test('math expressions and conversational queries never enqueue music in a voice session', async () => {
  const feedbackCalls: string[] = [];
  const h = harness({ classify: async () => ({ action: 'unknown', confidence: 1 }) },
    {
      playMusic: async () => { assert.fail('Math and conversational queries must never search for music'); },
      feedback: async (key) => { feedbackCalls.push(key); },
    });
  try {
    for (const phrase of ['2 + 2', 'два плюс два', 'посчитай 2 + 2', 'сколько будет 2 + 2', 'включи 2 + 2', 'что делаешь', 'как дела', 'погода', 'включи свет', 'поставь таймер']) {
      h.setText('Муза'); await h.session.begin('alice')!.complete(audio);
      h.setText(phrase); await h.session.begin('alice')!.complete(audio);
      assert.equal(h.actions.length, 0);
      assert.equal(h.ducked, false);
    }
    assert.ok(feedbackCalls.every((call) => call === 'unknown'));
  } finally { h.session.disable(); }
});

test('compound utterance with math expression does not enqueue music', async () => {
  const feedbackCalls: string[] = [];
  const h = harness({
    detectWake: async () => ({ wake: true, probability: 0.99 }),
    classify: async () => ({ action: 'unknown', confidence: 1 }),
    transcribe: async () => 'Бот, 2 + 2',
  }, {
    playMusic: async () => { assert.fail('Compound math must not search for music'); },
    feedback: async (key) => { feedbackCalls.push(key); },
  });
  try {
    const longAudio = Buffer.alloc(2000 * 32);
    await h.session.begin('alice')!.complete(longAudio);
    assert.equal(h.actions.length, 0);
    assert.deepEqual(feedbackCalls, ['unknown']);
  } finally { h.session.disable(); }
});

test('neural command detector executes control actions immediately without calling STT', async () => {
  let cloudTranscribed = false;
  let commandDetected = false;
  const shortWake = Buffer.alloc(800 * 32);
  const h = harness({
    detectWake: async () => ({ wake: true, probability: 0.99 }),
    detectCommand: async () => {
      commandDetected = true;
      return { class: 'pause', action: 'pause', confidence: 0.98, matched: true };
    },
    transcribe: async () => {
      cloudTranscribed = true;
      return 'пауза';
    },
  });
  try {
    await h.session.begin('alice')!.complete(shortWake);
    assert.equal(h.session.phase, 'awaiting');
    await h.session.begin('alice')!.complete(audio);
    assert.equal(commandDetected, true, 'command detector was invoked');
    assert.equal(cloudTranscribed, false, 'cloud STT was bypassed!');
    assert.deepEqual(h.actions, [{ action: 'pause' }]);
  } finally { h.session.disable(); }
});

test('neural command detector falls back to cloud STT for arbitrary music search', async () => {
  let cloudTranscribed = false;
  let commandDetected = false;
  let musicSearched = false;
  const shortWake = Buffer.alloc(800 * 32);
  const h = harness({
    detectWake: async () => ({ wake: true, probability: 0.99 }),
    detectCommand: async () => {
      commandDetected = true;
      return { class: 'play_search', action: 'play_search', confidence: 0.95, matched: true };
    },
    transcribe: async () => {
      cloudTranscribed = true;
      return 'включи Linkin Park Numb';
    },
  }, {
    playMusic: async () => {
      musicSearched = true;
      return true;
    },
  });
  try {
    await h.session.begin('alice')!.complete(shortWake);
    assert.equal(h.session.phase, 'awaiting');
    await h.session.begin('alice')!.complete(audio);
    assert.equal(commandDetected, true, 'command detector was invoked');
    assert.equal(cloudTranscribed, true, 'cloud STT was invoked for song search');
    assert.equal(musicSearched, true, 'music search was triggered');
  } finally { h.session.disable(); }
});


