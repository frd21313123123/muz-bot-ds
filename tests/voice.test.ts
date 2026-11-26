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
  let transcript = 'Муза';
  const h = harness({ detectWake: async () => ({ wake, probability: wake ? 1 : 0 }),
    transcribe: async () => { if (!transcript) throw new Error('STT offline'); return transcript; } },
    { archiveAudio: pcm => { archived.push(pcm); } });
  const short = Buffer.alloc(800 * 32);
  const command = Buffer.alloc(2000 * 32);
  try {
    await h.session.begin('alice')!.complete(short);
    assert.equal(archived.length, 0);
    wake = true;
    transcript = 'В разговоре упомянули Музу';
    await h.session.begin('alice')!.complete(command);
    assert.deepEqual(archived, []);
    transcript = 'Муза';
    await h.session.begin('alice')!.complete(short);
    assert.equal(archived.length, 0);
    assert.equal(h.session.begin('bob'), null);
    transcript = 'Следующий трек';
    await h.session.begin('alice')!.complete(command);
    assert.deepEqual(archived, [command]);
    transcript = 'Муза, следующий трек';
    await h.session.begin('alice')!.complete(command);
    assert.deepEqual(archived, [command, command]);
    transcript = 'Муза';
    await h.session.begin('alice')!.complete(short);
    transcript = '';
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

test('the full form still requires the configured wake word before any control', async () => {
  const h = harness({ classify: async text => ({ action: ruleIntent(text).action, confidence: 1 }) });
  try {
    h.setName('бот');
    for (const text of ['сделай тише', 'музыкальный', 'музыкальный вот', 'музыкальный кот, сделай тише',
      'музыкальный ботаник, сделай тише', 'мы обсуждали музыкального бота', 'он сказал музыкальный бот']) {
      h.setText(text); await h.session.begin('alice')!.complete(audio);
    }
    assert.equal(h.cues, 0); assert.deepEqual(h.actions, []);
    h.setText('Музыкальный бот!');
    await h.session.begin('alice')!.complete(audio);
    assert.ok(h.session.awaits('alice')); assert.equal(h.cues, 1);
    assert.equal(h.session.begin('bob'), null);
    h.setText('Сделай тише'); await h.session.begin('alice')!.complete(audio);
    h.setText('Музыкальный бот, поставь на паузу'); await h.session.begin('bob')!.complete(audio);
    assert.deepEqual(h.actions, [{ action: 'volume_down' }, { action: 'pause' }]); assert.equal(h.cues, 1);
    h.setName('Муза'); h.setText('Музыкальный бот'); await h.session.begin('bob')!.complete(audio);
    assert.equal(h.session.phase, 'idle'); assert.equal(h.cues, 1);
  } finally { h.session.disable(); }
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

test('bare artists close the listening window without searching or classifying', async () => {
  const calls: string[] = [];
  const h = harness({ classify: async () => { calls.push('classify'); return { action: 'skip', confidence: 0.999 }; } },
    { playMusic: async () => { calls.push('search'); return true; } });
  try {
    await h.session.begin('alice')!.complete(audio);
    h.setText('Монеточка'); await h.session.begin('alice')!.complete(audio);
    assert.deepEqual(calls, []); assert.equal(h.actions.length, 0);
    assert.equal(h.session.phase, 'idle'); assert.equal(h.ducked, false);
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

test('activation ducks music, binds the speaker, executes once and restores gain', async () => {
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

test('the activated speaker can start the command while the wake cue is still playing', async () => {
  let releaseCue!: () => void;
  let cueStarted!: () => void;
  const started = new Promise<void>((resolve) => { cueStarted = resolve; });
  const h = harness({ detectWake: async () => ({ wake: true, probability: 0.99 }) }, {
    cue: () => new Promise<void>((resolve) => { releaseCue = resolve; cueStarted(); }),
  });
  try {
    const wake = h.session.begin('alice')!;
    const activated = wake.complete(Buffer.alloc(800 * 32));
    await started;
    assert.equal(h.session.phase, 'signalling');
    assert.equal(h.ducked, true, 'music is already ducked while the cue is playing');
    assert.equal(h.session.begin('bob'), null, 'other speakers cannot take over the cue');
    const command = h.session.begin('alice')!;
    assert.equal(h.session.phase, 'capturing');
    releaseCue();
    await activated;
    assert.equal(h.session.phase, 'capturing', 'cue completion must not reopen the wait window');
    assert.equal(h.ducked, true, 'music stays ducked while the command is captured');
    h.setText('Следующий');
    await command.complete(audio);
    assert.deepEqual(h.actions, [{ action: 'skip' }]);
    assert.equal(h.ducked, false); assert.equal(h.session.phase, 'idle');
  } finally { h.session.disable(); }
});

test('a failed cue cancels an early command even when its audio has already ended', async () => {
  let failCue!: (error: Error) => void;
  let notifyStarted!: () => void;
  const started = new Promise<void>(resolve => { notifyStarted = resolve; });
  const h = harness({}, { cue: () => new Promise<void>((_resolve, reject) => {
    failCue = reject; notifyStarted();
  }) });
  try {
    const activation = h.session.begin('alice')!.complete(audio);
    await started;
    h.setText('Следующий');
    const capture = h.session.begin('alice')!;
    const processing = capture.complete(audio);
    await new Promise<void>(resolve => setImmediate(resolve));
    assert.deepEqual(h.actions, []);
    failCue(new Error('Cue playback failed'));
    await Promise.all([activation, processing]);
    assert.deepEqual(h.actions, []);
    assert.equal(capture.signal.aborted, true);
    assert.equal(h.session.phase, 'idle'); assert.equal(h.ducked, false);
  } finally { h.session.disable(); }
});

test('repeating the wake name re-arms the command window instead of searching for it', async () => {
  let cues = 0;
  let searches = 0;
  const h = harness({}, {
    cue: async () => { cues++; },
    playMusic: async () => { searches++; return false; },
  });
  try {
    await h.session.begin('alice')!.complete(audio);
    assert.equal(cues, 1); assert.equal(h.session.phase, 'awaiting');
    h.setText('Муза');
    await h.session.begin('alice')!.complete(audio);
    assert.equal(cues, 2, 'the cue tells the speaker the window is still open');
    assert.equal(h.session.phase, 'awaiting');
    assert.equal(h.ducked, true);
    assert.equal(searches, 0, 'the bot name must never become a music query');
    assert.deepEqual(h.actions, []);
    h.setText('Следующий');
    await h.session.begin('alice')!.complete(audio);
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

test('a verified compound request claims the turn while another wake recognition finishes', async () => {
  const transcripts: ((text: string) => void)[] = [];
  const h = harness({ transcribe: () => new Promise(resolve => transcripts.push(resolve)) });
  try {
    const alice = h.session.begin('alice')!;
    const bob = h.session.begin('bob')!;
    const a = alice.complete(audio), b = bob.complete(audio);
    transcripts[1]!('Муза, следующий трек');
    transcripts[0]!('Муза, следующий трек');
    await Promise.all([a, b]);
    assert.equal(alice.signal.aborted, true);
    assert.deepEqual(h.actions, [{ action: 'skip' }]);
    assert.equal(h.cues, 0); assert.equal(h.ducked, false);
    assert.equal(h.session.phase, 'idle');
  } finally { h.session.disable(); }
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

test('a second speaker can wake the bot when the first has filled the pending recognition slots', async () => {
  const pending: ((text: string) => void)[] = [];
  const h = harness({ transcribe: () => new Promise(resolve => pending.push(resolve)) });
  try {
    const requests = [];
    for (let i = 0; i < 3; i++) {
      const capture = h.session.begin('alice')!;
      requests.push({ capture, result: capture.complete(audio) });
    }
    const ongoing = h.session.begin('alice')!;
    const friend = h.session.begin('bob');
    assert.ok(friend);
    assert.ok(requests[0]!.capture.signal.aborted, 'evict an old recognition from the same microphone');
    assert.equal(ongoing.signal.aborted, false, 'never evict a live utterance to allocate a slot');
    const activation = friend.complete(audio);
    pending[3]!('Муза'); await activation;
    for (let i = 0; i < 3; i++) pending[i]!('Муза');
    await Promise.all(requests.map(request => request.result));
    assert.equal(h.cues, 1);
    assert.ok(h.session.awaits('bob'));
    assert.equal(h.session.begin('alice'), null);
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

test('a short local wake hit opens the window only after one verified transcription', async () => {
  const transcriptions: boolean[] = [];
  const h = harness({
    detectWake: async () => ({ wake: true, probability: 0.99 }),
    classify: async () => ({ action: 'pause', confidence: 0.99 }),
    transcribe: async (_pcm, _sig, command) => {
      transcriptions.push(command);
      return command ? 'Поставь на паузу' : 'Муза';
    },
  });
  try {
    const shortAudio = Buffer.alloc(800 * 32);
    await h.session.begin('alice')!.complete(shortAudio);
    assert.deepEqual(transcriptions, [false]);
    assert.equal(h.cues, 1, 'cue tone played');
    assert.equal(h.ducked, true, 'audio ducked');
    assert.equal(h.session.phase, 'awaiting');

    const commandCapture = h.session.begin('alice')!;
    await commandCapture.complete(shortAudio);
    assert.deepEqual(transcriptions, [false, true]);
    assert.deepEqual(h.actions, [{ action: 'pause' }]);
  } finally { h.session.disable(); }
});

test('neural wake detector rejects non-wake audio locally without invoking cloud STT', async () => {
  let transcriptions = 0;
  const h = harness({
    detectWake: async () => ({ wake: false, probability: 0.0001 }),
    transcribe: async () => { transcriptions++; return 'Муза'; },
  });
  try {
    const speechAudio = Buffer.alloc(1000 * 32);
    await h.session.begin('alice')!.complete(speechAudio);
    assert.equal(transcriptions, 0);
    assert.equal(h.cues, 0);
    assert.equal(h.ducked, false);
    assert.equal(h.session.phase, 'idle');
    assert.deepEqual(h.actions, []);
  } finally { h.session.disable(); }
});

test('a quiet isolated name missed locally can wake its speaker, while short non-names cannot dispatch', async () => {
  const burst = Buffer.alloc(500 * 32);
  for (let sample = 1600; sample < 4800; sample++) {
    burst.writeInt16LE(Math.round(150 * Math.sin(2 * Math.PI * 170 * sample / 16_000)), sample * 2);
  }
  const original = Buffer.from(burst);
  const inputs: Buffer[] = [];
  let text = 'Бот, стоп';
  const h = harness({
    detectWake: async () => ({ wake: false, probability: 0.76 }),
    transcribe: async pcm => { inputs.push(pcm); return text; },
  });
  h.setName('бот');
  try {
    await h.session.begin('alice')!.complete(burst);
    assert.equal(h.cues, 0); assert.deepEqual(h.actions, []);
    assert.equal(inputs.length, 1);
    assert.equal(inputs[0]!.readInt16LE(3250), 8 * original.readInt16LE(3250));
    assert.deepEqual(burst, original);
    text = 'бот';
    await h.session.begin('alice')!.complete(burst);
    assert.equal(inputs.length, 1, 'an unsuccessful recheck is rate-limited for that microphone');
    await h.session.begin('bob')!.complete(burst);
    assert.equal(inputs.length, 2, 'a different speaker may still use their first recheck');
    assert.equal(h.cues, 1); assert.ok(h.session.awaits('bob'));
    assert.equal(h.session.begin('alice'), null);
  } finally { h.session.disable(); }
});

function isolatedBurst(): Buffer {
  const pcm = Buffer.alloc(700 * 32);
  for (let i = 1600; i < 4800; i++) pcm.writeInt16LE(Math.round(280 * Math.sin(i * 0.09)), i * 2);
  return pcm;
}

test('the quiet tail can be shortened for the local detector while STT receives the complete utterance', async () => {
  const lengths: number[] = [];
  const stt: number[] = [];
  const h = harness({ detectWake: async pcm => {
    lengths.push(pcm.length);
    return { wake: pcm.length === 380 * 32, probability: pcm.length === 380 * 32 ? .99 : .01 };
  }, transcribe: async pcm => { stt.push(pcm.length); return 'бот'; } });
  try {
    h.setName('бот');
    await h.session.begin('bob')!.complete(isolatedBurst());
    assert.deepEqual(lengths, [700 * 32, 380 * 32]);
    assert.deepEqual(stt, [700 * 32]);
    assert.equal(h.cues, 1); assert.ok(h.session.awaits('bob'));
  } finally { h.session.disable(); }
});

test('voice off during a shortened wake check discards the late result', async () => {
  let trimmed!: () => void, finish!: (value: { wake: boolean; probability: number }) => void;
  const checking = new Promise<void>(resolve => { trimmed = resolve; });
  const h = harness({ detectWake: async pcm => {
    if (pcm.length === 700 * 32) return { wake: false, probability: .001 };
    trimmed();
    return new Promise(resolve => { finish = resolve; });
  }, transcribe: async () => assert.fail('cancelled audio must not reach speech recognition') });
  const pending = h.session.begin('bob')!.complete(isolatedBurst());
  await checking;
  h.session.disable();
  finish({ wake: true, probability: 1 }); await pending;
  assert.equal(h.session.phase, 'disabled'); assert.equal(h.cues, 0); assert.equal(h.ducked, false);
});

test('stale short-name verification cannot start a late cue', async context => {
  let now = 20;
  context.mock.method(performance, 'now', () => now);
  const h = harness({ detectWake: async () => ({ wake: false, probability: .5 }),
    transcribe: async () => { now += 5000; return 'бот'; } });
  try {
    h.setName('бот');
    await h.session.begin('bob')!.complete(isolatedBurst());
    assert.equal(h.cues, 0); assert.equal(h.ducked, false); assert.equal(h.session.phase, 'idle');
  } finally { h.session.disable(); }
});

test('two similar isolated misses from one speaker may verify only the exact name, once per cooldown', async context => {
  let now = 100, calls = 0, text = 'бот, стоп';
  context.mock.method(performance, 'now', () => now);
  const h = harness({ detectWake: async () => ({ wake: false, probability: .001 }),
    transcribe: async () => { calls++; return text; } });
  const burst = isolatedBurst();
  try {
    h.setName('бот');
    await h.session.begin('alice')!.complete(burst);
    now += 1000;
    await h.session.begin('bob')!.complete(burst);
    assert.equal(calls, 0);
    now += 1000;
    await h.session.begin('bob')!.complete(burst);
    assert.equal(calls, 1); assert.deepEqual(h.actions, []); assert.equal(h.cues, 0);
    text = 'бот';
    now += 1000;
    await h.session.begin('bob')!.complete(burst);
    assert.equal(calls, 1);
    now += 5000;
    await h.session.begin('bob')!.complete(burst);
    assert.equal(calls, 1, 'an old utterance cannot count as a retry');
    now += 900;
    await h.session.begin('bob')!.complete(burst);
    assert.equal(calls, 2); assert.equal(h.cues, 1); assert.ok(h.session.awaits('bob'));
  } finally { h.session.disable(); }
});

test('steady noise, long speech and a near-homophone cannot wake through the short recheck', async () => {
  let transcriptions = 0;
  let probability = 0.71;
  const h = harness({ detectWake: async () => ({ wake: false, probability }),
    transcribe: async () => { transcriptions++; return 'вот'; } });
  h.setName('бот');
  try {
    const tone = Buffer.alloc(1500 * 32);
    for (let i = 0; i < tone.length; i += 2) tone.writeInt16LE(i % 4 ? 200 : -200, i);
    await h.session.begin('alice')!.complete(tone.subarray(0, 500 * 32));
    tone.fill(0, 0, 120 * 32);
    await h.session.begin('alice')!.complete(tone);
    assert.equal(transcriptions, 0);
    probability = 0.001;
    await h.session.begin('alice')!.complete(tone.subarray(0, 500 * 32));
    assert.equal(transcriptions, 0, 'isolated sounds with negligible name probability stay local');
    probability = 0.71;
    await h.session.begin('alice')!.complete(tone.subarray(0, 500 * 32));
    assert.equal(transcriptions, 1);
    assert.equal(h.cues, 0); assert.equal(h.ducked, false);
  } finally { h.session.disable(); }
});

test('false local hits of either duration are discarded before cues, acoustic commands and feedback', async () => {
  const calls: string[] = [];
  const h = harness({
    detectWake: async () => ({ wake: true, probability: 0.9999 }),
    detectCommand: async () => { calls.push('acoustic'); return { class: 'stop', action: 'stop', confidence: 1, matched: true }; },
  }, {
    playMusic: async () => { calls.push('search'); return true; },
    feedback: async key => { calls.push(key); },
  });
  try {
    h.setName('Батон');
    for (const [duration, transcript] of [[800, 'потом'], [2600, 'передай батон пожалуйста'],
      [2200, 'следующий трек'], [3000, 'Батончик, включи музыку'], [600, '']] as const) {
      h.setText(transcript);
      await h.session.begin('alice')!.complete(Buffer.alloc(duration * 32));
      assert.equal(h.session.phase, 'idle', transcript);
      assert.equal(h.ducked, false, transcript);
    }
    assert.deepEqual(calls, []); assert.deepEqual(h.actions, []); assert.equal(h.cues, 0);
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

test('only the first two seconds reach the wake detector, and the full compound request reaches STT once', async () => {
  const full = Buffer.alloc(3400 * 32, 1);
  full.fill(2, 2000 * 32);
  const wakeInputs: Buffer[] = [];
  const sttInputs: Buffer[] = [];
  const h = harness({
    detectWake: async pcm => { wakeInputs.push(pcm); return { wake: true, probability: 0.99 }; },
    transcribe: async pcm => { sttInputs.push(pcm); return 'Бот, следующий трек'; },
  });
  try {
    h.setName('Бот');
    await h.session.begin('alice')!.complete(full);
    assert.deepEqual(wakeInputs, [full.subarray(0, 2000 * 32)]);
    assert.deepEqual(sttInputs, [full]);
    assert.deepEqual(h.actions, [{ action: 'skip' }]);
    assert.equal(h.cues, 0); assert.equal(h.session.phase, 'idle');
    const short = full.subarray(0, 900 * 32);
    await h.session.begin('bob')!.complete(short);
    assert.deepEqual(wakeInputs[1], short);
    assert.deepEqual(sttInputs, [full, short]);
    assert.deepEqual(h.actions, [{ action: 'skip' }, { action: 'skip' }]);
    assert.equal(h.cues, 0, 'a short addressed command is complete without prompting for another');
  } finally { h.session.disable(); }
});

test('silence, bare titles and conversation end the turn quietly and a new call still works', async () => {
  const feedbackCalls: string[] = [];
  const searches: string[] = [];
  const h = harness({},
    {
      playMusic: async message => { searches.push(message); return true; },
      feedback: async (key) => { feedbackCalls.push(key); },
    });
  try {
    for (const phrase of ['', '2 + 2', 'сколько будет два плюс два', 'что делаешь', 'как дела',
      'я поговорю с ним завтра', 'сегодня отличный день', 'Монеточка', 'Radio Ga Ga']) {
      h.setText('Муза'); await h.session.begin('alice')!.complete(audio);
      h.setText(phrase); await h.session.begin('alice')!.complete(audio);
      assert.equal(h.actions.length, 0);
      assert.equal(h.ducked, false); assert.equal(h.session.phase, 'idle', phrase);
    }
    assert.deepEqual(feedbackCalls, []); assert.deepEqual(searches, []);
    h.setText('Муза'); await h.session.begin('alice')!.complete(audio);
    h.setText('Следующий трек'); await h.session.begin('alice')!.complete(audio);
    assert.deepEqual(h.actions, [{ action: 'skip' }]);
  } finally { h.session.disable(); }
});

test('an exact multiword address works without a local model and leaves the song title intact', async () => {
  const searches: string[] = [];
  const h = harness({ supportsWake: () => false }, {
    playMusic: async message => { searches.push(message); return true; },
  });
  try {
    h.setName('Бот Ёнот');
    h.setText('Bot Enot, включи песню Radio Ga Ga (Live)!');
    await h.session.begin('alice')!.complete(audio);
    assert.deepEqual(searches, ['включи песню Radio Ga Ga (Live)!']);
    assert.equal(h.session.phase, 'idle'); assert.equal(h.cues, 0);
  } finally { h.session.disable(); }
});

test('an incomplete player request gives one reply and releases the speaker', async () => {
  const replies: string[] = [];
  const h = harness({}, { feedback: async key => { replies.push(key); } });
  try {
    await h.session.begin('alice')!.complete(audio);
    h.setText('Включи эту');
    const command = h.session.begin('alice')!;
    await command.complete(audio); await command.complete(audio);
    assert.deepEqual(replies, ['unknown']); assert.deepEqual(h.actions, []);
    assert.equal(h.session.phase, 'idle'); assert.equal(h.ducked, false);
    h.setText('как дела'); await h.session.begin('bob')!.complete(audio);
    assert.deepEqual(replies, ['unknown']); assert.equal(h.cues, 1);
  } finally { h.session.disable(); }
});

test('addressing the bot with a conversational fragment closes the turn without feedback', async () => {
  const feedbackCalls: string[] = [];
  const h = harness({
    detectWake: async () => ({ wake: true, probability: 0.99 }),
    classify: async () => ({ action: 'unknown', confidence: 1 }),
    transcribe: async () => 'Бот, 2 + 2',
  }, {
    playMusic: async () => { feedbackCalls.push('search'); return true; },
    feedback: async (key) => { feedbackCalls.push(key); },
  });
  try {
    h.setName('Бот');
    const longAudio = Buffer.alloc(2000 * 32);
    await h.session.begin('alice')!.complete(longAudio);
    assert.equal(h.actions.length, 0);
    assert.deepEqual(feedbackCalls, []);
    assert.equal(h.session.phase, 'idle'); assert.equal(h.cues, 0);
  } finally { h.session.disable(); }
});

test('the command detector executes a follow-up control without a second transcription', async () => {
  let cloudTranscribed = false;
  let commandDetected = false;
  const shortWake = Buffer.alloc(800 * 32);
  const h = harness({
    detectWake: async () => ({ wake: true, probability: 0.99 }),
    detectCommand: async () => {
      commandDetected = true;
      return { class: 'pause', action: 'pause', confidence: 0.98, matched: true };
    },
    transcribe: async (_pcm, _signal, command) => {
      if (command) cloudTranscribed = true;
      return command ? 'пауза' : 'Муза';
    },
  });
  try {
    await h.session.begin('alice')!.complete(shortWake);
    assert.equal(h.session.phase, 'awaiting');
    await h.session.begin('alice')!.complete(audio);
    assert.equal(commandDetected, true, 'command detector was invoked');
    assert.equal(cloudTranscribed, false, 'the command did not use STT');
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
    transcribe: async (_pcm, _signal, command) => {
      if (command) cloudTranscribed = true;
      return command ? 'включи Linkin Park Numb' : 'Муза';
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


