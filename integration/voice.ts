import 'dotenv/config';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { VoiceRuntime } from '../src/voice/runtime.js';
import { contextualCommand, isWakePhrase, normalizeSpeech, validateIntent, type VoiceAction } from '../src/voice/intents.js';
import { requireFfmpeg } from '../src/utils/stream.js';
import { extractMusicRequest, musicState, resolveMusicRequest, validConfidence, type MusicDecision } from '../src/voice/music.js';

const cases: [string, VoiceAction, boolean?][] = [
  ['Следующий трек', 'skip'], ['Пропусти текущую музыку', 'skip'],
  ['Поставь на паузу', 'pause'], ['Приостанови музыку', 'pause'],
  ['Продолжи музыку', 'resume'], ['Сними с паузы', 'resume'],
  ['Останови музыку', 'stop'], ['Отключись от канала', 'stop'],
  ['Громкость пятьдесят процентов', 'volume_set'], ['Установи громкость на 150 процентов', 'volume_set'],
  ['Сделай громче', 'volume_up'], ['Сделай тише', 'volume_down'],
  ['Включи песню', 'unknown'], ['Добавь трек в очередь', 'unknown'],
  ['Не останавливай музыку', 'unknown'], ['Следующий трек и пауза', 'unknown'],
  ['Как погода сегодня', 'unknown'], ['Расскажи анекдот', 'unknown'],
  ['Мне нравится эта музыка', 'unknown'], ['Музыка на паузе', 'unknown'],
  ['Как сделать музыку громче?', 'unknown'], ['Песня называется Стоп', 'unknown'],
  ['Привет', 'unknown'], ['Спасибо', 'unknown'], ['Выключи свет', 'unknown'],
  ['Сделай паузу и продолжи', 'unknown'], ['Громкость 900', 'unknown'],
  ['Покажи очередь', 'unknown'], ['Расскажи что ты умеешь', 'unknown'],
  ['Поставь Metallica', 'unknown'],
  ['Включи', 'resume', true], ['Включи музыку', 'resume', true],
  ['Включи обратно', 'resume', true], ['Включи музыку снова', 'resume', true],
  ['Включи', 'unknown', false], ['Включи музыку', 'unknown', false],
  ['Включи песню', 'unknown', true], ['Включи Metallica', 'unknown', true],
  ['Не включай музыку', 'unknown', true], ['Включи и следующий трек', 'unknown', true],
  ['сделай громче.', 'volume_up'], ['сделай тише.', 'volume_down'],
  ['на паузу', 'pause'], ['включи.', 'resume', true],
  ['Продолжай', 'resume', true], ['Продолжай музыку', 'resume', true],
  ['Возобнови', 'resume', true], ['Продолжай играть', 'resume', true],
  ['Не продолжай музыку', 'unknown', true], ['Продолжай и следующий трек', 'unknown', true],
  ['Продолжай читать', 'unknown', true], ['Продолжай рассказ', 'unknown', true],
  ['Поставь паузу', 'pause'], ['Вот, поставь на паузу', 'pause'],
];

async function decode(file: string): Promise<Buffer> {
  const ffmpeg = spawn(requireFfmpeg(), ['-nostdin', '-hide_banner', '-loglevel', 'error',
    '-i', file, '-t', '15', '-ar', '16000', '-ac', '1', '-f', 's16le', 'pipe:1'], { windowsHide: true });
  const chunks: Buffer[] = [];
  ffmpeg.stdout.on('data', (chunk: Buffer) => chunks.push(chunk));
  ffmpeg.stderr.resume();
  await new Promise<void>((resolve, reject) => {
    ffmpeg.once('error', reject);
    ffmpeg.once('close', (code) => code === 0 ? resolve() : reject(new Error('Cannot decode fixture')));
  });
  return Buffer.concat(chunks);
}

const runtime = new VoiceRuntime();
try {
  assert.equal(await runtime.start(), true, 'Run npm run setup:voice first');
  console.log(`Laya checkpoint: ${runtime.modelName ?? 'default'}`);
  const signal = new AbortController().signal;
  let correct = 0;
  let wrongActions = 0;
  const timings: number[] = [];
  for (const [transcript, expected, paused] of cases) {
    const text = contextualCommand(transcript, paused ?? false);
    const start = performance.now();
    const decision = await runtime.classify(text, signal);
    const actual = validateIntent(text, decision).action;
    timings.push(performance.now() - start);
    if (actual === expected) correct++;
    else if (actual !== 'unknown') wrongActions++;
    console.log(`${expected}: ${actual} [model=${decision.action}, confidence=${decision.confidence.toFixed(3)}] (${Math.round(timings.at(-1)!)} ms)`);
  }
  timings.sort((a, b) => a - b);
  console.log(`Laya: ${correct}/${cases.length} correct; ${wrongActions} incorrect actions; median ${Math.round(timings[Math.floor(timings.length / 2)]!)} ms`);
  assert.equal(wrongActions, 0, 'Wrong playback actions need investigation before use');
  assert.ok(correct >= cases.length - 2, 'Too many supported commands rejected');
  const musicCases: [string, MusicDecision['next_tool'], MusicDecision['search_result_policy']][] = [
    ['включи Numb', 'youtube_music_search', 'play_first_result'],
    ['поставь песню Rammstein Sonne', 'youtube_music_search', 'play_first_result'],
    ['Включи Moscow never sleep', 'youtube_music_search', 'play_first_result'],
    ['Включи Moscow Never Sleeps', 'youtube_music_search', 'play_first_result'],
    ['включи песню из Лунтика', 'youtube_music_search', 'play_first_result'],
    ['Вот включи песню из Лунтика', 'youtube_music_search', 'play_first_result'],
    ['Включи, песню из Лунтика', 'youtube_music_search', 'play_first_result'],
    ['Включить песню из Лунтика', 'youtube_music_search', 'play_first_result'],
    ['включи Numb live', 'youtube_music_search', 'rerank_results'],
    ['сыграй Numb remix', 'youtube_music_search', 'rerank_results'],
    ['воспроизведи Numb cover', 'youtube_music_search', 'rerank_results'],
    ['включи Numb acoustic', 'youtube_music_search', 'rerank_results'],
    ['включи Numb instrumental', 'youtube_music_search', 'rerank_results'],
    ['включи Numb концертная версия', 'youtube_music_search', 'rerank_results'],
    ['включи Numb ремикс', 'youtube_music_search', 'rerank_results'],
    ['включи Numb кавер', 'youtube_music_search', 'rerank_results'],
    ['включи Numb акустическая версия', 'youtube_music_search', 'rerank_results'],
    ['включи Numb инструментальная версия', 'youtube_music_search', 'rerank_results'],
    ['включи https://music.youtube.com/watch?v=dQw4w9WgXcQ', 'direct_youtube_video', 'play_first_result'],
    ['включи https://music.youtube.com/watch?v=dQw4w9WgXcQ&list=RDdQw4w9WgXcQ', 'direct_youtube_video', 'play_first_result'],
  ];
  // Legacy trained contract smoke check. Production music search bypasses it.
  for (const [index, [message, route]] of musicCases.entries()) {
    const request = extractMusicRequest(message);
    assert.ok(request);
    const start = performance.now();
    const decision = await runtime.decideMusic(musicState(message, request), signal);
    console.log(`Music route ${index + 1}: ${decision.next_tool}, ${decision.search_result_policy}, confidence=${decision.confidence.toFixed(3)} (${Math.round(performance.now() - start)} ms)`);
    assert.equal(decision.next_tool, route, `Music route ${index + 1}`);
    assert.ok(validConfidence(decision.confidence), `Music confidence ${index + 1}`);
    // The trained routing head may still suggest reranking; the bot deliberately
    // ignores that field and always keeps the search order.
  }
  for (const message of ['Включи Moscow never sleep', 'Включи Moscow Never Sleeps',
    'Вот включи песню из Лунтика', 'Включи, песню из Лунтика', 'Включить песню из Лунтика',
    'включи Numb live', 'включи Numb remix', 'включи Numb кавер',
    'Включи Монеточку', 'Вот, включи Монеточку', 'Монеточка', 'Поставь Кино',
    'Хочу послушать Земфиру', 'песни Монеточки', 'Можешь включить Монеточку']) {
    const expectedQuery = message.includes('Лунтика') ? 'песня из Лунтика' : extractMusicRequest(message, true)!.query;
    const track = await resolveMusicRequest(message, 'Voice check', {
      decideMusic: async () => { assert.fail('Search queries must bypass model routing'); },
      decideMusicResults: async () => { assert.fail('Plain song requests must bypass result policy'); },
      rerankMusic: async () => { assert.fail('Plain song requests must bypass reranking'); },
    }, { searchCandidates: async (query) => {
      assert.equal(query, expectedQuery);
      return [{ id: 'aaaaaaaaaaa', title: 'First search result' }, { id: 'bbbbbbbbbbb', title: 'Other candidate' }];
    },
      video: async () => { assert.fail('Song title must use search'); } }, signal);
    assert.equal(track?.videoId, 'aaaaaaaaaaa');
    console.log('First search result pipeline: PASS');
  }
  // Optional local WAV fixtures: [{file, action}] or [{file, wakeName}]. Never print the transcript.
  const manifest = process.env.VOICE_TEST_FIXTURES;
  if (manifest) {
    const fixtures = JSON.parse(await readFile(manifest, 'utf8')) as { file: string; action?: VoiceAction; level?: number; paused?: boolean; wakeName?: string; shouldWake?: boolean; musicQuery?: string }[];
    let passedCount = 0;
    const asrTimings: number[] = [];
    for (const [i, fixture] of fixtures.entries()) {
      const start = performance.now();
      const transcript = await runtime.transcribe(await decode(path.resolve(path.dirname(manifest), fixture.file)), signal, !fixture.wakeName, fixture.wakeName);
      asrTimings.push(performance.now() - start);
      const text = contextualCommand(transcript, fixture.paused ?? false);
      const intent = fixture.wakeName || fixture.musicQuery ? null : validateIntent(text, await runtime.classify(text, signal));
      const music = fixture.musicQuery ? extractMusicRequest(text, true) : null;
      const passed = fixture.wakeName ? isWakePhrase(text, fixture.wakeName) === (fixture.shouldWake ?? true)
        : fixture.musicQuery ? music !== null && normalizeSpeech(music.query) === normalizeSpeech(fixture.musicQuery)
        : intent?.action === fixture.action && (fixture.level === undefined || (intent?.action === 'volume_set' && intent.level === fixture.level));
      if (passed) passedCount++;
      console.log(`Audio fixture ${i + 1}: ${passed ? 'PASS' : 'FAIL'} [action=${intent?.action ?? (fixture.musicQuery ? 'play' : 'wake')}] (${Math.round(performance.now() - start)} ms)`);
    }
    asrTimings.sort((a, b) => a - b);
    console.log(`Audio: ${passedCount}/${fixtures.length} correct; median ASR ${Math.round(asrTimings[Math.floor(asrTimings.length / 2)] ?? 0)} ms`);
    assert.equal(passedCount, fixtures.length, 'Audio fixtures failed');
  } else console.log('Whisper audio check skipped: set VOICE_TEST_FIXTURES to a local WAV manifest.');
} finally { runtime.close(); }
