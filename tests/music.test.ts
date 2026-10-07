import assert from 'node:assert/strict';
import test from 'node:test';
import { extractMusicRequest, resolveMusicRequest, musicState, musicVariant, type MusicBackend, type MusicDiagnostic, type MusicMetadata } from '../src/voice/music.js';

const entries = [
  { id: 'aaaaaaaaaaa', title: 'Linkin Park - Numb', channel: 'Linkin Park', duration: 180 },
  { id: 'bbbbbbbbbbb', title: 'Linkin Park - Numb Live in Texas', artist: 'Linkin Park', duration: 200 },
];
const deferredRoute = async () => ({ next_tool: 'youtube_music_search' as const, query_source: 'message' as const,
  search_result_policy: 'play_first_result' as const, defer_result_policy: true, confidence: 1 });
function harness(backendOverrides: Partial<MusicBackend> = {}, metadataOverrides: Partial<MusicMetadata> = {}) {
  const calls: string[] = [];
  const events: MusicDiagnostic[] = [];
  const controller = new AbortController();
  const backend: MusicBackend = {
    decideMusic: async (state) => {
      assert.equal(state.selected_track, null);
      calls.push('route');
      return { next_tool: state.message.includes('https://') ? 'direct_youtube_video' : 'youtube_music_search',
        query_source: 'message', search_result_policy: state.message.includes('live') ? 'rerank_results' : 'play_first_result', confidence: 1 };
    },
    rerankMusic: async (query, candidates) => {
      calls.push('rerank'); assert.equal(query, 'Numb live');
      assert.equal(candidates[1]?.index, 1); assert.equal(candidates[1]?.artist, 'Linkin Park');
      return { best_track: 1, confidence: 0.99 };
    }, ...backendOverrides,
  };
  const metadata: MusicMetadata = {
    searchCandidates: async (query, limit, signal) => {
      calls.push(`search:${query}`); assert.equal(limit, 5); assert.equal(signal, controller.signal); return entries;
    },
    video: async (url) => { calls.push(`video:${url}`); return entries[0]!; }, ...metadataOverrides,
  };
  return { calls, events, controller, resolve: (message: string) => resolveMusicRequest(message, 'Alice', backend, metadata, controller.signal, (event) => events.push(event)) };
}

test('music extraction keeps title punctuation, conjunctions, artist and version', () => {
  for (const verb of ['Включи', 'Поставь', 'Сыграй', 'Воспроизведи']) {
    assert.deepEqual(extractMusicRequest(`${verb} песню Linkin Park — Numb/Encore live`), { kind: 'search', query: 'Linkin Park — Numb/Encore live' });
  }
  assert.equal(extractMusicRequest('включи Numb и Encore')?.query, 'Numb и Encore');
  assert.equal(extractMusicRequest('включи песню Не отпускай')?.query, 'Не отпускай');
  for (const text of ['не включи Numb', 'включи', 'включи песню', 'включи эту', 'включи этот трек',
    'поставь на паузу', 'включи музыку', 'включи обратно', 'включи Numb и поставь на паузу',
    'включи Numb потом пауза', 'включи Numb или включи Sonne', 'включи Numb и не останавливай музыку',
    'включи Numb и громче']) assert.equal(extractMusicRequest(text), null, text);
});

test('direct music links select one video even with an ordinary playlist', () => {
  assert.deepEqual(extractMusicRequest('включи https://music.youtube.com/watch?v=aaaaaaaaaaa&list=PLfoo'),
    { kind: 'video', query: 'https://www.youtube.com/watch?v=aaaaaaaaaaa' });
  assert.equal(extractMusicRequest('включи https://youtu.be/aaaaaaaaaaa')?.kind, 'video');
  for (const url of ['https://example.com/watch?v=aaaaaaaaaaa', 'https://youtube.com/playlist?list=PLfoo',
    'https://youtube.com/watch?v=bad', 'https://youtube.com.evil.test/watch?v=aaaaaaaaaaa']) assert.equal(extractMusicRequest(`включи ${url}`), null);
});

test('the command window accepts arbitrary artists, titles and descriptions like YouTube search', async () => {
  for (const [message, query] of [
    ['Включи Монеточку', 'Монеточку'], ['Вот, включи Монеточку', 'Монеточку'],
    ['Ключи монеточку.', 'монеточку.'],
    ['Монеточка', 'Монеточка'], ['Давай Монеточку', 'Монеточку'],
    ['песни Монеточки', 'песни Монеточки'], ['Хочу послушать Земфиру', 'Земфиру'],
    ['Можешь включить Монеточку', 'Монеточку'], ['запусти Кино', 'Кино'],
    ['Включи Всё', 'Всё'], ['Numb и Encore', 'Numb и Encore'],
    ['Включи музыку Земфиры', 'музыку Земфиры'],
  ]) {
    assert.deepEqual(extractMusicRequest(message!, true), { kind: 'search', query });
    const h = harness({ decideMusic: async () => { assert.fail('Artist queries must never require model approval'); } });
    assert.equal((await h.resolve(message!))?.videoId, 'aaaaaaaaaaa');
    assert.deepEqual(h.calls, [`search:${query}`]);
  }
  assert.equal(extractMusicRequest('Монеточка'), null, 'bare names are only accepted in the command window');
});

test('bare query support preserves control priority and rejects missing context, negation and multiple actions', () => {
  for (const message of ['Пауза', 'Поставь на паузу', 'Поставь паузу', 'Поставь громкость 50',
    'Сделай музыку громче', 'Продолжай', 'Следующий трек', 'Останови музыку', 'Громкость 70',
    'Не включай Монеточку', 'Вот не включи Кино', 'Я сказал включи Кино', 'Как включить музыку?',
    'Включи эту', 'эту', 'Включи то', 'Включи', 'Включи музыку', 'Включи песню из',
    'Включи Кино и поставь паузу', 'Включи Кино, поставь паузу', 'Кино потом пауза',
    'Включи Кино или включи Монеточку', 'Включи Кино и запусти Монеточку',
    'Включи Кино, пауза', 'Ключи Кино и проиграй Монеточку']) {
    assert.equal(extractMusicRequest(message, true), null, message);
  }
});

test('strict filters reject math expressions, calculations, assistant commands, and device targets', () => {
  for (const message of [
    '2 + 2', '2+2', '2 - 2', '5 * 5', '10 / 2', 'два плюс два',
    'Бот 2 + 2', 'Бот, 2 + 2', 'бот два плюс два',
    'включи 2 + 2', 'поставь 2 + 2', 'сыграй 2 + 2', 'включи два плюс два', 'поставь два плюс два',
    'сколько будет 2 + 2', 'сколько будет два плюс два', 'посчитай 2 плюс 2', 'вычисли 5 на 5',
    'включи свет', 'включи микрофон', 'включи звук', 'включи камеру', 'включи демонстрацию экрана',
    'поставь таймер', 'поставь будильник', 'поставь чайник', 'сыграй в города', 'сыграй в шахматы',
    '123', 'раз два три', 'сорок два',
    'что делаешь', 'как дела', 'кто ты', 'где ты', 'сколько времени', 'какая погода',
    'погода в Москве', 'анекдот', 'новости', 'переведи на английский',
    'привет', 'здравствуйте', 'до свидания', 'пока', 'спасибо',
    'ты тут', 'ты меня слышишь', 'проверка', 'проверка микрофона', 'тест связи',
  ]) {
    assert.equal(extractMusicRequest(message, true), null, message);
    assert.equal(extractMusicRequest(message, false), null, message);
  }
  assert.deepEqual(extractMusicRequest('включи песню Свет'), { kind: 'search', query: 'Свет' });
  assert.deepEqual(extractMusicRequest('поставь песню Таймер'), { kind: 'search', query: 'Таймер' });
});

test('conversational and ASR command forms retain a song-from request without accepting negation or context', () => {
  for (const text of ['включи песню из Лунтика', 'Вот включи песню из Лунтика',
    'Ну, включи песню из Лунтика', 'Пожалуйста, включи песню из Лунтика',
    'Включи, песню из Лунтика', 'Включить песню из Лунтика', 'Включай песню из Лунтика']) {
    assert.deepEqual(extractMusicRequest(text), { kind: 'search', query: 'песня из Лунтика' }, text);
  }
  assert.equal(extractMusicRequest('сыграй трек из Шрека')?.query, 'трек из Шрека');
  assert.equal(extractMusicRequest('Вот включи песню Linkin Park — Numb/Encore live')?.query, 'Linkin Park — Numb/Encore live');
  assert.equal(extractMusicRequest('Включить Numb и Encore')?.query, 'Numb и Encore');
  for (const text of ['Вот не включи Numb', 'Не надо, включи Numb', 'Я сказал включи Numb',
    'Если можешь включи Numb', 'Ну включи эту', 'Включить песню', 'Включи песню из',
    'Вот поставь на паузу', 'Вот включи музыку', 'Ну включи Numb и поставь на паузу',
    'Включить Numb или сыграть Sonne']) assert.equal(extractMusicRequest(text), null, text);
});

test('plain and version requests preserve search order without calling a selector', async () => {
  const plain = harness();
  assert.equal((await plain.resolve('включи Numb'))?.videoId, 'aaaaaaaaaaa');
  assert.deepEqual(plain.calls, ['search:Numb']);
  const live = harness();
  const track = await live.resolve('включи Numb live');
  assert.equal(track?.videoId, 'aaaaaaaaaaa'); assert.equal(track?.requestedBy, 'Alice');
  assert.deepEqual(live.calls, ['search:Numb live']);
  assert.equal(JSON.stringify(live.events).includes('Numb'), false);
});

test('a valid first YouTube Music song is kept even when its title resembles spoken content', async () => {
  const h = harness({}, { searchCandidates: async () => [
    { id: 'aaaaaaaaaaa', title: 'Interview', artist: 'Artist', duration: 180 }, entries[1]!,
  ] });
  assert.equal((await h.resolve('включи песню Interview'))?.videoId, 'aaaaaaaaaaa');
});

test('v3 state contains the parsed original query and canonical video URL', () => {
  const request = extractMusicRequest('включи песню Numb концертная версия')!;
  assert.deepEqual(musicState('включи песню Numb концертная версия', request).parsed_request,
    { kind: 'play', search_query: 'Numb концертная версия', request_variant: 'live' });
  const video = extractMusicRequest('включи https://music.youtube.com/watch?v=aaaaaaaaaaa&list=PLfoo')!;
  assert.deepEqual(musicState('original', video).parsed_request,
    { kind: 'play', search_query: null, url: 'https://www.youtube.com/watch?v=aaaaaaaaaaa', request_variant: null });
  assert.equal(musicVariant('Numb и Encore'), null);
  assert.equal(musicVariant('Numb акустическая версия'), 'acoustic');
});

test('v3 candidate policy and reranking are bypassed for every version qualifier', async () => {
  for (const query of ['Numb', 'Numb live', 'Numb remix', 'Numb cover', 'Numb acoustic',
    'Numb instrumental', 'Numb концертная версия', 'Numb ремикс', 'Numb кавер']) {
    let policyCalls = 0;
    const h = harness({ decideMusic: deferredRoute,
      decideMusicResults: async () => { policyCalls++; throw new Error('Must not call'); },
      rerankMusic: async () => { assert.fail('Must not compare candidates'); } });
    assert.equal((await h.resolve(`включи ${query}`))?.videoId, 'aaaaaaaaaaa');
    assert.equal(policyCalls, 0);
    assert.equal(h.calls.includes('rerank'), false);
    assert.deepEqual(h.events.filter((event) => ['search', 'policy'].includes(event.stage)).map((event) => event.stage),
      ['search']);
  }
});

test('Moscow never sleep and speech spelling variants cannot be vetoed by a second plain-title decision', async () => {
  for (const query of ['Moscow never sleep', 'Moscow Never Sleeps', 'москоу невер слип', 'Moscow never slip.']) {
    let policyCalls = 0;
    const h = harness({ decideMusic: deferredRoute,
      decideMusicResults: async () => { policyCalls++; return { search_result_policy: 'no_result', confidence: 0.9966 }; },
      rerankMusic: async () => { assert.fail('Plain titles must bypass reranking'); } },
    { searchCandidates: async actualQuery => {
      assert.equal(actualQuery, query);
      return [{ id: 'aaaaaaaaaaa', title: 'DJ SMASH — MOSCOW NEVER SLEEPS', channel: 'DJ SMASH', duration: 210 }];
    } });
    assert.equal((await h.resolve(`Включи ${query}`))?.title, 'DJ SMASH — MOSCOW NEVER SLEEPS');
    assert.equal(policyCalls, 0, 'An ordinary title must not be rejected after a successful search');
    assert.equal(h.events.some(event => event.stage === 'fallback'), false);
  }
  const original = harness({ decideMusic: async () => ({ next_tool: 'youtube_music_search', query_source: 'message',
    search_result_policy: 'rerank_results', confidence: 1 }) });
  assert.equal((await original.resolve('Включи Moscow never sleep'))?.videoId, 'aaaaaaaaaaa');
  assert.equal(original.calls.includes('rerank'), false);
});

test('obsolete candidate decisions cannot veto or replace the first search result', async () => {
  for (const confidence of [0.99, 0.59]) {
    const noResults = harness({ decideMusic: deferredRoute,
      decideMusicResults: async () => ({ search_result_policy: 'no_result', confidence }) });
    assert.equal((await noResults.resolve('включи Numb live'))?.videoId, 'aaaaaaaaaaa');
    assert.equal(noResults.events.some(event => event.reason === 'no_result'), false);
    const noMatch = harness({ rerankMusic: async () => ({ best_track: -1, no_match: true, confidence }) });
    assert.equal((await noMatch.resolve('включи Numb live'))?.videoId, 'aaaaaaaaaaa');
    assert.equal(noMatch.events.some(event => event.reason === 'no_match'), false);
  }
  const failed = harness({ decideMusic: deferredRoute, decideMusicResults: async () => { throw new Error('worker failed'); } });
  assert.equal((await failed.resolve('включи Numb live'))?.videoId, 'aaaaaaaaaaa');
});

test('direct route never searches or reranks', async () => {
  const h = harness();
  assert.equal((await h.resolve('включи https://music.youtube.com/watch?v=aaaaaaaaaaa&list=PLfoo'))?.videoId, 'aaaaaaaaaaa');
  assert.deepEqual(h.calls, ['video:https://www.youtube.com/watch?v=aaaaaaaaaaa']);
});

test('a parsed search bypasses unavailable or rejecting model routing', async () => {
  const fallback = harness({ decideMusic: async () => { throw new Error('unavailable'); } });
  assert.equal((await fallback.resolve('включи Numb live'))?.videoId, 'aaaaaaaaaaa');
  assert.deepEqual(fallback.calls, ['search:Numb live']);
  assert.equal(fallback.events.some((event) => event.stage === 'fallback'), false);
  for (const [next_tool, confidence] of [['unknown', 1], ['player_control', 1], ['youtube_music_search', 0.1], ['direct_youtube_video', 1]] as const) {
    const h = harness({ decideMusic: async () => ({ next_tool, confidence, query_source: 'message', search_result_policy: 'play_first_result' }) });
    assert.equal((await h.resolve('включи Numb'))?.videoId, 'aaaaaaaaaaa'); assert.deepEqual(h.calls, ['search:Numb']);
  }
});

test('obsolete invalid indices and candidate inference errors never affect search order', async () => {
  for (const selection of [{ best_track: -1, confidence: 1 }, { best_track: 9, confidence: 1 },
    { best_track: 0.5, confidence: 1 }, { best_track: 1, confidence: 0.59 },
    { best_track: 1, confidence: NaN }, { best_track: 1, confidence: 2 }]) {
    const h = harness({ rerankMusic: async () => selection });
    assert.equal((await h.resolve('включи Numb live'))?.videoId, 'aaaaaaaaaaa');
    assert.equal(h.events.some((event) => event.reason === 'selection'), false);
  }
  const h = harness({ rerankMusic: async () => { throw new Error('unavailable'); } });
  assert.equal((await h.resolve('включи Numb live'))?.videoId, 'aaaaaaaaaaa');
});

test('empty or failed searches do not create a track; invalid entries and duplicates are removed', async () => {
  for (const searchCandidates of [async () => [], async () => { throw new Error('network'); }]) {
    assert.equal(await harness({}, { searchCandidates }).resolve('включи Numb'), null);
  }
  const h = harness({}, { searchCandidates: async () => [{ id: 'bad' }, entries[0]!, entries[0]!, entries[1]!] });
  assert.equal((await h.resolve('включи Numb live'))?.videoId, 'aaaaaaaaaaa');
  assert.equal(h.events.find((event) => event.stage === 'search')?.candidates, 2);
});

test('already cancelled requests and abort during search never return a track', async () => {
  for (const stage of ['before', 'search']) {
    const h = harness(
      {},
      stage === 'search' ? { searchCandidates: async () => { h.controller.abort(); return entries; } } : {},
    );
    if (stage === 'before') h.controller.abort();
    await assert.rejects(h.resolve('включи Numb live'));
    assert.equal(h.events.some((event) => event.stage === 'fallback'), false);
  }
});
