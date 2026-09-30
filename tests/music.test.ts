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

test('plain requests take first result and version requests rerank', async () => {
  const plain = harness();
  assert.equal((await plain.resolve('включи Numb'))?.videoId, 'aaaaaaaaaaa');
  assert.deepEqual(plain.calls, ['route', 'search:Numb']);
  const live = harness();
  const track = await live.resolve('включи Numb live');
  assert.equal(track?.videoId, 'bbbbbbbbbbb'); assert.equal(track?.requestedBy, 'Alice');
  assert.deepEqual(live.calls, ['route', 'search:Numb live', 'rerank']);
  assert.equal(JSON.stringify(live.events).includes('Numb'), false);
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

test('v3 policy receives real candidates only for an explicit version; plain titles take the first', async () => {
  for (const query of ['Numb', 'Numb live']) {
    let policyCalls = 0;
    const h = harness({ decideMusic: deferredRoute,
      decideMusicResults: async (actualQuery, candidates) => {
        policyCalls++; assert.equal(actualQuery, query); assert.equal(candidates.length, 2);
        assert.equal(candidates[1]!.title, entries[1]!.title);
        return { search_result_policy: 'rerank_results', confidence: 0.99 };
      } });
    assert.equal((await h.resolve(`включи ${query}`))?.videoId, query === 'Numb' ? 'aaaaaaaaaaa' : 'bbbbbbbbbbb');
    assert.equal(policyCalls, query === 'Numb' ? 0 : 1);
    assert.equal(h.calls.includes('rerank'), query !== 'Numb');
    assert.deepEqual(h.events.filter((event) => ['search', 'policy'].includes(event.stage)).map((event) => event.stage),
      query === 'Numb' ? ['search'] : ['search', 'policy']);
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

test('v3 explicit no-result and no-match decisions abstain; weak or failed decisions fall back', async () => {
  for (const confidence of [0.99, 0.59]) {
    const noResults = harness({ decideMusic: deferredRoute,
      decideMusicResults: async () => ({ search_result_policy: 'no_result', confidence }) });
    assert.equal((await noResults.resolve('включи Numb live'))?.videoId ?? null, confidence >= 0.6 ? null : 'aaaaaaaaaaa');
    assert.equal(noResults.events.some(event => event.reason === 'no_result'), confidence >= 0.6);
    const noMatch = harness({ rerankMusic: async () => ({ best_track: -1, no_match: true, confidence }) });
    assert.equal((await noMatch.resolve('включи Numb live'))?.videoId ?? null, confidence >= 0.6 ? null : 'aaaaaaaaaaa');
    assert.equal(noMatch.events.some(event => event.reason === 'no_match'), confidence >= 0.6);
  }
  const failed = harness({ decideMusic: deferredRoute, decideMusicResults: async () => { throw new Error('worker failed'); } });
  assert.equal((await failed.resolve('включи Numb live'))?.videoId, 'aaaaaaaaaaa');
});

test('abort during v3 policy never falls back or returns a track', async () => {
  const h = harness({ decideMusic: deferredRoute, decideMusicResults: async () => {
    h.controller.abort(); throw new Error('cancelled');
  } });
  await assert.rejects(h.resolve('включи Numb live'));
  assert.equal(h.events.some((event) => event.stage === 'fallback'), false);
});

test('direct route never searches or reranks', async () => {
  const h = harness();
  assert.equal((await h.resolve('включи https://music.youtube.com/watch?v=aaaaaaaaaaa&list=PLfoo'))?.videoId, 'aaaaaaaaaaa');
  assert.deepEqual(h.calls, ['route', 'video:https://www.youtube.com/watch?v=aaaaaaaaaaa']);
});

test('failed routing falls back but explicit unknown and weak decisions abstain', async () => {
  const fallback = harness({ decideMusic: async () => { throw new Error('unavailable'); } });
  assert.equal((await fallback.resolve('включи Numb live'))?.videoId, 'aaaaaaaaaaa');
  assert.deepEqual(fallback.calls, ['search:Numb live']);
  assert.ok(fallback.events.some((event) => event.reason === 'model'));
  for (const [next_tool, confidence] of [['unknown', 1], ['player_control', 1], ['youtube_music_search', 0.1], ['direct_youtube_video', 1]] as const) {
    const h = harness({ decideMusic: async () => ({ next_tool, confidence, query_source: 'message', search_result_policy: 'play_first_result' }) });
    assert.equal(await h.resolve('включи Numb'), null); assert.deepEqual(h.calls, []);
  }
});

test('invalid reranking and inference errors fall back to first candidate', async () => {
  for (const selection of [{ best_track: -1, confidence: 1 }, { best_track: 9, confidence: 1 },
    { best_track: 0.5, confidence: 1 }, { best_track: 1, confidence: 0.59 },
    { best_track: 1, confidence: NaN }, { best_track: 1, confidence: 2 }]) {
    const h = harness({ rerankMusic: async () => selection });
    assert.equal((await h.resolve('включи Numb live'))?.videoId, 'aaaaaaaaaaa');
    assert.ok(h.events.some((event) => event.reason === 'selection'));
  }
  const h = harness({ rerankMusic: async () => { throw new Error('unavailable'); } });
  assert.equal((await h.resolve('включи Numb live'))?.videoId, 'aaaaaaaaaaa');
});

test('empty or failed searches do not create a track; invalid entries and duplicates are removed', async () => {
  for (const searchCandidates of [async () => [], async () => { throw new Error('network'); }]) {
    assert.equal(await harness({}, { searchCandidates }).resolve('включи Numb'), null);
  }
  const h = harness({}, { searchCandidates: async () => [{ id: 'bad' }, entries[0]!, entries[0]!, entries[1]!] });
  assert.equal((await h.resolve('включи Numb live'))?.videoId, 'bbbbbbbbbbb');
  assert.equal(h.events.find((event) => event.stage === 'search')?.candidates, 2);
});

test('aborts in routing, search and reranking never fall back or return a track', async () => {
  for (const stage of ['route', 'search', 'rerank']) {
    const h = harness(
      stage === 'route' ? { decideMusic: async () => { h.controller.abort(); throw new Error('cancel'); } }
        : stage === 'rerank' ? { rerankMusic: async () => { h.controller.abort(); return { best_track: 1, confidence: 1 }; } } : {},
      stage === 'search' ? { searchCandidates: async () => { h.controller.abort(); return entries; } } : {},
    );
    await assert.rejects(h.resolve('включи Numb live'));
    assert.equal(h.events.some((event) => event.stage === 'fallback'), false);
  }
});
