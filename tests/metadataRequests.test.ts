import assert from 'node:assert/strict';
import test from 'node:test';
import { MetadataRequests } from '../src/utils/MetadataRequests.js';
import { YtdlpClient, type VideoInfo } from '../src/utils/ytdlp.js';

const flush = async (): Promise<void> => { for (let i = 0; i < 8; i++) await Promise.resolve(); };
function gate<T>() { let resolve!: (value: T) => void; const promise = new Promise<T>(finish => { resolve = finish; }); return { promise, resolve }; }

test('shared metadata has independent cancellation and stops when the final consumer leaves', async () => {
  const requests = new MetadataRequests<number>();
  const result = gate<number>(); let calls = 0, sharedSignal: AbortSignal | undefined;
  const work = async (signal: AbortSignal) => { calls++; sharedSignal = signal; return result.promise; };
  const first = new AbortController(), second = new AbortController();
  const cancelled = assert.rejects(requests.run('same', work, { signal: first.signal }));
  const remaining = requests.run('same', work, { signal: second.signal });
  await flush(); first.abort(); await cancelled;
  assert.equal(calls, 1); assert.equal(sharedSignal?.aborted, false);
  result.resolve(7); assert.equal(await remaining, 7); await flush();
  const final = new AbortController();
  const abandoned = assert.rejects(requests.run('last', async signal => {
    sharedSignal = signal;
    return new Promise<number>((_resolve, reject) => signal.addEventListener('abort', () => reject(new Error('cancelled')), { once: true }));
  }, { signal: final.signal }));
  await flush(); final.abort(); await abandoned; await flush();
  assert.equal(sharedSignal?.aborted, true);
  assert.equal(await requests.run('last', async () => 9), 9, 'aborted jobs cannot be reused');
});

test('metadata reserves foreground capacity, prioritizes users and removes cancelled waiting jobs', async () => {
  const requests = new MetadataRequests<number>();
  const bg = gate<number>(), fg = gate<number>(), user = gate<number>();
  const starts: string[] = []; let active = 0, peak = 0, background = 0, peakBackground = 0;
  const task = (name: string, pending: Promise<number>, isBackground: boolean) => async () => {
    starts.push(name); active++; peak = Math.max(peak, active);
    if (isBackground) { background++; peakBackground = Math.max(peakBackground, background); }
    try { return await pending; } finally { active--; if (isBackground) background--; }
  };
  const a = requests.run('bg', task('bg', bg.promise, true), { priority: 'background' });
  const b = requests.run('bg2', task('bg2', Promise.resolve(2), true), { priority: 'background' });
  await flush(); assert.deepEqual(starts, ['bg']);
  const c = requests.run('fg', task('fg', fg.promise, false));
  await flush(); assert.deepEqual(starts, ['bg', 'fg']);
  const cancelled = new AbortController();
  const abandoned = assert.rejects(requests.run('abandoned', async () => { assert.fail('cancelled job ran'); }, { signal: cancelled.signal }));
  cancelled.abort(); await abandoned;
  const d = requests.run('user', task('user', user.promise, false));
  fg.resolve(3); await c; await flush(); assert.deepEqual(starts, ['bg', 'fg', 'user']);
  bg.resolve(1); user.resolve(4); await Promise.all([a, b, d]); await flush();
  assert.equal(peak, 2); assert.equal(peakBackground, 1);
});

test('video cache deduplicates cold requests, prunes expiry and drops unused yt-dlp fields', async context => {
  let calls = 0, now = 0;
  context.mock.method(Date, 'now', () => now);
  class Stub extends YtdlpClient {
    override async json(): Promise<VideoInfo> { calls++; return { id: 'aaaaaaaaaaa', title: 'Song', duration: 60, categories: ['Music'],
      formats: new Array(100).fill({ url: 'private-url' }) } as VideoInfo; }
  }
  const client = new Stub({ command: 'unused', prefix: [] });
  const [one, two] = await Promise.all([client.video('url'), client.video('url')]);
  assert.equal(calls, 1); assert.deepEqual(one, two); assert.equal('formats' in one, false);
  one.categories?.push('Changed');
  assert.deepEqual((await client.video('url')).categories, ['Music']);
  now = 30 * 60_000; await client.video('new');
  assert.equal(Reflect.get(client, 'videoCache').has('url'), false);
  await client.video('url'); assert.equal(calls, 3);
  const single = await client.video('single'); single.categories?.push('Mutated');
  assert.deepEqual((await client.video('single')).categories, ['Music']);
});

test('autoplay excludes history before metadata verification and caps newly verified songs', async () => {
  const checked: string[] = [];
  class Stub extends YtdlpClient {
    override async json(): Promise<VideoInfo> { return { entries: ['aaaaaaaaaaa', 'bbbbbbbbbbb', 'ccccccccccc', 'ddddddddddd', 'eeeeeeeeeee']
      .map(id => ({ id, title: 'Song' })) }; }
    override async video(url: string): Promise<VideoInfo> { const id = new URL(url).searchParams.get('v')!; checked.push(id); return { id, title: 'Song', categories: ['Music'] }; }
  }
  const client = new Stub({ command: 'unused', prefix: [] });
  const tracks = await client.related('aaaaaaaaaaa', 3, { excludeIds: new Set(['bbbbbbbbbbb']) });
  assert.deepEqual(checked, ['ccccccccccc', 'ddddddddddd', 'eeeeeeeeeee']);
  assert.equal(tracks.length, 3);
});
