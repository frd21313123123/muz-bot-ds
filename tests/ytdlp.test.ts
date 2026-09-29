import assert from 'node:assert/strict';
import test from 'node:test';
import { YtdlpClient, type VideoInfo } from '../src/utils/ytdlp.js';

test('radio recommendations skip the current video and duplicate entries', async () => {
  const calls: string[][] = [];
  class StubYtdlp extends YtdlpClient {
    override async json(args: string[]): Promise<VideoInfo> {
      calls.push(args);
      return { entries: [
        { id: 'aaaaaaaaaaa', title: 'Current' },
        { id: 'bbbbbbbbbbb', title: 'Related' },
        { id: 'bbbbbbbbbbb', title: 'Duplicate' },
        { id: 'invalid', title: 'Invalid' },
        { id: 'ccccccccccc', title: 'Another' },
      ] };
    }
  }
  const ytdlp = new StubYtdlp({ command: 'unused', prefix: [] });
  const related = await ytdlp.related('aaaaaaaaaaa', 5);
  assert.deepEqual(related.map((item) => item.videoId), ['bbbbbbbbbbb', 'ccccccccccc']);
  assert.ok(related.every((item) => item.isAutoplay && item.requestedBy === '🤖 Бесконечное'));
  assert.deepEqual(calls[0], [
    '--flat-playlist', '--playlist-items', '2:6',
    'https://www.youtube.com/watch?v=aaaaaaaaaaa&list=RDaaaaaaaaaaa',
  ]);
  assert.deepEqual(await ytdlp.related('bad'), []);
  assert.equal(calls.length, 1);
});
