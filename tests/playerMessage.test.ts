import assert from 'node:assert/strict';
import test from 'node:test';
import type { MusicClient } from '../src/types.js';
import { GuildQueue } from '../src/utils/GuildQueue.js';
import type { Message } from 'discord.js';

test('player skips identical paused payloads and collapses state changes during an edit', async () => {
  const queue = new GuildQueue('message', { queues: new Map() } as unknown as MusicClient);
  let edits = 0, finish: (() => void) | undefined;
  const message = { edit: async () => { edits++; if (edits === 2) await new Promise<void>(resolve => { finish = resolve; }); }, delete: async () => {} } as unknown as Message;
  try {
    queue.currentTrack = { videoId: 'aaaaaaaaaaa', url: 'https://www.youtube.com/watch?v=aaaaaaaaaaa', title: 'Song',
      duration: '3:00', requestedBy: 'Tester', thumbnail: null };
    Reflect.set(queue, 'pauseRequested', true);
    await queue.showPlayer(message, 'channel');
    await queue.playerMessage.update(); await queue.playerMessage.update(); assert.equal(edits, 1);
    queue.setVolume(100);
    await Promise.resolve();
    queue.setVolume(110); queue.setVolume(120); queue.setVolume(130);
    finish?.();
    await new Promise<void>(resolve => setImmediate(resolve));
    assert.equal(edits, 3, 'one in-flight edit and one final state edit');
    await queue.playerMessage.update(); assert.equal(edits, 3);
  } finally { finish?.(); await queue.stop(); }
});

test('a failed old edit cannot discard a replacement player message', async () => {
  const queue = new GuildQueue('message-replace', { queues: new Map() } as unknown as MusicClient);
  let edits = 0, newEdits = 0, fail: ((error: Error) => void) | undefined;
  const oldMessage = { edit: async () => {
    if (++edits === 2) await new Promise<void>((_resolve, reject) => { fail = reject; });
  }, delete: async () => {} } as unknown as Message;
  const newMessage = { edit: async () => { newEdits++; }, delete: async () => {} } as unknown as Message;
  try {
    queue.currentTrack = { videoId: 'aaaaaaaaaaa', url: 'https://www.youtube.com/watch?v=aaaaaaaaaaa', title: 'Song',
      duration: '3:00', requestedBy: 'Tester', thumbnail: null };
    Reflect.set(queue, 'pauseRequested', true);
    await queue.showPlayer(oldMessage, 'channel');
    await queue.playerMessage.update();
    queue.setVolume(100);
    await Promise.resolve();
    assert.ok(fail);
    await queue.showPlayer(newMessage, 'channel');
    await queue.playerMessage.update();
    fail?.(new Error('old message deleted'));
    await new Promise<void>(resolve => setImmediate(resolve));
    assert.equal(newEdits, 1);
    await queue.playerMessage.update();
    assert.equal(newEdits, 1);
  } finally { fail?.(new Error('cancel')); await queue.stop(); }
});
