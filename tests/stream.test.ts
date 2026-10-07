import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import test from 'node:test';
import { createProcessStream } from '../src/utils/stream.js';
import { terminateProcess } from '../src/utils/processes.js';

test('Windows teardown kills descendants that inherit stdout', { skip: process.platform !== 'win32' }, async () => {
  const source = spawn(process.execPath, ['-e', `
    const { spawn } = require('node:child_process');
    const descendant = spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000)'], { stdio: 'inherit', windowsHide: true });
    process.stdout.write(String(descendant.pid) + '\\n');
    setInterval(() => {}, 1000);
  `], { stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true });
  let pid: number | undefined;
  try {
    const data = await new Promise<Buffer>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('descendant did not start')), 5000);
      source.stdout.once('data', value => { clearTimeout(timer); resolve(value); });
      source.once('error', error => { clearTimeout(timer); reject(error); });
    });
    pid = Number(data.toString().trim()); assert.ok(Number.isInteger(pid) && pid > 0);
    await terminateProcess(source);
    assert.equal(source.stdout.destroyed, true);
    assert.throws(() => process.kill(pid!, 0), { code: 'ESRCH' });
  } finally {
    await terminateProcess(source);
    if (pid) try { process.kill(pid, 'SIGKILL'); } catch { /* Already stopped. */ }
  }
});

test('destroy ends both child processes', async () => {
  const source = spawn(process.execPath, ['-e', 'setInterval(() => process.stdout.write("x"), 10)'],
    { stdio: ['ignore', 'pipe', 'pipe'] });
  const sink = spawn(process.execPath, ['-e', 'process.stdin.pipe(process.stdout); setInterval(() => {}, 1000)'],
    { stdio: ['pipe', 'pipe', 'pipe'] });
  const media = createProcessStream(source, sink);
  const sourceClosed = once(source, 'close');
  const sinkClosed = once(sink, 'close');
  media.destroy();
  let timer: NodeJS.Timeout | undefined;
  await Promise.race([
    Promise.all([sourceClosed, sinkClosed]),
    new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('child processes survived destroy')), 5_000); }),
  ]).finally(() => { if (timer) clearTimeout(timer); });
  assert.equal(media.stream.destroyed, true);
});

test('nonzero source exit reports an error and closes the sink', async () => {
  const source = spawn(process.execPath, ['-e', 'process.stderr.write("test failure");process.exit(7)'],
    { stdio: ['ignore', 'pipe', 'pipe'] });
  const sink = spawn(process.execPath, ['-e', 'process.stdin.pipe(process.stdout);setInterval(() => {}, 1000)'],
    { stdio: ['pipe', 'pipe', 'pipe'] });
  const media = createProcessStream(source, sink);
  const sinkClosed = once(sink, 'close');
  const error = await new Promise<Error>((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error('no stream error')), 5_000);
    media.stream.once('error', (cause: Error) => { clearTimeout(timeout); resolve(cause); });
  });
  assert.match(error.message, /yt-dlp.*7.*test failure/);
  await sinkClosed;
});
