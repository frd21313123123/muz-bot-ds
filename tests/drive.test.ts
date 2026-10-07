import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdir, mkdtemp, readdir, readFile, rm } from 'node:fs/promises';
import path from 'node:path';
import { VoiceDriveArchive, VOICE_DRIVE_FOLDER } from '../src/voice/drive.js';
import { speechWav } from '../src/voice/groq.js';

const pcm = Buffer.from([1, 0, 2, 0, 3, 0]);
async function temporaryDirectory(): Promise<string> {
  // Windows sandbox supports atomic rename inside the workspace.
  const root = path.resolve('.runtime');
  await mkdir(root, { recursive: true });
  return mkdtemp(path.join(root, 'drive-test-'));
}
async function until(check: () => Promise<boolean>): Promise<void> {
  for (let i = 0; i < 200; i++) {
    if (await check()) return;
    await new Promise(resolve => setTimeout(resolve, 5));
  }
  assert.fail('Timed out waiting for archive');
}

test('archive is opt-in and stores a snapshot as a valid WAV without Google credentials', async () => {
  const root = await temporaryDirectory();
  const off = new VoiceDriveArchive({ directory: path.join(root, 'off'), enabled: false });
  const archive = new VoiceDriveArchive({ directory: root, enabled: true, clientId: '', clientSecret: '', refreshToken: '' });
  try {
    off.append(pcm); await off.flush();
    await assert.rejects(readdir(path.join(root, 'off')));
    const input = Buffer.from(pcm);
    archive.append(input); input.fill(0); await archive.flush();
    const names = await readdir(root);
    assert.equal(names.length, 1); assert.match(names[0]!, /^command-.*\.wav$/);
    assert.deepEqual(await readFile(path.join(root, names[0]!)), speechWav(pcm));
  } finally { await archive.close(); await off.close(); await rm(root, { recursive: true, force: true }); }
});

test('archive uploads WAV into the requested folder, caches tokens and removes only successful uploads', async () => {
  const root = await temporaryDirectory();
  let tokens = 0, uploads = 0;
  let release: (() => void) | undefined;
  const gate = new Promise<void>(resolve => { release = resolve; });
  const mock = (async (url: string | URL | Request, init?: RequestInit) => {
    if (String(url).includes('oauth2')) {
      tokens++;
      assert.equal((init!.body as URLSearchParams).get('refresh_token'), 'refresh');
      return Response.json({ access_token: 'access', expires_in: 3600 });
    }
    uploads++;
    assert.match(String(url), /uploadType=multipart/);
    assert.equal((init!.headers as Record<string, string>).Authorization, 'Bearer access');
    const body = Buffer.from(init!.body as Uint8Array);
    assert.ok(body.includes(Buffer.from(JSON.stringify([VOICE_DRIVE_FOLDER]))));
    assert.ok(body.includes(speechWav(pcm)));
    if (uploads === 1) await gate;
    return Response.json({ id: `file-${uploads}` });
  }) as typeof fetch;
  const archive = new VoiceDriveArchive({ directory: root, enabled: true, clientId: 'client',
    clientSecret: 'secret', refreshToken: 'refresh', fetch: mock });
  try {
    archive.append(pcm); archive.append(pcm); await archive.flush();
    await until(async () => uploads === 1);
    assert.equal((await readdir(root)).length, 2);
    release!();
    await until(async () => (await readdir(root)).length === 0);
    assert.equal(uploads, 2); assert.equal(tokens, 1);
  } finally { release!(); await archive.close(); await rm(root, { recursive: true, force: true }); }
});

test('failed uploads survive shutdown and are recovered on the next start', async () => {
  const root = await temporaryDirectory();
  let uploads = 0;
  const mock = (async (url: string | URL | Request) => {
    if (String(url).includes('oauth2')) return Response.json({ access_token: 'access', expires_in: 3600 });
    uploads++;
    return uploads === 1 ? new Response('', { status: 503 }) : Response.json({ id: 'recovered' });
  }) as typeof fetch;
  const options = { directory: root, enabled: true, clientId: 'client', clientSecret: 'secret', refreshToken: 'refresh', fetch: mock };
  const first = new VoiceDriveArchive(options);
  let second: VoiceDriveArchive | undefined;
  try {
    first.append(pcm); await first.flush();
    await until(async () => uploads === 1);
    await first.close();
    assert.equal((await readdir(root)).length, 1);
    second = new VoiceDriveArchive(options); second.start();
    await until(async () => (await readdir(root)).length === 0);
    assert.equal(uploads, 2);
  } finally { await first.close(); await second?.close(); await rm(root, { recursive: true, force: true }); }
});
