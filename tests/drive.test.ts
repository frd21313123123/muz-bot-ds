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

test('archive is opt-in and stores a snapshot as a valid WAV without an rclone remote', async () => {
  const root = await temporaryDirectory();
  const off = new VoiceDriveArchive({ directory: path.join(root, 'off'), enabled: false });
  const archive = new VoiceDriveArchive({ directory: root, enabled: true, remote: '' });
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

test('archive copies WAV to the configured remote and folder before removing the local file', async () => {
  const root = await temporaryDirectory();
  let uploads = 0;
  let release: (() => void) | undefined;
  const gate = new Promise<void>(resolve => { release = resolve; });
  const archive = new VoiceDriveArchive({
    directory: root, enabled: true, remote: 'voice-drive', rclonePath: 'test-rclone',
    runRclone: async (executable, args) => {
      const file = args[1]!;
      assert.equal(executable, 'test-rclone');
      assert.deepEqual(args, [
        'copyto', file, `voice-drive:${path.basename(file)}`,
        '--drive-root-folder-id', VOICE_DRIVE_FOLDER, '--checksum', '--retries', '1', '--dry-run=false',
      ]);
      assert.equal(path.dirname(file), root);
      assert.deepEqual(await readFile(file), speechWav(pcm));
      uploads++;
      if (uploads === 1) await gate;
    },
  });
  try {
    archive.append(pcm); archive.append(pcm); await archive.flush();
    await until(async () => uploads === 1);
    assert.equal((await readdir(root)).length, 2);
    release!();
    await until(async () => (await readdir(root)).length === 0);
    assert.equal(uploads, 2);
  } finally { release!(); await archive.close(); await rm(root, { recursive: true, force: true }); }
});

test('failed uploads survive shutdown and are recovered on the next start', async () => {
  const root = await temporaryDirectory();
  let uploads = 0;
  const destinations: string[] = [];
  const options = {
    directory: root, enabled: true, remote: 'voice-drive',
    runRclone: async (_executable: string, args: string[]) => {
      destinations.push(args[2]!);
      if (++uploads === 1) throw new Error('rclone failed');
    },
  };
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
    assert.equal(destinations[0], destinations[1]);
  } finally { await first.close(); await second?.close(); await rm(root, { recursive: true, force: true }); }
});
