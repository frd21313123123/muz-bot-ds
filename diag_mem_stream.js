'use strict';

const fs = require('fs');
const path = require('path');
const { createYtdlpStream, destroyStream } = require('./src/utils/stream');
const { resolveQuery } = require('./src/utils/resolve');

const REPORT_PATH = path.join(process.cwd(), 'diag-mem-stream.log');
const TEST_URL = 'https://music.youtube.com/watch?v=ETxmCCsMoD0&list=RDAMVMi5_asj1BGFs';
const TEST_DURATION_MS = 60_000;

function log(line) {
  const msg = `[${new Date().toISOString()}] ${line}\n`;
  fs.appendFileSync(REPORT_PATH, msg);
  process.stdout.write(msg);
}

async function main() {
  fs.writeFileSync(REPORT_PATH, '');
  log('diag_mem_stream start');
  log(`node=${process.version} pid=${process.pid} cwd=${process.cwd()}`);

  const resolved = await resolveQuery(TEST_URL, 'DiagMem');
  if (!resolved) {
    log('resolveQuery returned null');
    process.exit(2);
    return;
  }

  const track = resolved.type === 'single' ? resolved.track : resolved.tracks[0];
  log(`resolved type=${resolved.type} title="${track?.title}" url=${track?.url}`);

  const { stream } = createYtdlpStream(track.url);
  let bytes = 0;
  let lastBytes = 0;
  let ended = false;

  stream.on('data', (chunk) => {
    bytes += chunk.length;
  });
  stream.on('error', (err) => {
    log(`stream error: ${err.message}`);
  });
  stream.on('end', () => {
    ended = true;
    log('stream end event');
  });

  const started = Date.now();
  const timer = setInterval(() => {
    const rssMb = (process.memoryUsage().rss / 1024 / 1024).toFixed(2);
    const heapMb = (process.memoryUsage().heapUsed / 1024 / 1024).toFixed(2);
    const deltaKb = ((bytes - lastBytes) / 1024).toFixed(1);
    lastBytes = bytes;
    log(`tick rss=${rssMb}MiB heap=${heapMb}MiB bytes=${(bytes / 1024).toFixed(1)}KB delta=${deltaKb}KB ended=${ended}`);
  }, 1000);

  await new Promise((resolve) => setTimeout(resolve, TEST_DURATION_MS));
  clearInterval(timer);
  destroyStream(stream);

  const totalSec = ((Date.now() - started) / 1000).toFixed(1);
  log(`finished seconds=${totalSec} totalBytes=${(bytes / 1024).toFixed(1)}KB`);
}

main()
  .then(() => process.exit(0))
  .catch((err) => {
    try {
      log(`fatal: ${err.message}`);
    } catch (_) {}
    process.exit(1);
  });
