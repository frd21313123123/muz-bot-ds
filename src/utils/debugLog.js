'use strict';

const fs = require('fs');
const path = require('path');

const DEBUG_LOG_PATH = path.join(process.cwd(), 'debug-runtime.log');
const MAX_BUFFERED_BYTES = 256 * 1024;

let stream = null;
let streamErrored = false;
let waitingDrain = false;
let buffered = [];
let bufferedBytes = 0;
let dropped = 0;

function ensureStream() {
  if (stream || streamErrored) return stream;

  stream = fs.createWriteStream(DEBUG_LOG_PATH, { flags: 'a' });
  stream.on('error', (err) => {
    streamErrored = true;
    console.error(`[debugLog] ${err.message}`);
    try { stream?.destroy(); } catch (_) {}
    stream = null;
    waitingDrain = false;
    buffered = [];
    bufferedBytes = 0;
    dropped = 0;
  });
  stream.on('drain', () => {
    waitingDrain = false;
    flushBuffered();
  });

  return stream;
}

function enqueueLine(line) {
  const bytes = Buffer.byteLength(line);
  if (bufferedBytes + bytes > MAX_BUFFERED_BYTES) {
    dropped += 1;
    return;
  }
  buffered.push({ line, bytes });
  bufferedBytes += bytes;
}

function writeNow(line) {
  const out = ensureStream();
  if (!out) return;
  const ok = out.write(line);
  if (!ok) waitingDrain = true;
}

function flushBuffered() {
  if (waitingDrain) return;
  const out = ensureStream();
  if (!out) return;

  if (dropped > 0) {
    const droppedLine = `[${new Date().toISOString()}] [debugLog] dropped ${dropped} lines due to backpressure\n`;
    dropped = 0;
    const ok = out.write(droppedLine);
    if (!ok) {
      waitingDrain = true;
      return;
    }
  }

  while (buffered.length > 0) {
    const item = buffered.shift();
    bufferedBytes -= item.bytes;
    const ok = out.write(item.line);
    if (!ok) {
      waitingDrain = true;
      break;
    }
  }

  if (bufferedBytes < 0) bufferedBytes = 0;
}

function debugLog(message) {
  if (process.env.DEBUG_BOT !== '1') return;
  const line = `[${new Date().toISOString()}] ${message}\n`;
  if (waitingDrain) {
    enqueueLine(line);
    return;
  }
  writeNow(line);
}

module.exports = { debugLog, DEBUG_LOG_PATH };
