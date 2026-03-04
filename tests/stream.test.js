'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { destroyStream } = require('../src/utils/stream');

test('destroyStream unpipes and tears down child processes safely', () => {
  let unpiped = false;
  let stdinDestroyed = false;
  let ytdlpKilled = false;
  let ffmpegKilled = false;
  let streamDestroyed = false;

  const ffmpegStdin = {
    destroy: () => {
      stdinDestroyed = true;
    },
  };

  const stream = {
    _ytdlpProc: {
      stdout: {
        unpipe: (target) => {
          assert.equal(target, ffmpegStdin);
          unpiped = true;
        },
      },
      kill: () => {
        ytdlpKilled = true;
      },
    },
    _ffmpegProc: {
      stdin: ffmpegStdin,
      kill: () => {
        ffmpegKilled = true;
      },
    },
    destroy: () => {
      streamDestroyed = true;
    },
  };

  destroyStream(stream);

  assert.equal(unpiped, true);
  assert.equal(stdinDestroyed, true);
  assert.equal(ytdlpKilled, true);
  assert.equal(ffmpegKilled, true);
  assert.equal(streamDestroyed, true);
});
