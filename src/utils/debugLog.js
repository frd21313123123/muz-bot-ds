'use strict';

const fs = require('fs');
const path = require('path');

const DEBUG_LOG_PATH = path.join(process.cwd(), 'debug-runtime.log');

function debugLog(message) {
  if (process.env.DEBUG_BOT === '0') return;
  const line = `[${new Date().toISOString()}] ${message}\n`;
  fs.appendFile(DEBUG_LOG_PATH, line, () => {});
}

module.exports = { debugLog, DEBUG_LOG_PATH };
