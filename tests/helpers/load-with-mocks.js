'use strict';

const Module = require('module');

/**
 * Loads a module with temporary dependency mocks.
 * @param {string} resolvedModulePath Absolute path from require.resolve()
 * @param {Record<string, any>} mocks
 */
function loadWithMocks(resolvedModulePath, mocks) {
  const originalLoad = Module._load;
  delete require.cache[resolvedModulePath];

  Module._load = function patchedLoad(request, parent, isMain) {
    if (Object.prototype.hasOwnProperty.call(mocks, request)) {
      return mocks[request];
    }
    return originalLoad.call(this, request, parent, isMain);
  };

  try {
    return require(resolvedModulePath);
  } finally {
    Module._load = originalLoad;
  }
}

module.exports = { loadWithMocks };
