'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { loadWithMocks } = require('./helpers/load-with-mocks');

const resolveModulePath = require.resolve('../src/utils/resolve');

test('resolveQuery normalizes YouTube Music URL for single video', async () => {
  const calls = {};
  const playMock = {
    yt_validate: (url) => {
      calls.validated = url;
      return 'video';
    },
    video_basic_info: async (url) => {
      calls.basicInfo = url;
      return {
        video_details: {
          id: 'abc123',
          title: 'Song',
          durationRaw: '3:21',
          thumbnails: [{ url: 'https://img.example/1.jpg' }],
          url: 'https://music.youtube.com/watch?v=abc123',
        },
      };
    },
    playlist_info: async () => {
      throw new Error('should not be called');
    },
    search: async () => {
      throw new Error('should not be called');
    },
  };

  const { resolveQuery } = loadWithMocks(resolveModulePath, { 'play-dl': playMock });
  const result = await resolveQuery('  https://music.youtube.com/watch?v=abc123  ', 'Tester');

  assert.equal(calls.validated, 'https://www.youtube.com/watch?v=abc123');
  assert.equal(calls.basicInfo, 'https://www.youtube.com/watch?v=abc123');
  assert.equal(result.type, 'single');
  assert.equal(result.track.url, 'https://www.youtube.com/watch?v=abc123');
  assert.equal(result.track.videoId, 'abc123');
  assert.equal(result.track.requestedBy, 'Tester');
});

test('resolveQuery returns playlist tracks for playlist URL', async () => {
  const playMock = {
    yt_validate: () => 'playlist',
    playlist_info: async () => ({
      title: 'Playlist',
      all_videos: async () => [
        {
          id: 'v1',
          title: 'Track 1',
          durationRaw: '2:00',
          thumbnails: [{ url: 'https://img.example/1.jpg' }],
          url: '',
        },
        {
          id: 'v2',
          title: 'Track 2',
          durationRaw: '2:30',
          thumbnails: [{ url: 'https://img.example/2.jpg' }],
          url: 'https://music.youtube.com/watch?v=v2',
        },
      ],
    }),
    video_basic_info: async () => {
      throw new Error('should not be called');
    },
    search: async () => {
      throw new Error('should not be called');
    },
  };

  const { resolveQuery } = loadWithMocks(resolveModulePath, { 'play-dl': playMock });
  const result = await resolveQuery('https://www.youtube.com/playlist?list=PL123', 'Tester');

  assert.equal(result.type, 'playlist');
  assert.equal(result.name, 'Playlist');
  assert.equal(result.tracks.length, 2);
  assert.equal(result.tracks[0].url, 'https://www.youtube.com/watch?v=v1');
  assert.equal(result.tracks[1].url, 'https://www.youtube.com/watch?v=v2');
});

test('resolveQuery falls back to single video when playlist fetch fails', async () => {
  const calls = {};
  const playMock = {
    yt_validate: () => 'playlist',
    playlist_info: async () => {
      throw new Error('playlist unavailable');
    },
    video_basic_info: async (url) => {
      calls.basicInfo = url;
      return {
        video_details: {
          id: 'fallback-id',
          title: 'Fallback',
          durationRaw: '4:10',
          thumbnails: [],
          url: 'https://www.youtube.com/watch?v=fallback-id',
        },
      };
    },
    search: async () => {
      throw new Error('should not be called');
    },
  };

  const { resolveQuery } = loadWithMocks(resolveModulePath, { 'play-dl': playMock });
  const result = await resolveQuery(
    'https://www.youtube.com/watch?v=fallback-id&list=RDAMVM123',
    'Tester',
  );

  assert.equal(calls.basicInfo, 'https://www.youtube.com/watch?v=fallback-id');
  assert.equal(result.type, 'single');
  assert.equal(result.track.videoId, 'fallback-id');
});

test('resolveQuery uses search for plain text and returns null when empty', async () => {
  const emptySearchMock = {
    yt_validate: () => false,
    playlist_info: async () => {
      throw new Error('should not be called');
    },
    video_basic_info: async () => {
      throw new Error('should not be called');
    },
    search: async () => [],
  };

  {
    const { resolveQuery } = loadWithMocks(resolveModulePath, { 'play-dl': emptySearchMock });
    const noResult = await resolveQuery('some text', 'Tester');
    assert.equal(noResult, null);
  }

  const successSearchMock = {
    yt_validate: () => false,
    playlist_info: async () => {
      throw new Error('should not be called');
    },
    video_basic_info: async () => {
      throw new Error('should not be called');
    },
    search: async () => [
      {
        id: 's1',
        title: 'Search Song',
        durationRaw: '1:11',
        thumbnails: [{ url: 'https://img.example/s1.jpg' }],
        url: 'https://www.youtube.com/watch?v=s1',
      },
    ],
  };

  const { resolveQuery } = loadWithMocks(resolveModulePath, { 'play-dl': successSearchMock });
  const result = await resolveQuery('some text', 'Tester');

  assert.equal(result.type, 'single');
  assert.equal(result.track.videoId, 's1');
});
