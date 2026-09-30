import { createInterface } from 'node:readline';
console.log(JSON.stringify({ ready: true, modelName: process.argv.includes('--v3') ? 'laya-muz-bot-ds-v3' : 'test' }));
createInterface({ input: process.stdin }).on('line', (line) => {
  const request = JSON.parse(line);
  if (process.argv.includes('--hang')) return;
  if (request.op === 'transcribe' && process.argv.includes('--metrics')) {
    console.log(JSON.stringify({ id: request.id, result: { text: 'Бот', metrics: {
      vadMs: 320, segments: 1, rejectedSegments: 0, privateText: 'do not forward' } } })); return;
  }
  if (request.op === 'music_route') {
    console.log(JSON.stringify({ id: request.id, result: { next_tool: 'youtube_music_search', query_source: 'message',
      search_result_policy: 'rerank_results', defer_result_policy: process.argv.includes('--v3'), confidence: 0.99 } })); return;
  }
  if (request.op === 'music_policy') {
    console.log(JSON.stringify({ id: request.id, result: { search_result_policy: 'rerank_results', confidence: 0.99 } })); return;
  }
  if (request.op === 'music_rerank') {
    console.log(JSON.stringify({ id: request.id, result: { best_track: process.argv.includes('--bad-index') ? 99 : process.argv.includes('--none') ? -1 : 1,
      ...(process.argv.includes('--none') ? { no_match: true } : {}), confidence: 0.99 } })); return;
  }
  setTimeout(() => console.log(JSON.stringify({ id: request.id,
    result: request.op === 'transcribe' ? String(Buffer.from(request.pcm, 'base64')[0])
      : { action: 'skip', confidence: 0.99 } })), 30);
});
