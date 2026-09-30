import { createInterface } from 'node:readline';
console.log(JSON.stringify({ ready: true }));
createInterface({ input: process.stdin }).on('line', (line) => {
  const request = JSON.parse(line);
  if (process.argv.includes('--hang')) return;
  setTimeout(() => console.log(JSON.stringify({ id: request.id,
    result: request.op === 'transcribe' ? String(Buffer.from(request.pcm, 'base64')[0])
      : { action: 'skip', confidence: 0.99 } })), 30);
});
