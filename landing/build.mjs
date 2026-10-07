import { mkdir, copyFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
const root = fileURLToPath(new URL('.', import.meta.url));
await mkdir(`${root}dist`, { recursive: true });
for (const file of ['index.html', 'styles.css', 'script.js']) {
  await copyFile(`${root}${file}`, `${root}dist/${file}`);
}
console.log('MUZ landing page built.');
