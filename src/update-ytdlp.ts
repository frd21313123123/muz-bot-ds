import 'dotenv/config';
import { prepareYtdlp } from './utils/ytdlp.js';

prepareYtdlp(true)
  .then(() => console.log('✅ yt-dlp обновлён.'))
  .catch((error) => { console.error(error); process.exitCode = 1; });
