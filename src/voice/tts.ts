import { readFile } from 'node:fs/promises';
import path from 'node:path';
import phrases from './confirmations.json' with { type: 'json' };

export type Confirmation = keyof typeof phrases;
export interface VoiceTts { audio(key: Confirmation): Buffer | null }

// The local TTS model synthesizes fixed replies during setup. Playback needs neither
// a model process nor network access, and never stores a user's transcript.
export class LocalTts implements VoiceTts {
  private clips = new Map<Confirmation, Buffer>();
  voice = '';
  get ready(): boolean { return this.clips.size === Object.keys(phrases).length; }
  async load(directory = path.resolve('.runtime/tts')): Promise<boolean> {
    this.clips.clear();
    this.voice = '';
    if (process.env.VOICE_TTS === '0') return false;
    try {
      const manifest = JSON.parse(await readFile(path.join(directory, 'ready.json'), 'utf8'));
      if (!['piper', 'silero'].includes(manifest.engine) || typeof manifest.voice !== 'string'
        || manifest.sampleRate !== 48000 || manifest.channels !== 2
        || JSON.stringify(manifest.phrases) !== JSON.stringify(phrases)) return false;
      const loaded = new Map<Confirmation, Buffer>();
      for (const key of Object.keys(phrases) as Confirmation[]) {
        const pcm = await readFile(path.join(directory, `${key}.pcm`));
        if (!pcm.length || pcm.length % 4 || pcm.length > 48000 * 4 * 12) return false;
        loaded.set(key, pcm);
      }
      this.clips = loaded;
      this.voice = manifest.voice;
      return true;
    } catch { return false; }
  }
  audio(key: Confirmation): Buffer | null { return this.clips.get(key) ?? null; }
}
