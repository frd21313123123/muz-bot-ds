import type { Track } from '../types.js';

export class TrackQueue {
  readonly items: Track[] = [];

  add(track: Track): void {
    if (track.isAutoplay) {
      this.items.push(track);
      return;
    }
    const firstAutoplay = this.items.findIndex((item) => item.isAutoplay);
    this.items.splice(firstAutoplay < 0 ? this.items.length : firstAutoplay, 0, track);
  }

  addMany(tracks: Track[]): void {
    for (const track of tracks) this.add(track);
  }

  shift(): Track | undefined { return this.items.shift(); }

  clear(): number {
    const length = this.items.length;
    this.items.length = 0;
    return length;
  }
}
