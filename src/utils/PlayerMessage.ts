import type { Message } from 'discord.js';
import type { MusicClient } from '../types.js';
import type { GuildQueue } from './GuildQueue.js';
import { playerActionRow, playerEmbed } from './embeds.js';

export class PlayerMessage {
  private message: Message | null = null;
  private channelId: string | null = null;
  private timer: NodeJS.Timeout | null = null;
  private updating = false;
  private queued = false;
  private revision = 0;
  private lastPayload: string | null = null;

  constructor(private readonly client: MusicClient, private readonly queue: GuildQueue) {}

  setChannel(channelId: string): void {
    if (this.channelId !== channelId) this.revision++;
    this.channelId = channelId;
    if (!this.timer) this.timer = setInterval(() => void this.update(), 10_000);
  }

  async replace(message: Message, channelId: string): Promise<void> {
    await this.removeMessage();
    this.message = message;
    this.lastPayload = null;
    this.setChannel(channelId);
  }

  async update(): Promise<void> {
    if (!this.channelId || this.queue.closed) return;
    if (this.updating) { this.queued = true; return; }
    this.updating = true;
    const revision = this.revision;
    try {
      // Collapse a burst of synchronous state changes into one final payload.
      await Promise.resolve();
      if (revision !== this.revision || this.queue.closed || !this.channelId) return;
      const payload = { embeds: [playerEmbed(this.queue)], components: [playerActionRow(this.queue)] };
      const serialized = JSON.stringify(payload);
      if (this.message) {
        if (serialized === this.lastPayload) return;
        const message = this.message;
        try { await message.edit(payload); if (revision === this.revision && this.message === message) this.lastPayload = serialized; return; }
        catch {
          if (revision !== this.revision || this.message !== message) return;
          this.message = null;
        }
      }
      const channel = await this.client.channels.fetch(this.channelId).catch(() => null);
      if (revision !== this.revision || this.queue.closed) return;
      if (channel && 'send' in channel && typeof channel.send === 'function') {
        const message = await channel.send(payload) as Message;
        if (revision !== this.revision || this.queue.closed) await message.delete().catch(() => {});
        else { this.message = message; this.lastPayload = serialized; }
      }
    } catch (error) {
      console.error('[PlayerMessage]', error);
    } finally {
      this.updating = false;
      if (this.queued) {
        this.queued = false;
        queueMicrotask(() => void this.update());
      }
    }
  }

  async removeMessage(): Promise<void> {
    const previous = this.message;
    this.message = null;
    this.lastPayload = null;
    if (previous) await previous.delete().catch(() => {});
  }

  async destroy(): Promise<void> {
    this.revision++;
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    this.channelId = null;
    await this.removeMessage();
  }
}
