# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
npm start          # Run the bot (production)
npm run dev        # Run with --watch (auto-restart on file changes)
npm run deploy     # Register slash commands with Discord API
node test.js       # Run integration tests (URL resolution, streaming, autoplay)
```

`npm run deploy` must be re-run after adding or modifying any slash command.
With `GUILD_ID` in `.env` commands sync instantly; without it, global sync takes up to 1 hour.

## Architecture

The bot uses two separate engines:
- **play-dl** — metadata only: YouTube search, video info, playlist parsing
- **yt-dlp** (system binary) — actual audio streaming, autoplay recommendations

`play-dl`'s `stream()` function is broken with current YouTube API (returns `Invalid URL`). All audio goes through yt-dlp.

### Stream pipeline

```
YouTube URL → yt-dlp (bestaudio) → FFmpeg (libopus 48kHz stereo 128k) → Discord voice
```

Both yt-dlp and FFmpeg are spawned as child processes with stdio piping (`src/utils/stream.js`). The output is `StreamType.OggOpus` — Discord's native format.

### Per-guild state

`client.queues` is a `Map<guildId, GuildQueue>`. Each GuildQueue owns:
- An `AudioPlayer` and `VoiceConnection`
- A track queue (`tracks[]`) and `currentTrack`
- An `_advancing` flag to prevent race conditions during queue transitions
- A 5-minute idle timer that auto-disconnects

### Command/event loading

Dynamic filesystem discovery in `src/index.js`: reads all `.js` files from `src/commands/` and `src/events/`. No explicit imports — just drop a file to register.

Each command exports `{ data: SlashCommandBuilder, execute(interaction, client) }`.
Each event exports `{ name, once?, execute(...args, client) }`.

### YouTube Music URL handling

`music.youtube.com` URLs are normalized to `www.youtube.com` in two places:
1. `src/utils/resolve.js` — during query resolution
2. `src/utils/GuildQueue.js` `_playTrack()` — safety net before streaming

RDAMVM playlists (YouTube Music auto-mixes) often fail in `play-dl`'s `playlist_info()`. The resolver falls back to extracting the `?v=` parameter as a single video.

### Autoplay

Uses yt-dlp's Radio Mix: `?list=RD{videoId}` playlist. Fetches items 2–6 (skipping item 1 which is the current track). Runs `execFile('yt-dlp', ['--flat-playlist', ...])` to get IDs/titles without downloading.

## Key constraints

- **Windows without Visual Studio**: uses `opusscript` (pure JS) instead of `@discordjs/opus` (native). Do not switch to `@discordjs/opus`.
- **yt-dlp must be installed** system-wide (e.g., `pip install yt-dlp`).
- All bot replies are **ephemeral** (visible only to the command invoker). GuildQueue has no `textChannel.send()` calls.
- Volume range is 1–150% (internally 0.0–1.5 scale on `inlineVolume`).
- The `ready` event uses `clientReady` name (discord.js v15 migration).

## Environment variables

```
DISCORD_TOKEN   — Bot token (required)
CLIENT_ID       — Application ID (required)
GUILD_ID        — Server ID for instant command registration (optional)
```
