# Repository guidance

## Commands

```bash
npm run build             # strict TypeScript build
npm test                  # offline tests
npm run test:integration  # live YouTube/yt-dlp/FFmpeg smoke test
npm run deploy            # register slash commands
npm start                 # run compiled bot
npm run dev               # watch TypeScript sources
npm run update:ytdlp      # update managed Python yt-dlp
```

Run `npm run deploy` after changing slash-command definitions. Set `GUILD_ID` for fast registration in a test server.

## Architecture and constraints

- TypeScript ESM, Node.js 24.17+, strict type checking. `src/commands.ts` is the command registry used by runtime and deployment.
- `src/utils/ytdlp.ts` owns search, metadata, playlists, recommendations, and the yt-dlp process. It uses Node as the YouTube JavaScript runtime and installs `yt-dlp[default]` into `.runtime/yt-dlp` on first launch unless `YT_DLP_PATH` is set. Python 3.11+ is required.
- `src/utils/stream.ts` owns yt-dlp → FFmpeg child processes and their cleanup. Keep certificate verification enabled. FFmpeg must provide `libopus`.
- `GuildQueue` owns per-guild voice and player state; `TrackQueue` orders manual tracks before autoplay tracks; `PlayerMessage` updates the public player.
- Keep `opusscript` for Windows without Visual Studio. Do not add a native `@discordjs/opus` requirement.
- All command replies are ephemeral **except `/player`**, which deliberately creates a public, updating message. Control buttons reply ephemerally when appropriate. Do not add unrelated channel messages.
- Commands and buttons that change playback require the invoker to be in the bot's voice channel. `/play` can start a new queue from an empty voice connection.
- Volume is 1–150%; idle disconnect occurs after five minutes. Without `infinite`, `/play` takes only the first playlist track.
- Keep tracked secrets and runtime artifacts out of Git: `.env`, `.runtime/`, `dist/`, and logs are ignored.
