# Playback echo audio fixtures

- `stop.pcm.gz` is the word «стоп», recorded by 1640max (Forvo). The original is [Ru-стоп.ogg](https://commons.wikimedia.org/wiki/File:Ru-стоп.ogg), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
- `pause.pcm.gz` is the word «пауза» from the Shtooka Project. The original is [Ru-пауза.ogg](https://commons.wikimedia.org/wiki/File:Ru-пауза.ogg), [CC BY 2.0 France](https://creativecommons.org/licenses/by/2.0/fr/).
- `piano.pcm.gz` is a four-second excerpt starting at 18 seconds from [Erik Satie: Gymnopédie No. 1, performed by Kevin MacLeod](https://commons.wikimedia.org/wiki/File:Kevin_MacLeod_-_Erik_Satie_Gymnopedie_No_1.ogg), [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/).
- `ensemble.pcm.gz` is a four-second excerpt starting at 18 seconds from [Monkeys Spinning Monkeys by Kevin MacLeod (incompetech.com)](https://commons.wikimedia.org/wiki/File:Kevin_MacLeod_~_Monkeys_Spinning_Monkeys.ogg), [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/).

The speech recordings were converted to 16 kHz mono and the music to 48 kHz stereo. The fixtures are signed 16-bit little-endian PCM compressed with gzip. The licenses above cover the audio fixtures and their conversions.

## Run the benchmark

`npm test` includes the audio test. To run it alone and export WAVs from the repository root:

```sh
npm run build
ECHO_AUDIO_WAV_DIR=.runtime/echo-wavs node --test dist/tests/echoAudio.test.js
```

Playback uses real stereo Opus packets and the bot's mono playback-reference decoder. The test feeds the production filter 20 ms frames with a deterministic clock. It compares direct PCM echo (71 ms delay), direct echo with microphone Opus at 24 kbps (137 ms), and the same microphone codec with a room model (480 ms direct delay plus four attenuated reflections through 129 ms). The test scales echo to −24 dBFS RMS over each speech interval before mixing and sweeps the speech-to-echo RMS ratio from 0 to −30 dB in 10 dB steps.

Each mixture has a same-level speech-only control through the same microphone codec. Echo-only controls measure attenuation without speech. The test asserts sample preservation, clipping-free input, direct echo rejection, and speech retention for direct mixes at 0 dB. It reports the time in the command interval classified as echo-only and the projected speech level for the remaining cases. Projection is sensitive to codec changes and can be biased by leaked music, especially at the lowest speech levels.

For a separate local speech recognition check, install `vosk==0.3.45` in a Python environment and unpack [vosk-model-small-ru-0.22](https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip). Use a fresh WAV directory for each run.

```sh
python tests/echo_audio_stt.py /path/to/vosk-model-small-ru-0.22 .runtime/echo-wavs
```

The script uses the unrestricted Vosk model and an exact transcript match for each command. For echo-only controls, it checks for either command word in the transcript. Compare `clean`, `mixed`, and `filtered` at the same speech level. This measures recognition of the filter's output. Discord, voice activation, utterance segmentation, and the production Groq recognizer are outside this benchmark.
