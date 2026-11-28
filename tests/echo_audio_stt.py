"""Measure exact command recognition on the WAVs exported by echoAudio.test.ts."""

import argparse
import csv
import json
import re
import sys
import wave
from pathlib import Path

from vosk import KaldiRecognizer, Model, SetLogLevel


def recognize(model, path):
    with wave.open(str(path), "rb") as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (16000, 1, 2):
            raise ValueError(f"Expected 16 kHz mono s16 WAV: {path}")
        recognizer = KaldiRecognizer(model, audio.getframerate())
        parts = []
        while chunk := audio.readframes(3200):
            if recognizer.AcceptWaveform(chunk):
                parts.append(json.loads(recognizer.Result())["text"])
        parts.append(json.loads(recognizer.FinalResult())["text"])
    return " ".join(part for part in parts if part)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="unpacked Russian Vosk model")
    parser.add_argument("wav_dir", type=Path)
    args = parser.parse_args()
    pattern = re.compile(r"([a-z]+)-([a-z]+)-(stop|pause)(?:-(-?\d+))?-(clean|mixed|filtered|echo)")
    expected = {"stop": "стоп", "pause": "пауза"}
    inputs = [(match.groups(), path) for path in sorted(args.wav_dir.glob("*.wav"))
              if (match := pattern.fullmatch(path.stem))]
    if not inputs:
        parser.error(f"No exported test WAVs in {args.wav_dir}")
    SetLogLevel(-1)
    model = Model(str(args.model))
    writer = csv.writer(sys.stdout)
    writer.writerow(("track", "room", "command", "speech_db", "variant", "text", "correct"))
    for (track, room, command, speech_db, variant), path in inputs:
        text = recognize(model, path)
        correct = not set(text.split()).intersection(expected.values()) if variant == "echo" else text == expected[command]
        writer.writerow((track, room, command, speech_db, variant, text, int(correct)))


if __name__ == "__main__":
    main()
