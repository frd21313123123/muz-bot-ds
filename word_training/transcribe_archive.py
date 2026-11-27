"""Produce local, provisional transcripts for exactly the windows used by the detectors."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'models/wake-model'))
from wake_audio import read_wav


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recordings', type=Path)
    parser.add_argument('asr_model', type=Path, help='Local CTranslate2 Whisper directory')
    parser.add_argument('output', type=Path)
    parser.add_argument('--whole', action='store_true', help='Transcribe each complete recording for utterance-level intent labels')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if os.name == 'nt':
        libs = str(Path(torch.__file__).parent / 'lib')
        os.environ['PATH'] = libs + os.pathsep + os.environ['PATH']
        dll_handle = os.add_dll_directory(libs)
    from faster_whisper import WhisperModel
    model = WhisperModel(str(args.asr_model), device='cuda', compute_type='int8_float16', local_files_only=True)
    rows = []
    for path in sorted(args.recordings.glob('*.wav')):
        audio = read_wav(path)
        # Archive clips longer than two seconds need explicit window-local labels.
        for start in ([0] if args.whole else sorted({0, max(0, len(audio) - 32000)})):
            window = audio if args.whole else audio[start:start + 32000]
            segments, _ = model.transcribe(window, language='ru', beam_size=3, temperature=0,
                condition_on_previous_text=False, vad_filter=False, without_timestamps=True,
                max_new_tokens=32)
            segments = list(segments)
            rows.append(dict(file=path.name, start=start, samples=len(window),
                text=' '.join(s.text.strip() for s in segments),
                logprob=min((s.avg_logprob for s in segments), default=-10),
                no_speech=max((s.no_speech_prob for s in segments), default=1)))
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf8')
    print(f'{len(rows)} windows in {args.output}', flush=True)


if __name__ == '__main__':
    main()
