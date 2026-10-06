"""Scan mono 16 kHz PCM16 WAV for the trained whole word; never transcribe speech."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.runtime/wake-validation'),str(ROOT/'scripts/wakeword')]
from wake_audio import WakeDetector,read_wav,N_SAMPLES


def load_detector(directory):
    metadata=json.loads((Path(directory)/'wake_config.json').read_text(encoding='utf-8'))
    if metadata.get('architecture')=='phonetic_ctc_v1':
        from phonetic_runtime import PhoneticDetector
        return PhoneticDetector(directory)
    return WakeDetector(directory)


def scan(detector,audio,hop=1600,confirmation_seconds=.3):
    if hop<=0 or confirmation_seconds<0: raise ValueError('Invalid scan timing')
    # Include the complete final window and short files without dropping the ending.
    endpoints=sorted(set([*range(hop,len(audio)+1,hop),len(audio)]))
    results=[]; onset=None; emitted=False
    for end in endpoints:
        probability=detector.score(audio[max(0,end-N_SAMPLES):end])
        if probability>=detector.config['threshold']:
            if onset is None: onset=end
            # A clipped prefix of «боту» can sound like «бот». Require following
            # context; at EOF the final score already sees the complete recording.
            confirmed=end-onset>=round(confirmation_seconds*16000) or end==len(audio)
            if confirmed and not emitted:
                results.append(dict(time_seconds=end/16000,probability=probability))
                emitted=True
        else:
            onset=None; emitted=False
    return dict(word=detector.config['wake_word_ru'],detected=bool(results),detections=results)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_directory',type=Path)
    parser.add_argument('wav',type=Path)
    args=parser.parse_args()
    print(json.dumps(scan(load_detector(args.model_directory),read_wav(args.wav)),ensure_ascii=False))
