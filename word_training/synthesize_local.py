"""Render distinct source utterances before adding any acoustic variants."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

from local_prompts import CONFUSIONS, EXTRA, WAKE

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/wakeword'))
from wake_audio import write_wav
from wake_tts import Synthesizer

SPLITS = ('train', 'validation', 'test')
VOICES = (('aidar', 'baya', 'xenia'), ('kseniya',), ('eugene',))


def prompts():
    groups = defaultdict(set)
    for line in (ROOT / 'data/lia_action_v4_authored.txt').read_text(encoding='utf8').splitlines():
        if line.startswith('['):
            split = line[1:].split(':')[0]
        elif '|' in line and not line.startswith('#'):
            label, phrases = line.split('|', 1)
            if split in SPLITS and label in {'pause', 'resume', 'skip', 'stop', 'volume_up', 'volume_down'}:
                groups[split, label].update(p.strip() for p in phrases.split('||') if len(p.split()) <= 5)
    for i, split in enumerate(SPLITS):
        for label, columns in EXTRA.items():
            groups[split, label].update(columns[i].split('|'))
    seen = set()
    result = []
    for i, split in enumerate(SPLITS):
        for (fold, label), texts in sorted(groups.items()):
            if fold != split:
                continue
            options = sorted(texts - seen)
            random.Random(771 + i).shuffle(options)
            for n, text in enumerate(options[:(24, 10, 14)[i]]):
                seen.add(text)
                result.append((split, label, 0, text))
                if n % 2 == 0:
                    result.append((split, label, 1, 'бот, ' + text))
        result.extend((split, 'unknown', 1, t) for t in WAKE[i].split('|'))
        result.extend((split, 'unknown', 0, t) for t in CONFUSIONS[i].split('|'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=ROOT / '.runtime/local-voice')
    parser.add_argument('--tts-work', type=Path, default=ROOT / 'word_training/runs/bot-v2',
                        help='Existing wakeword TTS cache containing tts/v5_5_ru.pt')
    args = parser.parse_args()
    torch.set_num_threads(2)
    folder = args.work / 'synthetic'
    folder.mkdir(exist_ok=True, parents=True)
    rows = []
    for i, voices in enumerate(VOICES):
        for voice in voices:
            engine = Synthesizer(args.tts_work, dict(engine='silero', model='v5_5_ru', voice=voice), 'cpu')
            for split, label, wake, text in prompts():
                if split != SPLITS[i]:
                    continue
                key = hashlib.sha256((voice + '/' + text).encode()).hexdigest()[:20]
                path = folder / (key + '.wav')
                if not path.exists():
                    with torch.inference_mode():
                        audio = engine.say(text, seed=int(key[:8], 16))
                    active = np.flatnonzero(np.abs(audio) > max(.001, np.max(np.abs(audio)) * .012))
                    if not len(active):
                        raise ValueError('Silent synthesis: ' + text)
                    audio = audio[max(0, active[0]-320):active[-1]+321]
                    write_wav(path, audio)
                rows.append(dict(file=str(path.relative_to(args.work)), source=key, origin='tts', speaker=voice,
                                 split=split, text=text, wake=wake, command=label, start=0))
            print(voice, len(rows), flush=True)
    (args.work / 'synthetic.json').write_text(json.dumps(rows, ensure_ascii=False), encoding='utf8')


if __name__ == '__main__':
    main()
