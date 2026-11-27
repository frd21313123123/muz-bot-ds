"""Build an indexed PCM16 dataset. Every row represents the exact window used for scoring."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'models/wake-model'))
from wake_audio import read_wav, right_aligned_window


def historical_rows(manifest):
    # Earlier Windows runs may have been moved to a different drive.
    pools = defaultdict(list)
    folds = defaultdict(set)
    with manifest.open(encoding='utf8') as source:
        for line in source:
            r = json.loads(line)
            key = r.get('source_id', r['id'])
            folds[key].add(r['split'])
            pools[r['split'], r['label']].append((key, r))
    for split, limit in [('train', 2100), ('validation', 400), ('test', 800)]:
        for label in (0, 1):
            used = set()
            bucket = pools[split, label]
            random.Random(150 + label).shuffle(bucket)
            for key, r in bucket:
                if len(folds[key]) != 1 or key in used:
                    continue
                parts = r['wav'].replace('\\', '/').split('/')
                path = (ROOT.joinpath(*parts[parts.index('word_training'):]) if 'word_training' in parts
                        else manifest.parent / r['wav'])
                if not path.is_file():
                    continue
                used.add(key)
                yield dict(file=str(path), audio_index=r.get('audio_index'), source='legacy/' + key, origin='legacy', split=split, wake=label,
                           command=None, speaker=r.get('speaker'), start=0, text=r.get('text', ''))
                if len(used) == limit:
                    break


def variant(audio, index, rng):
    if not index:
        return right_aligned_window(audio)
    # Keep all annotated speech inside the window, including short final consonants.
    speed = max(rng.uniform(.90, 1.12), len(audio) / 32000)
    position = np.arange(0, len(audio), speed)
    speech = np.interp(position, np.arange(len(audio)), audio).astype(np.float32)[:32000]
    room = 32000 - len(speech)
    start = rng.integers(room + 1) if room else 0
    out = np.zeros(32000, dtype=np.float32)
    out[start:start + len(speech)] = speech * rng.uniform(.33, 1.15)
    if index % 3:
        delay = rng.integers(140, 1100)
        out[delay:] += out[:-delay] * rng.uniform(.05, .28)
    rms = np.sqrt(np.mean(out ** 2))
    noise = rng.normal(0, max(rms, .001) * 10 ** (-rng.uniform(9, 34) / 20), len(out))
    return np.clip(out + noise, -1, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=ROOT / '.runtime/local-voice')
    parser.add_argument('--legacy', type=Path, default=ROOT / 'word_training/runs/bot-v6-hardneg/dataset/manifest.jsonl')
    args = parser.parse_args()
    work = args.work.resolve()
    rows = json.loads((work / 'synthetic.json').read_text(encoding='utf8'))
    rows += json.loads((work / 'real-labels.json').read_text(encoding='utf-8-sig'))
    rows += list(historical_rows(args.legacy))
    sources, counts, manifest = {}, Counter(), []
    conflicts = set()
    owners, packs = {}, {}
    for r in rows:
        split, origin = r['split'], r['origin']
        recording = work / r['file']
        packed_index = r.get('audio_index')
        for group in [(str(recording), packed_index), r['source'], r.get('group'), r.get('speaker') if origin == 'tts' else None]:
            if group and owners.setdefault((origin, group), split) != split:
                raise ValueError('Source shared between splits')
        if packed_index is None:
            audio = read_wav(recording)[r['start']:]
        else:
            if recording not in packs:
                packs[recording] = np.load(recording, mmap_mode='r')
            packed = packs[recording][packed_index]
            if packed.dtype != np.int16 or packed.shape != (32000,):
                raise ValueError(f'Invalid packed PCM at {recording}')
            audio = packed.astype(np.float32) / 32768
        if origin == 'tts' and len(audio) > 32000:
            # The label describes the complete sentence.
            continue
        audio = audio[:32000]
        content = hashlib.sha256(audio.tobytes()).hexdigest()
        if content in sources:
            earlier = sources[content][0]
            if any(earlier[k] != r[k] for k in ('split', 'wake', 'command')):
                conflicts.add(content)
            continue
        copies = {'tts': {'train': 16, 'validation': 7, 'test': 14}[split],
                  'drive': 12 if split == 'train' else 1, 'legacy': 1}[origin]
        seed = int(content[:16], 16)
        sources[content] = (r, audio, copies, seed)
    for content in conflicts:
        del sources[content]
    for r, _, copies, _ in sources.values():
        counts[r['split'], r['origin']] += copies
    print(f'Excluded {len(conflicts)} conflicting or cross-split PCM hashes', flush=True)
    audio_temp = work / 'audio.tmp.npy'
    manifest_temp = work / 'windows.tmp.jsonl'
    store = np.lib.format.open_memmap(audio_temp, mode='w+', dtype='<i2', shape=(sum(counts.values()), 32000))
    for content, (r, audio, copies, seed) in sources.items():
        rng = np.random.default_rng(seed)
        for i in range(copies):
            j = len(manifest)
            store[j] = np.rint(variant(audio, i, rng) * 32768).clip(-32768, 32767).astype('<i2')
            manifest.append(dict(index=j, variant=i, pcm_sha256=content, **r))
    store.flush()
    del store
    with manifest_temp.open('w', encoding='utf8') as out:
        for r in manifest:
            out.write(json.dumps(r, ensure_ascii=False) + '\n')
    summary = {s: {o: counts[s, o] for o in ('tts', 'drive', 'legacy')} for s in ('train', 'validation', 'test')}
    audio_temp.replace(work / 'audio.npy')
    manifest_temp.replace(work / 'windows.jsonl')
    counts_temp = work / 'counts.tmp.json'
    counts_temp.write_text(json.dumps(summary, indent=2), encoding='utf8')
    counts_temp.replace(work / 'counts.json')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
