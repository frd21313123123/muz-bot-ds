"""Speaker-held-out multi-engine dataset with deliberate hard negatives and real speech."""
import argparse
import hashlib
import heapq
import json
import re
from pathlib import Path
import tarfile
import time
import zipfile
import numpy as np
import torch
from scipy.signal import fftconvolve, resample_poly
from wake_audio import SAMPLE_RATE, N_SAMPLES, read_wav, write_wav, right_aligned_window
from wake_data import atomic_json, background, negative_texts, prepare_source, stable_seed, SPLITS
from wake_tts import Synthesizer, voice_jobs, download_models, download

REAL_URLS = {'mini': 'https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip',
             'full': 'https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz'}
REAL_WORDS = set('backward bed bird cat dog down eight five follow forward four go happy house learn left marvin nine no off on one right seven sheila six stop three tree two up visual wow yes zero'.split())


def speaker_split(speaker):
    bucket = stable_seed('real-speaker:' + speaker) % 100
    return 'train' if bucket < 76 else 'validation' if bucket < 84 else 'calibration' if bucket < 92 else 'test'


def prepare_real(config):
    work = Path(config['work_dir'])
    count = config.get('real_speech_windows', 12000)
    if not count:
        atomic_json(work / 'real_sources.json', [])
        return
    source = config.get('real_speech_source', 'full')
    archive = work / 'downloads' / ('mini.zip' if source == 'mini' else 'speech-commands-v002.tar.gz')
    provenance = download(REAL_URLS[source], archive)
    provenance['license_reference'] = 'https://www.tensorflow.org/datasets/catalog/speech_commands'
    provenance['description'] = 'Known-word negatives only; speaker assigned before augmentation; no real wake positives.'
    target = {config['wake_word_ru'].lower(), config['wake_word_en'].lower()}
    destination = work / 'dataset/real-sources'
    destination.mkdir(parents=True, exist_ok=True)
    budgets = dict(train=int(count * .76), validation=int(count * .08), calibration=int(count * .08))
    budgets['test'] = count - sum(budgets.values())
    reservoirs = {split: [] for split in SPLITS}
    backgrounds = []
    def accept(name, data):
        parts = name.replace('\\', '/').split('/')
        if len(parts) < 2: return
        word, filename = parts[-2:]
        if not re.fullmatch(r'[A-Za-z0-9_]+\.wav', filename): return
        if word == '_background_noise_':
            path = destination / ('background-' + filename)
            path.write_bytes(data)
            backgrounds.append(str(path))
            return
        if word not in REAL_WORDS or word in target or '_nohash_' not in filename: return
        speaker = filename.split('_nohash_')[0]
        identifier = 'real-' + hashlib.sha256((word + '/' + filename).encode()).hexdigest()[:24]
        split = speaker_split(speaker)
        priority = stable_seed(identifier + str(config['seed']))
        entry = (-priority, identifier, dict(id=identifier, word=word, speaker='speechcommands/' + speaker,
                           speaker_group='speechcommands/' + speaker, split=split, data=data))
        reservoir, budget = reservoirs[split], budgets[split]
        if not budget: return
        if len(reservoir) < budget: heapq.heappush(reservoir, entry)
        elif priority < -reservoir[0][0]: heapq.heapreplace(reservoir, entry)
    # Read member bytes only; never extract archive paths or links to the filesystem.
    if source == 'mini':
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                if member.filename.endswith('.wav') and member.file_size <= 8_000_000:
                    accept(member.filename, zipped.read(member))
    else:
        with tarfile.open(archive, 'r|gz') as tar:
            for member in tar:
                if member.isfile() and member.name.endswith('.wav') and member.size <= 8_000_000:
                    accept(member.name, tar.extractfile(member).read())
    selected = []
    for split in SPLITS:
        group = [item[2] for item in reservoirs[split]]
        group.sort(key=lambda r: stable_seed(r['id'] + str(config['seed'])))
        for row in group[:budgets[split]]:
            payload = row.pop('data')
            file = destination / (row['id'] + '.wav')
            file.write_bytes(payload)
            row['file'] = str(file)
            selected.append(row)
    atomic_json(work / 'real_sources.json', selected)
    atomic_json(work / 'real_backgrounds.json', backgrounds)
    provenance['selected_windows'] = len(selected)
    provenance['corpus'] = source
    atomic_json(work / 'real_provenance.json', provenance)
    print(f'Real speech negatives: {len(selected)}; background recordings: {len(backgrounds)}', flush=True)


def negative_kind(text, target):
    tokens = re.findall(r"[\w]+", text.lower())
    if target.lower() in tokens: return 'context'
    return 'hard_word' if len(tokens) == 1 else 'ordinary'


def prepare_human(config):
    """Optional reviewed recordings: labels come from annotation, not an ASR teacher."""
    if not config.get('real_wake_manifest'): return
    work = Path(config['work_dir'])
    manifest = Path(config['real_wake_manifest'])
    records = [r for r in json.loads((work / 'real_sources.json').read_text(encoding='utf-8'))
               if r.get('generator') != 'human_reviewed']
    groups, audio_groups = {}, {}
    destination = work / 'dataset/real-sources'
    destination.mkdir(parents=True, exist_ok=True)
    for line in manifest.read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        row = json.loads(line)
        if row.get('reviewed') is not True or type(row.get('label')) is not int or row['label'] not in [0, 1]:
            raise ValueError('Human recordings require reviewed=true and an explicit binary label')
        if row.get('split') not in SPLITS or row.get('language') not in ['ru', 'en'] or not row.get('speaker'):
            raise ValueError('Human recordings require speaker, language and split')
        group = 'human/' + str(row['speaker'])
        if groups.setdefault(group, row['split']) != row['split']: raise ValueError('Human speaker leakage')
        file = Path(row['wav'])
        if not file.is_absolute(): file = manifest.parent / file
        audio = prepare_source(read_wav(file))
        if len(audio) > N_SAMPLES - 480: raise ValueError('Human annotations must contain a complete utterance shorter than two seconds')
        sha = hashlib.sha256(audio.tobytes()).hexdigest()
        signature = (row['split'], row['label'])
        if audio_groups.setdefault(sha, signature) != signature: raise ValueError('Human audio leakage or contradictory labels')
        identifier = 'human-' + sha[:24]
        if any(r['id'] == identifier for r in records): continue
        saved = destination / (identifier + '.wav')
        write_wav(saved, audio)
        records.append(dict(id=identifier, word=row.get('text', ''), speaker=group, speaker_group=group,
                            split=row['split'], label=row['label'], language=row['language'], generator='human_reviewed', file=str(saved)))
    atomic_json(work / 'real_sources.json', records)


def augment(audio, rng, label, preserve_context=False, *, train=False, noise_pool=None):
    ratio = int(rng.integers(93, 108))
    speech = resample_poly(audio, ratio, 100).astype(np.float32)
    if len(speech) > N_SAMPLES - 480:
        if label or preserve_context: return None
        start = int(rng.integers(len(speech) - (N_SAMPLES - 480) + 1))
        speech = speech[start:start + N_SAMPLES - 480]
    if rng.random() < .3:
        ir = rng.normal(0, .025, int(rng.integers(400, 2400)))
        ir *= np.exp(-np.arange(len(ir)) / rng.uniform(160, 850))
        ir[0] = 1
        speech = fftconvolve(speech, ir)[:len(speech)].astype(np.float32)
    speech *= 10 ** (rng.uniform(-22, -3) / 20) / (np.max(np.abs(speech)) + 1e-6)
    window = np.zeros(N_SAMPLES, np.float32)
    # Match exported inference's right-aligned utterance; retain some position variation.
    start = N_SAMPLES - len(speech) if rng.random() < .65 else int(rng.integers(N_SAMPLES - len(speech) + 1))
    window[start:start + len(speech)] = speech
    if rng.random() < .65:
        if train and noise_pool and rng.random() < .6:
            noise = noise_pool[int(rng.integers(len(noise_pool)))]
            offset = int(rng.integers(max(1, len(noise) - N_SAMPLES + 1)))
            noise = np.resize(noise[offset:offset + N_SAMPLES], N_SAMPLES).copy()
            noise /= max(float(np.sqrt(np.mean(noise ** 2))), 1e-6)
        else: noise = background(rng, N_SAMPLES)
        snr = rng.uniform(8, 35)  # Most samples remain audible; no wholesale low-SNR label corruption.
        window += noise * max(float(np.sqrt(np.mean(speech ** 2))), 1e-4) / 10 ** (snr / 20)
    if rng.random() < .15:
        window = resample_poly(resample_poly(window, 1, 2), 2, 1)[:N_SAMPLES]
    if train and rng.random() < .12:
        # Mild saturation/quantization; does not claim exact Opus packet simulation.
        window = np.tanh(window * rng.uniform(1, 2))
        window = np.round(window * 2047) / 2047
    peak = np.max(np.abs(window))
    if peak > .98: window *= .98 / peak
    return window.astype(np.float32)


def generate(config, rank=0, world=1):
    work = Path(config['work_dir'])
    torch.set_num_threads(config.get('cpu_threads', 2))
    device = f'cuda:{rank}' if torch.cuda.is_available() and not config.get('force_cpu') else 'cpu'
    if device != 'cpu': torch.cuda.set_device(rank)
    destination = work / 'dataset'
    for folder in ['wav', 'features', 'sources']: (destination / folder).mkdir(parents=True, exist_ok=True)
    real = json.loads((work / 'real_sources.json').read_text(encoding='utf-8')) if (work / 'real_sources.json').exists() else []
    backgrounds = json.loads((work / 'real_backgrounds.json').read_text(encoding='utf-8')) if (work / 'real_backgrounds.json').exists() else []
    noise_pool = [read_wav(p) for p in backgrounds]
    # Training-only real background speech. Held-out speakers never enter training mixtures.
    noise_pool += [read_wav(r['file']) for r in real if r['split'] == 'train' and r.get('generator', 'real_speech') == 'real_speech'][:64]
    rows, audit = [], []
    started = time.monotonic()
    synthesizer, key = None, None
    from wake_audio import make_frontend
    frontend = make_frontend(config, work / ('whisper-encoder' if config.get('architecture') == 'whisper_encoder_v4' else 'speech-embedding'))
    def save(identifier, audio, row):
        write_wav(destination / 'wav' / (identifier + '.wav'), audio)
        np.save(destination / 'features' / (identifier + '.npy'), frontend(read_wav(destination / 'wav' / (identifier + '.wav'))))
        rows.append(dict(id=identifier, wav=f'wav/{identifier}.wav', feature=f'features/{identifier}.npy', **row))
    for job in voice_jobs(config)[rank::world]:
        engine, model, language, speaker, split = [job[k] for k in ['engine', 'model', 'language', 'voice', 'split']]
        model_key = (engine, model, speaker if engine == 'piper' else None)
        if model_key != key:
            if synthesizer is not None: del synthesizer
            if device != 'cpu': torch.cuda.empty_cache()
            synthesizer = Synthesizer(work, job, device, config.get('cpu_threads', 2))
            key = model_key
        synthesizer.job = job
        target = config['wake_word_ru' if language == 'ru' else 'wake_word_en'].strip()
        positives = [target + mark for mark in ['.', '!', '?']]
        if config.get('positive_bare_word'): positives.append(target)
        if not config['standalone_only']: positives += [f'{target}, привет.' if language == 'ru' else f'{target}, hello.']
        negatives = negative_texts(language, target, config['standalone_only'],
                                   config.get('negative_exclusions', {}).get(language, []))
        if config.get('negative_source_limit'):
            # Bounded development experiments retain every negative category.
            categories = [[t for t in negatives if negative_kind(t, target) == kind]
                          for kind in ['hard_word', 'context', 'ordinary']]
            negatives = [t for i in range(max(map(len, categories))) for category in categories for t in category[i:i+1]][:config['negative_source_limit']]
        if config.get('smoke'):
            negatives = [next(t for t in negatives if negative_kind(t, target) == kind)
                         for kind in ['hard_word', 'context', 'ordinary']] if config['standalone_only'] else negatives[:3]
        buckets = {kind: [] for kind in ['positive', 'hard_word', 'context', 'ordinary']}
        for label, texts in [(1, positives), (0, negatives)]:
            for text in texts:
                for variant in range(config.get('positive_tts_variants', 6) if label else (
                        config.get('negative_tts_variants', 1) if split == 'train' else 1)):
                    rate = ['normal', 'slow', 'fast'][variant % 3]
                    seed = stable_seed(f'{engine}:{model}:{speaker}:{text}:{variant}:{config["seed"]}')
                    source_id = hashlib.sha256(f'v2|{engine}|{model}|{speaker}|{text}|{rate}|{seed}'.encode()).hexdigest()[:24]
                    file = destination / 'sources' / (source_id + '.wav')
                    if not file.exists():
                        generated = synthesizer.say(text, rate, seed)
                        try: audio = prepare_source(generated)
                        except ValueError as error:
                            audit.append(dict(source_id=source_id, text=text, rate=rate, reason=str(error), speaker=job['speaker_group']))
                            continue
                        write_wav(file, audio)
                    audio = read_wav(file)
                    kind = 'positive' if label else negative_kind(text, target)
                    if len(audio) > N_SAMPLES - 480 and (label or kind == 'context'):
                        audit.append(dict(source_id=source_id, text=text, reason='overlong complete wake/context', speaker=job['speaker_group']))
                        continue
                    # Independent pitch shift, cached per source. Both classes get the same policy.
                    options = [audio]
                    if config.get('independent_pitch', True) and not config.get('smoke') and split == 'train':
                        import librosa
                        for steps in [-2., 2.]:
                            shifted_file = destination / 'sources' / f'{source_id}-pitch-{int(steps)}.wav'
                            if not shifted_file.exists():
                                shifted = librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=steps).astype(np.float32)
                                write_wav(shifted_file, shifted)
                            options.append(read_wav(shifted_file))
                    buckets[kind].append((source_id, text, options, kind))
        if not buckets['positive']: raise RuntimeError(f'No complete wake sources for {job}')
        multiplier = config.get('russian_multiplier', 2) if language == 'ru' else 1
        # Versions do not inflate the Russian sample budget; split it between renderings.
        if engine == 'silero' and language == 'ru': multiplier /= len(config.get('silero_ru_versions', ['v5_5_ru', 'v4_ru']))
        counts = {1: max(1, round(config['positive_per_voice'] * multiplier)),
                  0: max(1, round(config['negative_per_voice'] * multiplier))}
        if split != 'train' and not config.get('smoke'):
            counts = {1: max(1, round(config.get('eval_positive_per_voice', 200) * multiplier)),
                      0: max(1, round(config.get('eval_negative_per_voice', 800) * multiplier))}
        for label in [1, 0]:
            rng = np.random.default_rng(config['seed'] + stable_seed(f'{engine}:{model}:{speaker}:{label}'))
            written, attempts = 0, 0
            while written < counts[label]:
                attempts += 1
                if attempts > counts[label] * 40: raise RuntimeError(f'Too many rejected augmentations for {job}')
                if label: kind = 'positive'
                else:
                    available = [k for k in ['hard_word', 'context', 'ordinary'] if buckets[k]]
                    proportions = config.get('negative_kind_weights', {}) if split == 'train' else {}
                    weights = np.array([proportions.get(k, .5 if k == 'hard_word' else .25) for k in available])
                    kind = rng.choice(available, p=weights / weights.sum())
                source_id, text, variants, _ = buckets[kind][int(rng.integers(len(buckets[kind])))]
                audio = variants[int(rng.integers(len(variants)))]
                window = augment(audio, rng, label, kind == 'context', train=split == 'train', noise_pool=noise_pool)
                if window is None: continue
                identifier = f'{engine}-{model.split("/")[-1]}-{speaker}-{label}-{written:05d}'
                save(identifier, window, dict(label=label, split=split, language=language, speaker=f'{model}/{speaker}',
                     speaker_group=job['speaker_group'], source_id=source_id, text=text, generator=engine, kind=kind))
                written += 1
        print(f'[TTS {rank}] {engine}/{model}/{speaker} -> {split}: {sum(counts.values())} windows; provider={synthesizer.provider}', flush=True)
    for row in real[rank::world]:
        rng = np.random.default_rng(config['seed'] + stable_seed(row['id']))
        audio = read_wav(row['file'])
        try: audio = prepare_source(audio)
        except ValueError:
            # A quiet/empty negative word is still non-wake. Never relabel it positive.
            if row.get('label', 0) or not len(audio):
                audit.append(dict(source_id=row['id'], reason='empty real recording')); continue
        label = row.get('label', 0)
        human = row.get('generator') == 'human_reviewed'
        window = augment(audio, rng, label, preserve_context=human, train=row['split'] == 'train', noise_pool=noise_pool)
        if window is None and human:
            # A valid complete annotation may no longer fit after slowing it down.
            # Keep its original whole utterance instead of cropping or aborting.
            window = right_aligned_window(audio)
        if window is None: raise ValueError('Annotated human utterance cannot be cropped to preserve its label')
        save(row['id'], window, dict(label=label, split=row['split'], language=row.get('language', 'en'), speaker=row['speaker'],
             speaker_group=row['speaker_group'], source_id=row['id'], text=row['word'],
             generator=row.get('generator', 'real_speech'), kind='human_utterance' if human else 'real_word'))
    for index in range(rank, config['noise_windows'], world):
        rng = np.random.default_rng(config['seed'] + 100000 + index)
        identifier = f'noise-{index:05d}'
        save(identifier, background(rng, N_SAMPLES) * 10 ** (rng.uniform(-45, -12) / 20),
             dict(label=0, split=SPLITS[index % 4], language='none', speaker='noise', speaker_group='noise',
                  source_id=identifier, text='', generator='procedural_noise', kind='noise'))
    shard = destination / f'manifest-rank-{rank}.jsonl'
    temp = shard.with_suffix('.tmp')
    temp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    temp.replace(shard)
    atomic_json(work / f'source_audit-{rank}.json', audit)
    print(f'[TTS {rank}] finished {len(rows)} windows in {time.monotonic()-started:.1f}s', flush=True)


def merge_manifest(config, world):
    from wake_data import merge_manifest as merge
    rows, summary = merge(config, world)
    groups = {}
    for row in rows:
        if row.get('speaker_group', row['speaker']) != 'noise':
            identity = row.get('speaker_group', row['speaker'])
            assert groups.setdefault(identity, row['split']) == row['split'], 'Voice identity leakage across TTS versions'
    for split in SPLITS:
        selected = [r for r in rows if r['split'] == split and r['speaker'] != 'noise']
        summary[split]['voices'] = len({r.get('speaker_group', r['speaker']) for r in selected})
        summary[split]['tts_voice_groups'] = len({r['speaker_group'] for r in selected if r['generator'] in ['silero', 'piper', 'kokoro', 'mms']})
        summary[split]['tts_renderings'] = len({r['speaker'] for r in selected if r['generator'] in ['silero', 'piper', 'kokoro', 'mms']})
    atomic_json(Path(config['work_dir']) / 'dataset_summary.json', summary)
    details = {split: {language: {label: sum(r['split'] == split and r['language'] == language and r['label'] == label for r in rows)
                                 for label in [0, 1]} for language in ['ru', 'en', 'none']} for split in SPLITS}
    atomic_json(Path(config['work_dir']) / 'dataset_groups.json', details)
    return rows, summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world', type=int, default=1)
    args = parser.parse_args()
    generate(json.loads(Path(args.config).read_text(encoding='utf-8')), args.rank, args.world)
