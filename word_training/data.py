"""Russian whole-word audio labels: target anywhere in an utterance is positive."""
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np
import torch
from wake_audio import N_SAMPLES, WhisperMelFrontend, read_wav, write_wav
from wake_data import HARD_RU, RU_SHORT, SPLITS, atomic_json, background, prepare_source, stable_seed
from wake_data_v2 import augment
from wake_tts import Synthesizer, voice_jobs


def contains_word(text, target):
    return target.casefold() in re.findall(r'\w+', text.casefold())


def text_sets(target):
    templates = ['{w}', 'эй {w}', 'привет {w}', '{w} привет', 'это {w}', 'где {w}',
                 'наш {w}', 'мне нужен {w}', '{w} слушай', 'слушай {w}', '{w} включи',
                 '{w} музыка', '{w} включи музыку', '{w} пожалуйста', 'сегодня {w}', '{w} работает']
    positives = [template.format(w=target) for template in templates]
    negatives = [*HARD_RU, *RU_SHORT, 'включи музыку', 'выключи музыку', 'покажи песню',
                 'найди радио', 'добавь трек', 'убери песню', 'послушай меня', 'я тебя слышу',
                 'почему так', 'позвони мне', 'сейчас играет', 'песня закончилась']
    for other in ['кот','борт','вот','робот','болт','боту']:
        negatives += [template.format(w=other) for template in templates[1:7]]
    negatives = sorted({text for text in negatives if not contains_word(text,target) and text.casefold() != 'бод'})
    assert all(contains_word(text,target) for text in positives)
    return positives, negatives


def prepare_real_audio(audio):
    """Trim quiet annotated speech relative to its peak; reject genuinely empty audio."""
    peak=float(np.max(np.abs(audio))) if len(audio) else 0.
    if np.isfinite(peak) and peak>1e-7: audio=audio*(.95/peak)
    return prepare_source(audio)


def generate(config):
    work = Path(config['work_dir']); destination = work/'dataset'
    for name in ['sources','features','wav']: (destination/name).mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(config['cpu_threads'])
    device='cuda:0'
    torch.cuda.set_device(0)
    frontend=WhisperMelFrontend(work/'whisper-encoder')
    positives, negatives = text_sets(config['wake_word_ru'])
    jobs=voice_jobs(config)
    rows=[]; audit=[]; synthesizer=None; previous=None
    real=json.loads((work/'real_sources.json').read_text(encoding='utf-8'))
    noise_pool=[read_wav(row['file']) for row in real if row['split']=='train'][:64]
    start=time.monotonic()
    def save(identifier,audio,metadata):
        file=destination/'wav'/(identifier+'.wav')
        write_wav(file,audio)
        np.save(destination/'features'/(identifier+'.npy'),frontend(read_wav(file)))
        rows.append(dict(id=identifier,wav=f'wav/{identifier}.wav',feature=f'features/{identifier}.npy',**metadata))
    for job in jobs:
        key=(job['engine'],job['model'],job['voice'] if job['engine']=='piper' else None)
        if previous!=key:
            if synthesizer is not None: del synthesizer
            torch.cuda.empty_cache()
            synthesizer=Synthesizer(work,job,device,config['cpu_threads']); previous=key
        synthesizer.job=job
        split=job['split']; speaker=f'{job["model"]}/{job["voice"]}'
        buckets={0:[],1:[]}
        for label,texts in [(1,positives),(0,negatives)]:
            variants=(config['positive_tts_variants'] if label else config['negative_tts_variants']) if split=='train' else (4 if label else 1)
            for text in texts:
                for variant in range(variants):
                    # Prosody is balanced across classes; punctuation never defines the label.
                    mark=['','.','!','?'][(stable_seed(text)+variant)%4]
                    rendered=text+mark
                    rate=['normal','slow','fast'][variant%3] if split=='train' else 'normal'
                    seed=stable_seed(f'{speaker}|{rendered}|{rate}|{variant}|{config["seed"]}')
                    source_id=hashlib.sha256(f'word-v1|{speaker}|{rendered}|{rate}|{seed}'.encode()).hexdigest()[:24]
                    file=destination/'sources'/(source_id+'.wav')
                    if not file.exists():
                        generated=synthesizer.say(rendered,rate,seed)
                        try: audio=prepare_source(generated)
                        except ValueError as error:
                            audit.append(dict(source_id=source_id,text=rendered,speaker=speaker,reason=str(error))); continue
                        temporary=file.with_suffix('.tmp')
                        write_wav(temporary,audio)
                        temporary.replace(file)
                    audio=read_wav(file)
                    # Never retain a positive label after cropping away the keyword.
                    if len(audio)>N_SAMPLES-480:
                        audit.append(dict(source_id=source_id,text=rendered,speaker=speaker,reason='Complete utterance exceeds 2-second contract')); continue
                    buckets[label].append((source_id,rendered))
        for label in [0,1]:
            if not buckets[label]: raise ValueError(f'No usable class {label} for {speaker}')
            count=config['positive_per_voice' if label else 'negative_per_voice'] if split=='train' else config['eval_positive_per_voice' if label else 'eval_negative_per_voice']
            rng=np.random.default_rng(config['seed']+stable_seed(f'{speaker}:{label}'))
            # Round-robin sources before repetition keeps whole-word/phrase coverage visible.
            pool=[buckets[label][i] for i in rng.permutation(len(buckets[label]))]
            isolated=[entry for entry in pool if len(re.findall(r'\w+',entry[1]))==1] if label else []
            phrases=[entry for entry in pool if len(re.findall(r'\w+',entry[1]))>1] if label else []
            fraction=config.get('isolated_positive_fraction')
            isolated_count=int(round(count*fraction)) if label and fraction is not None else None
            for index in range(count):
                if isolated_count is not None:
                    selected=isolated if index<isolated_count else phrases
                    if not selected: raise ValueError(f'Incomplete positive coverage for {speaker}')
                    offset=index if index<isolated_count else index-isolated_count
                    source_id,text=selected[offset%len(selected)]
                else: source_id,text=pool[index%len(pool)]
                identifier=f'{job["engine"]}-{job["model"].replace("/","-")}-{job["voice"]}-{label}-{index:05d}'
                kind='positive_isolated' if label and len(re.findall(r'\w+',text))==1 else ('positive_phrase' if label else ('hard_word' if len(re.findall(r'\w+',text))==1 else 'ordinary'))
                metadata=dict(label=label,split=split,language='ru',speaker=speaker,
                    speaker_group=job['speaker_group'],source_id=source_id,text=text,generator=job['engine'],kind=kind)
                if (destination/'wav'/(identifier+'.wav')).exists() and (destination/'features'/(identifier+'.npy')).exists():
                    rows.append(dict(id=identifier,wav=f'wav/{identifier}.wav',feature=f'features/{identifier}.npy',**metadata))
                    continue
                audio=read_wav(destination/'sources'/(source_id+'.wav'))
                if config['independent_pitch']:
                    import librosa
                    audio=librosa.effects.pitch_shift(audio,sr=16000,n_steps=float(rng.uniform(-1.8,1.8)))
                window=augment(audio,rng,label,preserve_context=True,train=split=='train',noise_pool=noise_pool)
                if window is None:
                    # Fallback preserves the complete annotated utterance, never crops positives.
                    from wake_audio import right_aligned_window
                    window=right_aligned_window(audio)
                save(identifier,window,metadata)
        atomic_json(work/'generation_progress.json',dict(stage='generation',completed_voices=jobs.index(job)+1,total_voices=len(jobs),windows=len(rows),speaker=speaker,elapsed_seconds=time.monotonic()-start))
        print(f'[TTS] {speaker} -> {split}: {sum(r["speaker"]==speaker for r in rows)} windows; usable sources positive={len(buckets[1])}, negative={len(buckets[0])}; {time.monotonic()-start:.0f}s',flush=True)
    if synthesizer is not None: del synthesizer
    torch.cuda.empty_cache()
    for row in real:
        rng=np.random.default_rng(config['seed']+stable_seed(row['id']))
        try: audio=prepare_real_audio(read_wav(row['file']))
        except ValueError as error:
            audit.append(dict(source_id=row['id'],text=row['word'],speaker=row['speaker'],reason=str(error)))
            continue
        window=augment(audio,rng,0,train=row['split']=='train',noise_pool=noise_pool)
        if window is None: continue
        save(row['id'],window,dict(label=0,split=row['split'],language='en',speaker=row['speaker'],speaker_group=row['speaker_group'],
             source_id=row['id'],text=row['word'],generator='real_speech',kind='real_word'))
    for index in range(config['noise_windows']):
        rng=np.random.default_rng(config['seed']+100000+index)
        save(f'noise-{index:05d}',background(rng,N_SAMPLES)*10**(rng.uniform(-45,-12)/20),
             dict(label=0,split=SPLITS[index%4],language='none',speaker='noise',speaker_group='noise',source_id=f'noise-{index}',text='',generator='procedural_noise',kind='noise'))
    identities={}; sources={}
    for row in rows:
        if row['speaker_group']!='noise':
            if identities.setdefault(row['speaker_group'],row['split'])!=row['split']: raise ValueError('Speaker leakage')
        if sources.setdefault(row['source_id'],row['split'])!=row['split']: raise ValueError('Source leakage')
    manifest=destination/'manifest.jsonl'
    temp=manifest.with_suffix('.tmp')
    temp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8'); temp.replace(manifest)
    summary={split:{'positive':sum(r['split']==split and r['label']==1 for r in rows),'negative':sum(r['split']==split and r['label']==0 for r in rows)} for split in SPLITS}
    atomic_json(work/'dataset_summary.json',summary); atomic_json(work/'source_audit.json',audit)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return rows
