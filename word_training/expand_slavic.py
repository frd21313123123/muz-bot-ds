"""Add three Cyrillic BOT voices; keep Russian held-out audio immutable."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.runtime/wake-validation'),str(ROOT/'scripts/wakeword'),str(Path(__file__).parent)]
import numpy as np
import torch
from scipy.signal import resample_poly
from wake_data import atomic_json,prepare_source,stable_seed
from wake_data_v2 import augment
from wake_audio import WhisperMelFrontend,read_wav,write_wav,N_SAMPLES,right_aligned_window
from wake_tts import download,Synthesizer
from data import text_sets,contains_word
from train import cache_files

MODEL='uk_UA-ukrainian_tts-medium'
REVISION='c9f71aea53a7ebfe5934f33aa28016141392c8cf'
BASE='https://huggingface.co/rhasspy/piper-voices/resolve/'+REVISION+'/uk/uk_UA/ukrainian_tts/medium/'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',default='bot-data-v3');parser.add_argument('--run',default='bot-data-v4')
    parser.add_argument('--download-only',action='store_true');args=parser.parse_args()
    runs=ROOT/'word_training/runs';source=runs/args.source;work=runs/args.run
    if source.parent.resolve()!=runs.resolve() or work.parent.resolve()!=runs.resolve():raise ValueError('Use local run names')
    work.mkdir(exist_ok=True);tts=work/'tts/piper'
    provenance={}
    for name in [MODEL+'.onnx',MODEL+'.onnx.json','MODEL_CARD']:
        provenance[name]=download(BASE+name,tts/name)
    atomic_json(work/'tts_provenance.json',dict(repository='rhasspy/piper-voices',revision=REVISION,files=provenance,
        voices=['lada','mykyta','tetiana'],purpose='Cyrillic phonetic BOT proxies; spoken locale uk_UA, not native Russian samples'))
    if args.download_only:print('[slavic] public TTS weights ready',flush=True);return
    destination=work/'dataset';destination.mkdir(exist_ok=True)
    if (destination/'manifest.jsonl').exists():print('[slavic] dataset already complete',flush=True);return
    for name in ['sources','waveforms','features']:(destination/name).mkdir(exist_ok=True)
    config=json.loads((source/'run_config.json').read_text(encoding='utf-8'))
    config.update(work_dir=str(work),data_recipe='V3 Russian holdouts unchanged; three Ukrainian Cyrillic BOT proxy voices')
    atomic_json(work/'run_config.json',config);cache_files(source/'whisper-encoder',work/'whisper-encoder')
    rows=[json.loads(line) for line in (source/'dataset/manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    for row in rows:
        for key in ['wav','feature']:row[key]=str((source/'dataset'/row[key]).resolve())
    original=[(r['id'],r['label'],r['source_id']) for r in rows if r['split']!='train']
    neural=Synthesizer(work,dict(engine='piper',voice=MODEL,model='piper'),'cpu',2)
    voice_config=json.loads((tts/(MODEL+'.onnx.json')).read_text(encoding='utf-8'))
    if voice_config['num_speakers']!=3:raise ValueError('Wrong voice model')
    supported=set(voice_config['phoneme_id_map']);positive,negative=text_sets('бот')
    # Text phonemization silently drops unknown Cyrillic letters. A negative
    # such as «боты» must never become audible «бот» after dropping «ы».
    def valid(text):return all(char in supported for char in text if char.isalnum())
    positive=[text for text in positive if valid(text)];negative=[text for text in negative if valid(text)]
    frontend=WhisperMelFrontend(work/'whisper-encoder');torch.set_num_threads(2)
    noise_rows=[r for r in rows if r['split']=='train' and r['generator']=='real_speech'][:64]
    noise_pool=[read_wav(r['wav']) for r in noise_rows]
    from piper import SynthesisConfig
    for voice,speaker in voice_config['speaker_id_map'].items():
        pool={0:[],1:[]}
        for label,texts in [(0,negative),(1,positive)]:
            for text in texts:
                for variant in range(6 if label else 3):
                    rendered=text+['.','!','?'][variant%3]
                    speed=[1.,.85,1.15][variant%3]
                    identifier=hashlib.sha256(f'uk-bot|{voice}|{rendered}|{speed}|{config["seed"]}'.encode()).hexdigest()[:24]
                    file=destination/'sources'/(identifier+'.wav')
                    if not file.exists():
                        np.random.seed(stable_seed(identifier));chunks=list(neural.model.synthesize(rendered,SynthesisConfig(speaker_id=speaker,length_scale=1/speed,noise_scale=.667,noise_w_scale=.8)))
                        audio=np.concatenate([chunk.audio_float_array for chunk in chunks]);sr=chunks[0].sample_rate
                        from math import gcd
                        divisor=gcd(sr,16000);audio=resample_poly(audio,16000//divisor,sr//divisor).astype(np.float32)
                        write_wav(file,prepare_source(audio))
                    if len(read_wav(file))<=N_SAMPLES-480:pool[label].append((identifier,rendered))
        waves=np.lib.format.open_memmap(destination/'waveforms'/(voice+'.npy'),mode='w+',dtype=np.int16,shape=(3000,N_SAMPLES))
        features=np.lib.format.open_memmap(destination/'features'/(voice+'.npy'),mode='w+',dtype=np.float16,shape=(3000,1,80,200))
        offset=0
        for label,count in [(0,1800),(1,1200)]:
            rng=np.random.default_rng(config['seed']+stable_seed(f'{voice}|{label}'))
            ordered=[pool[label][i] for i in rng.permutation(len(pool[label]))]
            isolated=[r for r in ordered if len(re.findall(r'\w+',r[1]))==1]
            phrases=[r for r in ordered if len(re.findall(r'\w+',r[1]))>1]
            if not ordered or label and (not isolated or not phrases):raise ValueError('Incomplete voice coverage')
            for index in range(count):
                group=isolated if label and index<600 else (phrases if label else ordered)
                identifier,text=group[index%len(group)]
                if contains_word(text,'бот')!=bool(label):raise ValueError('Wrong whole-word label')
                audio=read_wav(destination/'sources'/(identifier+'.wav'))
                window=augment(audio,rng,label,preserve_context=True,train=True,noise_pool=noise_pool)
                if window is None:window=right_aligned_window(audio)
                pcm=np.rint(np.clip(window,-1,32767/32768)*32768).astype(np.int16)
                waves[offset]=pcm;features[offset]=frontend(pcm.astype(np.float32)/32768)
                rows.append(dict(id=f'piper-uk-{voice}-{label}-{index:05d}',wav=f'waveforms/{voice}.npy',audio_index=offset,feature=f'features/{voice}.npy',feature_index=offset,
                    label=label,split='train',language='uk',sampling_language='ru',speaker=f'piper-uk/{voice}',speaker_group=f'piper/uk/{voice}',source_id=f'piper-uk/{identifier}',
                    text=text,phonetic_text=text,generator='piper_uk',kind='positive_isolated' if label and index<600 else ('positive_phrase' if label else 'ordinary')))
                offset+=1
        waves.flush();features.flush();del waves,features
        print(f'[slavic] {voice}: 1200 BOT + 1800 negatives',flush=True)
    assert original==[(r['id'],r['label'],r['source_id']) for r in rows if r['split']!='train']
    assert len({r['id'] for r in rows})==len(rows)
    manifest=destination/'manifest.tmp';manifest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8');manifest.replace(destination/'manifest.jsonl')
    atomic_json(work/'dataset_summary.json',dict(windows=len(rows),train_windows=sum(r['split']=='train' for r in rows),added_positive=3600,added_negative=5400,
        added_voices=list(voice_config['speaker_id_map']),spoken_locale='uk_UA',held_out_windows_unchanged=True,unsupported_letters_rejected=True))
    print('[slavic] dataset complete',flush=True)


if __name__=='__main__':main()
