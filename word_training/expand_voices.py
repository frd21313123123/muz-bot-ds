"""Add eight British neural voices; keep Russian validation/calibration/test fixed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.runtime/wake-validation'),str(ROOT/'scripts/wakeword'),str(Path(__file__).parent)]
import numpy as np
import torch
from train import cache_files
from wake_tts import Synthesizer
from wake_audio import WhisperMelFrontend,read_wav,write_wav,N_SAMPLES
from wake_data import prepare_source,atomic_json,stable_seed
from wake_data_v2 import augment

WORDS={'bot':'бот','vot':'вот','cot':'кот','pot':'пот','rot':'рот','dot':'дот','hot':'хот','not':'нот',
       'boat':'боут','boot':'бут','bat':'бэт','bit':'бит','bet':'бет','bolt':'боулт','robot':'роубот','bottle':'ботл','body':'боди','box':'бокс','botu':'боту'}
VOICES=['bf_alice','bf_emma','bf_isabella','bf_lily','bm_daniel','bm_fable','bm_george','bm_lewis']


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',default='bot-v2');parser.add_argument('--run',default='bot-data-v3');args=parser.parse_args()
    base=ROOT/'word_training/runs'/args.source;work=ROOT/'word_training/runs'/args.run
    if base.parent.resolve()!=(ROOT/'word_training/runs').resolve() or work.parent.resolve()!=base.parent.resolve():raise ValueError('Use local run names')
    destination=work/'dataset';destination.mkdir(parents=True,exist_ok=True)
    if (destination/'manifest.jsonl').exists():print('[voices] already complete',flush=True);return
    for name in ['sources','waveforms','features']:(destination/name).mkdir(exist_ok=True)
    config=json.loads((base/'run_config.json').read_text(encoding='utf-8'))
    config.update(work_dir=str(work),data_recipe='Russian v2 holdouts unchanged; eight British Kokoro training voices, phonetic minimal pairs',target_aliases=['бот','bot (en-gb)'])
    atomic_json(work/'run_config.json',config)
    cache_files(base/'whisper-encoder',work/'whisper-encoder')
    cache_files(ROOT/'.runtime/wakeword-v2-local/tts/kokoro',work/'tts/kokoro')
    rows=[json.loads(line) for line in (base/'dataset/manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    for row in rows:
        row['wav']=str((base/'dataset'/row['wav']).resolve());row['feature']=str((base/'dataset'/row['feature']).resolve())
    frontend=WhisperMelFrontend(work/'whisper-encoder');torch.set_num_threads(2)
    real=[row for row in rows if row['split']=='train' and row['generator']=='real_speech'][:64]
    noise_pool=[read_wav(row['wav']) for row in real]
    synthesizer=Synthesizer(work,dict(engine='kokoro',model='kokoro-v1.0',voice=VOICES[0]),'cpu',2)
    for voice in VOICES:
        synthesizer.job['voice']=voice;pool={0:[],1:[]}
        for text,canonical in WORDS.items():
            for variant in range(6):
                rendered=text+['.','!'][variant%2];rate=['normal','slow','fast'][variant%3]
                seed=stable_seed(f'gb-bot|{voice}|{rendered}|{rate}|{config["seed"]}')
                identifier=hashlib.sha256(f'{voice}|{rendered}|{rate}|{seed}'.encode()).hexdigest()[:24]
                file=destination/'sources'/(identifier+'.wav')
                if not file.exists():write_wav(file,prepare_source(synthesizer.say(rendered,rate,seed)))
                if len(read_wav(file))>N_SAMPLES-480:continue
                pool[int(text=='bot')].append((identifier,text,canonical))
        waves=np.lib.format.open_memmap(destination/'waveforms'/(voice+'.npy'),mode='w+',dtype=np.int16,shape=(2000,N_SAMPLES))
        features=np.lib.format.open_memmap(destination/'features'/(voice+'.npy'),mode='w+',dtype=np.float16,shape=(2000,1,80,200))
        for label in [0,1]:
            rng=np.random.default_rng(config['seed']+stable_seed(f'{voice}:{label}'))
            ordered=[pool[label][i] for i in rng.permutation(len(pool[label]))]
            for index in range(1000):
                source_id,text,canonical=ordered[index%len(ordered)]
                audio=read_wav(destination/'sources'/(source_id+'.wav'))
                window=augment(audio,rng,label,preserve_context=True,train=True,noise_pool=noise_pool)
                if window is None:raise ValueError('Unexpected overlong one-word English utterance')
                offset=label*1000+index
                pcm=np.rint(np.clip(window,-1,32767/32768)*32768).astype(np.int16)
                waves[offset]=pcm;features[offset]=frontend(pcm.astype(np.float32)/32768)
                rows.append(dict(id=f'kokoro-gb-{voice}-{label}-{index:05d}',wav=f'waveforms/{voice}.npy',audio_index=offset,feature=f'features/{voice}.npy',feature_index=offset,label=label,split='train',language='en',speaker=f'kokoro-v1.0/{voice}',speaker_group=f'kokoro/{voice}',source_id=f'kokoro-gb/{source_id}',text=text,phonetic_text=canonical,generator='kokoro',kind='positive_isolated' if label else 'hard_word'))
        waves.flush();features.flush();del waves,features
        print(f'[voices] {voice}: 1000 bot + 1000 hard negatives',flush=True)
    assert len({r['id'] for r in rows})==len(rows)
    original=[r for r in rows if r['split']!='train']
    old=[json.loads(line) for line in (base/'dataset/manifest.jsonl').read_text(encoding='utf-8').splitlines() if json.loads(line)['split']!='train']
    assert [(r['id'],r['label'],r['source_id']) for r in original]==[(r['id'],r['label'],r['source_id']) for r in old]
    temporary=destination/'manifest.tmp';temporary.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8');temporary.replace(destination/'manifest.jsonl')
    atomic_json(work/'dataset_summary.json',dict(windows=len(rows),train_windows=sum(r['split']=='train' for r in rows),added_voices=VOICES,added_positive=8000,added_negative=8000,held_out_windows_unchanged=True,neural_tts='Kokoro ONNX v1.0, en-gb',phonetic_annotations=WORDS))
    print('[voices] dataset complete',flush=True)


if __name__=='__main__':main()
