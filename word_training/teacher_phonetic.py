"""Cache six acoustic posteriors from a pinned Russian teacher on the local GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.runtime/wake-validation'),str(ROOT/'scripts/wakeword'),str(Path(__file__).parent)]
import numpy as np
import torch
from wake_audio import read_wav
from wake_data import atomic_json
from wake_metrics import point_for_config
from phonetic import ALPHABET,keyword_probability,posterior

TEACHER='jonatasgrosman/wav2vec2-large-xlsr-53-russian'
REVISION='ab9492f472b7de03e79117b4187ed7bf385af3e1'


def rows_for(source,split):
    return [json.loads(line) for line in (source/'dataset/manifest.jsonl').read_text(encoding='utf-8').splitlines()
            if json.loads(line)['split']==split]


def read_audio(source,row,cache):
    path=source/'dataset'/row['wav']
    if 'audio_index' not in row:return read_wav(path)
    if path not in cache:cache[path]=np.load(path,mmap_mode='r')
    return cache[path][row['audio_index']].astype(np.float32)/32768


def load_teacher(directory):
    from transformers import Wav2Vec2ForCTC
    model=Wav2Vec2ForCTC.from_pretrained(str(directory),local_files_only=True,torch_dtype=torch.float16,
                                       attn_implementation='sdpa').cuda().eval()
    vocab=json.loads((directory/'vocab.json').read_text(encoding='utf-8'))
    mapping=np.full(model.config.vocab_size,5,np.int64)
    for character,index in vocab.items():
        mapping[index]={'б':1,'о':2,'т':3,'|':4}.get(character.casefold(),5)
    mapping[model.config.pad_token_id]=0
    return model,[torch.tensor(np.flatnonzero(mapping==i),device='cuda') for i in range(6)]


@torch.inference_mode()
def infer(model,groups,audio):
    value=torch.from_numpy(np.stack(audio)).cuda()
    value=(value-value.mean(-1,keepdim=True))/torch.sqrt(value.var(-1,keepdim=True,unbiased=False)+1e-7)
    probability=model(value.half()).logits.float().softmax(-1)
    # Filler represents the strongest competing phone, not the sum of dozens
    # of unrelated letters: that sum would turn uncertain silence into speech.
    reduced=torch.stack([probability.index_select(-1,indices).amax(-1) for indices in groups],-1)
    reduced=reduced/reduced.sum(-1,keepdim=True)
    return reduced.cpu().numpy()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='bot-v2')
    parser.add_argument('--run',default='bot-phonetic-v1')
    parser.add_argument('--splits',nargs='+',default=['validation','train'],choices=['train','validation','calibration','test'])
    parser.add_argument('--batch',type=int,default=32)
    args=parser.parse_args()
    source=ROOT/'word_training/runs'/args.source;work=ROOT/'word_training/runs'/args.run
    if source.parent.resolve()!=(ROOT/'word_training/runs').resolve() or work.parent.resolve()!=source.parent.resolve():
        raise ValueError('Use local run names')
    work.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256((source/'dataset/manifest.jsonl').read_bytes()).hexdigest()
    provenance=dict(teacher=TEACHER,revision=REVISION,source_run=args.source,manifest_sha256=digest,alphabet=ALPHABET,
                    competing_phone_pool='max')
    record=work/'teacher_provenance.json'
    if record.exists() and json.loads(record.read_text(encoding='utf-8'))!=provenance:raise ValueError('Different teacher/dataset already cached')
    atomic_json(record,provenance)
    torch.set_num_threads(4)
    model,groups=load_teacher(ROOT/'.runtime/wake-phonetic-teacher')
    config=json.loads((source/'run_config.json').read_text(encoding='utf-8'))
    start=time.monotonic()
    audio_cache={}
    for split in args.splits:
        rows=rows_for(source,split)
        array_file=work/f'teacher-{split}.npy';status_file=work/f'teacher-{split}-status.json'
        completed=0
        if status_file.exists():
            status=json.loads(status_file.read_text(encoding='utf-8'))
            if status['manifest_sha256']!=digest:raise ValueError('Cached split belongs to another dataset')
            completed=status['completed']
        array=np.lib.format.open_memmap(array_file,mode='r+') if array_file.exists() else None
        for begin in range(completed,len(rows),args.batch):
            batch=rows[begin:begin+args.batch]
            result=infer(model,groups,[read_audio(source,row,audio_cache) for row in batch])
            if array is None:array=np.lib.format.open_memmap(array_file,mode='w+',dtype=np.float16,shape=(len(rows),*result.shape[1:]))
            if result.shape[1:]!=array.shape[1:] or not np.isfinite(result).all():raise ValueError('Invalid teacher output')
            array[begin:begin+len(batch)]=result
            array.flush()
            atomic_json(status_file,dict(completed=begin+len(batch),total=len(rows),manifest_sha256=digest))
            if begin//args.batch%20==0 or begin+len(batch)==len(rows):
                print(f'[teacher] {split}: {begin+len(batch)}/{len(rows)}, {time.monotonic()-start:.0f}s',flush=True)
        if split=='validation':
            candidates=[]
            for temperature in [.5,1.]:
                scores=keyword_probability(posterior(np.log(np.maximum(np.asarray(array,dtype=np.float32),1e-9)),temperature))
                point=point_for_config(dict(labels=[r['label'] for r in rows],scores=scores.tolist()),rows,config)
                candidates.append(dict(temperature=temperature,point=point))
                print(f'[teacher validation] temperature={temperature}, recall={point["macro_recall"]:.4f}, worst={point["worst_source_recall"]:.4f}',flush=True)
            atomic_json(work/'teacher_validation.json',dict(candidates=candidates,scope='Validation only; acoustic teacher, not exported student'))
        del array
    print('[teacher] complete',flush=True)


if __name__=='__main__':main()
