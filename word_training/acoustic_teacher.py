"""Learn only BOT presence on frozen multilingual Russian acoustic features."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.runtime/wake-validation'),str(ROOT/'scripts/wakeword'),str(Path(__file__).parent)]
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from teacher_phonetic import rows_for,read_audio,TEACHER,REVISION
from wake_data import atomic_json
from wake_train import BalancedSampler
from wake_metrics import point_for_config,selection_score


class AcousticHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm=nn.LayerNorm(1024)
        self.project=nn.Sequential(nn.Conv1d(1024,128,1),nn.SiLU(),nn.Dropout(.15))
        self.temporal=nn.ModuleList([nn.Sequential(nn.Conv1d(128,128,3,padding=d,dilation=d,groups=128),
                nn.Conv1d(128,128,1),nn.GroupNorm(8,128),nn.SiLU(),nn.Dropout(.15)) for d in [1,2,4]])
        self.attention=nn.Conv1d(128,1,1)
        self.head=nn.Sequential(nn.Dropout(.2),nn.Linear(256,64),nn.SiLU(),nn.Linear(64,1))
    def forward(self,value):
        value=self.project(self.norm(value).transpose(1,2))
        for block in self.temporal:value=value+block(value)
        weights=self.attention(value).softmax(-1)
        return self.head(torch.cat([(value*weights).sum(-1),value.amax(-1)],1)).squeeze(-1)


@torch.inference_mode()
def extract(source,work,batch):
    from transformers import Wav2Vec2Model
    # Load only acoustics; tokenizer, language vocabulary and text head are absent.
    model=Wav2Vec2Model.from_pretrained(str(ROOT/'.runtime/wake-phonetic-teacher'),local_files_only=True,
             dtype=torch.float16,attn_implementation='sdpa').cuda().eval()
    manifest=hashlib.sha256((source/'dataset/manifest.jsonl').read_bytes()).hexdigest();cache={}
    for split in ['validation','train']:
        rows=rows_for(source,split);file=work/(split+'-features.npy');status_file=work/(split+'-features-status.json')
        completed=0
        if status_file.exists():
            state=json.loads(status_file.read_text(encoding='utf-8'))
            if state['manifest_sha256']!=manifest:raise ValueError('Wrong acoustic cache')
            completed=state['completed']
        array=np.lib.format.open_memmap(file,mode='r+') if file.exists() else np.lib.format.open_memmap(file,mode='w+',dtype=np.float16,shape=(len(rows),33,1024))
        for begin in range(completed,len(rows),batch):
            audio=np.stack([read_audio(source,row,cache) for row in rows[begin:begin+batch]])
            value=torch.from_numpy(audio).cuda()
            value=(value-value.mean(-1,keepdim=True))/torch.sqrt(value.var(-1,keepdim=True,unbiased=False)+1e-7)
            hidden=model(value.half()).last_hidden_state
            hidden=nn.functional.avg_pool1d(hidden.transpose(1,2),3,3).transpose(1,2).float().cpu().numpy()
            if hidden.shape[1:]!=(33,1024) or not np.isfinite(hidden).all():raise ValueError('Invalid acoustic features')
            array[begin:begin+len(hidden)]=hidden;array.flush()
            atomic_json(status_file,dict(completed=begin+len(hidden),total=len(rows),manifest_sha256=manifest))
            if begin%640==0 or begin+len(hidden)==len(rows):print(f'[acoustics] {split}: {begin+len(hidden)}/{len(rows)}',flush=True)
        del array
    atomic_json(work/'teacher_provenance.json',dict(repository=TEACHER,revision=REVISION,manifest_sha256=manifest,
        task='Frozen speech acoustics plus supervised binary whole-word head; no text output',pooling='99 frames -> 33 mean frames'))


class AcousticDataset(Dataset):
    def __init__(self,source,work,split):
        self.rows=rows_for(source,split);self.features=np.load(work/(split+'-features.npy'),mmap_mode='r')
        state=json.loads((work/(split+'-features-status.json')).read_text(encoding='utf-8'))
        if state['completed']!=len(self.rows):raise ValueError('Incomplete acoustic features')
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):return torch.from_numpy(self.features[index].astype(np.float32)),torch.tensor(self.rows[index]['label'],dtype=torch.float32)


@torch.inference_mode()
def predict(model,dataset):
    model.eval();values=[]
    for feature,_ in DataLoader(dataset,batch_size=256):
        with torch.autocast('cuda',dtype=torch.float16):output=model(feature.cuda())
        values.extend(output.float().cpu().tolist())
    return np.asarray(values,np.float32)


def train(source,work):
    config=dict(seed=202610062,epochs=70,min_epochs=15,patience=10,target_fpr=.005,source_groups=True,source_group_fpr_constraint=False,
                manifest_sha256=hashlib.sha256((source/'dataset/manifest.jsonl').read_bytes()).hexdigest(),learning_rate=.0007,batch_per_gpu=256)
    atomic_json(work/'run_config.json',config)
    training=AcousticDataset(source,work,'train');validation=AcousticDataset(source,work,'validation')
    model=AcousticHead().cuda();ema=copy.deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=.001)
    scaler=torch.amp.GradScaler('cuda');sampler=BalancedSampler([dict(row,language=row.get('sampling_language',row['language'])) for row in training.rows],config['seed'])
    loader=DataLoader(training,batch_size=256,sampler=sampler,num_workers=0,pin_memory=True)
    history=[];best_score=-np.inf;stale=0;steps=0
    for epoch in range(config['epochs']):
        tick=time.monotonic();model.train();sampler.set_epoch(epoch);loss_sum=0.
        for feature,label in loader:
            optimizer.zero_grad(set_to_none=True);feature=feature.cuda();label=label.cuda()
            feature=feature+torch.randn_like(feature)*.005
            with torch.autocast('cuda',dtype=torch.float16):loss=nn.functional.binary_cross_entropy_with_logits(model(feature),label)
            if not torch.isfinite(loss):raise ValueError('Invalid acoustic head loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),5.)
            scaler.step(optimizer);scaler.update();steps+=1
            decay=min(.995,(1+steps)/(10+steps))
            with torch.no_grad():
                for averaged,current in zip(ema.parameters(),model.parameters(),strict=True):averaged.lerp_(current,1-decay)
            loss_sum+=float(loss.detach())*len(feature)
        trials=[]
        for name,network in [('raw',model),('ema',ema)]:
            logits=predict(network,validation);scores=1/(1+np.exp(-logits.astype(np.float64)))
            point=point_for_config(dict(labels=[r['label'] for r in validation.rows],scores=scores),validation.rows,config)
            trials.append(dict(weights=name,point=point,score=selection_score(point,0,config)))
        chosen=max(trials,key=lambda item:item['score']);improved=chosen['score']>best_score
        if improved:
            best_score=chosen['score'];stale=0;network=ema if chosen['weights']=='ema' else model
            temporary=work/'best.tmp';torch.save(dict(model={k:v.cpu() for k,v in network.state_dict().items()},epoch=epoch+1,selection=chosen,config=config),temporary);temporary.replace(work/'best.pt')
        else:stale+=1
        history.append(dict(epoch=epoch+1,loss=loss_sum/len(training),validation=chosen,seconds=time.monotonic()-tick))
        atomic_json(work/'training_history.json',history)
        print(f'[acoustic head {epoch+1}] recall={chosen["point"]["macro_recall"]:.3f}, worst={chosen["point"]["worst_source_recall"]:.3f}, {time.monotonic()-tick:.1f}s',flush=True)
        if epoch+1>=config['min_epochs'] and stale>=config['patience']:break
    best=torch.load(work/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(best['model'])
    values=predict(model,training)
    np.savez_compressed(work/'teacher-targets.npz',ids=np.array([r['id'] for r in training.rows]),logits=values,
                source_manifest_sha256=np.array(config['manifest_sha256']),teacher_checkpoint_sha256=np.array(hashlib.sha256((work/'best.pt').read_bytes()).hexdigest()))
    atomic_json(work/'training_run.json',dict(device='cuda:0',gpu=torch.cuda.get_device_name(0),head_parameters=sum(p.numel() for p in model.parameters()),
                frozen_acoustic_encoder=True,epochs_completed=len(history),best_validation=best['selection'],test_used=False))
    print(json.dumps(best['selection'],indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',default='bot-data-v3');parser.add_argument('--run',default='bot-large-acoustic-v3')
    parser.add_argument('--batch',type=int,default=8);parser.add_argument('--train-only',action='store_true');args=parser.parse_args()
    runs=ROOT/'word_training/runs';source=runs/args.source;work=runs/args.run
    if source.parent.resolve()!=runs.resolve() or work.parent.resolve()!=runs.resolve():raise ValueError('Use local run names')
    work.mkdir(exist_ok=True);torch.set_num_threads(4);torch.manual_seed(202610062);torch.cuda.manual_seed_all(202610062)
    if not args.train_only:extract(source,work,args.batch)
    train(source,work)


if __name__=='__main__':main()
