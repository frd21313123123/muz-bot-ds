"""Distil Russian acoustic alignment into a compact whole-word BOT detector."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.runtime/wake-validation'),str(ROOT/'scripts/wakeword'),str(Path(__file__).parent)]
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from wake_data import atomic_json
from wake_train import BalancedSampler,confusion
from wake_metrics import point_for_config,selection_score,calibrate
from teacher_phonetic import rows_for
from phonetic import targets,posterior,keyword_probability,ALPHABET,forced_alignment
from phonetic_model import make_student,augment_features


class AcousticData(Dataset):
    def __init__(self,source,work,split,teacher=False):
        self.rows=rows_for(source,split)
        file=work/f'mel-{split}.npy'
        if not file.exists():
            temporary=file.with_suffix('.tmp.npy')
            array=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.float16,shape=(len(self.rows),1,80,200))
            packed={}
            for index,row in enumerate(self.rows):
                path=source/'dataset'/row['feature']
                if 'feature_index' in row:
                    if path not in packed:packed[path]=np.load(path,mmap_mode='r')
                    array[index]=packed[path][row['feature_index']]
                else:array[index]=np.load(path)
                if (index+1)%5000==0:print(f'[features] {split}: {index+1}/{len(self.rows)}',flush=True)
            array.flush();del array;temporary.replace(file)
        self.features=np.load(file,mmap_mode='r')
        if len(self.features)!=len(self.rows):raise ValueError('Feature cache has wrong row count')
        self.teacher=np.load(work/f'teacher-{split}.npy',mmap_mode='r') if teacher else None
        if teacher:
            status=json.loads((work/f'teacher-{split}-status.json').read_text(encoding='utf-8'))
            if status['completed']!=len(self.rows):raise ValueError('Complete teacher extraction before training')
        self.annotated=[row['language']=='ru' or 'phonetic_text' in row for row in self.rows]
        self.text_targets=[targets(row.get('phonetic_text',row['text'])) if annotated else [] for row,annotated in zip(self.rows,self.annotated,strict=True)]
        self.alignment=None
        if teacher:
            file=work/f'alignment-{split}.npy'
            if not file.exists():
                aligned=np.zeros((len(self.rows),99),np.int8)
                for index,row in enumerate(self.rows):
                    if self.annotated[index]:aligned[index]=forced_alignment(self.teacher[index],self.text_targets[index])
                    if (index+1)%5000==0:print(f'[alignment] {index+1}/{len(self.rows)}',flush=True)
                temporary=file.with_suffix('.tmp.npy');np.save(temporary,aligned);temporary.replace(file)
            self.alignment=np.load(file)
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        feature=torch.from_numpy(self.features[index].astype(np.float32))
        if self.teacher is None:return feature,index
        row=self.rows[index]
        teacher=self.teacher[index].astype(np.float32)
        supervised=self.annotated[index] or row['language']=='none'
        if supervised:teacher=.1*teacher+.9*np.eye(6,dtype=np.float32)[self.alignment[index]]
        return feature,torch.from_numpy(teacher),torch.tensor(self.text_targets[index],dtype=torch.long),float(supervised)


def collate(batch):
    feature,teacher,text,mask=zip(*batch)
    return torch.stack(feature),torch.stack(teacher),torch.cat(text),torch.tensor([len(t) for t in text]),torch.tensor(mask)


@torch.inference_mode()
def infer(model,dataset,batch=128):
    model.eval();values=[]
    for feature,_ in DataLoader(dataset,batch_size=batch,num_workers=0):
        with torch.autocast('cuda',dtype=torch.float16):logits=model(feature.cuda())
        values.append(logits.float().cpu().numpy())
    return np.concatenate(values)


def choose(logits,rows,config):
    candidates=[]
    for temperature in [.15,.35,.7,1.]:
        scores=keyword_probability(posterior(logits,temperature))
        point=point_for_config(dict(labels=[r['label'] for r in rows],scores=scores.tolist()),rows,config)
        candidates.append(dict(temperature=temperature,point=point,score=selection_score(point,0,config)))
    return max(candidates,key=lambda item:item['score']),candidates


def save_torch(path,value):
    temporary=path.with_suffix('.tmp');torch.save(value,temporary);temporary.replace(path)


def train(config,source,work,resume):
    training=AcousticData(source,work,'train',teacher=True);validation=AcousticData(source,work,'validation')
    model=make_student(config,source).cuda();ema=copy.deepcopy(model).eval().requires_grad_(False)
    sampler=BalancedSampler([dict(row,language=row.get('sampling_language',row['language'])) for row in training.rows],config['seed'])
    loader=DataLoader(training,batch_size=config['batch_per_gpu'],sampler=sampler,collate_fn=collate,num_workers=0,pin_memory=True)
    encoder=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('encoder.')]
    head=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('encoder.')]
    if config.get('backbone')=='tiny':
        optimizer=torch.optim.AdamW([dict(params=head),dict(params=encoder,lr=config['encoder_learning_rate'])],lr=config['learning_rate'],weight_decay=.001)
    else:optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=.001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['epochs'])
    scaler=torch.amp.GradScaler('cuda');ctc=nn.CTCLoss(blank=0,reduction='none',zero_infinity=True)
    checkpoint=work/'latest.pt';best_file=work/'best.pt'
    start_epoch=0;steps=0;stale=0;best_score=-float('inf');history=[]
    if resume and checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['config']!=config:raise ValueError('Resume requires the same recipe')
        model.load_state_dict(saved['model']);ema.load_state_dict(saved['ema']);optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']);scaler.load_state_dict(saved['scaler'])
        start_epoch=saved['epoch'];steps=saved['steps'];stale=saved['stale'];best_score=saved['best_score'];history=saved['history']
        torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
        sampler.weights=saved['sampler_weights']
    print(f'[student] RTX={torch.cuda.get_device_name(0)}, parameters={sum(p.numel() for p in model.parameters())}, windows={len(training)}',flush=True)
    for epoch in range(start_epoch,config['epochs']):
        tick=time.monotonic();sampler.set_epoch(epoch);model.train();loss_sum=0.;count=0
        for feature,teacher,text,length,mask in loader:
            feature=augment_features(feature.cuda());teacher=teacher.cuda();mask=mask.cuda()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):logits=model(feature)
            log_probability=logits.float().log_softmax(-1)
            teacher=teacher/teacher.sum(-1,keepdim=True).clamp_min(1e-7)
            weights=1+4*(1-teacher[:,:,0])
            kd=(-(teacher*log_probability).sum(-1)*weights).sum()/weights.sum()
            sequence=ctc(log_probability.transpose(0,1),text.cuda(),torch.full((len(feature),),99,dtype=torch.long),length)
            sequence=sequence/length.cuda().clamp_min(5)
            supervised=(sequence*mask).sum()/mask.sum().clamp_min(1)
            loss=config.get('alignment_weight',1.)*kd+config.get('ctc_weight',.3)*supervised
            if not torch.isfinite(loss):raise ValueError('Non-finite student training loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),3.)
            scaler.step(optimizer);scaler.update();steps+=1
            decay=min(.999,(steps+1)/(steps+10))
            with torch.no_grad():
                for averaged,current in zip(ema.parameters(),model.parameters(),strict=True):averaged.lerp_(current,1-decay)
            loss_sum+=float(loss.detach())*len(feature);count+=len(feature)
        scheduler.step()
        trials=[]
        for kind,network in [('raw',model),('ema',ema)]:
            chosen,candidates=choose(infer(network,validation),validation.rows,config)
            trials.append(dict(weights=kind,**chosen))
        chosen=max(trials,key=lambda item:item['score'])
        improved=chosen['score']>best_score
        if improved:
            best_score=chosen['score'];stale=0
            network=ema if chosen['weights']=='ema' else model
            save_torch(best_file,dict(model={k:v.detach().cpu() for k,v in network.state_dict().items()},epoch=epoch+1,selection=chosen,config=config))
        else:stale+=1
        row=dict(epoch=epoch+1,loss=loss_sum/count,validation=trials,selected=chosen,seconds=time.monotonic()-tick)
        history.append(row);atomic_json(work/'training_history.json',history)
        print(f'[epoch {epoch+1}] loss={row["loss"]:.4f}, recall={chosen["point"]["macro_recall"]:.3f}, worst={chosen["point"]["worst_source_recall"]:.3f}, {chosen["weights"]}, T={chosen["temperature"]}, {row["seconds"]:.1f}s',flush=True)
        save_torch(checkpoint,dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),epoch=epoch+1,steps=steps,stale=stale,best_score=best_score,history=history,config=config,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),sampler_weights=sampler.weights))
        if epoch+1>=config['min_epochs'] and stale>=config['patience']:break
    atomic_json(work/'training_run.json',dict(device='cuda:0',gpu=torch.cuda.get_device_name(0),amp=True,epochs_completed=len(history),parameters=sum(p.numel() for p in model.parameters()),manifest_sha256=config['manifest_sha256']))


def export(config,source,work):
    import onnx
    import onnxruntime as ort
    output=work.parent/(work.name+'-output');output.mkdir(exist_ok=True)
    runtime=work/'model';runtime.mkdir(exist_ok=True)
    best=torch.load(work/'best.pt',map_location='cpu',weights_only=False)
    model=make_student(config,source).eval();model.load_state_dict(best['model'])
    probe=np.load(source/'dataset'/rows_for(source,'validation')[0]['feature']).astype(np.float32)[None]
    file=runtime/'wake_model.onnx'
    torch.onnx.export(model,torch.from_numpy(probe),str(file),input_names=['log_mel'],output_names=['logits'],dynamic_axes={'log_mel':{0:'batch'},'logits':{0:'batch'}},opset_version=17,dynamo=False)
    onnx.checker.check_model(str(file))
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(file),options,providers=['CPUExecutionProvider'])
    with torch.inference_mode():expected=model(torch.from_numpy(probe)).numpy()
    onnx_logits=session.run(None,{'log_mel':probe})[0]
    np.testing.assert_allclose(onnx_logits,expected,atol=2e-5,rtol=1e-4)
    for name in ['wake_audio.py']:shutil.copy2(ROOT/'scripts/wakeword'/name,runtime/name)
    for name in ['phonetic.py','phonetic_runtime.py','detect.py']:shutil.copy2(ROOT/'word_training'/name,runtime/name)
    for name in ['whisper_mel_filters.npy','provenance.json']:shutil.copy2(source/'whisper-encoder'/name,runtime/name)
    temperature=best['selection']['temperature']
    from phonetic_onnx import add_keyword_output
    logits_file=work/'student_logits.onnx';shutil.copy2(file,logits_file)
    add_keyword_output(logits_file,file,temperature)
    session=ort.InferenceSession(str(file),options,providers=['CPUExecutionProvider'])
    np.testing.assert_allclose(session.run(None,{'log_mel':probe})[0],keyword_probability(posterior(onnx_logits,temperature)),atol=1e-10,rtol=1e-8)
    # Keep the familiar binary ONNX contract: log_mel input -> probability output.
    frontend_file=runtime/'wake_audio.py'
    frontend_file.write_text(frontend_file.read_text(encoding='utf-8').replace("if config.get('architecture') == 'whisper_encoder_v4':","if config.get('architecture') in ['whisper_encoder_v4', 'phonetic_ctc_v1']:"),encoding='utf-8')
    def predict(split):
        dataset=AcousticData(source,work,split);values=[]
        for feature,_ in DataLoader(dataset,batch_size=128):values.append(session.run(['probability'],{'log_mel':feature.numpy()})[0])
        scores=np.concatenate(values)
        return dataset.rows,scores
    validation_rows,validation_scores=predict('validation')
    validation=point_for_config(dict(labels=[r['label'] for r in validation_rows],scores=validation_scores.tolist()),validation_rows,config)
    cal_rows,cal_scores=predict('calibration')
    threshold,calibration=calibrate([r['label'] for r in cal_rows],cal_scores,config['target_fpr'],[r['language'] for r in cal_rows])
    test_rows,test_scores=predict('test');labels=[r['label'] for r in test_rows]
    test=confusion(labels,test_scores,threshold);groups={}
    for field in ['kind','speaker','language']:
        groups[field]={}
        for value in sorted({r[field] for r in test_rows}):
            indices=[i for i,r in enumerate(test_rows) if r[field]==value]
            groups[field][value]=confusion([labels[i] for i in indices],test_scores[indices],threshold)
    metadata=dict(architecture='phonetic_ctc_v1',task='whole_word_anywhere',wake_word_ru='бот',wake_word_en=None,threshold=threshold,temperature=temperature,alphabet=ALPHABET,sample_rate=16000,window_samples=32000,standalone_only=False,quality_validated_on_real_users=False,model_sha256=hashlib.sha256(file.read_bytes()).hexdigest())
    atomic_json(runtime/'wake_config.json',metadata)
    report=dict(validation=validation,calibration=calibration,test=test,test_groups=groups,threshold=threshold,temperature=temperature,best_epoch=best['epoch'],model_bytes=file.stat().st_size,real_user_validation=False,decision_data='Validation selects weights and temperature; calibration fixes threshold; test reports only',quality_gate=dict(passed=test['recall']>=.9 and test['false_positive_rate_per_window']<=config['target_fpr'],target_recall=.9,target_fpr=config['target_fpr']))
    atomic_json(output/'training_report.json',report)
    for split,rows,scores in [('calibration',cal_rows,cal_scores),('test',test_rows,test_scores)]:
        atomic_json(output/f'{split}_predictions.json',[dict(id=r['id'],text=r['text'],label=r['label'],kind=r['kind'],speaker=r['speaker'],probability=float(score)) for r,score in zip(rows,scores,strict=True)])
    (runtime/'README.txt').write_text('BOT whole-word wake detector. ONNX log_mel input -> probability output. No transcript or command output.\nInstall numpy onnxruntime. Run detect.py MODEL_FOLDER recording.wav (mono PCM16, 16 kHz).\nTeacher and datasets are unnecessary for inference. See accompanying training_report.json for measured quality.\n',encoding='utf-8')
    files=['wake_model.onnx','wake_config.json','wake_audio.py','whisper_mel_filters.npy','provenance.json','phonetic.py','phonetic_runtime.py','detect.py','README.txt']
    with zipfile.ZipFile(output/'wake-model.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in files:archive.write(runtime/name,name)
    with zipfile.ZipFile(output/'wake-model.zip') as archive:assert archive.testzip() is None and set(archive.namelist())==set(files)
    for name in ['teacher_provenance.json','training_history.json','training_run.json','run_config.json']:shutil.copy2(work/name,output/name)
    atomic_json(output/'output_summary.json',dict(archive=str(output/'wake-model.zip'),model_bytes=file.stat().st_size,test=test,dataset_exported=False,teacher_exported=False))
    print(json.dumps(dict(model_bytes=file.stat().st_size,test=test,quality_gate=report['quality_gate']),ensure_ascii=False,indent=2),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',default='bot-phonetic-v2');parser.add_argument('--source',default='bot-v2');parser.add_argument('--resume',action='store_true');parser.add_argument('--export-only',action='store_true')
    parser.add_argument('--backbone',choices=['cnn','tiny'],default='cnn');parser.add_argument('--encoder-train-layers',type=int,default=1)
    args=parser.parse_args();source=ROOT/'word_training/runs'/args.source;work=ROOT/'word_training/runs'/args.run
    if source.parent.resolve()!=(ROOT/'word_training/runs').resolve() or work.parent.resolve()!=source.parent.resolve():raise ValueError('Use local run names')
    work.mkdir(exist_ok=True);base=json.loads((source/'run_config.json').read_text(encoding='utf-8'))
    config=dict(seed=base['seed'],learning_rate=.0003,batch_per_gpu=128,epochs=60,min_epochs=20,patience=12,target_fpr=.005,target_recall=.9,source_groups=True,source_group_fpr_constraint=False,manifest_sha256=hashlib.sha256((source/'dataset/manifest.jsonl').read_bytes()).hexdigest(),architecture='phonetic_ctc_v1',distillation='Russian teacher forced alignment of annotated text, 90% aligned labels + 10% teacher soft states, plus CTC',feature_warp=[.9,1.1])
    if args.backbone=='tiny':config.update(backbone='tiny',encoder_train_layers=args.encoder_train_layers,encoder_learning_rate=3e-6,batch_per_gpu=64,alignment_weight=.3,ctc_weight=1.)
    previous=work/'run_config.json'
    if previous.exists() and json.loads(previous.read_text(encoding='utf-8'))!=config:raise ValueError('Different recipe already exists')
    provenance=json.loads((work/'teacher_provenance.json').read_text(encoding='utf-8'))
    if provenance['manifest_sha256']!=config['manifest_sha256'] or provenance.get('competing_phone_pool')!='max':raise ValueError('Teacher cache is incompatible')
    if (work/'best.pt').exists() and not(args.resume or args.export_only):raise ValueError('Use --resume for an existing run')
    atomic_json(previous,config);torch.set_num_threads(4);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
    def stage(value,**extra):atomic_json(work/'status.json',dict(stage=value,updated_at=time.time(),**extra));print(f'[{value}]',flush=True)
    try:
        if not args.export_only:stage('training');train(config,source,work,args.resume)
        stage('exporting');report=export(config,source,work);stage('complete',test=report['test'],model_bytes=report['model_bytes'])
    except BaseException as error:stage('failed',error=str(error));raise


if __name__=='__main__':main()
