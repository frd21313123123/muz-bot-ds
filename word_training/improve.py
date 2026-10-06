"""Continue binary word training on additional voices; preserve all held-out windows."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from train import ROOT,cache_files
sys.path[:0]=[str(ROOT/'.runtime/wake-validation'),str(ROOT/'scripts/wakeword')]
import numpy as np
import torch
from wake_data import atomic_json
import wake_train


class PackedFeatureDataset(wake_train.FeatureDataset):
    def __init__(self,root,split,preload=False):
        self.root=Path(root)
        self.rows=[json.loads(line) for line in (self.root/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
        self.rows=[row for row in self.rows if row['split']==split]
        if not self.rows:raise ValueError('Empty split')
        self.packed={}
        for row in self.rows:
            if 'feature_index' in row:
                path=self.root/row['feature']
                if path not in self.packed:self.packed[path]=np.load(path,mmap_mode='r')
        self.cached=np.stack([self.read(row) for row in self.rows]) if preload else None
    def read(self,row):
        path=self.root/row['feature']
        return self.packed[path][row['feature_index']] if 'feature_index' in row else np.load(path)
    def __getitem__(self,index):
        row=self.rows[index]
        feature=(self.cached[index] if self.cached is not None else self.read(row)).astype(np.float32)
        return torch.from_numpy(feature),torch.tensor(row['label'],dtype=torch.float32)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='bot-data-v3');parser.add_argument('--run',default='bot-binary-v3')
    parser.add_argument('--initial',default='bot-v2/checkpoints/member-0/best.pt')
    parser.add_argument('--teacher',help='Local run containing train-only teacher-targets.npz')
    parser.add_argument('--temporal',action='store_true',help='Add residual local context before word pooling')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--export-only',action='store_true')
    args=parser.parse_args();runs=ROOT/'word_training/runs';source=runs/args.source;work=runs/args.run
    if source.parent.resolve()!=runs.resolve() or work.parent.resolve()!=runs.resolve():raise ValueError('Use local run names')
    initial=(runs/args.initial).resolve()
    if runs.resolve() not in initial.parents:raise ValueError('Initial checkpoint must be inside local runs')
    work.mkdir(exist_ok=True)
    (work/'dataset').mkdir(exist_ok=True)
    rows=[json.loads(line) for line in (source/'dataset/manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    for row in rows:
        for key in ['feature','wav']:row[key]=str((source/'dataset'/row[key]).resolve())
    manifest=work/'dataset/manifest.jsonl'
    manifest.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
    config=json.loads((source/'run_config.json').read_text(encoding='utf-8'))
    config.update(work_dir=str(work),output_dir=str(runs/(args.run+'-output')),dataset_dir=str(work/'dataset'),
                  encoder_dir=str(source/'whisper-encoder'),ensemble_members=1,epochs=45,min_epochs=15,patience=10,
                  learning_rate=.0003,encoder_train_layers=4,encoder_learning_rate=3e-6,
                  member_overrides={'0':dict(distill_weight=0,encoder_train_layers=4,encoder_learning_rate=3e-6)},
                  initial_checkpoint=str(initial),initial_sha256=hashlib.sha256(initial.read_bytes()).hexdigest(),
                  manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),resume_checkpoint=None,feature_warp=[.9,1.1])
    if args.temporal:config.update(temporal_head='residual_v1')
    if args.teacher:
        teacher=runs/args.teacher
        if teacher.parent.resolve()!=runs.resolve():raise ValueError('Use local teacher name')
        file=teacher/'teacher-targets.npz'
        source_digest=hashlib.sha256((source/'dataset/manifest.jsonl').read_bytes()).hexdigest()
        import wake_distill
        class TrainTeacherDataset(torch.utils.data.Dataset):
            def __init__(self,dataset,targets_file):
                self.dataset,self.rows,self.root=dataset,dataset.rows,dataset.root
                if any(r['split']!='train' for r in self.rows):raise ValueError('Distillation is train-only')
                with np.load(targets_file,allow_pickle=False) as saved:
                    if str(saved['source_manifest_sha256'].item())!=source_digest:raise ValueError('Wrong teacher source manifest')
                    if saved['ids'].tolist()!=[r['id'] for r in self.rows]:raise ValueError('Wrong teacher row order')
                    self.logits=saved['logits'].copy()
                if self.logits.shape!=(len(self.rows),) or not np.isfinite(self.logits).all():raise ValueError('Invalid teacher logits')
            def __len__(self):return len(self.rows)
            def __getitem__(self,index):
                feature,label=self.dataset[index]
                return feature,label,torch.tensor(self.logits[index],dtype=torch.float32)
        wake_distill.DistillationDataset=TrainTeacherDataset
        config.update(teacher_targets=str(file),teacher_targets_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),distill_weight=.4,
                      distill_temperature=2.,member_overrides={'0':dict(distill_weight=.4,encoder_train_layers=4,encoder_learning_rate=3e-6)})
    recipe=work/'run_config.json'
    if recipe.exists() and json.loads(recipe.read_text(encoding='utf-8'))!=config:raise ValueError('Existing run has another recipe')
    if (work/'checkpoints/best.pt').exists() and not(args.resume or args.export_only):raise ValueError('Use --resume')
    atomic_json(recipe,config)
    cache_files(source/'whisper-encoder',work/'whisper-encoder')
    for name in ['dataset_summary.json','dataset_recipe.json','tts_provenance.json','environment.json']:
        if (source/name).exists():shutil.copy2(source/name,work/name)
    wake_train.FeatureDataset=PackedFeatureDataset
    original_sampler=wake_train.BalancedSampler
    class ProxyBalancedSampler(original_sampler):
        def __init__(self,rows,*args,**kwargs):
            canonical=[dict(row,language=row.get('sampling_language',row['language'])) for row in rows]
            super().__init__(canonical,*args,**kwargs)
    wake_train.BalancedSampler=ProxyBalancedSampler
    from phonetic_model import augment_features
    original_augmentation=wake_train.spec_augment
    wake_train.spec_augment=lambda value:original_augmentation(augment_features(value))
    original=wake_train.make_model
    def initialized(configuration):
        if configuration.get('temporal_head'):
            from temporal_model import TemporalTinyWakeModel
            model=TemporalTinyWakeModel(configuration)
        else:model=original(configuration)
        missing,unexpected=model.load_state_dict(torch.load(initial,map_location='cpu',weights_only=False)['model'],strict=False)
        if unexpected or any(not key.startswith('temporal.') for key in missing):raise ValueError('Incompatible initial model')
        return model
    wake_train.make_model=initialized
    import wake_model
    wake_model.make_model=initialized
    if args.resume:config['resume_checkpoint']=str(work/'checkpoints/latest.pt')
    def stage(value,**extra):
        atomic_json(work/'status.json',dict(stage=value,updated_at=time.time(),**extra));print(f'[{value}]',flush=True)
    try:
        if not args.export_only:
            stage('training')
            with torch.inference_mode(False),torch.enable_grad():wake_train.train(config)
        stage('exporting')
        from wake_package_v5 import export_student,package,quality_gate
        export_student(config);reports,selection=package(config)
        report=reports[selection['selected']]
        report['quality_gate']=quality_gate(report['test'],{'ru':report['test_groups']['language']['ru']},.9,.005)
        output=Path(config['output_dir']);atomic_json(output/'training_report.json',report)
        for name in ['training_history.json','training_run.json','run_config.json']:
            shutil.copy2(work/name,output/name)
        summary=dict(test=report['test'],quality_gate=report['quality_gate'],model_bytes=report['model_bytes'],
                     archive=str(output/'wake-model.zip'),word='бот',task='whole_word_anywhere',dataset_exported=False,
                     real_user_validation=False,format=selection['selected'])
        atomic_json(output/'output_summary.json',summary);stage('complete',result=summary)
        print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    except BaseException as error:stage('failed',error=str(error));raise


if __name__=='__main__':main()
