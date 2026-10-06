"""Generate and train one Russian word detector on the local CUDA GPU."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.runtime/wake-validation'),str(ROOT/'scripts/wakeword'),str(Path(__file__).parent)]
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')


def cache_files(source,destination):
    source,destination=Path(source),Path(destination)
    for file in source.rglob('*'):
        if not file.is_file() or '.cache' in file.parts: continue
        target=destination/file.relative_to(source)
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists(): continue
        # Immutable weights may share disk blocks. Metadata always gets its own copy.
        if file.suffix in ['.pt','.onnx','.bin','.safetensors','.npy','.zip']:
            try: os.link(file,target); continue
            except OSError: pass
        shutil.copy2(file,target)


def configuration(work,word):
    return dict(work_dir=str(work),output_dir=str(work.parent/(work.name+'-output')),
        wake_word_ru=word,wake_word_en='bot',use_english=False,standalone_only=False,
        recipe_version='local-word-2',seed=202610062,tts_engines=['silero','piper','mms'],isolated_positive_fraction=.5,
        silero_ru_versions=['v5_5_ru','v4_ru'],gender_balanced_ru=True,
        piper_ru_splits={'train':['ru_RU-irina-medium'],'validation':['ru_RU-ruslan-medium'],
                         'calibration':['ru_RU-dmitri-medium'],'test':['ru_RU-denis-medium']},
        positive_per_voice=1200,negative_per_voice=1800,eval_positive_per_voice=250,eval_negative_per_voice=800,
        positive_tts_variants=12,negative_tts_variants=3,noise_windows=800,
        real_speech_source='mini',real_speech_windows=1600,real_wake_manifest=None,
        independent_pitch=True,architecture='whisper_encoder_v4',encoder_train_layers=4,
        encoder_learning_rate=3e-6,learning_rate=.0007,batch_per_gpu=64,
        epochs=70,min_epochs=20,patience=14,loader_workers=0,cpu_threads=4,
        mixed_precision=True,ema=True,balanced_sampling=True,preload_features=True,mining_interval=8,
        ensemble_members=1,member_overrides={'0':dict(distill_weight=0,encoder_train_layers=4,encoder_learning_rate=3e-6)},
        member_seed_stride=0,target_fpr=.005,target_recall=.90,source_groups=True,
        source_group_fpr_constraint=False,threshold_diagnostics=True,export_sample_metadata=True,
        export_calibration_predictions=True,quantization_recall_tolerance=.01,smoke=False,
        persist_latest=True,resume_checkpoint=None)


def same_recipe(saved,requested):
    return {k:v for k,v in saved.items() if k!='resume_checkpoint'} == {k:v for k,v in requested.items() if k!='resume_checkpoint'}


def run_training(config):
    import torch
    from wake_train import train
    # Packaged TTS may leave thread-local grad/inference mode disabled for training.
    with torch.inference_mode(False),torch.enable_grad():
        train(config)


def reuse_data_sources(config,run_name):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',run_name): raise ValueError('Seed run must be a local run name')
    source=ROOT/'word_training/runs'/run_name
    old=json.loads((source/'run_config.json').read_text(encoding='utf-8'))
    keys=['wake_word_ru','seed','standalone_only','use_english','tts_engines','silero_ru_versions','piper_ru_splits',
          'positive_tts_variants','negative_tts_variants','negative_per_voice','eval_negative_per_voice']
    if any(old[key]!=config[key] for key in keys): raise ValueError('Seed run uses incompatible source/negative labels')
    destination=Path(config['work_dir'])/'dataset'
    cache_files(source/'dataset/sources',destination/'sources')
    for folder in ['wav','features']:
        target=destination/folder; target.mkdir(parents=True,exist_ok=True)
        for file in (source/'dataset'/folder).glob('*-0-*'):
            if file.is_file() and not (target/file.name).exists(): shutil.copy2(file,target/file.name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--word',default='бот')
    parser.add_argument('--run',default='bot-v2')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--seed-run',help='Reuse immutable TTS sources and copied negative windows from a compatible local run')
    args=parser.parse_args()
    word=args.word.strip().casefold()
    if not re.fullmatch(r'\w+',word): raise ValueError('Specify one whole word')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.run): raise ValueError('Use an ASCII run name without paths')
    work=ROOT/'word_training/runs'/args.run
    work.mkdir(parents=True,exist_ok=True)
    config=configuration(work,word)
    from wake_data import atomic_json
    status=work/'status.json'
    if args.resume and status.exists():
        existing=json.loads(status.read_text(encoding='utf-8'))
        if existing.get('stage')=='complete':
            if existing.get('word')!=word: raise ValueError('Completed run belongs to another word')
            print(json.dumps(existing['result'],ensure_ascii=False,indent=2),flush=True)
            return
    started=time.time()
    def stage(name,**extra):
        atomic_json(status,dict(stage=name,word=word,started_at=started,updated_at=time.time(),**extra))
        print(f'[{name}] {word}',flush=True)
    try:
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU is required for this local training run')
        torch.set_num_threads(config['cpu_threads'])
        torch.cuda.set_device(0)
        stage('preparing',gpu=torch.cuda.get_device_name(0))
        previous=work/'run_config.json'
        if previous.exists():
            saved=json.loads(previous.read_text(encoding='utf-8'))
            if not same_recipe(saved,config):
                raise ValueError('This run already contains a different recipe; choose a new --run')
            if not args.resume: raise ValueError('Run already exists; use --resume or a new --run')
        atomic_json(previous,config)
        atomic_json(work/'environment.json',dict(python=sys.version,pytorch=torch.__version__,gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda))
        cache=ROOT/'.runtime/wakeword-v2-local'
        cache_files(cache/'tts/mms/mms-tts-rus',work/'tts/mms/mms-tts-rus')
        piper=work/'tts/piper'; piper.mkdir(parents=True,exist_ok=True)
        for file in (cache/'tts/piper').glob('ru_RU-*'):
            destination=piper/file.name
            if destination.exists(): continue
            if file.suffix=='.onnx':
                try: os.link(file,destination); continue
                except OSError: pass
            shutil.copy2(file,destination)
        for name in ['v5_5_ru.pt','v4_ru.pt']:
            destination=work/'tts'/name; destination.parent.mkdir(exist_ok=True)
            if not destination.exists():
                try: os.link(cache/'tts'/name,destination)
                except OSError: shutil.copy2(cache/'tts'/name,destination)
        cache_files(ROOT/'.runtime/wakeword-v4-local/whisper-encoder',work/'whisper-encoder')
        cache_files(cache/'downloads',work/'downloads')
        from wake_tts import download_models,prepare_whisper_encoder,file_sha256
        required=[work/'tts'/name for name in ['v5_5_ru.pt','v4_ru.pt']]
        required += [piper/(voice+suffix) for voices in config['piper_ru_splits'].values() for voice in voices for suffix in ['.onnx','.onnx.json']]
        mms=work/'tts/mms/mms-tts-rus'
        required += [mms/'config.json',mms/'vocab.json']
        missing=[str(file) for file in required if not file.exists()]
        if missing or not any((mms/name).exists() for name in ['model.safetensors','pytorch_model.bin']):
            download_models(work,config)
        else:
            prepare_whisper_encoder(work)
            atomic_json(work/'tts_provenance.json',dict(cache_source=str(cache/'tts'),
                files_sha256={str(file.relative_to(work)):file_sha256(file) for file in [*required,*mms.glob('*.safetensors'),*mms.glob('pytorch_model.bin')]},
                references=['https://github.com/snakers4/silero-models','https://huggingface.co/rhasspy/piper-voices','https://huggingface.co/facebook/mms-tts-rus']))
        from wake_data_v2 import prepare_real
        if not (work/'dataset/manifest.jsonl').exists():
            if args.seed_run: reuse_data_sources(config,args.seed_run)
            prepare_real(config)
            from data import generate
            stage('generating')
            generate(config)
        if args.prepare_only:
            stage('prepared',dataset=str(work/'dataset/manifest.jsonl'))
            return
        checkpoint=work/'checkpoints/latest.pt'
        if args.resume and checkpoint.exists(): config['resume_checkpoint']=str(checkpoint)
        atomic_json(previous,config)
        stage('training',gpu=torch.cuda.get_device_name(0))
        run_training(config)
        stage('exporting')
        from wake_package_v5 import export_student,package,quality_gate
        export_student(config)
        reports,selection=package(config)
        # English recordings are negative background; this model has a Russian target only.
        for name,report in reports.items():
            report['quality_gate']=quality_gate(report['test'],{'ru':report['test_groups']['language']['ru']},config['target_recall'],config['target_fpr'])
            report['interpretation']='Russian whole word anywhere in a complete utterance; synthetic held-out voices, no real positive user validation. No FAR/hour claim.'
            atomic_json(Path(config['output_dir'])/f'training_report_{name}.json',report)
        selected=reports[selection['selected']]
        atomic_json(Path(config['output_dir'])/'training_report.json',selected)
        summary=dict(format=selection['selected'],test=selected['test'],ru=selected['test_groups']['language']['ru'],
                     quality_gate=selected['quality_gate'],model_bytes=selected['model_bytes'],
                     archive=str(Path(config['output_dir'])/'wake-model.zip'),word=word,task='whole_word_anywhere',
                     real_user_validation=False,dataset_exported=False,teacher_exported=False)
        atomic_json(Path(config['output_dir'])/'output_summary.json',summary)
        stage('complete',result=summary)
        print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    except BaseException as error:
        stage('failed',error=f'{type(error).__name__}: {error}')
        traceback.print_exc()
        raise


if __name__=='__main__': main()
