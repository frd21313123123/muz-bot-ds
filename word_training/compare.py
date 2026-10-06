"""Compare regularized Tiny recipes on validation, then export one local word detector."""
import argparse
import json
from pathlib import Path
import shutil
import time
from train import ROOT,run_training
from wake_data import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='bot-v1')
    parser.add_argument('--comparison',choices=['regularization','learning-rate'],default='regularization')
    args=parser.parse_args()
    work=ROOT/'word_training/runs'/args.run
    if work.parent.resolve()!=(ROOT/'word_training/runs').resolve(): raise ValueError('Use a run name, not a path')
    status=work/'status.json'
    config=json.loads((work/'run_config.json').read_text(encoding='utf-8'))
    baseline=json.loads(status.read_text(encoding='utf-8'))
    if baseline.get('stage')!='complete': raise ValueError('Complete the initial run before comparing recipes')
    output=Path(config['output_dir'])
    backup=output.with_name(output.name+'-initial')
    if not backup.exists(): shutil.copytree(output,backup)
    # Preserve the initial recipe, checkpoints and quality report; no model is selected on test.
    atomic_json(work/'initial_result.json',baseline)
    member=work/'checkpoints/member-0'; member.mkdir(exist_ok=True)
    shutil.copy2(work/'checkpoints/best.pt',member/'best.pt')
    shutil.copy2(work/'training_history.json',work/'training_history-member-0.json')
    shutil.copy2(work/'training_run.json',work/'training_run-member-0.json')
    initial=dict(config['member_overrides']['0'])
    overrides={'0':initial}
    if args.comparison=='learning-rate':
        overrides['1']=dict(initial,encoder_learning_rate=1e-5)
        if overrides['1']==initial: raise ValueError('The initial recipe already uses this learning rate')
    else:
        overrides.update({'1':dict(distill_weight=0,encoder_train_layers=2,encoder_learning_rate=3e-6),
                          '2':dict(distill_weight=0,encoder_train_layers=4,encoder_learning_rate=3e-6)})
    config.update(ensemble_members=len(overrides),resume_checkpoint=None,member_overrides=overrides)
    atomic_json(work/'comparison_config.json',config)
    def stage(name,**extra):
        atomic_json(status,dict(stage=name,word=config['wake_word_ru'],updated_at=time.time(),**extra))
        print(f'[{name}] {extra}',flush=True)
    try:
        import torch
        from wake_train import train
        for candidate in range(1,len(overrides)):
            stage('comparison_training',member=candidate,recipe=config['member_overrides'][str(candidate)])
            with torch.inference_mode(False),torch.enable_grad(): train(config,member=candidate)
        stage('comparison_export')
        from wake_package_v5 import export_student,package,quality_gate
        export_student(config)
        reports,selection=package(config)
        for name,report in reports.items():
            report['quality_gate']=quality_gate(report['test'],{'ru':report['test_groups']['language']['ru']},config['target_recall'],config['target_fpr'])
            report['interpretation']='Russian whole word anywhere in a phrase. Recipes/format selected only on validation. Reused synthetic regression benchmark; no independent real user positives.'
            atomic_json(output/f'training_report_{name}.json',report)
        selected=reports[selection['selected']]
        atomic_json(output/'training_report.json',selected)
        selected_members=json.loads((work/'ensemble_selection.json').read_text(encoding='utf-8'))['selected_members']
        for stem in ['training_history','training_run']:
            for candidate in range(len(overrides)):
                file=work/f'{stem}-member-{candidate}.json'
                shutil.copy2(file,output/file.name)
            shutil.copy2(work/f'{stem}-member-{selected_members[0]}.json',output/f'{stem}.json')
        atomic_json(output/'run_config.json',config)
        summary=dict(format=selection['selected'],test=selected['test'],ru=selected['test_groups']['language']['ru'],
            quality_gate=selected['quality_gate'],model_bytes=selected['model_bytes'],archive=str(output/'wake-model.zip'),
            word=config['wake_word_ru'],task='whole_word_anywhere',real_user_validation=False,dataset_exported=False,teacher_exported=False,
            selected_members=selected_members)
        atomic_json(output/'output_summary.json',summary)
        stage('complete',result=summary)
        print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    except BaseException as error:
        stage('failed',error=f'{type(error).__name__}: {error}'); raise


if __name__=='__main__': main()
