"""Paired comparison on complete held-out WAVs through the bot's acoustic command worker."""
import argparse
import base64
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import wave

ROOT = Path(__file__).resolve().parents[1]
CONTROLS = {'skip','pause','resume','stop','volume_up','volume_down','loop_on','autoplay_on','queue_clear'}


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def worker_results(python, rows, archive, directory, expected_name):
    env = dict(os.environ, VOICE_PROFILE='light', STT_PROVIDER='groq', PYTHONIOENCODING='utf8',
               OPENBLAS_NUM_THREADS='1')
    env.pop('PYTHONPATH', None)  # Use the voice worker's installed runtime libraries.
    if directory is None:
        env.pop('COMMAND_MODEL_PATH', None)
    else:
        env['COMMAND_MODEL_PATH'] = str(directory)
    requests = []
    for i,r in enumerate(rows):
        path = archive/'drive'/r['file']
        if digest(path) != r['sha256']:
            raise ValueError('Changed holdout recording: ' + path.name)
        with wave.open(str(path)) as wav:
            if (wav.getframerate(),wav.getnchannels(),wav.getsampwidth()) != (16000,1,2):
                raise ValueError('Expected 16 kHz mono PCM16')
            pcm = wav.readframes(wav.getnframes())
        requests.append(dict(id=i,op='command_detect',pcm=base64.b64encode(pcm).decode('ascii')))
    run = subprocess.run([str(python),'-u',str(ROOT/'scripts/voice_worker.py')],cwd=ROOT,env=env,
                         input=''.join(json.dumps(x)+'\n' for x in requests),capture_output=True,
                         text=True,encoding='utf8',timeout=120,check=True)
    ready,*replies = [json.loads(x) for x in run.stdout.splitlines()]
    if not ready.get('ready') or ready.get('commandModelName') != expected_name or len(replies)!=len(rows):
        raise ValueError('Unexpected worker model or reply count')
    results = []
    for i,reply in enumerate(replies):
        pred=reply['result']
        if reply['id']!=i or not math.isfinite(pred['confidence']):
            raise ValueError('Invalid worker result')
        results.append({k:pred[k] for k in ('class','action','matched','confidence')})
    return results


def scores(rows, key):
    correct = commands = right_commands = right_controls = wrong_controls = 0
    for r in rows:
        pred = r[key]
        hit = r['command'] == pred['class']
        correct += hit
        if r['command'] != 'unknown':
            commands += 1
            right_commands += hit
        if pred['action'] in CONTROLS and pred['matched']:
            right_controls += hit
            wrong_controls += not hit
    return dict(recordings=len(rows), correct=correct, commands=commands, correct_commands=right_commands,
                direct_correct=right_controls, direct_wrong=wrong_controls)


def better(old, new):
    gain = new['correct_commands'] > old['correct_commands'] or new['direct_correct'] > old['direct_correct']
    return (gain and new['correct_commands'] >= old['correct_commands'] and new['direct_correct'] >= old['direct_correct']
            and new['correct'] >= old['correct'] and new['direct_wrong'] <= old['direct_wrong'])


def replace_file(source, target):
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
        staged = Path(tmp.name)
    try:
        shutil.copyfile(source, staged)
        staged.replace(target)
    finally:
        staged.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,default=ROOT/'.runtime/personal-command-v2')
    parser.add_argument('--archive',type=Path,default=ROOT/'.runtime/local-voice')
    parser.add_argument('--worker-python',type=Path,default=ROOT/'.runtime/voice-light/Scripts/python.exe' if os.name=='nt' else sys.executable)
    parser.add_argument('--install-if-better',action='store_true')
    parser.add_argument('--criterion', choices=('class-and-action','direct-controls'), default='class-and-action',
                        help='Compare all class decisions or prioritize correct controls accepted by the bot')
    args=parser.parse_args()
    work, archive=args.work.resolve(),args.archive.resolve()
    training=json.loads((work/'training.json').read_text(encoding='utf8'))
    if digest(work/'labels.json') != training['labels_sha256']:
        raise ValueError('Reference labels changed after training')
    rows=[r for r in json.loads((work/'labels.json').read_text(encoding='utf8'))
          if r['split']=='holdout' and r['command'] is not None]
    if not rows or any(r['group'] in training['groups'] for r in rows) or len({r['sha256'] for r in rows}) != len(rows):
        raise ValueError('Invalid recording holdout')
    active=ROOT/'models/command-model'
    original, candidate=archive/'baseline/command-model',work/'candidate/command-model'
    model='command_model.onnx'
    config='command_config.json'
    configs={}
    for key,folder in [('baseline',original),('candidate',candidate)]:
        cfg=json.loads((folder/config).read_text(encoding='utf8'))
        if (digest(folder/model) != training[key+'_sha256'] or cfg['default_threshold'] != training['threshold']
                or cfg['classes'] != training['classes']):
            raise ValueError('Model or decision contract changed')
        configs[key]=cfg
        preds=worker_results(args.worker_python,rows,archive,folder,cfg['model_name'])
        for r, pred in zip(rows, preds, strict=True):
            if pred['matched'] != (pred['confidence'] >= cfg['default_threshold']):
                raise ValueError('Unexpected decision threshold')
            r[key]=pred
    if configs['baseline']['class_to_action'] != configs['candidate']['class_to_action']:
        raise ValueError('Changed command-to-action mapping')
    subsets={'all':rows,'previously_unused':[r for r in rows if r['previous_command_split'] is None],
             'previous_test':[r for r in rows if r['previous_command_split']=='test']}
    summary={s:{k:scores(v,k) for k in configs} for s,v in subsets.items()}
    old, new = summary['all']['baseline'], summary['all']['candidate']
    gates = {'class-and-action': better(old, new),
             'direct-controls': (new['direct_correct'] > old['direct_correct']
                                 and new['direct_wrong'] <= old['direct_wrong'] and new['correct'] >= old['correct'])}
    won = gates[args.criterion]
    report=dict(passes=won,counts=summary,classes=dict(Counter(r['command'] for r in rows)),
                criterion=args.criterion, criteria_results=gates,
                candidate_sha256=training['candidate_sha256'],baseline_sha256=training['baseline_sha256'],
                labels_sha256=training['labels_sha256'],threshold=training['threshold'],rows=rows)
    if won and args.install_if_better:
        files=(model,config)
        if not all(digest(active/f)==digest(original/f) for f in files) and not all(
                digest(active/f)==digest(candidate/f) for f in files):
            raise ValueError('Installed model changed since the saved baseline')
        for f in ('command_audio.py','whisper_mel_filters.npy','provenance.json'):
            if digest(active/f)!=digest(candidate/f):
                raise ValueError('Installed frontend differs from the candidate')
        try:
            for f in files:
                replace_file(candidate/f,active/f)
            verified=worker_results(args.worker_python,rows,archive,None,configs['candidate']['model_name'])
            if verified != [r['candidate'] for r in rows]:
                raise ValueError('Installed worker disagrees with the candidate')
        except BaseException:
            for f in files:
                replace_file(original/f,active/f)
            raise
    report['active_sha256']=digest(active/model)
    report['installed_verified']=won and args.install_if_better
    (work/'comparison.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps(dict(passes=won, installed=report['installed_verified'], counts=summary),indent=2))


if __name__=='__main__':
    main()
