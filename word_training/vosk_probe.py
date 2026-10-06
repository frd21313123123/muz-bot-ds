"""Validation-only comparison: Russian phonetic word graph, output only BOT score."""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.runtime/wake-validation'),str(ROOT/'scripts/wakeword'),str(Path(__file__).parent)]
import numpy as np
from wake_audio import read_wav,right_aligned_window
from wake_data import atomic_json
from wake_metrics import point_for_config
from data import text_sets
from teacher_phonetic import rows_for


class WordGraphDetector:
    def __init__(self,model_directory,grammar):
        from vosk import Model,KaldiRecognizer,SetLogLevel
        SetLogLevel(-1)
        self.model=Model(str(model_directory))
        self.recognizer=KaldiRecognizer(self.model,16000,json.dumps(grammar,ensure_ascii=False))
        self.recognizer.SetWords(True)
    def score(self,audio):
        value=right_aligned_window(np.asarray(audio,dtype=np.float32))
        pcm=np.rint(np.clip(value,-1,32767/32768)*32768).astype(np.int16)
        self.recognizer.Reset();self.recognizer.AcceptWaveform(pcm.tobytes())
        result=json.loads(self.recognizer.FinalResult())
        return max([item['conf'] for item in result.get('result',[]) if item['word']=='бот'],default=0.)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',default='bot-v2');args=parser.parse_args()
    source=ROOT/'word_training/runs'/args.source
    if source.parent.resolve()!=(ROOT/'word_training/runs').resolve():raise ValueError('Use a local run name')
    output=ROOT/'word_training/runs/bot-word-graph';output.mkdir(exist_ok=True)
    positives,negatives=text_sets('бот')
    words=sorted(set(' '.join(positives+negatives).split()))
    grammar=words+['[unk]'];atomic_json(output/'grammar.json',grammar)
    detector=WordGraphDetector(ROOT/'.runtime/wake-vosk/vosk-model-small-ru-0.22',grammar)
    rows=rows_for(source,'validation');values=[]
    for index,row in enumerate(rows):
        values.append(detector.score(read_wav(source/'dataset'/row['wav'])))
        if (index+1)%300==0:print(f'[word graph validation] {index+1}/{len(rows)}',flush=True)
    config=json.loads((source/'run_config.json').read_text(encoding='utf-8'))
    point=point_for_config(dict(labels=[r['label'] for r in rows],scores=values),rows,config)
    atomic_json(output/'validation.json',point)
    atomic_json(output/'validation_predictions.json',[dict(id=r['id'],label=r['label'],probability=v) for r,v in zip(rows,values,strict=True)])
    print(json.dumps(dict(recall=point['macro_recall'],worst=point['worst_source_recall'],threshold=point['threshold'],metrics=point['metrics']),indent=2),flush=True)


if __name__=='__main__':main()
