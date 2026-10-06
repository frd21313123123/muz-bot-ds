"""Choose single/combined word detectors on validation, calibrate, then report test."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.runtime/wake-validation'),str(ROOT/'scripts/wakeword'),str(Path(__file__).parent)]
import numpy as np
from wake_data import atomic_json
from wake_metrics import point_for_config,selection_score,calibrate,confusion
from improve import PackedFeatureDataset


def blend(left,right,weight):
    def logit(value):
        value=np.clip(np.asarray(value,dtype=np.float64),1e-7,1-1e-7)
        return np.log(value)-np.log1p(-value)
    value=weight*logit(left)+(1-weight)*logit(right)
    return 1/(1+np.exp(-value))


def merge(left,right,path,weight):
    import onnx
    from onnx import helper,numpy_helper,compose,TensorProto
    a=compose.add_prefix(onnx.load(str(left)),'a_');b=compose.add_prefix(onnx.load(str(right)),'b_')
    # Both branches use exactly the same mel tensor. Keep one dynamic batch input.
    model=compose.merge_models(a,b,io_map=[])
    for node in model.graph.node:
        for i,name in enumerate(node.input):
            if name=='b_log_mel':node.input[i]='a_log_mel'
    inputs=[value for value in model.graph.input if value.name!='b_log_mel']
    del model.graph.input[:];model.graph.input.extend(inputs)
    nodes=model.graph.node;initializers=model.graph.initializer
    def const(name,value):initializers.append(numpy_helper.from_array(np.asarray(value,np.float64),name))
    const('lower',1e-7);const('upper',1-1e-7);const('one',1.);const('wa',weight);const('wb',1-weight)
    for prefix in ['a','b']:
        nodes.append(helper.make_node('Cast',[prefix+'_probability'],[prefix+'_double'],to=TensorProto.DOUBLE))
        nodes.append(helper.make_node('Clip',[prefix+'_double','lower','upper'],[prefix+'_p']))
        nodes.append(helper.make_node('Sub',['one',prefix+'_p'],[prefix+'_q']))
        nodes.append(helper.make_node('Log',[prefix+'_p'],[prefix+'_lp']))
        nodes.append(helper.make_node('Log',[prefix+'_q'],[prefix+'_lq']))
        nodes.append(helper.make_node('Sub',[prefix+'_lp',prefix+'_lq'],[prefix+'_logit']))
        nodes.append(helper.make_node('Mul',[prefix+'_logit','w'+prefix],[prefix+'_weighted']))
    nodes.append(helper.make_node('Add',['a_weighted','b_weighted'],['sum_logit']))
    nodes.append(helper.make_node('Sigmoid',['sum_logit'],['probability']))
    del model.graph.output[:];model.graph.output.append(helper.make_tensor_value_info('probability',TensorProto.DOUBLE,['batch']))
    model.graph.input[0].name='log_mel'
    for node in nodes:
        for i,name in enumerate(node.input):
            if name=='a_log_mel':node.input[i]='log_mel'
    onnx.checker.check_model(model);onnx.save(model,str(path))


def scores(model,dataset):
    import onnxruntime as ort
    from torch.utils.data import DataLoader
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(model),options,providers=['CPUExecutionProvider'])
    values=[]
    for features,_ in DataLoader(dataset,batch_size=64):values.extend(session.run(['probability'],{'log_mel':features.numpy()})[0].tolist())
    result=np.asarray(values,np.float64)
    if result.shape!=(len(dataset),) or not np.isfinite(result).all():raise ValueError('Invalid word scores')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',default='bot-best-v3')
    parser.add_argument('--extra',nargs='+',help='Additional completed local Tiny binary training runs')
    parser.add_argument('--source',default='bot-data-v3')
    args=parser.parse_args();runs=ROOT/'word_training/runs';work=runs/args.run
    if work.parent.resolve()!=runs.resolve():raise ValueError('Use local run name')
    work.mkdir(exist_ok=True);source=runs/args.source
    if source.parent.resolve()!=runs.resolve():raise ValueError('Use local source name')
    config=dict(target_fpr=.005,target_recall=.9,source_groups=True,source_group_fpr_constraint=False)
    members=[dict(name='binary_v2',directory=runs/'bot-v2/wake-model'),dict(name='binary_v3',directory=runs/'bot-binary-v3/wake-model'),
             dict(name='phonetic_tiny_v3',directory=runs/'bot-phonetic-tiny-v3/model')]
    for name in args.extra or []:
        extra=runs/name
        if extra.parent.resolve()!=runs.resolve():raise ValueError('Use local run name')
        members.append(dict(name=name,directory=extra/'wake-model'))
    for member in members:
        directory=member['directory'];metadata=json.loads((directory/'wake_config.json').read_text(encoding='utf-8'))
        if metadata['model_sha256']!=hashlib.sha256((directory/'wake_model.onnx').read_bytes()).hexdigest():raise ValueError('Model export incomplete or changed')
    validation=PackedFeatureDataset(source/'dataset','validation')
    predictions=[]
    for member in members:
        values=scores(member['directory']/'wake_model.onnx',validation);predictions.append(values)
        print(f'[validation] {member["name"]}',flush=True)
    labels=[row['label'] for row in validation.rows];candidates=[]
    for index,values in enumerate(predictions):
        point=point_for_config(dict(labels=labels,scores=values),validation.rows,config)
        candidates.append(dict(members=[index],weight=1.,point=point,score=selection_score(point,0,config)))
    for a,b in itertools.combinations(range(len(members)),2):
        for weight in [.25,.5,.75]:
            point=point_for_config(dict(labels=labels,scores=blend(predictions[a],predictions[b],weight)),validation.rows,config)
            candidates.append(dict(members=[a,b],weight=weight,point=point,score=selection_score(point,0,config)))
    selected=max(candidates,key=lambda item:item['score'])
    atomic_json(work/'selection.json',dict(selected=selected,candidates=candidates,models=[m['name'] for m in members],decision_data='Validation only; model and blend weight frozen before calibration/test'))
    print('[selected] '+json.dumps(dict(models=[members[i]['name'] for i in selected['members']],weight=selected['weight'],
          recall=selected['point']['macro_recall'],worst_source=selected['point']['worst_source_recall']),ensure_ascii=False),flush=True)
    runtime=work/'model';runtime.mkdir(exist_ok=True);model_file=runtime/'wake_model.onnx'
    chosen=selected['members']
    if len(chosen)==1:
        directory=members[chosen[0]]['directory'];shutil.copy2(directory/'wake_model.onnx',model_file)
        expected=predictions[chosen[0]]
    else:
        merge(members[chosen[0]]['directory']/'wake_model.onnx',members[chosen[1]]['directory']/'wake_model.onnx',model_file,selected['weight'])
        expected=blend(predictions[chosen[0]],predictions[chosen[1]],selected['weight'])
    actual=scores(model_file,validation);np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=1e-8)
    atomic_json(work/'onnx_validation.json',dict(max_probability_difference=float(np.max(np.abs(actual-expected))),windows=len(validation)))
    for name in ['wake_audio.py']:shutil.copy2(ROOT/'scripts/wakeword'/name,runtime/name)
    for name in ['whisper_mel_filters.npy','provenance.json']:shutil.copy2(source/'whisper-encoder'/name,runtime/name)
    shutil.copy2(ROOT/'word_training/detect.py',runtime/'detect.py')
    cal=PackedFeatureDataset(source/'dataset','calibration');cal_scores=scores(model_file,cal)
    threshold,calibration=calibrate([row['label'] for row in cal.rows],cal_scores,.005,[row['language'] for row in cal.rows])
    atomic_json(runtime/'wake_config.json',dict(architecture='whisper_encoder_v4',sample_rate=16000,window_samples=32000,
                wake_word_ru='бот',wake_word_en=None,standalone_only=False,task='whole_word_anywhere',threshold=threshold,
                quality_validated_on_real_users=False,selected_models=[members[i]['name'] for i in chosen],blend_weight=selected['weight'],
                model_sha256=hashlib.sha256(model_file.read_bytes()).hexdigest(),input='log_mel float32 [batch,1,80,200]',output='probability [batch]'))
    # The test split is opened only after selection, graph validation and calibration.
    test=PackedFeatureDataset(source/'dataset','test');test_scores=scores(model_file,test);test_labels=[row['label'] for row in test.rows]
    metrics=confusion(test_labels,test_scores,threshold);groups={}
    for field in ['speaker','kind','generator','language']:
        groups[field]={}
        for value in sorted({row[field] for row in test.rows}):
            indices=[i for i,row in enumerate(test.rows) if row[field]==value]
            groups[field][value]=confusion(np.asarray(test_labels)[indices],test_scores[indices],threshold)
    report=dict(validation=selected['point'],calibration=calibration,threshold=threshold,test=metrics,test_groups=groups,
                model_bytes=model_file.stat().st_size,selected_models=[members[i]['name'] for i in chosen],blend_weight=selected['weight'],
                quality_gate=dict(passed=metrics['recall']>=.9 and metrics['false_positive_rate_per_window']<=.005,target_recall=.9,target_fpr=.005),
                real_user_validation=False,decision_data='Validation selects models/weight; calibration fixes threshold; test reports only')
    output=runs/(args.run+'-output');output.mkdir(exist_ok=True)
    atomic_json(output/'training_report.json',report);shutil.copy2(work/'selection.json',output/'selection.json')
    for split,dataset,values in [('calibration',cal,cal_scores),('test',test,test_scores)]:
        atomic_json(output/(split+'_predictions.json'),[dict(id=r['id'],label=r['label'],text=r['text'],speaker=r['speaker'],probability=float(v)) for r,v in zip(dataset.rows,values,strict=True)])
    (runtime/'README.txt').write_text('Whole-word BOT detector. No transcript or command output.\nInstall numpy onnxruntime. Run python detect.py MODEL_FOLDER recording.wav (mono PCM16 16kHz).\nDataset, TTS, teacher and training checkpoints are unnecessary for inference. See training_report.json for measured quality.\n',encoding='utf-8')
    files=['wake_model.onnx','wake_config.json','wake_audio.py','whisper_mel_filters.npy','provenance.json','detect.py','README.txt']
    with zipfile.ZipFile(output/'wake-model.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in files:archive.write(runtime/name,name)
    with zipfile.ZipFile(output/'wake-model.zip') as archive:assert archive.testzip() is None and set(archive.namelist())==set(files)
    atomic_json(output/'output_summary.json',dict(archive=str(output/'wake-model.zip'),model_bytes=model_file.stat().st_size,
                archive_bytes=(output/'wake-model.zip').stat().st_size,test=metrics,dataset_exported=False,teacher_exported=False))
    print(json.dumps(dict(selected_models=report['selected_models'],test=metrics,model_bytes=report['model_bytes'],
          quality_gate=report['quality_gate']),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
