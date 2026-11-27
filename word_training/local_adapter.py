"""Fit a CUDA residual head on frozen, deployed acoustic embeddings and fold it into ONNX."""
import copy
import json
import shutil
import time

import numpy as np
import onnx
from onnx import helper, numpy_helper
import torch
from torch import nn

from local_features import session, sha, taps
from wake_audio import WhisperMelFrontend


class ResidualHead(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.hidden = nn.Linear(inputs, 64)
        self.output = nn.Linear(64, outputs)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        return self.output(torch.tanh(self.hidden(x)))


def probabilities(logits):
    if logits.shape[1] == 1:
        return 1 / (1 + np.exp(-np.clip(logits[:, 0], -80, 80)))
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def metrics(task, labels, p, threshold, classes):
    result = dict(windows=len(labels))
    if task == 'wake':
        hit = p >= threshold
        pos = labels == 1
        tp, fp = int(np.sum(hit & pos)), int(np.sum(hit & ~pos))
        positives, negatives = int(pos.sum()), int((~pos).sum())
        result.update(tp=tp, fp=fp, positives=positives, negatives=negatives,
                      recall=tp/positives if positives else None, fpr=fp/negatives if negatives else None)
    else:
        prediction = p.argmax(axis=1)
        per_class = {name: float(np.mean(prediction[labels == i] == i))
                     for i, name in enumerate(classes) if (labels == i).any()}
        control = ~np.isin(prediction, [classes.index('play_search'), classes.index('unknown')])
        acts = control & (p.max(axis=1) >= threshold)
        result.update(correct=int((prediction == labels).sum()), accuracy=float(np.mean(prediction == labels)),
                      macro_recall=float(np.mean(list(per_class.values()))), class_recall=per_class,
                      correct_controls=int((acts & (prediction == labels)).sum()),
                      wrong_controls=int((acts & (prediction != labels)).sum()))
    return result


def by_origin(task, rows, mask, labels, probabilities_, threshold, classes):
    origins = np.array([r['origin'] for r in rows])
    masks = {'all': mask, **{g: mask & (origins == g) for g in sorted(set(origins[mask]))}}
    return {k: metrics(task, labels[m], probabilities_[m], threshold, classes) for k, m in masks.items()}


def selection(task, result, base):
    if task == 'wake':
        # Protect each negative population, including the unaugmented personal recordings.
        safe = all(m['fp'] <= base[k]['fp'] + int(.005 * m['negatives'])
                   for k, m in result.items() if k != 'all')
        score = np.mean([m['recall'] - 2 * m['fpr'] for k, m in result.items()
                         if k != 'all' and m['recall'] is not None and m['fpr'] is not None])
    else:
        safe = all(m['wrong_controls'] <= base[k]['wrong_controls'] + int(.005 * m['windows'])
                   for k, m in result.items() if k != 'all')
        score = np.mean([m['macro_recall'] + (m['correct_controls'] - 2*m['wrong_controls']) / m['windows']
                         for k, m in result.items() if k != 'all'])
    return bool(safe), float(score)


def export(model_file, target, task, head, mean, scale):
    model = onnx.load(model_file)
    pooled, logits = taps(model, task)
    insertion = next(i for i, n in enumerate(model.graph.node) if logits in n.input)
    downstream = model.graph.node[insertion]
    downstream.input[list(downstream.input).index(logits)] = 'adapt_logits'
    def parameter(name, value):
        model.graph.initializer.append(numpy_helper.from_array(value.astype(np.float32), name))
        return name
    # Fold the training normalization into the first affine layer.
    h = head.hidden.weight.detach().cpu().numpy().T / scale[:, None]
    b = head.hidden.bias.detach().cpu().numpy() - mean @ h
    w = head.output.weight.detach().cpu().numpy().T
    c = head.output.bias.detach().cpu().numpy()
    nodes = [
        helper.make_node('Gemm', [pooled, parameter('adapt_w1', h), parameter('adapt_b1', b)], ['adapt_hidden']),
        helper.make_node('Tanh', ['adapt_hidden'], ['adapt_tanh']),
        helper.make_node('Gemm', ['adapt_tanh', parameter('adapt_w2', w), parameter('adapt_b2', c)], ['adapt_delta']),
        helper.make_node('Add', [logits, 'adapt_delta'], ['adapt_logits']),
    ]
    for i, node in enumerate(nodes):
        model.graph.node.insert(insertion + i, node)
    onnx.checker.check_model(model)
    onnx.save(model, target)
    return session(model)


def fit_and_export(work, base, task, all_rows, cache, deadline):
    started = time.monotonic()
    ids, embeddings, logits = cache
    rows = [all_rows[i] for i in ids]
    source = base / (task + '-model')
    config_file = task + '_config.json'
    config = json.loads((source / config_file).read_text(encoding='utf8'))
    classes = config.get('classes', [])
    threshold = config['threshold' if task == 'wake' else 'default_threshold']
    labels = np.array([int(r[task]) if task == 'wake' else classes.index(r[task]) for r in rows])
    folds = np.array([r['split'] for r in rows])
    train, val, test = (folds == f for f in ('train', 'validation', 'test'))
    if not all(m.any() for m in (train, val, test)):
        raise ValueError('Missing split')
    mean = embeddings[train].mean(axis=0)
    scale = embeddings[train].std(axis=0).clip(.05)
    x = torch.from_numpy((embeddings - mean) / scale).cuda()
    z = torch.from_numpy(logits).cuda()
    y = torch.from_numpy(labels).long().cuda()
    torch.manual_seed(177)
    head = ResidualHead(x.shape[1], z.shape[1]).cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=.003, weight_decay=.015)
    train_ids = np.flatnonzero(train)
    origins = np.array([r['origin'] for r in rows])
    weights = np.zeros(len(rows), np.float64)
    for group in set(origins[train]):
        group_mask = train & (origins == group)
        present = np.unique(labels[group_mask])
        mass = {'drive': .10, 'legacy': .45, 'tts': .45 if task == 'wake' else .90}[group]
        for cls in present:
            members = group_mask & (labels == cls)
            weights[members] = mass / (len(present) * members.sum())
    weight = torch.from_numpy(weights[train]).float().cuda()
    indices = torch.from_numpy(train_ids).cuda()
    baseline = probabilities(logits)
    val_base = by_origin(task, rows, val, labels, baseline, threshold, classes)
    best_key = (False, -np.inf)
    history = []
    valid_rows = [r for r, keep in zip(rows, val) if keep]
    for epoch in range(1, 31):
        if time.monotonic() >= deadline - 5:
            raise TimeoutError('Training budget exhausted')
        samples = indices[torch.multinomial(weight, min(6144, len(train_ids)), replacement=True)]
        for batch in samples.split(256):
            delta = head(x[batch])
            pred = z[batch] + delta
            loss = (nn.functional.binary_cross_entropy_with_logits(pred[:, 0], y[batch].float()) if task == 'wake'
                    else nn.functional.cross_entropy(pred, y[batch], label_smoothing=.015))
            loss = loss + .0005 * delta.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            # Only validation is consulted for epoch selection.
            scores = probabilities((z[val] + head(x[val])).cpu().numpy())
        measured = by_origin(task, valid_rows, np.ones(len(scores), bool), labels[val], scores, threshold, classes)
        key = selection(task, measured, val_base)
        history.append(dict(epoch=epoch, safe=key[0], validation_score=key[1]))
        if key > best_key:
            best_key, best_epoch, best_state = key, epoch, copy.deepcopy(head.state_dict())
    head.load_state_dict(best_state)
    with torch.no_grad():
        predicted = probabilities((z + head(x)).cpu().numpy())
    out = work / 'candidates' / (task+'-model')
    out.mkdir(parents=True, exist_ok=True)
    filename = task + '_model.onnx'
    runner = export(source / filename, out / filename, task, head, mean, scale)
    for name in (task + '_audio.py', 'provenance.json', 'whisper_mel_filters.npy', 'README.txt'):
        shutil.copy2(source / name, out / name)
    for stale in ('comparison', 'avg_latency_ms', 'accuracy', 'dataset_samples', 'best_epoch'):
        config.pop(stale, None)
    config.update(recipe_version='local-archive-residual-v1', model_sha256=sha(out / filename),
                  model_size_bytes=(out / filename).stat().st_size, base_model_sha256=sha(source / filename),
                  best_epoch=best_epoch, quantization='int8_backbone_fp32_residual')
    if task == 'command':
        config['model_name'] = 'whisper_tiny_spoken_commands_local_v1'
    (out / config_file).write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    frontend = WhisperMelFrontend(out)
    audio = np.load(work / 'audio.npy', mmap_mode='r')
    # Score a stratified sample with the exported graph; dynamic INT8 must use batch one.
    checked = set()
    max_error = 0.
    for i in np.flatnonzero(test):
        key = rows[i]['origin'], labels[i]
        if key in checked:
            continue
        checked.add(key)
        mel = frontend(audio[ids[i]].astype(np.float32) / 32768)
        actual = runner.run(None, {'log_mel': mel[None]})[0][0]
        max_error = max(max_error, float(np.max(np.abs(actual - predicted[i]))))
    if max_error > 2e-4:
        raise ValueError(f'ONNX/PyTorch mismatch: {max_error}')
    report = dict(threshold=threshold, base_model_sha256=config['base_model_sha256'],
                  model_sha256=config['model_sha256'], selected_epoch=best_epoch, validation_history=history,
                  onnx_max_error=max_error, windows={f:int((folds == f).sum()) for f in set(folds)})
    for name, mask in [('validation', val), ('test', test)]:
        b = by_origin(task, rows, mask, labels, baseline, threshold, classes)
        c = by_origin(task, rows, mask, labels, predicted, threshold, classes)
        report[name] = dict(baseline=b, candidate=c)
        safe, score = selection(task, c, b)
        report[name]['passes'] = safe and score > selection(task, b, b)[1]
    report['seconds'] = time.monotonic() - started
    print(task, 'selected epoch', best_epoch, 'test', report['test']['candidate']['all'], flush=True)
    return report
