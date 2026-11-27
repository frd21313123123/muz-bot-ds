"""Fit a personal command adapter, selecting its recipe by held-out recording hours within development."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from local_adapter import ResidualHead, export, probabilities
from local_corpus import variant
from local_features import ROOT, cache_features, sha
from wake_audio import read_wav, right_aligned_window

STEPS = (100, 280, 520)
PERSONAL_WEIGHTS = (.25, .50, .75)
# Search requests and out-of-scope speech both go to transcription, but only
# the former is a supported command. Compare shrinkage of the three outputs.
SCALES = ((.35,.35,.35), (.70,.70,.70), (1.,1.,1.), (.35,1.,1.), (1.,.35,.35),
          (.35,.35,1.), (.70,.35,1.), (1.,.35,1.), (.70,1.,.35))


def output_scales(classes, control_scale, search_scale, unknown_scale):
    other = {'unknown': unknown_scale, 'play_search': search_scale}
    return np.array([other.get(k, control_scale) for k in classes],
                    dtype=np.float32)


def load_labels(work):
    rows = json.loads((work / 'labels.json').read_text(encoding='utf8'))
    for key in ('file', 'sha256', 'group'):
        groups = defaultdict(set)
        for row in rows:
            groups[row[key]].add(row['split'])
        if any(len(v) != 1 for v in groups.values()):
            raise ValueError(f'{key} overlaps development and final holdout')
    if len({r['file'] for r in rows}) != len(rows):
        raise ValueError('Expected one reference label per original recording')
    return [r for r in rows if r['command'] is not None]


def prepare_personal(work, archive, base, development, deadline):
    directory = work / 'development'
    directory.mkdir(exist_ok=True)
    rows, examples = [], []
    for row in development:
        path = archive / 'drive' / row['file']
        if sha(path) != row['sha256']:
            raise ValueError(f'Recording changed: {path.name}')
        clip = read_wav(path)
        # The clean example always follows CommandDetector's right alignment.
        # For long training clips, allow temporal jitter around the active speech.
        active = np.flatnonzero(np.abs(clip) > max(.0015, np.max(np.abs(clip)) * .02))
        speech = clip[max(0, active[0]-320):active[-1]+641] if len(active) else clip
        rng = np.random.default_rng(int(row['sha256'][:16], 16))
        for i in range(65):
            if i == 0:
                sample = right_aligned_window(clip)
            else:
                source = speech
                if len(source) > 32000:
                    start = int(rng.integers(len(source) - 32000 + 1))
                    source = source[start:start+32000]
                sample = variant(source, i, rng)
            rows.append(dict(**row, variant=i, index=len(rows)))
            examples.append(np.rint(sample * 32768).clip(-32768, 32767).astype('<i2'))
    np.save(directory / 'audio.npy', np.stack(examples))
    (directory / 'windows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf8')
    _, x, z = cache_features(directory, base, 'command', rows, sha(directory/'audio.npy'), deadline, 5)
    return rows, x, z


def synthetic_replay(archive, base):
    manifest = archive / 'windows.jsonl'
    rows = [json.loads(line) for line in manifest.read_text(encoding='utf8').splitlines()]
    folder = base / 'command-model'
    expected = [sha(folder/'command_model.onnx'), sha(folder/'whisper_mel_filters.npy'),
                sha(archive/'audio.npy'), sha(manifest)]
    with np.load(archive/'command-features.npz', allow_pickle=False) as cache:
        if cache['identity'].tolist() != expected:
            raise ValueError('The synthetic feature cache does not match the baseline or dataset')
        ids = cache['indices']
        keep = [j for j, i in enumerate(ids) if rows[i]['origin'] == 'tts' and rows[i]['split'] in ('train', 'validation')]
        return [rows[ids[j]] for j in keep], cache['pooled'][keep], cache['logits'][keep]


def measurements(y, p, classes, threshold):
    top = p.argmax(1)
    correct = top == y
    supported = y != classes.index('unknown')
    direct = ~np.isin(top, [classes.index('unknown'), classes.index('play_search')]) & (p.max(1) >= threshold)
    counts = dict(recordings=len(y), correct=int(correct.sum()), supported=int(supported.sum()),
                  supported_correct=int((supported & correct).sum()), direct_correct=int((direct & correct).sum()),
                  direct_wrong=int((direct & ~correct).sum()))
    counts['log_loss'] = float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-9)).mean())
    return counts


def distribution(y, ids, clean=None):
    # Temper class imbalance and give clean natural recordings a third of the sampling mass.
    labels = y[ids]
    counts = Counter(labels.tolist())
    weights = np.array([1 / math.sqrt(counts[k]) for k in labels])
    if clean is not None:
        raw = clean[ids]
        weights[raw] *= weights[~raw].sum() / (2 * weights[raw].sum())
    return torch.as_tensor(weights / weights.sum(), device='cuda', dtype=torch.float32)


def train_heads(x, z, y, y_np, personal, synth, clean, real_weight, max_steps, deadline):
    torch.manual_seed(20261127)
    head = ResidualHead(x.shape[1], z.shape[1]).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=.0018, weight_decay=.02)
    wp = distribution(y_np, personal, clean)
    ws = distribution(y_np, synth)
    personal, synth = (torch.as_tensor(i, device='cuda') for i in (personal, synth))
    for step in range(1, max_steps+1):
        if step % 25 == 1 and time.monotonic() >= deadline:
            raise TimeoutError('Personal training budget exhausted')
        pi = personal[torch.multinomial(wp, 80, replacement=True)]
        si = synth[torch.multinomial(ws, 176, replacement=True)]
        batch = torch.cat((pi, si))
        delta = head(x[batch])
        new = z[batch] + delta
        human_loss = F.cross_entropy(new[:len(pi)], y[pi], label_smoothing=.01)
        speech_loss = F.cross_entropy(new[len(pi):], y[si], label_smoothing=.02)
        # Keep the synthetic replay close to the old model where it is informative.
        distill = F.kl_div(F.log_softmax(new[len(pi):] / 2, dim=1),
                           F.softmax(z[si] / 2, dim=1), reduction='batchmean') * 4
        loss = real_weight * human_loss + (1-real_weight) * (.83 * speech_loss + .17 * distill)
        loss = loss + .002 * delta.square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 3.)
        opt.step()
        if step in STEPS:
            yield step, head


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=ROOT/'.runtime/personal-command-v2')
    parser.add_argument('--archive', type=Path, default=ROOT/'.runtime/local-voice')
    parser.add_argument('--minutes', type=float, default=40)
    args = parser.parse_args()
    if not 0 < args.minutes <= 40:
        parser.error('Training budget must be between 0 and 40 minutes')
    start = time.monotonic()
    deadline = start + 60 * args.minutes
    work, archive = args.work.resolve(), args.archive.resolve()
    base = archive / 'baseline'
    folder = base / 'command-model'
    cfg = json.loads((folder/'command_config.json').read_text(encoding='utf8'))
    classes, threshold = cfg['classes'], cfg['default_threshold']
    if not torch.cuda.is_available():
        raise RuntimeError('Expected CUDA-enabled PyTorch')
    torch.set_num_threads(2)
    dev = [r for r in load_labels(work) if r['split'] == 'development']
    real, rx, rz = prepare_personal(work, archive, base, dev, deadline)
    tts, sx, sz = synthetic_replay(archive, base)
    rows = real + tts
    si = np.array([len(real)+i for i,r in enumerate(tts) if r['split']=='train'])
    sv = np.array([len(real)+i for i,r in enumerate(tts) if r['split']=='validation'])
    emb = np.concatenate((rx, sx))
    mean, std = emb[si].mean(0), emb[si].std(0).clip(.05)
    logits = np.concatenate((rz, sz))
    labels = np.array([classes.index(r['command']) for r in rows])
    x, z, y = [torch.from_numpy(v).cuda() for v in ((emb-mean)/std, logits, labels.astype(np.int64))]
    groups = np.array([r['group'] for r in real])
    clean = np.array([r['variant']==0 for r in real])
    raw = np.flatnonzero(clean)
    groups_used = sorted(set(groups))
    if len(groups_used) < 2:
        raise ValueError('At least two development recording groups are needed')
    oof, replay_score = {}, defaultdict(list)
    for group in groups_used:
        fit = np.flatnonzero(groups != group)
        valid = np.flatnonzero((groups == group) & clean)
        for w in PERSONAL_WEIGHTS:
            for step, head in train_heads(x, z, y, labels, fit, si, clean, w, max(STEPS), deadline):
                with torch.no_grad():
                    rd, sd = head(x[valid]).cpu().numpy(), head(x[sv]).cpu().numpy()
                for scales in SCALES:
                    strength = output_scales(classes, *scales)
                    key = (w, step, *scales)
                    if key not in oof:
                        oof[key] = np.full((len(real), len(classes)), np.nan, dtype=np.float32)
                    oof[key][valid] = probabilities(logits[valid] + strength * rd)
                    p = probabilities(logits[sv] + strength * sd)
                    replay_score[key].append(measurements(labels[sv], p, classes, threshold)['log_loss'])
        print('Validated recording group', group, flush=True)
    old = measurements(labels[raw], probabilities(logits[raw]), classes, threshold)
    scores = []
    for key, predictions in oof.items():
        if not np.isfinite(predictions[raw]).all():
            raise ValueError('Incomplete out-of-fold predictions')
        m = measurements(labels[raw], predictions[raw], classes, threshold)
        scores.append(dict(weight=key[0], steps=key[1], scales=key[2:], personal=m,
                           synthetic_loss=float(np.mean(replay_score[key]))))
    def rank(s):
        m = s['personal']
        retains_commands = (m['supported_correct'] >= old['supported_correct'] and
                            m['direct_correct'] >= old['direct_correct'] and m['correct'] >= old['correct'])
        return (m['direct_wrong'] <= old['direct_wrong'], retains_commands, m['supported_correct'] + m['direct_correct'],
                -m['direct_wrong'], m['correct'], -m['log_loss'] - .25 * s['synthetic_loss'])
    best = max(scores, key=rank)
    final_fit = train_heads(x, z, y, labels, np.arange(len(real)), si, clean,
                            best['weight'], best['steps'], deadline)
    head = next(model for step, model in final_fit if step == best['steps'])
    with torch.no_grad():
        strength = torch.from_numpy(output_scales(classes, *best['scales'])).cuda()
        head.output.weight.mul_(strength[:, None])
        head.output.bias.mul_(strength)
        probe = np.r_[raw, sv[:32]]
        expected = probabilities((z[probe] + head(x[probe])).cpu().numpy())
    out = work / 'candidate' / 'command-model'
    out.mkdir(parents=True, exist_ok=True)
    runner = export(folder/'command_model.onnx', out/'command_model.onnx', 'command', head, mean, std)
    from wake_audio import WhisperMelFrontend
    frontend = WhisperMelFrontend(folder)
    dev_audio = np.load(work/'development/audio.npy', mmap_mode='r')
    old_audio = np.load(archive/'audio.npy', mmap_mode='r')
    for i,k in enumerate(probe):
        pcm = dev_audio[k] if k<len(real) else old_audio[rows[k]['index']]
        p = runner.run(['probabilities'], {'log_mel': frontend(pcm.astype(np.float32)/32768)[None]})[0][0]
        np.testing.assert_allclose(p, expected[i], atol=2e-4, rtol=0)
    for name in ('command_audio.py','whisper_mel_filters.npy','provenance.json','README.txt'):
        shutil.copy2(folder/name, out/name)
    for k in ('accuracy','avg_latency_ms','comparison','dataset_samples'):
        cfg.pop(k,None)
    cfg.update(model_name='whisper_tiny_personal_commands_v2', recipe_version='personal-command-group-cv-v2',
               base_model_sha256=sha(folder/'command_model.onnx'), model_sha256=sha(out/'command_model.onnx'),
               model_size_bytes=(out/'command_model.onnx').stat().st_size, quantization='int8_backbone_fp32_residual')
    (out/'command_config.json').write_text(json.dumps(cfg,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    report = dict(gpu=torch.cuda.get_device_name(), seconds=time.monotonic()-start, classes=classes,
                  threshold=threshold, labels_sha256=sha(work/'labels.json'), baseline_sha256=cfg['base_model_sha256'],
                  candidate_sha256=cfg['model_sha256'], developmental_sources=len(dev), groups=groups_used,
                  developmental_windows=len(real), synthetic_train=len(si), synthetic_validation=len(sv),
                  baseline_cv=old, selected=best, validation_recipes=scores)
    (work/'training.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print('Selected',best,'elapsed',round(report['seconds'],1),'seconds',flush=True)


if __name__ == '__main__':
    main()
