"""torchrun entry point: train, validation and mining share all available ranks."""
import argparse
import copy
import contextlib
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Sampler, Subset
from wake_model import make_model, spec_augment
from wake_data import atomic_json
from wake_metrics import point_for_config, selection_score


class FeatureDataset(Dataset):
    def __init__(self, root, split, preload=False):
        self.root = Path(root)
        self.rows = [json.loads(line) for line in (self.root / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
        self.rows = [row for row in self.rows if row['split'] == split]
        if not self.rows: raise ValueError(f'Empty {split} split')
        self.cached = np.stack([np.load(self.root / row['feature']) for row in self.rows]) if preload else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        feature = (self.cached[index] if self.cached is not None else np.load(self.root / row['feature'])).astype(np.float32)
        return torch.from_numpy(feature), torch.tensor(row['label'], dtype=torch.float32)


class BalancedSampler(Sampler):
    """One deterministic weighted stream, partitioned equally across DDP ranks."""
    def __init__(self, rows, seed, rank=0, world=1):
        self.rows, self.seed, self.rank, self.world, self.epoch = rows, seed, rank, world, 0
        self.keys = [(r['language'], r['label'], r['generator']) for r in rows]
        counts = {}
        for key in self.keys: counts[key] = counts.get(key, 0) + 1
        groups = {}
        for language, label, engine in counts: groups[(language, label)] = groups.get((language, label), 0) + 1
        weights = []
        for key in self.keys:
            language, label, _ = key
            language_mass = {'ru': .7, 'en': .3}.get(language, 0) if label else {'ru': .6, 'en': .3, 'none': .1}.get(language, .1)
            weights.append(language_mass / (counts[key] * groups[(language, label)]))
        self.base = torch.tensor(weights, dtype=torch.double)
        for label in [0, 1]:
            selected = torch.tensor([r['label'] == label for r in rows])
            self.base[selected] *= .5 / self.base[selected].sum()
        self.weights = self.base.clone()
        self.size = (len(rows) + world - 1) // world

    def set_epoch(self, epoch): self.epoch = epoch
    def __len__(self): return self.size
    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(self.weights, self.size * self.world, replacement=True, generator=generator).tolist()
        return iter(indices[self.rank::self.world])

    def mine(self, scores):
        scores = np.asarray(scores)
        difficulty = torch.tensor([1 + min(2., 2 * (1 - s if r['label'] else s)) for r, s in zip(self.rows, scores, strict=True)])
        self.weights = self.base * difficulty
        # Mining changes examples within groups, while preserving language/class/engine masses.
        for key in set(self.keys):
            selected = torch.tensor([k == key for k in self.keys])
            self.weights[selected] *= self.base[selected].sum() / self.weights[selected].sum()


def confusion(labels, scores, threshold=0.5):
    labels, scores = np.asarray(labels), np.asarray(scores)
    predictions = scores >= threshold
    positive, negative = labels == 1, labels == 0
    tp, fp = int(np.sum(predictions & positive)), int(np.sum(predictions & negative))
    fn, tn = int(np.sum(~predictions & positive)), int(np.sum(~predictions & negative))
    recall = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'positive_windows': tp + fn, 'negative_windows': tn + fp,
            'recall': recall, 'precision': precision, 'false_positive_rate_per_window': fp / max(1, fp + tn),
            'false_negative_rate': fn / max(1, tp + fn), 'f1': 2 * precision * recall / max(1e-12, precision + recall)}


@torch.inference_mode()
def evaluate(model, loader, device, amp=False):
    model.eval()
    labels, scores, logits = [], [], []
    loss_sum = 0
    for batch in loader:
        feature, label = batch[:2]
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == 'cuda'):
            output = model(feature.to(device, non_blocking=True))
        output = output.float()
        loss_sum += float(nn.functional.binary_cross_entropy_with_logits(output, label.to(device), reduction='sum'))
        logits.extend(output.cpu().tolist())
        scores.extend(output.sigmoid().cpu().tolist())
        labels.extend(label.tolist())
    return {'labels': labels, 'scores': scores, 'logits': logits, 'loss': loss_sum / max(1, len(labels))}


def merge_evaluations(shards, size):
    """Restore manifest order and reject padded, missing or duplicated evaluation rows."""
    records, loss_sum = {}, 0.
    for shard in shards:
        prediction, indices = shard['prediction'], shard['indices']
        loss_sum += prediction['loss'] * len(indices)
        for index, label, score, logit in zip(indices, prediction['labels'], prediction['scores'], prediction['logits'], strict=True):
            if index in records or not 0 <= index < size:
                raise ValueError('Evaluation shards contain duplicate/out-of-range rows')
            records[index] = (label, score, logit)
    if len(records) != size:
        raise ValueError('Evaluation shards do not cover the dataset')
    return dict(labels=[records[i][0] for i in range(size)], scores=[records[i][1] for i in range(size)],
                logits=[records[i][2] for i in range(size)], loss=loss_sum / max(1, size))


def evaluate_shared(model, dataset, batch_size, device, amp=False):
    """Every rank evaluates an unpadded shard using its raw model, then gathers CPU results."""
    distributed = dist.is_available() and dist.is_initialized()
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    indices = list(range(rank, len(dataset), world))
    prediction = evaluate(model, DataLoader(Subset(dataset, indices), batch_size=batch_size, num_workers=0,
                                           pin_memory=device.type == 'cuda'), device, amp=amp)
    shard = dict(indices=indices, prediction=prediction)
    shards = [shard]
    if distributed:
        shards = [None] * world
        dist.all_gather_object(shards, shard)
    return merge_evaluations(shards, len(dataset))


def save_torch(file, value):
    temporary = file.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(file)


def train(config, member=0):
    config = dict(config)
    config.update(config.get('member_overrides', {}).get(str(member), {}))
    rank, local_rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('LOCAL_RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    cuda = torch.cuda.is_available() and not config.get('force_cpu')
    device = torch.device(f'cuda:{local_rank}' if cuda else 'cpu')
    distributed = world > 1
    if cuda: torch.cuda.set_device(local_rank)
    if distributed:
        dist.init_process_group('nccl' if cuda else 'gloo', timeout=timedelta(seconds=config.get('ddp_timeout_seconds', 1800)))
    try:
        torch.set_num_threads(config.get('cpu_threads', 2))
        seed = config['seed'] + member * config.get('member_seed_stride', 1009)
        torch.manual_seed(seed + rank)
        random.seed(seed + rank)
        np.random.seed(seed + rank)
        if cuda: torch.cuda.manual_seed_all(seed + rank)
        torch.backends.cudnn.benchmark = False
        work = Path(config['work_dir'])
        multiple = config.get('ensemble_members', 1) > 1
        tag = f'-member-{member}' if multiple else ''
        checkpoints = work / 'checkpoints' / f'member-{member}' if multiple else work / 'checkpoints'
        checkpoints.mkdir(parents=True, exist_ok=True)
        dataset_root = Path(config.get('dataset_dir') or work / 'dataset')
        dataset = FeatureDataset(dataset_root, 'train', config.get('preload_features', False))
        if config.get('distill_weight', 0):
            from wake_distill import DistillationDataset
            dataset = DistillationDataset(dataset, config['teacher_targets'])
        validation = FeatureDataset(dataset_root, 'validation', config.get('preload_features', False))
        # Padding at the sampler is intentional; every rank executes the same number of steps.
        sampler = BalancedSampler(dataset.rows, seed, rank, world) if config.get('balanced_sampling') else (
            DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=seed) if distributed else None)
        generator = torch.Generator().manual_seed(seed + rank)
        loader = DataLoader(dataset, batch_size=config['batch_per_gpu'], sampler=sampler, shuffle=sampler is None,
                            num_workers=config['loader_workers'], pin_memory=cuda, drop_last=False, generator=generator)
        model = make_model(config).to(device)
        if distributed: model = DDP(model, device_ids=[local_rank] if cuda else None, broadcast_buffers=False)
        raw_model = model.module if distributed else model
        ema = copy.deepcopy(raw_model).eval().requires_grad_(False) if config.get('ema', False) else None
        ema_steps = 0
        if config.get('architecture') == 'whisper_encoder_v4':
            backbone = [p for name, p in model.named_parameters() if 'encoder.' in name and p.requires_grad]
            head = [p for name, p in model.named_parameters() if 'encoder.' not in name and p.requires_grad]
            optimizer = torch.optim.AdamW([dict(params=backbone, lr=config.get('encoder_learning_rate', 1e-5)),
                                           dict(params=head, lr=config['learning_rate'])], weight_decay=1e-3)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
        amp = cuda and config.get('mixed_precision', True)
        scaler = torch.amp.GradScaler('cuda', enabled=amp)
        positive = sum(row['label'] == 1 for row in dataset.rows)
        negative = len(dataset) - positive
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1. if config.get('balanced_sampling') else negative / positive, device=device))
        manifest_hash = hashlib.sha256((dataset_root / 'manifest.jsonl').read_bytes()).hexdigest()
        recipe_keys = ['seed', 'epochs', 'batch_per_gpu', 'learning_rate', 'architecture', 'balanced_sampling', 'ema', 'mining_interval', 'target_fpr', 'encoder_learning_rate', 'encoder_train_layers']
        recipe_keys += [key for key in ['encoder_dir', 'distill_weight', 'distill_temperature', 'member_seed_stride', 'source_groups', 'source_group_fpr_constraint'] if key in config]
        if config.get('distill_weight', 0):
            config['teacher_targets_sha256'] = hashlib.sha256(Path(config['teacher_targets']).read_bytes()).hexdigest()
            recipe_keys.append('teacher_targets_sha256')
        training_recipe_hash = hashlib.sha256(json.dumps({k: config.get(k) for k in recipe_keys}, sort_keys=True).encode()).hexdigest()
        start_epoch, best_loss, history, best_score, stale = 0, float('inf'), [], -float('inf'), 0
        resume = config.get('resume_checkpoint')
        if resume and multiple: resume = str(Path(resume) / f'member-{member}' / 'latest.pt')
        if resume:
            saved = torch.load(resume, map_location='cpu', weights_only=False)
            if saved['manifest_sha256'] != manifest_hash or saved['world_size'] != world:
                raise ValueError('Resume requires the same dataset and number of processes')
            if saved.get('training_recipe_sha256', training_recipe_hash) != training_recipe_hash:
                raise ValueError('Resume requires the same training recipe, including total epochs')
            raw_model.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler'])
            scaler.load_state_dict(saved['scaler'])
            start_epoch, best_loss, history = saved['epoch'] + 1, saved['best_loss'], saved['history']
            if saved.get('architecture', 'legacy') != config.get('architecture', 'legacy') or saved.get('member', 0) != member:
                raise ValueError('Resume requires the same architecture and ensemble member')
            best_score, stale = saved.get('best_score', -float('inf')), saved.get('stale', 0)
            if ema is not None:
                ema.load_state_dict(saved['ema'])
                ema_steps = saved['ema_steps']
            if isinstance(sampler, BalancedSampler) and saved.get('sampler_weights') is not None:
                sampler.weights = saved['sampler_weights']
            state = torch.load(Path(resume).with_name(f'latest-rng-{rank}.pt'), weights_only=False, map_location='cpu')
            torch.set_rng_state(state['torch'])
            generator.set_state(state['loader'])
            if cuda: torch.cuda.set_rng_state(state['cuda'], device=device)
            # best weights are embedded so a copied latest.pt can restore them in the new output directory.
            if rank == 0: save_torch(checkpoints / 'best.pt', saved['best'])
        if rank == 0:
            print(f'Training: {world} processes, {device.type}, effective batch {config["batch_per_gpu"] * world}, '
                  f'{sum(p.numel() for p in raw_model.parameters()):,} parameters, {len(dataset)} train windows', flush=True)
        best = saved['best'] if resume else None
        end_epoch = min(config['epochs'], config.get('stop_after_epoch') or config['epochs'])
        for epoch in range(start_epoch, end_epoch):
            started = time.monotonic()
            if sampler is not None: sampler.set_epoch(epoch)
            model.train()
            totals = torch.zeros(2, dtype=torch.float64, device=device)
            for batch in loader:
                feature, label = batch[:2]
                feature = spec_augment(feature.to(device, non_blocking=True))
                label = label.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    output = model(feature)
                    loss = criterion(output, label)
                    if config.get('distill_weight', 0):
                        from wake_distill import distillation_loss
                        teacher_logit = batch[2].to(device, non_blocking=True)
                        loss = loss + config['distill_weight'] * distillation_loss(
                            output, teacher_logit, label, config.get('distill_temperature', 2.))
                if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                scaler.step(optimizer)
                scaler.update()
                if ema is not None:
                    ema_steps += 1
                    decay = min(.995, (1 + ema_steps) / (10 + ema_steps))
                    with torch.no_grad():
                        for averaged, parameter in zip(ema.parameters(), raw_model.parameters(), strict=True):
                            averaged.lerp_(parameter.detach(), 1 - decay)
                totals[0] += loss.detach().double() * len(label)
                totals[1] += len(label)
            scheduler.step()
            if distributed: dist.all_reduce(totals)
            if rank == 0: print(f'Epoch {epoch + 1}: validation on {world} rank(s)', flush=True)
            validation_started = time.monotonic()
            result = evaluate_shared(raw_model, validation, config['batch_per_gpu'] * 2, device, amp=amp)
            averaged_result = evaluate_shared(ema, validation, config['batch_per_gpu'] * 2, device, amp=amp) if ema is not None else None
            if rank == 0:
                chosen, source = raw_model, 'raw'
                point = point_for_config(result, validation.rows, config) if config.get('balanced_sampling') else None
                selection = selection_score(point, result['loss'], config) if point else -result['loss']
                if ema is not None:
                    averaged_point = point_for_config(averaged_result, validation.rows, config)
                    averaged_selection = selection_score(averaged_point, averaged_result['loss'], config)
                    if averaged_selection > selection:
                        result, point, selection, chosen, source = averaged_result, averaged_point, averaged_selection, ema, 'ema'
                row = {'epoch': epoch + 1, 'train_loss': float(totals[0] / totals[1]),
                       'validation_loss': result['loss'], 'seconds': time.monotonic() - started,
                       'validation_seconds': time.monotonic() - validation_started, 'evaluation_world_size': world,
                       **confusion(result['labels'], result['scores'])}
                if point: row.update(validation_operating_point=point, selection_score=selection, weights_source=source)
                history.append(row)
                improved = selection > best_score + 1e-5
                stale = 0 if improved else stale + 1
                if improved:
                    best_loss = result['loss']
                    best_score = selection
                    best = {'model': {key: value.detach().cpu().clone() for key, value in chosen.state_dict().items()},
                            'epoch': epoch + 1, 'validation_loss': best_loss, 'selection_score': best_score,
                            'architecture': config.get('architecture', 'legacy'), 'member': member, 'weights_source': source}
                    save_torch(checkpoints / 'best.pt', best)
            # Mine training examples only. All ranks receive the same new sampling stream.
            if isinstance(sampler, BalancedSampler) and (epoch + 1) % config.get('mining_interval', 10) == 0:
                if rank == 0: print(f'Epoch {epoch + 1}: train-only mining on {world} rank(s)', flush=True)
                mined = evaluate_shared(raw_model, dataset, config['batch_per_gpu'] * 2, device, amp=amp)
                sampler.mine(mined['scores'])
            if rank == 0 and config.get('persist_latest', True):
                save_torch(checkpoints / 'latest.pt', {'model': raw_model.state_dict(), 'optimizer': optimizer.state_dict(),
                           'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'epoch': epoch,
                           'best_loss': best_loss, 'best_score': best_score, 'stale': stale, 'best': best, 'history': history,
                           'ema': ema.state_dict() if ema is not None else None, 'ema_steps': ema_steps,
                           'sampler_weights': sampler.weights if isinstance(sampler, BalancedSampler) else None,
                           'architecture': config.get('architecture', 'legacy'), 'member': member,
                           'training_recipe_sha256': training_recipe_hash,
                           'manifest_sha256': manifest_hash, 'world_size': world})
            if rank == 0:
                atomic_json(work / f'training_history{tag}.json', history)
                print(f'Epoch {epoch + 1}/{config["epochs"]}: train {row["train_loss"]:.4f}, '
                      f'validation {row["validation_loss"]:.4f}, recall {row["recall"]:.3f}, '
                      f'FP/window {row["false_positive_rate_per_window"]:.4f}', flush=True)
                if point: print(f'  Validation at target FPR: macro recall {point["macro_recall"]:.3f}, worst language {point["worst_language_recall"]:.3f}, weights {source}', flush=True)
            if config.get('persist_latest', True):
                save_torch(checkpoints / f'latest-rng-{rank}.pt', {'torch': torch.get_rng_state(), 'loader': generator.get_state(),
                           'cuda': torch.cuda.get_rng_state(device) if cuda else None})
            if distributed: dist.barrier()
            stop = torch.tensor(int(rank == 0 and stale >= config.get('patience', config['epochs'] + 1)
                                    and epoch + 1 >= config.get('min_epochs', 15)), device=device)
            if distributed: dist.broadcast(stop, src=0)
            if stop.item():
                if rank == 0: print('Early stop: no validation operating-point improvement', flush=True)
                break
        if not (checkpoints / 'best.pt').exists(): raise RuntimeError('No trained checkpoint to export')
        if rank == 0:
            run = {'world_size': world, 'device': str(device), 'amp': amp,
                        'manifest_sha256': manifest_hash, 'pytorch': torch.__version__,
                        'epochs_completed': len(history), 'smoke_only': config.get('smoke', False),
                        'architecture': config.get('architecture', 'legacy'), 'member': member,
                        'ensemble_members': config.get('ensemble_members', 1),
                        'evaluation_world_size': world, 'ddp_timeout_seconds': config.get('ddp_timeout_seconds', 1800),
                        'distill_weight': config.get('distill_weight', 0), 'encoder_dir': config.get('encoder_dir')}
            atomic_json(work / f'training_run{tag}.json', run)
            atomic_json(work / 'training_run.json', run)
    finally:
        if distributed and dist.is_initialized(): dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--member', type=int, default=0)
    args = parser.parse_args()
    train(json.loads(Path(args.config).read_text(encoding='utf-8')), args.member)
