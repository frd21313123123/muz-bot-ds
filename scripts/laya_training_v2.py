"""Balanced, task-separated Laya fine tuning; imports no GPU packages at import time."""
from collections import Counter, defaultdict
import gc
import json
import math
from pathlib import Path
import random
import laya_training as base


def encode(agent, records, groups):
    items = base.encode_records(agent, records)
    for item in items:
        item["group_id"] = groups[item["event_id"]]
    return items


def balanced_indices(items, draws, seed):
    """Equal task mass, then equal class mass, then equal phrase-family mass."""
    pools = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for index, item in enumerate(items):
        pools[item["task"] + "/" + item["qid"]][item["label"]][item["group_id"]].append(index)
    tasks = sorted(pools)
    if not tasks or draws < len(tasks):
        raise ValueError("Not enough balanced samples")
    rng, order = random.Random(seed), []
    # Each task has a separately shuffled class cycle: support is never dominated by unknown.
    cycles = {task: [] for task in tasks}
    for step in range(draws):
        task = tasks[step % len(tasks)]
        if not cycles[task]:
            cycles[task] = sorted(pools[task])
            rng.shuffle(cycles[task])
        label = cycles[task].pop()
        group = rng.choice(sorted(pools[task][label]))
        order.append(rng.choice(pools[task][label][group]))
    rng.shuffle(order)
    return order


def selection_score(metrics):
    """A frequent unknown class cannot hide a completely failed positive class."""
    recalls = [c["recall"] for m in metrics.values() for c in m["per_class"].values() if c["support"]]
    return (min(recalls), sum(m["macro_f1"] for m in metrics.values()) / len(metrics),
            -sum(m["nll"] for m in metrics.values()) / len(metrics))


def configure_trainable(agent, last_layers):
    encoder = agent.model.encoder
    layers = getattr(encoder, "layers", None)
    if layers is None or not 0 <= last_layers <= len(layers):
        raise ValueError("Expected ModernBERT encoder.layers for partial fine tuning")
    for name, parameter in agent.model.named_parameters():
        parameter.requires_grad_(not name.startswith(("encoder.", "act_head.")))
    for layer in list(layers)[len(layers) - last_layers:] if last_layers else []:
        for parameter in layer.parameters():
            parameter.requires_grad_(True)
    if last_layers:
        encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    agent.model.head_checkpointing = True
    return layers


def train(agent, train_items, validation_items, directory, model_name, epochs=6, draws=3072,
          batch_size=2, accumulation=8, head_lr=5e-5, encoder_lr=3e-6, last_layers=2, seed=42):
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    layers = configure_trainable(agent, last_layers)
    head = [p for n, p in agent.model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    encoder = [p for n, p in agent.model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    groups = [{"params": head, "lr": head_lr}]
    if encoder:
        groups.append({"params": encoder, "lr": encoder_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=agent.device.type == "cuda" and agent.dtype == torch.float16)
    micro_batches = math.ceil(draws / batch_size)
    total_updates = epochs * math.ceil(micro_batches / accumulation)
    warmup = max(1, int(0.1 * total_updates))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / warmup,
        max(0.0, (total_updates - step) / max(1, total_updates - warmup))))
    baseline, _ = base.evaluate_logits(base.raw_logits(agent, validation_items, batch_size))
    best = selection_score(baseline)
    base.save_checkpoint(agent, directory, model_name)
    history, audits = [], []
    torch.manual_seed(seed)
    for epoch in range(epochs):
        agent.model.train()
        agent.model.encoder.eval()
        for layer in list(layers)[len(layers) - last_layers:] if last_layers else []:
            layer.train()
        order = balanced_indices(train_items, draws, seed + epoch)
        audits.append(dict(Counter(f'{train_items[i]["task"]}/{train_items[i]["qid"]}:{train_items[i]["labels"][train_items[i]["label"]]}' for i in order)))
        optimizer.zero_grad(set_to_none=True)
        total_loss, count = 0.0, 0
        selected_items = [train_items[i] for i in order]
        for step, (selected, batch) in enumerate(base.batches(selected_items, batch_size)):
            window_start = (step // accumulation) * accumulation * batch_size
            window_size = min(accumulation * batch_size, draws - window_start)
            with base.amp_context(agent):
                logits, _ = agent.model(**{k: batch[k].to(agent.device) for k in base.MODEL_KEYS})
                loss = F.cross_entropy(logits.float(), batch["label"].to(agent.device), reduction="sum")
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; candidate is not exportable")
            scaler.scale(loss / window_size).backward()
            if last_layers and step == 0 and not any(p.grad is not None for p in encoder):
                raise RuntimeError("Partial encoder received no gradients; refuse silent head-only training")
            total_loss += loss.item(); count += len(selected)
            if (step + 1) % accumulation == 0 or step + 1 == micro_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in agent.model.parameters() if p.requires_grad], 1.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        metrics, _ = base.evaluate_logits(base.raw_logits(agent, validation_items, batch_size))
        score = selection_score(metrics)
        history.append({"epoch": epoch + 1, "balanced_train_nll": total_loss / count,
                        "validation_score": list(score), "validation": metrics})
        print({"epoch": epoch + 1, "balanced_train_nll": total_loss / count, "min_recall": score[0], "macro_f1": score[1]}, flush=True)
        if score > best:
            best = score
            base.save_checkpoint(agent, directory, model_name)
    agent.model.load_state_dict(load_file(str(Path(directory) / "model.safetensors")), strict=True)
    agent.model.eval()
    return {"history": history, "baseline_validation": baseline, "selected_score": list(best), "sampling_audit": audits,
            "sampling": "equal task, class, family; random variant", "last_encoder_layers": last_layers,
            "trainable_encoder_parameters": sum(p.numel() for p in encoder),
            "trainable_head_parameters": sum(p.numel() for p in head)}


def calibrate(rows):
    """Fit option-count buckets only on calibration; balance class/family mass."""
    import torch
    import torch.nn.functional as F
    from laya.common import temp_bucket
    buckets = defaultdict(list)
    for row in rows:
        item = row["item"]
        buckets[temp_bucket(item["qtype"], len(item["labels"]))].append(row)
    temperatures = {"temperature": [1.0, 1.0, 1.0], "temperature_by_options": {}}
    for key, records in buckets.items():
        if len(records) < 40:
            continue
        counts = Counter((r["item"]["task"], r["item"]["label"], r["item"]["group_id"]) for r in records)
        families = defaultdict(set)
        for task, label, group in counts:
            families[task, label].add(group)
        logits = torch.tensor([r["logits"] for r in records], dtype=torch.float32)
        labels = torch.tensor([r["item"]["label"] for r in records])
        weights = torch.tensor([1 / (counts[(r["item"]["task"], r["item"]["label"], r["item"]["group_id"])]
                    * len(families[r["item"]["task"], r["item"]["label"]])) for r in records])
        weights /= weights.sum()
        candidates = [0.5 * (10 ** (i / 80)) for i in range(81)]  # 0.5..5, deterministic one-parameter fit.
        losses = [float((F.cross_entropy(logits / t, labels, reduction="none") * weights).sum()) for t in candidates]
        temperatures["temperature_by_options"][key] = candidates[min(range(len(losses)), key=losses.__getitem__)]
    return temperatures


def install_temperatures(agent, temperatures):
    import torch
    agent.temperature = list(temperatures["temperature"])
    agent.temperature_by_options = dict(temperatures["temperature_by_options"])
    agent.cfg.update(temperatures)
    agent.model.temperature.copy_(torch.tensor(agent.temperature, device=agent.device, dtype=agent.model.temperature.dtype))


def metric_gate(metrics, minimum_recall=0.8, minimum_f1=0.8):
    failures = []
    for task, values in metrics.items():
        if values["macro_f1"] < minimum_f1:
            failures.append(f"{task}: macro-F1 {values['macro_f1']:.3f} < {minimum_f1}")
        for label, value in values["per_class"].items():
            if value["support"] and value["recall"] < minimum_recall:
                failures.append(f"{task}/{label}: recall {value['recall']:.3f} < {minimum_recall}")
    return {"passed": not failures, "failures": failures}


def release(agent):
    import torch
    agent.model.cpu()
    agent.device = torch.device("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
