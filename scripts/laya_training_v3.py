"""Text-only semantics, stratified negatives, measured optimizer updates and epoch resume."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import re
import laya_training as base
import laya_training_v2 as previous

CONTROLS = {"skip", "pause", "resume", "stop", "volume_set", "volume_up", "volume_down"}


def joint_split(rows, seed=20261010):
    """Balance effective texts separately for each task; reserve real-log components for train."""
    parent, owners = list(range(len(rows))), {}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, row in enumerate(rows):
        for key in [("family", row["group_id"])] + [("text", base.normalized(row[k])) for k in ("message", "canonical")]:
            if key in owners: parent[find(i)] = find(owners[key])
            owners[key] = i
    grouped = defaultdict(list)
    for i, row in enumerate(rows): grouped[find(i)].append(row)
    components = []
    for group in grouped.values():
        effective = defaultdict(set)
        for row in group:
            task = "semantic" if row.get("task") == "semantic" else "action"
            label = row["intent"] if task == "semantic" or row["intent"] in CONTROLS | {"unknown"} else "unknown"
            effective[task + "/" + label].add(base.normalized(row["canonical"]))
        components.append((group, Counter({k: len(v) for k, v in effective.items()}), any(r["origin"] == "bot_log" for r in group)))
    totals, family_counts = Counter(), Counter()
    for _, counts, _ in components: totals.update(counts); family_counts.update(counts.keys())
    if any(n < 4 for n in family_counts.values()): raise ValueError("Need four disjoint components per task/class")
    fractions, rng, best, best_cost = (.6, .15, .1, .15), random.Random(seed), None, math.inf
    for attempt in range(500):
        assignments, counts = [[] for _ in fractions], [Counter() for _ in fractions]
        movable = []
        for group, gc, forced in components:
            if forced: assignments[0].extend(group); counts[0].update(gc)
            else: movable.append((group, gc))
        rng.shuffle(movable)
        movable.sort(key=lambda v: min(family_counts[k] for k in v[1]))
        for group, gc in movable:
            def cost(j):
                missing = sum(counts[j][k] < 2 for k in gc)
                fill = sum(counts[j][k] / max(1, totals[k] * fractions[j]) for k in gc)
                return -100 * missing + fill + rng.random() * .01
            j = min(range(4), key=cost)
            assignments[j].extend(group); counts[j].update(gc)
        if any(counts[j][k] < 2 for j in range(4) for k in totals): continue
        score = sum(abs(counts[j][k] / totals[k] - fractions[j]) for j in range(4) for k in totals)
        if score < best_cost: best, best_cost = assignments, score
    if best is None: raise ValueError("Cannot retain effective per-task class coverage; add disjoint phrases")
    return dict(zip(base.SPLITS, best, strict=True))


def runtime_text(text):
    text = text.strip().rstrip('.!…').strip().lower()
    return text[:1].upper() + text[1:]


def load_commands(path, bridge, labels):
    rows = base.read_jsonl([path])
    seen = set()
    for row in rows:
        declared = row.get("origin") == "synthetic" and row.get("label_status") == "template_declared" and row.get("label_source") == "declared_semantics"
        if row.get("reviewed") is not True and not (base.automatic_label_allowed(row, True) or declared):
            raise ValueError("No predictions or unlabelled speech as gold")
        if row["intent"] not in labels or row["event_id"] in seen:
            raise ValueError("Invalid label/duplicate event")
        seen.add(row["event_id"])
        if "phase" in row or "is_owner" in row or "gold" in row or "model_decision" in row:
            raise ValueError("Session policy and model hints do not belong in semantic commands")
    return base.prepare_commands(rows, bridge)


def negative_kind(row):
    intent = row["intent"]
    if intent == "play": return "music"
    if intent != "unknown": return "mode" if intent not in CONTROLS else "positive"
    text = row["message"].lower()
    if re.search(r"(?:^|\s)(?:не|нет|not|never)\s", text): return "negation"
    if "?" in text or re.match(r"(?:как|почему|кто|когда|какая)\b", text): return "question"
    if re.search(r"\b(?:и|или|потом|затем|and|or)\b", text): return "multiple"
    if re.search(r"\d|процент|громче на|тише на|громкость", text): return "invalid_number"
    if re.search(r"свет|камер|яркость|микрофон|телевизор|таймер|погод", text): return "other_device"
    return "other"


def records(rows, worker, semantic_question=None):
    if semantic_question is None:
        return base.runtime_records(rows, worker)
    return [{"event_id": r["event_id"], "task": "semantic", "state": runtime_text(r["canonical"]),
             "questions": semantic_question, "gold": {"intent": r["intent"]}} for r in rows]


def encode(agent, record_rows, sources):
    items = base.encode_records(agent, record_rows)
    for item, record in zip(items, [r for r in record_rows for _ in r["questions"]], strict=True):
        row = sources[item["event_id"]]
        item.update(group_id=row["group_id"], category=negative_kind(row), text_key=runtime_text(row["canonical"]))
    return items


def pools(items):
    result = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))))
    for index, item in enumerate(items):
        label = item["labels"][item["label"]]
        category = item["category"] if label == "unknown" else "positive"
        result[item["task"] + "/" + item["qid"]][item["label"]][category][item["group_id"]][item["text_key"]].append(index)
    return result


def balanced_indices(items, draws, seed):
    tree, rng = pools(items), random.Random(seed)
    tasks = sorted(tree)
    if not tasks or draws < len(tasks): raise ValueError("Not enough balanced samples")
    cycles, order = {task: [] for task in tasks}, []
    for step in range(draws):
        task = tasks[step % len(tasks)]
        if not cycles[task]:
            cycles[task] = sorted(tree[task]); rng.shuffle(cycles[task])
        label = cycles[task].pop()
        categories = tree[task][label]
        families = categories[rng.choice(sorted(categories))]
        texts = families[rng.choice(sorted(families))]
        order.append(rng.choice(texts[rng.choice(sorted(texts))]))
    rng.shuffle(order)
    return order


def selection_score(metrics):
    recalls = [v["recall"] for task in metrics.values() for v in task["per_class"].values() if v["support"]]
    return (sum(task["macro_f1"] for task in metrics.values()) / len(metrics),
            sum(recalls) / len(recalls), min(recalls), -sum(task["nll"] for task in metrics.values()) / len(metrics))


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor): return value.detach().cpu()
    if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    if hasattr(value, "item"): return value.item()
    return value


def train(agent, train_items, validation_items, directory, model_name, state_path, epochs=10, draws=4096,
          batch_size=2, accumulation=8, head_lr=5e-5, encoder_lr=5e-6, last_layers=4, seed=20261010,
          patience=4, signature="", resume=True, _max_new_epochs=None):
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    layers = previous.configure_trainable(agent, last_layers)
    head = [p for n, p in agent.model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    encoder = [p for n, p in agent.model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    parameters = head + encoder
    groups = [{"params": head, "lr": head_lr}] + ([{"params": encoder, "lr": encoder_lr}] if encoder else [])
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", init_scale=512, enabled=agent.device.type == "cuda" and agent.dtype == torch.float16)
    micro_batches = math.ceil(draws / batch_size)
    total_updates = epochs * math.ceil(micro_batches / accumulation)
    warmup = max(1, int(total_updates * .1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / warmup,
        max(0., (total_updates - step) / max(1, total_updates - warmup))))
    config = {"signature": signature, "epochs": epochs, "draws": draws, "batch": batch_size, "accumulation": accumulation,
              "head_lr": head_lr, "encoder_lr": encoder_lr, "layers": last_layers, "seed": seed}
    directory, state_path = Path(directory), Path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = state_path.with_suffix(".json")
    history, audits, start_epoch, no_improvement = [], [], 0, 0
    if resume and state_path.exists():
        saved = torch.load(state_path, map_location="cpu", weights_only=True)
        if saved["config"] != config: raise ValueError("Resume config/data differ; use a new WORKDIR or original parameters")
        if not (directory / "model.safetensors").exists(): raise FileNotFoundError("Resume also needs the saved best candidate folder")
        agent.model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"]); scheduler.load_state_dict(saved["scheduler"]); scaler.load_state_dict(saved["scaler"])
        baseline, best, history, audits = saved["baseline"], tuple(saved["best"]), saved["history"], saved["audits"]
        start_epoch, no_improvement = saved["epoch"], saved["no_improvement"]
        torch.set_rng_state(saved["rng_cpu"])
        if agent.device.type == "cuda": torch.cuda.set_rng_state(saved["rng_cuda"], device=0)
        del saved
        print("RESUME at completed epoch", start_epoch, flush=True)
    else:
        baseline, _ = base.evaluate_logits(base.raw_logits(agent, validation_items, batch_size))
        best = selection_score(baseline)
        base.save_checkpoint(agent, directory, model_name)
        torch.manual_seed(seed)
    stop_epoch = epochs if _max_new_epochs is None else min(epochs, start_epoch + _max_new_epochs)
    for epoch in range(start_epoch, stop_epoch):
        agent.model.train(); agent.model.encoder.eval()
        for layer in list(layers)[len(layers) - last_layers:] if last_layers else []: layer.train()
        order = balanced_indices(train_items, draws, seed + epoch)
        audits.append(dict(Counter(f'{train_items[i]["task"]}:{train_items[i]["labels"][train_items[i]["label"]]}:{train_items[i]["category"]}' for i in order)))
        optimizer.zero_grad(set_to_none=True)
        loss_sum, samples, updates, skipped = 0., 0, 0, 0
        selected_items = [train_items[i] for i in order]
        for step, (selected, batch) in enumerate(base.batches(selected_items, batch_size)):
            window_start = (step // accumulation) * accumulation * batch_size
            window_size = min(accumulation * batch_size, draws - window_start)
            with base.amp_context(agent):
                logits, _ = agent.model(**{k: batch[k].to(agent.device) for k in base.MODEL_KEYS})
                loss = F.cross_entropy(logits.float(), batch["label"].to(agent.device), reduction="sum")
            if not torch.isfinite(loss): raise RuntimeError("Non-finite loss: stop and preserve last completed epoch")
            scaler.scale(loss / window_size).backward()
            if step == 0 and encoder and not any(p.grad is not None for p in encoder): raise RuntimeError("Encoder received no gradients")
            loss_sum += loss.item(); samples += len(selected)
            if (step + 1) % accumulation == 0 or step + 1 == micro_batches:
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(parameters, 1.)
                old_scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
                if scaler.get_scale() >= old_scale: scheduler.step(); updates += 1
                else: skipped += 1
                optimizer.zero_grad(set_to_none=True)
        if not updates: raise RuntimeError("No optimizer updates in epoch; inspect AMP. Last checkpoint preserved")
        metrics, _ = base.evaluate_logits(base.raw_logits(agent, validation_items, batch_size))
        score = selection_score(metrics)
        history.append({"epoch": epoch + 1, "balanced_train_nll": loss_sum / samples, "validation_score": list(score), "validation": metrics,
                        "optimizer_updates": updates, "skipped_updates": skipped, "amp_scale": scaler.get_scale()})
        if score > best:
            best = score; no_improvement = 0
            base.save_checkpoint(agent, directory, model_name)
        else: no_improvement += 1
        payload = {"config": config, "model": cpu_tree(agent.model.state_dict()), "optimizer": cpu_tree(optimizer.state_dict()),
                   "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "baseline": baseline, "best": list(best),
                   "history": cpu_tree(history), "audits": audits, "epoch": epoch + 1, "no_improvement": no_improvement,
                   "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state(0) if agent.device.type == "cuda" else None}
        temporary = state_path.with_suffix(".pt.tmp")
        torch.save(cpu_tree(payload), temporary); temporary.replace(state_path)
        del payload
        atomic_json(progress_path, {"completed_epoch": epoch + 1, "history": history, "best_score": list(best),
                    "config": config, "qualification": "not evaluated; completed training progress only"})
        print({"epoch": epoch + 1, "macro_f1": score[0], "min_recall": score[2], "updates": updates, "skipped": skipped,
               "saved_resume": str(state_path)}, flush=True)
        if epoch + 1 >= 4 and no_improvement >= patience: break
    agent.model.load_state_dict(load_file(str(directory / "model.safetensors")), strict=True); agent.model.eval()
    return {"history": history, "baseline_validation": baseline, "selected_score": list(best), "sampling_audit": audits,
            "last_encoder_layers": last_layers, "sampling": "equal task/class; equal negative subtype; equal family/text/variant",
            "trainable_encoder_parameters": sum(p.numel() for p in encoder), "trainable_head_parameters": sum(p.numel() for p in head),
            "optimizer_updates": sum(r["optimizer_updates"] for r in history), "skipped_updates": sum(r["skipped_updates"] for r in history),
            "resumed_from_epoch": start_epoch}


def calibrate(rows):
    import torch
    import torch.nn.functional as F
    from laya.common import temp_bucket
    buckets = defaultdict(list)
    for row in rows: buckets[temp_bucket(row["item"]["qtype"], len(row["item"]["labels"]))].append(row)
    result = {"temperature": [1., 1., 1.], "temperature_by_options": {}}
    for bucket, records in buckets.items():
        if len(records) < 40: continue
        items, weights = [r["item"] for r in records], [0.] * len(records)
        tree = pools(items)
        for classes in tree.values():
            for categories in classes.values():
                for families in categories.values():
                    for texts in families.values():
                        for variants in texts.values():
                            for i in variants: weights[i] = 1 / (len(tree) * len(classes) * len(categories) * len(families) * len(texts) * len(variants))
        logits = torch.tensor([r["logits"] for r in records], dtype=torch.float32)
        labels = torch.tensor([r["item"]["label"] for r in records]); weight = torch.tensor(weights)
        candidates = [0.5 * 10 ** (i / 80) for i in range(81)]
        losses = [float((F.cross_entropy(logits / t, labels, reduction="none") * weight).sum()) for t in candidates]
        result["temperature_by_options"][bucket] = candidates[min(range(81), key=losses.__getitem__)]
    return result


def policy_decision(state, predict_intent, probe):
    """Policy never invokes the semantic model in a forbidden session or for wake."""
    if any(type(state.get(k)) is not bool for k in ("voice_enabled", "in_bot_channel", "is_owner")):
        raise ValueError("Explicit session flags required")
    phase = state["phase"]
    if not state["voice_enabled"] or not state["in_bot_channel"] or phase not in {"idle", "awaiting"}:
        return {"should_respond": "ignore", "next_tool": "no_tool", "intent": "unknown"}
    if phase == "idle":
        return {"should_respond": "respond" if probe["wake_match"] else "ignore",
                "next_tool": "wake_ack" if probe["wake_match"] else "no_tool", "intent": "unknown"}
    if not state["is_owner"]:
        return {"should_respond": "ignore", "next_tool": "no_tool", "intent": "unknown"}
    intent = predict_intent(runtime_text(probe["canonical"]))
    if intent not in CONTROLS | {"play", "queue_clear", "loop_on", "loop_off", "autoplay_on", "autoplay_off", "voice_on", "voice_off", "unknown"}:
        intent = "unknown"
    if intent == "play": tool = "direct_youtube_video" if probe["music_kind"] == "video" else "youtube_music_search"
    elif intent == "unknown": tool = "no_tool"
    else: tool = "player_control"
    return {"should_respond": "respond", "next_tool": tool, "intent": intent}
