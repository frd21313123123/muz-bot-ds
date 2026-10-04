"""Action-only paired prompts, explicit synthetic provenance, and quality-aware selection."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import re
import tempfile
import laya_training as base
import laya_training_v2 as legacy
import laya_training_v3 as v3

LABELS = {"pause", "resume", "skip", "stop", "volume_up", "volume_down", "volume_set", "unknown"}
SELECTION = "validation: class recall and prompt macro-F1 gate coverage at .80, worst recall, worst prompt macro-F1, mean macro-F1, negative NLL"


def category_of(row):
    if row["intent"] != "unknown": return "positive"
    ordinary = v3.negative_kind(row)
    if ordinary in {"negation", "question", "multiple"}: return ordinary
    if row.get("historical_semantic_intent") == "play" or re.match(r"(?:включи|поставь|найди) (?:песню|трек|альбом|композицию|музыку)\b", row["message"].lower()): return "music"
    if row.get("direct_action") not in LABELS | {None} or row.get("historical_semantic_intent") not in LABELS | {None, "play"}: return "mode"
    return ordinary


def load_commands(path, bridge):
    rows, seen = base.read_jsonl([path]), set()
    for row in rows:
        authored = (row.get("origin") == "synthetic" and row.get("label_status") == "assistant_authored"
                    and row.get("label_source") == "codex_explicit_declaration" and row.get("source_role") == "authored")
        historical = row.get("source_role") == "legacy_replay" and base.automatic_label_allowed(row, True)
        if not (authored or historical or row.get("reviewed") is True): raise ValueError("Missing explicit declaration/approved historical label")
        if row["intent"] not in LABELS or row["event_id"] in seen: raise ValueError("Invalid label/duplicate ID")
        if any(k in row for k in ("gold", "phase", "is_owner", "model_decision", "suggested_intent")): raise ValueError("Model predictions/session hints cannot be labels or inputs")
        if row["declared_split"] not in base.SPLITS: raise ValueError("Invalid split")
        if row.get("origin") == "bot_log" and row["declared_split"] != "train": raise ValueError("Automatic logs are train-only")
        seen.add(row["event_id"])
    return base.prepare_commands(rows, bridge)


def encode(agent, records, sources):
    items = base.encode_records(agent, records)
    for item in items:
        row = sources[item["event_id"]]
        item.update(group_id=row["group_id"], text_key=v3.runtime_text(row["canonical"]),
                    category=row["category"], source_role=row["source_role"])
    return items


def paired_indices(items, draws, seed, replay_fraction=.20):
    """Each draw pair is one text with BOTH production prompts, same target.

    Exact class balance; deterministic bounded replay per class; family/text
    cycles prevent a large template family or duplicated event dominating.
    draws counts encoded question inputs, not unique commands.
    """
    if draws % 16 or not 0 <= replay_fraction <= .5: raise ValueError("Use draws divisible by 16 and replay fraction 0..0.5")
    by_event = defaultdict(dict)
    for i, item in enumerate(items): by_event[item["event_id"]][item["task"]] = i
    tree = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))))
    for pair in by_event.values():
        if set(pair) != {"action_primary", "action_short"}: raise ValueError("Both prompts are required for every event")
        first, second = pair["action_primary"], pair["action_short"]
        a, b = items[first], items[second]
        if a["labels"] != b["labels"] or a["label"] != b["label"]: raise ValueError("Paired option order/target mismatch")
        label = a["labels"][a["label"]]
        tree[label][a["source_role"]][a["category"]][a["group_id"]][a["text_key"]].append((first, second))
    if set(tree) != LABELS: raise ValueError("Train requires all eight classes")
    rng, cycles, pairs, counts = random.Random(seed), {}, [], Counter()
    def cycling(key, choices):
        if not cycles.get(key):
            cycles[key] = sorted(choices); rng.shuffle(cycles[key])
        return cycles[key].pop()
    for _ in range(draws // 16):
        labels = sorted(tree); rng.shuffle(labels)
        for label in labels:
            counts[label] += 1
            replay_now = math.floor(counts[label] * replay_fraction) > math.floor((counts[label] - 1) * replay_fraction)
            role = "legacy_replay" if replay_now and "legacy_replay" in tree[label] else "authored"
            if role not in tree[label]: raise ValueError("Every train class needs assistant-authored phrases")
            categories = tree[label][role]
            category = cycling((label, role), categories)
            families = categories[category]
            family = cycling((label, role, category), families)
            texts = families[family]
            text = cycling((label, role, category, family), texts)
            pairs.append(rng.choice(texts[text]))
    rng.shuffle(pairs)
    return [index for pair in pairs for index in pair]


def selection_score(metrics):
    recalls = [c["recall"] for m in metrics.values() for c in m["per_class"].values() if c["support"]]
    f1 = [m["macro_f1"] for m in metrics.values()]
    return ((sum(r >= .80 for r in recalls) + sum(f >= .80 for f in f1)) / (len(recalls) + len(f1)), min(recalls), min(f1), sum(f1) / len(f1),
            -sum(m["nll"] for m in metrics.values()) / len(metrics))


def paired_loss(logits, labels, consistency_weight=.15):
    import torch
    import torch.nn.functional as F
    if len(labels) % 2 or not torch.equal(labels[::2], labels[1::2]): raise ValueError("Batch must contain adjacent equal-target prompt pairs")
    z = logits.float()[:, :8]
    ce = F.cross_entropy(z, labels, reduction="mean")
    logs = F.log_softmax(z, dim=-1).reshape(-1, 2, 8)
    mixture = torch.logaddexp(logs[:, 0], logs[:, 1]) - math.log(2.)
    js = (logs.exp() * (logs - mixture[:, None, :])).sum(-1).mean().clamp_min(0.)
    return ce + consistency_weight * js, ce, js


def atomic_checkpoint(agent, directory, model_name, temperatures=None):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v4-checkpoint-", dir=directory.parent) as name:
        temporary = Path(name)
        temporary.resolve().relative_to(directory.parent.resolve())
        base.save_checkpoint(agent, temporary, model_name, temperatures)
        files = sorted((p for p in temporary.rglob("*") if p.is_file()), key=lambda p: p.name == "model.safetensors")
        for p in files:
            target = directory / p.relative_to(temporary)
            target.parent.mkdir(parents=True, exist_ok=True)
            p.replace(target)


def training_config(agent, signature, epochs, draws, batch_size, accumulation, head_lr, encoder_lr,
                    last_layers, seed, consistency_weight, replay_fraction, patience):
    return {"signature": signature, "epochs": epochs, "draws": draws, "batch": batch_size, "accumulation": accumulation,
        "head_lr": head_lr, "encoder_lr": encoder_lr, "layers": last_layers, "seed": seed,
        "consistency_weight": consistency_weight, "replay_fraction": replay_fraction, "selection": SELECTION,
        "dtype": str(agent.dtype), "patience": patience}


def info_from_progress(progress, resumed_from_epoch=0):
    history = progress["history"]
    return {"history": history, "baseline_validation": progress["baseline"], "selected_score": progress["best_score"],
        "selected_epoch": progress["selected_epoch"], "sampling_audit": progress["sampling_audit"],
        "last_encoder_layers": progress["config"]["layers"],
        "sampling": "equal class; paired primary+short; authored 80%, historical replay at most 20%; category/family/text cycles",
        "trainable_encoder_parameters": progress["trainable_encoder_parameters"], "trainable_head_parameters": progress["trainable_head_parameters"],
        "optimizer_updates": sum(r["optimizer_updates"] for r in history), "skipped_updates": sum(r["skipped_updates"] for r in history),
        "resumed_from_epoch": resumed_from_epoch, "checkpoint_selection": SELECTION}


def finished_info(progress_path, config):
    progress = json.loads(Path(progress_path).read_text(encoding="utf-8"))
    if progress["config"] != config: raise ValueError("Finish-only needs the same data, code, settings and precision")
    completed = progress["completed_epoch"] >= config["epochs"]
    stopped_early = progress["completed_epoch"] >= 4 and progress["no_improvement"] >= config["patience"]
    if not (completed or stopped_early): raise ValueError("Training is incomplete: use RESUME, not FINISH_ONLY")
    result = info_from_progress(progress, progress["completed_epoch"])
    result["recovery"] = "final checks only, no epochs or optimizer restore"
    return result


def train(agent, train_items, validation_items, directory, model_name, state_path,
          epochs=12, draws=4096, batch_size=2, accumulation=8, head_lr=2e-5, encoder_lr=1e-5,
          last_layers=4, consistency_weight=.15, replay_fraction=.20, seed=20261003,
          patience=4, signature="", resume=True, _max_new_epochs=None):
    import torch
    from safetensors.torch import load_file
    if batch_size < 2 or batch_size % 2: raise ValueError("Paired training needs an even batch size, at least 2")
    layers = legacy.configure_trainable(agent, last_layers)
    head = [p for n, p in agent.model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    encoder = [p for n, p in agent.model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    parameters = head + encoder
    optimizer = torch.optim.AdamW([{"params": head, "lr": head_lr}] + ([{"params": encoder, "lr": encoder_lr}] if encoder else []), weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda", init_scale=512, enabled=agent.device.type == "cuda" and agent.dtype == torch.float16)
    micro_batches = math.ceil(draws / batch_size)
    total_updates = epochs * math.ceil(micro_batches / accumulation)
    warmup = max(1, int(total_updates * .08))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / warmup,
        max(0., (total_updates - step) / max(1, total_updates - warmup))))
    config = training_config(agent, signature, epochs, draws, batch_size, accumulation, head_lr, encoder_lr,
                             last_layers, seed, consistency_weight, replay_fraction, patience)
    directory, state_path = Path(directory), Path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if not resume and state_path.exists(): raise ValueError("Use a new WORKDIR for fresh training; existing resume state preserved")
    history, audits, start_epoch, no_improvement, selected_epoch = [], [], 0, 0, 0
    if resume and state_path.exists():
        saved = torch.load(state_path, map_location="cpu", weights_only=True)
        if saved["config"] != config: raise ValueError("Resume config/data/code/precision differ; keep original notebook/settings")
        if not (directory / "model.safetensors").exists(): raise FileNotFoundError("Resume needs candidates/ AND resume/")
        agent.model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"]); scheduler.load_state_dict(saved["scheduler"]); scaler.load_state_dict(saved["scaler"])
        baseline, best, history, audits = saved["baseline"], tuple(saved["best"]), saved["history"], saved["audits"]
        start_epoch, no_improvement, selected_epoch = saved["epoch"], saved["no_improvement"], saved["selected_epoch"]
        torch.set_rng_state(saved["rng_cpu"])
        if agent.device.type == "cuda" and saved["rng_cuda"] is not None: torch.cuda.set_rng_state(saved["rng_cuda"], device=0)
        del saved
        print("RESUME: completed epochs", start_epoch, flush=True)
    else:
        baseline, _ = base.evaluate_logits(base.raw_logits(agent, validation_items, 1))
        best = selection_score(baseline)
        atomic_checkpoint(agent, directory, model_name)
        torch.manual_seed(seed)
    stop_epoch = epochs if _max_new_epochs is None else min(epochs, start_epoch + _max_new_epochs)
    if start_epoch >= 4 and no_improvement >= patience: stop_epoch = start_epoch
    for epoch in range(start_epoch, stop_epoch):
        agent.model.train(); agent.model.encoder.eval()
        for layer in list(layers)[len(layers) - last_layers:] if last_layers else []: layer.train()
        order = paired_indices(train_items, draws, seed + epoch, replay_fraction)
        audits.append(dict(Counter(f'{train_items[i]["task"]}:{train_items[i]["labels"][train_items[i]["label"]]}:{train_items[i]["source_role"]}:{train_items[i]["category"]}' for i in order)))
        optimizer.zero_grad(set_to_none=True)
        loss_sum, ce_sum, js_sum, samples, updates, skipped = 0., 0., 0., 0, 0, 0
        selected_items = [train_items[i] for i in order]
        for step, (selected, batch) in enumerate(base.batches(selected_items, batch_size)):
            window_start = (step // accumulation) * accumulation * batch_size
            window_size = min(accumulation * batch_size, draws - window_start)
            with base.amp_context(agent):
                logits, _ = agent.model(**{k: batch[k].to(agent.device) for k in base.MODEL_KEYS})
                loss, ce, js = paired_loss(logits, batch["label"].to(agent.device), consistency_weight)
            if not torch.isfinite(loss): raise RuntimeError("Non-finite loss; latest completed epoch preserved")
            scaler.scale(loss * len(selected) / window_size).backward()
            if step == 0 and encoder and not any(p.grad is not None for p in encoder): raise RuntimeError("Encoder has no gradients")
            loss_sum += loss.item() * len(selected); ce_sum += ce.item() * len(selected); js_sum += js.item() * len(selected); samples += len(selected)
            if (step + 1) % accumulation == 0 or step + 1 == micro_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=not scaler.is_enabled())
                old_scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
                if scaler.get_scale() >= old_scale: scheduler.step(); updates += 1
                else: skipped += 1
                optimizer.zero_grad(set_to_none=True)
        if not updates: raise RuntimeError("No optimizer updates; inspect precision. Previous epoch preserved")
        metrics, _ = base.evaluate_logits(base.raw_logits(agent, validation_items, 1))
        score = selection_score(metrics)
        history.append({"epoch": epoch + 1, "balanced_train_loss": loss_sum / samples, "train_cross_entropy": ce_sum / samples,
            "prompt_js_divergence": js_sum / samples, "validation_score": list(score), "validation": metrics,
            "optimizer_updates": updates, "skipped_updates": skipped, "amp_scale": scaler.get_scale()})
        if score > best:
            best = score; no_improvement = 0; selected_epoch = epoch + 1
            atomic_checkpoint(agent, directory, model_name)
        else: no_improvement += 1
        payload = {"config": config, "model": v3.cpu_tree(agent.model.state_dict()), "optimizer": v3.cpu_tree(optimizer.state_dict()),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "baseline": baseline, "best": list(best),
            "history": v3.cpu_tree(history), "audits": audits, "epoch": epoch + 1, "selected_epoch": selected_epoch,
            "no_improvement": no_improvement, "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state(0) if agent.device.type == "cuda" else None}
        temporary = state_path.with_suffix(".pt.tmp")
        torch.save(v3.cpu_tree(payload), temporary); temporary.replace(state_path); del payload
        v3.atomic_json(state_path.with_suffix(".json"), {"completed_epoch": epoch + 1, "selected_epoch": selected_epoch,
            "history": history, "config": config, "best_score": list(best), "baseline": baseline, "sampling_audit": audits,
            "no_improvement": no_improvement, "trainable_encoder_parameters": sum(p.numel() for p in encoder),
            "trainable_head_parameters": sum(p.numel() for p in head), "qualification": "not evaluated"})
        print({"epoch": epoch + 1, "worst_recall": score[1], "worst_prompt_macro_f1": score[2],
            "selected_epoch": selected_epoch, "updates": updates, "skipped": skipped}, flush=True)
        if epoch + 1 >= 4 and no_improvement >= patience: break
    agent.model.load_state_dict(load_file(str(directory / "model.safetensors")), strict=True); agent.model.eval()
    for parameter in agent.model.parameters(): parameter.requires_grad_(False)
    return {"history": history, "baseline_validation": baseline, "selected_score": list(best), "selected_epoch": selected_epoch,
        "sampling_audit": audits, "last_encoder_layers": last_layers,
        "sampling": "equal class; paired primary+short; authored 80%, historical replay at most 20%; category/family/text cycles",
        "trainable_encoder_parameters": sum(p.numel() for p in encoder), "trainable_head_parameters": sum(p.numel() for p in head),
        "optimizer_updates": sum(r["optimizer_updates"] for r in history), "skipped_updates": sum(r["skipped_updates"] for r in history),
        "resumed_from_epoch": start_epoch, "checkpoint_selection": SELECTION}


def worker_decision(primary, short):
    """Mirror voice_worker.classify thresholds; both logits already evaluated."""
    primary = {**primary, "confidence": round(primary["confidence"], 4)}
    short = {**short, "confidence": round(short["confidence"], 4)}
    answer = primary
    if primary["prediction"] != "unknown" and primary["confidence"] < .6:
        answer = short
        if answer["confidence"] < .75: return {"action": "unknown", "confidence": answer["confidence"]}
    return {"action": answer["prediction"], "confidence": answer["confidence"]}


def runtime_evaluation(predictions, commands, bridge):
    by_event = defaultdict(dict)
    for p in predictions: by_event[p["event_id"]][p["task"]] = p
    decisions = [worker_decision(by_event[r["event_id"]]["action_primary"], by_event[r["event_id"]]["action_short"]) for r in commands]
    probes = base.node_bridge([{**r, "decision": d} for r, d in zip(commands, decisions, strict=True)], bridge)
    rows = [{"event_id": r["event_id"], "gold": r["intent"], "prediction": p["final_action"] if p["final_action"] in LABELS else "unknown",
             "raw_pipeline_action": p["final_action"], "model_required": p["model_required"], "worker": d}
            for r, p, d in zip(commands, probes, decisions, strict=True)]
    required = [r for r in rows if r["model_required"]]
    negatives = [r for r in rows if r["gold"] == "unknown"]
    return {"n": len(rows), "pipeline_accuracy": sum(r["gold"] == r["prediction"] for r in rows) / len(rows),
        "model_required_n": len(required), "model_required_accuracy": sum(r["gold"] == r["prediction"] for r in required) / len(required) if required else None,
        "negative_n": len(negatives), "false_control_rate": sum(r["prediction"] != "unknown" for r in negatives) / len(negatives) if negatives else None,
        "scope": "8-class controls; code-handled play/modes map to unknown for the action model; not audio/Discord accuracy"}, rows
