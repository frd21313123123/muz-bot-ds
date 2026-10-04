"""Portable Laya training utilities, embedded in the Colab/Kaggle notebook.

Data preparation uses only the standard library. Torch/Laya imports are lazy.
Training uses supervised option-level cross entropy, not text generation or RLCD.
"""
from collections import Counter, defaultdict
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import unicodedata


SPLITS = ("train", "validation", "calibration", "test")
MODEL_KEYS = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")


def normalized(text):
    text = unicodedata.normalize("NFKC", text).lower().replace("ё", "е")
    return " ".join("".join(c if c.isalnum() else " " for c in text).split())


def json_object(value):
    value = json.loads(value) if isinstance(value, str) else value
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    return value


def read_jsonl(paths):
    rows = []
    for path in paths:
        for line_number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json_object(line))
            except (ValueError, TypeError) as error:
                raise ValueError(f"Invalid JSONL at {Path(path).name}:{line_number}") from error
    return rows


def automatic_label_allowed(row, allow_automatic_labels):
    return (allow_automatic_labels and row.get("reviewed") is False
            and row.get("label_status") == "rule_verified"
            and ((row.get("origin") == "synthetic" and row.get("label_source") == "synthetic_template_checked")
                 or (row.get("origin") == "bot_log" and row.get("label_source") == "bot_transcript_rules")))


def load_commands(paths, reviews_path=None, intent_labels=(), allow_automatic_labels=False):
    """Accept the existing export, reviewed command rows, or logs + separate reviews."""
    reviews = {}
    if reviews_path:
        for row in read_jsonl([reviews_path]):
            key = row.get("event_id")
            if not isinstance(key, str) or not key or key in reviews:
                raise ValueError("Invalid/duplicate review event_id")
            reviews[key] = row["intent"]
    found, seen, commands = set(), set(), []
    for row in read_jsonl(paths):
        exported = "human-reviewed" in row.get("tags", []) and "source_event_id" in row
        key = row.get("source_event_id") if exported else row.get("event_id")
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError("Every source row needs a unique event_id/source_event_id")
        seen.add(key)
        if key in reviews:
            intent = reviews[key]
            found.add(key)
        elif exported:
            intent = json_object(row.get("gold"))["intent"]
        elif row.get("reviewed") is True and row.get("label_status") != "unreviewed":
            intent = row.get("intent")
        elif automatic_label_allowed(row, allow_automatic_labels):
            intent = row.get("intent")
        else:
            continue  # Predictions, outcomes and suggested_intent are never gold.
        if intent not in intent_labels:
            raise ValueError(f"Unknown reviewed intent for {key}")
        state = json_object(row["state"]) if "state" in row else {"message": row.get("message"), "player": row.get("player")}
        message = state.get("message")
        if not isinstance(message, str) or not message.strip() or len(message) > 1000:
            raise ValueError(f"Invalid message for {key}")
        # No canonical_message, model_decision, diagnostics, outcome or gold hints.
        player = state.get("player")
        if player is not None:
            if not isinstance(player, dict) or any(type(player.get(k)) is not bool for k in ("connected", "playing", "paused", "autoplay")):
                raise ValueError(f"Invalid player state for {key}")
            if type(player.get("queue_length")) is not int or player["queue_length"] < 0:
                raise ValueError(f"Invalid queue_length for {key}")
        group = row.get("group_id", key)
        if not isinstance(group, str) or not group:
            raise ValueError("group_id must be a non-empty string")
        commands.append({"event_id": key, "message": message, "player": player,
                         "intent": intent, "group_id": group, "origin": row.get("origin", "human-reviewed"),
                         "label_source": row.get("label_source", "human-review")})
    if set(reviews) != found:
        raise ValueError("Some reviewed events were not found in source files")
    if not commands:
        raise ValueError("No manually reviewed commands; label/export data first, or explicitly allow rule-verified automatic labels")
    return commands


def node_bridge(rows, script):
    executable = shutil.which("node")
    if not executable:
        raise RuntimeError("Node.js required for exact bot preprocessing; install nodejs and rerun")
    result = subprocess.run([executable, str(script)], input=json.dumps(rows, ensure_ascii=False),
                            text=True, encoding="utf-8", capture_output=True, check=True)
    return json.loads(result.stdout)


def prepare_commands(commands, bridge_script):
    processed = node_bridge(commands, bridge_script)
    return [{**row, **processed[i]} for i, row in enumerate(commands)]


def grouped_split(commands, seed=42, fractions=(0.6, 0.15, 0.1, 0.15), min_groups_per_label=4):
    """Union raw/canonical duplicates and explicit families before stratification.

    Context variants of the same phrase stay together, even with different gold.
    Search over deterministic candidate assignments; fail rather than silently
    lose a class in one of the four held-out partitions.
    """
    if len(fractions) != 4 or any(x <= 0 for x in fractions) or not math.isclose(sum(fractions), 1):
        raise ValueError("Four positive split fractions must sum to 1")
    parent = list(range(len(commands)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        parent[find(a)] = find(b)
    owners = {}
    for i, row in enumerate(commands):
        keys = [("family", row["group_id"])]
        keys += [("text", normalized(row[k])) for k in ("message", "canonical") if row.get(k)]
        for key in keys:
            if key in owners:
                union(i, owners[key])
            owners[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(commands):
        groups[find(i)].append(row)
    label_groups = Counter(label for group in groups.values() for label in {r["intent"] for r in group})
    scarce = {label: n for label, n in label_groups.items() if n < min_groups_per_label}
    if scarce:
        raise ValueError(f"Too few independent phrase families (need {min_groups_per_label}): {scarce}")
    labels = sorted(label_groups)
    totals = Counter(r["intent"] for r in commands)
    rng, best, best_score = random.Random(seed), None, float("inf")
    # Group assignment is chosen only from labels and sizes, never model results.
    for _ in range(1000):
        bins = {name: [] for name in SPLITS}
        for group in groups.values():
            name = rng.choices(SPLITS, weights=fractions)[0]
            bins[name].extend(group)
        counts = {s: Counter(r["intent"] for r in bins[s]) for s in SPLITS}
        if any(not counts[s][label] for s in SPLITS for label in labels):
            continue
        score = sum(abs(counts[s][label] / totals[label] - fractions[j])
                    for j, s in enumerate(SPLITS) for label in labels)
        if score < best_score:
            best, best_score = bins, score
    if best is None:
        # Deterministic greedy coverage fallback for small homogeneous datasets.
        bins = {s: [] for s in SPLITS}
        counts = {s: Counter() for s in SPLITS}
        ordered = list(groups.values())
        rng.shuffle(ordered)
        ordered.sort(key=lambda g: min(label_groups[r["intent"]] for r in g))
        for group in ordered:
            gc = Counter(r["intent"] for r in group)
            def cost(s):
                j = SPLITS.index(s)
                uncovered = sum(counts[s][label] == 0 for label in gc)
                fill = sum(counts[s][label] / max(1, totals[label] * fractions[j]) for label in gc)
                return -1000 * uncovered + fill
            target = min(SPLITS, key=cost)
            bins[target].extend(group)
            counts[target].update(gc)
        if any(not counts[s][label] for s in SPLITS for label in labels):
            raise ValueError("Cannot split these overlapping groups with class coverage; add independent families")
        best = bins
    return best


def runtime_records(commands, worker):
    records = []
    allowed = worker.QUESTIONS["action"]["criteria"]
    for command in commands:
        text = command["canonical"].strip().rstrip('.!…').strip().lower()
        text = text[:1].upper() + text[1:]
        # Both production prompts learn the same target; they share a split.
        target = command["intent"] if command["intent"] in allowed else "unknown"
        for prompt, questions in (("primary", worker.QUESTIONS), ("short", worker.SHORT_QUESTIONS)):
            records.append({"event_id": command["event_id"], "task": "action_" + prompt,
                            "state": text, "questions": questions, "gold": {"action": target}})
    return records


def planner_questions(intent_questions):
    return {
        "should_respond": {"type": "choice", "instructions": "Нужно ли реагировать на эту реплику в текущей голосовой сессии?",
                           "criteria": {"respond": "Обращение по имени в idle либо команда владельца в awaiting; пользователь в канале бота, голос включён",
                                        "ignore": "Фоновый разговор, другой пользователь, нет доступа, голос выключен или сессия занята"}},
        "next_tool": {"type": "choice", "instructions": "Какую функцию музыкального бота выбрать? Не выполняй команды без активной сессии владельца.",
                      "criteria": {"wake_ack": "Сигнал на отдельное имя активации, затем ждать команду",
                                   "youtube_music_search": "Поиск и включение музыки по тексту",
                                   "direct_youtube_video": "Конкретное видео YouTube по ссылке",
                                   "player_control": "Команда управления плеером или его режимами",
                                   "no_tool": "Игнорировать или сообщить о неподдерживаемой команде без вызова инструмента"}},
        "intent": deepcopy(intent_questions["intent"]),
    }


def load_planner(paths, questions, bridge_script, allow_automatic_labels=False):
    records = []
    required = {"message", "phase", "voice_enabled", "in_bot_channel", "is_owner", "wake_name", "player"}
    source_rows = read_jsonl(paths)
    contexts = [json_object(row["state"]) for row in source_rows]
    wakes = node_bridge([{"message": state.get("message", ""), "wake_name": state.get("wake_name", "")}
                         for state in contexts], bridge_script)
    for row, wake in zip(source_rows, wakes, strict=True):
        if row.get("reviewed") is not True and not automatic_label_allowed(row, allow_automatic_labels):
            raise ValueError("Planner examples must be manually reviewed")
        state, gold = json_object(row["state"]), json_object(row["gold"])
        if set(state) != required or state["phase"] not in {"idle", "awaiting", "busy", "disabled"}:
            raise ValueError("Planner state must use the documented voice context, without answer hints")
        if not isinstance(state["message"], str) or len(state["message"]) > 1000 or not isinstance(state["wake_name"], str):
            raise ValueError("Invalid planner text")
        if any(type(state[k]) is not bool for k in ("voice_enabled", "in_bot_channel", "is_owner")):
            raise ValueError("Planner context flags must be booleans")
        if set(gold) != set(questions) or any(gold[k] not in questions[k]["criteria"] for k in questions):
            raise ValueError("Invalid planner gold")
        can_wake = state["phase"] == "idle" and wake["wake_match"]
        can_command = state["phase"] == "awaiting" and state["is_owner"]
        eligible = state["voice_enabled"] and state["in_bot_channel"] and (can_wake or can_command)
        if (gold["should_respond"] == "respond") != eligible:
            raise ValueError("Planner respond label contradicts explicit session context")
        tool, intent = gold["next_tool"], gold["intent"]
        if (not eligible and (tool != "no_tool" or intent != "unknown")
                or (can_wake and eligible and (tool != "wake_ack" or intent != "unknown"))
                or (tool == "wake_ack" and not can_wake)
                or (tool in {"youtube_music_search", "direct_youtube_video"} and intent != "play")
                or (tool == "player_control" and intent in {"play", "unknown"})
                or (tool == "no_tool" and intent != "unknown")):
            raise ValueError("Inconsistent planner function/intent/context labels")
        key, group = row.get("event_id"), row.get("group_id")
        if not isinstance(key, str) or not key or not isinstance(group, str) or not group:
            raise ValueError("Planner requires event_id and group_id")
        records.append({"event_id": key, "group_id": group, "message": state["message"],
                        "canonical": state["message"], "intent": "|".join(gold.values()),
                        "task": "planner", "state": state, "questions": questions, "gold": gold,
                        "origin": row.get("origin", "human-reviewed"), "label_source": row.get("label_source", "human-review")})
    return records


def encode_records(agent, records):
    items = []
    for record in records:
        questions, gold = record["questions"], record["gold"]
        for qid, question in questions.items():
            agent._check_question(qid, question)
            internal = agent._to_internal(question)
            item = agent._encode_state(record["state"], [qid], {qid: internal})[0]
            labels = list(question["criteria"])
            if item["state_stats"]["truncated"]:
                raise ValueError("Input was truncated; increase token budget")
            if item["options"]["options_distinct"] != item["options"]["options"]:
                raise ValueError("Option truncation made labels indistinguishable")
            item.update(label=labels.index(gold[qid]), labels=labels,
                        event_id=record["event_id"], task=record["task"], qid=qid)
            items.append(item)
    return items


def batches(items, batch_size, shuffle=False, seed=42):
    from laya.common import collate_items
    order = list(range(len(items)))
    if shuffle:
        random.Random(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        selected = [items[i] for i in order[start:start + batch_size]]
        yield selected, collate_items([[item] for item in selected], batches.pad_id)


def amp_context(agent):
    import torch
    return torch.autocast("cuda", dtype=agent.dtype) if agent.device.type == "cuda" else nullcontext()


def raw_logits(agent, items, batch_size=8):
    import torch
    agent.model.eval()
    rows = []
    with torch.no_grad():
        for selected, batch in batches(items, batch_size):
            with amp_context(agent):
                logits, _ = agent.model(**{k: batch[k].to(agent.device) for k in MODEL_KEYS})
            for item, z in zip(selected, logits, strict=True):
                rows.append({"item": item, "logits": z[:len(item["labels"])].float().cpu().tolist()})
    return rows


def evaluate_logits(rows, temperatures=None):
    import numpy as np
    from laya.common import temp_bucket, ece_score
    predictions, results = [], {}
    for row in rows:
        item = row["item"]
        t = 1.0 if temperatures is None else temperatures["temperature_by_options"].get(
            temp_bucket(item["qtype"], len(item["labels"])), temperatures["temperature"][item["qtype"]])
        z = np.asarray(row["logits"], dtype=float) / t
        p = np.exp(z - z.max()); p /= p.sum()
        y, pred = item["label"], int(p.argmax())
        predictions.append({"event_id": item["event_id"], "task": item["task"], "qid": item["qid"],
                            "gold": item["labels"][y], "prediction": item["labels"][pred],
                            "confidence": float(p[pred]), "correct": y == pred,
                            "nll": float(-np.log(max(p[y], 1e-12))),
                            "brier": float(((p - np.eye(len(p))[y]) ** 2).sum())})
    groups = defaultdict(list)
    for row in predictions:
        groups[row["task"] + "/" + row["qid"]].append(row)
    for key, group in groups.items():
        labels = sorted({r["gold"] for r in group} | {r["prediction"] for r in group})
        per_class = {}
        for label in labels:
            tp = sum(r["gold"] == label and r["prediction"] == label for r in group)
            actual = sum(r["gold"] == label for r in group)
            predicted = sum(r["prediction"] == label for r in group)
            per_class[label] = {"support": actual, "precision": tp / predicted if predicted else 0.0,
                                "recall": tp / actual if actual else 0.0,
                                "f1": 2 * tp / (actual + predicted) if actual + predicted else 0.0}
        results[key] = {"n": len(group), "accuracy": sum(r["correct"] for r in group) / len(group),
                       "macro_f1": sum(x["f1"] for x in per_class.values()) / len(per_class),
                       "nll": sum(r["nll"] for r in group) / len(group),
                       "brier": sum(r["brier"] for r in group) / len(group),
                       "ece": ece_score(np.array([r["confidence"] for r in group]), np.array([r["correct"] for r in group])),
                       "per_class": per_class}
    return results, predictions


def save_checkpoint(agent, directory, model_name, temperatures=None):
    import torch
    from safetensors.torch import save_file
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = deepcopy(agent.cfg)
    config["model_name"] = model_name  # Never pretend to be the legacy music-v3 profile.
    config["temperature"] = temperatures["temperature"] if temperatures else [1.0, 1.0, 1.0]
    config["temperature_by_options"] = temperatures["temperature_by_options"] if temperatures else {}
    config.pop("lang_temperatures", None)
    config["training"] = {"method": "supervised_option_cross_entropy", "fine_tuned_from_checkpoint": True}
    (directory / "rl_agent_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    tensors = {k: v.detach().cpu().contiguous().clone() for k, v in agent.model.state_dict().items()}
    tensors["temperature"] = torch.tensor(config["temperature"], dtype=tensors["temperature"].dtype)
    save_file(tensors, str(directory / "model.safetensors"))
    agent.model.encoder.config.save_pretrained(str(directory / "encoder"))
    agent.tok.save_pretrained(str(directory / "tokenizer"))


def train(agent, train_items, validation_items, directory, epochs=4, batch_size=2,
          accumulation=8, head_lr=2e-5, encoder_lr=2e-6, train_encoder=False, seed=42):
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    torch.manual_seed(seed)
    for name, parameter in agent.model.named_parameters():
        parameter.requires_grad_(not name.startswith("act_head.") and (train_encoder or not name.startswith("encoder.")))
    # The separate act/escalate head is not the bot's action classifier and has no gold here.
    head = [p for n, p in agent.model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    encoder = [p for n, p in agent.model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    groups = [{"params": head, "lr": head_lr}]
    if encoder:
        groups.append({"params": encoder, "lr": encoder_lr})
        agent.model.encoder.gradient_checkpointing_enable()
    agent.model.head_checkpointing = True
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=agent.device.type == "cuda" and agent.dtype == torch.float16)
    micro_batches = math.ceil(len(train_items) / batch_size)
    updates_per_epoch = math.ceil(micro_batches / accumulation)
    total_updates = epochs * updates_per_epoch
    warmup = max(1, int(total_updates * 0.1))
    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total_updates - step) / max(1, total_updates - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    best, history = float("inf"), []
    for epoch in range(epochs):
        agent.model.train()
        if not train_encoder:
            agent.model.encoder.eval()  # Freeze encoder dropout as well as its parameters.
        optimizer.zero_grad(set_to_none=True)
        loss_sum, count, updates = 0.0, 0, 0
        for index, (selected, batch) in enumerate(batches(train_items, batch_size, True, seed + epoch)):
            # Correct sample denominator also for an incomplete last accumulation window.
            window_start = (index // accumulation) * accumulation * batch_size
            window_size = min(accumulation * batch_size, len(train_items) - window_start)
            with amp_context(agent):
                logits, _ = agent.model(**{k: batch[k].to(agent.device) for k in MODEL_KEYS})
                loss = F.cross_entropy(logits.float(), batch["label"].to(agent.device), reduction="sum")
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; reduce LR/use fp32")
            scaler.scale(loss / window_size).backward()
            loss_sum += loss.item(); count += len(selected)
            if (index + 1) % accumulation == 0 or index + 1 == micro_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in agent.model.parameters() if p.requires_grad], 1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                if scaler.get_scale() >= scale_before:
                    scheduler.step(); updates += 1
                optimizer.zero_grad(set_to_none=True)
        validation, _ = evaluate_logits(raw_logits(agent, validation_items, batch_size))
        val_nll = sum(x["nll"] * x["n"] for x in validation.values()) / sum(x["n"] for x in validation.values())
        history.append({"epoch": epoch + 1, "train_nll": loss_sum / count, "validation_nll": val_nll, "updates": updates})
        print(history[-1], flush=True)
        if val_nll < best:
            best = val_nll
            save_checkpoint(agent, directory, "laya-muz-bot-controls-v1")
    agent.model.load_state_dict(load_file(str(Path(directory) / "model.safetensors")), strict=True)
    agent.model.eval()
    return history


def calibrate(rows):
    from laya.calibrate import fit_temperature_map
    records = []
    # Do not count primary and fallback prompts as two independent events.
    seen = set()
    for row in rows:
        item = row["item"]
        key = (item["event_id"], item["qid"])
        if key in seen:
            continue
        seen.add(key)
        k = len(item["labels"])
        target = [float(i == item["label"]) for i in range(k)]
        records.append((item["qtype"], row["logits"], target, k))
    result = fit_temperature_map(records, compute_ece=False)
    return {"temperature": result["temperature"], "temperature_by_options": result["temperature_by_options"]}


def runtime_evaluation(agent, commands, worker, bridge_script):
    predictions = [worker.classify(agent, row["canonical"]) for row in commands]
    checked = node_bridge([{**row, "decision": decision} for row, decision in zip(commands, predictions, strict=True)], bridge_script)
    rows = [{"event_id": row["event_id"], "gold": row["intent"], "prediction": result["final_action"],
             "model_required": row["model_required"], "confidence": decision["confidence"]}
            for row, result, decision in zip(commands, checked, predictions, strict=True)]
    negatives = [r for r in rows if r["gold"] in {"unknown", "play"}]
    required = [r for r in rows if r["model_required"]]
    false_controls = sum(r["prediction"] not in {"unknown", "play"} for r in negatives)
    return {"n": len(rows), "pipeline_accuracy": sum(r["gold"] == r["prediction"] for r in rows) / len(rows),
            "model_required_n": len(required),
            "model_required_accuracy": sum(r["gold"] == r["prediction"] for r in required) / len(required) if required else None,
            "negative_n": len(negatives), "false_control_count": false_controls,
            "false_control_rate": false_controls / len(negatives) if negatives else None,
            "note": "Current text pipeline only; not an audio, Discord, cancellation or live YouTube test"}, rows


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
