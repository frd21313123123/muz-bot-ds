"""Notebook cell: measured Action V4 training, recovery, runtime parity and export."""
def diagnostics_export(workdir):
    diagnostics = Path(workdir) / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    for folder, pattern in (("candidates", "*/training_report.json"), ("resume", "*.json"), ("failures", "*.json")):
        for path in (Path(workdir) / folder).glob(pattern):
            shutil.copyfile(path, diagnostics / (path.parent.name + "-" + path.name))
    shutil.copyfile(DATA_DIR / "manifest.json", diagnostics / "dataset-manifest.json")
    return shutil.make_archive(str(Path(workdir) / "lia-action-v4-diagnostics"), "zip", root_dir=diagnostics)


def package_candidate(directory, qualified):
    v3.atomic_json(directory / "qualification.json", {"qualified_on_synthetic_holdouts": qualified, "smoke_only": SMOKE_TEST,
        "scope": "current compatible text control syntax, not real speech", "warning": "Do not deploy CANDIDATE/SMOKE; validate fresh reviewed real speech even after synthetic gates pass"})
    hashes = {p.relative_to(directory).as_posix(): training.sha256_file(p) for p in directory.rglob("*") if p.is_file() and p.name != "sha256.json"}
    v3.atomic_json(directory / "sha256.json", hashes)
    if SMOKE_TEST: return None
    name = directory.name if qualified else directory.name + "-CANDIDATE"
    filename = shutil.make_archive(str(WORKDIR / name), "zip", root_dir=directory.parent, base_dir=directory.name)
    with zipfile.ZipFile(filename) as archive:
        assert archive.testzip() is None, "Archive CRC failed"
    return filename


def runtime_gate(predictions, commands):
    metrics, outputs = v4.runtime_evaluation(predictions, commands, bridge_path)
    failures, recalls, supports = [], {}, {}
    for label in sorted(v4.LABELS - {"unknown"}):
        relevant = [r for r in outputs if r["gold"] == label and r["model_required"]]
        supports[label] = len(relevant)
        recalls[label] = sum(r["prediction"] == label for r in relevant) / len(relevant) if relevant else None
        if not relevant: failures.append(f"No model-required examples: {label}")
        elif recalls[label] < MIN_RECALL: failures.append(f"runtime/{label}: recall={recalls[label]:.3f}")
    if metrics["model_required_accuracy"] is None or metrics["model_required_accuracy"] < .85: failures.append("model_required accuracy < .85")
    if metrics["false_control_rate"] is None or metrics["false_control_rate"] > .02: failures.append("false control rate > .02")
    return {"passed": not failures, "failures": failures, "per_class_recall": recalls, "per_class_support": supports, "metrics": metrics}, outputs


def production_parity(agent, predictions, commands):
    # Actual worker API, not only a replica of its thresholds.
    indexed = {}
    for p in predictions: indexed.setdefault(p["event_id"], {})[p["task"]] = p
    selected, counters, results = [], {}, []
    for r in commands:
        count = counters.get(r["intent"], 0)
        if count < 3: selected.append(r); counters[r["intent"]] = count + 1
    for row in selected:
        pair = indexed[row["event_id"]]
        expected = v4.worker_decision(pair["action_primary"], pair["action_short"])
        actual = worker.classify(agent, row["canonical"])
        assert actual["action"] == expected["action"], (row["event_id"], actual, expected)
        assert abs(actual["confidence"] - expected["confidence"]) <= .005, (row["event_id"], "Batch and worker probabilities differ", actual, expected)
        results.append({"event_id": row["event_id"], **actual})
    return results


def run_action():
    model_name = "laya-muz-bot-controls-v4" + ("-SMOKE-ONLY" if SMOKE_TEST else "")
    directory = WORKDIR / "candidates" / model_name
    directory.mkdir(parents=True, exist_ok=True)
    # Remove stale qualification artifacts before an attempt; never delete weights/resume.
    for path in (WORKDIR / (model_name + ".zip"), WORKDIR / (model_name + "-CANDIDATE.zip"), directory / "qualification.json",
                 directory / "sha256.json", directory / "training_report.json", WORKDIR / "failures" / "action.json"):
        if path.exists(): path.unlink()
    agent = None
    try:
        print("LOAD", "best candidate for finish-only" if FINISH_ONLY else "pinned base for a fresh V4 run / epoch resume", flush=True)
        kwargs = {} if FINISH_ONLY else {"revision": BASE_REVISION}
        agent = laya.load(str(directory) if FINISH_ONLY else BASE_MODEL, device=DEVICE, **kwargs)
        if agent.device.type != DEVICE: raise RuntimeError("Requested device unavailable; inspect GPU memory")
        if DEVICE == "cuda": agent.dtype = AMP_DTYPE
        legacy.install_temperatures(agent, {"temperature": [1., 1., 1.], "temperature_by_options": {}})
        training.batches.pad_id = agent.tok.pad_token_id
        print("TOKENIZE", sum(len(r) for r in action_records.values()), "question inputs", flush=True)
        items = {name: v4.encode(agent, records, by_id) for name, records in action_records.items()}
        signature = hashlib.sha256(json.dumps({"data": DATA_MANIFEST["file_sha256"], "code": SNAPSHOT["source_sha256"],
            "base": PINNED_BASE_REVISION, "smoke": SMOKE_TEST}, sort_keys=True).encode()).hexdigest()
        config = v4.training_config(agent, signature, EPOCHS, ACTION_DRAWS, BATCH_SIZE, ACCUMULATION, HEAD_LR,
            ENCODER_LR, LAST_ENCODER_LAYERS, SEED, CONSISTENCY_WEIGHT, REPLAY_FRACTION, PATIENCE)
        if FINISH_ONLY:
            info = v4.finished_info(WORKDIR / "resume/action.json", config)
            for p in agent.model.parameters(): p.requires_grad_(False)
        else:
            print("TRAIN", EPOCHS, "max epochs", flush=True)
            info = v4.train(agent, items["train"], items["validation"], directory, model_name, WORKDIR / "resume/action.pt",
                epochs=EPOCHS, draws=ACTION_DRAWS, batch_size=BATCH_SIZE, accumulation=ACCUMULATION,
                head_lr=HEAD_LR, encoder_lr=ENCODER_LR, last_layers=LAST_ENCODER_LAYERS,
                consistency_weight=CONSISTENCY_WEIGHT, replay_fraction=REPLAY_FRACTION,
                seed=SEED, patience=PATIENCE, signature=signature, resume=RESUME)
        print("CALIBRATION", flush=True)
        temperatures = v3.calibrate(training.raw_logits(agent, items["calibration"], 1))
        legacy.install_temperatures(agent, temperatures)
        metrics, predictions, gates, pipeline_outputs = {}, {}, {}, {}
        for name in ("validation", "test"):
            print("EVALUATION", name, "and exact bot parser", flush=True)
            metrics[name], predictions[name] = training.evaluate_logits(training.raw_logits(agent, items[name], 1), temperatures)
            gates[name] = legacy.metric_gate(metrics[name], MIN_RECALL, MIN_MACRO_F1)
            gates[name + "_pipeline"], pipeline_outputs[name] = runtime_gate(predictions[name], action_splits[name])
        gates["optimizer"] = {"passed": info["optimizer_updates"] > 0 and info["skipped_updates"] == 0,
            "failures": [] if info["optimizer_updates"] > 0 and info["skipped_updates"] == 0 else ["Zero updates or skipped AMP steps"]}
        before = production_parity(agent, predictions["test"], action_splits["test"])
        print("SAVE + RELOAD", flush=True)
        v4.atomic_checkpoint(agent, directory, model_name, temperatures)
        cfg_path = directory / "rl_agent_config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if DEVICE == "cuda": cfg["amp_dtype"] = "bf16" if AMP_DTYPE == torch.bfloat16 else "fp16"
        cfg["training"].update({"sampling": info["sampling"], "checkpoint_selection": v4.SELECTION,
            "consistency_weight": CONSISTENCY_WEIGHT, "replay_fraction": REPLAY_FRACTION, "separate_task": "action"})
        v3.atomic_json(cfg_path, cfg)
        legacy.release(agent); del agent; agent = None; gc.collect()
        agent = laya.load(str(directory), device=DEVICE)
        if DEVICE == "cuda": agent.dtype = AMP_DTYPE
        after = [dict(event_id=r["event_id"], **worker.classify(agent, by_id[r["event_id"]]["canonical"])) for r in before]
        assert [r["action"] for r in before] == [r["action"] for r in after], "Reload changed decisions"
        assert all(abs(a["confidence"] - b["confidence"]) <= .005 for a, b in zip(before, after, strict=True)), "Reload changed probabilities"
        qualified = not SMOKE_TEST and all(g["passed"] for g in gates.values())
        report = {"stage": "action", "smoke_only": SMOKE_TEST, "qualified_on_synthetic_holdouts": qualified,
            "training": info, "training_config": config, "temperatures": temperatures, "metrics": metrics, "gates": gates,
            "checkpoint_roundtrip": "passed", "runtime_batch_worker_parity": "passed", "roundtrip_probes": len(before),
            "data_manifest": DATA_MANIFEST, "source_sha256": SNAPSHOT["source_sha256"], "base_revision": PINNED_BASE_REVISION,
            "environment": {k: importlib.metadata.version(k) for k in ("torch", "laya", "transformers", "huggingface-hub")},
            "amp_dtype": str(AMP_DTYPE), "split_summary": split_summary,
            "limitation": "Assistant-authored related-language paraphrases; small holdouts, parser exclusions; no independently reviewed real speech"}
        v3.atomic_json(directory / "training_report.json", report)
        for fold in ("validation", "test"):
            errors = [{**p, "text": by_id[p["event_id"]]["message"]} for p in predictions[fold] if not p["correct"]]
            v3.atomic_json(WORKDIR / "diagnostics" / (fold + "-errors.json"), errors)
            v3.atomic_json(WORKDIR / "diagnostics" / (fold + "-predictions.json"), predictions[fold])
            v3.atomic_json(WORKDIR / "diagnostics" / (fold + "-pipeline.json"), pipeline_outputs[fold])
        report["export_archive"] = package_candidate(directory, qualified)
        report["diagnostics_archive"] = diagnostics_export(WORKDIR)
        print("ACTION V4 FINISHED", "QUALIFIED" if qualified else "CANDIDATE / SMOKE: do not deploy", flush=True)
        return report
    except Exception as error:
        v3.atomic_json(WORKDIR / "failures/action.json", {"error": type(error).__name__, "message": str(error), "qualified": False,
            "recovery": "Resume incomplete epochs; FINISH_ONLY only after completed training. Preserve both candidates/ and resume/."})
        if (directory / "model.safetensors").exists(): package_candidate(directory, False)
        diagnostics_export(WORKDIR)
        raise
    finally:
        if agent is not None: legacy.release(agent)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
