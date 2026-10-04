"""Embedded notebook cell: training, recovery artifacts and measured acceptance."""
def diagnostics_export(workdir):
    workdir = Path(workdir)
    diagnostics = workdir / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    saved = list((workdir / "candidates").glob("*/training_report.json"))
    saved += list((workdir / "resume").glob("*.json"))
    saved += list((workdir / "failures").glob("*.json"))
    for path in saved:
        shutil.copyfile(path, diagnostics / (path.parent.name + "-" + path.name))
    manifest = workdir / "data" / "manifest.json"
    if manifest.exists(): shutil.copyfile(manifest, diagnostics / "dataset-manifest.json")
    archive = shutil.make_archive(str(workdir / "lia-v3-diagnostics"), "zip", root_dir=diagnostics)
    print("DIAGNOSTICS:", archive, flush=True)
    return archive


def package_candidate(directory, model_name, qualified=False):
    qualification = {"qualified_on_synthetic_holdouts": qualified, "smoke_only": SMOKE_TEST,
                     "semantic_model_enabled_in_bot": False,
                     "warning": "Only qualified action weights match the current bot; review fresh real speech before deployment"}
    v3.atomic_json(directory / "qualification.json", qualification)
    hashes = {str(p.relative_to(directory)): training.sha256_file(p) for p in directory.rglob("*")
              if p.is_file() and p.name != "sha256.json"}
    v3.atomic_json(directory / "sha256.json", hashes)
    if SMOKE_TEST: return None
    name = model_name if qualified else model_name + "-CANDIDATE"
    archive = shutil.make_archive(str(WORKDIR / name), "zip", root_dir=directory.parent, base_dir=directory.name)
    print("QUALIFIED" if qualified else "UNVERIFIED CANDIDATE — DO NOT DEPLOY", archive, flush=True)
    return archive


def runtime_gate(agent, rows):
    metrics, predictions = training.runtime_evaluation(agent, rows, worker, bridge_path)
    selected = [r for r in predictions if r["model_required"]]
    failures, recalls = [], {}
    for label in sorted({r["gold"] for r in selected}):
        relevant = [r for r in selected if r["gold"] == label]
        recalls[label] = sum(r["prediction"] == label for r in relevant) / len(relevant)
        if recalls[label] < MIN_RECALL: failures.append(f"pipeline/{label} recall={recalls[label]:.3f}")
    if metrics["model_required_accuracy"] is None: failures.append("No model-required examples")
    elif metrics["model_required_accuracy"] < .85: failures.append("model_required accuracy < .85")
    if metrics["false_control_rate"] is None: failures.append("No negative examples")
    elif metrics["false_control_rate"] > .02: failures.append("false control rate > .02")
    return {"passed": not failures, "failures": failures, "per_class_recall": recalls, "metrics": metrics}, predictions


def effective_semantic_gate(predictions):
    # Explicit proposed interface: low-confidence positives abstain; no session labels in NLI.
    rows = [{**r, "prediction": r["prediction"] if r["confidence"] >= .60 else "unknown"} for r in predictions]
    labels = sorted(SEMANTIC_QUESTIONS["intent"]["criteria"])
    per_class = {}
    for label in labels:
        actual = sum(r["gold"] == label for r in rows)
        predicted = sum(r["prediction"] == label for r in rows)
        tp = sum(r["gold"] == label and r["prediction"] == label for r in rows)
        per_class[label] = {"support": actual, "recall": tp / actual if actual else 0.,
                            "f1": 2 * tp / (actual + predicted) if actual + predicted else 0.}
    metrics = {"semantic/intent@0.60": {"macro_f1": sum(v["f1"] for v in per_class.values()) / len(labels), "per_class": per_class}}
    return {**legacy.metric_gate(metrics, MIN_RECALL, MIN_MACRO_F1), "metrics": metrics,
            "confidence_threshold": .60, "interface_enabled_in_bot": False}


def probe_outputs(agent, records, stage):
    seen, selected = set(), []
    for r in records:
        label = tuple(r["gold"].values())
        if label not in seen: selected.append(r); seen.add(label)
    outputs = []
    for r in selected:
        if stage == "action": outputs.append(worker.classify(agent, r["state"]))
        else:
            a = agent.predict(r["state"], r["questions"])["answers"]["intent"]
            outputs.append({"action": a["choice"], "confidence": a["answer_confidence"]})
    return outputs


def run_stage(stage, records, draws):
    model_name = "laya-muz-bot-controls-v3" if stage == "action" else "laya-muz-bot-semantic-v3"
    if SMOKE_TEST: model_name += "-SMOKE-ONLY"
    directory = WORKDIR / "candidates" / model_name
    directory.mkdir(parents=True, exist_ok=True)
    for path in (WORKDIR / (model_name + ".zip"), WORKDIR / (model_name + "-CANDIDATE.zip"),
                 directory / "sha256.json", directory / "training_report.json", directory / "qualification.json",
                 WORKDIR / "diagnostics" / (model_name + "-training_report.json"),
                 WORKDIR / "diagnostics" / (stage + "-test-errors.json"),
                 WORKDIR / "failures" / (stage + ".json")):
        if path.exists(): path.unlink()
    agent = None
    try:
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        agent = laya.load(BASE_MODEL, device=DEVICE, revision=BASE_REVISION)
        if agent.device.type != DEVICE: raise RuntimeError("Laya fell back from requested GPU; inspect available GPU memory")
        legacy.install_temperatures(agent, {"temperature": [1., 1., 1.], "temperature_by_options": {}})
        training.batches.pad_id = agent.tok.pad_token_id
        items = {name: v3.encode(agent, rows, by_id) for name, rows in records.items()}
        signature = hashlib.sha256(json.dumps({"data": DATA_MANIFEST["file_sha256"], "sources": SNAPSHOT["source_sha256"],
                                               "base": BASE_REVISION, "stage": stage, "smoke": SMOKE_TEST}, sort_keys=True).encode()).hexdigest()
        info = v3.train(agent, items["train"], items["validation"], directory, model_name,
            WORKDIR / "resume" / (stage + ".pt"), epochs=EPOCHS, draws=draws, batch_size=BATCH_SIZE,
            accumulation=ACCUMULATION, head_lr=HEAD_LR, encoder_lr=ENCODER_LR,
            last_layers=LAST_ENCODER_LAYERS, seed=SEED, patience=4, signature=signature, resume=RESUME)
        temperatures = v3.calibrate(training.raw_logits(agent, items["calibration"], BATCH_SIZE))
        legacy.install_temperatures(agent, temperatures)
        metrics, predictions, gates = {}, {}, {}
        for name in ("validation", "test"):
            metrics[name], predictions[name] = training.evaluate_logits(training.raw_logits(agent, items[name], BATCH_SIZE), temperatures)
            gates[name] = legacy.metric_gate(metrics[name], MIN_RECALL, MIN_MACRO_F1)
            if stage == "action":
                gates[name + "_pipeline"], _ = runtime_gate(agent, action_splits[name])
            else: gates[name + "_abstention"] = effective_semantic_gate(predictions[name])
        training.save_checkpoint(agent, directory, model_name, temperatures)
        config_path = directory / "rl_agent_config.json"
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        cfg["training"].update({"sampling": info["sampling"], "separate_task": stage,
            "last_encoder_layers": LAST_ENCODER_LAYERS, "checkpoint_selection": "validation_macro_f1_then_mean_recall_then_min_recall_then_nll",
            "semantic_input": "canonical text only; session eligibility remains deterministic"})
        v3.atomic_json(config_path, cfg)
        before = probe_outputs(agent, records["test"], stage)
        legacy.release(agent); del agent; agent = None; gc.collect()
        agent = laya.load(str(directory), device=DEVICE)
        after = probe_outputs(agent, records["test"], stage)
        assert [r["action"] for r in before] == [r["action"] for r in after], "Reload changed actions"
        assert all(abs(a["confidence"] - b["confidence"]) <= .005 for a, b in zip(before, after, strict=True)), "Reload changed probabilities"
        qualified = not SMOKE_TEST and all(g["passed"] for g in gates.values())
        report = {"stage": stage, "smoke_only": SMOKE_TEST, "qualified_on_synthetic_holdouts": qualified,
            "semantic_enabled_in_bot": False, "policy_is_rule_based": True, "data_manifest": DATA_MANIFEST,
            "source_sha256": SNAPSHOT["source_sha256"], "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
            "epochs": EPOCHS, "draws_per_epoch": draws, "batch_size": BATCH_SIZE, "accumulation": ACCUMULATION,
            "head_lr": HEAD_LR, "encoder_lr": ENCODER_LR, "seed": SEED,
            "min_recall": MIN_RECALL, "min_macro_f1": MIN_MACRO_F1,
            "training": info, "temperatures": temperatures, "metrics": metrics, "gates": gates,
            "checkpoint_roundtrip": "passed", "roundtrip_probes": len(before), "confidence_tolerance": .005,
            "policy_fixture_check": policy_report, "split_summary": split_summary,
            "environment": {k: importlib.metadata.version(k) for k in ("torch", "laya", "transformers", "huggingface-hub")},
            "limitation": "Synthetic/automatic development holdouts; no independently reviewed real-user accuracy"}
        v3.atomic_json(directory / "training_report.json", report)
        errors = [r for r in predictions["test"] if not r["correct"]]
        v3.atomic_json(WORKDIR / "diagnostics" / (stage + "-test-errors.json"), errors)
        report["export_archive"] = package_candidate(directory, model_name, qualified)
        diagnostics_export(WORKDIR)
        print(stage, "QUALIFIED:", qualified, "UPDATES:", info["optimizer_updates"], "SKIPPED:", info["skipped_updates"], flush=True)
        return report
    except Exception as error:
        v3.atomic_json(WORKDIR / "failures" / (stage + ".json"),
            {"stage": stage, "error": type(error).__name__, "message": str(error), "qualified": False,
             "recovery": "Resume needs the resume/*.pt state AND candidates/ best model with identical parameters"})
        if (directory / "model.safetensors").exists(): package_candidate(directory, model_name, False)
        diagnostics_export(WORKDIR)
        raise
    finally:
        if agent is not None: legacy.release(agent)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
