"""Validate every input, then GPU paired backward, actual resume and finish-only.

No network install, no full quality training, no deployable smoke weights.
Executed QA notebook and figures are preserved apart from the clean handoff.
"""
import ast
import base64
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".runtime/notebook-validation"))


def full_calibration_check(work, ns):
    """Exercise fitted, nonempty temperature maps, not smoke's identity shortcut."""
    agent = ns["laya"].load(str(work / "candidates/laya-muz-bot-controls-v4-SMOKE-ONLY"), device=ns["DEVICE"])
    if ns["DEVICE"] == "cuda": agent.dtype = ns["AMP_DTYPE"]
    ns["training"].batches.pad_id = agent.tok.pad_token_id
    ns["legacy"].install_temperatures(agent, {"temperature": [1., 1., 1.], "temperature_by_options": {}})
    rows = [r for r in ns["commands"] if r["declared_split"] == "calibration"]
    records = ns["training"].runtime_records(rows, ns["worker"])
    items = ns["v4"].encode(agent, records, ns["by_id"])
    temps = ns["v3"].calibrate(ns["training"].raw_logits(agent, items, 1))
    assert temps["temperature_by_options"] and all(.5 <= t <= 5 for t in temps["temperature_by_options"].values())
    ns["legacy"].install_temperatures(agent, temps)
    test_rows = ns["action_splits"]["test"]
    test_items = ns["v4"].encode(agent, ns["training"].runtime_records(test_rows, ns["worker"]), ns["by_id"])
    _, predictions = ns["training"].evaluate_logits(ns["training"].raw_logits(agent, test_items, 1), temps)
    before = ns["production_parity"](agent, predictions, test_rows)
    destination = work / "calibration-check/candidate-SMOKE-ONLY"
    ns["v4"].atomic_checkpoint(agent, destination, "v4-temperature-SMOKE-ONLY", temps)
    ns["legacy"].release(agent); del agent; ns["gc"].collect()
    agent = ns["laya"].load(str(destination), device=ns["DEVICE"])
    if ns["DEVICE"] == "cuda": agent.dtype = ns["AMP_DTYPE"]
    for probe in before:
        actual = ns["worker"].classify(agent, ns["by_id"][probe["event_id"]]["canonical"])
        assert actual["action"] == probe["action"]
        assert abs(actual["confidence"] - probe["confidence"]) <= .005
    ns["legacy"].release(agent); del agent; ns["gc"].collect()
    result = {"full_calibration_inputs": len(items), "fitted_temperatures": temps, "worker_and_reload_with_fitted_temperatures": "passed"}
    (work / "calibration-check/validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    mode = "fp16" if "--fp16" in sys.argv else "auto"
    work = ROOT / (".runtime/kaggle-v4-qa-" + mode)
    if "--workdir" in sys.argv:
        work = Path(sys.argv[sys.argv.index("--workdir") + 1]).resolve()
        work.relative_to((ROOT / ".runtime").resolve())
    os.environ["MPLCONFIGDIR"] = str(work / "matplotlib-cache")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    import nbformat
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    notebook = nbformat.read(ROOT / ".runtime/datasets/lia_action_v4_embedded.ipynb", as_version=4)
    nbformat.validate(notebook)
    work.mkdir(parents=True, exist_ok=True)
    cache = ROOT / ".runtime/voice/cache/hub/models--convaiinnovations--laya-multilingual/snapshots/e4e9ddf21a7b1903b7acffd8814ad4307bf63a67"
    ns, figures, new_figures = {}, [], []
    def save_plot():
        filename = work / f"figure-{len(figures) + 1}.png"
        plt.gcf().savefig(filename, dpi=120, bbox_inches="tight"); figures.append(str(filename)); plt.close()
        new_figures.append(filename)
    plt.show = save_plot
    start, execution = time.monotonic(), 0
    original_train, full_input_count = None, 0
    for cell in notebook.cells:
        if cell.cell_type != "code": continue
        source, tags = cell.source, cell.metadata.get("tags", [])
        ast.parse(source)
        if "install" in tags:
            cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text="Skipped package installation; existing pinned local packages, no network.\n")]
            continue
        if "parameters" in tags:
            source = source.replace('WORKDIR = output_root / "lia-action-v4"', f'WORKDIR = Path({str(work)!r})')
            source = source.replace('SMOKE_TEST = False', 'SMOKE_TEST = True')
            source = source.replace('BASE_MODEL = "convaiinnovations/laya-multilingual"', f'BASE_MODEL = {str(cache)!r}')
            source = source.replace('BASE_REVISION = PINNED_BASE_REVISION', 'BASE_REVISION = None')
            if mode == "fp16": source = source.replace('AMP_PRECISION = "auto"', 'AMP_PRECISION = "fp16"')
        print("EXECUTE", cell.id, tags, flush=True)
        capture = io.StringIO(); new_figures.clear()
        with contextlib.redirect_stdout(capture): exec(compile(source, cell.id, "exec"), ns)
        cell.source = source
        execution += 1; cell.execution_count = execution
        cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text=capture.getvalue()[-6000:])]
        for filename in new_figures:
            cell.outputs.append(nbformat.v4.new_output("display_data", data={"image/png": base64.b64encode(filename.read_bytes()).decode()}, metadata={}))
        if "contract" in tags:
            original_train = ns["v4"].train
            def limited_train(*args, **kwargs): return original_train(*args, **kwargs, _max_new_epochs=1)
            ns["v4"].train = limited_train
        if "data-loading" in tags:
            # Full corpus encoding checks before bounded smoke selection can hide a bad input.
            agent = ns["laya"].load(str(cache), device=ns["DEVICE"])
            if ns["DEVICE"] == "cuda": agent.dtype = ns["AMP_DTYPE"]
            full_records = ns["training"].runtime_records(ns["commands"], ns["worker"])
            full_items = ns["v4"].encode(agent, full_records, ns["by_id"])
            full_input_count = len(full_items)
            assert full_input_count == 2 * ns["DATA_MANIFEST"]["rows"]
            ns["legacy"].release(agent); del agent, full_items; ns["gc"].collect()
    assert ns["report"]["smoke_only"] and ns["report"]["checkpoint_roundtrip"] == "passed"
    assert ns["report"]["runtime_batch_worker_parity"] == "passed"
    assert ns["report"]["training"]["trainable_encoder_parameters"] > 0
    assert ns["report"]["training"]["optimizer_updates"] > 0 and ns["report"]["training"]["skipped_updates"] == 0
    ns["v4"].train = original_train
    before = ns["report"]["training"]["optimizer_updates"]
    if ns["report"]["training"]["resumed_from_epoch"] == 1:
        # Recover an interrupted QA attempt without repeating its first epoch.
        resumed = ns["report"]
        assert len(resumed["training"]["history"]) == 2
        assert resumed["training"]["history"][1]["optimizer_updates"] > 0
    else:
        resumed = ns["run_action"]()
        assert resumed["training"]["resumed_from_epoch"] == 1
        assert resumed["training"]["optimizer_updates"] > before
    assert resumed["training"]["skipped_updates"] == 0
    # A completed run must use the best checkpoint without any further training.
    ns["FINISH_ONLY"] = True
    recovered = ns["run_action"]()
    assert recovered["training"]["optimizer_updates"] == resumed["training"]["optimizer_updates"]
    assert recovered["training"]["recovery"] == "final checks only, no epochs or optimizer restore"
    assert recovered["checkpoint_roundtrip"] == "passed"
    assert not list(work.glob("laya-muz-bot-*.zip")), "Smoke must not export deployable weights"
    ns["report"] = recovered
    for cell in notebook.cells:
        if "train-action" in cell.metadata.get("tags", []):
            cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text="QA: actual second-epoch resume and finish-only both passed; no extra epochs during finish-only.\n"))
        if "plots" in cell.metadata.get("tags", []) or "diagnostics" in cell.metadata.get("tags", []):
            capture = io.StringIO(); new_figures.clear()
            with contextlib.redirect_stdout(capture): exec(compile(cell.source, cell.id, "exec"), ns)
            cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text=capture.getvalue()[-6000:])]
            for filename in new_figures:
                cell.outputs.append(nbformat.v4.new_output("display_data", data={"image/png": base64.b64encode(filename.read_bytes()).decode()}, metadata={}))
    finish = (ROOT / "scripts/kaggle_v4_finish.py").read_text(encoding="utf-8")
    finish = finish.replace('default_root / "lia-action-v4"', f'Path({str(work)!r})')
    exec(compile(finish, "empty-training-globals", "exec"), {})
    calibration_result = full_calibration_check(work, ns)
    nbformat.validate(notebook); nbformat.write(notebook, work / "executed-smoke.ipynb")
    result = {"schema_and_all_code_syntax": "passed", "full_inputs_encoded": full_input_count,
        "all_gold_labels_and_no_truncation": "passed", "GPU_paired_backward": "passed", "amp_dtype": str(ns["AMP_DTYPE"]),
        "real_epoch_resume": "passed", "finish_only_no_retraining": "passed", "worker_and_batch_parity": "passed",
        "checkpoint_reload": "passed", "optimizer_updates": recovered["training"]["optimizer_updates"],
        "skipped_updates": recovered["training"]["skipped_updates"], "empty_training_globals_final_cell": "passed",
        "full_quality_training_executed": False, "install_cell": "skipped, existing pinned dependencies", "figures": figures,
        "seconds": round(time.monotonic() - start, 1), "full_calibration": calibration_result}
    (work / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__": main()
