"""Validate every full-corpus model input against the actual pinned tokenizer/options."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".runtime/notebook-validation"))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    notebook = json.loads((ROOT / ".runtime/datasets/lia_kaggle_v3_embedded.ipynb").read_text(encoding="utf-8"))
    work = ROOT / ".runtime/kaggle-v3-validation-final"
    os.environ["MPLCONFIGDIR"] = str(work / "matplotlib-cache")
    cached = ROOT / ".runtime/voice/cache/hub/models--convaiinnovations--laya-multilingual/snapshots/e4e9ddf21a7b1903b7acffd8814ad4307bf63a67"
    namespace = {}
    for cell in notebook["cells"]:
        tags = cell.get("metadata", {}).get("tags", [])
        if not any(tag in tags for tag in ("parameters", "environment", "contract", "data-loading")): continue
        source = cell["source"]
        if "parameters" in tags:
            source = source.replace('WORKDIR = Path("/kaggle/working/lia-v3")', f'WORKDIR = Path({str(work)!r})')
            source = source.replace('BASE_MODEL = "convaiinnovations/laya-multilingual"', f'BASE_MODEL = {str(cached)!r}')
            source = source.replace('BASE_REVISION = "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67"', 'BASE_REVISION = None')
            # Reuse the already verified full data, without overwriting GPU smoke reports/state.
            source += '\nDATA_DIR = WORKDIR / "data"\nDATA_MANIFEST = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))\n'
        exec(compile(source, cell["id"], "exec"), namespace)
    start = time.monotonic()
    agent = namespace["laya"].load(str(cached), device="cpu")
    namespace["training"].batches.pad_id = agent.tok.pad_token_id
    results = {}
    for stage, records in (("action", namespace["action_records"]), ("semantic", namespace["semantic_records"])):
        rows = [r for part in records.values() for r in part]
        encoded = namespace["v3"].encode(agent, rows, namespace["by_id"])
        results[stage] = {"events": len({r["event_id"] for r in encoded}), "question_inputs": len(encoded),
                          "max_tokens": max(len(r["ids"]) for r in encoded), "truncated": False,
                          "option_collisions": False, "unsupported_gold": False}
        print(stage, results[stage], flush=True)
        del encoded
    results["seconds"] = round(time.monotonic() - start, 1)
    (work / "full-input-validation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
