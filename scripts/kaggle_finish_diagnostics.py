"""Paste into the final Kaggle cell; restore completed reports without training."""
from pathlib import Path
import json
import shutil

WORKDIR = Path(globals().get("WORKDIR", "/kaggle/working/lia-v2"))
reports = globals().get("reports")
if not isinstance(reports, dict) or not reports:
    reports = {}
    saved_reports = sorted((WORKDIR / "candidates").glob("*/training_report.json"),
                           key=lambda path: path.stat().st_mtime)
    for path in saved_reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        stage = report.get("stage")
        if stage in {"action", "planner"}:
            reports[stage] = report
            print("Read completed report:", path)
if not reports:
    raise RuntimeError(
        "Нет завершённых отчётов обучения в " + str(WORKDIR / "candidates") + ". "
        "Проверьте вывод ячеек 7–8. Если обучение ещё не запускалось, выполните Run All с первой ячейки. "
        "Если среда сбросилась и файлы исчезли, нужен сохранённый Kaggle Output или новый запуск обучения."
    )
diagnostics = WORKDIR / "diagnostics"
diagnostics.mkdir(parents=True, exist_ok=True)
for stage, report in reports.items():
    if stage not in {"action", "planner"}:
        raise ValueError("Unexpected report stage")
    (diagnostics / (stage + "-report.json")).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(stage, "qualified:", report.get("qualified_on_synthetic_holdouts", False),
          "smoke:", report.get("smoke_only", False))
diagnostics_zip = shutil.make_archive(str(WORKDIR / "lia-v2-diagnostics"), "zip", root_dir=diagnostics)
print("Diagnostics:", diagnostics_zip)
print("Existing model ZIPs:", [p.name for p in WORKDIR.glob("laya-muz-bot-*.zip")])
if any(report.get("smoke_only", False) for report in reports.values()):
    print("SMOKE ONLY: пробные веса нельзя подключать к боту")
