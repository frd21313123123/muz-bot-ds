"""Package the delivered private dataset inside a single portable notebook."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / ".runtime/datasets/lia-starter-v1"
OUTPUT = DATA.parent / "lia_training_with_embedded_dataset.ipynb"


def main():
    notebook = json.loads((DATA / "lia_dataset_training_colab_kaggle.ipynb").read_text(encoding="utf-8"))
    template = json.loads((ROOT / "notebooks/laya_training_colab_kaggle.ipynb").read_text(encoding="utf-8"))
    export_cell = next(c for c in template["cells"] if c["cell_type"] == "code" and "training.save_checkpoint(agent, export_dir" in c["source"])
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code" and "training.save_checkpoint(agent, export_dir" in cell["source"]:
            notebook["cells"][index] = export_cell
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(DATA.iterdir()):
            if path.is_file() and path.suffix != ".ipynb":
                archive.write(path, path.name)
    payload = buffer.getvalue()
    encoded = base64.b64encode(payload).decode("ascii")
    chunks = "\n".join(repr(encoded[i:i + 120]) for i in range(0, len(encoded), 120))
    code = f'''# Датасет встроен в этот файл; отдельная загрузка не требуется.
import base64
import hashlib
import io
import json
import zipfile

EMBEDDED_DATASET_BASE64 = (
{chunks}
)
embedded_bytes = base64.b64decode(EMBEDDED_DATASET_BASE64, validate=True)
assert hashlib.sha256(embedded_bytes).hexdigest() == {hashlib.sha256(payload).hexdigest()!r}, "Dataset checksum mismatch"
EMBEDDED_DATA_DIR = WORKDIR / "embedded-dataset"
EMBEDDED_DATA_DIR.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(embedded_bytes)) as archive:
    assert archive.testzip() is None, "Dataset archive is corrupted"
    for member in archive.infolist():
        (EMBEDDED_DATA_DIR / member.filename).resolve().relative_to(EMBEDDED_DATA_DIR.resolve())
    archive.extractall(EMBEDDED_DATA_DIR)
embedded_manifest = json.loads((EMBEDDED_DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
for filename, expected in embedded_manifest["file_sha256"].items():
    assert hashlib.sha256((EMBEDDED_DATA_DIR / filename).read_bytes()).hexdigest() == expected, filename
DATA_PATHS = [str(EMBEDDED_DATA_DIR / name) for name in ("commands-synthetic.jsonl", "bot-commands-labeled.jsonl")]
PLANNER_PATHS = [str(EMBEDDED_DATA_DIR / "planner-synthetic.jsonl")] if AUTO_LOAD_PLANNER else []
print("Встроенный датасет готов: 3560 команд, 10640 planner-сценариев, 44 записи бота (31 для обучения, 13 для проверки).")
'''
    ast.parse(code)
    for cell in notebook["cells"]:
        if cell["id"] == "dataset-upload":
            cell.update(id="dataset-embedded", source=code,
                        metadata={"tags": ["dataset-embedded"], "jupyter": {"source_hidden": True}})
        elif cell["id"] == "dataset-upload-description":
            cell.update(id="dataset-embedded-description", source="### Встроенный датасет\n\nСледующая ячейка распакует данные из самого ноутбука и проверит SHA-256. Все 44 записи бота сохранены; 13 неоднозначных записей исключены из обучения. Автоматическая разметка требует проверки человеком.\n")
    notebook["cells"][0]["source"] = """# LIA / Laya: обучение со встроенным датасетом

Откройте этот единственный файл в Google Colab или Kaggle, включите GPU и выполните все ячейки. Датасет уже встроен: ZIP и JSONL загружать не нужно. Интернет нужен для установки библиотек и скачивания базовой модели.

Внутри: 3560 синтетических команд, 10640 контекстных сценариев и все 44 записи бота. 31 запись имеет автоматическую разметку и участвует в обучении; 13 неоднозначных записей сохранены для ручной проверки. Метки не подтверждены человеком. Тестовая выборка синтетическая, поэтому её метрики не показывают точность на реальных пользователях.

Обучение action совместимо с текущим ботом. Задачи should_respond / next_tool / intent подготовлены для дальнейшего подключения. Чтобы обучать только action, установите AUTO_LOAD_PLANNER=False в параметрах.

Ноутбук содержит реальные распознанные реплики из логов: учитывайте это при публикации файла или предоставлении доступа.
"""
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(OUTPUT)
    print(f"Notebook bytes: {OUTPUT.stat().st_size}; embedded archive bytes: {len(payload)}")


if __name__ == "__main__":
    main()
