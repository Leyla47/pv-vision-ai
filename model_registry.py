"""Eğitilmiş model ağırlıklarını güvenli biçimde yayımlama ve tanımlama araçları."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import yaml

from config import MODEL_METADATA_PATH, MODEL_WEIGHTS_PATH, PROJECT_ROOT


METRIC_COLUMNS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}
VALID_MODEL_STAGES = {"smoke", "interim", "candidate", "final"}


def publish_run_best(
    run_dir: Path,
    *,
    stage: str,
    destination: Path = MODEL_WEIGHTS_PATH,
    metadata_path: Path = MODEL_METADATA_PATH,
    target_epochs: int | None = None,
    evaluation_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bir eğitim çalışmasının best.pt dosyasını atomik olarak merkezi konuma yayımlar."""
    if stage not in VALID_MODEL_STAGES:
        raise ValueError(f"Geçersiz model aşaması: {stage}")

    run_dir = run_dir.resolve()
    source = run_dir / "weights" / "best.pt"
    if not source.exists():
        raise FileNotFoundError(f"Eğitim çıktısında best.pt bulunamadı: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=destination.parent,
        prefix=".pv-vision-weights-",
        suffix=".pt",
        delete=False,
    ) as temporary_file:
        temporary_weights = Path(temporary_file.name)

    try:
        shutil.copy2(source, temporary_weights)
        temporary_weights.replace(destination)
    finally:
        temporary_weights.unlink(missing_ok=True)

    metadata = build_model_metadata(
        run_dir,
        model_path=destination,
        stage=stage,
        target_epochs=target_epochs,
        evaluation_metadata=evaluation_metadata,
    )
    _write_json_atomic(metadata_path, metadata)
    return metadata


def build_model_metadata(
    run_dir: Path,
    *,
    model_path: Path,
    stage: str,
    target_epochs: int | None = None,
    evaluation_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Eğitim sonuçlarından uygulamanın göstereceği model kimlik bilgisini üretir."""
    args = _read_yaml(run_dir / "args.yaml")
    rows = _read_result_rows(run_dir / "results.csv")
    best_row = _best_result_row(rows)

    completed_epochs = max((_row_epoch(row) for row in rows), default=0)
    best_epoch = _row_epoch(best_row) if best_row else None
    configured_target = _safe_int(args.get("epochs"))
    resolved_target = target_epochs if target_epochs is not None else configured_target

    metrics = {
        key: _safe_float(best_row.get(column)) if best_row else None
        for key, column in METRIC_COLUMNS.items()
    }

    metadata = {
        "schema_version": 1,
        "stage": stage,
        "source_run": run_dir.name,
        "model_name": _resolve_model_name(args, run_dir),
        "weights_path": _project_relative(model_path),
        "sha256": _sha256(model_path),
        "published_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "completed_epochs": completed_epochs,
        "target_epochs": resolved_target,
        "best_epoch": best_epoch,
        "metrics": metrics,
    }
    if evaluation_metadata is not None:
        metadata["evaluation"] = evaluation_metadata
    return metadata


def load_model_metadata(metadata_path: Path = MODEL_METADATA_PATH) -> dict[str, Any] | None:
    """Model kimlik bilgisini okur; dosya yoksa None döndürür."""
    if not metadata_path.exists():
        return None

    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Model bilgi dosyası okunamadı: {metadata_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Model bilgi dosyası nesne biçiminde değil: {metadata_path}")
    return payload


def model_sha256(path: Path) -> str:
    """Model dosyasının SHA-256 kimliğini döndürür."""
    return _sha256(path)


def _read_result_rows(results_path: Path) -> list[dict[str, str]]:
    if not results_path.exists():
        return []
    with results_path.open(encoding="utf-8", newline="") as results_file:
        return list(csv.DictReader(results_file))


def _best_result_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    scored_rows = [row for row in rows if _row_fitness(row) is not None]
    if not scored_rows:
        return rows[-1] if rows else None
    return max(scored_rows, key=lambda row: _row_fitness(row) or float("-inf"))


def _row_fitness(row: dict[str, str]) -> float | None:
    """Ultralytics DetectMetrics ile aynı mAP ağırlıklı fitness değerini hesaplar."""
    map50 = _safe_float(row.get(METRIC_COLUMNS["map50"]))
    map50_95 = _safe_float(row.get(METRIC_COLUMNS["map50_95"]))
    if map50 is None or map50_95 is None:
        return None
    return 0.1 * map50 + 0.9 * map50_95


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _row_epoch(row: dict[str, str] | None) -> int:
    if not row:
        return 0
    return _safe_int(row.get("epoch")) or 0


def _safe_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _safe_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return round(float(str(value).strip()), 6)
    except (TypeError, ValueError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _resolve_model_name(args: dict[str, Any], run_dir: Path) -> str:
    candidates = (run_dir.name, str(args.get("model", "")))
    for candidate in candidates:
        match = re.search(r"(yolov\d+[nslmx])", candidate.lower())
        if match:
            return match.group(1)
    return Path(str(args.get("model", "YOLO"))).name


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=".pv-vision-model-info-",
        suffix=".json",
        delete=False,
    ) as temporary_file:
        json.dump(payload, temporary_file, ensure_ascii=False, indent=2)
        temporary_file.write("\n")
        temporary_path = Path(temporary_file.name)

    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
