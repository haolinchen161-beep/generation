# -*- coding: utf-8 -*-
"""创新点一 v3 B-Rep 结构基元图抽取的通用 IO 工具。"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


NODE_TYPES = [
    "face_group",
    "sheet_region",
    "thin_wall_pair",
    "loop_or_hole",
    "transition_group",
    "repeated_feature",
    "boundary_group",
]

RELATION_TYPES = [
    "adjacent_to",
    "parallel_to",
    "opposite_to",
    "orthogonal_to",
    "coplanar_with",
    "smooth_connected",
    "embedded_in",
    "repeated_with",
    "bounded_by",
    "thin_wall_pair",
    "has_member",
    "hosted_by",
]

REQUIRED_PARSED_FIELDS = [
    "uid",
    "face_count",
    "edge_count",
    "vertex_count",
    "face_wcs",
    "edge_wcs",
    "vert_wcs",
    "face_bbox_wcs",
    "edge_bbox_wcs",
    "edgeFace_adj",
    "edgeVert_adj",
    "faceEdge_adj",
    "global_bbox",
    "parser_backend",
    "geometry_sampling_quality",
]


def normalize_path(path: os.PathLike[str] | str) -> str:
    return str(Path(path).resolve()).replace("\\", "/")


def ensure_dir(path: os.PathLike[str] | str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return normalize_path(path)


def ensure_workdir(workdir: os.PathLike[str] | str) -> Dict[str, str]:
    root = Path(workdir)
    dirs = {
        "root": root,
        "outputs": root / "outputs",
        "parsed": root / "outputs" / "parsed",
        "dataset": root / "outputs" / "dataset",
        "reports": root / "outputs" / "reports",
        "visualizations": root / "outputs" / "visualizations",
        "examples": root / "outputs" / "examples",
    }
    return {key: ensure_dir(value) for key, value in dirs.items()}


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: os.PathLike[str] | str, data: Dict[str, Any]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(data), f, ensure_ascii=False, indent=2)


def read_json(path: os.PathLike[str] | str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: os.PathLike[str] | str, records: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(json_safe(rec), ensure_ascii=False) + "\n")


def read_jsonl(path: os.PathLike[str] | str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not Path(path).exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_csv(path: os.PathLike[str] | str, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    ensure_dir(Path(path).parent)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            safe_row = json_safe(row)
            writer.writerow({key: safe_row.get(key, "") for key in fieldnames})


def read_csv(path: os.PathLike[str] | str) -> List[Dict[str, str]]:
    if not Path(path).exists():
        return []
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_text(path: os.PathLike[str] | str, lines: Sequence[str] | str) -> None:
    ensure_dir(Path(path).parent)
    text = lines if isinstance(lines, str) else "\n".join(lines)
    # 报告主要在 Windows 环境查看，使用 BOM 让常见编辑器自动识别中文编码。
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write(text)


def write_pickle(path: os.PathLike[str] | str, data: Dict[str, Any]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def read_pickle(path: os.PathLike[str] | str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_dataset_split(path: os.PathLike[str] | str) -> Tuple[Dict[str, str], Dict[str, int]]:
    """Load a trusted DTG/DeepCAD split and canonicalize labels to train/val/test."""
    split_path = Path(path)
    if not split_path.is_file():
        raise FileNotFoundError(f"dataset split is missing: {split_path}")
    if split_path.suffix.lower() == ".pkl":
        with split_path.open("rb") as handle:
            payload = pickle.load(handle)
    elif split_path.suffix.lower() == ".json":
        with split_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        raise ValueError(f"unsupported split format: {split_path.suffix}")
    if not isinstance(payload, dict):
        raise ValueError("dataset split must be a dictionary")

    split_lists: Dict[str, List[Any]] = {"train": [], "val": [], "test": []}
    for source_key, target_key in {
        "train": "train",
        "val": "val",
        "validation": "val",
        "test": "test",
    }.items():
        values = payload.get(source_key)
        if values is None:
            continue
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"split field {source_key!r} must be a list")
        split_lists[target_key].extend(values)
    if any(not split_lists[key] for key in ("train", "val", "test")):
        raise ValueError("dataset split must contain non-empty train, val/validation and test lists")

    uid_to_split: Dict[str, str] = {}
    counts: Dict[str, int] = {}
    for split_name in ("train", "val", "test"):
        counts[split_name] = len(split_lists[split_name])
        for item in split_lists[split_name]:
            uid = Path(str(item).replace("\\", "/")).stem
            if uid in uid_to_split:
                raise ValueError(f"duplicate UID across dataset splits: {uid}")
            uid_to_split[uid] = split_name
    return uid_to_split, counts


def scan_step_files(step_root: os.PathLike[str] | str, limit: int = 0) -> List[str]:
    root = Path(step_root)
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".step", ".stp"}
    )
    if limit and limit > 0:
        files = files[: int(limit)]
    return [normalize_path(path) for path in files]


def make_uid(step_path: os.PathLike[str] | str, step_root: os.PathLike[str] | str | None = None) -> str:
    path = Path(step_path)
    if step_root is not None:
        try:
            raw = path.with_suffix("").relative_to(Path(step_root)).as_posix()
        except Exception:
            raw = path.with_suffix("").as_posix()
    else:
        raw = path.with_suffix("").as_posix()
    uid = re.sub(r"[^0-9A-Za-z._-]+", "_", raw).strip("_")
    if len(uid) > 120:
        digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
        uid = f"{uid[:96]}_{digest}"
    return uid or hashlib.md5(str(path).encode("utf-8")).hexdigest()


def summarize_numeric(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    if not vals:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": round(min(vals), 6),
        "mean": round(sum(vals) / len(vals), 6),
        "max": round(max(vals), 6),
    }


def count_by_type(records: Sequence[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for rec in records:
        typ = str(rec.get(key, "unknown"))
        counts[typ] = counts.get(typ, 0) + 1
    return counts


def copy_text_file(src: os.PathLike[str] | str, dst: os.PathLike[str] | str) -> None:
    path = Path(src)
    if path.exists():
        write_text(dst, path.read_text(encoding="utf-8", errors="ignore"))
