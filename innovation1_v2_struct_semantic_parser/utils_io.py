# -*- coding: utf-8 -*-
"""Shared IO and schema helpers for innovation1 v2."""

from __future__ import annotations

import csv
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


PARAMETER_KEYS = [
    "length",
    "width",
    "thickness",
    "height",
    "flange_width",
    "rib_width",
    "rib_height",
    "rib_count",
    "fillet_radius",
    "root_fillet_radius",
    "cutout_corner_radius",
    "hole_radius",
    "hole_width",
    "hole_height",
    "hole_count",
    "taper_ratio",
    "curvature_radius",
    "sweep_angle",
    "runout_length",
    "notch_depth",
    "offset_ratio",
    "height_start",
    "height_end",
    "flange_width_start",
    "flange_width_end",
    "cap_width",
]

NODE_TYPES = [
    "panel",
    "web",
    "flange",
    "stiffener",
    "transition",
    "boundary",
    "hole",
    "cutout",
    "runout",
]

RELATION_TYPES = [
    "attached_to",
    "connected_to",
    "smooth_connected",
    "symmetric_to",
    "opposite_side_of",
    "parallel_to",
    "hole_of",
    "cutout_of",
    "runout_of",
    "boundary_of",
]

FACE_ROLES = NODE_TYPES + ["unassigned"]

ENHANCED_PART_TYPES = [
    "panel_with_circular_cutout",
    "panel_with_rectangular_cutout",
    "stiffened_panel_with_cutout",
    "multi_stiffened_panel",
    "asymmetric_stiffened_panel",
    "tapered_c_channel",
    "tapered_hat_stiffener",
    "curved_panel",
    "curved_stiffened_panel",
    "stiffener_runout_panel",
]

MAX_DIM_LIMITS = {"max_faces": 30, "max_edges": 68, "max_vertices": 40}


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
        "enhanced_dataset": root / "outputs" / "enhanced_dataset",
        "enhanced_parsed": root / "outputs" / "enhanced_parsed",
        "inferred_semantics": root / "outputs" / "inferred_semantics",
        "reports": root / "outputs" / "reports",
        "logs": root / "outputs" / "logs",
    }
    return {key: ensure_dir(value) for key, value in dirs.items()}


def read_json(path: os.PathLike[str] | str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: os.PathLike[str] | str, data: Dict[str, Any]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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


def write_jsonl(path: os.PathLike[str] | str, records: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_text(path: os.PathLike[str] | str, lines: Sequence[str] | str) -> None:
    ensure_dir(Path(path).parent)
    text = lines if isinstance(lines, str) else "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_csv(path: os.PathLike[str] | str, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    ensure_dir(Path(path).parent)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def blank_parameters() -> Dict[str, float]:
    return {key: 0.0 for key in PARAMETER_KEYS}


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def summarize_numeric(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    vals = [float(v) for v in values]
    return {
        "min": round(min(vals), 4),
        "mean": round(sum(vals) / len(vals), 4),
        "max": round(max(vals), 4),
    }


def build_tensor_schema(part_types: Sequence[str] | None = None) -> Dict[str, Any]:
    part_types = list(part_types or ENHANCED_PART_TYPES)
    return {
        "version": "innovation1_v2_struct_semantic_parser",
        "part_type_to_id": {name: idx for idx, name in enumerate(part_types)},
        "id_to_part_type": {str(idx): name for idx, name in enumerate(part_types)},
        "node_type_to_id": {name: idx for idx, name in enumerate(NODE_TYPES)},
        "relation_type_to_id": {name: idx for idx, name in enumerate(RELATION_TYPES)},
        "face_role_to_id": {name: idx for idx, name in enumerate(FACE_ROLES)},
        "parameter_keys": list(PARAMETER_KEYS),
        "brep_required_fields": [
            "face_bbox_wcs",
            "face_surface_type",
            "face_curvature_proxy",
            "edge_bbox_wcs",
            "vert_wcs",
            "face_wcs",
            "edge_wcs",
            "edgeFace_adj",
            "edgeVert_adj",
            "faceEdge_adj",
            "face_count",
            "edge_count",
            "vertex_count",
        ],
        "innovation2_dim_limits": dict(MAX_DIM_LIMITS),
    }


def make_data_splits(uids: Sequence[str], seed: int = 42) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    shuffled = list(uids)
    rng.shuffle(shuffled)
    n = len(shuffled)
    train_end = int(round(0.70 * n))
    val_end = train_end + int(round(0.15 * n))
    rows: List[Dict[str, str]] = []
    for idx, uid in enumerate(shuffled):
        if idx < train_end:
            split = "train"
        elif idx < val_end:
            split = "val"
        else:
            split = "test"
        rows.append({"uid": uid, "split": split})
    return sorted(rows, key=lambda item: item["uid"])


def scan_uid_files(dataset_dir: os.PathLike[str] | str, suffix: str) -> List[str]:
    root = Path(dataset_dir)
    return sorted(path.stem for path in root.glob(f"*{suffix}"))


def count_by_key(records: Sequence[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for rec in records:
        val = str(rec.get(key, "unknown"))
        counts[val] = counts.get(val, 0) + 1
    return counts


def relation_key(rel: Dict[str, Any]) -> tuple[str, str, str]:
    src = str(rel.get("source", rel.get("src", "")))
    dst = str(rel.get("target", rel.get("dst", "")))
    typ = str(rel.get("type", rel.get("relation", "")))
    return src, dst, typ
