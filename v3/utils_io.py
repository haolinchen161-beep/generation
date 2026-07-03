# -*- coding: utf-8 -*-
"""Utility helpers for innovation1 v3 B-Rep motif graph extraction."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


OUTPUT_SUBDIRS = {
    "parsed_public": "outputs/parsed_public",
    "motif_graphs": "outputs/motif_graphs",
    "visualizations": "outputs/visualizations",
    "reports": "outputs/reports",
    "logs": "outputs/logs",
}


def ensure_workdir(workdir: str) -> Dict[str, str]:
    root = Path(workdir)
    dirs = {"root": str(root)}
    root.mkdir(parents=True, exist_ok=True)
    for key, rel in OUTPUT_SUBDIRS.items():
        path = root / rel
        path.mkdir(parents=True, exist_ok=True)
        dirs[key] = str(path)
    return dirs


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def sanitize_uid(text: str, max_len: int = 120) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_\-]+", "_", text).strip("_")
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned[:max_len] if cleaned else "sample"


def uid_from_step(step_path: str, source_root: str) -> str:
    rel = os.path.relpath(step_path, source_root)
    stem = os.path.splitext(rel)[0]
    digest = hashlib.sha1(rel.replace("\\", "/").encode("utf-8")).hexdigest()[:8]
    return f"{sanitize_uid(stem)}_{digest}"


def detect_source_name(path: str) -> str:
    lower = str(path).lower()
    if "abc" in lower:
        return "abc"
    if "deepcad" in lower:
        return "deepcad"
    return "public_brep"


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_text(path: str, lines: Sequence[str] | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) if isinstance(lines, (list, tuple)) else str(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def write_csv(path: str, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def scan_step_files(source_dir: str, max_files: int = 0) -> List[str]:
    root = Path(source_dir)
    files = sorted(str(path) for path in root.rglob("*.step"))
    files.extend(sorted(str(path) for path in root.rglob("*.stp")))
    files = sorted(set(files))
    if max_files and max_files > 0:
        return files[: int(max_files)]
    return files


def clean_dir(path: str, suffixes: Sequence[str]) -> None:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    for suffix in suffixes:
        for item in root.glob(suffix):
            if item.is_file():
                item.unlink()


def short_hash(values: Any) -> str:
    payload = json.dumps(values, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]

