"""路线2适配器训练数据集。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from route2_adapter_models import vectorize_s_faces, vectorize_s_prior
from utils_io import load_pickle


def _read_prior_map(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row["uid"]] = row
    return rows


def _read_manifest(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["uid"]: row for row in csv.DictReader(f)}


def _read_uids(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _adjacency_classes(edge_face_adj: np.ndarray, max_faces: int, edge_classes: int) -> np.ndarray:
    mat = np.zeros((max_faces, max_faces), dtype=np.int64)
    for pair in np.asarray(edge_face_adj, dtype=np.int64):
        a, b = int(pair[0]), int(pair[1])
        if 0 <= a < max_faces and 0 <= b < max_faces and a != b:
            mat[a, b] += 1
            mat[b, a] += 1
    return np.clip(mat, 0, edge_classes - 1)


class Route2MacroDataset(Dataset):
    """S prior 到 DTG 宏观布局目标的数据集。"""

    def __init__(
        self,
        dataset_dir: Path,
        split: str,
        max_faces: int = 30,
        max_edges: int = 108,
        max_vertices: int = 186,
        edge_classes: int = 5,
        bbox_scaled: float = 3.0,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.max_faces = int(max_faces)
        self.max_edges = int(max_edges)
        self.max_vertices = int(max_vertices)
        self.edge_classes = int(edge_classes)
        self.bbox_scaled = float(bbox_scaled)
        self.prior_map = _read_prior_map(self.dataset_dir / "motif_prior_index_ready.jsonl")
        self.manifest = _read_manifest(self.dataset_dir / "manifest.csv")
        self.uids = _read_uids(self.dataset_dir / "splits" / f"{split}_uids.txt")

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        uid = self.uids[idx]
        prior = self.prior_map[uid]
        manifest_row = self.manifest[uid]
        local_pkl = Path(manifest_row["local_pkl"])
        if not local_pkl.is_absolute():
            local_pkl = self.dataset_dir.parents[1] / local_pkl
        parsed = load_pickle(local_pkl)

        face_bbox = np.asarray(parsed["face_bbox_wcs"], dtype=np.float32) * self.bbox_scaled
        nf = min(len(face_bbox), self.max_faces)
        bbox_target = np.zeros((self.max_faces, 6), dtype=np.float32)
        bbox_target[:nf] = face_bbox[:nf]
        face_mask = np.zeros((self.max_faces,), dtype=np.float32)
        face_mask[:nf] = 1.0
        adj_target = _adjacency_classes(parsed["edgeFace_adj"], self.max_faces, self.edge_classes)
        s_vec = vectorize_s_prior(prior, self.max_faces, self.max_edges, self.max_vertices)
        face_features = vectorize_s_faces(prior, self.max_faces)

        return {
            "s_vec": torch.from_numpy(s_vec),
            "face_features": torch.from_numpy(face_features),
            "bbox": torch.from_numpy(bbox_target),
            "face_mask": torch.from_numpy(face_mask),
            "adj": torch.from_numpy(adj_target),
            "num_faces": torch.tensor(nf, dtype=torch.long),
        }
