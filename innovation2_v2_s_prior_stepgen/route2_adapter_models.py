"""路线2：S-conditioned face-aware 宏观生成适配器。

核心改动：
1. 保留全局 S 统计向量；
2. 从 S 中的 motif_nodes / face_ids 构造逐 face 条件 token；
3. 用 face-aware Transformer 预测 face bbox 与 face-edge topology。
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
from torch import nn


NODE_TYPES = [
    "sheet_like_group",
    "thin_wall_pair",
    "loop_or_hole",
    "transition_group",
    "repeated_feature",
    "boundary_group",
]

RELATION_TYPES = [
    "parallel_to",
    "opposite_to",
    "orthogonal_to",
    "coplanar_with",
    "repeated_with",
    "bounded_by",
]


def s_feature_dim() -> int:
    return 5 + len(NODE_TYPES) + len(RELATION_TYPES) + 8


def face_feature_dim() -> int:
    return 3 + 2 * len(NODE_TYPES) + 6 + 3 + 3 + 5


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float32)
    return float(arr.std())


def vectorize_s_prior(row: Dict[str, Any], max_faces: int = 30, max_edges: int = 108, max_vertices: int = 186) -> np.ndarray:
    """把一个 S prior 压成全局固定维向量。"""
    nodes = row.get("motif_nodes", [])
    relations = row.get("motif_relations", [])
    node_counts = Counter(node.get("type", "unknown") for node in nodes)
    relation_counts = Counter(rel.get("type", "unknown") for rel in relations)

    values: List[float] = [
        _safe_float(row.get("num_faces")) / max(max_faces, 1),
        _safe_float(row.get("num_edges")) / max(max_edges, 1),
        _safe_float(row.get("num_vertices")) / max(max_vertices, 1),
        len(nodes) / max(max_faces, 1),
        len(relations) / 60.0,
    ]
    values += [node_counts.get(name, 0) / max(max_faces, 1) for name in NODE_TYPES]
    values += [relation_counts.get(name, 0) / 60.0 for name in RELATION_TYPES]

    confidences = [_safe_float(node.get("confidence")) for node in nodes]
    rel_confidences = [_safe_float(rel.get("confidence")) for rel in relations]
    rel_areas = []
    thinness = []
    aspect = []
    for node in nodes:
        features = node.get("features", {})
        rel_areas.append(_safe_float(features.get("relative_area_sum")))
        thinness.append(_safe_float(features.get("bbox_thinness")))
        aspect.append(_safe_float(features.get("mean_aspect_ratio")))
    values += [
        _mean(confidences),
        _std(confidences),
        _mean(rel_confidences),
        _std(rel_confidences),
        _mean(rel_areas),
        _std(rel_areas),
        _mean(thinness),
        min(_mean(aspect) / 30.0, 1.0),
    ]
    return np.asarray(values, dtype=np.float32)


def vectorize_s_faces(row: Dict[str, Any], max_faces: int = 30) -> np.ndarray:
    """把 S prior 展开为逐 face 条件 token。"""
    node_type_to_idx = {name: idx for idx, name in enumerate(NODE_TYPES)}
    num_faces = int(row.get("num_faces", 0) or 0)
    dim = face_feature_dim()
    out = np.zeros((max_faces, dim), dtype=np.float32)
    count = np.zeros((max_faces, 1), dtype=np.float32)

    for face_id in range(min(num_faces, max_faces)):
        out[face_id, 0] = face_id / max(max_faces - 1, 1)
        out[face_id, 1] = 1.0

    type_count_start = 3
    type_conf_start = type_count_start + len(NODE_TYPES)
    bbox_start = type_conf_start + len(NODE_TYPES)
    centroid_start = bbox_start + 6
    normal_start = centroid_start + 3
    scalar_start = normal_start + 3

    for node in row.get("motif_nodes", []):
        node_type = node.get("type", "")
        type_idx = node_type_to_idx.get(node_type)
        confidence = _safe_float(node.get("confidence"))
        features = node.get("features", {})

        bbox = np.asarray(features.get("bbox", [0, 0, 0, 0, 0, 0]), dtype=np.float32).reshape(-1)
        if bbox.size != 6:
            bbox = np.zeros((6,), dtype=np.float32)
        centroid = np.asarray(features.get("centroid", [0, 0, 0]), dtype=np.float32).reshape(-1)
        if centroid.size != 3:
            centroid = np.zeros((3,), dtype=np.float32)
        normal = np.asarray(features.get("normal_proxy", [0, 0, 0]), dtype=np.float32).reshape(-1)
        if normal.size != 3:
            normal = np.zeros((3,), dtype=np.float32)
        scalars = np.asarray(
            [
                _safe_float(features.get("relative_area_sum")),
                _safe_float(features.get("bbox_thinness")),
                min(_safe_float(features.get("mean_aspect_ratio")) / 30.0, 1.0),
                min(_safe_float(features.get("mean_face_degree")) / 20.0, 1.0),
                1.0 if features.get("boundary_flag", False) else 0.0,
            ],
            dtype=np.float32,
        )

        for face_id in node.get("face_ids", []):
            try:
                face_id = int(face_id)
            except Exception:
                continue
            if not (0 <= face_id < max_faces):
                continue
            count[face_id, 0] += 1.0
            if type_idx is not None:
                out[face_id, type_count_start + type_idx] += 1.0
                out[face_id, type_conf_start + type_idx] += confidence
            out[face_id, bbox_start : bbox_start + 6] += bbox
            out[face_id, centroid_start : centroid_start + 3] += centroid
            out[face_id, normal_start : normal_start + 3] += normal
            out[face_id, scalar_start : scalar_start + 5] += scalars

    active = count[:, 0] > 0
    if np.any(active):
        out[active, bbox_start : bbox_start + 6] /= count[active]
        out[active, centroid_start : centroid_start + 3] /= count[active]
        out[active, normal_start : normal_start + 3] /= count[active]
        out[active, scalar_start : scalar_start + 5] /= count[active]
        out[active, type_count_start : type_count_start + len(NODE_TYPES)] /= np.maximum(count[active], 1.0)
        out[active, type_conf_start : type_conf_start + len(NODE_TYPES)] /= np.maximum(count[active], 1.0)
    out[:, 2] = np.minimum(count[:, 0] / 4.0, 1.0)
    return out.astype(np.float32)


class SRoute2MacroAdapter(nn.Module):
    """S -> face bbox + face-edge adjacency 的 face-aware 适配器。"""

    def __init__(
        self,
        s_dim: int,
        max_faces: int = 30,
        edge_classes: int = 5,
        hidden_dim: int = 512,
        face_dim: int | None = None,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.s_dim = int(s_dim)
        self.max_faces = int(max_faces)
        self.edge_classes = int(edge_classes)
        self.face_dim = int(face_dim or face_feature_dim())
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)

        self.global_trunk = nn.Sequential(
            nn.Linear(self.s_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.count_head = nn.Linear(hidden_dim, self.max_faces + 1)
        self.face_in = nn.Sequential(
            nn.Linear(self.face_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.face_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.face_pos = nn.Parameter(torch.randn(1, self.max_faces, hidden_dim) * 0.02)
        self.bbox_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6),
        )
        self.adj_pair_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, self.edge_classes),
        )

    def forward(self, s_vec: torch.Tensor, face_features: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        global_context = self.global_trunk(s_vec)
        count_logits = self.count_head(global_context)
        if face_features is None:
            face_features = torch.zeros(
                (s_vec.shape[0], self.max_faces, self.face_dim),
                dtype=s_vec.dtype,
                device=s_vec.device,
            )

        global_repeated = global_context.unsqueeze(1).expand(-1, self.max_faces, -1)
        face_tokens = self.face_in(torch.cat([face_features, global_repeated], dim=-1)) + self.face_pos
        face_tokens = self.face_encoder(face_tokens)
        bbox = self.bbox_head(face_tokens)

        left = face_tokens.unsqueeze(2).expand(-1, -1, self.max_faces, -1)
        right = face_tokens.unsqueeze(1).expand(-1, self.max_faces, -1, -1)
        pair = torch.cat([left, right, torch.abs(left - right), left * right], dim=-1)
        adj_logits = self.adj_pair_head(pair)
        adj_logits = 0.5 * (adj_logits + adj_logits.transpose(1, 2))
        diag = torch.arange(self.max_faces, device=s_vec.device)
        adj_logits[:, diag, diag, 1:] = -1e4
        adj_logits[:, diag, diag, 0] = 1e4
        return {"count_logits": count_logits, "bbox": bbox, "adj_logits": adj_logits}


def sort_minmax_bbox_tensor(bbox: torch.Tensor) -> torch.Tensor:
    """保证 bbox 为 [min_xyz, max_xyz] 格式。"""
    reshaped = bbox.view(*bbox.shape[:-1], 2, 3)
    mins = reshaped.min(dim=-2).values
    maxs = reshaped.max(dim=-2).values
    return torch.cat([mins, maxs], dim=-1)
