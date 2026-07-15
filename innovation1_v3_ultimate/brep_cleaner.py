# -*- coding: utf-8 -*-
"""解析后 B-Rep 字典的字段规范化与清洗校验。"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def _as_float_array(value: Any, shape_tail: int | None = None) -> np.ndarray:
    if value is None:
        return np.zeros((0,), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if shape_tail is not None and arr.size > 0:
        arr = arr.reshape((-1, shape_tail))
    return arr


def _bbox_from_points(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.zeros(6, dtype=np.float32)
    pts = points.reshape(-1, 3)
    mn = np.min(pts, axis=0)
    mx = np.max(pts, axis=0)
    return np.concatenate([mn, mx]).astype(np.float32)


def _bbox_stack_from_wcs(wcs: Any) -> np.ndarray:
    arr = _as_float_array(wcs)
    if arr.size == 0:
        return np.zeros((0, 6), dtype=np.float32)
    return np.stack([_bbox_from_points(arr[i]) for i in range(arr.shape[0])], axis=0).astype(np.float32)


def _normalize_edge_face_adj(value: Any, edge_count: int | None = None) -> np.ndarray:
    if value is None:
        count = int(edge_count or 0)
        return -np.ones((count, 2), dtype=np.int64)
    rows: List[List[int]] = []
    for row in value:
        vals = [int(x) for x in np.asarray(row).reshape(-1).tolist() if int(x) >= 0]
        if len(vals) == 0:
            rows.append([-1, -1])
        elif len(vals) == 1:
            rows.append([vals[0], -1])
        else:
            rows.append(vals[:2])
    if edge_count is not None and len(rows) < edge_count:
        rows.extend([[-1, -1] for _ in range(edge_count - len(rows))])
    return np.asarray(rows, dtype=np.int64)


def _normalize_edge_vert_adj(value: Any, edge_count: int | None = None) -> np.ndarray:
    if value is None:
        count = int(edge_count or 0)
        return -np.ones((count, 2), dtype=np.int64)
    rows: List[List[int]] = []
    for row in value:
        vals = [int(x) for x in np.asarray(row).reshape(-1).tolist() if int(x) >= 0]
        if len(vals) == 0:
            rows.append([-1, -1])
        elif len(vals) == 1:
            rows.append([vals[0], -1])
        else:
            rows.append(vals[:2])
    if edge_count is not None and len(rows) < edge_count:
        rows.extend([[-1, -1] for _ in range(edge_count - len(rows))])
    return np.asarray(rows, dtype=np.int64)


def build_face_edge_adj(edge_face_adj: np.ndarray, face_count: int) -> List[List[int]]:
    face_edges: List[List[int]] = [[] for _ in range(face_count)]
    if edge_face_adj.ndim != 2:
        return face_edges
    for edge_id, row in enumerate(edge_face_adj):
        for face_id in row:
            fid = int(face_id)
            if 0 <= fid < face_count and edge_id not in face_edges[fid]:
                face_edges[fid].append(int(edge_id))
    return [sorted(edges) for edges in face_edges]


def build_fef_adj(face_edge_adj: List[List[int]], face_count: int) -> np.ndarray:
    fef_adj = np.zeros((face_count, face_count), dtype=np.int64)
    edge_sets = [set(int(edge) for edge in edges) for edges in face_edge_adj]
    for i in range(face_count):
        for j in range(i + 1, face_count):
            shared = len(edge_sets[i].intersection(edge_sets[j]))
            fef_adj[i, j] = shared
            fef_adj[j, i] = shared
    return fef_adj


def build_vert_face_adj(edge_face_adj: np.ndarray, edge_vert_adj: np.ndarray, vertex_count: int, face_count: int) -> List[List[int]]:
    vertex_faces: List[set[int]] = [set() for _ in range(vertex_count)]
    if edge_face_adj.ndim != 2 or edge_vert_adj.ndim != 2:
        return [[] for _ in range(vertex_count)]
    for edge_id in range(min(edge_face_adj.shape[0], edge_vert_adj.shape[0])):
        faces = [int(fid) for fid in edge_face_adj[edge_id].reshape(-1).tolist() if 0 <= int(fid) < face_count]
        verts = [int(vid) for vid in edge_vert_adj[edge_id].reshape(-1).tolist() if 0 <= int(vid) < vertex_count]
        for vid in verts:
            vertex_faces[vid].update(faces)
    return [sorted(vals) for vals in vertex_faces]


def _normalize_face_edge_adj(value: Any, edge_face_adj: np.ndarray, face_count: int) -> List[List[int]]:
    if value is None:
        return build_face_edge_adj(edge_face_adj, face_count)
    rows: List[List[int]] = []
    for row in value:
        vals = sorted({int(x) for x in np.asarray(row).reshape(-1).tolist() if int(x) >= 0})
        rows.append(vals)
    if len(rows) < face_count:
        rows.extend([[] for _ in range(face_count - len(rows))])
    return rows[:face_count]


def ensure_minimal_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(data)
    
    # 记录原始拓扑邻接度数，用于防御非流形静默截断
    if "max_raw_edge_face_degree" not in data:
        raw_ef = data.get("edgeFace_adj")
        if raw_ef is not None:
            max_ef = 0
            for row in raw_ef:
                vals = [int(x) for x in np.asarray(row).reshape(-1).tolist() if int(x) >= 0]
                max_ef = max(max_ef, len(vals))
            data["max_raw_edge_face_degree"] = max_ef
        else:
            data["max_raw_edge_face_degree"] = 0

    if "max_raw_edge_vert_degree" not in data:
        raw_ev = data.get("edgeVert_adj")
        if raw_ev is not None:
            max_ev = 0
            for row in raw_ev:
                vals = [int(x) for x in np.asarray(row).reshape(-1).tolist() if int(x) >= 0]
                max_ev = max(max_ev, len(vals))
            data["max_raw_edge_vert_degree"] = max_ev
        else:
            data["max_raw_edge_vert_degree"] = 0

    if "face_bbox_wcs" not in data or np.asarray(data.get("face_bbox_wcs")).size == 0:
        data["face_bbox_wcs"] = _bbox_stack_from_wcs(data.get("face_wcs"))
    else:
        data["face_bbox_wcs"] = _as_float_array(data.get("face_bbox_wcs"), 6)

    if "edge_bbox_wcs" not in data or np.asarray(data.get("edge_bbox_wcs")).size == 0:
        data["edge_bbox_wcs"] = _bbox_stack_from_wcs(data.get("edge_wcs"))
    else:
        data["edge_bbox_wcs"] = _as_float_array(data.get("edge_bbox_wcs"), 6)

    data["face_wcs"] = _as_float_array(data.get("face_wcs"))
    data["edge_wcs"] = _as_float_array(data.get("edge_wcs"))
    data["vert_wcs"] = _as_float_array(data.get("vert_wcs"), 3)

    data["face_count"] = int(data.get("face_count", data["face_bbox_wcs"].shape[0]))
    data["edge_count"] = int(data.get("edge_count", data["edge_bbox_wcs"].shape[0]))
    data["vertex_count"] = int(data.get("vertex_count", data["vert_wcs"].shape[0]))

    data["face_bbox_wcs"] = data["face_bbox_wcs"][: data["face_count"]].astype(np.float32)
    data["edge_bbox_wcs"] = data["edge_bbox_wcs"][: data["edge_count"]].astype(np.float32)
    data["vert_wcs"] = data["vert_wcs"][: data["vertex_count"]].astype(np.float32)

    data["edgeFace_adj"] = _normalize_edge_face_adj(data.get("edgeFace_adj"), data["edge_count"])
    data["edgeVert_adj"] = _normalize_edge_vert_adj(data.get("edgeVert_adj"), data["edge_count"])
    data["faceEdge_adj"] = _normalize_face_edge_adj(data.get("faceEdge_adj"), data["edgeFace_adj"], data["face_count"])
    if "fef_adj" not in data or np.asarray(data.get("fef_adj")).shape != (data["face_count"], data["face_count"]):
        data["fef_adj"] = build_fef_adj(data["faceEdge_adj"], data["face_count"])
    else:
        data["fef_adj"] = np.asarray(data["fef_adj"], dtype=np.int64)
    if "vertFace_adj" not in data or not isinstance(data.get("vertFace_adj"), list):
        data["vertFace_adj"] = build_vert_face_adj(data["edgeFace_adj"], data["edgeVert_adj"], data["vertex_count"], data["face_count"])

    if data["face_bbox_wcs"].shape[0] > 0:
        mn = np.min(data["face_bbox_wcs"][:, :3], axis=0)
        mx = np.max(data["face_bbox_wcs"][:, 3:], axis=0)
        data["global_bbox"] = np.concatenate([mn, mx]).astype(np.float32)
    elif data["vert_wcs"].shape[0] > 0:
        mn = np.min(data["vert_wcs"], axis=0)
        mx = np.max(data["vert_wcs"], axis=0)
        data["global_bbox"] = np.concatenate([mn, mx]).astype(np.float32)
    else:
        data["global_bbox"] = np.zeros(6, dtype=np.float32)

    data.setdefault("parser_backend", "unknown")
    data.setdefault("geometry_sampling_quality", "unknown")
    data.setdefault("solid_count", 1)
    return data


def _has_valid_controls(data: Dict[str, Any]) -> bool:
    for key in ["face_ctrs", "edge_ctrs"]:
        if key not in data or data.get(key) is None:
            return False
        arr = np.asarray(data.get(key))
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return False
    return True


def _has_duplicate_bbox(bboxes: np.ndarray, scaled_value: float = 3.0, threshold_value: float = 0.05) -> bool:
    if bboxes.ndim != 2 or bboxes.shape[1] != 6 or bboxes.shape[0] == 0:
        return True
    reshaped = (bboxes.astype(np.float32) * float(scaled_value)).reshape(bboxes.shape[0], 2, 3)
    non_repeat = reshaped[:1]
    for bbox in reshaped:
        diff = np.max(np.max(np.abs(non_repeat - bbox), axis=-1), axis=-1)
        same = diff < float(threshold_value)
        if int(same.sum()) >= 1:
            continue
        non_repeat = np.concatenate([non_repeat, bbox[np.newaxis, :, :]], axis=0)
    return len(non_repeat) != len(reshaped)


def check_dtg_train_compatible(
    data: Dict[str, Any],
    max_face: int = 50,
    max_edge_per_face: int = 30,
    edge_classes: int = 5,
    max_vert_face: int = 15,
) -> Tuple[bool, str, Dict[str, Any]]:
    """近似复现 DTG 训练阶段的 `check_step_ok`，但不直接依赖基线源码。"""
    try:
        data = ensure_minimal_fields(data)
    except Exception as exc:
        return False, f"dtg_minimal_field_error:{exc}", {}

    face_count = int(data.get("face_count", 0))
    edge_count = int(data.get("edge_count", 0))
    vertex_count = int(data.get("vertex_count", 0))
    face_edge_adj = data.get("faceEdge_adj", [])
    edge_vert_adj = np.asarray(data.get("edgeVert_adj"), dtype=np.int64)
    fef_adj = np.asarray(data.get("fef_adj"), dtype=np.int64)
    vert_face_adj = data.get("vertFace_adj", [])
    stats = {
        "dtg_max_face": max_face,
        "dtg_max_edge_per_face": max_edge_per_face,
        "dtg_edge_classes": edge_classes,
        "face_count": face_count,
        "edge_count": edge_count,
        "vertex_count": vertex_count,
    }

    if face_count > int(max_face):
        return False, "dtg_face_count_over_limit", stats
    if not _has_valid_controls(data):
        return False, "dtg_missing_or_invalid_bspline_controls", stats
    if not isinstance(vert_face_adj, list) or not vert_face_adj:
        return False, "dtg_vertFace_adj_missing", stats
    if max([len(item) for item in vert_face_adj] or [0]) > int(max_vert_face):
        return False, "dtg_vertFace_degree_over_limit", stats
    if fef_adj.shape != (face_count, face_count):
        return False, "dtg_fef_adj_invalid", stats
    if fef_adj.size and int(np.max(fef_adj)) >= int(edge_classes):
        return False, "dtg_edge_class_over_limit", stats

    sorted_edges = np.sort(edge_vert_adj, axis=1)
    unique_edges = np.unique(sorted_edges, axis=0)
    if unique_edges.shape[0] < edge_vert_adj.shape[0]:
        return False, "dtg_duplicate_edge_vertices", stats

    for face_edges in face_edge_adj:
        edges = [int(edge) for edge in face_edges]
        if len(edges) > int(max_edge_per_face):
            return False, "dtg_face_edge_count_over_limit", stats
        vertices = set()
        for edge_id in edges:
            if edge_id < 0 or edge_id >= edge_vert_adj.shape[0]:
                return False, "dtg_faceEdge_adj_out_of_range", stats
            vertices.update(int(v) for v in edge_vert_adj[edge_id].reshape(-1).tolist())
        if len(edges) != len(vertices):
            return False, "dtg_face_loop_not_simple", stats

    face_bbox = np.asarray(data.get("face_bbox_wcs"), dtype=np.float32)
    if _has_duplicate_bbox(face_bbox):
        return False, "dtg_duplicate_or_too_close_face_bbox", stats

    edge_bbox = np.asarray(data.get("edge_bbox_wcs"), dtype=np.float32)
    for face_edges in face_edge_adj:
        if len(face_edges) == 0:
            return False, "dtg_empty_face_edges", stats
        local_edge_bbox = edge_bbox[[int(edge) for edge in face_edges]]
        if _has_duplicate_bbox(local_edge_bbox):
            return False, "dtg_duplicate_or_too_close_edge_bbox_in_face", stats

    # 验证主路径下 B-Rep 实体映射与解析路径面片顺序完全一致，禁止顺序发生错位的噪声数据
    if data.get("parser_backend") == "dtg_occwl" and not bool(data.get("surface_metadata_order_verified", False)):
        return False, "dtg_surface_metadata_order_mismatch", stats

    return True, "dtg_train_compatible", stats


def _finite_array(data: Dict[str, Any], key: str) -> bool:
    arr = np.asarray(data.get(key))
    return bool(arr.size > 0 and np.all(np.isfinite(arr)))


def validate_brep(data: Dict[str, Any], max_faces: int = 70) -> Tuple[bool, str, Dict[str, Any]]:
    try:
        data = ensure_minimal_fields(data)
    except Exception as exc:
        return False, f"minimal_field_error:{exc}", {}

    face_count = int(data.get("face_count", 0))
    edge_count = int(data.get("edge_count", 0))
    vertex_count = int(data.get("vertex_count", 0))
    stats = {
        "face_count": face_count,
        "edge_count": edge_count,
        "vertex_count": vertex_count,
        "parser_backend": data.get("parser_backend", "unknown"),
        "geometry_sampling_quality": data.get("geometry_sampling_quality", "unknown"),
    }

    solid_count = int(data.get("solid_count", 1))
    if solid_count != 1:
        return False, "not_single_solid", stats
    if face_count <= 0:
        return False, "zero_faces", stats
    if face_count > int(max_faces):
        return False, "face_count_over_limit", stats
    if edge_count <= 0 or vertex_count <= 0:
        return False, "zero_edges_or_vertices", stats

    # 校验防御性规则：禁止任何非流形边界（边连接面数 > 2 或 边连接顶点数 > 2）
    if data.get("max_raw_edge_face_degree", 0) > 2:
        return False, "non_manifold_edge_face_adjacency", stats
    if data.get("max_raw_edge_vert_degree", 0) > 2:
        return False, "non_manifold_edge_vert_adjacency", stats

    edge_face_adj = np.asarray(data.get("edgeFace_adj"))
    edge_vert_adj = np.asarray(data.get("edgeVert_adj"))
    face_edge_adj = data.get("faceEdge_adj")
    if edge_face_adj.ndim != 2 or edge_face_adj.shape[0] != edge_count:
        return False, "edgeFace_adj_not_constructible", stats
    if edge_vert_adj.ndim != 2 or edge_vert_adj.shape[0] != edge_count:
        return False, "edgeVert_adj_not_constructible", stats
    if not isinstance(face_edge_adj, list) or len(face_edge_adj) != face_count:
        return False, "faceEdge_adj_not_constructible", stats

    for key in ["face_wcs", "edge_wcs", "vert_wcs", "face_bbox_wcs", "edge_bbox_wcs", "global_bbox"]:
        if not _finite_array(data, key):
            return False, f"{key}_invalid_or_nonfinite", stats

    global_bbox = np.asarray(data["global_bbox"], dtype=np.float32)
    dims = np.maximum(global_bbox[3:] - global_bbox[:3], 0.0)
    global_scale = float(np.max(dims)) if dims.size else 0.0
    stats["global_scale"] = global_scale
    if global_scale <= 1e-6:
        return False, "global_bbox_scale_too_small", stats

    return True, "clean", stats
