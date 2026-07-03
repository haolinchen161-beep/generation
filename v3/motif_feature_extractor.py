# -*- coding: utf-8 -*-
"""Feature extraction from parsed public B-Rep tensors for motif graph M."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def _as_array(data: Any, dtype=np.float32) -> np.ndarray:
    if data is None:
        return np.zeros((0,), dtype=dtype)
    try:
        return np.asarray(data, dtype=dtype)
    except Exception:
        return np.zeros((0,), dtype=dtype)


def _bbox_from_points(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.zeros(6, dtype=np.float32)
    pts = points.reshape(-1, 3)
    return np.concatenate([np.min(pts, axis=0), np.max(pts, axis=0)]).astype(np.float32)


def _face_bboxes(data: Dict[str, Any]) -> np.ndarray:
    arr = _as_array(data.get("face_bbox_wcs"))
    if arr.ndim == 2 and arr.shape[1] == 6:
        return arr.astype(np.float32)
    face_wcs = _as_array(data.get("face_wcs"))
    if face_wcs.ndim >= 3:
        return np.stack([_bbox_from_points(face_wcs[i]) for i in range(face_wcs.shape[0])], axis=0)
    return np.zeros((0, 6), dtype=np.float32)


def _edge_bboxes(data: Dict[str, Any]) -> np.ndarray:
    arr = _as_array(data.get("edge_bbox_wcs"))
    if arr.ndim == 2 and arr.shape[1] == 6:
        return arr.astype(np.float32)
    edge_wcs = _as_array(data.get("edge_wcs"))
    if edge_wcs.ndim >= 2:
        return np.stack([_bbox_from_points(edge_wcs[i]) for i in range(edge_wcs.shape[0])], axis=0)
    return np.zeros((0, 6), dtype=np.float32)


def _normal_from_bbox_dims(dims: np.ndarray) -> np.ndarray:
    axis = int(np.argmin(np.maximum(dims, 1e-8)))
    normal = np.zeros(3, dtype=np.float32)
    normal[axis] = 1.0
    return normal


def _normal_from_grid(grid: np.ndarray, dims: np.ndarray) -> np.ndarray:
    try:
        pts = grid.reshape(grid.shape[0], grid.shape[1], 3)
        candidates = []
        probes = [(0, 0), (0, pts.shape[1] - 2), (pts.shape[0] - 2, 0), (pts.shape[0] // 2, pts.shape[1] // 2)]
        for u, v in probes:
            a = pts[u, v]
            b = pts[min(u + 1, pts.shape[0] - 1), v]
            c = pts[u, min(v + 1, pts.shape[1] - 1)]
            n = np.cross(b - a, c - a)
            norm = np.linalg.norm(n)
            if norm > 1e-8:
                candidates.append(n / norm)
        if candidates:
            n = np.mean(candidates, axis=0)
            norm = np.linalg.norm(n)
            if norm > 1e-8:
                return (n / norm).astype(np.float32)
    except Exception:
        pass
    return _normal_from_bbox_dims(dims)


def _area_proxy_from_bbox(dims: np.ndarray) -> float:
    sorted_dims = np.sort(np.maximum(dims, 0.0))
    return float(sorted_dims[1] * sorted_dims[2])


def _build_face_adjacency(edge_face_adj: np.ndarray, face_count: int) -> np.ndarray:
    adj = np.zeros((face_count, face_count), dtype=np.int32)
    if edge_face_adj.ndim != 2:
        return adj
    for item in edge_face_adj:
        faces = [int(x) for x in item if 0 <= int(x) < face_count]
        for i in range(len(faces)):
            for j in range(i + 1, len(faces)):
                adj[faces[i], faces[j]] = 1
                adj[faces[j], faces[i]] = 1
    return adj


def _face_edge_degrees(face_edge_adj: Any, edge_face_adj: np.ndarray, face_count: int) -> List[int]:
    if isinstance(face_edge_adj, list) and len(face_edge_adj) >= face_count:
        return [len(set(int(x) for x in face_edge_adj[i] if int(x) >= 0)) for i in range(face_count)]
    degrees = [0] * face_count
    if edge_face_adj.ndim == 2:
        for item in edge_face_adj:
            for fid in item:
                fid = int(fid)
                if 0 <= fid < face_count:
                    degrees[fid] += 1
    return degrees


def _projection_overlap(box_a: np.ndarray, box_b: np.ndarray, axes: List[int]) -> float:
    ratios = []
    for axis in axes:
        a0, a1 = float(box_a[axis]), float(box_a[axis + 3])
        b0, b1 = float(box_b[axis]), float(box_b[axis + 3])
        inter = max(0.0, min(a1, b1) - max(a0, b0))
        union = max(a1, b1) - min(a0, b0)
        ratios.append(inter / max(union, 1e-8))
    return float(np.prod(ratios)) if ratios else 0.0


def extract_motif_features(data: Dict[str, Any]) -> Dict[str, Any]:
    face_bbox = _face_bboxes(data)
    edge_bbox = _edge_bboxes(data)
    face_count = int(data.get("face_count", face_bbox.shape[0]))
    edge_count = int(data.get("edge_count", edge_bbox.shape[0]))
    vertex_count = int(data.get("vertex_count", len(data.get("vert_wcs", []))))
    face_bbox = face_bbox[:face_count]
    edge_bbox = edge_bbox[:edge_count]

    face_wcs = _as_array(data.get("face_wcs"))
    edge_face_adj = _as_array(data.get("edgeFace_adj"), dtype=np.int64)
    edge_vert_adj = _as_array(data.get("edgeVert_adj"), dtype=np.int64)
    face_edge_adj = data.get("faceEdge_adj", [])
    curvature_proxy = _as_array(data.get("face_curvature_proxy"), dtype=np.float32).reshape(-1)

    if face_bbox.shape[0] == 0:
        global_min = np.zeros(3, dtype=np.float32)
        global_max = np.zeros(3, dtype=np.float32)
    else:
        global_min = np.min(face_bbox[:, :3], axis=0)
        global_max = np.max(face_bbox[:, 3:], axis=0)
    global_dims = np.maximum(global_max - global_min, 1e-8)
    global_scale = float(np.max(global_dims)) if global_dims.size else 1.0
    tol = np.maximum(global_dims * 0.025, max(1e-4 * global_scale, 1e-8))

    face_adj = _build_face_adjacency(edge_face_adj, face_count)
    edge_degrees = _face_edge_degrees(face_edge_adj, edge_face_adj, face_count)

    face_features: List[Dict[str, Any]] = []
    for idx in range(face_count):
        bbox = face_bbox[idx]
        mn, mx = bbox[:3], bbox[3:]
        dims = np.maximum(mx - mn, 0.0)
        centroid = 0.5 * (mn + mx)
        normal = _normal_from_grid(face_wcs[idx], dims) if face_wcs.ndim >= 4 and idx < face_wcs.shape[0] else _normal_from_bbox_dims(dims)
        sorted_dims = np.sort(np.maximum(dims, 1e-8))
        boundary_axes = []
        for axis, name in enumerate(["x", "y", "z"]):
            if abs(float(mn[axis] - global_min[axis])) <= tol[axis] or abs(float(mx[axis] - global_max[axis])) <= tol[axis]:
                boundary_axes.append(name)
        internal_xz = (
            mn[0] > global_min[0] + tol[0]
            and mx[0] < global_max[0] - tol[0]
            and mn[2] > global_min[2] + tol[2]
            and mx[2] < global_max[2] - tol[2]
        )
        face_features.append(
            {
                "face_id": idx,
                "bbox": bbox.astype(float).tolist(),
                "dims": dims.astype(float).tolist(),
                "centroid": centroid.astype(float).tolist(),
                "normal": normal.astype(float).tolist(),
                "area_proxy": _area_proxy_from_bbox(dims),
                "aspect_ratio": float(min(sorted_dims[2] / max(sorted_dims[0], 1e-8), 1e5)),
                "degree": int(edge_degrees[idx]) if idx < len(edge_degrees) else int(face_adj[idx].sum()),
                "adjacent_degree": int(face_adj[idx].sum()) if idx < face_adj.shape[0] else 0,
                "curvature_proxy": float(curvature_proxy[idx]) if idx < curvature_proxy.shape[0] else 0.0,
                "boundary_axes": boundary_axes,
                "is_global_boundary": bool(boundary_axes),
                "is_internal_xz": bool(internal_xz),
            }
        )

    return {
        "uid": str(data.get("uid", "")),
        "source": str(data.get("source", data.get("source_dataset", "public_brep"))),
        "face_count": face_count,
        "edge_count": edge_count,
        "vertex_count": vertex_count,
        "global_min": global_min.astype(float).tolist(),
        "global_max": global_max.astype(float).tolist(),
        "global_dims": global_dims.astype(float).tolist(),
        "global_scale": global_scale,
        "face_features": face_features,
        "edge_bbox_wcs": edge_bbox.astype(float).tolist(),
        "edgeFace_adj": edge_face_adj.astype(int).tolist() if edge_face_adj.size else [],
        "edgeVert_adj": edge_vert_adj.astype(int).tolist() if edge_vert_adj.size else [],
        "faceEdge_adj": face_edge_adj,
        "face_adjacency": face_adj.astype(int).tolist(),
        "projection_overlap": _projection_overlap,
        "parser_backend": data.get("parser_backend", "unknown"),
        "geometry_sampling_quality": data.get("geometry_sampling_quality", "unknown"),
    }

