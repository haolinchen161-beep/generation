# -*- coding: utf-8 -*-
"""Geometry/topology feature extraction for weak B-Rep semantics."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def _as_array(data: Any, dtype=np.float32) -> np.ndarray:
    if data is None:
        return np.zeros((0,), dtype=dtype)
    return np.asarray(data, dtype=dtype)


def _bbox_from_points(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.zeros(6, dtype=np.float32)
    pts = points.reshape(-1, 3)
    mn = np.min(pts, axis=0)
    mx = np.max(pts, axis=0)
    return np.concatenate([mn, mx]).astype(np.float32)


def _face_bboxes(data: Dict[str, Any]) -> np.ndarray:
    if "face_bbox_wcs" in data:
        arr = _as_array(data["face_bbox_wcs"])
        if arr.ndim == 2 and arr.shape[1] == 6:
            return arr.astype(np.float32)
    face_wcs = _as_array(data.get("face_wcs"))
    if face_wcs.ndim >= 3:
        return np.stack([_bbox_from_points(face_wcs[i]) for i in range(face_wcs.shape[0])], axis=0)
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
        for u, v in [(0, 0), (0, pts.shape[1] - 2), (pts.shape[0] - 2, 0), (pts.shape[0] // 2, pts.shape[1] // 2)]:
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
        faces = [int(x) for x in item if int(x) >= 0 and int(x) < face_count]
        for i in range(len(faces)):
            for j in range(i + 1, len(faces)):
                adj[faces[i], faces[j]] = 1
                adj[faces[j], faces[i]] = 1
    return adj


def extract_semantic_features(data: Dict[str, Any]) -> Dict[str, Any]:
    face_bbox = _face_bboxes(data)
    face_count = int(data.get("face_count", face_bbox.shape[0]))
    face_bbox = face_bbox[:face_count]
    face_wcs = _as_array(data.get("face_wcs"))
    edge_face_adj = _as_array(data.get("edgeFace_adj"), dtype=np.int64)
    curvature_proxy = _as_array(data.get("face_curvature_proxy"), dtype=np.float32).reshape(-1)

    if face_bbox.shape[0] == 0:
        global_min = np.zeros(3, dtype=np.float32)
        global_max = np.zeros(3, dtype=np.float32)
    else:
        global_min = np.min(face_bbox[:, :3], axis=0)
        global_max = np.max(face_bbox[:, 3:], axis=0)
    global_dims = np.maximum(global_max - global_min, 1e-6)
    scale = float(np.max(global_dims)) if global_dims.size else 1.0
    tol = np.maximum(global_dims * 0.025, max(1e-4 * scale, 1e-6))

    face_adj = _build_face_adjacency(edge_face_adj, face_count)
    features: List[Dict[str, Any]] = []
    for idx in range(face_count):
        bbox = face_bbox[idx]
        mn = bbox[:3]
        mx = bbox[3:]
        dims = np.maximum(mx - mn, 0.0)
        centroid = 0.5 * (mn + mx)
        if face_wcs.ndim >= 4 and idx < face_wcs.shape[0]:
            normal = _normal_from_grid(face_wcs[idx], dims)
        else:
            normal = _normal_from_bbox_dims(dims)
        sorted_dims = np.sort(np.maximum(dims, 1e-6))
        aspect = float(sorted_dims[2] / sorted_dims[0])
        area_proxy = _area_proxy_from_bbox(dims)
        boundary_axes = []
        for axis, name in enumerate(["x", "y", "z"]):
            if abs(float(mn[axis] - global_min[axis])) <= tol[axis] or abs(float(mx[axis] - global_max[axis])) <= tol[axis]:
                boundary_axes.append(name)
        is_internal_xz = (
            mn[0] > global_min[0] + tol[0]
            and mx[0] < global_max[0] - tol[0]
            and mn[2] > global_min[2] + tol[2]
            and mx[2] < global_max[2] - tol[2]
        )
        features.append(
            {
                "face_id": idx,
                "bbox": bbox.astype(float).tolist(),
                "dims": dims.astype(float).tolist(),
                "centroid": centroid.astype(float).tolist(),
                "normal": normal.astype(float).tolist(),
                "area_proxy": area_proxy,
                "aspect_ratio": min(aspect, 1e4),
                "degree": int(face_adj[idx].sum()) if idx < face_adj.shape[0] else 0,
                "curvature_proxy": float(curvature_proxy[idx]) if idx < curvature_proxy.shape[0] else 0.0,
                "boundary_axes": boundary_axes,
                "is_global_boundary": bool(boundary_axes),
                "is_internal_xz": bool(is_internal_xz),
            }
        )

    return {
        "uid": data.get("uid", ""),
        "face_count": face_count,
        "edge_count": int(data.get("edge_count", len(data.get("edgeFace_adj", [])))),
        "vertex_count": int(data.get("vertex_count", len(data.get("vert_wcs", [])))),
        "global_min": global_min.astype(float).tolist(),
        "global_max": global_max.astype(float).tolist(),
        "global_dims": global_dims.astype(float).tolist(),
        "face_features": features,
        "face_adjacency": face_adj.astype(int).tolist(),
    }
