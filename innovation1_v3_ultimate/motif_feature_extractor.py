# -*- coding: utf-8 -*-
"""为结构基元图 M 抽取 face-level 几何与拓扑证据。"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Tuple

import numpy as np


def _as_array(data: Any, dtype=np.float32) -> np.ndarray:
    if data is None:
        return np.zeros((0,), dtype=dtype)
    return np.asarray(data, dtype=dtype)


def _bbox_from_points(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.zeros(6, dtype=np.float32)
    pts = points.reshape(-1, 3)
    return np.concatenate([np.min(pts, axis=0), np.max(pts, axis=0)]).astype(np.float32)


def _face_bboxes(data: Dict[str, Any]) -> np.ndarray:
    bbox = _as_array(data.get("face_bbox_wcs"), dtype=np.float32)
    if bbox.ndim == 2 and bbox.shape[1] == 6 and bbox.shape[0] > 0:
        return bbox
    face_wcs = _as_array(data.get("face_wcs"), dtype=np.float32)
    if face_wcs.ndim >= 3 and face_wcs.shape[0] > 0:
        return np.stack([_bbox_from_points(face_wcs[i]) for i in range(face_wcs.shape[0])], axis=0)
    return np.zeros((0, 6), dtype=np.float32)


def _normal_from_bbox(dims: np.ndarray) -> np.ndarray:
    dims = np.maximum(np.asarray(dims, dtype=np.float32), 1e-8)
    axis = int(np.argmin(dims))
    normal = np.zeros(3, dtype=np.float32)
    normal[axis] = 1.0
    return normal


def _normal_from_pca(points: np.ndarray, dims: np.ndarray) -> np.ndarray:
    try:
        pts = points.reshape(-1, 3).astype(np.float64)
        if pts.shape[0] < 3:
            return _normal_from_bbox(dims)
        pts = pts - np.mean(pts, axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(pts, full_matrices=False)
        n = vh[-1].astype(np.float32)
        norm = float(np.linalg.norm(n))
        if norm <= 1e-8:
            return _normal_from_bbox(dims)
        n = n / norm
        major = int(np.argmax(np.abs(n)))
        if n[major] < 0:
            n = -n
        return n.astype(np.float32)
    except Exception:
        return _normal_from_bbox(dims)


def _normal_from_grid(grid: np.ndarray, dims: np.ndarray) -> np.ndarray:
    try:
        pts = np.asarray(grid, dtype=np.float32)
        if pts.ndim < 3 or pts.shape[-1] != 3:
            return _normal_from_pca(pts, dims)
        pts = pts.reshape(pts.shape[0], pts.shape[1], 3)
        candidates = []
        anchors = [
            (0, 0),
            (0, max(0, pts.shape[1] - 2)),
            (max(0, pts.shape[0] - 2), 0),
            (max(0, pts.shape[0] // 2 - 1), max(0, pts.shape[1] // 2 - 1)),
        ]
        for u, v in anchors:
            a = pts[u, v]
            b = pts[min(u + 1, pts.shape[0] - 1), v]
            c = pts[u, min(v + 1, pts.shape[1] - 1)]
            n = np.cross(b - a, c - a)
            norm = float(np.linalg.norm(n))
            if norm > 1e-8:
                candidates.append(n / norm)
        if candidates:
            n = np.mean(candidates, axis=0)
            norm = float(np.linalg.norm(n))
            if norm > 1e-8:
                return (n / norm).astype(np.float32)
    except Exception:
        pass
    return _normal_from_pca(np.asarray(grid), dims)


def _area_proxy_from_dims(dims: np.ndarray) -> float:
    sorted_dims = np.sort(np.maximum(dims, 0.0))
    return float(sorted_dims[-1] * sorted_dims[-2])


def _face_edge_rows(face_edge_adj: Any, face_count: int) -> List[List[int]]:
    rows: List[List[int]] = []
    if isinstance(face_edge_adj, list):
        for row in face_edge_adj[:face_count]:
            rows.append(sorted({int(x) for x in np.asarray(row).reshape(-1).tolist() if int(x) >= 0}))
    while len(rows) < face_count:
        rows.append([])
    return rows[:face_count]


def _build_face_adjacency(edge_face_adj: np.ndarray, face_edge_adj: Any, face_count: int) -> Tuple[np.ndarray, np.ndarray]:
    adj = np.zeros((face_count, face_count), dtype=np.int32)
    shared_edges = np.zeros((face_count, face_count), dtype=np.int32)
    if edge_face_adj.ndim == 2 and edge_face_adj.shape[0] > 0:
        for row in edge_face_adj:
            faces = sorted({int(x) for x in row if 0 <= int(x) < face_count})
            for i, j in combinations(faces, 2):
                adj[i, j] = 1
                adj[j, i] = 1
                shared_edges[i, j] += 1
                shared_edges[j, i] += 1
    if adj.sum() == 0:
        face_edges = _face_edge_rows(face_edge_adj, face_count)
        edge_to_faces: Dict[int, List[int]] = {}
        for fid, edges in enumerate(face_edges):
            for eid in edges:
                edge_to_faces.setdefault(eid, []).append(fid)
        for faces in edge_to_faces.values():
            for i, j in combinations(sorted(set(faces)), 2):
                adj[i, j] = 1
                adj[j, i] = 1
                shared_edges[i, j] += 1
                shared_edges[j, i] += 1
    return adj, shared_edges


def _angle_deg_from_absdot(absdot: float) -> float:
    val = max(-1.0, min(1.0, float(absdot)))
    return float(np.degrees(np.arccos(val)))


def _confidence_from_margin(value: float, good: float, bad: float, invert: bool = False) -> float:
    value = float(value)
    if invert:
        score = (bad - value) / max(bad - good, 1e-8)
    else:
        score = (value - bad) / max(good - bad, 1e-8)
    return float(max(0.05, min(0.99, score)))


def extract_motif_features(data: Dict[str, Any]) -> Dict[str, Any]:
    face_bbox = _face_bboxes(data)
    face_count = int(data.get("face_count", face_bbox.shape[0]))
    face_bbox = face_bbox[:face_count]
    edge_face_adj = _as_array(data.get("edgeFace_adj"), dtype=np.int64)
    face_wcs = _as_array(data.get("face_wcs"), dtype=np.float32)
    curvature_proxy = _as_array(data.get("face_curvature_proxy"), dtype=np.float32).reshape(-1)

    if face_bbox.shape[0] > 0:
        global_min = np.min(face_bbox[:, :3], axis=0)
        global_max = np.max(face_bbox[:, 3:], axis=0)
    else:
        global_min = np.zeros(3, dtype=np.float32)
        global_max = np.ones(3, dtype=np.float32)
    global_dims = np.maximum(global_max - global_min, 1e-6)
    global_scale = float(np.max(global_dims))
    global_area_proxy = float(np.prod(np.sort(global_dims)[-2:]))
    boundary_tol = np.maximum(global_dims * 0.02, max(1e-5, 1e-4 * global_scale))

    face_adj, shared_edges = _build_face_adjacency(edge_face_adj, data.get("faceEdge_adj"), face_count)
    face_features: List[Dict[str, Any]] = []
    for fid in range(face_count):
        bbox = face_bbox[fid]
        mn = bbox[:3]
        mx = bbox[3:]
        dims = np.maximum(mx - mn, 0.0)
        centroid = 0.5 * (mn + mx)
        if face_wcs.ndim >= 4 and fid < face_wcs.shape[0]:
            normal = _normal_from_grid(face_wcs[fid], dims)
        else:
            normal = _normal_from_bbox(dims)
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-8:
            normal = _normal_from_bbox(dims)
        else:
            normal = (normal / norm).astype(np.float32)

        sorted_dims = np.sort(np.maximum(dims, 1e-8))
        area_proxy = _area_proxy_from_dims(dims)
        planar_aspect = float(sorted_dims[-1] / max(sorted_dims[-2], 1e-8))
        bbox_thinness = float(sorted_dims[0] / max(sorted_dims[-1], 1e-8))
        boundary_axes = []
        for axis, name in enumerate(["x", "y", "z"]):
            if abs(float(mn[axis] - global_min[axis])) <= boundary_tol[axis]:
                boundary_axes.append(f"{name}-")
            if abs(float(mx[axis] - global_max[axis])) <= boundary_tol[axis]:
                boundary_axes.append(f"{name}+")
        adjacency_faces = [int(i) for i in np.where(face_adj[fid] > 0)[0].tolist()]
        face_features.append(
            {
                "face_id": fid,
                "centroid": centroid.astype(float).tolist(),
                "bbox": bbox.astype(float).tolist(),
                "bbox_dims": dims.astype(float).tolist(),
                "area_proxy": area_proxy,
                "relative_area": float(area_proxy / max(global_area_proxy, 1e-8)),
                "normal_proxy": normal.astype(float).tolist(),
                "aspect_ratio": min(planar_aspect, 1e4),
                "bbox_thinness": min(bbox_thinness, 1e4),
                "face_degree": int(face_adj[fid].sum()),
                "shared_edge_degree": int(shared_edges[fid].sum()),
                "boundary_flag": bool(boundary_axes),
                "boundary_axes": boundary_axes,
                "adjacency_faces": adjacency_faces,
                "curvature_proxy": float(curvature_proxy[fid]) if fid < curvature_proxy.shape[0] else 0.0,
            }
        )

    parallel_cos = float(np.cos(np.radians(12.0)))
    ortho_sin = float(np.sin(np.radians(12.0)))
    coplanar_tol = max(0.015 * global_scale, 1e-5)
    opposite_gap_min = max(0.015 * global_scale, 1e-5)
    smooth_cos = float(np.cos(np.radians(18.0)))

    face_relations: List[Dict[str, Any]] = []
    for i, j in combinations(range(face_count), 2):
        fi = face_features[i]
        fj = face_features[j]
        ni = np.asarray(fi["normal_proxy"], dtype=np.float32)
        nj = np.asarray(fj["normal_proxy"], dtype=np.float32)
        ci = np.asarray(fi["centroid"], dtype=np.float32)
        cj = np.asarray(fj["centroid"], dtype=np.float32)
        delta = cj - ci
        dot = float(np.dot(ni, nj))
        absdot = abs(dot)
        angle_to_parallel = _angle_deg_from_absdot(absdot)
        center_distance = float(np.linalg.norm(delta))
        normal_gap = float(abs(np.dot(delta, ni)))
        plane_distance = normal_gap
        area_i = float(fi["area_proxy"])
        area_j = float(fj["area_proxy"])
        area_ratio = float(min(area_i, area_j) / max(max(area_i, area_j), 1e-8))
        base_evidence = {
            "face_pair": [i, j],
            "normal_dot": dot,
            "abs_normal_dot": absdot,
            "angle_to_parallel_deg": angle_to_parallel,
            "center_distance": center_distance,
            "normal_gap": normal_gap,
            "plane_distance": plane_distance,
            "area_ratio_min_over_max": area_ratio,
            "shared_edges": int(shared_edges[i, j]),
        }
        if face_adj[i, j] > 0:
            face_relations.append(
                {
                    "source_face": i,
                    "target_face": j,
                    "type": "adjacent_to",
                    "confidence": 1.0,
                    "evidence": dict(base_evidence),
                }
            )
        if absdot >= parallel_cos:
            conf = 0.65 + 0.34 * _confidence_from_margin(absdot, 1.0, parallel_cos)
            face_relations.append(
                {
                    "source_face": i,
                    "target_face": j,
                    "type": "parallel_to",
                    "confidence": float(min(conf, 0.99)),
                    "evidence": dict(base_evidence),
                }
            )
            if plane_distance <= coplanar_tol:
                evidence = dict(base_evidence)
                evidence["coplanar_tol"] = coplanar_tol
                face_relations.append(
                    {
                        "source_face": i,
                        "target_face": j,
                        "type": "coplanar_with",
                        "confidence": _confidence_from_margin(plane_distance, coplanar_tol * 0.2, coplanar_tol, invert=True),
                        "evidence": evidence,
                    }
                )
            if normal_gap >= opposite_gap_min:
                evidence = dict(base_evidence)
                evidence["opposite_gap_min"] = opposite_gap_min
                evidence["orientation_status"] = "法向相反" if dot < 0 else "无向平行且中心分离"
                confidence = 0.45 + 0.35 * _confidence_from_margin(absdot, 1.0, parallel_cos)
                confidence += 0.15 * _confidence_from_margin(normal_gap, 0.08 * global_scale, opposite_gap_min)
                confidence += 0.05 * area_ratio
                face_relations.append(
                    {
                        "source_face": i,
                        "target_face": j,
                        "type": "opposite_to",
                        "confidence": float(max(0.1, min(0.98, confidence))),
                        "evidence": evidence,
                    }
                )
        if abs(dot) <= ortho_sin:
            evidence = dict(base_evidence)
            evidence["orthogonal_absdot_tol"] = ortho_sin
            face_relations.append(
                {
                    "source_face": i,
                    "target_face": j,
                    "type": "orthogonal_to",
                    "confidence": _confidence_from_margin(abs(dot), ortho_sin * 0.25, ortho_sin, invert=True),
                    "evidence": evidence,
                }
            )
        if face_adj[i, j] > 0 and absdot >= smooth_cos:
            evidence = dict(base_evidence)
            evidence["smooth_cos_tol"] = smooth_cos
            face_relations.append(
                {
                    "source_face": i,
                    "target_face": j,
                    "type": "smooth_connected",
                    "confidence": 0.55 + 0.4 * _confidence_from_margin(absdot, 1.0, smooth_cos),
                    "evidence": evidence,
                }
            )

    relation_stats: Dict[str, int] = {}
    for rel in face_relations:
        typ = str(rel["type"])
        relation_stats[typ] = relation_stats.get(typ, 0) + 1

    return {
        "uid": data.get("uid", ""),
        "source": data.get("source", "unknown"),
        "face_count": face_count,
        "edge_count": int(data.get("edge_count", 0)),
        "vertex_count": int(data.get("vertex_count", 0)),
        "global_bbox": np.concatenate([global_min, global_max]).astype(float).tolist(),
        "global_dims": global_dims.astype(float).tolist(),
        "global_scale": global_scale,
        "thresholds": {
            "parallel_angle_deg": 12.0,
            "orthogonal_angle_deg": 12.0,
            "coplanar_tol": coplanar_tol,
            "opposite_gap_min": opposite_gap_min,
            "smooth_angle_deg": 18.0,
        },
        "face_features": face_features,
        "face_adjacency": face_adj.astype(int).tolist(),
        "face_shared_edges": shared_edges.astype(int).tolist(),
        "face_relations": face_relations,
        "face_relation_stats": relation_stats,
        "parser_backend": data.get("parser_backend", "unknown"),
        "geometry_sampling_quality": data.get("geometry_sampling_quality", "unknown"),
    }
