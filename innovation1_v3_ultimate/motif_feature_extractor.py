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


def _get_boundary_normals(face_idx: int, face_wcs: np.ndarray) -> List[np.ndarray]:
    try:
        if face_wcs is None or face_wcs.ndim < 4 or face_idx >= face_wcs.shape[0]:
            return []
        grid = face_wcs[face_idx]  # (32, 32, 3)
        v1 = grid[1:, :31] - grid[:-1, :31]
        v2 = grid[:31, 1:] - grid[:31, :-1]
        n_grid = np.cross(v1, v2)
        n_norm = np.linalg.norm(n_grid, axis=-1, keepdims=True)
        n_grid = n_grid / np.maximum(n_norm, 1e-8)
        
        boundaries = [
            n_grid[0, :, :],    # top
            n_grid[-1, :, :],   # bottom
            n_grid[:, 0, :],    # left
            n_grid[:, -1, :]    # right
        ]
        b_normals = []
        for b in boundaries:
            b_mean = np.mean(b, axis=0)
            b_norm = np.linalg.norm(b_mean)
            if b_norm > 1e-5:
                b_normals.append(b_mean / b_norm)
        return b_normals
    except Exception:
        return []


def _get_shared_boundary_normals(i: int, j: int, face_wcs: np.ndarray) -> Tuple[np.ndarray, np.ndarray] | None:
    try:
        if face_wcs is None or face_wcs.ndim < 4 or i >= face_wcs.shape[0] or j >= face_wcs.shape[0]:
            return None
        grid_i = face_wcs[i] # (32, 32, 3)
        grid_j = face_wcs[j]
        
        v1_i = grid_i[1:, :31] - grid_i[:-1, :31]
        v2_i = grid_i[:31, 1:] - grid_i[:31, :-1]
        n_grid_i = np.cross(v1_i, v2_i)
        n_norm_i = np.linalg.norm(n_grid_i, axis=-1, keepdims=True)
        n_grid_i = n_grid_i / np.maximum(n_norm_i, 1e-8)
        
        v1_j = grid_j[1:, :31] - grid_j[:-1, :31]
        v2_j = grid_j[:31, 1:] - grid_j[:31, :-1]
        n_grid_j = np.cross(v1_j, v2_j)
        n_norm_j = np.linalg.norm(n_grid_j, axis=-1, keepdims=True)
        n_grid_j = n_grid_j / np.maximum(n_norm_j, 1e-8)
        
        # 3D points midpoints for: 0=top, 1=bottom, 2=left, 3=right
        midpoints_i = [
            grid_i[0, 16],
            grid_i[-1, 16],
            grid_i[16, 0],
            grid_i[16, -1]
        ]
        midpoints_j = [
            grid_j[0, 16],
            grid_j[-1, 16],
            grid_j[16, 0],
            grid_j[16, -1]
        ]
        
        # Boundary normal matrices
        boundaries_i = [
            n_grid_i[0, :, :],
            n_grid_i[-1, :, :],
            n_grid_i[:, 0, :],
            n_grid_i[:, -1, :]
        ]
        boundaries_j = [
            n_grid_j[0, :, :],
            n_grid_j[-1, :, :],
            n_grid_j[:, 0, :],
            n_grid_j[:, -1, :]
        ]
        
        min_dist = float('inf')
        best_pair = None
        for p in range(4):
            for q in range(4):
                dist = np.linalg.norm(midpoints_i[p] - midpoints_j[q])
                if dist < min_dist:
                    min_dist = dist
                    best_pair = (p, q)
                    
        if best_pair is not None:
            p, q = best_pair
            
            b_mean_i = np.mean(boundaries_i[p], axis=0)
            norm_i = np.linalg.norm(b_mean_i)
            if norm_i > 1e-5:
                b_mean_i = b_mean_i / norm_i
            else:
                b_mean_i = np.array([0.0, 1.0, 0.0])
                
            b_mean_j = np.mean(boundaries_j[q], axis=0)
            norm_j = np.linalg.norm(b_mean_j)
            if norm_j > 1e-5:
                b_mean_j = b_mean_j / norm_j
            else:
                b_mean_j = np.array([0.0, 1.0, 0.0])
                
            return b_mean_i, b_mean_j
    except Exception:
        pass
    return None


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
    
    mean_curvatures = data.get("face_mean_curvature")
    if mean_curvatures is not None:
        mean_curvatures = np.asarray(mean_curvatures, dtype=np.float32)
        
    max_curvatures = data.get("face_max_curvature")
    if max_curvatures is not None:
        max_curvatures = np.asarray(max_curvatures, dtype=np.float32)
        
    var_curvatures = data.get("face_var_curvature")
    if var_curvatures is not None:
        var_curvatures = np.asarray(var_curvatures, dtype=np.float32)
        
    gaussian_signs = data.get("face_gaussian_sign")
    if gaussian_signs is not None:
        gaussian_signs = np.asarray(gaussian_signs, dtype=np.int32)
    
    # 提取面片类型元数据，用于后续过滤非平面面片的共面关系
    face_types = data.get("face_surface_type")
    if face_types is None:
        face_types = np.full(face_count, -1, dtype=np.int64)
    else:
        face_types = np.asarray(face_types, dtype=np.int64)

    cylinder_radii = data.get("face_cylinder_radius")
    if cylinder_radii is not None:
        cylinder_radii = np.asarray(cylinder_radii, dtype=np.float32)
    else:
        cylinder_radii = np.zeros(face_count, dtype=np.float32)

    cylinder_axes = data.get("face_cylinder_axis")
    if cylinder_axes is not None:
        cylinder_axes = np.asarray(cylinder_axes, dtype=np.float32)
    else:
        cylinder_axes = np.zeros((face_count, 3), dtype=np.float32)

    cylinder_locations = data.get("face_cylinder_location")
    if cylinder_locations is not None:
        cylinder_locations = np.asarray(cylinder_locations, dtype=np.float32)
    else:
        cylinder_locations = np.zeros((face_count, 3), dtype=np.float32)

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

    plane_locations = data.get("face_plane_location")
    area_centroids = data.get("face_area_centroid")

    # 提取面片真实面积并计算高精度旋转不变的相对面积分母
    exact_areas = data.get("face_area")
    if exact_areas is not None and len(exact_areas) >= face_count:
        actual_areas = [float(a) for a in exact_areas[:face_count]]
    else:
        # Fallback: 使用面片 OBB 面积代理计算
        actual_areas = []
        for fid in range(face_count):
            bbox = face_bbox[fid]
            mn_fb = bbox[:3]
            mx_fb = bbox[3:]
            dims_fb = np.maximum(mx_fb - mn_fb, 0.0)
            local_dims_fb = np.maximum(dims_fb, 1e-8)
            if face_wcs.ndim >= 4 and fid < face_wcs.shape[0]:
                try:
                    pts_fb = face_wcs[fid].reshape(-1, 3).astype(np.float64)
                    if pts_fb.shape[0] >= 3:
                        pts_fb = pts_fb - np.mean(pts_fb, axis=0, keepdims=True)
                        _, _, vh_fb = np.linalg.svd(pts_fb, full_matrices=False)
                        pts_local_fb = pts_fb @ vh_fb.T
                        local_dims_fb = np.max(pts_local_fb, axis=0) - np.min(pts_local_fb, axis=0)
                except Exception:
                    pass
            sorted_local_dims_fb = np.sort(np.maximum(local_dims_fb, 1e-8))
            area_proxy_fb = float(sorted_local_dims_fb[-1] * sorted_local_dims_fb[-2])
            actual_areas.append(area_proxy_fb)
            
    total_area_sum = sum(actual_areas)

    face_adj, shared_edges = _build_face_adjacency(edge_face_adj, data.get("faceEdge_adj"), face_count)
    face_features: List[Dict[str, Any]] = []
    for fid in range(face_count):
        bbox = face_bbox[fid]
        mn = bbox[:3]
        mx = bbox[3:]
        dims = np.maximum(mx - mn, 0.0)
        centroid = 0.5 * (mn + mx)
        # 确认法向来源质量以进行下游先验图过滤
        normal_source = "pca"
        has_analytical = False
        if "face_analytical_normals" in data and fid < len(data["face_analytical_normals"]):
            analytical_normal = np.asarray(data["face_analytical_normals"][fid], dtype=np.float32)
            face_type = int(data["face_surface_type"][fid]) if "face_surface_type" in data and fid < len(data["face_surface_type"]) else -1
            if face_type == 0:  # 平面 Plane
                norm_an = float(np.linalg.norm(analytical_normal))
                if norm_an > 1e-5:
                    normal = analytical_normal / norm_an
                    has_analytical = True
                    normal_source = "analytical_plane"
        
        if not has_analytical:
            if face_wcs.ndim >= 4 and fid < face_wcs.shape[0]:
                normal = _normal_from_grid(face_wcs[fid], dims)
                normal_source = "sampled_grid"
            else:
                normal = _normal_from_bbox(dims)
                normal_source = "bbox_fallback"
            norm = float(np.linalg.norm(normal))
            if norm <= 1e-8:
                normal = _normal_from_bbox(dims)
                normal_source = "bbox_fallback"
            else:
                normal = (normal / norm).astype(np.float32)

        # 计算基于点云 PCA 的有向包围盒（OBB）尺寸，确保特征在任意刚体旋转/平移下具有严格的一致性（旋转不变性）
        local_dims = np.maximum(dims, 1e-8)
        if face_wcs.ndim >= 4 and fid < face_wcs.shape[0]:
            try:
                pts = face_wcs[fid].reshape(-1, 3).astype(np.float64)
                if pts.shape[0] >= 3:
                    pts = pts - np.mean(pts, axis=0, keepdims=True)
                    _, _, vh = np.linalg.svd(pts, full_matrices=False)
                    pts_local = pts @ vh.T
                    local_dims = np.max(pts_local, axis=0) - np.min(pts_local, axis=0)
            except Exception:
                pass
        sorted_local_dims = np.sort(np.maximum(local_dims, 1e-8))
        area_proxy = float(sorted_local_dims[-1] * sorted_local_dims[-2])
        planar_aspect = float(sorted_local_dims[-1] / max(sorted_local_dims[-2], 1e-8))
        bbox_thinness = float(sorted_local_dims[0] / max(sorted_local_dims[-1], 1e-8))
        boundary_axes = []
        for axis, name in enumerate(["x", "y", "z"]):
            if abs(float(mn[axis] - global_min[axis])) <= boundary_tol[axis]:
                boundary_axes.append(f"{name}-")
            if abs(float(mx[axis] - global_max[axis])) <= boundary_tol[axis]:
                boundary_axes.append(f"{name}+")
        is_cyl = (int(face_types[fid]) == 1) if fid < len(face_types) else False
        axis_proxy = cylinder_axes[fid].tolist() if is_cyl else None
        plane_loc = plane_locations[fid] if plane_locations is not None and fid < len(plane_locations) else None
        area_cent = area_centroids[fid] if area_centroids is not None and fid < len(area_centroids) else None
        adjacency_faces = [int(i) for i in np.where(face_adj[fid] > 0)[0].tolist()]
        face_features.append(
            {
                "face_id": fid,
                "centroid": centroid.astype(float).tolist(),
                "bbox": bbox.astype(float).tolist(),
                "bbox_dims": dims.astype(float).tolist(),
                "area_proxy": area_proxy,
                "relative_area": float(actual_areas[fid] / max(total_area_sum, 1e-8)),
                "normal_proxy": normal.astype(float).tolist(),
                "normal_source": normal_source,
                "axis_proxy": axis_proxy,
                "plane_location": plane_loc,
                "area_centroid": area_cent,
                "aspect_ratio": min(planar_aspect, 1e4),
                "bbox_thinness": min(bbox_thinness, 1e4),
                "face_degree": int(face_adj[fid].sum()),
                "shared_edge_degree": int(shared_edges[fid].sum()),
                "boundary_flag": bool(boundary_axes),
                "boundary_axes": boundary_axes,
                "adjacency_faces": adjacency_faces,
                "curvature_proxy": float(curvature_proxy[fid]) if fid < curvature_proxy.shape[0] else 0.0,
                "mean_curvature": float(mean_curvatures[fid]) if mean_curvatures is not None and fid < mean_curvatures.shape[0] else None,
                "max_curvature": float(max_curvatures[fid]) if max_curvatures is not None and fid < max_curvatures.shape[0] else None,
                "var_curvature": float(var_curvatures[fid]) if var_curvatures is not None and fid < var_curvatures.shape[0] else None,
                "gaussian_sign": int(gaussian_signs[fid]) if gaussian_signs is not None and fid < gaussian_signs.shape[0] else None,
            }
        )

    parallel_cos = float(np.cos(np.radians(0.1)))
    ortho_sin = float(np.sin(np.radians(12.0)))
    coplanar_tol = max(1e-4, 1e-5 * global_scale)
    opposite_gap_min = max(1e-4, 1e-5 * global_scale)
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
        
        # 使用高精度平面点计算面面距离
        pi = fi.get("plane_location")
        pj = fj.get("plane_location")
        type_i = int(face_types[i]) if i < len(face_types) else -1
        type_j = int(face_types[j]) if j < len(face_types) else -1
        is_plane_plane = (type_i == 0 and type_j == 0)
        
        if pi is not None and pj is not None and is_plane_plane:
            plane_distance = float(abs(np.dot(np.asarray(pj) - np.asarray(pi), ni)))
        else:
            plane_distance = normal_gap

        # 对于平面—平面关系，统一定义有效间距为精确平面距离
        effective_gap = plane_distance if is_plane_plane else normal_gap
        # 兼容原有特征提取，同时将有效间距覆盖为 normal_gap 供下游提取
        normal_gap_val = effective_gap

        area_i = float(fi["area_proxy"])
        area_j = float(fj["area_proxy"])
        area_ratio = float(min(area_i, area_j) / max(max(area_i, area_j), 1e-8))
        
        # 计算基于面片 i PCA 主向投影的旋转不变投影重叠率
        projection_overlap_ratio = None
        projection_overlap_valid = False
        projection_overlap_source = "unavailable"
        
        if absdot >= parallel_cos and is_plane_plane:
            # 只有在平面且采样有效时执行计算
            if face_wcs is not None and face_wcs.ndim == 4:
                if i < face_wcs.shape[0] and j < face_wcs.shape[0]:
                    try:
                        grid_i = face_wcs[i]
                        grid_j = face_wcs[j]
                        pts_i = grid_i.reshape(-1, 3)
                        pts_j = grid_j.reshape(-1, 3)
                        
                        center_i = np.mean(pts_i, axis=0)
                        cov_i = np.cov(pts_i.T)
                        evals, evecs = np.linalg.eigh(cov_i)
                        e1 = evecs[:, 1]
                        e2 = evecs[:, 2]
                        
                        proj_i = np.stack([np.dot(pts_i - center_i, e1), np.dot(pts_i - center_i, e2)], axis=-1)
                        proj_j = np.stack([np.dot(pts_j - center_i, e1), np.dot(pts_j - center_i, e2)], axis=-1)
                        
                        min_i = np.min(proj_i, axis=0)
                        max_i = np.max(proj_i, axis=0)
                        min_j = np.min(proj_j, axis=0)
                        max_j = np.max(proj_j, axis=0)
                        
                        inter_min = np.maximum(min_i, min_j)
                        inter_max = np.minimum(max_i, max_j)
                        inter_dims = np.maximum(inter_max - inter_min, 0.0)
                        area_inter = inter_dims[0] * inter_dims[1]
                        
                        area_i_proj = (max_i[0] - min_i[0]) * (max_i[1] - min_i[1])
                        area_j_proj = (max_j[0] - min_j[0]) * (max_j[1] - min_j[1])
                        
                        if (np.isfinite(area_inter) and np.isfinite(area_i_proj) 
                                and np.isfinite(area_j_proj) and min(area_i_proj, area_j_proj) > 1e-10):
                            projection_overlap_ratio = float(np.clip(area_inter / min(area_i_proj, area_j_proj), 0.0, 1.0))
                            projection_overlap_valid = True
                            projection_overlap_source = "pca_projected_rectangle"
                    except Exception:
                        projection_overlap_ratio = 0.0
                        projection_overlap_valid = False
                        projection_overlap_source = "calculation_failed"

        # 计算关系证据质量类别 (relation_evidence_quality)
        src_src = fi.get("normal_source", "pca")
        dst_src = fj.get("normal_source", "pca")
        src_is_plane_analytical = (src_src == "analytical_plane")
        dst_is_plane_analytical = (dst_src == "analytical_plane")
        if src_is_plane_analytical and dst_is_plane_analytical:
            relation_evidence_quality = "analytical"
        elif src_is_plane_analytical or dst_is_plane_analytical:
            relation_evidence_quality = "mixed_analytical"
        elif "sampled_grid" in (src_src, dst_src):
            relation_evidence_quality = "sampled"
        else:
            relation_evidence_quality = "heuristic"

        base_evidence = {
            "face_pair": [i, j],
            "normal_dot": dot,
            "abs_normal_dot": absdot,
            "angle_to_parallel_deg": angle_to_parallel,
            "center_distance": center_distance,
            "normal_gap": normal_gap_val,
            "normal_gap_aabb_center": normal_gap,
            "plane_distance": plane_distance,
            "effective_gap": effective_gap,
            "area_ratio_min_over_max": area_ratio,
            "shared_edges": int(shared_edges[i, j]),
            "relation_evidence_quality": relation_evidence_quality,
            "projection_overlap_ratio": projection_overlap_ratio,
            "projection_overlap_valid": projection_overlap_valid,
            "projection_overlap_source": projection_overlap_source,
        }
        
        type_i = int(face_types[i]) if i < len(face_types) else -1
        type_j = int(face_types[j]) if j < len(face_types) else -1
        is_plane_plane = (type_i == 0 and type_j == 0)

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
        if absdot >= parallel_cos and is_plane_plane:
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
            # 核心改进：仅允许平面面片 (face_surface_type == 0) 之间建立共面关系，直接排除圆柱面等弯曲曲面
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
            # 实体薄壁 vs 空气空腔 双向有向外法向指向检查
            facing_each_other = (
                dot < 0.0
                and float(np.dot(delta, ni)) < 0.0
                and float(np.dot(-delta, nj)) < 0.0
            )
            if opposite_gap_min <= effective_gap <= 0.15 * global_scale and facing_each_other:
                evidence = dict(base_evidence)
                evidence["opposite_gap_min"] = opposite_gap_min
                evidence["orientation_status"] = "法向相反"
                confidence = 0.45 + 0.35 * _confidence_from_margin(absdot, 1.0, parallel_cos)
                confidence += 0.15 * _confidence_from_margin(effective_gap, opposite_gap_min, 0.08 * global_scale, invert=True)
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
        if is_plane_plane and abs(dot) <= ortho_sin:
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
        if face_adj[i, j] > 0:
            is_smooth = False
            effective_absdot = absdot
            
            # 优先使用基于网格中点距离最小化精确定位共享边的算法
            shared_norms = _get_shared_boundary_normals(i, j, face_wcs)
            if shared_norms is not None:
                ni_edge, nj_edge = shared_norms
                dot_val = abs(float(np.dot(ni_edge, nj_edge)))
                is_smooth = (dot_val >= smooth_cos)
                effective_absdot = dot_val
            else:
                # 降级退回到旧有的边界面法向匹配。非平面对绝对不允许退化（否则圆柱面会在其分割缝线处产生大量误判）
                if not is_plane_plane:
                    is_smooth = False
                else:
                    b_norms_i = _get_boundary_normals(i, face_wcs)
                    b_norms_j = _get_boundary_normals(j, face_wcs)
                    
                    if not b_norms_i and i < len(face_features):
                        b_norms_i = [np.asarray(face_features[i]["normal_proxy"], dtype=np.float32)]
                    if not b_norms_j and j < len(face_features):
                        b_norms_j = [np.asarray(face_features[j]["normal_proxy"], dtype=np.float32)]
                        
                    if b_norms_i and b_norms_j:
                        max_boundary_absdot = 0.0
                        for ni_b in b_norms_i:
                            for nj_b in b_norms_j:
                                dot_val = abs(float(np.dot(ni_b, nj_b)))
                                if dot_val > max_boundary_absdot:
                                    max_boundary_absdot = dot_val
                        is_smooth = (max_boundary_absdot >= smooth_cos)
                        effective_absdot = max_boundary_absdot
                
            if is_smooth:
                evidence = dict(base_evidence)
                evidence["smooth_cos_diff_tol"] = 0.05  # 使用法向夹角余弦差阈值（余弦 >= 0.95）
                face_relations.append(
                    {
                        "source_face": i,
                        "target_face": j,
                        "type": "smooth_connected",
                        "confidence": 0.55 + 0.40 * _confidence_from_margin(effective_absdot, good=1.0, bad=smooth_cos, invert=False),
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
            "parallel_angle_deg": 0.1,
            "orthogonal_angle_deg": 12.0,
            "coplanar_tol": coplanar_tol,
            "opposite_gap_min": opposite_gap_min,
            "smooth_angle_deg": 18.0,
        },
        "face_surface_type": face_types.tolist(),  # 导出面片类型，方便下游合并逻辑读取
        "face_cylinder_radius": cylinder_radii.tolist(),
        "face_cylinder_axis": cylinder_axes.tolist(),
        "face_cylinder_location": cylinder_locations.tolist(),
        "face_features": face_features,
        "face_adjacency": face_adj.astype(int).tolist(),
        "face_shared_edges": shared_edges.astype(int).tolist(),
        "face_relations": face_relations,
        "face_relation_stats": relation_stats,
        "face_wcs": face_wcs,
        "parser_backend": data.get("parser_backend", "unknown"),
        "geometry_sampling_quality": data.get("geometry_sampling_quality", "unknown"),
    }
