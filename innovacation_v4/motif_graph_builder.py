# -*- coding: utf-8 -*-
"""由 face-level evidence 构建弱结构基元图 M=(Vm, Em, Pm)。"""

from __future__ import annotations

import copy
import csv
import json
import os
import time
from collections import Counter
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

try:  # pragma: no cover
    from .brep_cleaner import check_dtg_train_compatible, ensure_minimal_fields, validate_brep
    from .brep_loader import BrepParseError, _terminate_executor_workers, parse_step_file
    from .motif_feature_extractor import extract_motif_features
    from .utils_io import NODE_TYPES, RELATION_TYPES, ensure_workdir, load_dataset_split, make_uid, normalize_path, read_csv, read_pickle, scan_step_files, write_csv, write_json, write_jsonl
except ImportError:  # pragma: no cover
    from brep_cleaner import check_dtg_train_compatible, ensure_minimal_fields, validate_brep
    from brep_loader import BrepParseError, _terminate_executor_workers, parse_step_file
    from motif_feature_extractor import extract_motif_features
    from utils_io import NODE_TYPES, RELATION_TYPES, ensure_workdir, load_dataset_split, make_uid, normalize_path, read_csv, read_pickle, scan_step_files, write_csv, write_json, write_jsonl


STRUCTURAL_RELATION_TYPES = {
    "parallel_to",
    "opposite_to",
    "orthogonal_to",
    "coplanar_with",
    "smooth_connected",
    "repeated_with",
    "bounded_by",
}

SUPPORT_RELATION_TYPES = {"embedded_in"}
TOPOLOGY_SUPPORT_RELATION_TYPES = {"adjacent_to"}

PRIOR_NODE_TYPES = {
    "sheet_region",
    "loop_or_hole",
    "repeated_feature",
}

PRIOR_RELATION_TYPES = {
    "thin_wall_pair",
    "hosted_by",
}

PRIOR_NODE_CONFIDENCE_MIN = {
    "sheet_region": 0.58,
    "loop_or_hole": 0.58,
    "repeated_feature": 0.55,
}

PRIOR_RELATION_CONFIDENCE_MIN = {
    "thin_wall_pair": 0.58,
    "hosted_by": 0.58,
}

EMBEDDED_IN_NODE_CAPS = {
    "sheet_region": 1,
    "loop_or_hole": 1,
    "transition_group": 1,
    "repeated_feature": 3,
    "boundary_group": 1,
}


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def groups(self) -> List[List[int]]:
        buckets: Dict[int, List[int]] = {}
        for idx in range(len(self.parent)):
            buckets.setdefault(self.find(idx), []).append(idx)
        return [sorted(vals) for vals in buckets.values()]


def _relation_pair_key(rel: Dict[str, Any]) -> Tuple[int, int]:
    a = int(rel.get("source_face", rel.get("face_pair", [0, 0])[0]))
    b = int(rel.get("target_face", rel.get("face_pair", [a, a])[1]))
    return (min(a, b), max(a, b))


def _relation_type_sets(face_relations: Sequence[Dict[str, Any]]) -> Dict[Tuple[int, int], Set[str]]:
    pair_types: Dict[Tuple[int, int], Set[str]] = {}
    for rel in face_relations:
        pair_types.setdefault(_relation_pair_key(rel), set()).add(str(rel.get("type", "")))
    return pair_types


def _safe_normal(normals: Sequence[Sequence[float]]) -> List[float]:
    arr = np.asarray(normals, dtype=np.float32)
    if arr.size == 0:
        return [1.0, 0.0, 0.0]
    n = np.mean(arr.reshape(-1, 3), axis=0)
    norm = float(np.linalg.norm(n))
    if norm <= 1e-8:
        n = arr.reshape(-1, 3)[0]
        norm = float(np.linalg.norm(n))
    if norm <= 1e-8:
        return [1.0, 0.0, 0.0]
    return (n / norm).astype(float).tolist()


def _group_features(face_ids: Sequence[int], face_features: Sequence[Dict[str, Any]], global_scale: float, face_types: np.ndarray, face_wcs: np.ndarray | None = None) -> Dict[str, Any]:
    faces = [face_features[fid] for fid in face_ids]
    bboxes = np.asarray([f["bbox"] for f in faces], dtype=np.float32)
    centroids = np.asarray([f["centroid"] for f in faces], dtype=np.float32)
    mn = np.min(bboxes[:, :3], axis=0)
    mx = np.max(bboxes[:, 3:], axis=0)
    dims = np.maximum(mx - mn, 0.0)
    
    # 核心改进：拼接面片三维点云执行 PCA 求解严格旋转不变的 OBB 包围盒主维度与薄度
    local_dims = dims
    if face_wcs is not None and face_wcs.ndim == 4:
        group_pts = []
        for fid in face_ids:
            if fid < face_wcs.shape[0]:
                group_pts.append(face_wcs[fid].reshape(-1, 3))
        if group_pts:
            pts = np.concatenate(group_pts, axis=0).astype(np.float64)
            if pts.shape[0] >= 3:
                try:
                    pts = pts - np.mean(pts, axis=0, keepdims=True)
                    _, _, vh = np.linalg.svd(pts, full_matrices=False)
                    pts_local = pts @ vh.T
                    local_dims = np.max(pts_local, axis=0) - np.min(pts_local, axis=0)
                except Exception:
                    pass
    
    sorted_dims = np.sort(np.maximum(local_dims, 1e-8))
    area_sum = float(sum(float(f["area_proxy"]) for f in faces))
    relative_area_sum = float(sum(float(f.get("relative_area", 0.0)) for f in faces))
    boundary_ratio = float(sum(1 for f in faces if f.get("boundary_flag")) / max(len(faces), 1))
    aspect_values = [float(f.get("aspect_ratio", 1.0)) for f in faces]
    thinness_values = [float(f.get("bbox_thinness", 1.0)) for f in faces]
    adjacency_faces = sorted(
        {
            int(adj)
            for f in faces
            for adj in f.get("adjacency_faces", [])
            if int(adj) not in set(face_ids)
        }
    )
    
    # 计算新增的四个属性：surface_family, planarity_score, curvature_level, orientation_definition
    total_area = float(sum(float(f["area_proxy"]) for f in faces))
    plane_area = float(sum(float(f["area_proxy"]) for f in faces if int(face_types[f["face_id"]]) == 0))
    cyl_area = float(sum(float(f["area_proxy"]) for f in faces if int(face_types[f["face_id"]]) == 1))
    
    planarity_score = float(plane_area / max(total_area, 1e-8))
    
    if planarity_score >= 0.8:
        surface_family = "plane"
    elif cyl_area > plane_area and cyl_area > 0.5 * total_area:
        surface_family = "cylinder"
    else:
        surface_family = "complex"
        
    curvature_values = [float(f.get("mean_curvature")) for f in faces if f.get("mean_curvature") is not None]
    curvature_level = float(np.mean(curvature_values)) if curvature_values else 0.0
    
    normal_proxy = _safe_normal([f["normal_proxy"] for f in faces])
    axis_values = [f["axis_proxy"] for f in faces if f.get("axis_proxy") is not None]
    axis_proxy = _safe_normal(axis_values) if axis_values else [0.0, 0.0, 0.0]
    
    if surface_family == "plane":
        orientation_definition = {"type": "normal", "vector": normal_proxy}
    elif surface_family == "cylinder":
        orientation_definition = {"type": "axis", "vector": axis_proxy}
    else:
        orientation_definition = {"type": "complex", "vector": [0.0, 0.0, 0.0]}

    return {
        "face_count": len(face_ids),
        "centroid": np.mean(centroids, axis=0).astype(float).tolist(),
        "bbox": np.concatenate([mn, mx]).astype(float).tolist(),
        "bbox_dims": dims.astype(float).tolist(),
        "bbox_dims_sorted": sorted_dims.astype(float).tolist(),
        "bbox_extent_ratio": float(sorted_dims[-1] / max(global_scale, 1e-8)),
        "bbox_thinness": float(sorted_dims[0] / max(sorted_dims[-1], 1e-8)),
        "area_proxy_sum": area_sum,
        "relative_area_sum": relative_area_sum,
        "normal_proxy": normal_proxy,
        "mean_aspect_ratio": float(np.mean(aspect_values)) if aspect_values else 1.0,
        "max_aspect_ratio": float(np.max(aspect_values)) if aspect_values else 1.0,
        "mean_face_degree": float(np.mean([float(f.get("face_degree", 0)) for f in faces])) if faces else 0.0,
        "boundary_ratio": boundary_ratio,
        "boundary_flag": bool(boundary_ratio >= 0.5),
        "adjacent_faces_outside": adjacency_faces,
        "curvature_proxy_mean": float(np.mean([float(f.get("curvature_proxy", 0.0)) for f in faces])) if faces else 0.0,
        "mean_curvature_mean": float(np.mean([float(f.get("mean_curvature")) for f in faces if f.get("mean_curvature") is not None])) if any(f.get("mean_curvature") is not None for f in faces) else None,
        "max_curvature_mean": float(np.mean([float(f.get("max_curvature")) for f in faces if f.get("max_curvature") is not None])) if any(f.get("max_curvature") is not None for f in faces) else None,
        "var_curvature_mean": float(np.mean([float(f.get("var_curvature")) for f in faces if f.get("var_curvature") is not None])) if any(f.get("var_curvature") is not None for f in faces) else None,
        "gaussian_sign_mean": float(np.mean([float(f.get("gaussian_sign")) for f in faces if f.get("gaussian_sign") is not None])) if any(f.get("gaussian_sign") is not None for f in faces) else None,
        "planarity_score": planarity_score,
        "surface_family": surface_family,
        "curvature_level": curvature_level,
        "orientation_definition": orientation_definition,
    }


def _is_cylinder_hole(fid: int, features: Dict[str, Any]) -> bool:
    sampling_quality = features.get("geometry_sampling_quality", "unknown")
    if sampling_quality != "true_or_dtg_sampling":
        return False
    face_wcs = features.get("face_wcs")
    face_types = features.get("face_surface_type", [])
    locations = features.get("face_cylinder_location", [])
    axes = features.get("face_cylinder_axis", [])
    if face_wcs is not None and fid < len(face_types) and face_types[fid] == 1 and fid < len(locations) and fid < len(axes):
        face_wcs = np.asarray(face_wcs, dtype=np.float32)
        if fid < face_wcs.shape[0]:
            grid = face_wcs[fid]
            p = grid[16, 16]
            c0 = np.asarray(locations[fid], dtype=np.float32)
            a = np.asarray(axes[fid], dtype=np.float32)
            norm_a = np.linalg.norm(a)
            if norm_a > 1e-5:
                a = a / norm_a
                v0 = p - c0
                v_radial = v0 - np.dot(v0, a) * a
                norm_radial = np.linalg.norm(v_radial)
                if norm_radial > 1e-5:
                    v_radial = v_radial / norm_radial
                    v1 = grid[17, 16] - grid[15, 16]
                    v2 = grid[16, 17] - grid[16, 15]
                    n_local = np.cross(v1, v2)
                    norm_n = np.linalg.norm(n_local)
                    if norm_n > 1e-5:
                        n_local = n_local / norm_n
                        if np.dot(v_radial, n_local) < -0.2:
                            return True
    return False


def _build_base_face_groups(features: Dict[str, Any]) -> List[List[int]]:
    face_count = int(features.get("face_count", 0))
    uf = UnionFind(face_count)
    pair_types = _relation_type_sets(features.get("face_relations", []))
    
    # 获取面类型与解析几何参数进行有针对性的相邻拼接合并
    face_types = features.get("face_surface_type")
    if face_types is None:
        face_types = [-1] * face_count
        
    radii = features.get("face_cylinder_radius")
    if radii is None:
        radii = [0.0] * face_count
        
    axes = features.get("face_cylinder_axis")
    if axes is None:
        axes = [[0.0, 0.0, 0.0]] * face_count
        
    locations = features.get("face_cylinder_location")
    if locations is None:
        locations = [[0.0, 0.0, 0.0]] * face_count
        
    for (i, j), types in pair_types.items():
        if "adjacent_to" in types:
            type_i = face_types[i] if i < len(face_types) else 0
            type_j = face_types[j] if j < len(face_types) else 0
            
            # 平面之间：通过相邻且共面合并
            if type_i == 0 and type_j == 0 and "coplanar_with" in types:
                uf.union(i, j)
            # 圆柱面之间：合并相邻分瓣面，但必须满足轴线平行且共线、以及半径一致的严格几何约束
            elif type_i == 1 and type_j == 1:
                # 1. 校验轴线平行度（允许反向，点积绝对值 >= cos(1度)）
                axis_i = np.asarray(axes[i], dtype=np.float32)
                axis_j = np.asarray(axes[j], dtype=np.float32)
                norm_i = np.linalg.norm(axis_i)
                norm_j = np.linalg.norm(axis_j)
                axis_parallel = False
                a_i = None
                if norm_i > 1e-5 and norm_j > 1e-5:
                    a_i = axis_i / norm_i
                    a_j = axis_j / norm_j
                    axis_parallel = abs(np.dot(a_i, a_j)) >= np.cos(np.radians(1.0))
                
                # 2. 校验半径一致性（相对误差 < 2%）
                r_i = float(radii[i])
                r_j = float(radii[j])
                radius_equal = False
                if r_i > 1e-5 and r_j > 1e-5:
                    radius_equal = abs(r_i - r_j) / max(r_i, r_j) < 0.02
                
                # 3. 校验轴线共线性（计算两条轴线间距距离 <= 1e-3）
                coaxial = False
                if axis_parallel and a_i is not None:
                    loc_i = np.asarray(locations[i], dtype=np.float32)
                    loc_j = np.asarray(locations[j], dtype=np.float32)
                    v = loc_j - loc_i
                    v_proj = np.dot(v, a_i) * a_i
                    v_perp = v - v_proj
                    dist_axis = float(np.linalg.norm(v_perp))
                    if dist_axis <= 1e-3:
                        coaxial = True
                
                if axis_parallel and radius_equal and coaxial:
                    uf.union(i, j)
                
    groups = uf.groups()
    return sorted(groups, key=lambda item: (min(item), len(item)))


def _percentile(values: Sequence[float], q: float, default: float = 0.0) -> float:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    if not vals:
        return default
    # Keep threshold precision consistent with the float64 feature values.
    # A float32 percentile can round just above an equal-area face and make a
    # mathematically inclusive ``value >= threshold`` test fail.
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q))


def _node_type_count(nodes: Sequence[Dict[str, Any]], node_type: str) -> int:
    return sum(1 for node in nodes if node.get("type") == node_type)


def _relation_confidence(relations: Sequence[Dict[str, Any]]) -> float:
    if not relations:
        return 0.0
    vals = [float(rel.get("confidence", 0.0)) for rel in relations]
    return float(max(vals) * 0.65 + np.mean(vals) * 0.35)


def _spacing_regular_score(centroids: np.ndarray, member_directions: np.ndarray = None) -> Dict[str, Any]:
    N = centroids.shape[0]
    if N <= 1:
        return {
            "pattern_type": "irregular",
            "count": N,
            "direction": [0.0, 0.0, 0.0],
            "pitch": 0.0,
            "pattern_residual": 0.0,
            "regular_score": 0.0
        }
        
    if N == 2:
        v = centroids[1] - centroids[0]
        dist = float(np.linalg.norm(v))
        direction = (v / max(dist, 1e-8)).tolist()
        
        # 校验镜像对称性：仅当两个特征满足反射矩阵重合时，才标记为 mirror，否则标记为 pair（对特征）
        is_validated_mirror = False
        if member_directions is not None and len(member_directions) == 2:
            n1 = np.asarray(member_directions[0], dtype=np.float32)
            n2 = np.asarray(member_directions[1], dtype=np.float32)
            norm1 = np.linalg.norm(n1)
            norm2 = np.linalg.norm(n2)
            if norm1 > 1e-5 and norm2 > 1e-5:
                n1 = n1 / norm1
                n2 = n2 / norm2
                nsym = np.asarray(direction, dtype=np.float32)
                # 计算反射矩阵作用下的 n1: n1_ref = n1 - 2 * (n1 . nsym) * nsym
                n1_ref = n1 - 2.0 * np.dot(n1, nsym) * nsym
                if abs(np.dot(n1_ref, n2)) >= 0.95:
                    is_validated_mirror = True
                    
        pattern_type = "mirror" if is_validated_mirror else "pair"
        return {
            "pattern_type": pattern_type,
            "count": 2,
            "direction": direction,
            "pitch": dist,
            "pattern_residual": 0.0,
            # Two arbitrary instances do not establish a repeat pattern.  A
            # validated mirror pair is useful evidence; an unverified pair is
            # retained only as an audit result and is rejected downstream.
            "regular_score": 0.75 if is_validated_mirror else 0.25
        }
        
    centered = centroids - np.mean(centroids, axis=0, keepdims=True)
    try:
        _, svals, vh = np.linalg.svd(centered, full_matrices=False)
        total_var = float(np.sum(svals**2))
        linearity = float((svals[0] ** 2) / max(total_var, 1e-8))
        plane_ratio = float((svals[0]**2 + svals[1]**2) / max(total_var, 1e-8))
        
        # 1. 线性阵列 (Linear Pattern)
        if linearity >= 0.88:
            direction = vh[0].tolist()
            proj = np.sort(centered @ vh[0])
            gaps = np.diff(proj)
            pitch = float(np.mean(gaps))
            spacing_cv = float(np.std(gaps) / max(pitch, 1e-8))
            regular_score = float(max(0.0, min(1.0, 1.0 - spacing_cv)))
            return {
                "pattern_type": "linear",
                "count": N,
                "direction": direction,
                "pitch": pitch,
                "pattern_residual": spacing_cv,
                "regular_score": regular_score
            }
            
        # 2. 共面非线性阵列 (Radial or Grid Pattern)
        if plane_ratio >= 0.95:
            pts_2d = centered @ vh[:2].T  # 投影到 2D 主平面，形如 (N, 2)
            
            # 圆周阵列检测 (Radial Pattern)
            radii = np.linalg.norm(pts_2d, axis=1)
            r_mean = float(np.mean(radii))
            r_std = float(np.std(radii))
            cv_r = r_std / max(r_mean, 1e-8)
            
            if cv_r <= 0.08:
                # 拟合成功：点集基本处于同心圆周上
                normal = np.cross(vh[0], vh[1])
                normal = (normal / max(float(np.linalg.norm(normal)), 1e-8)).tolist()
                angles = np.sort(np.arctan2(pts_2d[:, 1], pts_2d[:, 0]))
                ang_gaps = np.diff(angles)
                wrap_gap = 2.0 * np.pi - (angles[-1] - angles[0])
                all_ang_gaps = np.append(ang_gaps, wrap_gap)
                mean_ang_gap = float(np.mean(all_ang_gaps))
                cv_ang = float(np.std(all_ang_gaps) / max(mean_ang_gap, 1e-8))
                
                regular_score = float(max(0.0, min(1.0, (1.0 - cv_r) * (1.0 - cv_ang))))
                return {
                    "pattern_type": "radial",
                    "count": N,
                    "direction": normal,
                    "pitch": r_mean,  # 圆周阵列的 pitch 记为圆半径
                    "pattern_residual": float(cv_r + cv_ang),
                    "regular_score": regular_score
                }
                
            # 二维网格阵列检测 (Grid Pattern)
            if N >= 4:
                proj1 = np.sort(pts_2d[:, 0])
                proj2 = np.sort(pts_2d[:, 1])
                
                # 提取两个主方向上的间隔
                diffs1 = np.diff(proj1)
                diffs1_nonzero = diffs1[diffs1 > 0.05 * (proj1[-1] - proj1[0])]
                diffs2 = np.diff(proj2)
                diffs2_nonzero = diffs2[diffs2 > 0.05 * (proj2[-1] - proj2[0])]
                
                if len(diffs1_nonzero) > 0 and len(diffs2_nonzero) > 0:
                    pitch1 = float(np.mean(diffs1_nonzero))
                    pitch2 = float(np.mean(diffs2_nonzero))
                    cv1 = float(np.std(diffs1_nonzero) / pitch1) if len(diffs1_nonzero) > 1 else 0.0
                    cv2 = float(np.std(diffs2_nonzero) / pitch2) if len(diffs2_nonzero) > 1 else 0.0
                    
                    if cv1 <= 0.18 and cv2 <= 0.18:
                        regular_score = float(max(0.0, min(1.0, (1.0 - cv1) * (1.0 - cv2))))
                        return {
                            "pattern_type": "grid",
                            "count": N,
                            "direction": [vh[0].tolist(), vh[1].tolist()],
                            "pitch": [pitch1, pitch2],
                            "pattern_residual": float(cv1 + cv2),
                            "regular_score": regular_score
                        }
                        
        return {
            "pattern_type": "irregular",
            "count": N,
            "direction": vh[0].tolist(),
            "pitch": 0.0,
            "pattern_residual": 1.0,
            "regular_score": 0.1
        }
    except Exception:
        return {
            "pattern_type": "irregular",
            "count": N,
            "direction": [1.0, 0.0, 0.0],
            "pitch": 0.0,
            "pattern_residual": 1.0,
            "regular_score": 0.0
        }


def _similar_group_signature(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    fa = a["features"]
    fb = b["features"]
    dims_a = np.asarray(fa.get("bbox_dims_sorted", [0.0, 0.0, 0.0]), dtype=np.float32)
    dims_b = np.asarray(fb.get("bbox_dims_sorted", [0.0, 0.0, 0.0]), dtype=np.float32)
    dims_den = np.maximum(np.maximum(dims_a, dims_b), 1e-8)
    dims_rel_diff = float(np.mean(np.abs(dims_a - dims_b) / dims_den))
    area_a = float(fa.get("area_proxy_sum", 0.0))
    area_b = float(fb.get("area_proxy_sum", 0.0))
    area_ratio = float(min(area_a, area_b) / max(max(area_a, area_b), 1e-8))
    n_a = np.asarray(fa.get("normal_proxy", [1.0, 0.0, 0.0]), dtype=np.float32)
    n_b = np.asarray(fb.get("normal_proxy", [1.0, 0.0, 0.0]), dtype=np.float32)
    normal_absdot = float(abs(np.dot(n_a, n_b)))
    face_count_gap = abs(int(fa.get("face_count", 1)) - int(fb.get("face_count", 1)))
    degree_gap = abs(float(fa.get("mean_face_degree", 0.0)) - float(fb.get("mean_face_degree", 0.0)))
    surface_a = str(fa.get("surface_family", "unknown"))
    surface_b = str(fb.get("surface_family", "unknown"))
    planarity_gap = abs(float(fa.get("planarity_score", 0.0)) - float(fb.get("planarity_score", 0.0)))
    evidence = {
        "dims_rel_diff": dims_rel_diff,
        "area_ratio": area_ratio,
        "normal_absdot": normal_absdot,
        "face_count_gap": float(face_count_gap),
        "degree_gap": degree_gap,
        "surface_family_a": surface_a,
        "surface_family_b": surface_b,
        "planarity_gap": planarity_gap,
    }
    similar = (
        surface_a == surface_b
        and dims_rel_diff <= 0.22
        and area_ratio >= 0.72
        and normal_absdot >= 0.94
        and face_count_gap <= 1
        and degree_gap <= 2.0
        and planarity_gap <= 0.15
    )
    return similar, evidence


def _add_relation(
    relations: Dict[Tuple[str, str, str], Dict[str, Any]],
    source: str,
    target: str,
    relation_type: str,
    confidence: float,
    evidence: Dict[str, Any],
) -> None:
    if source == target or relation_type not in RELATION_TYPES:
        return
    key = (source, target, relation_type)
    record = {
        "source": source,
        "target": target,
        "type": relation_type,
        "confidence": float(max(0.0, min(0.99, confidence))),
        "evidence": evidence,
    }
    if key not in relations or record["confidence"] > float(relations[key].get("confidence", 0.0)):
        relations[key] = record


def _relation_role(relation_type: str) -> str:
    if relation_type == "embedded_in":
        return "support"
    if relation_type == "adjacent_to":
        return "topology_support"
    return "structural"


def _annotate_relation_roles(relations: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    annotated: List[Dict[str, Any]] = []
    for rel in relations:
        record = dict(rel)
        record["relation_role"] = _relation_role(str(record.get("type", "")))
        annotated.append(record)
    return annotated


def _embedded_supported_node_id(rel: Dict[str, Any], node_by_id: Dict[str, Dict[str, Any]]) -> str:
    src = str(rel.get("source"))
    dst = str(rel.get("target"))
    dst_type = str(node_by_id.get(dst, {}).get("type", ""))
    src_type = str(node_by_id.get(src, {}).get("type", ""))
    if dst_type != "face_group":
        return dst
    if src_type != "face_group":
        return src
    return dst


def _embedded_support_score(rel: Dict[str, Any]) -> Tuple[float, int, int]:
    evidence = rel.get("evidence", {}) or {}
    overlap = evidence.get("overlap_faces", []) or []
    if not isinstance(overlap, list):
        overlap = []
    explicit = 0
    if evidence.get("rule"):
        explicit += 2
    if evidence.get("support_status"):
        explicit += 2
    if evidence.get("compressed_graph_added"):
        explicit += 1
    return (float(rel.get("confidence", 0.0)), int(len(overlap)), explicit)


def _limit_embedded_in_relations(
    nodes: Sequence[Dict[str, Any]],
    relations: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    node_by_id = {str(node.get("id")): node for node in nodes}
    non_embedded = [dict(rel) for rel in relations if str(rel.get("type")) != "embedded_in"]
    embedded = [dict(rel) for rel in relations if str(rel.get("type")) == "embedded_in"]
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for rel in embedded:
        supported_id = _embedded_supported_node_id(rel, node_by_id)
        grouped.setdefault(supported_id, []).append(rel)

    kept_embedded: List[Dict[str, Any]] = []
    per_node_caps: Dict[str, int] = {}
    for supported_id, rels in grouped.items():
        supported_type = str(node_by_id.get(supported_id, {}).get("type", "face_group"))
        cap = EMBEDDED_IN_NODE_CAPS.get(supported_type, 1 if supported_type != "face_group" else 0)
        per_node_caps[supported_id] = int(cap)
        if cap <= 0:
            continue
        ordered = sorted(
            rels,
            key=lambda rel: (
                -_embedded_support_score(rel)[0],
                -_embedded_support_score(rel)[1],
                -_embedded_support_score(rel)[2],
                str(rel.get("source")),
                str(rel.get("target")),
            ),
        )
        kept_embedded.extend(ordered[:cap])

    node_count = len(nodes)
    global_cap = int(max(0, np.floor(1.2 * max(node_count, 0))))
    if len(kept_embedded) > global_cap:
        kept_embedded = sorted(
            kept_embedded,
            key=lambda rel: (
                -_embedded_support_score(rel)[0],
                -_embedded_support_score(rel)[1],
                -_embedded_support_score(rel)[2],
                str(rel.get("source")),
                str(rel.get("target")),
            ),
        )[:global_cap]

    kept = _annotate_relation_roles(non_embedded + kept_embedded)
    kept = sorted(kept, key=lambda rel: (str(rel.get("source")), str(rel.get("target")), str(rel.get("type"))))
    info = {
        "raw_embedded_in_count": len(embedded),
        "kept_embedded_in_count": len(kept_embedded),
        "embedded_in_global_cap": global_cap,
        "embedded_in_per_node_caps": per_node_caps,
        "policy": "对每个非 face_group motif node 限制 embedded_in 支撑边数量，并全局限制为 1.2 * motif node count",
    }
    return kept, info


def _relation_role_stats(nodes: Sequence[Dict[str, Any]], relations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    node_count = max(len(nodes), 1)
    structural_count = sum(1 for rel in relations if rel.get("relation_role") == "structural")
    support_count = sum(1 for rel in relations if rel.get("relation_role") == "support")
    topology_support_count = sum(1 for rel in relations if rel.get("relation_role") == "topology_support")
    embedded_count = sum(1 for rel in relations if rel.get("type") == "embedded_in")
    return {
        "num_structural_relations": structural_count,
        "num_support_relations": support_count,
        "num_topology_support_relations": topology_support_count,
        "structural_relation_density": float(structural_count / node_count),
        "support_relation_density": float(support_count / node_count),
        "embedded_in_per_node": float(embedded_count / node_count),
        "embedded_in_per_sample": int(embedded_count),
    }


def _refresh_relation_views(graph: Dict[str, Any], relations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    graph = copy.deepcopy(graph)
    relations = _annotate_relation_roles(relations)
    graph["motif_relations"] = relations
    relation_type_counts = {typ: sum(1 for rel in relations if rel.get("type") == typ) for typ in RELATION_TYPES}
    role_stats = _relation_role_stats(graph.get("motif_nodes", []), relations)
    stats = dict(graph.get("motif_stats", {}))
    stats["relation_type_counts"] = relation_type_counts
    stats["relation_role_counts"] = {
        "structural": role_stats["num_structural_relations"],
        "support": role_stats["num_support_relations"],
        "topology_support": role_stats["num_topology_support_relations"],
    }
    stats.update(role_stats)
    graph["motif_stats"] = stats
    motif_prior = dict(graph.get("motif_prior", {}))
    motif_prior["motif_relation_type_ids"] = [RELATION_TYPES.index(rel["type"]) for rel in relations]
    motif_prior["relation_role_vocab"] = ["structural", "support", "topology_support"]
    role_vocab = motif_prior["relation_role_vocab"]
    motif_prior["motif_relation_role_ids"] = [role_vocab.index(str(rel.get("relation_role", "structural"))) for rel in relations]
    graph["motif_prior"] = motif_prior
    return graph


def make_structural_graph(graph: Dict[str, Any]) -> Dict[str, Any]:
    structural_relations = [
        rel
        for rel in graph.get("motif_relations", [])
        if str(rel.get("relation_role", _relation_role(str(rel.get("type", ""))))) == "structural"
    ]
    structural_graph = _refresh_relation_views(graph, structural_relations)
    structural_graph["graph_view"] = "structural_only"
    structural_graph["motif_prior"]["policy"] = "structural_relations_only_default_training_view"
    structural_graph["motif_prior"]["support_relations_available_in"] = "仅少量 examples 中保留；正式紧凑数据不持久化完整支撑图"
    return structural_graph


def _prior_node_keep(node: Dict[str, Any]) -> bool:
    node_type = str(node.get("type", ""))
    if node_type not in PRIOR_NODE_TYPES:
        return False
    confidence = float(node.get("confidence", 0.0))
    return confidence >= PRIOR_NODE_CONFIDENCE_MIN.get(node_type, 0.65)


def _prior_relation_sort_key(rel: Dict[str, Any]) -> Tuple[int, float, str, str, str]:
    priority = {
        "thin_wall_pair": 0,
        "hosted_by": 1,
    }
    return (
        priority.get(str(rel.get("type", "")), 99),
        -float(rel.get("confidence", 0.0)),
        str(rel.get("source", "")),
        str(rel.get("target", "")),
        str(rel.get("type", "")),
    )


def _prune_prior_relations(relations: Sequence[Dict[str, Any]], node_count: int) -> List[Dict[str, Any]]:
    per_type_degree_caps = {
        "thin_wall_pair": 2,
        "hosted_by": 2,
    }
    max_edges = max(2, min(2 * max(node_count, 1), 32))
    degree_by_type: Dict[Tuple[str, str], int] = {}
    kept: List[Dict[str, Any]] = []
    for rel in sorted(relations, key=_prior_relation_sort_key):
        rel_type = str(rel.get("type", ""))
        if rel_type not in PRIOR_RELATION_TYPES:
            continue
        if float(rel.get("confidence", 0.0)) < PRIOR_RELATION_CONFIDENCE_MIN.get(rel_type, 0.65):
            continue
        cap = per_type_degree_caps.get(rel_type, 2)
        src = str(rel.get("source", ""))
        dst = str(rel.get("target", ""))
        source_full = degree_by_type.get((src, rel_type), 0) >= cap
        target_full = degree_by_type.get((dst, rel_type), 0) >= cap
        # A local feature may attach to at most two host sheets; one sheet can
        # legitimately host several different local features.
        if source_full or (rel_type != "hosted_by" and target_full):
            continue
        kept.append(dict(rel))
        degree_by_type[(src, rel_type)] = degree_by_type.get((src, rel_type), 0) + 1
        if rel_type != "hosted_by":
            degree_by_type[(dst, rel_type)] = degree_by_type.get((dst, rel_type), 0) + 1
        if len(kept) >= max_edges:
            break
    return kept


def _make_prior_stats(graph: Dict[str, Any], nodes: Sequence[Dict[str, Any]], relations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    num_faces = max(int(graph.get("num_faces", 0)), 1)
    candidate_nodes = max(len(graph.get("motif_nodes", [])), 1)
    candidate_relations = max(len(graph.get("motif_relations", [])), 1)
    candidate_size = candidate_nodes + candidate_relations
    size_prior = len(nodes) + len(relations)
    retention = float(size_prior / candidate_size)
    covered_faces = sorted({int(fid) for node in nodes for fid in node.get("face_ids", [])})
    prior_types = sorted({str(node.get("type", "")) for node in nodes})
    prior_relation_types = sorted({str(rel.get("type", "")) for rel in relations})
    
    reduction = 1.0 - retention
    compression = max(0.0, reduction)
    core_types = [typ for typ in prior_types if typ in PRIOR_NODE_TYPES]

    # 结合高精先验正确性的严格校验
    r_isolated = 0.0
    if nodes:
        node_ids_in_relations = set()
        for rel in relations:
            node_ids_in_relations.add(str(rel.get("source")))
            node_ids_in_relations.add(str(rel.get("target")))
        num_isolated = sum(1 for node in nodes if str(node.get("id")) not in node_ids_in_relations)
        r_isolated = float(num_isolated / len(nodes))
        
    r_coverage = float(len(covered_faces) / num_faces)
    
    r_redundancy = 0.0
    sum_m_f = 0
    sum_max_m_f_minus_1 = 0
    if nodes:
        face_counts = {}
        for node in nodes:
            for fid in node.get("face_ids", []):
                face_counts[fid] = face_counts.get(fid, 0) + 1
        sum_m_f = sum(face_counts.values())
        sum_max_m_f_minus_1 = sum(max(m - 1, 0) for m in face_counts.values())
        r_redundancy = float(sum_max_m_f_minus_1 / sum_m_f) if sum_m_f > 0 else 0.0

    prior_ready = bool(
        "sheet_region" in core_types
        and len(nodes) >= 1
        and len(relations) <= max(4, 2 * max(len(nodes), 1))
        and r_coverage >= 0.25       # 简单实体不因缺少局部 motif 被误判为低质量
        and r_redundancy <= 0.60     # 面片过度覆盖冗余度上限
    )
    return {
        "num_prior_nodes": len(nodes),
        "num_prior_relations": len(relations),
        "prior_node_density": float(len(nodes) / num_faces),
        "prior_relation_density": float(len(relations) / max(len(nodes), 1)),
        "prior_retention_ratio": retention,
        "prior_reduction_ratio": reduction,
        "prior_compression_ratio": compression,
        "prior_size_change_ratio": retention - 1.0,
        "prior_is_size_reduced": bool(retention <= 1.0),
        "ratio_Mraw_to_S": retention,
        "prior_node_retention_ratio": float(len(nodes) / candidate_nodes),
        "prior_node_compression_ratio": max(0.0, 1.0 - float(len(nodes) / candidate_nodes)),
        "prior_relation_retention_ratio": float(len(relations) / candidate_relations),
        "prior_relation_compression_ratio": max(0.0, 1.0 - float(len(relations) / candidate_relations)),
        "prior_node_to_face_ratio": float(len(nodes) / num_faces),
        "prior_node_face_reduction_ratio": max(0.0, 1.0 - float(len(nodes) / num_faces)),
        "prior_coverage_faces": len(covered_faces),
        "prior_coverage_face_ratio": float(len(covered_faces) / num_faces),
        "prior_isolated_ratio": r_isolated,
        "prior_redundancy_ratio": r_redundancy,
        "prior_motif_types": prior_types,
        "prior_relation_types": prior_relation_types,
        "prior_ready": prior_ready,
    }


def make_motif_prior_graph(graph: Dict[str, Any]) -> Dict[str, Any]:
    """Project the candidate audit graph M_raw to the reliable prior S."""
    raw_nodes = list(graph.get("motif_nodes", []))
    node_by_id = {str(node.get("id")): node for node in raw_nodes}
    selected_raw_nodes = [copy.deepcopy(node) for node in raw_nodes if _prior_node_keep(node)]

    relation_records: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    # Map physical faces to the reliable high-level nodes.  v2 constructs its
    # two semantic relations directly from thin-wall and hosting evidence; no
    # Cartesian promotion of orientation labels is performed.
    face_to_structural: Dict[int, Set[str]] = {}
    for node in selected_raw_nodes:
        node_id = str(node.get("id"))
        for fid in node.get("face_ids", []):
            face_to_structural.setdefault(int(fid), set()).add(node_id)

    # 1. Link thin-wall sheet pairs.
    thin_wall_nodes = [node for node in raw_nodes if node.get("type") == "thin_wall_pair"]
    for tw_node in thin_wall_nodes:
        pair_faces = tw_node.get("features", {}).get("pair_face_ids", [])
        if len(pair_faces) == 2:
            a, b = pair_faces[0], pair_faces[1]
            a_structural = face_to_structural.get(int(a), set())
            b_structural = face_to_structural.get(int(b), set())
            
            for u_s in a_structural:
                for v_s in b_structural:
                    if u_s == v_s:
                        continue
                    u_node = node_by_id.get(u_s)
                    v_node = node_by_id.get(v_s)
                    if u_node and v_node and u_node.get("type") == "sheet_region" and v_node.get("type") == "sheet_region":
                        src, dst = sorted((u_s, v_s))
                        key = (src, dst, "thin_wall_pair")
                        relation_records[key] = {
                            "source": src,
                            "target": dst,
                            "type": "thin_wall_pair",
                            "confidence": float(tw_node.get("confidence", 0.85)),
                            "relation_role": "structural",
                            "evidence": {
                                "rule": "由 thin_wall_pair 节点转换来的两个 sheet 之间的薄壁对关系",
                                "normal_gap": tw_node.get("features", {}).get("normal_gap"),
                                "area_ratio_min_over_max": tw_node.get("features", {}).get("area_ratio_min_over_max"),
                            }
                        }
                
    # 2. 链接重复模式成员边 has_member
    repeated_nodes = [node for node in selected_raw_nodes if node.get("type") == "repeated_feature"]
    for rep_node in repeated_nodes:
        rep_id = str(rep_node.get("id"))
        for fid in rep_node.get("face_ids", []):
            for target_id in face_to_structural.get(int(fid), []):
                if target_id == rep_id:
                    continue
                target_node = node_by_id.get(target_id)
                if target_node and target_node.get("type") == "repeated_feature":
                    continue
                key = (rep_id, target_id, "has_member")
                relation_records[key] = {
                    "source": rep_id,
                    "target": target_id,
                    "type": "has_member",
                    "confidence": float(rep_node.get("confidence", 0.8)),
                    "relation_role": "structural",
                    "evidence": {
                        "rule": "链接模式节点与其成员节点",
                    }
                }

    # 3. 链接局部特征挂载边 hosted_by
    local_nodes = [node for node in selected_raw_nodes if node.get("type") in {"loop_or_hole", "transition_group"}]
    for loc_node in local_nodes:
        loc_id = str(loc_node.get("id"))
        adj_faces = loc_node.get("features", {}).get("adjacent_faces_outside", [])
        for adj_fid in adj_faces:
            for target_id in face_to_structural.get(int(adj_fid), []):
                other_node = node_by_id.get(target_id)
                if other_node and other_node.get("type") in {"sheet_region", "boundary_group"}:
                    key = (loc_id, target_id, "hosted_by")
                    relation_records[key] = {
                        "source": loc_id,
                        "target": target_id,
                        "type": "hosted_by",
                        "confidence": float(loc_node.get("confidence", 0.8)),
                        "relation_role": "structural",
                        "evidence": {
                            "rule": "Fillet/Hole is hosted_by an adjacent sheet or boundary",
                        }
                    }

    prior_relations_raw = _prune_prior_relations(list(relation_records.values()), len(selected_raw_nodes))
    boundary_connection_types = {"bounded_by", "hosted_by"}
    connected_boundary_ids = {
        str(rel.get(endpoint))
        for rel in prior_relations_raw
        if str(rel.get("type")) in boundary_connection_types
        for endpoint in ("source", "target")
    }
    pruned_nodes: List[Dict[str, Any]] = []
    for node in selected_raw_nodes:
        raw_id = str(node.get("id"))
        node_type = str(node.get("type", ""))
        if node_type == "boundary_group" and raw_id not in connected_boundary_ids:
            continue
        pruned_nodes.append(node)

    kept_raw_ids = {str(node.get("id")) for node in pruned_nodes}
    prior_relations_raw = [
        rel
        for rel in prior_relations_raw
        if str(rel.get("source")) in kept_raw_ids and str(rel.get("target")) in kept_raw_ids
    ]

    id_map = {str(node.get("id")): f"s{idx}" for idx, node in enumerate(pruned_nodes)}
    prior_nodes: List[Dict[str, Any]] = []
    for node in pruned_nodes:
        raw_id = str(node.get("id"))
        record = copy.deepcopy(node)
        record["id"] = id_map[raw_id]
        
        # 剥离底层悬空 ID 进入 support_metadata，保持 features 精炼与纯正的旋转不变性
        support_metadata = {
            "raw_motif_id": raw_id,
        }
        features = record.get("features", {})
        for k in ["base_face_group_id", "base_face_group_ids", "member_face_group_ids", "member_motif_ids", "larger_neighbor_group_ids"]:
            if k in features:
                support_metadata[k] = features.pop(k)
        record["support_metadata"] = support_metadata
        record["prior_role"] = "generation_prior_node"
        prior_nodes.append(record)

    prior_relations: List[Dict[str, Any]] = []
    for rel in prior_relations_raw:
        record = copy.deepcopy(rel)
        record["raw_source"] = str(rel.get("source", ""))
        record["raw_target"] = str(rel.get("target", ""))
        record["source"] = id_map[str(rel.get("source"))]
        record["target"] = id_map[str(rel.get("target"))]
        record["prior_role"] = "generation_prior_relation"
        prior_relations.append(record)

    # 蒸馏图完整性检验，防止悬空节点或悬空边
    prior_node_ids = {str(node["id"]) for node in prior_nodes}
    for rel in prior_relations:
        if rel["source"] not in prior_node_ids:
            raise ValueError(f"dangling prior relation source: {rel}")
        if rel["target"] not in prior_node_ids:
            raise ValueError(f"dangling prior relation target: {rel}")

    prior_graph = copy.deepcopy(graph)
    prior_graph["graph_view"] = "distilled_motif_prior"
    prior_graph["prior_definition"] = "S = R(M_raw)"
    prior_graph["source_graph_view"] = "raw_candidate_audit_graph"
    prior_graph["motif_nodes"] = prior_nodes
    prior_graph["motif_relations"] = _annotate_relation_roles(prior_relations)
    node_type_counts = {typ: _node_type_count(prior_nodes, typ) for typ in NODE_TYPES}
    relation_type_counts = {typ: sum(1 for rel in prior_graph["motif_relations"] if rel["type"] == typ) for typ in RELATION_TYPES}
    prior_stats = _make_prior_stats(graph, prior_nodes, prior_graph["motif_relations"])
    relation_role_stats = _relation_role_stats(prior_nodes, prior_graph["motif_relations"])
    stats = dict(prior_graph.get("motif_stats", {}))
    stats.update(
        {
            "node_type_counts": node_type_counts,
            "relation_type_counts": relation_type_counts,
            "relation_role_counts": {
                "structural": relation_role_stats["num_structural_relations"],
                "support": relation_role_stats["num_support_relations"],
                "topology_support": relation_role_stats["num_topology_support_relations"],
            },
            **relation_role_stats,
            **prior_stats,
        }
    )
    prior_graph["motif_stats"] = stats
    prior_quality = dict(prior_graph.get("motif_quality", {}))
    prior_quality["motif_prior_ready"] = bool(prior_stats["prior_ready"])
    prior_quality["prior_ready_policy"] = "reliable sheet anchor + coverage/redundancy constraints; local motifs and structural edges are optional"
    prior_graph["motif_quality"] = prior_quality
    face_to_prior_nodes: Dict[str, List[str]] = {}
    for node in prior_nodes:
        for fid in node.get("face_ids", []):
            face_to_prior_nodes.setdefault(str(fid), []).append(str(node.get("id")))
    prior_graph["motif_prior"] = {
        "version": "innovation1_compact_control_prior_v2",
        "policy": "three stable motifs with two semantic relations; membership is rebuilt downstream",
        "prior_symbol": "S",
        "source_symbol": "M_raw",
        "source_raw_symbol": "M_raw",
        "node_type_vocab": NODE_TYPES,
        "relation_type_vocab": RELATION_TYPES,
        "motif_node_type_ids": [NODE_TYPES.index(node["type"]) for node in prior_nodes],
        "motif_relation_type_ids": [RELATION_TYPES.index(rel["type"]) for rel in prior_graph["motif_relations"]],
        "relation_role_vocab": ["structural", "support", "topology_support"],
        "motif_relation_role_ids": [0 for _ in prior_graph["motif_relations"]],
        "face_to_motif_nodes": face_to_prior_nodes,
        "distillation_policy": "M_raw 保留候选审计；正式 S 直接保留可靠 sheet/loop/repeat 与 hosted/thin-wall，创新点2再重建 embedded_in。",
    }
    prior_graph["motif_prior_distillation"] = {
        "raw_graph_node_count": len(graph.get("motif_nodes", [])),
        "raw_graph_relation_count": len(graph.get("motif_relations", [])),
        "projection_input_node_count": len(raw_nodes),
        "projection_input_relation_count": len(graph.get("motif_relations", [])),
        "selected_input_node_count": len(selected_raw_nodes),
        "preprune_prior_relation_count": len(relation_records),
        "postprune_prior_relation_count": len(prior_relations_raw),
        "prior_node_count": len(prior_nodes),
        "prior_relation_count": len(prior_graph["motif_relations"]),
        "kept_node_types": sorted(PRIOR_NODE_TYPES),
        "kept_relation_types": sorted(PRIOR_RELATION_TYPES),
        "dropped_default_node_types": ["face_group", "transition_group", "boundary_group", "thin_wall_pair"],
        "dropped_default_relation_types": [
            "adjacent_to", "parallel_to", "opposite_to", "orthogonal_to", "coplanar_with",
            "smooth_connected", "repeated_with", "bounded_by", "has_member", "embedded_in"
        ],
        "policy": "M_raw 直接执行置信筛选、关系规范化与限流；正式 S 保留三类层级节点和两类高层关系。",
    }
    return prior_graph


def _summarize_face_relation_evidence(rels: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    confidences = [float(rel.get("confidence", 0.0)) for rel in rels]
    evidences = [rel.get("evidence", {}) for rel in rels]
    normal_gaps = [float(ev.get("normal_gap", 0.0)) for ev in evidences]
    center_distances = [float(ev.get("center_distance", 0.0)) for ev in evidences]
    plane_distances = [float(ev.get("plane_distance", ev.get("normal_gap", 0.0))) for ev in evidences]
    area_ratios = [float(ev.get("area_ratio_min_over_max", 0.0)) for ev in evidences]
    angle_values = [float(ev.get("angle_to_parallel_deg", 90.0)) for ev in evidences]
    return {
        "support_face_pairs": len(rels),
        "max_confidence": max(confidences) if confidences else 0.0,
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "min_normal_gap": min(normal_gaps) if normal_gaps else 0.0,
        "mean_normal_gap": float(np.mean(normal_gaps)) if normal_gaps else 0.0,
        "min_center_distance": min(center_distances) if center_distances else 0.0,
        "mean_center_distance": float(np.mean(center_distances)) if center_distances else 0.0,
        "min_plane_distance": min(plane_distances) if plane_distances else 0.0,
        "max_area_ratio": max(area_ratios) if area_ratios else 0.0,
        "mean_area_ratio": float(np.mean(area_ratios)) if area_ratios else 0.0,
        "min_angle_to_parallel_deg": min(angle_values) if angle_values else 90.0,
        "example_evidence": evidences[0] if evidences else {},
    }


def _group_pair_area_score(src_node: Dict[str, Any], dst_node: Dict[str, Any]) -> float:
    a = float(src_node.get("features", {}).get("relative_area_sum", 0.0))
    b = float(dst_node.get("features", {}).get("relative_area_sum", 0.0))
    return float(max(a, b))


def _keep_sparse_base_relation(
    src_node: Dict[str, Any],
    dst_node: Dict[str, Any],
    relation_type: str,
    evidence: Dict[str, Any],
    available_types: Set[str],
    global_scale: float,
) -> bool:
    support = int(evidence.get("support_face_pairs", 0))
    max_conf = float(evidence.get("max_confidence", 0.0))
    mean_conf = float(evidence.get("mean_confidence", 0.0))
    area_ratio = float(evidence.get("max_area_ratio", 0.0))
    min_gap = float(evidence.get("min_normal_gap", 0.0))
    min_center = float(evidence.get("min_center_distance", 0.0))
    min_plane = float(evidence.get("min_plane_distance", 0.0))
    area_score = _group_pair_area_score(src_node, dst_node)
    scale = max(float(global_scale), 1e-8)

    if relation_type == "adjacent_to":
        return support >= 1
    if relation_type == "smooth_connected":
        return "adjacent_to" in available_types and max_conf >= 0.70
    if relation_type == "coplanar_with":
        return max_conf >= 0.68 and min_plane <= 0.025 * scale and (area_score >= 0.03 or support >= 2)
    if relation_type == "orthogonal_to":
        return "adjacent_to" in available_types and max_conf >= 0.60
    if relation_type == "opposite_to":
        return (
            max_conf >= 0.62
            and area_ratio >= 0.45
            and min_gap <= 0.35 * scale
            and (area_score >= 0.035 or support >= 2)
        )
    if relation_type == "parallel_to":
        if ("coplanar_with" in available_types or "opposite_to" in available_types) and area_score < 0.06 and support < 2:
            return False
        return (
            max_conf >= 0.78
            and mean_conf >= 0.58
            and area_ratio >= 0.42
            and min_center <= 0.80 * scale
            and area_score >= 0.025
        )
    return False


def _prune_training_relations(relations: Sequence[Dict[str, Any]], face_group_count: int) -> List[Dict[str, Any]]:
    caps_per_node = {
        "adjacent_to": 4,
        "parallel_to": 2,
        "opposite_to": 2,
        "orthogonal_to": 3,
        "coplanar_with": 2,
        "smooth_connected": 2,
        "embedded_in": 8,
        "repeated_with": 2,
        "bounded_by": 2,
    }
    global_caps = {
        "adjacent_to": max(12, 3 * face_group_count),
        "parallel_to": max(8, 2 * face_group_count),
        "opposite_to": max(8, 2 * face_group_count),
        "orthogonal_to": max(10, 3 * face_group_count),
        "coplanar_with": max(8, 2 * face_group_count),
        "smooth_connected": max(6, 2 * face_group_count),
        "embedded_in": max(16, 4 * face_group_count),
        "repeated_with": max(8, 2 * face_group_count),
        "bounded_by": max(8, 2 * face_group_count),
    }
    priority = {
        "embedded_in": 0,
        "repeated_with": 1,
        "bounded_by": 2,
        "adjacent_to": 3,
        "opposite_to": 4,
        "coplanar_with": 5,
        "orthogonal_to": 6,
        "parallel_to": 7,
        "smooth_connected": 8,
    }
    sorted_relations = sorted(
        relations,
        key=lambda rel: (
            priority.get(str(rel.get("type")), 99),
            -float(rel.get("confidence", 0.0)),
            str(rel.get("source")),
            str(rel.get("target")),
        ),
    )
    node_type_degree: Dict[Tuple[str, str], int] = {}
    type_count: Dict[str, int] = {}
    kept: List[Dict[str, Any]] = []
    for rel in sorted_relations:
        typ = str(rel.get("type"))
        src = str(rel.get("source"))
        dst = str(rel.get("target"))
        cap = caps_per_node.get(typ, 2)
        global_cap = global_caps.get(typ, max(8, face_group_count))
        if type_count.get(typ, 0) >= global_cap:
            continue
        if node_type_degree.get((src, typ), 0) >= cap or node_type_degree.get((dst, typ), 0) >= cap:
            continue
        kept.append(rel)
        type_count[typ] = type_count.get(typ, 0) + 1
        node_type_degree[(src, typ)] = node_type_degree.get((src, typ), 0) + 1
        node_type_degree[(dst, typ)] = node_type_degree.get((dst, typ), 0) + 1
    return sorted(kept, key=lambda r: (str(r.get("source")), str(r.get("target")), str(r.get("type"))))


def _node_face_group_refs(node: Dict[str, Any]) -> Set[str]:
    features = node.get("features", {})
    refs: Set[str] = set()
    for key in ["base_face_group_id"]:
        value = features.get(key)
        if value:
            refs.add(str(value))
    for key in ["base_face_group_ids", "member_face_group_ids", "larger_neighbor_group_ids"]:
        for value in features.get(key, []) or []:
            refs.add(str(value))
    return refs


def _remap_ids_in_value(value: Any, id_map: Dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: _remap_ids_in_value(val, id_map) for key, val in value.items()}
    if isinstance(value, list):
        return [_remap_ids_in_value(item, id_map) for item in value]
    if isinstance(value, tuple):
        return [_remap_ids_in_value(item, id_map) for item in value]
    if isinstance(value, str) and value in id_map:
        return id_map[value]
    return value


def _compress_to_structural_motif_graph(
    nodes: Sequence[Dict[str, Any]],
    relations: Sequence[Dict[str, Any]],
    face_group_nodes: Sequence[Dict[str, Any]],
    face_count: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Keep M as a structural motif graph instead of a full face-group graph."""
    face_count = max(int(face_count), 1)
    target_max_nodes = face_count
    node_by_id = {str(node["id"]): node for node in nodes}
    face_group_by_id = {str(node["id"]): node for node in face_group_nodes}
    structural_nodes = [node for node in nodes if node.get("type") != "face_group"]
    priority = {
        "thin_wall_pair": 0,
        "repeated_feature": 1,
        "loop_or_hole": 2,
        "transition_group": 3,
        "sheet_region": 4,
        "boundary_group": 5,
    }
    ordered_structural = sorted(
        structural_nodes,
        key=lambda node: (
            priority.get(str(node.get("type")), 99),
            -float(node.get("confidence", 0.0)),
            str(node.get("id")),
        ),
    )

    kept_structural: List[Dict[str, Any]] = []
    for node in ordered_structural:
        if len(kept_structural) < target_max_nodes:
            kept_structural.append(node)

    support_face_group_ids: Set[str] = set()
    for node in kept_structural:
        for ref in _node_face_group_refs(node):
            if ref in face_group_by_id:
                support_face_group_ids.add(ref)

    support_rank: Dict[str, float] = {}
    for node in kept_structural:
        weight = 1.0 + float(node.get("confidence", 0.0))
        for ref in _node_face_group_refs(node):
            if ref in face_group_by_id:
                support_rank[ref] = support_rank.get(ref, 0.0) + weight

    # Structural nodes already carry face_ids. Add only as many support groups as fit.
    support_capacity = max(0, target_max_nodes - len(kept_structural))
    ordered_support_ids = sorted(
        support_face_group_ids,
        key=lambda item: (-support_rank.get(item, 0.0), item),
    )[:support_capacity]
    support_face_group_ids = set(ordered_support_ids)

    if not kept_structural:
        support_face_group_ids = {str(node["id"]) for node in face_group_nodes[:target_max_nodes]}

    kept_face_groups = [node for node in face_group_nodes if str(node["id"]) in support_face_group_ids]
    kept_node_ids = {str(node["id"]) for node in kept_structural} | {str(node["id"]) for node in kept_face_groups}

    filtered_relations = [
        rel
        for rel in relations
        if str(rel.get("source")) in kept_node_ids and str(rel.get("target")) in kept_node_ids
    ]

    # Re-add explicit support links if pruning removed them.
    relation_keys = {(str(rel.get("source")), str(rel.get("target")), str(rel.get("type"))) for rel in filtered_relations}
    for node in kept_structural:
        for fg_id in _node_face_group_refs(node):
            if fg_id in support_face_group_ids:
                key = (fg_id, str(node["id"]), "embedded_in")
                if key not in relation_keys:
                    filtered_relations.append(
                        {
                            "source": fg_id,
                            "target": str(node["id"]),
                            "type": "embedded_in",
                            "confidence": 0.82,
                            "evidence": {"support_status": "结构 motif 支撑", "compressed_graph_added": True},
                        }
                    )
                    relation_keys.add(key)

    ordered_nodes = kept_face_groups + kept_structural
    id_map = {str(node["id"]): f"m{idx}" for idx, node in enumerate(ordered_nodes)}
    remapped_nodes: List[Dict[str, Any]] = []
    for node in ordered_nodes:
        new_node = _remap_ids_in_value(dict(node), id_map)
        new_node["id"] = id_map[str(node["id"])]
        remapped_nodes.append(new_node)

    remapped_relations: List[Dict[str, Any]] = []
    seen_rel_keys: Set[Tuple[str, str, str]] = set()
    for rel in filtered_relations:
        src_old = str(rel.get("source"))
        dst_old = str(rel.get("target"))
        if src_old not in id_map or dst_old not in id_map:
            continue
        new_rel = _remap_ids_in_value(dict(rel), id_map)
        new_rel["source"] = id_map[src_old]
        new_rel["target"] = id_map[dst_old]
        key = (str(new_rel["source"]), str(new_rel["target"]), str(new_rel["type"]))
        if key in seen_rel_keys:
            continue
        seen_rel_keys.add(key)
        remapped_relations.append(new_rel)

    remapped_face_groups = [node for node in remapped_nodes if node.get("type") == "face_group"]
    info = {
        "target_max_nodes": target_max_nodes,
        "raw_node_count": len(nodes),
        "raw_face_group_count": len(face_group_nodes),
        "raw_relation_count": len(relations),
        "kept_node_count": len(remapped_nodes),
        "kept_face_group_count": len(remapped_face_groups),
        "kept_relation_count": len(remapped_relations),
        "policy": "优先导出结构 motif 节点，再补充必要且不超预算的支撑 face_group 节点",
    }
    return remapped_nodes, remapped_relations, remapped_face_groups, info


def _assess_motif_quality(
    face_count: int,
    node_type_counts: Dict[str, int],
    relation_type_counts: Dict[str, int],
    node_count: int,
    relation_count: int,
    geometry_sampling_quality: str,
    dtg_train_compatible: bool,
) -> Dict[str, Any]:
    non_base_types = [
        typ
        for typ in ["sheet_region", "thin_wall_pair", "loop_or_hole", "transition_group", "repeated_feature", "boundary_group"]
        if int(node_type_counts.get(typ, 0)) > 0
    ]
    core_motif_types = [
        typ
        for typ in ["sheet_region", "thin_wall_pair", "loop_or_hole", "transition_group", "repeated_feature"]
        if int(node_type_counts.get(typ, 0)) > 0
    ]
    key_relation_types = [
        typ
        for typ in ["opposite_to", "orthogonal_to", "coplanar_with", "parallel_to", "repeated_with", "bounded_by"]
        if int(relation_type_counts.get(typ, 0)) > 0
    ]
    reasons: List[str] = []
    score = 0.0
    if dtg_train_compatible:
        score += 0.18
        reasons.append("dtg_train_compatible")
    if 6 <= int(face_count) <= 70:
        score += 0.10
        reasons.append("dtg_size_compatible")
    if int(node_type_counts.get("face_group", 0)) >= 2:
        score += 0.14
        reasons.append("has_face_group_support")
    if len(core_motif_types) >= 1:
        score += 0.18
        reasons.append("has_core_motif")
    if len(core_motif_types) >= 2:
        score += 0.14
        reasons.append("motif_rich")
    if int(relation_type_counts.get("adjacent_to", 0)) > 0:
        score += 0.10
        reasons.append("has_topology_skeleton")
    if len(key_relation_types) >= 1:
        score += 0.12
        reasons.append("has_structural_relation")
    if len(key_relation_types) >= 2:
        score += 0.08
        reasons.append("multi_relation_prior")
    if relation_count <= max(12, 4 * max(node_count, 1)):
        score += 0.04
        reasons.append("relation_density_controlled")
    if geometry_sampling_quality == "true_or_dtg_sampling":
        score += 0.06
        reasons.append("dtg_geometry_sampling")
    elif geometry_sampling_quality == "bbox_fallback_sampling":
        score += 0.03
        reasons.append("fallback_geometry_sampling")

    score = float(min(score, 1.0))
    if score >= 0.72 and dtg_train_compatible and len(core_motif_types) >= 2 and len(key_relation_types) >= 1:
        grade = "high"
    elif score >= 0.48 and dtg_train_compatible and len(core_motif_types) >= 1:
        grade = "medium"
    else:
        grade = "low"
    return {
        "motif_ready": bool(grade == "high"),
        "strict_ready_policy": "high_quality_only",
        "dtg_train_compatible_required": True,
        "motif_quality_grade": grade,
        "motif_quality_score": round(score, 6),
        "non_base_motif_types": non_base_types,
        "core_motif_types": core_motif_types,
        "key_relation_types": key_relation_types,
        "quality_reasons": reasons,
    }


def build_face_evidence(data: Dict[str, Any], features: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    抽取完整的面级几何与拓扑证据图 E_face，不对关系或度数进行任何裁剪或过滤。
    参数:
        data: 从原始 STEP 解析出来的基础面片及曲率数据字典。
    返回:
        包含面级特征、无损拓扑及几何邻接/平行/垂直关系的完整证据字典。
    """
    features = features or extract_motif_features(data)
    face_features = features.get("face_features", [])
    face_relations = features.get("face_relations", [])
    face_types = features.get("face_surface_type", [])
    
    # 统计各类面片几何/拓扑关系的原始数量
    face_relation_stats = {}
    for rel in face_relations:
        rtype = str(rel.get("type", "unknown"))
        face_relation_stats[rtype] = face_relation_stats.get(rtype, 0) + 1
        
    return {
        "uid": data.get("uid", ""),
        "source": data.get("source", "unknown"),
        "num_faces": len(face_features),
        "face_features": face_features,
        "face_relations": face_relations,
        "face_types": [int(x) for x in face_types],
        "face_evidence_summary": {
            "global_bbox": features.get("global_bbox", []),
            "global_dims": features.get("global_dims", []),
            "global_scale": float(features.get("global_scale", 1.0)),
            "thresholds": features.get("thresholds", {}),
            "face_relation_stats": face_relation_stats,
        }
    }


def build_motif_graph(
    data: Dict[str, Any],
    raw_mode: bool = False,
    features: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    features = features or extract_motif_features(data)
    face_features = features["face_features"]
    global_scale = float(features.get("global_scale", 1.0))
    nodes: List[Dict[str, Any]] = []
    face_group_nodes: List[Dict[str, Any]] = []
    face_to_group: Dict[int, str] = {}

    # 1. 采用法向聚类与平行对识别板状主体，适应倾斜件、L型件等复杂模型
    face_types = features.get("face_surface_type", [])
    clusters = []
    for idx in range(len(face_features)):
        if idx >= len(face_types) or face_types[idx] != 0:
            continue
        n = np.asarray(face_features[idx].get("normal_proxy", [0.0, 1.0, 0.0]))
        n_norm = np.linalg.norm(n)
        if n_norm > 1e-8:
            n = n / n_norm
        else:
            n = np.array([0.0, 1.0, 0.0])
        area = float(face_features[idx].get("area_proxy", 0.0))
        
        matched = False
        for c in clusters:
            if abs(np.dot(c["normal"], n)) >= 0.996:  # 容差为 5 度的平行/相反法向
                c["faces"].append(idx)
                c["total_area"] += area
                matched = True
                break
        if not matched:
            clusters.append({
                "normal": n,
                "faces": [idx],
                "total_area": area
            })
            
    clusters = sorted(clusters, key=lambda c: c["total_area"], reverse=True)
    plate_faces = set()
    thick_vector = np.array([0.0, 1.0, 0.0])
    
    if clusters:
        # 主平板系统
        c_primary = clusters[0]
        plate_faces.update(c_primary["faces"])
        thick_vector = c_primary["normal"]
        
        # 寻找次级正交板状系统（例如 L型支架的正交板面）
        for c in clusters[1:]:
            is_orthogonal = abs(np.dot(c["normal"], thick_vector)) <= 0.08
            if is_orthogonal and c["total_area"] >= 0.25 * c_primary["total_area"]:
                plate_faces.update(c["faces"])
                
    normal1_abs = [abs(x) for x in thick_vector]
    thick_axis = normal1_abs.index(max(normal1_abs))
    thick_axis_name = ['x', 'y', 'z'][thick_axis]
    side_faces = [idx for idx in range(len(face_features)) if idx not in plate_faces]
    
    # 2. 侧面 BFS 连通性传播，加入平面“防火墙”阻断，防止泄漏至孔洞或圆角面
    seeds = []
    for f in side_faces:
        if f >= len(face_types) or face_types[f] != 0:
            continue
        b_axes = face_features[f].get("boundary_axes", [])
        has_other_boundary = False
        for ax in b_axes:
            ax_name = ax[:-1]
            if ax_name != thick_axis_name:
                has_other_boundary = True
                break
        if has_other_boundary:
            seeds.append(f)

    side_adj = {f: [] for f in side_faces}
    for f in side_faces:
        adj_list = face_features[f].get("adjacency_faces", [])
        for adj in adj_list:
            if adj in side_adj:
                side_adj[f].append(adj)

    outer_side_faces = set()
    queue = list(seeds)
    for s in seeds:
        outer_side_faces.add(s)
    head = 0
    while head < len(queue):
        curr = queue[head]
        head += 1
        for neighbor in side_adj.get(curr, []):
            # 防火墙：阻止 BFS 向内圆柱孔（Hole）传播，从而将整个外侧边界面严格限制在全局外壳上，防止向内腔泄露
            if neighbor not in outer_side_faces:
                if neighbor < len(face_types) and face_types[neighbor] == 1:
                    if _is_cylinder_hole(neighbor, features):
                        continue
                outer_side_faces.add(neighbor)
                queue.append(neighbor)

    def add_node(node_type: str, face_ids: Iterable[int], confidence: float, node_features: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
        node = {
            "id": f"m{len(nodes)}",
            "type": node_type,
            "face_ids": sorted({int(fid) for fid in face_ids}),
            "confidence": float(max(0.0, min(0.99, confidence))),
            "features": node_features,
            "evidence": evidence,
        }
        nodes.append(node)
        return node

    face_wcs = features.get("face_wcs")
    base_groups = _build_base_face_groups(features)
    for group in base_groups:
        gf = _group_features(group, face_features, global_scale, face_types, face_wcs)
        node = add_node(
            "face_group",
            group,
            0.9 if len(group) > 1 else 0.78,
            gf,
            {"rule": "由相邻且共面/平滑连接的 face 形成连通面组", "is_singleton": len(group) == 1},
        )
        face_group_nodes.append(node)
        for fid in group:
            face_to_group[int(fid)] = node["id"]

    group_area_values = [float(node["features"].get("relative_area_sum", 0.0)) for node in face_group_nodes]
    planar_group_area_values = [
        float(node["features"].get("relative_area_sum", 0.0))
        for node in face_group_nodes
        if float(node["features"].get("planarity_score", 0.0)) >= 0.90
    ]
    # Use a per-part planar percentile.  A fixed 8% floor incorrectly removes
    # the two cap planes of cylindrical extrusion parts even though they are
    # the only stable planar supports.
    sheet_area_cut = max(1.0e-6, _percentile(planar_group_area_values, 70.0, 1.0e-6))
    small_area_cut = max(0.015, _percentile(group_area_values, 35.0, 0.03))
    face_group_by_id = {node["id"]: node for node in face_group_nodes}

    sheet_candidates: List[Tuple[float, Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = []
    boundary_candidates: List[Tuple[float, Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = []
    for group_node in face_group_nodes:
        gf = group_node["features"]
        rel_area = float(gf.get("relative_area_sum", 0.0))
        thinness = float(gf.get("bbox_thinness", 1.0))
        boundary_ratio = float(gf.get("boundary_ratio", 0.0))
        planarity = float(gf.get("planarity_score", 0.0))
        if planarity >= 0.90 and (rel_area >= sheet_area_cut or (rel_area >= 0.06 and thinness <= 0.08)):
            confidence = 0.48 + min(0.30, rel_area) + 0.10 * planarity
            confidence += (0.07 if thinness <= 0.08 else 0.0) + 0.04 * boundary_ratio
            sheet_candidates.append(
                (
                    confidence,
                    group_node,
                    dict(gf, base_face_group_id=group_node["id"]),
                    {
                        "rule": "高平面性 + 相对面积/薄尺度主体面组",
                        "relative_area_cut": sheet_area_cut,
                        "planarity_min": 0.90,
                    },
                )
            )
        if boundary_ratio >= 0.5:
            boundary_candidates.append(
                (
                    0.58 + 0.35 * boundary_ratio,
                    group_node,
                    dict(gf, base_face_group_id=group_node["id"]),
                    {"rule": "face 接触全局 bbox 边界", "boundary_ratio": boundary_ratio},
                )
            )

    face_count_for_caps = max(int(features.get("face_count", 0)), 1)
    max_sheet_groups = max(2, min(5, int(0.10 * face_count_for_caps) + 1))
    max_boundary_groups = max(2, min(5, int(0.10 * face_count_for_caps) + 1))
    for confidence, group_node, node_features, evidence in sorted(sheet_candidates, key=lambda x: -x[0])[:max_sheet_groups]:
        add_node("sheet_region", group_node["face_ids"], confidence, node_features, evidence)
    for confidence, group_node, node_features, evidence in sorted(boundary_candidates, key=lambda x: -x[0])[:max_boundary_groups]:
        add_node("boundary_group", group_node["face_ids"], confidence, node_features, evidence)

    face_relations = features.get("face_relations", [])
    opposite_relations = [rel for rel in face_relations if rel.get("type") == "opposite_to"]
    used_thin_pairs: Set[Tuple[str, str]] = set()
    thin_candidates: List[Tuple[float, Tuple[str, str], Dict[str, Any]]] = []
    for rel in opposite_relations:
        evidence = rel.get("evidence", {})
        i, j = [int(x) for x in evidence.get("face_pair", [rel.get("source_face"), rel.get("target_face")])]
        gid_i = face_to_group.get(i)
        gid_j = face_to_group.get(j)
        if not gid_i or not gid_j or gid_i == gid_j:
            continue
        key = tuple(sorted([gid_i, gid_j]))
        if key in used_thin_pairs:
            continue
        gap = float(evidence.get("effective_gap", evidence.get("plane_distance", evidence.get("normal_gap", 0.0))))
        area_ratio = float(evidence.get("area_ratio_min_over_max", 0.0))
        group_a = face_group_by_id[key[0]]
        group_b = face_group_by_id[key[1]]
        dims_a = group_a["features"].get("bbox_dims_sorted", [0.0, 0.0, 0.0])
        dims_b = group_b["features"].get("bbox_dims_sorted", [0.0, 0.0, 0.0])
        major_span = min(float(dims_a[-1]), float(dims_b[-1]))
        thin_gap_cut = max(1e-5, min(0.08 * global_scale, 0.22 * max(major_span, 1e-5)))
        area_score = max(
            float(group_a["features"].get("relative_area_sum", 0.0)),
            float(group_b["features"].get("relative_area_sum", 0.0)),
        )
        
        # 严格校验重叠率：必须有效，且重叠率 >= 0.50
        overlap_valid = bool(evidence.get("projection_overlap_valid", False))
        overlap_value = evidence.get("projection_overlap_ratio")
        if not overlap_valid or overlap_value is None:
            continue
        overlap = float(overlap_value)
        
        if gap <= thin_gap_cut and area_ratio >= 0.62 and area_score >= 0.015 and overlap >= 0.50:
            gap_score = 1.0 - min(gap / max(thin_gap_cut, 1e-8), 1.0)
            confidence = float(0.42 + 0.30 * rel.get("confidence", 0.0) + 0.18 * area_ratio + 0.10 * gap_score)
            thin_candidates.append(
                (
                    confidence,
                    key,
                    {
                        "face_ids": sorted(set(group_a["face_ids"] + group_b["face_ids"])),
                        "pair_face_ids": [i, j],
                        "normal_gap": gap,
                        "effective_gap": gap,
                        "thin_gap_cut": thin_gap_cut,
                        "area_ratio_min_over_max": area_ratio,
                        "opposite_relation": evidence,
                    },
                )
            )

    thin_group_degree: Dict[str, int] = {}
    max_thin_pairs = 999999 if raw_mode else max(2, min(7, len(face_group_nodes), int(0.12 * face_count_for_caps) + 1))
    for confidence, key, item in sorted(thin_candidates, key=lambda x: (-x[0], x[2]["normal_gap"])):
        if not raw_mode and len(used_thin_pairs) >= max_thin_pairs:
            break
        if key in used_thin_pairs:
            continue
        if not raw_mode and (thin_group_degree.get(key[0], 0) >= 2 or thin_group_degree.get(key[1], 0) >= 2):
            continue
        used_thin_pairs.add(key)
        thin_group_degree[key[0]] = thin_group_degree.get(key[0], 0) + 1
        thin_group_degree[key[1]] = thin_group_degree.get(key[1], 0) + 1
        add_node(
            "thin_wall_pair",
            item["face_ids"],
            confidence,
            {
                "base_face_group_ids": list(key),
                "pair_face_ids": item["pair_face_ids"],
                "normal_gap": item["normal_gap"],
                "effective_gap": item["effective_gap"],
                "gap_source": "analytical_plane_distance",
                "thin_gap_cut": item["thin_gap_cut"],
                "area_ratio_min_over_max": item["area_ratio_min_over_max"],
            },
            {"rule": "稀疏 opposite_to + 小间距 + 相似面积", "opposite_relation": item["opposite_relation"]},
        )

    loop_candidates: List[Tuple[float, Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = []
    face_types = features.get("face_surface_type") or []
    radii = features.get("face_cylinder_radius") or [0.0] * len(face_features)
    axes = features.get("face_cylinder_axis") or [[0.0, 0.0, 0.0]] * len(face_features)
    locations = features.get("face_cylinder_location") or [[0.0, 0.0, 0.0]] * len(face_features)
    
    for group_node in face_group_nodes:
        gf = group_node["features"]
        rel_area = float(gf.get("relative_area_sum", 0.0))
        boundary_ratio = float(gf.get("boundary_ratio", 0.0))
        outside_degree = len(gf.get("adjacent_faces_outside", []))
        loop_area_cut = min(0.08, max(0.025, small_area_cut * 1.25))
        has_cylinder = any(idx < len(face_types) and face_types[idx] == 1 for idx in group_node["face_ids"])
        
        # 排除外部圆柱凸台/销轴：如果组内圆柱面的法向朝外（远离轴心），则判定为外部凸台而非内孔
        is_boss = False
        for fid in group_node["face_ids"]:
            if fid < len(face_types) and face_types[fid] == 1:
                if not _is_cylinder_hole(fid, features):
                    is_boss = True
                    break

        # 排除与外部侧边界面相连的圆角面片（即面片本身在外部侧面上，或邻接了外部侧面）
        is_corner_fillet = False
        for fid in group_node["face_ids"]:
            if int(fid) in outer_side_faces:
                is_corner_fillet = True
                break
            for adj in gf.get("adjacent_faces_outside", []):
                adj_idx = int(adj)
                if adj_idx in outer_side_faces:
                    is_corner_fillet = True
                    break
            if is_corner_fillet:
                break

        if (
            (boundary_ratio <= 0.05 or has_cylinder)
            and not is_corner_fillet
            and not is_boss  # 过滤外部圆柱凸台
            and rel_area <= loop_area_cut
            and outside_degree >= 2
            and int(gf.get("face_count", 1)) <= 8
        ):
            # 对于标准板厚穿透孔，外部邻接度为2，应给予极高置信度
            if outside_degree == 2:
                confidence = 0.82 + 0.10 * (1.0 - min(rel_area / loop_area_cut, 1.0))
            else:
                confidence = 0.42 + 0.24 * min(outside_degree / 5.0, 1.0) + 0.20 * (1.0 - min(rel_area / loop_area_cut, 1.0))
                
            loop_candidates.append(
                (
                    confidence,
                    group_node,
                    dict(gf, base_face_group_id=group_node["id"], wall_group_count=1),
                    {
                        "rule": "内部有界局部闭合候选",
                        "note": "仅为候选，不是真实工程孔洞标签",
                        "loop_area_cut": loop_area_cut,
                        "outside_degree": outside_degree,
                    },
                )
            )
    # A polygonal hole/slot is often represented by several adjacent planar
    # wall groups.  Treating every wall as a separate hole inflates node count
    # and destroys the hierarchy.  Merge adjacent candidates only when they
    # share at least two external host faces (typically the two cap sheets).
    if len(loop_candidates) > 1:
        loop_uf = UnionFind(len(loop_candidates))
        loop_face_sets = [set(int(fid) for fid in item[1]["face_ids"]) for item in loop_candidates]
        loop_adjacent_sets = [
            set(int(fid) for fid in item[1]["features"].get("adjacent_faces_outside", []))
            for item in loop_candidates
        ]
        for left, right in combinations(range(len(loop_candidates)), 2):
            adjacent = bool(loop_adjacent_sets[left] & loop_face_sets[right]) or bool(
                loop_adjacent_sets[right] & loop_face_sets[left]
            )
            if not adjacent:
                continue
            external_left = loop_adjacent_sets[left] - loop_face_sets[right]
            external_right = loop_adjacent_sets[right] - loop_face_sets[left]
            if len(external_left & external_right) >= 2:
                loop_uf.union(left, right)

        merged_loop_candidates: List[Tuple[float, Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = []
        for cluster in loop_uf.groups():
            if len(cluster) == 1:
                merged_loop_candidates.append(loop_candidates[cluster[0]])
                continue
            members = [loop_candidates[index] for index in cluster]
            face_ids = sorted({int(fid) for item in members for fid in item[1]["face_ids"]})
            merged_features = _group_features(face_ids, face_features, global_scale, face_types, face_wcs)
            base_ids = [str(item[1]["id"]) for item in members]
            merged_features["base_face_group_ids"] = base_ids
            merged_features["wall_group_count"] = len(base_ids)
            confidence = min(0.99, float(np.mean([item[0] for item in members])) + 0.04 * min(len(cluster), 4))
            merged_loop_candidates.append(
                (
                    confidence,
                    {"id": "+".join(base_ids), "face_ids": face_ids, "features": merged_features},
                    merged_features,
                    {
                        "rule": "相邻局部壁面 + 至少两个共同宿主面组成一个 loop/hole",
                        "merged_wall_group_ids": base_ids,
                        "merged_wall_group_count": len(base_ids),
                    },
                )
            )
        loop_candidates = merged_loop_candidates

    max_loop_groups = 999999 if raw_mode else max(4, min(16, int(0.25 * face_count_for_caps) + 1))
    loop_motif_nodes: List[Dict[str, Any]] = []
    for confidence, group_node, node_features, evidence in sorted(loop_candidates, key=lambda x: -x[0])[:max_loop_groups]:
        loop_motif_nodes.append(add_node("loop_or_hole", group_node["face_ids"], confidence, node_features, evidence))

    group_id_to_node = {node["id"]: node for node in face_group_nodes}
    face_to_group_index = {fid: gid for fid, gid in face_to_group.items()}
    orthogonal_pairs = {_relation_pair_key(rel) for rel in face_relations if rel.get("type") == "orthogonal_to"}
    smooth_pairs = {_relation_pair_key(rel) for rel in face_relations if rel.get("type") == "smooth_connected"}
    
    transition_candidates: List[Tuple[float, Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = []
    for group_node in face_group_nodes:
        gf = group_node["features"]
        rel_area = float(gf.get("relative_area_sum", 0.0))
        high_aspect = float(gf.get("max_aspect_ratio", 1.0)) >= 3.0
        
        # 提前计算 ortho_support 以免在 else 降级分支中被提早引用
        ortho_support = 0
        for fid in group_node["face_ids"]:
            for adj in gf.get("adjacent_faces_outside", []):
                if tuple(sorted([int(fid), int(adj)])) in orthogonal_pairs:
                    ortho_support += 1
                    
        mean_curvature_mean = gf.get("mean_curvature_mean")
        var_curvature_mean = gf.get("var_curvature_mean")
        gaussian_sign_mean = gf.get("gaussian_sign_mean")
        
        has_smooth_boundary = False
        for fid in group_node["face_ids"]:
            for adj in gf.get("adjacent_faces_outside", []):
                if tuple(sorted([int(fid), int(adj)])) in smooth_pairs:
                    has_smooth_boundary = True
                    break
            if has_smooth_boundary:
                break
                
        is_planar = all(idx < len(face_types) and face_types[idx] == 0 for idx in group_node["face_ids"])
        normal = gf.get("normal_proxy", [0.0, 0.0, 0.0])
        max_normal_comp = max(abs(x) for x in normal) if normal else 0.0
        is_axis_aligned = max_normal_comp > 0.95

        if mean_curvature_mean is not None:
            is_fillet = (not is_planar) and (mean_curvature_mean >= 1.5) and (var_curvature_mean < 5.0) and (gaussian_sign_mean == 0) and has_smooth_boundary
            is_chamfer = is_planar and high_aspect and (not is_axis_aligned)
            is_transition = is_fillet or is_chamfer
            curvature_flag = is_fillet
        else:
            curvature_flag = float(gf.get("curvature_proxy_mean", 0.0)) >= 0.5
            is_transition = (high_aspect and (not (is_planar and is_axis_aligned))) or curvature_flag or ortho_support > 0

        neighbor_groups = sorted({face_to_group_index.get(fid) for fid in gf.get("adjacent_faces_outside", []) if face_to_group_index.get(fid)})
        neighbor_groups = [gid for gid in neighbor_groups if gid != group_node["id"]]
        larger_neighbors = [
            gid
            for gid in neighbor_groups
            if float(group_id_to_node[gid]["features"].get("relative_area_sum", 0.0)) > rel_area * 1.4
        ]
        transition_area_cut = min(0.10, max(0.03, small_area_cut * 1.8))
        neighbor_support = len(set(larger_neighbors))
        connector_support = neighbor_support >= 2 or (
            neighbor_support >= 1 and (curvature_flag or (high_aspect and ortho_support > 0))
        )
        is_already_hole = any(set(group_node["face_ids"]) == set(lc[1]["face_ids"]) for lc in loop_candidates)
        
        non_plate_planes = 0
        for adj in neighbor_groups:
            adj_node = group_id_to_node[adj]
            adj_area = float(adj_node["features"].get("relative_area_sum", 0.0))
            has_plane = any(idx < len(face_types) and face_types[idx] == 0 for idx in adj_node["face_ids"])
            if has_plane and adj_area < 0.3:
                non_plate_planes += 1

        if (
            not is_already_hole
            and non_plate_planes >= 1
            and rel_area <= transition_area_cut
            and is_transition
            and connector_support
        ):
            confidence = 0.43 + 0.16 * min(len(set(larger_neighbors)) / 2.0, 1.0)
            confidence += 0.14 if high_aspect else 0.0
            confidence += 0.14 if curvature_flag else 0.0
            confidence += 0.10 * min(ortho_support / 3.0, 1.0)
            transition_candidates.append(
                (
                    confidence,
                    group_node,
                    dict(gf, base_face_group_id=group_node["id"], larger_neighbor_group_ids=sorted(set(larger_neighbors))),
                    {
                        "rule": "小面积/高长宽比/曲面连接候选",
                        "note": "几何拓扑过渡候选，不保证真实圆角语义",
                        "orthogonal_support": ortho_support,
                        "transition_area_cut": transition_area_cut,
                    },
                )
            )
    max_transition_groups = 999999 if raw_mode else max(2, min(5, len(face_group_nodes), int(0.10 * face_count_for_caps) + 1))
    used_transition_faces: Set[int] = set()
    for confidence, group_node, node_features, evidence in sorted(transition_candidates, key=lambda x: -x[0]):
        if not raw_mode and _node_type_count(nodes, "transition_group") >= max_transition_groups:
            break
        face_ids = set(int(fid) for fid in group_node["face_ids"])
        if face_ids & used_transition_faces:
            continue
        used_transition_faces |= face_ids
        add_node(
            "transition_group",
            group_node["face_ids"],
            confidence,
            node_features,
            evidence,
        )

    # A repeated feature is a higher-order pattern of local motifs, not merely
    # two similar B-rep faces.  This prevents the opposite faces of an ordinary
    # box from being mislabeled as a repeated engineering feature.
    repeat_candidates = [
        node
        for node in loop_motif_nodes
        if float(node["features"].get("relative_area_sum", 0.0)) <= 0.25
        and float(node["features"].get("area_proxy_sum", 0.0)) > 1e-8
    ]
    repeat_pair_evidence: Dict[Tuple[int, int], Dict[str, Any]] = {}
    similarity_matrix = np.zeros((len(repeat_candidates), len(repeat_candidates)), dtype=bool)
    for a_idx, b_idx in combinations(range(len(repeat_candidates)), 2):
        similar, evidence = _similar_group_signature(repeat_candidates[a_idx], repeat_candidates[b_idx])
        if similar:
            repeat_pair_evidence[(a_idx, b_idx)] = evidence
            similarity_matrix[a_idx, b_idx] = True
            similarity_matrix[b_idx, a_idx] = True
            
    # 采用 Complete-Linkage 聚类，约束簇内任何两个元素必须两两相似，避免并查集链式扩散问题
    raw_clusters = [[i] for i in range(len(repeat_candidates))]
    for i in range(len(repeat_candidates)):
        similarity_matrix[i, i] = True
        
    changed = True
    while changed:
        changed = False
        best_pair = None
        for i in range(len(raw_clusters)):
            for j in range(i + 1, len(raw_clusters)):
                all_similar = True
                for idx_a in raw_clusters[i]:
                    for idx_b in raw_clusters[j]:
                        if not similarity_matrix[idx_a, idx_b]:
                            all_similar = False
                            break
                    if not all_similar:
                        break
                if all_similar:
                    best_pair = (i, j)
                    break
            if best_pair:
                break
        if best_pair:
            i, j = best_pair
            raw_clusters[i].extend(raw_clusters[j])
            raw_clusters.pop(j)
            changed = True
            
    repeat_clusters = sorted(raw_clusters, key=len, reverse=True)
    max_repeated_features = 999999 if raw_mode else max(1, min(3, int(0.06 * face_count_for_caps) + 1))
    repeated_added = 0
    for cluster in repeat_clusters:
        if not raw_mode and repeated_added >= max_repeated_features:
            break
        if len(cluster) < 2:
            continue
        members = [repeat_candidates[idx] for idx in cluster]
        boundary_mean = float(np.mean([float(m["features"].get("boundary_ratio", 0.0)) for m in members]))
        rel_area_mean = float(np.mean([float(m["features"].get("relative_area_sum", 0.0)) for m in members]))
        if not raw_mode and (len(members) < 3 and rel_area_mean > 0.12):
            continue
        centroids = np.asarray([m["features"]["centroid"] for m in members], dtype=np.float32)
        # 提取聚类内每个成员代表面片的法向，用来在后面对二元组进行严格的镜像反射对称性校验
        member_directions = []
        for m in members:
            fids = m.get("face_ids", [])
            if fids:
                fid = fids[0]
                if fid < len(face_features):
                    member_directions.append(face_features[fid].get("normal_proxy", [0.0, 0.0, 0.0]))
                else:
                    member_directions.append([0.0, 0.0, 0.0])
            else:
                member_directions.append([0.0, 0.0, 0.0])
        spacing = _spacing_regular_score(centroids, np.asarray(member_directions, dtype=np.float32))
        pattern_type = str(spacing.get("pattern_type", "irregular"))
        regular_score = float(spacing.get("regular_score", 0.0))
        if len(members) == 2:
            if pattern_type != "mirror" or regular_score < 0.70:
                continue
        elif pattern_type not in {"linear", "radial", "grid"} or regular_score < 0.55:
            continue
        pair_support = sum(1 for key in repeat_pair_evidence if key[0] in cluster and key[1] in cluster)
        support_score = pair_support / max(len(members) * (len(members) - 1) / 2.0, 1.0)
        face_ids = sorted({fid for m in members for fid in m["face_ids"]})
        confidence = 0.30 + 0.16 * min(len(members) / 4.0, 1.0) + 0.18 * support_score
        confidence += 0.26 * regular_score + 0.10 * min(1.0, max(0.0, 1.0 - rel_area_mean / 0.25))
        base_group_ids = sorted(
            {
                str(group_id)
                for member in members
                for key in ("base_face_group_id", "base_face_group_ids")
                for group_id in (
                    [member["features"].get(key)]
                    if key == "base_face_group_id" and member["features"].get(key)
                    else member["features"].get(key, []) or []
                )
            }
        )
        add_node(
            "repeated_feature",
            face_ids,
            confidence,
            {
                "member_motif_ids": [m["id"] for m in members],
                "base_face_group_ids": base_group_ids,
                "member_count": len(members),
                "mean_relative_area": rel_area_mean,
                "boundary_ratio_mean": boundary_mean,
                **spacing,
            },
            {
                "rule": "相似局部 loop/hole + 完全链接一致性 + 经验证的镜像/线性/径向/网格排列",
                "pair_support": pair_support,
                "support_score": support_score,
            },
        )
        repeated_added += 1

    relation_records: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    node_face_sets = {node["id"]: set(node["face_ids"]) for node in nodes}
    evidence_relation_nodes = [node for node in nodes if node["type"] == "face_group"]
    node_order = {node["id"]: idx for idx, node in enumerate(evidence_relation_nodes)}
    face_membership: Dict[int, List[str]] = {}
    for node in evidence_relation_nodes:
        for fid in node["face_ids"]:
            face_membership.setdefault(int(fid), []).append(node["id"])
    face_relation_by_type: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for rel in face_relations:
        typ = str(rel.get("type"))
        i, j = _relation_pair_key(rel)
        for src_id in face_membership.get(i, []):
            for dst_id in face_membership.get(j, []):
                if src_id == dst_id:
                    continue
                if node_face_sets[src_id] & node_face_sets[dst_id]:
                    continue
                a_id, b_id = (src_id, dst_id) if node_order[src_id] <= node_order[dst_id] else (dst_id, src_id)
                bucket = face_relation_by_type.setdefault((a_id, b_id, typ), [])
                if rel not in bucket:
                    bucket.append(rel)

    pair_available_types: Dict[Tuple[str, str], Set[str]] = {}
    for src, dst, typ in face_relation_by_type:
        pair_available_types.setdefault((src, dst), set()).add(typ)
    face_group_by_id = {node["id"]: node for node in face_group_nodes}
    for (src, dst, typ), rels in face_relation_by_type.items():
        if typ in {"adjacent_to", "parallel_to", "opposite_to", "orthogonal_to", "coplanar_with", "smooth_connected"}:
            evidence_summary = _summarize_face_relation_evidence(rels)
            if not _keep_sparse_base_relation(
                face_group_by_id[src],
                face_group_by_id[dst],
                typ,
                evidence_summary,
                pair_available_types.get((src, dst), set()),
                global_scale,
            ):
                continue
            confidence = _relation_confidence(rels)
            _add_relation(
                relation_records,
                src,
                dst,
                typ,
                confidence,
                evidence_summary,
            )

    for node_a, node_b in combinations(nodes, 2):
        set_a = node_face_sets[node_a["id"]]
        set_b = node_face_sets[node_b["id"]]
        if set_a == set_b and set_a:
            if node_a["type"] == "face_group" and node_b["type"] != "face_group":
                _add_relation(
                    relation_records,
                    node_a["id"],
                    node_b["id"],
                    "embedded_in",
                    0.86,
                    {"overlap_faces": sorted(set_a), "support_status": "同 face 支撑"},
                )
            elif node_b["type"] == "face_group" and node_a["type"] != "face_group":
                _add_relation(
                    relation_records,
                    node_b["id"],
                    node_a["id"],
                    "embedded_in",
                    0.86,
                    {"overlap_faces": sorted(set_b), "support_status": "同 face 支撑"},
                )
            continue
        if set_a and set_a.issubset(set_b):
            _add_relation(relation_records, node_a["id"], node_b["id"], "embedded_in", 0.78, {"overlap_faces": sorted(set_a)})
        elif set_b and set_b.issubset(set_a):
            _add_relation(relation_records, node_b["id"], node_a["id"], "embedded_in", 0.78, {"overlap_faces": sorted(set_b)})

    structural_node_by_id = {str(node["id"]): node for node in nodes}
    for node in nodes:
        if node["type"] == "repeated_feature":
            members = [
                str(member_id)
                for member_id in node["features"].get("member_motif_ids", [])
                if str(member_id) in structural_node_by_id
            ]
            member_nodes = [structural_node_by_id[member_id] for member_id in members]
            if len(member_nodes) > 2:
                centroids = np.asarray([m["features"]["centroid"] for m in member_nodes], dtype=np.float32)
                centered = centroids - np.mean(centroids, axis=0, keepdims=True)
                try:
                    _, _, vh = np.linalg.svd(centered, full_matrices=False)
                    direction = vh[0]
                    order = np.argsort(centered @ direction).tolist()
                except Exception:
                    order = list(range(len(member_nodes)))
                ordered_members = [member_nodes[idx]["id"] for idx in order]
                repeated_pairs = list(zip(ordered_members[:-1], ordered_members[1:]))
            else:
                repeated_pairs = list(combinations(sorted(members), 2))
            for gid_a, gid_b in repeated_pairs:
                _add_relation(
                    relation_records,
                    gid_a,
                    gid_b,
                    "repeated_with",
                    node["confidence"],
                    {"repeated_feature_node": node["id"], "member_count": len(members)},
                )

    boundary_nodes = [node for node in nodes if node["type"] == "boundary_group"]
    adjacent_face_pairs = {
        _relation_pair_key(rel)
        for rel in face_relations
        if rel.get("type") == "adjacent_to"
    }
    for node in nodes:
        if node["type"] == "boundary_group":
            continue
        for boundary in boundary_nodes:
            if node_face_sets[node["id"]] & node_face_sets[boundary["id"]]:
                continue
            touches_boundary = any(
                tuple(sorted([int(fid), int(bfid)])) in adjacent_face_pairs
                for fid in node_face_sets[node["id"]]
                for bfid in node_face_sets[boundary["id"]]
            )
            if touches_boundary:
                _add_relation(
                    relation_records,
                    node["id"],
                    boundary["id"],
                    "bounded_by",
                    0.62,
                    {"boundary_group": boundary["id"], "rule": "motif 与全局 boundary_group 相邻"},
                )

    if raw_mode:
        motif_relations = sorted(relation_records.values(), key=lambda r: (r["source"], r["target"], r["type"]))
        for r in motif_relations:
            r["relation_role"] = "support" if r["type"] == "embedded_in" else "structural"
        raw_node_count = len(nodes)
        raw_relation_count = len(motif_relations)
        compression_info = {"status": "skipped_in_raw_mode"}
    else:
        motif_relations = _prune_training_relations(
            sorted(relation_records.values(), key=lambda r: (r["source"], r["target"], r["type"])),
            face_group_count=len(face_group_nodes),
        )
        raw_node_count = len(nodes)
        raw_relation_count = len(motif_relations)
        nodes, motif_relations, face_group_nodes, compression_info = _compress_to_structural_motif_graph(
            nodes,
            motif_relations,
            face_group_nodes,
            face_count=int(features.get("face_count", 0)),
        )
        motif_relations, support_limit_info = _limit_embedded_in_relations(nodes, motif_relations)
        compression_info["embedded_in_limit"] = support_limit_info
    face_relation_stats = features.get("face_relation_stats", {})
    node_type_counts = {typ: _node_type_count(nodes, typ) for typ in NODE_TYPES}
    relation_type_counts = {typ: sum(1 for rel in motif_relations if rel["type"] == typ) for typ in RELATION_TYPES}
    relation_role_stats = _relation_role_stats(nodes, motif_relations)
    motif_quality = _assess_motif_quality(
        face_count=int(features.get("face_count", 0)),
        node_type_counts=node_type_counts,
        relation_type_counts=relation_type_counts,
        node_count=len(nodes),
        relation_count=len(motif_relations),
        geometry_sampling_quality=str(features.get("geometry_sampling_quality", data.get("geometry_sampling_quality", "unknown"))),
        dtg_train_compatible=bool(int(data.get("dtg_train_compatible", 0))),
    )
    face_to_motif_nodes: Dict[str, List[str]] = {}
    for node in nodes:
        for fid in node["face_ids"]:
            face_to_motif_nodes.setdefault(str(fid), []).append(node["id"])

    return {
        "uid": features.get("uid", data.get("uid", "")),
        "source": features.get("source", data.get("source", "unknown")),
        "num_faces": int(features.get("face_count", 0)),
        "num_edges": int(features.get("edge_count", 0)),
        "num_vertices": int(features.get("vertex_count", 0)),
        "parser_backend": features.get("parser_backend", data.get("parser_backend", "unknown")),
        "geometry_sampling_quality": features.get("geometry_sampling_quality", data.get("geometry_sampling_quality", "unknown")),
        "dtg_train_compatible": bool(int(data.get("dtg_train_compatible", 0))),
        "dtg_filter_reason": data.get("dtg_filter_reason", ""),
        "motif_nodes": nodes,
        "motif_relations": motif_relations,
        "motif_stats": {
            "num_parallel_pairs": int(face_relation_stats.get("parallel_to", 0)),
            "num_opposite_pairs": int(face_relation_stats.get("opposite_to", 0)),
            "num_orthogonal_pairs": int(face_relation_stats.get("orthogonal_to", 0)),
            "num_coplanar_pairs": int(face_relation_stats.get("coplanar_with", 0)),
            "num_adjacent_pairs": int(face_relation_stats.get("adjacent_to", 0)),
            "num_smooth_connected_pairs": int(face_relation_stats.get("smooth_connected", 0)),
            "num_thin_wall_pairs": node_type_counts.get("thin_wall_pair", 0),
            "num_loop_candidates": node_type_counts.get("loop_or_hole", 0),
            "num_transition_groups": node_type_counts.get("transition_group", 0),
            "num_repeated_features": node_type_counts.get("repeated_feature", 0),
            "node_type_counts": node_type_counts,
            "relation_type_counts": relation_type_counts,
            "relation_role_counts": {
                "structural": relation_role_stats["num_structural_relations"],
                "support": relation_role_stats["num_support_relations"],
                "topology_support": relation_role_stats["num_topology_support_relations"],
            },
            **relation_role_stats,
        },
        "motif_quality": motif_quality,
        "motif_compression": compression_info,
        "motif_prior": {
            "version": "innovation1_v3_motif_prior_v1",
            "policy": "sparse_key_structural_prior",
            "node_type_vocab": NODE_TYPES,
            "relation_type_vocab": RELATION_TYPES,
            "motif_node_type_ids": [NODE_TYPES.index(node["type"]) for node in nodes],
            "motif_relation_type_ids": [RELATION_TYPES.index(rel["type"]) for rel in motif_relations],
            "relation_role_vocab": ["structural", "support", "topology_support"],
            "motif_relation_role_ids": [
                ["structural", "support", "topology_support"].index(str(rel.get("relation_role", "structural")))
                for rel in motif_relations
            ],
            "default_training_relation_role": "structural",
            "support_relation_weight": 0.15,
            "face_to_motif_nodes": face_to_motif_nodes,
            "base_face_group_node_ids": [node["id"] for node in face_group_nodes],
        },
        "motif_selection_policy": {
            "training_graph_policy": "默认训练和论文统计只使用 structural relations；已限量的 support relations 保留在完整图中用于追溯和消融。",
            "structural_node_budget": "导出的 motif node 数量尽量压缩到不超过原始 face_count",
            "support_relation_policy": "embedded_in 按非 face_group motif node 和全局预算限量；adjacent_to 属于 topology_support，不作为核心结构先验。",
            "raw_node_count_before_compression": raw_node_count,
            "raw_relation_count_before_compression": raw_relation_count,
            "kept_node_families": [
                "基础 face_group 支撑",
                "主要 sheet_region 锚点",
                "高置信 thin_wall_pair 候选",
                "内部 loop_or_hole / 局部闭合候选",
                "小面积 transition_group 连接候选",
                "规则 repeated_feature 簇",
                "全局 boundary_group",
            ],
            "kept_relation_families": [
                "face_group 之间的拓扑邻接支撑",
                "稀疏 parallel/opposite/orthogonal/coplanar 结构支撑",
                "motif 到 face_group 的 embedded_in 支撑链接",
                "链式 repeated_with 链接",
                "指向 boundary_group 的 bounded_by 链接",
            ],
            "not_kept_as_training_edges": [
                "全部两两 parallel face pairs",
                "全部两两 orthogonal face pairs",
                "低置信长距离关系对",
            ],
        },
        "face_evidence_summary": {
            "global_bbox": features.get("global_bbox", []),
            "global_dims": features.get("global_dims", []),
            "global_scale": features.get("global_scale", 0.0),
            "thresholds": features.get("thresholds", {}),
            "face_relation_stats": face_relation_stats,
        },
    }


# Persist only controller inputs plus a few compact hierarchy-audit fields.
# Full geometric evidence remains limited to explicit debug examples.
COMPACT_NODE_FEATURES = {
    "face_count",
    "relative_area_sum",
    "bbox_thinness",
    "bbox_extent_ratio",
    "mean_aspect_ratio",
    "mean_face_degree",
    "boundary_ratio",
    "curvature_proxy_mean",
    "planarity_score",
    "curvature_level",
    "surface_family",
    "bbox",
    "bbox_dims",
    "normal_proxy",
    "wall_group_count",
    "member_count",
    "pattern_type",
    "regular_score",
    "pattern_residual",
}


def _compact_node(node: Dict[str, Any]) -> Dict[str, Any]:
    features = dict(node.get("features", {}) or {})
    return {
        "id": str(node.get("id", "")),
        "type": str(node.get("type", "")),
        "face_ids": [int(value) for value in node.get("face_ids", [])],
        "confidence": float(node.get("confidence", 0.0)),
        "features": {key: copy.deepcopy(features[key]) for key in COMPACT_NODE_FEATURES if key in features},
    }


def _compact_relation(relation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source": str(relation.get("source", "")),
        "target": str(relation.get("target", "")),
        "type": str(relation.get("type", "")),
        "confidence": float(relation.get("confidence", 0.0)),
    }


def _compact_face_attributes(raw_graph: Dict[str, Any], num_faces: int) -> List[Dict[str, Any]]:
    by_face: Dict[int, Dict[str, Any]] = {}
    for node in raw_graph.get("motif_nodes", []):
        if str(node.get("type", "")) != "face_group":
            continue
        compact = _compact_node(node)
        for face_id in compact["face_ids"]:
            by_face.setdefault(int(face_id), compact)
    attributes: List[Dict[str, Any]] = []
    for face_id in range(int(num_faces)):
        source = by_face.get(face_id, {"confidence": 0.5, "features": {}})
        features = copy.deepcopy(source.get("features", {}))
        features["face_count"] = 1
        attributes.append(
            {
                "face_id": face_id,
                "confidence": float(source.get("confidence", 0.5)),
                "features": features,
            }
        )
    return attributes


def make_compact_dataset_record(
    prior_graph: Dict[str, Any],
    raw_graph: Dict[str, Any],
    parsed_data: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Create the single persisted Innovation-1 record used by Innovation-2."""
    num_faces = int(prior_graph.get("num_faces", 0) or 0)
    quality = dict(prior_graph.get("motif_quality", {}) or {})
    record: Dict[str, Any] = {
        "schema_version": "innovation1_compact_motif_v2",
        "uid": str(prior_graph.get("uid", "")),
        "source": str(prior_graph.get("source", "deepcad")),
        "num_faces": num_faces,
        "num_edges": int(prior_graph.get("num_edges", 0) or 0),
        "num_vertices": int(prior_graph.get("num_vertices", 0) or 0),
        "parser_backend": str(prior_graph.get("parser_backend", "unknown")),
        "geometry_sampling_quality": str(prior_graph.get("geometry_sampling_quality", "unknown")),
        "dtg_train_compatible": bool(prior_graph.get("dtg_train_compatible", False)),
        "dtg_filter_reason": str(prior_graph.get("dtg_filter_reason", "")),
        "motif_nodes": [_compact_node(node) for node in prior_graph.get("motif_nodes", [])],
        "motif_relations": [_compact_relation(rel) for rel in prior_graph.get("motif_relations", [])],
        "face_attributes": _compact_face_attributes(raw_graph, num_faces),
        "quality_hint": {
            "motif_ready": bool(quality.get("motif_ready", False)),
            "motif_prior_ready": bool(quality.get("motif_prior_ready", False)),
            "motif_quality_score": float(quality.get("motif_quality_score", 0.0) or 0.0),
        },
        "excludes_explicit_face_adjacency": True,
    }
    if parsed_data is not None:
        dtg_ok, dtg_reason, _ = check_dtg_train_compatible(parsed_data, dataset="deepcad")
        record["dtg_train_compatible"] = bool(dtg_ok)
        record["dtg_filter_reason"] = str(dtg_reason)
        fef = np.asarray(parsed_data.get("fef_adj"), dtype=np.int8)
        if fef.shape != (num_faces, num_faces):
            raise ValueError(f"invalid fef_adj for {record['uid']}: {fef.shape}")
        record["fef_adj"] = fef.tolist()
        record["source_step_rel"] = str(parsed_data.get("source_step_rel", ""))
    return record


def _iter_manifest(path: str, limit: int = 0) -> Iterable[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if limit and index >= int(limit):
                break
            yield row


def _repair_compact_dataset(
    path: Path,
    split_by_uid: Dict[str, str] | None = None,
) -> Tuple[Set[str], Dict[str, int]]:
    """Atomically discard interrupted lines and duplicate UIDs without losing later valid records."""
    if not path.exists():
        return set(), {
            "invalid_lines_removed": 0,
            "duplicate_lines_removed": 0,
            "split_labels_repaired": 0,
        }
    temp = path.with_suffix(path.suffix + ".repair.tmp")
    uids: Set[str] = set()
    invalid = duplicates = split_repaired = 0
    with path.open("r", encoding="utf-8", errors="replace") as source, temp.open("w", encoding="utf-8") as target:
        for line in source:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue
            uid = str(record.get("uid", ""))
            if not uid:
                invalid += 1
                continue
            if uid in uids:
                duplicates += 1
                continue
            if split_by_uid is not None:
                expected_split = split_by_uid.get(uid)
                if expected_split is None:
                    invalid += 1
                    continue
                if str(record.get("split", "")) != expected_split:
                    record["split"] = expected_split
                    split_repaired += 1
            target.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            uids.add(uid)
    os.replace(temp, path)
    return uids, {
        "invalid_lines_removed": invalid,
        "duplicate_lines_removed": duplicates,
        "split_labels_repaired": split_repaired,
    }


def _summarize_compact_dataset(path: Path, failures: int = 0) -> Dict[str, Any]:
    node_counts: Counter = Counter()
    relation_counts: Counter = Counter()
    node_samples: Counter = Counter()
    relation_samples: Counter = Counter()
    split_counts: Counter = Counter()
    ready_split_counts: Counter = Counter()
    records = 0
    ready = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records += 1
            split = str(record.get("split", "unknown"))
            is_ready = bool(record.get("quality_hint", {}).get("motif_prior_ready", False))
            ready += int(is_ready)
            split_counts[split] += 1
            if is_ready:
                ready_split_counts[split] += 1
            nodes = [str(node.get("type", "")) for node in record.get("motif_nodes", [])]
            relations = [str(rel.get("type", "")) for rel in record.get("motif_relations", [])]
            node_counts.update(nodes)
            relation_counts.update(relations)
            node_samples.update(set(nodes))
            relation_samples.update(set(relations))
    return {
        "schema_version": "innovation1_compact_dataset_summary_v2",
        "records": records,
        "failures_in_current_run": int(failures),
        "motif_prior_ready": ready,
        "split_counts": dict(split_counts),
        "motif_prior_ready_split_counts": dict(ready_split_counts),
        "node_counts": dict(node_counts),
        "node_sample_counts": dict(node_samples),
        "relation_counts": dict(relation_counts),
        "relation_sample_counts": dict(relation_samples),
        "dataset": str(path.resolve()).replace("\\", "/"),
        "storage_policy": "one compact JSONL record per accepted solid; no duplicated per-sample JSON",
        "topology_policy": "fef_adj is retained as a compact target; explicit face adjacency is excluded from motif inputs",
    }


def _compact_source_keys(path: Path) -> Set[str]:
    keys: Set[str] = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            source_rel = str(json.loads(line).get("source_step_rel", "")).replace("\\", "/").lstrip("./")
            if source_rel:
                keys.add(source_rel.lower())
    return keys


def _step_source_key(step_path: str, step_root: str) -> str:
    try:
        return str(Path(step_path).resolve().relative_to(Path(step_root).resolve().parent)).replace("\\", "/").lower()
    except Exception:
        return normalize_path(step_path).lower()


def extract_motif_graphs(
    workdir: str,
    parsed_dir: str | None = None,
    manifest_path: str | None = None,
    limit: int = 0,
    resume: bool = True,
    debug_samples: int = 0,
) -> Dict[str, Any]:
    """Stream a resumable, compact corpus instead of seven duplicated indexes."""
    dirs = ensure_workdir(workdir)
    parsed_dir = parsed_dir or dirs["parsed"]
    manifest_path = manifest_path or os.path.join(parsed_dir, "clean_manifest.csv")
    dataset_path = Path(dirs["dataset"]) / "motif_dataset.jsonl"
    failure_path = Path(dirs["reports"]) / "motif_failures.csv"
    if not resume:
        dataset_path.unlink(missing_ok=True)
        failure_path.unlink(missing_ok=True)
    processed, repair_info = _repair_compact_dataset(dataset_path) if resume else (set(), {})
    mode = "a" if dataset_path.exists() else "w"
    failure_exists = failure_path.exists() and failure_path.stat().st_size > 0
    attempted = skipped = succeeded = failed = 0
    debug_written = 0
    with dataset_path.open(mode, encoding="utf-8") as dataset_handle, failure_path.open(
        "a" if failure_exists else "w", encoding="utf-8-sig", newline=""
    ) as failure_handle:
        failure_writer = csv.DictWriter(failure_handle, fieldnames=["uid", "error"])
        if not failure_exists:
            failure_writer.writeheader()
        for row in _iter_manifest(manifest_path, limit=limit):
            uid = str(row.get("uid", ""))
            if uid in processed:
                skipped += 1
                continue
            attempted += 1
            try:
                pkl_path = os.path.join(parsed_dir, f"{uid}.pkl")
                if not os.path.isfile(pkl_path):
                    raise FileNotFoundError(f"missing parsed cache: {pkl_path}")
                data = read_pickle(pkl_path)
                features = extract_motif_features(data)
                motif_raw = build_motif_graph(data, raw_mode=True, features=features)
                # v2 projects the audit graph directly.  The legacy M_c node
                # budget can delete reliable sheet anchors before S is built.
                prior_graph = make_motif_prior_graph(motif_raw)
                compact_record = make_compact_dataset_record(prior_graph, motif_raw, data)
                dataset_handle.write(json.dumps(compact_record, ensure_ascii=False, separators=(",", ":")) + "\n")
                processed.add(uid)
                succeeded += 1
                if debug_written < max(0, int(debug_samples)):
                    motif_compact = build_motif_graph(data, raw_mode=False, features=features)
                    evidence = build_face_evidence(data, features=features)
                    example_dir = Path(dirs["examples"]) / uid
                    write_json(example_dir / "face_evidence.json", evidence)
                    write_json(example_dir / "motif_raw.json", motif_raw)
                    write_json(example_dir / "motif_compact.json", motif_compact)
                    write_json(example_dir / "motif_prior.json", prior_graph)
                    debug_written += 1
            except Exception as exc:
                failed += 1
                failure_writer.writerow({"uid": uid, "error": str(exc)})
            if attempted % 100 == 0:
                dataset_handle.flush()
                failure_handle.flush()
            if attempted % 250 == 0:
                print(
                    f"[extract_motif] processed={attempted} new={succeeded} failed={failed} resumed={skipped}",
                    flush=True,
                )
    summary = _summarize_compact_dataset(dataset_path, failures=failed)
    summary.update({"attempted_in_current_run": attempted, "resumed_records": skipped, "new_records": succeeded, **repair_info})
    summary_path = Path(dirs["reports"]) / "motif_dataset_summary.json"
    write_json(summary_path, summary)
    return {**summary, "summary": str(summary_path), "failures": str(failure_path)}


def _direct_reject_record(
    idx: int,
    step_path: str,
    step_root: str,
    source: str,
    stage: str,
    reason: str,
    error: str,
    data: Dict[str, Any] | None = None,
    stats: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    data = data or {}
    stats = stats or {}
    return {
        "idx": int(idx),
        "uid": make_uid(step_path, step_root),
        "split": str(data.get("split", "")),
        "source": source,
        "step_path": normalize_path(step_path),
        "stage": stage,
        "reject_reason": reason,
        "parser_backend": stats.get("parser_backend", data.get("parser_backend", "none")),
        "geometry_sampling_quality": stats.get(
            "geometry_sampling_quality", data.get("geometry_sampling_quality", "none")
        ),
        "face_count": int(stats.get("face_count", data.get("face_count", 0)) or 0),
        "edge_count": int(stats.get("edge_count", data.get("edge_count", 0)) or 0),
        "vertex_count": int(stats.get("vertex_count", data.get("vertex_count", 0)) or 0),
        "dtg_train_compatible": int(bool(data.get("dtg_train_compatible", False))),
        "dtg_filter_reason": str(data.get("dtg_filter_reason", "parse_rejected")),
        "error": str(error),
    }


def _build_compact_one_step(
    idx: int,
    step_path: str,
    step_root: str,
    source: str,
    max_faces: int,
    dataset_split: str,
) -> Dict[str, Any]:
    """Parse one STEP and return its final compact record without writing a pickle."""
    uid = make_uid(step_path, step_root)
    try:
        data = parse_step_file(step_path)
        data["uid"] = uid
        data["source"] = source
        data["split"] = dataset_split
        data["source_step"] = normalize_path(step_path)
        try:
            # Keep ``cad_step/...`` so records remain relocatable with deepcad_data.
            data["source_step_rel"] = str(
                Path(step_path).resolve().relative_to(Path(step_root).resolve().parent)
            ).replace("\\", "/")
        except Exception:
            data["source_step_rel"] = normalize_path(step_path)
        data = ensure_minimal_fields(data)
        ok, reason, stats = validate_brep(data, max_faces=max_faces)
        dtg_ok, dtg_reason, _ = check_dtg_train_compatible(data, dataset=source)
        data["dtg_train_compatible"] = int(dtg_ok)
        data["dtg_filter_reason"] = str(dtg_reason)
        if not ok:
            return {
                "idx": idx,
                "record": None,
                "rejected": _direct_reject_record(
                    idx,
                    step_path,
                    step_root,
                    source,
                    "filter",
                    str(reason),
                    str(data.get("parser_warning", "")),
                    data=data,
                    stats=stats,
                ),
            }
        features = extract_motif_features(data)
        motif_raw = build_motif_graph(data, raw_mode=True, features=features)
        # The formal v2 path is M_raw -> reliability projection S.  M_c remains
        # available only in explicit debug examples for historical comparison.
        prior_graph = make_motif_prior_graph(motif_raw)
        record = make_compact_dataset_record(prior_graph, motif_raw, data)
        record["split"] = dataset_split
        return {"idx": idx, "record": record, "rejected": None}
    except BrepParseError as exc:
        stage = "filter" if exc.reason == "not_single_solid" else "parse"
        return {
            "idx": idx,
            "record": None,
            "rejected": _direct_reject_record(
                idx,
                step_path,
                step_root,
                source,
                stage,
                exc.reason,
                exc.detail or str(exc),
                data={"split": dataset_split},
            ),
        }
    except Exception as exc:
        return {
            "idx": idx,
            "record": None,
            "rejected": _direct_reject_record(
                idx,
                step_path,
                step_root,
                source,
                "extract",
                "compact_build_failed",
                f"{type(exc).__name__}: {exc}",
                data={"split": dataset_split},
            ),
        }


def build_compact_dataset_from_steps(
    step_root: str,
    workdir: str,
    source: str = "deepcad",
    limit: int = 0,
    max_faces: int = 30,
    num_workers: int = 4,
    task_timeout_sec: int = 900,
    resume: bool = True,
    split_file: str | None = None,
) -> Dict[str, Any]:
    """Build Innovation-1's only formal dataset directly from STEP files.

    Accepted solids are appended immediately to one repairable JSONL file. Parsed
    geometry stays inside a worker and is released after its compact record is
    returned, which avoids a full-dataset pickle cache during large runs.
    """
    dirs = ensure_workdir(workdir)
    step_root = str(Path(step_root).resolve())
    step_files = scan_step_files(step_root, limit=limit)
    if split_file:
        split_by_uid, source_split_counts = load_dataset_split(split_file)
        unknown_uids = sorted(
            make_uid(step_path, step_root)
            for step_path in step_files
            if make_uid(step_path, step_root) not in split_by_uid
        )
        if unknown_uids:
            raise ValueError(
                f"{len(unknown_uids)} STEP files are absent from dataset split "
                f"(first: {unknown_uids[0]})"
            )
    else:
        split_by_uid = {make_uid(step_path, step_root): "all" for step_path in step_files}
        source_split_counts = {"all": len(step_files)}
    dataset_path = Path(dirs["dataset"]) / "motif_dataset.jsonl"
    rejected_path = Path(dirs["reports"]) / "compact_build_rejected.csv"
    if not resume:
        dataset_path.unlink(missing_ok=True)
        rejected_path.unlink(missing_ok=True)
    processed, repair_info = (
        _repair_compact_dataset(dataset_path, split_by_uid=split_by_uid)
        if resume
        else (set(), {})
    )
    processed_sources = _compact_source_keys(dataset_path) if resume else set()
    rejected_by_uid = {
        str(row.get("uid", "")): row for row in read_csv(rejected_path) if str(row.get("uid", ""))
    }
    queue = deque(
        (idx, step_path, split_by_uid[make_uid(step_path, step_root)])
        for idx, step_path in enumerate(step_files, start=1)
        if make_uid(step_path, step_root) not in processed
        and _step_source_key(step_path, step_root) not in processed_sources
    )
    resumed = len(step_files) - len(queue)
    new_records = failed = completed = 0
    worker_count = min(max(1, int(num_workers or 1)), max(1, len(queue)))
    timeout_sec = max(60, int(task_timeout_sec or 900))
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if dataset_path.exists() else "w"

    def consume(result: Dict[str, Any], handle: Any) -> None:
        nonlocal new_records, failed, completed
        completed += 1
        record = result.get("record")
        rejected = result.get("rejected")
        if record:
            uid = str(record.get("uid", ""))
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            processed.add(uid)
            source_rel = str(record.get("source_step_rel", "")).replace("\\", "/").lstrip("./").lower()
            if source_rel:
                processed_sources.add(source_rel)
            rejected_by_uid.pop(uid, None)
            new_records += 1
        elif rejected:
            rejected_by_uid[str(rejected.get("uid", ""))] = rejected
            failed += 1
        if completed % 50 == 0:
            handle.flush()
        if completed % 250 == 0 or completed == len(step_files) - resumed:
            print(
                f"[build_compact] current={completed}/{len(step_files) - resumed} "
                f"new={new_records} rejected={failed} resumed={resumed}",
                flush=True,
            )

    with dataset_path.open(mode, encoding="utf-8") as dataset_handle:
        if worker_count <= 1:
            while queue:
                idx, step_path, dataset_split = queue.popleft()
                consume(
                    _build_compact_one_step(
                        idx,
                        step_path,
                        step_root,
                        source,
                        max_faces,
                        dataset_split,
                    ),
                    dataset_handle,
                )
        else:
            print(
                f"[build_compact] workers={worker_count}, pending={len(queue)}, "
                f"timeout={timeout_sec}s, resumed={resumed}",
                flush=True,
            )
            while queue:
                active: Dict[Any, Tuple[int, str, str, float]] = {}
                executor = ProcessPoolExecutor(max_workers=worker_count)
                terminated = False
                try:
                    while queue or active:
                        while queue and len(active) < worker_count:
                            idx, step_path, dataset_split = queue.popleft()
                            future = executor.submit(
                                _build_compact_one_step,
                                idx,
                                step_path,
                                step_root,
                                source,
                                max_faces,
                                dataset_split,
                            )
                            active[future] = (idx, step_path, dataset_split, time.time())
                        if not active:
                            break
                        done, _ = wait(active.keys(), timeout=5.0, return_when=FIRST_COMPLETED)
                        now = time.time()
                        if done:
                            for future in done:
                                idx, step_path, dataset_split, _ = active.pop(future)
                                try:
                                    result = future.result()
                                except Exception as exc:
                                    result = {
                                        "idx": idx,
                                        "record": None,
                                        "rejected": _direct_reject_record(
                                            idx,
                                            step_path,
                                            step_root,
                                            source,
                                            "worker",
                                            "compact_worker_failed",
                                            f"{type(exc).__name__}: {exc}",
                                            data={"split": dataset_split},
                                        ),
                                    }
                                consume(result, dataset_handle)
                            continue
                        stalled = [
                            (future, meta) for future, meta in active.items() if now - meta[3] >= timeout_sec
                        ]
                        if not stalled:
                            continue
                        stalled_futures = {future for future, _ in stalled}
                        retry_items = [
                            (idx, step_path, dataset_split)
                            for future, (idx, step_path, dataset_split, _) in active.items()
                            if future not in stalled_futures
                        ]
                        for future, (idx, step_path, dataset_split, _) in stalled:
                            consume(
                                {
                                    "idx": idx,
                                    "record": None,
                                    "rejected": _direct_reject_record(
                                        idx,
                                        step_path,
                                        step_root,
                                        source,
                                        "parse",
                                        "compact_build_timeout",
                                        f"single STEP exceeded {timeout_sec} seconds",
                                        data={"split": dataset_split},
                                    ),
                                },
                                dataset_handle,
                            )
                        for item in reversed(retry_items):
                            queue.appendleft(item)
                        active.clear()
                        _terminate_executor_workers(executor)
                        terminated = True
                        print(
                            f"[build_compact] timed out={len(stalled)}; requeued in-flight={len(retry_items)}",
                            flush=True,
                        )
                        break
                finally:
                    try:
                        executor.shutdown(wait=not terminated, cancel_futures=True)
                    except TypeError:
                        executor.shutdown(wait=not terminated)
                    except Exception:
                        pass
        dataset_handle.flush()

    rejected_fields = [
        "uid",
        "split",
        "source",
        "step_path",
        "stage",
        "reject_reason",
        "parser_backend",
        "geometry_sampling_quality",
        "face_count",
        "edge_count",
        "vertex_count",
        "dtg_train_compatible",
        "dtg_filter_reason",
        "error",
    ]
    write_csv(rejected_path, list(rejected_by_uid.values()), rejected_fields)
    summary = _summarize_compact_dataset(dataset_path, failures=failed)
    summary.update(
        {
            "step_files_scanned": len(step_files),
            "resumed_records": resumed,
            "new_records": new_records,
            "rejected_in_current_run": failed,
            "rejected_total": len(rejected_by_uid),
            "temporary_pickle_written": False,
            "split_file": normalize_path(split_file) if split_file else "",
            "source_split_counts": source_split_counts,
            "split_policy": "DTG deepcad_data_split_6bit labels; no random reassignment" if split_file else "all",
            **repair_info,
        }
    )
    summary_path = Path(dirs["reports"]) / "motif_dataset_summary.json"
    write_json(summary_path, summary)
    return {
        **summary,
        "summary": str(summary_path),
        "rejected_manifest": str(rejected_path),
    }
