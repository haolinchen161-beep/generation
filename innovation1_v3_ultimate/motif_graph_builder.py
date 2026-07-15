# -*- coding: utf-8 -*-
"""由 face-level evidence 构建弱结构基元图 M=(Vm, Em, Pm)。"""

from __future__ import annotations

import copy
import os
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

try:  # pragma: no cover
    from .motif_feature_extractor import extract_motif_features
    from .utils_io import NODE_TYPES, RELATION_TYPES, ensure_workdir, read_csv, read_pickle, write_json, write_jsonl
except ImportError:  # pragma: no cover
    from motif_feature_extractor import extract_motif_features
    from utils_io import NODE_TYPES, RELATION_TYPES, ensure_workdir, read_csv, read_pickle, write_json, write_jsonl


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
    "transition_group",
    "repeated_feature",
    "boundary_group",
}

PRIOR_RELATION_TYPES = {
    "parallel_to",
    "opposite_to",
    "orthogonal_to",
    "coplanar_with",
    "repeated_with",
    "bounded_by",
    "thin_wall_pair",
    "has_member",
    "hosted_by",
}

PRIOR_NODE_CONFIDENCE_MIN = {
    "sheet_region": 0.58,
    "loop_or_hole": 0.58,
    "transition_group": 0.60,
    "repeated_feature": 0.55,
    "boundary_group": 0.72,
}

PRIOR_RELATION_CONFIDENCE_MIN = {
    "parallel_to": 0.72,
    "opposite_to": 0.62,
    "orthogonal_to": 0.62,
    "coplanar_with": 0.64,
    "repeated_with": 0.50,
    "bounded_by": 0.58,
    "thin_wall_pair": 0.58,
    "has_member": 0.50,
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
    return float(np.percentile(np.asarray(vals, dtype=np.float32), q))


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
            "regular_score": 0.75
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


def _similar_group_signature(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[bool, Dict[str, float]]:
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
    evidence = {
        "dims_rel_diff": dims_rel_diff,
        "area_ratio": area_ratio,
        "normal_absdot": normal_absdot,
        "face_count_gap": float(face_count_gap),
        "degree_gap": degree_gap,
    }
    similar = (
        dims_rel_diff <= 0.28
        and area_ratio >= 0.62
        and normal_absdot >= 0.92
        and face_count_gap <= 1
        and degree_gap <= 2.5
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
    structural_graph["motif_prior"]["support_relations_available_in"] = "motif_graph_index.jsonl，完整图中保留已限量的 support relations"
    return structural_graph


def _prior_node_keep(node: Dict[str, Any]) -> bool:
    node_type = str(node.get("type", ""))
    if node_type not in PRIOR_NODE_TYPES:
        return False
    confidence = float(node.get("confidence", 0.0))
    return confidence >= PRIOR_NODE_CONFIDENCE_MIN.get(node_type, 0.65)


def _prior_relation_keep(rel: Dict[str, Any]) -> bool:
    rel_type = str(rel.get("type", ""))
    if rel_type not in PRIOR_RELATION_TYPES:
        return False
    if str(rel.get("relation_role", _relation_role(rel_type))) != "structural":
        return False
    confidence = float(rel.get("confidence", 0.0))
    return confidence >= PRIOR_RELATION_CONFIDENCE_MIN.get(rel_type, 0.65)


def _node_support_group_ids(node: Dict[str, Any]) -> Set[str]:
    node_id = str(node.get("id", ""))
    node_type = str(node.get("type", ""))
    if node_type == "face_group":
        return {node_id}
    features = node.get("features", {}) or {}
    support_ids: Set[str] = set()
    for key in ["base_face_group_id"]:
        value = features.get(key)
        if value:
            support_ids.add(str(value))
    for key in ["base_face_group_ids", "member_face_group_ids"]:
        for value in features.get(key, []) or []:
            support_ids.add(str(value))
    return support_ids


def _prior_relation_sort_key(rel: Dict[str, Any]) -> Tuple[int, float, str, str, str]:
    priority = {
        "thin_wall_pair": 0,
        "has_member": 1,
        "hosted_by": 2,
        "repeated_with": 3,
        "bounded_by": 4,
        "opposite_to": 5,
        "coplanar_with": 6,
        "orthogonal_to": 7,
        "parallel_to": 8,
        "smooth_connected": 9,
    }
    return (
        priority.get(str(rel.get("type", "")), 99),
        -float(rel.get("confidence", 0.0)),
        str(rel.get("source", "")),
        str(rel.get("target", "")),
        str(rel.get("type", "")),
    )


def _add_prior_relation(
    relation_records: Dict[Tuple[str, str, str], Dict[str, Any]],
    src_id: str,
    dst_id: str,
    rel: Dict[str, Any],
    source_relation: Dict[str, Any],
) -> None:
    if src_id == dst_id or not _prior_relation_keep(rel):
        return
    rel_type = str(rel.get("type", ""))
    key = (src_id, dst_id, rel_type)
    evidence = copy.deepcopy(rel.get("evidence", {}) or {})
    evidence["distilled_from"] = {
        "source": source_relation.get("source", rel.get("source", "")),
        "target": source_relation.get("target", rel.get("target", "")),
        "type": source_relation.get("type", rel_type),
    }
    record = {
        "source": src_id,
        "target": dst_id,
        "type": rel_type,
        "confidence": float(rel.get("confidence", 0.0)),
        "relation_role": "structural",
        "evidence": evidence,
    }
    if key not in relation_records or record["confidence"] > float(relation_records[key].get("confidence", 0.0)):
        relation_records[key] = record


def _prune_prior_relations(relations: Sequence[Dict[str, Any]], node_count: int) -> List[Dict[str, Any]]:
    per_type_degree_caps = {
        "parallel_to": 2,
        "opposite_to": 3,
        "orthogonal_to": 3,
        "coplanar_with": 2,
        "smooth_connected": 1,
        "repeated_with": 3,
        "bounded_by": 2,
        "thin_wall_pair": 2,
        "has_member": 6,
        "hosted_by": 6,
    }
    max_edges = max(2, min(3 * max(node_count, 1), 48))
    degree_by_type: Dict[Tuple[str, str], int] = {}
    kept: List[Dict[str, Any]] = []
    for rel in sorted(relations, key=_prior_relation_sort_key):
        rel_type = str(rel.get("type", ""))
        cap = per_type_degree_caps.get(rel_type, 2)
        src = str(rel.get("source", ""))
        dst = str(rel.get("target", ""))
        if degree_by_type.get((src, rel_type), 0) >= cap or degree_by_type.get((dst, rel_type), 0) >= cap:
            continue
        kept.append(dict(rel))
        degree_by_type[(src, rel_type)] = degree_by_type.get((src, rel_type), 0) + 1
        degree_by_type[(dst, rel_type)] = degree_by_type.get((dst, rel_type), 0) + 1
        if len(kept) >= max_edges:
            break
    return kept


def _make_prior_stats(graph: Dict[str, Any], nodes: Sequence[Dict[str, Any]], relations: Sequence[Dict[str, Any]], raw_graph: Dict[str, Any] = None) -> Dict[str, Any]:
    num_faces = max(int(graph.get("num_faces", 0)), 1)
    
    # 原始图的大小 (M_raw)
    ref_raw = raw_graph if raw_graph is not None else graph
    raw_nodes = max(len(ref_raw.get("motif_nodes", [])), 1)
    raw_relations = max(len(ref_raw.get("motif_relations", [])), 1)
    
    # 压缩图的大小 (M_c)
    compact_nodes = max(len(graph.get("motif_nodes", [])), 1)
    compact_relations = max(len(graph.get("motif_relations", [])), 1)
    
    # 三个层级的节点+关系总数
    size_raw = raw_nodes + raw_relations
    size_compact = compact_nodes + compact_relations
    size_prior = len(nodes) + len(relations)
    
    # 三个比率的计算
    ratio_Mraw_to_Mc = float(size_compact / size_raw)
    ratio_Mc_to_S = float(size_prior / size_compact)
    ratio_Mraw_to_S = float(size_prior / size_raw)
    
    covered_faces = sorted({int(fid) for node in nodes for fid in node.get("face_ids", [])})
    prior_types = sorted({str(node.get("type", "")) for node in nodes})
    prior_relation_types = sorted({str(rel.get("type", "")) for rel in relations})
    
    compression = ratio_Mc_to_S
    core_types = [typ for typ in prior_types if typ != "boundary_group"]

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
        graph.get("motif_quality", {}).get("motif_ready", False)
        and len(core_types) >= 1
        and len(nodes) >= 2
        and len(relations) <= max(4, 3 * max(len(nodes), 1))
        and r_isolated <= 0.40       # 孤立节点占比上限
        and r_coverage >= 0.45       # 核心面片覆盖率下限
        and r_redundancy <= 0.60     # 面片过度覆盖冗余度上限
    )
    return {
        "num_prior_nodes": len(nodes),
        "num_prior_relations": len(relations),
        "prior_node_density": float(len(nodes) / num_faces),
        "prior_relation_density": float(len(relations) / max(len(nodes), 1)),
        "prior_retention_ratio": compression,
        "prior_reduction_ratio": 1.0 - compression,
        "prior_compression_ratio": compression,  # 兼容旧代码
        "ratio_Mraw_to_Mc": ratio_Mraw_to_Mc,
        "ratio_Mc_to_S": ratio_Mc_to_S,
        "ratio_Mraw_to_S": ratio_Mraw_to_S,
        "prior_node_retention_ratio": float(len(nodes) / compact_nodes),
        "prior_node_compression_ratio": float(len(nodes) / compact_nodes),  # 兼容旧代码
        "prior_relation_retention_ratio": float(len(relations) / compact_relations),
        "prior_relation_compression_ratio": float(len(relations) / compact_relations),  # 兼容旧代码
        "prior_node_face_reduction_ratio": 1.0 - float(len(nodes) / num_faces),
        "prior_coverage_faces": len(covered_faces),
        "prior_coverage_face_ratio": float(len(covered_faces) / num_faces),
        "prior_isolated_ratio": r_isolated,
        "prior_redundancy_ratio": r_redundancy,
        "prior_motif_types": prior_types,
        "prior_relation_types": prior_relation_types,
        "prior_ready": prior_ready,
    }


SYMMETRIC_PRIOR_RELATIONS = {
    "parallel_to",
    "coplanar_with",
    "orthogonal_to",
    "opposite_to",
}

PROMOTED_GEOMETRIC_RELATIONS = set(SYMMETRIC_PRIOR_RELATIONS)


def _unordered_type_pair(src_type: str, dst_type: str) -> Tuple[str, str]:
    return tuple(sorted((str(src_type), str(dst_type))))


def _is_prior_relation_compatible(rel_type: str, src_type: str, dst_type: str) -> bool:
    # 严格先验关系类型兼容表，防止底层关系的笛卡尔积扩散造成宏观先验的语义错乱
    rel_type = str(rel_type)
    src_type = str(src_type)
    dst_type = str(dst_type)

    if rel_type in SYMMETRIC_PRIOR_RELATIONS:
        # 平行、共面、正交、相对关系仅限在两个主要板系统 (sheet_region) 之间建立
        return _unordered_type_pair(src_type, dst_type) == ("sheet_region", "sheet_region")

    if rel_type == "bounded_by":
        # 有界关系仅限在局部开口/圆角特征与主体主板/边界角色之间建立（单向：局部基元 bounded_by 宿主）
        return src_type in {"loop_or_hole", "transition_group"} and dst_type in {"sheet_region", "boundary_group"}

    if rel_type == "thin_wall_pair":
        return _unordered_type_pair(src_type, dst_type) == ("sheet_region", "sheet_region")
    if rel_type == "has_member":
        return src_type == "repeated_feature" and dst_type in PRIOR_NODE_TYPES
    if rel_type == "hosted_by":
        return src_type in {"loop_or_hole", "transition_group"} and dst_type in {"sheet_region", "boundary_group"}

    # 重复关系 (repeated_with) 等其它关系不采用底层关系提升方式
    return False


def make_motif_prior_graph(graph: Dict[str, Any], raw_graph: Dict[str, Any] = None) -> Dict[str, Any]:
    """由压缩图 M_c 蒸馏生成网络使用的稀疏结构先验 S。"""
    raw_nodes = list(graph.get("motif_nodes", []))
    raw_relations = _annotate_relation_roles(graph.get("motif_relations", []))
    node_by_id = {str(node.get("id")): node for node in raw_nodes}
    selected_raw_nodes = [copy.deepcopy(node) for node in raw_nodes if _prior_node_keep(node)]
    selected_raw_ids = {str(node.get("id")) for node in selected_raw_nodes}

    support_to_prior: Dict[str, Set[str]] = {}
    for node in selected_raw_nodes:
        raw_id = str(node.get("id"))
        for support_id in _node_support_group_ids(node):
            support_to_prior.setdefault(support_id, set()).add(raw_id)

    relation_records: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    relation_buckets = {}

    # 构建物理面到高层结构节点的索引，以实现跨节点连接
    raw_node_by_id = {}
    if raw_graph is not None:
        raw_node_by_id = {str(node.get("id")): node for node in raw_graph.get("motif_nodes", [])}
    else:
        raw_node_by_id = {str(node.get("id")): node for node in raw_nodes}

    face_to_structural: Dict[int, Set[str]] = {}
    for node in selected_raw_nodes:
        node_id = str(node.get("id"))
        for fid in node.get("face_ids", []):
            face_to_structural.setdefault(int(fid), set()).add(node_id)

    for rel in raw_relations:
        if not _prior_relation_keep(rel):
            continue
        rel_type = str(rel.get("type", ""))
        src = str(rel.get("source", ""))
        dst = str(rel.get("target", ""))

        src_node = node_by_id.get(src)
        dst_node = node_by_id.get(dst)

        # 步骤一：处理原本已经是高层节点之间的关系（例如直接构建的 bounded_by 关系等）
        if src in selected_raw_ids and dst in selected_raw_ids:
            if src_node is not None and dst_node is not None:
                src_type = str(src_node.get("type", ""))
                dst_type = str(dst_node.get("type", ""))
                if _is_prior_relation_compatible(rel_type, src_type, dst_type):
                    _add_prior_relation(relation_records, src, dst, rel, rel)
            continue

        # 步骤二：只允许特定的解析几何关系从底层面组传播提升
        if rel_type not in PROMOTED_GEOMETRIC_RELATIONS:
            continue

        src_candidates = set(support_to_prior.get(src, set()))
        dst_candidates = set(support_to_prior.get(dst, set()))

        for src_id in src_candidates:
            for dst_id in dst_candidates:
                if src_id == dst_id:
                    continue

                src_prior = node_by_id[src_id]
                dst_prior = node_by_id[dst_id]
                src_type = str(src_prior.get("type", ""))
                dst_type = str(dst_prior.get("type", ""))

                if not _is_prior_relation_compatible(rel_type, src_type, dst_type):
                    continue

                # 保护条件 1：如果支撑面片区域存在重叠，则不能建立结构方向关系（例如同一面组同时扮演板和边界角色）
                src_faces = set(src_prior.get("face_ids", []))
                dst_faces = set(dst_prior.get("face_ids", []))
                if src_faces & dst_faces:
                    continue

                # 保护条件 2：生成先验 S 时，方向关系（平行、共面、正交、相对）必须要求解析级的高质量证据
                evidence_quality = rel.get("evidence", {}).get("example_evidence", {}).get("relation_evidence_quality", "heuristic")
                if evidence_quality not in {"analytical", "mixed_analytical"}:
                    continue

                # 对称关系按端点排序统一 key，方便步骤三进行见证聚合
                a, b = sorted((src_id, dst_id))
                key = (a, b, rel_type)
                relation_buckets.setdefault(key, []).append(rel)

    # 步骤三：聚合底层见证（Witness Aggregation）后再建立高层关系边
    for (src_id, dst_id, rel_type), witnesses in relation_buckets.items():
        confidences = [float(item.get("confidence", 0.0)) for item in witnesses]
        
        unique_support_pairs = {
            tuple(sorted((str(item.get("source", "")), str(item.get("target", "")))))
            for item in witnesses
        }
        
        max_conf = max(confidences)
        mean_conf = sum(confidences) / max(len(confidences), 1)
        
        # 计算底层面组见证覆盖率（以两端高层节点所对应的基底面组的笛卡尔积组合数作为分母）
        src_node = node_by_id.get(src_id)
        dst_node = node_by_id.get(dst_id)
        g_u = _node_support_group_ids(src_node) if src_node else set()
        g_v = _node_support_group_ids(dst_node) if dst_node else set()
        total_possible_pairs = len(g_u) * len(g_v)
        if total_possible_pairs > 0:
            support_ratio = min(1.0, float(len(unique_support_pairs)) / total_possible_pairs)
        else:
            support_ratio = 0.0
        
        # 综合计算见证聚合置信度
        aggregated_conf = 0.50 * max_conf + 0.30 * mean_conf + 0.20 * support_ratio
        
        # 寻找拥有最高置信度的底层证据，作为基础来进行字典拷贝
        source_rel = max(witnesses, key=lambda item: float(item.get("confidence", 0.0)))
        merged_rel = dict(source_rel)
        merged_rel["source"] = src_id
        merged_rel["target"] = dst_id
        merged_rel["confidence"] = aggregated_conf
        merged_rel["evidence"] = {
            "promotion_policy": "compatible_witness_aggregation",
            "support_relation_count": len(witnesses),
            "support_group_pairs": sorted(list(unique_support_pairs)),
            "max_support_confidence": max_conf,
            "mean_support_confidence": mean_conf,
            "support_ratio": support_ratio,
        }
        
        _add_prior_relation(relation_records, src_id, dst_id, merged_rel, source_rel)

    # 步骤三点五：链接新增的三类高层结构边（避免孤立节点）
    # 1. 链接薄壁关系边 thin_wall_pair
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
    connected_boundary_ids = {
        str(rel.get(end))
        for rel in prior_relations_raw
        if str(rel.get("type")) == "bounded_by"
        for end in ["source", "target"]
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
        for k in ["base_face_group_id", "base_face_group_ids", "member_face_group_ids", "larger_neighbor_group_ids"]:
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

    prior_graph = copy.deepcopy(graph)
    prior_graph["graph_view"] = "distilled_motif_prior"
    prior_graph["prior_definition"] = "M_c = C(M_raw); S = D(M_c)"
    prior_graph["source_graph_view"] = "compact_motif_graph"
    prior_graph["motif_nodes"] = prior_nodes
    prior_graph["motif_relations"] = _annotate_relation_roles(prior_relations)
    node_type_counts = {typ: _node_type_count(prior_nodes, typ) for typ in NODE_TYPES}
    relation_type_counts = {typ: sum(1 for rel in prior_graph["motif_relations"] if rel["type"] == typ) for typ in RELATION_TYPES}
    prior_stats = _make_prior_stats(graph, prior_nodes, prior_graph["motif_relations"], raw_graph)
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
    prior_quality["prior_ready_policy"] = "motif_ready + 至少一个核心结构类型 + 稀疏关系密度受控"
    prior_graph["motif_quality"] = prior_quality
    face_to_prior_nodes: Dict[str, List[str]] = {}
    for node in prior_nodes:
        for fid in node.get("face_ids", []):
            face_to_prior_nodes.setdefault(str(fid), []).append(str(node.get("id")))
    prior_graph["motif_prior"] = {
        "version": "innovation1_v3_distilled_prior_v1",
        "policy": "distilled_sparse_generation_prior",
        "prior_symbol": "S",
        "source_symbol": "M_raw",
        "node_type_vocab": NODE_TYPES,
        "relation_type_vocab": RELATION_TYPES,
        "motif_node_type_ids": [NODE_TYPES.index(node["type"]) for node in prior_nodes],
        "motif_relation_type_ids": [RELATION_TYPES.index(rel["type"]) for rel in prior_graph["motif_relations"]],
        "relation_role_vocab": ["structural", "support", "topology_support"],
        "motif_relation_role_ids": [0 for _ in prior_graph["motif_relations"]],
        "face_to_motif_nodes": face_to_prior_nodes,
        "distillation_policy": "从 M_raw 中去除 face_group / embedded_in / adjacent_to 等支撑信息，仅保留可用于无条件生成的稀疏结构骨架 S。",
    }
    prior_graph["motif_prior_distillation"] = {
        "source_raw_node_count": len(raw_nodes),
        "source_raw_relation_count": len(raw_relations),
        "selected_raw_node_count": len(selected_raw_nodes),
        "selected_prior_node_count": len(prior_nodes),
        "selected_prior_relation_count": len(prior_graph["motif_relations"]),
        "kept_node_types": sorted(PRIOR_NODE_TYPES),
        "kept_relation_types": sorted(PRIOR_RELATION_TYPES),
        "dropped_default_node_types": ["face_group"],
        "dropped_default_relation_types": ["embedded_in", "adjacent_to"],
        "policy": "M_raw 用于审计和监督信号来源；S 用于创新点二的无条件层级生成。",
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


def build_face_evidence(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    抽取完整的面级几何与拓扑证据图 E_face，不对关系或度数进行任何裁剪或过滤。
    参数:
        data: 从原始 STEP 解析出来的基础面片及曲率数据字典。
    返回:
        包含面级特征、无损拓扑及几何邻接/平行/垂直关系的完整证据字典。
    """
    features = extract_motif_features(data)
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


def build_motif_graph(data: Dict[str, Any], raw_mode: bool = False) -> Dict[str, Any]:
    features = extract_motif_features(data)
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
    sheet_area_cut = max(0.08, _percentile(group_area_values, 70.0, 0.08))
    small_area_cut = max(0.015, _percentile(group_area_values, 35.0, 0.03))
    face_group_by_id = {node["id"]: node for node in face_group_nodes}

    sheet_candidates: List[Tuple[float, Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = []
    boundary_candidates: List[Tuple[float, Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = []
    for group_node in face_group_nodes:
        gf = group_node["features"]
        rel_area = float(gf.get("relative_area_sum", 0.0))
        thinness = float(gf.get("bbox_thinness", 1.0))
        boundary_ratio = float(gf.get("boundary_ratio", 0.0))
        if rel_area >= sheet_area_cut or (rel_area >= 0.06 and thinness <= 0.08):
            confidence = 0.52 + min(0.32, rel_area) + (0.08 if thinness <= 0.08 else 0.0) + 0.04 * boundary_ratio
            sheet_candidates.append(
                (
                    confidence,
                    group_node,
                    dict(gf, base_face_group_id=group_node["id"]),
                    {"rule": "相对面积较大且 bbox 薄尺度明显", "relative_area_cut": sheet_area_cut},
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
        gap = float(evidence.get("normal_gap", 0.0))
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
        overlap = float(evidence.get("projection_overlap_ratio", 1.0))
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
                    dict(gf, base_face_group_id=group_node["id"]),
                    {
                        "rule": "内部有界局部闭合候选",
                        "note": "仅为候选，不是真实工程孔洞标签",
                        "loop_area_cut": loop_area_cut,
                        "outside_degree": outside_degree,
                    },
                )
            )
    max_loop_groups = 999999 if raw_mode else max(4, min(16, int(0.25 * face_count_for_caps) + 1))
    for confidence, group_node, node_features, evidence in sorted(loop_candidates, key=lambda x: -x[0])[:max_loop_groups]:
        add_node("loop_or_hole", group_node["face_ids"], confidence, node_features, evidence)

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

    repeat_candidates = [
        node
        for node in face_group_nodes
        if float(node["features"].get("relative_area_sum", 0.0)) <= 0.35
        and float(node["features"].get("area_proxy_sum", 0.0)) > 1e-8
    ]
    repeat_pair_evidence: Dict[Tuple[int, int], Dict[str, float]] = {}
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
        pair_support = sum(1 for key in repeat_pair_evidence if key[0] in cluster and key[1] in cluster)
        support_score = pair_support / max(len(members) * (len(members) - 1) / 2.0, 1.0)
        face_ids = sorted({fid for m in members for fid in m["face_ids"]})
        confidence = 0.38 + 0.22 * min(len(members) / 4.0, 1.0) + 0.16 * support_score
        confidence += 0.16 * spacing["regular_score"] + 0.08 * (1.0 - min(boundary_mean, 1.0))
        add_node(
            "repeated_feature",
            face_ids,
            confidence,
            {
                "member_face_group_ids": [m["id"] for m in members],
                "member_count": len(members),
                "mean_relative_area": rel_area_mean,
                "boundary_ratio_mean": boundary_mean,
                **spacing,
            },
            {
                "rule": "相似 bbox 尺寸 + 相似法向 + 相似邻接模式 + 规则间距",
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

    for node in nodes:
        if node["type"] == "repeated_feature":
            members = [gid for gid in node["features"].get("member_face_group_ids", []) if gid in face_group_by_id]
            member_nodes = [face_group_by_id[gid] for gid in members]
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


def _cleanup_generated_motif_outputs(motif_graph_dir: str) -> Dict[str, int]:
    """写入新抽取结果前清理旧的 motif JSON / index 文件。"""
    root = Path(motif_graph_dir)
    removed = 0
    for pattern in [
        "*_motif_graph.json",
        "*_motif_prior.json",
        "*_motif_raw.json",
        "*_face_evidence.json",
        "motif_graph_index.jsonl",
        "motif_graph_index_structural.jsonl",
        "motif_graph_index_ready.jsonl",
        "motif_prior_index.jsonl",
        "motif_prior_index_ready.jsonl",
        "face_evidence_index.jsonl",
        "motif_raw_index.jsonl",
    ]:
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            path.unlink()
            removed += 1
    return {"removed_stale_motif_files": removed}


def extract_motif_graphs(workdir: str, parsed_dir: str | None = None, manifest_path: str | None = None) -> Dict[str, Any]:
    dirs = ensure_workdir(workdir)
    parsed_dir = parsed_dir or dirs["parsed"]
    manifest_path = manifest_path or os.path.join(parsed_dir, "clean_manifest.csv")
    rows = read_csv(manifest_path)
    cleanup_info = _cleanup_generated_motif_outputs(dirs["motif_graphs"])
    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    index_records: List[Dict[str, Any]] = []
    structural_index_records: List[Dict[str, Any]] = []
    prior_index_records: List[Dict[str, Any]] = []
    face_evidence_records: List[Dict[str, Any]] = []
    motif_raw_records: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        uid = str(row.get("uid", ""))
        pkl_path = row.get("pkl_path") or os.path.join(parsed_dir, f"{uid}.pkl")
        try:
            data = read_pickle(pkl_path)
            
            # 分层抽取：
            # 1. E_face (无损面级证据)
            face_evidence = build_face_evidence(data)
            # 2. M_raw (高层候选基元，不进行数量和度数限制)
            motif_raw = build_motif_graph(data, raw_mode=True)
            # 3. M_compact (压缩后的基元图)
            motif_compact = build_motif_graph(data, raw_mode=False)
            # 4. S (用于神经网络生成的稀疏先验图)
            prior_graph = make_motif_prior_graph(motif_compact, raw_graph=motif_raw)
            
            structural_graph = make_structural_graph(motif_compact)
            
            # 各层文件的独立写入
            evidence_path = os.path.join(dirs["motif_graphs"], f"{uid}_face_evidence.json")
            raw_path = os.path.join(dirs["motif_graphs"], f"{uid}_motif_raw.json")
            graph_path = os.path.join(dirs["motif_graphs"], f"{uid}_motif_graph.json")
            prior_path = os.path.join(dirs["motif_graphs"], f"{uid}_motif_prior.json")
            
            write_json(evidence_path, face_evidence)
            write_json(raw_path, motif_raw)
            write_json(graph_path, motif_compact)
            write_json(prior_path, prior_graph)
            
            face_evidence_records.append(face_evidence)
            motif_raw_records.append(motif_raw)
            index_records.append(motif_compact)
            structural_index_records.append(structural_graph)
            prior_index_records.append(prior_graph)
            
            records.append(
                {
                    "uid": uid,
                    "status": "SUCCESS",
                    "motif_graph_path": graph_path,
                    "motif_prior_path": prior_path,
                    "num_nodes": len(motif_compact.get("motif_nodes", [])),
                    "num_relations": len(motif_compact.get("motif_relations", [])),
                    "num_prior_nodes": len(prior_graph.get("motif_nodes", [])),
                    "num_prior_relations": len(prior_graph.get("motif_relations", [])),
                    "num_faces": motif_compact.get("num_faces", 0),
                    "error": "",
                }
            )
        except Exception as exc:
            failures.append({"uid": uid, "status": "FAILED", "error": str(exc)})
            records.append(
                {
                    "uid": uid,
                    "status": "FAILED",
                    "motif_graph_path": "",
                    "motif_prior_path": "",
                    "num_nodes": 0,
                    "num_relations": 0,
                    "num_prior_nodes": 0,
                    "num_prior_relations": 0,
                    "num_faces": row.get("face_count", 0),
                    "error": str(exc),
                }
            )
        if idx % 250 == 0:
            print(f"[extract_motif] 已处理 {idx}/{len(rows)}；成功={len(index_records)} 失败={len(failures)}")

    index_path = os.path.join(dirs["motif_graphs"], "motif_graph_index.jsonl")
    structural_index_path = os.path.join(dirs["motif_graphs"], "motif_graph_index_structural.jsonl")
    ready_index_path = os.path.join(dirs["motif_graphs"], "motif_graph_index_ready.jsonl")
    prior_index_path = os.path.join(dirs["motif_graphs"], "motif_prior_index.jsonl")
    prior_ready_index_path = os.path.join(dirs["motif_graphs"], "motif_prior_index_ready.jsonl")
    
    # 论文三层架构的对应索引输出
    face_evidence_index_path = os.path.join(dirs["motif_graphs"], "face_evidence_index.jsonl")
    motif_raw_index_path = os.path.join(dirs["motif_graphs"], "motif_raw_index.jsonl")
    
    ready_index_records = []
    prior_ready_index_records = []
    for full_graph, structural_graph, prior_graph in zip(index_records, structural_index_records, prior_index_records):
        if bool(full_graph.get("motif_quality", {}).get("motif_ready", False)):
            ready_index_records.append(structural_graph)
        if bool(prior_graph.get("motif_quality", {}).get("motif_prior_ready", False)):
            prior_ready_index_records.append(prior_graph)
            
    write_jsonl(index_path, index_records)
    write_jsonl(structural_index_path, structural_index_records)
    write_jsonl(ready_index_path, ready_index_records)
    write_jsonl(prior_index_path, prior_index_records)
    write_jsonl(prior_ready_index_path, prior_ready_index_records)
    
    write_jsonl(face_evidence_index_path, face_evidence_records)
    write_jsonl(motif_raw_index_path, motif_raw_records)
    
    write_json(
        os.path.join(dirs["reports"], "motif_extraction_summary.json"),
        {
            "input_clean_samples": len(rows),
            "motif_graph_success_count": len(index_records),
            "motif_graph_structural_count": len(structural_index_records),
            "motif_ready_count": len(ready_index_records),
            "motif_prior_count": len(prior_index_records),
            "motif_prior_ready_count": len(prior_ready_index_records),
            "motif_graph_failure_count": len(failures),
            "motif_graph_index": index_path,
            "motif_graph_index_structural": structural_index_path,
            "motif_graph_index_ready": ready_index_path,
            "motif_prior_index": prior_index_path,
            "motif_prior_index_ready": prior_ready_index_path,
            "face_evidence_index": face_evidence_index_path,
            "motif_raw_index": motif_raw_index_path,
            "ready_index_graph_view": "structural_only",
            "default_generation_prior_index": prior_ready_index_path,
            **cleanup_info,
        },
    )
    return {
        "records": records,
        "failures": failures,
        "graphs": index_records,
        "structural_graphs": structural_index_records,
        "ready_graphs": ready_index_records,
        "prior_graphs": prior_index_records,
        "prior_ready_graphs": prior_ready_index_records,
        "face_evidence_graphs": face_evidence_records,
        "motif_raw_graphs": motif_raw_records,
        "motif_graph_index": index_path,
        "motif_graph_index_structural": structural_index_path,
        "motif_graph_index_ready": ready_index_path,
        "motif_prior_index": prior_index_path,
        "motif_prior_index_ready": prior_ready_index_path,
        "face_evidence_index": face_evidence_index_path,
        "motif_raw_index": motif_raw_index_path,
        **cleanup_info,
    }
