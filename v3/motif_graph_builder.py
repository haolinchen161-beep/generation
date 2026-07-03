# -*- coding: utf-8 -*-
"""Build weak B-Rep motif graph M=(Vm, Em, Pm) from face evidence."""

from __future__ import annotations

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


def _group_features(face_ids: Sequence[int], face_features: Sequence[Dict[str, Any]], global_scale: float) -> Dict[str, Any]:
    faces = [face_features[fid] for fid in face_ids]
    bboxes = np.asarray([f["bbox"] for f in faces], dtype=np.float32)
    centroids = np.asarray([f["centroid"] for f in faces], dtype=np.float32)
    mn = np.min(bboxes[:, :3], axis=0)
    mx = np.max(bboxes[:, 3:], axis=0)
    dims = np.maximum(mx - mn, 0.0)
    sorted_dims = np.sort(np.maximum(dims, 1e-8))
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
        "normal_proxy": _safe_normal([f["normal_proxy"] for f in faces]),
        "mean_aspect_ratio": float(np.mean(aspect_values)) if aspect_values else 1.0,
        "max_aspect_ratio": float(np.max(aspect_values)) if aspect_values else 1.0,
        "mean_face_degree": float(np.mean([float(f.get("face_degree", 0)) for f in faces])) if faces else 0.0,
        "boundary_ratio": boundary_ratio,
        "boundary_flag": bool(boundary_ratio >= 0.5),
        "adjacent_faces_outside": adjacency_faces,
        "curvature_proxy_mean": float(np.mean([float(f.get("curvature_proxy", 0.0)) for f in faces])) if faces else 0.0,
    }


def _build_base_face_groups(features: Dict[str, Any]) -> List[List[int]]:
    face_count = int(features.get("face_count", 0))
    uf = UnionFind(face_count)
    pair_types = _relation_type_sets(features.get("face_relations", []))
    for (i, j), types in pair_types.items():
        if "adjacent_to" in types and ("coplanar_with" in types or "smooth_connected" in types):
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


def _spacing_regular_score(centroids: np.ndarray) -> Dict[str, float]:
    if centroids.shape[0] <= 1:
        return {"linearity": 0.0, "spacing_cv": 1.0, "regular_score": 0.0}
    if centroids.shape[0] == 2:
        return {"linearity": 1.0, "spacing_cv": 0.0, "regular_score": 0.55}
    centered = centroids - np.mean(centroids, axis=0, keepdims=True)
    try:
        _, svals, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0]
        total_var = float(np.sum(svals**2))
        linearity = float((svals[0] ** 2) / max(total_var, 1e-8))
        proj = np.sort(centered @ direction)
        gaps = np.diff(proj)
        mean_gap = float(np.mean(np.abs(gaps)))
        spacing_cv = float(np.std(gaps) / max(mean_gap, 1e-8))
        regular_score = float(max(0.0, min(1.0, linearity * (1.0 - min(spacing_cv, 1.0)))))
        return {"linearity": linearity, "spacing_cv": spacing_cv, "regular_score": regular_score}
    except Exception:
        return {"linearity": 0.0, "spacing_cv": 1.0, "regular_score": 0.0}


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
        "sheet_like_group": 4,
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
                            "evidence": {"support_status": "structural_motif_support", "compressed_graph_added": True},
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
        "policy": "export structural motif nodes first and add only fitting support face_group nodes",
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
        for typ in ["sheet_like_group", "thin_wall_pair", "loop_or_hole", "transition_group", "repeated_feature", "boundary_group"]
        if int(node_type_counts.get(typ, 0)) > 0
    ]
    core_motif_types = [
        typ
        for typ in ["sheet_like_group", "thin_wall_pair", "loop_or_hole", "transition_group", "repeated_feature"]
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


def build_motif_graph(data: Dict[str, Any]) -> Dict[str, Any]:
    features = extract_motif_features(data)
    face_features = features["face_features"]
    global_scale = float(features.get("global_scale", 1.0))
    nodes: List[Dict[str, Any]] = []
    face_group_nodes: List[Dict[str, Any]] = []
    face_to_group: Dict[int, str] = {}

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

    base_groups = _build_base_face_groups(features)
    for group in base_groups:
        gf = _group_features(group, face_features, global_scale)
        node = add_node(
            "face_group",
            group,
            0.9 if len(group) > 1 else 0.78,
            gf,
            {"rule": "connected component of adjacent coplanar/smooth faces", "is_singleton": len(group) == 1},
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
                    {"rule": "large relative area and thin bbox scale", "relative_area_cut": sheet_area_cut},
                )
            )
        if boundary_ratio >= 0.5:
            boundary_candidates.append(
                (
                    0.58 + 0.35 * boundary_ratio,
                    group_node,
                    dict(gf, base_face_group_id=group_node["id"]),
                    {"rule": "faces touch global bbox boundary", "boundary_ratio": boundary_ratio},
                )
            )

    face_count_for_caps = max(int(features.get("face_count", 0)), 1)
    max_sheet_groups = max(2, min(5, int(0.10 * face_count_for_caps) + 1))
    max_boundary_groups = max(2, min(5, int(0.10 * face_count_for_caps) + 1))
    for confidence, group_node, node_features, evidence in sorted(sheet_candidates, key=lambda x: -x[0])[:max_sheet_groups]:
        add_node("sheet_like_group", group_node["face_ids"], confidence, node_features, evidence)
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
        if gap <= thin_gap_cut and area_ratio >= 0.62 and area_score >= 0.015:
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
    max_thin_pairs = max(2, min(7, len(face_group_nodes), int(0.12 * face_count_for_caps) + 1))
    for confidence, key, item in sorted(thin_candidates, key=lambda x: (-x[0], x[2]["normal_gap"])):
        if len(used_thin_pairs) >= max_thin_pairs:
            break
        if key in used_thin_pairs:
            continue
        if thin_group_degree.get(key[0], 0) >= 2 or thin_group_degree.get(key[1], 0) >= 2:
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
            {"rule": "sparse opposite_to + small spacing + similar area", "opposite_relation": item["opposite_relation"]},
        )

    loop_candidates: List[Tuple[float, Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = []
    for group_node in face_group_nodes:
        gf = group_node["features"]
        rel_area = float(gf.get("relative_area_sum", 0.0))
        boundary_ratio = float(gf.get("boundary_ratio", 0.0))
        outside_degree = len(gf.get("adjacent_faces_outside", []))
        loop_area_cut = min(0.08, max(0.025, small_area_cut * 1.25))
        if (
            boundary_ratio <= 0.05
            and rel_area <= loop_area_cut
            and outside_degree >= 3
            and int(gf.get("face_count", 1)) <= 8
        ):
            confidence = 0.42 + 0.24 * min(outside_degree / 5.0, 1.0) + 0.20 * (1.0 - min(rel_area / loop_area_cut, 1.0))
            loop_candidates.append(
                (
                    confidence,
                    group_node,
                    dict(gf, base_face_group_id=group_node["id"]),
                    {
                        "rule": "internal bounded local closure candidate",
                        "note": "candidate only; not a true engineering hole label",
                        "loop_area_cut": loop_area_cut,
                        "outside_degree": outside_degree,
                    },
                )
            )
    max_loop_groups = max(1, min(4, int(0.08 * face_count_for_caps) + 1))
    for confidence, group_node, node_features, evidence in sorted(loop_candidates, key=lambda x: -x[0])[:max_loop_groups]:
        add_node("loop_or_hole", group_node["face_ids"], confidence, node_features, evidence)

    group_id_to_node = {node["id"]: node for node in face_group_nodes}
    face_to_group_index = {fid: gid for fid, gid in face_to_group.items()}
    orthogonal_pairs = {_relation_pair_key(rel) for rel in face_relations if rel.get("type") == "orthogonal_to"}
    transition_candidates: List[Tuple[float, Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = []
    for group_node in face_group_nodes:
        gf = group_node["features"]
        rel_area = float(gf.get("relative_area_sum", 0.0))
        high_aspect = float(gf.get("max_aspect_ratio", 1.0)) >= 3.0
        curvature_flag = float(gf.get("curvature_proxy_mean", 0.0)) >= 0.5
        neighbor_groups = sorted({face_to_group_index.get(fid) for fid in gf.get("adjacent_faces_outside", []) if face_to_group_index.get(fid)})
        neighbor_groups = [gid for gid in neighbor_groups if gid != group_node["id"]]
        larger_neighbors = [
            gid
            for gid in neighbor_groups
            if float(group_id_to_node[gid]["features"].get("relative_area_sum", 0.0)) > rel_area * 1.4
        ]
        ortho_support = 0
        for fid in group_node["face_ids"]:
            for adj in gf.get("adjacent_faces_outside", []):
                if tuple(sorted([int(fid), int(adj)])) in orthogonal_pairs:
                    ortho_support += 1
        transition_area_cut = min(0.10, max(0.03, small_area_cut * 1.8))
        neighbor_support = len(set(larger_neighbors))
        connector_support = neighbor_support >= 2 or (
            neighbor_support >= 1 and (curvature_flag or (high_aspect and ortho_support > 0))
        )
        if rel_area <= transition_area_cut and (high_aspect or curvature_flag or ortho_support > 0) and connector_support:
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
                        "rule": "small/high-aspect or curved connector candidate",
                        "note": "geometric-topological transition candidate, not guaranteed fillet semantics",
                        "orthogonal_support": ortho_support,
                        "transition_area_cut": transition_area_cut,
                    },
                )
            )
    max_transition_groups = max(2, min(5, len(face_group_nodes), int(0.10 * face_count_for_caps) + 1))
    used_transition_faces: Set[int] = set()
    for confidence, group_node, node_features, evidence in sorted(transition_candidates, key=lambda x: -x[0]):
        if _node_type_count(nodes, "transition_group") >= max_transition_groups:
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
    repeat_uf = UnionFind(len(repeat_candidates))
    repeat_pair_evidence: Dict[Tuple[int, int], Dict[str, float]] = {}
    for a_idx, b_idx in combinations(range(len(repeat_candidates)), 2):
        similar, evidence = _similar_group_signature(repeat_candidates[a_idx], repeat_candidates[b_idx])
        if similar:
            repeat_uf.union(a_idx, b_idx)
            repeat_pair_evidence[(a_idx, b_idx)] = evidence
    repeat_clusters = sorted((repeat_uf.groups() if repeat_candidates else []), key=len, reverse=True)
    max_repeated_features = max(1, min(3, int(0.06 * face_count_for_caps) + 1))
    repeated_added = 0
    for cluster in repeat_clusters:
        if repeated_added >= max_repeated_features:
            break
        if len(cluster) < 2:
            continue
        members = [repeat_candidates[idx] for idx in cluster]
        boundary_mean = float(np.mean([float(m["features"].get("boundary_ratio", 0.0)) for m in members]))
        rel_area_mean = float(np.mean([float(m["features"].get("relative_area_sum", 0.0)) for m in members]))
        if len(members) < 3 and rel_area_mean > 0.12:
            continue
        centroids = np.asarray([m["features"]["centroid"] for m in members], dtype=np.float32)
        spacing = _spacing_regular_score(centroids)
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
                "rule": "similar bbox dimensions + normal + adjacency signature + spacing pattern",
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
                    {"overlap_faces": sorted(set_a), "support_status": "same_face_support"},
                )
            elif node_b["type"] == "face_group" and node_a["type"] != "face_group":
                _add_relation(
                    relation_records,
                    node_b["id"],
                    node_a["id"],
                    "embedded_in",
                    0.86,
                    {"overlap_faces": sorted(set_b), "support_status": "same_face_support"},
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
                    {"boundary_group": boundary["id"], "rule": "motif adjacent to global boundary group"},
                )

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
    face_relation_stats = features.get("face_relation_stats", {})
    node_type_counts = {typ: _node_type_count(nodes, typ) for typ in NODE_TYPES}
    relation_type_counts = {typ: sum(1 for rel in motif_relations if rel["type"] == typ) for typ in RELATION_TYPES}
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
            "face_to_motif_nodes": face_to_motif_nodes,
            "base_face_group_node_ids": [node["id"] for node in face_group_nodes],
        },
        "motif_selection_policy": {
            "training_graph_policy": "Keep only generation-relevant weak structural priors. Dense face-level relation evidence is used for scoring/audit but is not exported as training edges.",
            "structural_node_budget": "exported motif node count is compressed to be no larger than the original face count when possible",
            "raw_node_count_before_compression": raw_node_count,
            "raw_relation_count_before_compression": raw_relation_count,
            "kept_node_families": [
                "base face groups",
                "dominant sheet-like anchors",
                "high-confidence thin-wall pairs",
                "internal loop/closure candidates",
                "small transition connector candidates",
                "regular repeated-feature clusters",
                "global boundary groups",
            ],
            "kept_relation_families": [
                "topological adjacency between face groups",
                "sparse parallel/opposite/orthogonal/coplanar supports",
                "motif-to-face-group embedded_in links",
                "chain-like repeated_with links",
                "bounded_by links to boundary groups",
            ],
            "not_kept_as_training_edges": [
                "all pairwise parallel face pairs",
                "all pairwise orthogonal face pairs",
                "low-confidence long-range relation pairs",
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
    """Remove stale motif JSON/index files before writing a fresh extraction."""
    root = Path(motif_graph_dir)
    removed = 0
    for pattern in ["*_motif_graph.json", "motif_graph_index.jsonl", "motif_graph_index_ready.jsonl"]:
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

    for idx, row in enumerate(rows, start=1):
        uid = str(row.get("uid", ""))
        pkl_path = row.get("pkl_path") or os.path.join(parsed_dir, f"{uid}.pkl")
        try:
            data = read_pickle(pkl_path)
            graph = build_motif_graph(data)
            graph_path = os.path.join(dirs["motif_graphs"], f"{uid}_motif_graph.json")
            write_json(graph_path, graph)
            index_records.append(graph)
            records.append(
                {
                    "uid": uid,
                    "status": "SUCCESS",
                    "motif_graph_path": graph_path,
                    "num_nodes": len(graph.get("motif_nodes", [])),
                    "num_relations": len(graph.get("motif_relations", [])),
                    "num_faces": graph.get("num_faces", 0),
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
                    "num_nodes": 0,
                    "num_relations": 0,
                    "num_faces": row.get("face_count", 0),
                    "error": str(exc),
                }
            )
        if idx % 250 == 0:
            print(f"[extract_motif] processed {idx}/{len(rows)}; success={len(index_records)} failed={len(failures)}")

    index_path = os.path.join(dirs["motif_graphs"], "motif_graph_index.jsonl")
    ready_index_path = os.path.join(dirs["motif_graphs"], "motif_graph_index_ready.jsonl")
    ready_index_records = [
        graph
        for graph in index_records
        if bool(graph.get("motif_quality", {}).get("motif_ready", False))
    ]
    write_jsonl(index_path, index_records)
    write_jsonl(ready_index_path, ready_index_records)
    write_json(
        os.path.join(dirs["reports"], "motif_extraction_summary.json"),
        {
            "input_clean_samples": len(rows),
            "motif_graph_success_count": len(index_records),
            "motif_ready_count": len(ready_index_records),
            "motif_graph_failure_count": len(failures),
            "motif_graph_index": index_path,
            "motif_graph_index_ready": ready_index_path,
            **cleanup_info,
        },
    )
    return {
        "records": records,
        "failures": failures,
        "graphs": index_records,
        "ready_graphs": ready_index_records,
        "motif_graph_index": index_path,
        "motif_graph_index_ready": ready_index_path,
        **cleanup_info,
    }
