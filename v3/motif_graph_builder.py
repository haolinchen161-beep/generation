# -*- coding: utf-8 -*-
"""Build a conservative, inspectable B-Rep motif graph M=(Vm, Em, Pm).

v3.2 is intentionally sparse.  It avoids relation flooding by:
  1) grouping faces into a small set of candidate motif nodes;
  2) emitting only relations with explicit local/topological evidence;
  3) applying per-relation caps;
  4) writing rejected candidates for audit.

The output is a weak, algorithm-extracted structural prior, not manual
semantic truth.
"""

from __future__ import annotations

import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from motif_feature_extractor import extract_motif_features
from utils_io import clean_dir, ensure_workdir, timestamp, write_csv, write_jsonl, write_text

NODE_TYPES = {
    "face_group",
    "sheet_like_group",
    "thin_wall_pair",
    "loop_or_hole",
    "transition_group",
    "repeated_feature",
    "boundary_group",
}

RELATION_TYPES = {
    "adjacent_to",
    "parallel_to",
    "opposite_to",
    "orthogonal_to",
    "coplanar_with",
    "smooth_connected",
    "embedded_in",
    "repeated_with",
    "bounded_by",
}

RELATION_CAPS = {
    "embedded_in": 6,
    "bounded_by": 6,
    "opposite_to": 8,
    "parallel_to": 8,
    "coplanar_with": 4,
    "orthogonal_to": 10,
    "smooth_connected": 5,
    "repeated_with": 5,
    "adjacent_to": 6,
}

RELATION_PRIORITY = {
    "embedded_in": 0,
    "bounded_by": 1,
    "opposite_to": 2,
    "parallel_to": 3,
    "coplanar_with": 4,
    "orthogonal_to": 5,
    "smooth_connected": 6,
    "repeated_with": 7,
    "adjacent_to": 8,
}


def _unit(v: Sequence[float]) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(arr))
    return arr / n if n > 1e-8 else np.asarray([1.0, 0.0, 0.0], dtype=float)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.clip(np.dot(_unit(a), _unit(b)), -1.0, 1.0))


def _bbox(face_ids: Sequence[int], faces: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    boxes = [np.asarray(faces[int(fid)]["bbox"], dtype=float) for fid in face_ids if 0 <= int(fid) < len(faces)]
    if not boxes:
        return np.zeros(3), np.zeros(3)
    arr = np.asarray(boxes, dtype=float)
    return np.min(arr[:, :3], axis=0), np.max(arr[:, 3:], axis=0)


def _center(face_ids: Sequence[int], faces: Sequence[Dict[str, Any]]) -> np.ndarray:
    pts = [np.asarray(faces[int(fid)]["centroid"], dtype=float) for fid in face_ids if 0 <= int(fid) < len(faces)]
    return np.mean(np.asarray(pts, dtype=float), axis=0) if pts else np.zeros(3)


def _normal(face_ids: Sequence[int], faces: Sequence[Dict[str, Any]]) -> np.ndarray:
    normals = [_unit(faces[int(fid)]["normal"]) for fid in face_ids if 0 <= int(fid) < len(faces)]
    if not normals:
        return np.asarray([1.0, 0.0, 0.0], dtype=float)
    base = normals[0]
    aligned = [n if np.dot(n, base) >= 0 else -n for n in normals]
    return _unit(np.mean(np.asarray(aligned), axis=0))


def _area(face_ids: Sequence[int], faces: Sequence[Dict[str, Any]]) -> float:
    return float(sum(float(faces[int(fid)]["area_proxy"]) for fid in face_ids if 0 <= int(fid) < len(faces)))


def _adjacent_count(ids_a: Sequence[int], ids_b: Sequence[int], face_adj: np.ndarray) -> int:
    count = 0
    for a in ids_a:
        if int(a) >= face_adj.shape[0]:
            continue
        for b in ids_b:
            if int(b) < face_adj.shape[1] and face_adj[int(a), int(b)] > 0:
                count += 1
    return count


def _components(face_ids: Sequence[int], face_adj: np.ndarray, faces: Sequence[Dict[str, Any]] | None = None, normal_dot_min: float = 0.92) -> List[List[int]]:
    id_set = {int(x) for x in face_ids}
    seen = set()
    comps: List[List[int]] = []
    face_map = {int(f["face_id"]): f for f in faces or []}
    for seed in sorted(id_set):
        if seed in seen:
            continue
        stack = [seed]
        seen.add(seed)
        comp: List[int] = []
        seed_normal = _unit(face_map.get(seed, {}).get("normal", [1, 0, 0])) if face_map else None
        while stack:
            cur = stack.pop()
            comp.append(cur)
            if cur >= face_adj.shape[0]:
                continue
            for nb in np.where(face_adj[cur] > 0)[0].tolist():
                if nb not in id_set or nb in seen:
                    continue
                if face_map and seed_normal is not None:
                    nb_normal = _unit(face_map.get(int(nb), {}).get("normal", [1, 0, 0]))
                    if abs(float(np.dot(seed_normal, nb_normal))) < normal_dot_min:
                        continue
                seen.add(int(nb))
                stack.append(int(nb))
        comps.append(sorted(comp))
    return comps


def _projection_overlap(mn_a: np.ndarray, mx_a: np.ndarray, mn_b: np.ndarray, mx_b: np.ndarray, axes: Sequence[int]) -> float:
    ratios = []
    for axis in axes:
        inter = max(0.0, min(mx_a[axis], mx_b[axis]) - max(mn_a[axis], mn_b[axis]))
        union = max(mx_a[axis], mx_b[axis]) - min(mn_a[axis], mn_b[axis])
        ratios.append(float(inter / max(union, 1e-8)))
    return float(np.prod(ratios)) if ratios else 0.0


def _bbox_gap(mn_a: np.ndarray, mx_a: np.ndarray, mn_b: np.ndarray, mx_b: np.ndarray) -> float:
    gap = np.maximum(0.0, np.maximum(mn_a - mx_b, mn_b - mx_a))
    return float(np.linalg.norm(gap))


def _make_node(node_id: str, node_type: str, face_ids: Sequence[int], faces: Sequence[Dict[str, Any]], extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    ids = sorted(int(x) for x in set(face_ids))
    mn, mx = _bbox(ids, faces)
    props = {
        "area_proxy": round(_area(ids, faces), 6),
        "normal": [round(float(x), 6) for x in _normal(ids, faces).tolist()],
        "centroid": [round(float(x), 6) for x in _center(ids, faces).tolist()],
        "bbox": [round(float(x), 6) for x in np.concatenate([mn, mx]).tolist()],
    }
    if extra:
        props.update(extra)
    return {"id": node_id, "type": node_type, "face_ids": ids, "properties": props}


def _make_rel(source: str, target: str, typ: str, score: float, evidence: Dict[str, Any] | None = None) -> Dict[str, Any]:
    rel = {"source": source, "target": target, "type": typ, "score": round(float(score), 6)}
    if evidence:
        rel["evidence"] = evidence
    return rel


def _select_relations(candidates: List[Dict[str, Any]], node_count: int) -> List[Dict[str, Any]]:
    unique: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    undirected = {"parallel_to", "opposite_to", "orthogonal_to", "coplanar_with", "adjacent_to"}
    for rel in candidates:
        src, dst, typ = str(rel["source"]), str(rel["target"]), str(rel["type"])
        if src == dst or typ not in RELATION_TYPES:
            continue
        if typ in undirected and dst < src:
            src, dst = dst, src
            rel = dict(rel)
            rel["source"], rel["target"] = src, dst
        key = (src, dst, typ)
        if key not in unique or float(rel.get("score", 0.0)) > float(unique[key].get("score", 0.0)):
            unique[key] = rel

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rel in unique.values():
        grouped[str(rel["type"])].append(rel)

    kept: List[Dict[str, Any]] = []
    per_node_degree: Dict[str, int] = defaultdict(int)
    max_total = max(6, min(24, int(0.55 * max(node_count, 1)) + 6))

    for typ in sorted(grouped, key=lambda t: RELATION_PRIORITY.get(t, 99)):
        items = sorted(grouped[typ], key=lambda r: -float(r.get("score", 0.0)))
        cap = min(RELATION_CAPS.get(typ, 4), max(2, node_count // 3 + 2))
        used_for_type = 0
        for rel in items:
            src, dst = str(rel["source"]), str(rel["target"])
            if per_node_degree[src] >= 4 or per_node_degree[dst] >= 4:
                continue
            kept.append(rel)
            per_node_degree[src] += 1
            per_node_degree[dst] += 1
            used_for_type += 1
            if used_for_type >= cap or len(kept) >= max_total:
                break
        if len(kept) >= max_total:
            break
    return sorted(kept, key=lambda r: (str(r["source"]), str(r["target"]), str(r["type"])))


def _node_lookup(nodes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(n["id"]): n for n in nodes}


def _similar_nodes_for_repetition(nodes: List[Dict[str, Any]], global_scale: float) -> List[List[str]]:
    buckets: Dict[Tuple[str, int, int], List[Dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        if node["type"] not in {"sheet_like_group", "transition_group", "boundary_group"}:
            continue
        box = np.asarray(node["properties"].get("bbox", [0, 0, 0, 0, 0, 0]), dtype=float)
        dims = np.maximum(box[3:] - box[:3], 0.0)
        dims_sorted = np.sort(dims / max(global_scale, 1e-8))
        normal = np.asarray(node["properties"].get("normal", [1, 0, 0]), dtype=float)
        axis = int(np.argmax(np.abs(normal)))
        size_bin = int(round(float(np.sum(np.round(dims_sorted, 2))) * 10))
        buckets[(node["type"], axis, size_bin)].append(node)

    clusters: List[List[str]] = []
    for items in buckets.values():
        if len(items) < 4:
            continue
        centers = np.asarray([item["properties"]["centroid"] for item in items], dtype=float)
        spread_axis = int(np.argmax(np.ptp(centers, axis=0)))
        order = np.argsort(centers[:, spread_axis])
        sorted_items = [items[int(i)] for i in order]
        coords = centers[order, spread_axis]
        diffs = np.diff(coords)
        if len(diffs) < 3:
            continue
        cv = float(np.std(diffs) / max(abs(float(np.mean(diffs))), 1e-8))
        if cv <= 0.40 and float(np.ptp(coords)) >= 0.15 * global_scale:
            clusters.append([str(item["id"]) for item in sorted_items[:6]])
    return clusters[:2]


def build_motif_graph_for_sample(data: Dict[str, Any]) -> Dict[str, Any]:
    features = extract_motif_features(data)
    faces = features["face_features"]
    face_adj = np.asarray(features["face_adjacency"], dtype=int)
    global_scale = float(features.get("global_scale", 1.0))
    areas = np.asarray([float(f["area_proxy"]) for f in faces], dtype=float) if faces else np.zeros(0)
    p10 = float(np.percentile(areas, 10)) if areas.size else 0.0
    p25 = float(np.percentile(areas, 25)) if areas.size else 0.0
    p70 = float(np.percentile(areas, 70)) if areas.size else 0.0

    used: set[int] = set()
    nodes: List[Dict[str, Any]] = []
    rejected_candidates: List[Dict[str, Any]] = []
    gmin = np.asarray(features["global_min"], dtype=float)
    gmax = np.asarray(features["global_max"], dtype=float)
    tol = max(0.025 * global_scale, 1e-6)

    loop_ids: List[int] = []
    for f in faces:
        fid = int(f["face_id"])
        box = np.asarray(f["bbox"], dtype=float)
        mn, mx = box[:3], box[3:]
        internal_axes = sum(1 for axis in range(3) if mn[axis] > gmin[axis] + tol and mx[axis] < gmax[axis] - tol)
        dims = np.maximum(mx - mn, 0.0)
        compact_extent = float(np.max(dims)) <= 0.35 * global_scale
        compact_area = float(f["area_proxy"]) <= max(p25, 1e-8)
        high_degree = int(f["degree"]) >= 4 or int(f.get("adjacent_degree", 0)) >= 3
        if (not bool(f["is_global_boundary"])) and internal_axes >= 2 and compact_extent and compact_area and high_degree:
            loop_ids.append(fid)

    for comp in _components(loop_ids, face_adj):
        if 1 <= len(comp) <= 6:
            nodes.append(_make_node(f"loop_{len(nodes)}", "loop_or_hole", comp, faces, {"detector": "strict_internal_compact_high_degree"}))
            used.update(comp)

    transition_ids: List[int] = []
    for f in faces:
        fid = int(f["face_id"])
        if fid in used:
            continue
        area = float(f["area_proxy"])
        aspect = float(f["aspect_ratio"])
        degree = int(f.get("adjacent_degree", f.get("degree", 0)))
        curved = float(f.get("curvature_proxy", 0.0)) > 0.5
        if (curved and area <= max(p25, 1e-8)) or ((not bool(f["is_global_boundary"])) and degree >= 2 and area <= max(p10, 1e-8) and aspect >= 12.0):
            transition_ids.append(fid)

    for comp in _components(transition_ids, face_adj):
        if 1 <= len(comp) <= 4:
            nodes.append(_make_node(f"transition_{len(nodes)}", "transition_group", comp, faces, {"detector": "small_curved_or_narrow_bridge"}))
            used.update(comp)

    sheet_ids = [int(f["face_id"]) for f in faces if int(f["face_id"]) not in used and float(f["area_proxy"]) >= max(p70, 1e-8)]
    for comp in _components(sheet_ids, face_adj, faces, normal_dot_min=0.94):
        if comp:
            nodes.append(_make_node(f"sheet_{len(nodes)}", "sheet_like_group", comp, faces, {"detector": "large_area_normal_component"}))
            used.update(comp)

    boundary_ids = [int(f["face_id"]) for f in faces if int(f["face_id"]) not in used and bool(f["is_global_boundary"])]
    for comp in _components(boundary_ids, face_adj):
        if comp:
            nodes.append(_make_node(f"boundary_{len(nodes)}", "boundary_group", comp, faces, {"detector": "global_bbox_boundary"}))
            used.update(comp)

    remaining_ids = [int(f["face_id"]) for f in faces if int(f["face_id"]) not in used]
    for comp in _components(remaining_ids, face_adj):
        if comp:
            nodes.append(_make_node(f"group_{len(nodes)}", "face_group", comp, faces, {"detector": "remaining_connected_component"}))

    candidates: List[Dict[str, Any]] = []

    def add_candidate(src: str, dst: str, typ: str, score: float, evidence: Dict[str, Any]) -> None:
        candidates.append(_make_rel(src, dst, typ, score, evidence))

    for i, a in enumerate(nodes):
        a_ids = a["face_ids"]
        a_mn, a_mx = _bbox(a_ids, faces)
        a_c = _center(a_ids, faces)
        a_n = _normal(a_ids, faces)
        a_area = _area(a_ids, faces)
        for b in nodes[i + 1 :]:
            b_ids = b["face_ids"]
            b_mn, b_mx = _bbox(b_ids, faces)
            b_c = _center(b_ids, faces)
            b_n = _normal(b_ids, faces)
            b_area = _area(b_ids, faces)
            adj = _adjacent_count(a_ids, b_ids, face_adj)
            gap = _bbox_gap(a_mn, a_mx, b_mn, b_mx)
            abs_dot = abs(_dot(a_n, b_n))
            area_ratio = min(a_area, b_area) / max(max(a_area, b_area), 1e-8)
            dominant = int(np.argmax(np.abs(a_n)))
            overlap_axes = [axis for axis in range(3) if axis != dominant]
            overlap = _projection_overlap(a_mn, a_mx, b_mn, b_mx, overlap_axes)
            plane_dist = abs(float(np.dot(b_c - a_c, a_n)))
            structural_touch = adj > 0 or gap <= 0.015 * global_scale

            if adj >= 2 and (a["type"] in {"transition_group", "loop_or_hole"} or b["type"] in {"transition_group", "loop_or_hole"}):
                add_candidate(a["id"], b["id"], "adjacent_to", min(1.0, adj / 3.0), {"adjacent_face_pairs": adj})

            if abs_dot >= 0.985 and area_ratio >= 0.35 and overlap >= 0.65:
                if plane_dist <= 0.010 * global_scale:
                    add_candidate(a["id"], b["id"], "coplanar_with", 1.0 - plane_dist / max(0.010 * global_scale, 1e-8), {"plane_dist": round(plane_dist, 6), "overlap": round(overlap, 6), "area_ratio": round(area_ratio, 6)})
                elif structural_touch or plane_dist <= 0.18 * global_scale:
                    typ = "opposite_to" if plane_dist > 0.025 * global_scale else "parallel_to"
                    add_candidate(a["id"], b["id"], typ, min(1.0, overlap * abs_dot), {"plane_dist": round(plane_dist, 6), "overlap": round(overlap, 6), "area_ratio": round(area_ratio, 6), "gap": round(gap, 6), "adjacent_face_pairs": adj})
                else:
                    rejected_candidates.append({"source": a["id"], "target": b["id"], "type": "parallel_or_opposite", "reason": "not_near_no_structural_support", "overlap": round(overlap, 6), "plane_dist": round(plane_dist, 6)})
            elif abs_dot <= 0.08 and adj >= 1 and area_ratio >= 0.25:
                add_candidate(a["id"], b["id"], "orthogonal_to", 1.0 - abs_dot, {"adjacent_face_pairs": adj, "area_ratio": round(area_ratio, 6), "gap": round(gap, 6)})

    preliminary = _select_relations(candidates, len(nodes))
    lookup = {str(n["id"]): n for n in nodes}
    thin_nodes: List[Dict[str, Any]] = []
    for rel in preliminary:
        if rel["type"] != "opposite_to" or float(rel.get("score", 0.0)) < 0.70:
            continue
        src = lookup.get(str(rel["source"]))
        dst = lookup.get(str(rel["target"]))
        if not src or not dst:
            continue
        c0 = _center(src["face_ids"], faces)
        c1 = _center(dst["face_ids"], faces)
        n0 = _normal(src["face_ids"], faces)
        distance = abs(float(np.dot(c1 - c0, n0)))
        if 1e-6 < distance <= 0.08 * global_scale:
            node = _make_node(
                f"thinwall_{len(nodes) + len(thin_nodes)}",
                "thin_wall_pair",
                sorted(set(src["face_ids"]) | set(dst["face_ids"])),
                faces,
                {"distance": round(distance, 6), "source_nodes": [src["id"], dst["id"]]},
            )
            thin_nodes.append(node)
    nodes.extend(thin_nodes)

    for node in thin_nodes:
        for src_id in node.get("properties", {}).get("source_nodes", []):
            candidates.append(_make_rel(node["id"], src_id, "bounded_by", 1.0, {"thin_wall_source": True}))

    for node in nodes:
        if node["type"] != "loop_or_hole":
            continue
        loop_c = _center(node["face_ids"], faces)
        best = None
        best_area = 0.0
        for other in nodes:
            if other["id"] == node["id"] or other["type"] not in {"sheet_like_group", "face_group", "boundary_group"}:
                continue
            mn, mx = _bbox(other["face_ids"], faces)
            margin = 0.02 * global_scale
            inside = bool(np.all(loop_c >= mn - margin) and np.all(loop_c <= mx + margin))
            area = _area(other["face_ids"], faces)
            if inside and area > best_area:
                best = other
                best_area = area
        if best is not None:
            candidates.append(_make_rel(node["id"], best["id"], "embedded_in", 1.0, {"container_area": round(best_area, 6)}))

    adjacent_candidates = [c for c in candidates if c["type"] == "adjacent_to"]
    for node in nodes:
        if node["type"] != "transition_group":
            continue
        neigh = [rel for rel in adjacent_candidates if rel["source"] == node["id"] or rel["target"] == node["id"]]
        neigh.sort(key=lambda r: -float(r.get("score", 0.0)))
        if neigh:
            rel = neigh[0]
            other = rel["target"] if rel["source"] == node["id"] else rel["source"]
            candidates.append(_make_rel(node["id"], other, "smooth_connected", rel.get("score", 0.67), {"from_adjacent_to": True}))

    for cluster in _similar_nodes_for_repetition(nodes, global_scale):
        lookup = _node_lookup(nodes)
        face_ids: List[int] = []
        for node_id in cluster:
            face_ids.extend(lookup[node_id]["face_ids"])
        rep_node = _make_node(f"repeated_{len(nodes)}", "repeated_feature", face_ids, faces, {"member_nodes": cluster, "detector": "regular_similar_size_normal_spacing"})
        nodes.append(rep_node)
        for node_id in cluster[:3]:
            candidates.append(_make_rel(rep_node["id"], node_id, "repeated_with", 1.0, {"cluster_size": len(cluster)}))

    relations = _select_relations(candidates, len(nodes))
    rel_counts = Counter(str(r["type"]) for r in relations)
    node_counts = Counter(str(n["type"]) for n in nodes)
    summary = {
        "has_parallel": rel_counts["parallel_to"] > 0,
        "has_thin_wall": node_counts["thin_wall_pair"] > 0,
        "has_loop_or_hole": node_counts["loop_or_hole"] > 0,
        "has_transition": node_counts["transition_group"] > 0,
        "has_repeated_feature": node_counts["repeated_feature"] > 0,
        "motif_rich": len(nodes) >= 4 and (rel_counts["opposite_to"] + rel_counts["embedded_in"] + rel_counts["bounded_by"] + rel_counts["repeated_with"] >= 1 or rel_counts["orthogonal_to"] >= 2),
        "raw_relation_candidates": len(candidates),
        "kept_relation_count": len(relations),
        "rejected_candidate_count": len(rejected_candidates),
    }
    return {
        "uid": features["uid"],
        "source": features["source"],
        "num_faces": int(features["face_count"]),
        "num_edges": int(features["edge_count"]),
        "num_vertices": int(features["vertex_count"]),
        "motif_nodes": nodes,
        "motif_relations": relations,
        "motif_summary": summary,
        "rejected_relation_candidates": rejected_candidates[:80],
        "parser_backend": features.get("parser_backend", "unknown"),
        "geometry_sampling_quality": features.get("geometry_sampling_quality", "unknown"),
        "label_source": "algorithm_extracted_motif_sparse_v3_2",
        "is_manual_ground_truth": False,
    }


def build_motif_graphs(workdir: str) -> Dict[str, Any]:
    dirs = ensure_workdir(workdir)
    parsed_dir = dirs["parsed_public"]
    motif_dir = dirs["motif_graphs"]
    reports_dir = dirs["reports"]
    clean_dir(motif_dir, ["*.jsonl", "*.csv"])

    graphs: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for pkl_path in sorted(Path(parsed_dir).glob("*.pkl")):
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            graphs.append(build_motif_graph_for_sample(data))
        except Exception as exc:
            failures.append({"uid": pkl_path.stem, "error": str(exc)})

    write_jsonl(os.path.join(motif_dir, "motif_graphs.jsonl"), graphs)

    node_rows: List[Dict[str, Any]] = []
    rel_rows: List[Dict[str, Any]] = []
    reject_rows: List[Dict[str, Any]] = []
    for graph in graphs:
        node_counter = Counter(node["type"] for node in graph["motif_nodes"])
        rel_counter = Counter(rel["type"] for rel in graph["motif_relations"])
        row = {"uid": graph["uid"], "num_nodes": len(graph["motif_nodes"])}
        row.update({f"node_{key}": node_counter.get(key, 0) for key in sorted(NODE_TYPES)})
        node_rows.append(row)
        rrow = {"uid": graph["uid"], "num_relations": len(graph["motif_relations"])}
        rrow.update({f"rel_{key}": rel_counter.get(key, 0) for key in sorted(RELATION_TYPES)})
        rel_rows.append(rrow)
        for item in graph.get("rejected_relation_candidates", []):
            reject_rows.append({"uid": graph["uid"], **item})

    write_csv(os.path.join(motif_dir, "motif_node_stats.csv"), node_rows, ["uid", "num_nodes"] + [f"node_{key}" for key in sorted(NODE_TYPES)])
    write_csv(os.path.join(motif_dir, "motif_relation_stats.csv"), rel_rows, ["uid", "num_relations"] + [f"rel_{key}" for key in sorted(RELATION_TYPES)])
    write_csv(os.path.join(reports_dir, "motif_rejected_relation_candidates.csv"), reject_rows, ["uid", "source", "target", "type", "reason", "overlap", "plane_dist"])

    rich_count = sum(1 for graph in graphs if graph["motif_summary"].get("motif_rich"))
    avg_nodes = float(np.mean([len(graph["motif_nodes"]) for graph in graphs])) if graphs else 0.0
    avg_relations = float(np.mean([len(graph["motif_relations"]) for graph in graphs])) if graphs else 0.0
    raw_candidates = sum(int(graph.get("motif_summary", {}).get("raw_relation_candidates", 0)) for graph in graphs)
    kept_relations = sum(int(graph.get("motif_summary", {}).get("kept_relation_count", 0)) for graph in graphs)
    report = [
        "Innovation1 v3.2 Sparse Motif Extraction Report",
        "=" * 72,
        f"Time: {timestamp()}",
        f"Parsed samples: {len(list(Path(parsed_dir).glob('*.pkl')))}",
        f"Motif graph success: {len(graphs)}",
        f"Motif graph failures: {len(failures)}",
        f"Motif-rich samples: {rich_count}",
        f"Average motif nodes: {avg_nodes:.3f}",
        f"Average kept motif relations: {avg_relations:.3f}",
        f"Raw relation candidates: {raw_candidates}",
        f"Kept relation candidates: {kept_relations}",
        "",
        "Interpretation:",
        "  v3.2 intentionally keeps a sparse motif graph.",
        "  Orthogonal relations require real face adjacency.",
        "  Parallel/opposite relations require high overlap and area-ratio evidence.",
        "  Adjacency is only emitted around local features and is capped.",
        "  Motif nodes and relations are algorithm-extracted weak structural priors, not human labels.",
    ]
    if failures:
        report.extend(["", "Failures:"])
        for item in failures[:100]:
            report.append(f"  - {item['uid']}: {item['error']}")
    write_text(os.path.join(reports_dir, "motif_extraction_report.txt"), report)
    return {"graphs": graphs, "failures": failures}
