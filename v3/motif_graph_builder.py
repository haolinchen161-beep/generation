# -*- coding: utf-8 -*-
"""Build algorithm-extracted B-Rep motif graph M=(Vm, Em, Pm)."""

from __future__ import annotations

import os
import pickle
from collections import Counter
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


def _unit(v: Sequence[float]) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(arr))
    return arr / n if n > 1e-8 else np.asarray([1.0, 0.0, 0.0], dtype=float)


def _angle_dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.clip(np.dot(_unit(a), _unit(b)), -1.0, 1.0))


def _group_bbox(face_ids: Sequence[int], face_features: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    boxes = [np.asarray(face_features[int(fid)]["bbox"], dtype=float) for fid in face_ids if int(fid) < len(face_features)]
    if not boxes:
        return np.zeros(3), np.zeros(3)
    arr = np.asarray(boxes, dtype=float)
    return np.min(arr[:, :3], axis=0), np.max(arr[:, 3:], axis=0)


def _group_centroid(face_ids: Sequence[int], face_features: Sequence[Dict[str, Any]]) -> np.ndarray:
    pts = [np.asarray(face_features[int(fid)]["centroid"], dtype=float) for fid in face_ids if int(fid) < len(face_features)]
    return np.mean(np.asarray(pts, dtype=float), axis=0) if pts else np.zeros(3)


def _group_normal(face_ids: Sequence[int], face_features: Sequence[Dict[str, Any]]) -> np.ndarray:
    normals = [_unit(face_features[int(fid)]["normal"]) for fid in face_ids if int(fid) < len(face_features)]
    if not normals:
        return np.asarray([1.0, 0.0, 0.0], dtype=float)
    n = np.mean(np.asarray(normals), axis=0)
    return _unit(n)


def _group_area(face_ids: Sequence[int], face_features: Sequence[Dict[str, Any]]) -> float:
    return float(sum(float(face_features[int(fid)]["area_proxy"]) for fid in face_ids if int(fid) < len(face_features)))


def _adjacent_faces(face_ids_a: Sequence[int], face_ids_b: Sequence[int], face_adj: np.ndarray) -> int:
    count = 0
    for a in face_ids_a:
        if int(a) >= face_adj.shape[0]:
            continue
        for b in face_ids_b:
            if int(b) < face_adj.shape[1] and face_adj[int(a), int(b)] > 0:
                count += 1
    return count


def _connected_components(face_ids: Sequence[int], face_adj: np.ndarray) -> List[List[int]]:
    id_set = {int(x) for x in face_ids}
    seen = set()
    comps: List[List[int]] = []
    for seed in sorted(id_set):
        if seed in seen:
            continue
        stack = [seed]
        seen.add(seed)
        comp: List[int] = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            if cur >= face_adj.shape[0]:
                continue
            for nb in np.where(face_adj[cur] > 0)[0].tolist():
                if nb in id_set and nb not in seen:
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


def _make_node(node_id: str, node_type: str, face_ids: Sequence[int], face_features: Sequence[Dict[str, Any]], extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    mn, mx = _group_bbox(face_ids, face_features)
    props = {
        "area_proxy": round(_group_area(face_ids, face_features), 6),
        "normal": [round(float(x), 6) for x in _group_normal(face_ids, face_features).tolist()],
        "centroid": [round(float(x), 6) for x in _group_centroid(face_ids, face_features).tolist()],
        "bbox": [round(float(x), 6) for x in np.concatenate([mn, mx]).tolist()],
    }
    if extra:
        props.update(extra)
    return {"id": node_id, "type": node_type, "face_ids": sorted(int(x) for x in set(face_ids)), "properties": props}


def _relation(source: str, target: str, rel_type: str, score: float) -> Dict[str, Any]:
    return {"source": source, "target": target, "type": rel_type, "score": round(float(score), 6)}


def _find_repeated_groups(nodes: List[Dict[str, Any]], face_features: Sequence[Dict[str, Any]], global_scale: float) -> List[List[str]]:
    candidates = [n for n in nodes if n["type"] in {"sheet_like_group", "face_group", "transition_group"} and n.get("face_ids")]
    clusters: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = {}
    for node in candidates:
        props = node["properties"]
        normal = np.asarray(props.get("normal", [1, 0, 0]), dtype=float)
        axis = int(np.argmax(np.abs(normal)))
        area_bin = int(round(float(props.get("area_proxy", 0.0)) / max(global_scale * global_scale * 0.03, 1e-8)))
        degree_bin = len(node.get("face_ids", []))
        clusters.setdefault((axis, area_bin, degree_bin), []).append(node)
    repeated: List[List[str]] = []
    for items in clusters.values():
        if len(items) < 3:
            continue
        centers = np.asarray([item["properties"]["centroid"] for item in items], dtype=float)
        spans = np.ptp(centers, axis=0)
        if float(np.max(spans)) >= 0.12 * global_scale:
            repeated.append([str(item["id"]) for item in items])
    return repeated


def build_motif_graph_for_sample(data: Dict[str, Any]) -> Dict[str, Any]:
    features = extract_motif_features(data)
    faces = features["face_features"]
    face_count = int(features["face_count"])
    face_adj = np.asarray(features["face_adjacency"], dtype=int)
    gd = np.asarray(features["global_dims"], dtype=float)
    global_scale = float(features.get("global_scale", np.max(gd) if gd.size else 1.0))
    areas = np.asarray([float(f["area_proxy"]) for f in faces], dtype=float) if faces else np.zeros(0)
    area_p25 = float(np.percentile(areas, 25)) if areas.size else 0.0
    area_p60 = float(np.percentile(areas, 60)) if areas.size else 0.0

    boundary_ids = [f["face_id"] for f in faces if f["is_global_boundary"]]
    loop_ids = [
        f["face_id"]
        for f in faces
        if bool(f["is_internal_xz"]) or (not f["is_global_boundary"] and float(f["area_proxy"]) <= max(area_p25, 1e-8) and int(f["degree"]) >= 2)
    ]
    loop_set = set(loop_ids)
    transition_ids = []
    for f in faces:
        fid = int(f["face_id"])
        if fid in loop_set:
            continue
        area = float(f["area_proxy"])
        degree = int(f["degree"])
        aspect = float(f["aspect_ratio"])
        is_boundary = bool(f["is_global_boundary"])
        curved_small = float(f["curvature_proxy"]) > 0.5 and area <= max(area_p60, 1e-8)
        small_connected = (not is_boundary) and area <= max(area_p25, 1e-8) and degree >= 2
        narrow_small_connected = (not is_boundary) and aspect > 45.0 and area <= max(area_p60, 1e-8) and degree >= 2
        if curved_small or small_connected or narrow_small_connected:
            transition_ids.append(fid)
    used = set()
    nodes: List[Dict[str, Any]] = []

    def add_components(face_ids: Sequence[int], node_type: str, prefix: str) -> None:
        nonlocal nodes, used
        for comp in _connected_components(face_ids, face_adj):
            if not comp:
                continue
            nodes.append(_make_node(f"{prefix}_{len(nodes)}", node_type, comp, faces))
            used.update(comp)

    add_components(loop_ids, "loop_or_hole", "loop")
    add_components(transition_ids, "transition_group", "transition")

    sheet_ids = [
        f["face_id"]
        for f in faces
        if f["face_id"] not in used and float(f["area_proxy"]) >= area_p60 and float(f["aspect_ratio"]) <= 200.0
    ]
    for fid in sheet_ids:
        nodes.append(_make_node(f"sheet_{len(nodes)}", "sheet_like_group", [fid], faces))
        used.add(fid)

    remaining_boundary = [fid for fid in boundary_ids if fid not in used]
    add_components(remaining_boundary, "boundary_group", "boundary")

    remaining = [fid for fid in range(face_count) if fid not in used]
    for comp in _connected_components(remaining, face_adj):
        nodes.append(_make_node(f"face_group_{len(nodes)}", "face_group", comp, faces))

    relations: List[Dict[str, Any]] = []
    relation_keys = set()

    def add_rel(src: str, dst: str, typ: str, score: float) -> None:
        if src == dst or typ not in RELATION_TYPES:
            return
        key = (src, dst, typ)
        rev_key = (dst, src, typ)
        if typ in {"parallel_to", "opposite_to", "orthogonal_to", "coplanar_with", "adjacent_to"} and rev_key in relation_keys:
            return
        if key not in relation_keys:
            relation_keys.add(key)
            relations.append(_relation(src, dst, typ, score))

    for i, a in enumerate(nodes):
        a_faces = a["face_ids"]
        a_mn, a_mx = _group_bbox(a_faces, faces)
        a_c = _group_centroid(a_faces, faces)
        a_n = _group_normal(a_faces, faces)
        for b in nodes[i + 1 :]:
            b_faces = b["face_ids"]
            b_mn, b_mx = _group_bbox(b_faces, faces)
            b_c = _group_centroid(b_faces, faces)
            b_n = _group_normal(b_faces, faces)
            adj_count = _adjacent_faces(a_faces, b_faces, face_adj)
            if adj_count:
                add_rel(a["id"], b["id"], "adjacent_to", min(1.0, adj_count / 3.0))
            dot_abs = abs(_angle_dot(a_n, b_n))
            dot = _angle_dot(a_n, b_n)
            if dot_abs >= 0.94:
                add_rel(a["id"], b["id"], "parallel_to", dot_abs)
                dominant = int(np.argmax(np.abs(a_n)))
                plane_dist = abs(float(np.dot(b_c - a_c, a_n)))
                overlap_axes = [axis for axis in range(3) if axis != dominant]
                overlap = _projection_overlap(a_mn, a_mx, b_mn, b_mx, overlap_axes)
                if plane_dist <= 0.025 * global_scale and overlap >= 0.35:
                    add_rel(a["id"], b["id"], "coplanar_with", 1.0 - plane_dist / max(0.025 * global_scale, 1e-8))
                if dot <= -0.75 or plane_dist > 0.025 * global_scale:
                    if overlap >= 0.35:
                        add_rel(a["id"], b["id"], "opposite_to", min(1.0, overlap))
            elif dot_abs <= 0.18:
                add_rel(a["id"], b["id"], "orthogonal_to", 1.0 - dot_abs)

    thin_nodes: List[Dict[str, Any]] = []
    for rel in list(relations):
        if rel["type"] != "opposite_to":
            continue
        src = next((n for n in nodes if n["id"] == rel["source"]), None)
        dst = next((n for n in nodes if n["id"] == rel["target"]), None)
        if not src or not dst:
            continue
        c0 = _group_centroid(src["face_ids"], faces)
        c1 = _group_centroid(dst["face_ids"], faces)
        n0 = _group_normal(src["face_ids"], faces)
        distance = abs(float(np.dot(c1 - c0, n0)))
        if distance <= 0.12 * global_scale and rel["score"] >= 0.35:
            node = _make_node(
                f"thinwall_{len(nodes) + len(thin_nodes)}",
                "thin_wall_pair",
                sorted(set(src["face_ids"]) | set(dst["face_ids"])),
                faces,
                {"distance": round(distance, 6), "source_nodes": [src["id"], dst["id"]]},
            )
            thin_nodes.append(node)
            add_rel(node["id"], src["id"], "bounded_by", 1.0)
            add_rel(node["id"], dst["id"], "bounded_by", 1.0)
    nodes.extend(thin_nodes)

    repeated_clusters = _find_repeated_groups(nodes, faces, global_scale)
    for cluster in repeated_clusters:
        face_ids: List[int] = []
        for node_id in cluster:
            node = next((n for n in nodes if n["id"] == node_id), None)
            if node:
                face_ids.extend(node["face_ids"])
        rep_node = _make_node(f"repeated_{len(nodes)}", "repeated_feature", face_ids, faces, {"member_nodes": cluster})
        nodes.append(rep_node)
        for node_id in cluster:
            add_rel(rep_node["id"], node_id, "repeated_with", 1.0)

    for node in nodes:
        if node["type"] == "transition_group":
            neighbors = [rel for rel in relations if rel["type"] == "adjacent_to" and (rel["source"] == node["id"] or rel["target"] == node["id"])]
            for item in neighbors[:2]:
                other = item["target"] if item["source"] == node["id"] else item["source"]
                add_rel(node["id"], other, "smooth_connected", item["score"])
        if node["type"] == "loop_or_hole":
            loop_mn, loop_mx = _group_bbox(node["face_ids"], faces)
            loop_c = 0.5 * (loop_mn + loop_mx)
            container = None
            best_score = 0.0
            for other in nodes:
                if other["id"] == node["id"] or other["type"] not in {"sheet_like_group", "face_group", "boundary_group"}:
                    continue
                mn, mx = _group_bbox(other["face_ids"], faces)
                inside = all(loop_c[axis] >= mn[axis] - 1e-8 and loop_c[axis] <= mx[axis] + 1e-8 for axis in range(3))
                if inside:
                    score = float(_group_area(other["face_ids"], faces))
                    if score > best_score:
                        best_score = score
                        container = other
            if container:
                add_rel(node["id"], container["id"], "embedded_in", 1.0)

    relation_counts = Counter(rel["type"] for rel in relations)
    node_counts = Counter(node["type"] for node in nodes)
    summary = {
        "has_parallel": relation_counts["parallel_to"] > 0,
        "has_thin_wall": node_counts["thin_wall_pair"] > 0,
        "has_loop_or_hole": node_counts["loop_or_hole"] > 0,
        "has_transition": node_counts["transition_group"] > 0,
        "has_repeated_feature": node_counts["repeated_feature"] > 0,
        "motif_rich": len(nodes) >= 4 and len(relations) >= 4 and (relation_counts["parallel_to"] + relation_counts["opposite_to"] + relation_counts["orthogonal_to"] > 0),
    }
    return {
        "uid": features["uid"],
        "source": features["source"],
        "num_faces": int(features["face_count"]),
        "num_edges": int(features["edge_count"]),
        "num_vertices": int(features["vertex_count"]),
        "motif_nodes": nodes,
        "motif_relations": sorted(relations, key=lambda x: (x["source"], x["target"], x["type"])),
        "motif_summary": summary,
        "parser_backend": features.get("parser_backend", "unknown"),
        "geometry_sampling_quality": features.get("geometry_sampling_quality", "unknown"),
        "label_source": "algorithm_extracted_motif",
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
            graph = build_motif_graph_for_sample(data)
            graphs.append(graph)
        except Exception as exc:
            failures.append({"uid": pkl_path.stem, "error": str(exc)})

    write_jsonl(os.path.join(motif_dir, "motif_graphs.jsonl"), graphs)
    node_rows: List[Dict[str, Any]] = []
    rel_rows: List[Dict[str, Any]] = []
    for graph in graphs:
        node_counter = Counter(node["type"] for node in graph["motif_nodes"])
        rel_counter = Counter(rel["type"] for rel in graph["motif_relations"])
        row = {"uid": graph["uid"], "num_nodes": len(graph["motif_nodes"])}
        row.update({f"node_{key}": node_counter.get(key, 0) for key in sorted(NODE_TYPES)})
        node_rows.append(row)
        rrow = {"uid": graph["uid"], "num_relations": len(graph["motif_relations"])}
        rrow.update({f"rel_{key}": rel_counter.get(key, 0) for key in sorted(RELATION_TYPES)})
        rel_rows.append(rrow)
    write_csv(os.path.join(motif_dir, "motif_node_stats.csv"), node_rows, ["uid", "num_nodes"] + [f"node_{key}" for key in sorted(NODE_TYPES)])
    write_csv(os.path.join(motif_dir, "motif_relation_stats.csv"), rel_rows, ["uid", "num_relations"] + [f"rel_{key}" for key in sorted(RELATION_TYPES)])

    rich_count = sum(1 for graph in graphs if graph["motif_summary"].get("motif_rich"))
    avg_nodes = float(np.mean([len(graph["motif_nodes"]) for graph in graphs])) if graphs else 0.0
    avg_relations = float(np.mean([len(graph["motif_relations"]) for graph in graphs])) if graphs else 0.0
    report = [
        "Innovation1 v3 Motif Extraction Report",
        "=" * 72,
        f"Time: {timestamp()}",
        f"Parsed samples: {len(list(Path(parsed_dir).glob('*.pkl')))}",
        f"Motif graph success: {len(graphs)}",
        f"Motif graph failures: {len(failures)}",
        f"Motif-rich samples: {rich_count}",
        f"Average motif nodes: {avg_nodes:.3f}",
        f"Average motif relations: {avg_relations:.3f}",
        "",
        "Interpretation:",
        "  Motif nodes and relations are algorithm-extracted weak structural priors.",
        "  They are not human labels and are not real engineering semantic truth.",
    ]
    if failures:
        report.extend(["", "Failures:"])
        for item in failures[:100]:
            report.append(f"  - {item['uid']}: {item['error']}")
    write_text(os.path.join(reports_dir, "motif_extraction_report.txt"), report)
    return {"graphs": graphs, "failures": failures}
