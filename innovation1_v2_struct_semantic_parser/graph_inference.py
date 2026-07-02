# -*- coding: utf-8 -*-
"""Rule-based weak semantic parser from B-Rep features to inferred Gc."""

from __future__ import annotations

import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from semantic_feature_extractor import extract_semantic_features
from utils_io import (
    FACE_ROLES,
    PARAMETER_KEYS,
    ensure_workdir,
    read_json,
    scan_uid_files,
    write_jsonl,
    write_text,
)


def _load_pkl(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _axis_dims(face: Dict[str, Any]) -> Tuple[float, float, float]:
    d = face["dims"]
    return float(d[0]), float(d[1]), float(d[2])


def _classify_faces(features: Dict[str, Any], part_type: str = "") -> Dict[int, str]:
    gd = np.asarray(features["global_dims"], dtype=float)
    gmin = np.asarray(features["global_min"], dtype=float)
    gmax = np.asarray(features["global_max"], dtype=float)
    faces = features["face_features"]
    if not faces:
        return {}
    areas = np.asarray([f["area_proxy"] for f in faces], dtype=float)
    large_area = float(np.percentile(areas, 70)) if areas.size else 0.0
    small_area = float(np.percentile(areas, 25)) if areas.size else 0.0
    roles: Dict[int, str] = {}
    surface_order_verified = bool(features.get("surface_metadata_order_verified", True))
    scale = float(np.max(gd)) if gd.size else 1.0
    abs_floor = max(1e-4 * scale, 1e-6)
    tol = np.maximum(gd * 0.03, abs_floor)
    thin_floor = max(0.01 * scale, abs_floor)
    end_face_max = max(0.08 * gd[2], 0.015 * scale)
    narrow_floor = max(0.04 * scale, abs_floor)

    for face in faces:
        fid = int(face["face_id"])
        dx, dy, dz = _axis_dims(face)
        c = np.asarray(face["centroid"], dtype=float)
        bbox = np.asarray(face["bbox"], dtype=float)
        area = float(face["area_proxy"])
        curved_face = surface_order_verified and float(face.get("curvature_proxy", 0.0)) > 0.5
        internal_xz = bool(face["is_internal_xz"])
        near_z_end = (abs(bbox[2] - gmin[2]) <= tol[2]) or (abs(bbox[5] - gmax[2]) <= tol[2])
        near_x_side = (abs(bbox[0] - gmin[0]) <= tol[0]) or (abs(bbox[3] - gmax[0]) <= tol[0])
        thin_y = dy <= max(0.12 * gd[1], thin_floor)
        long_z = dz >= 0.42 * gd[2]
        wide_x = dx >= 0.32 * gd[0]
        narrow_x = dx <= max(0.22 * gd[0], narrow_floor)

        if internal_xz and dy <= max(0.45 * gd[1], thin_floor) and dx <= 0.45 * gd[0] and dz <= 0.45 * gd[2]:
            if abs(dx - dz) / max(dx, dz, 1e-6) <= 0.35:
                roles[fid] = "hole"
            else:
                roles[fid] = "cutout"
            continue

        if near_z_end and dz <= end_face_max:
            roles[fid] = "boundary"
            continue

        if "runout" in part_type and c[1] > gmin[1] + 0.22 * gd[1] and near_z_end and narrow_x:
            roles[fid] = "runout"
            continue

        if "hat" in part_type:
            mid_x = 0.5 * (gmin[0] + gmax[0])
            near_top = abs(bbox[4] - gmax[1]) <= tol[1] or c[1] > gmin[1] + 0.70 * gd[1]
            centered_x = abs(c[0] - mid_x) <= 0.25 * gd[0]
            cap_wide = dx >= 0.15 * gd[0]
            if near_top and centered_x and long_z and cap_wide:
                roles[fid] = "cap"
                continue

        if "curved" in part_type and curved_face and wide_x and long_z and area >= large_area:
            roles[fid] = "panel"
            continue

        if thin_y and wide_x and long_z and area >= small_area:
            roles[fid] = "panel"
            continue

        if curved_face and (area <= large_area or narrow_x or face["aspect_ratio"] > 18.0):
            roles[fid] = "transition"
            continue

        if c[1] > gmin[1] + 0.25 * gd[1] and long_z and narrow_x:
            roles[fid] = "stiffener"
            continue

        if dy >= 0.35 * gd[1] and long_z and narrow_x:
            roles[fid] = "web"
            continue

        if thin_y and long_z and (wide_x or near_x_side):
            roles[fid] = "flange"
            continue

        if area <= max(small_area, 1.0) or face["aspect_ratio"] > 35.0:
            roles[fid] = "transition"
            continue

        roles[fid] = "unassigned"
    return roles


def _cluster_faces(face_ids: Sequence[int], features: Dict[str, Any], axes: Tuple[int, ...], radius: float) -> List[List[int]]:
    centers = {
        int(f["face_id"]): np.asarray(f["centroid"], dtype=float)
        for f in features["face_features"]
        if int(f["face_id"]) in set(face_ids)
    }
    clusters: List[List[int]] = []
    cluster_centers: List[np.ndarray] = []
    for fid in sorted(face_ids, key=lambda item: tuple(centers[int(item)][list(axes)])):
        c = centers[int(fid)][list(axes)]
        assigned = False
        for idx, cc in enumerate(cluster_centers):
            if float(np.linalg.norm(c - cc)) <= radius:
                clusters[idx].append(int(fid))
                pts = np.array([centers[x][list(axes)] for x in clusters[idx]], dtype=float)
                cluster_centers[idx] = np.mean(pts, axis=0)
                assigned = True
                break
        if not assigned:
            clusters.append([int(fid)])
            cluster_centers.append(c)
    return clusters


def _adjacency_components(face_ids: Sequence[int], features: Dict[str, Any]) -> List[List[int]]:
    id_set = {int(fid) for fid in face_ids}
    if not id_set:
        return []
    adj = np.asarray(features.get("face_adjacency", []), dtype=int)
    seen = set()
    components: List[List[int]] = []
    for seed in sorted(id_set):
        if seed in seen:
            continue
        stack = [seed]
        seen.add(seed)
        comp: List[int] = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            if cur >= adj.shape[0]:
                continue
            neighbors = [idx for idx in np.where(adj[cur] > 0)[0].tolist() if idx in id_set and idx not in seen]
            for nb in neighbors:
                seen.add(nb)
                stack.append(int(nb))
        components.append(sorted(comp))
    return components


def _group_faces(roles: Dict[int, str], features: Dict[str, Any], part_type: str = "") -> List[Dict[str, Any]]:
    gd = np.asarray(features["global_dims"], dtype=float)
    face_by_id = {int(f["face_id"]): f for f in features["face_features"]}
    groups: List[Dict[str, Any]] = []

    def add_group(role: str, face_ids: Sequence[int], node_id: str) -> None:
        if face_ids:
            for group in groups:
                if group["node_id"] == node_id:
                    group["face_ids"] = sorted(set(group["face_ids"]) | {int(x) for x in face_ids})
                    return
            groups.append({"node_id": node_id, "role": role, "face_ids": sorted(int(x) for x in face_ids)})

    beam_like = part_type in {"tapered_c_channel", "tapered_hat_stiffener"}
    panel_faces = [fid for fid, role in roles.items() if role == "panel"]
    if not beam_like:
        add_group("panel", panel_faces, "panel_0")
    elif panel_faces and not any(role == "web" for role in roles.values()):
        add_group("web", panel_faces, "web_0")

    boundary_faces = [fid for fid, role in roles.items() if role == "boundary"]
    if boundary_faces:
        start = [fid for fid in boundary_faces if face_by_id[fid]["centroid"][2] <= 0.5 * (features["global_min"][2] + features["global_max"][2])]
        end = [fid for fid in boundary_faces if fid not in start]
        add_group("boundary", start, "boundary_start")
        add_group("boundary", end, "boundary_end")

    for role, axes, scale in [
        ("hole", (0, 2), 0.18),
        ("cutout", (0, 2), 0.18),
        ("stiffener", (0,), 0.12),
        ("web", (0,), 0.16),
        ("cap", (0, 1), 0.16),
        ("flange", (0, 1), 0.18),
        ("runout", (0, 2), 0.16),
        ("transition", (0, 1, 2), 0.12),
    ]:
        ids = [fid for fid, item_role in roles.items() if item_role == role]
        if not ids:
            continue
        if role in {"hole", "cutout", "stiffener", "transition", "runout"}:
            clusters = _adjacency_components(ids, features)
        else:
            scale_floor = max(0.01 * float(np.max(gd)), 1e-6)
            radius = max(float(np.linalg.norm(gd[list(axes)])) * scale, scale_floor)
            clusters = _cluster_faces(ids, features, axes, radius)
        for idx, cluster in enumerate(clusters):
            add_group(role, cluster, f"{role}_{idx}")
    return groups


def _group_centroid(group: Dict[str, Any], features: Dict[str, Any]) -> np.ndarray:
    face_by_id = {int(f["face_id"]): f for f in features["face_features"]}
    pts = [face_by_id[fid]["centroid"] for fid in group["face_ids"] if fid in face_by_id]
    if not pts:
        return np.zeros(3)
    return np.mean(np.asarray(pts, dtype=float), axis=0)


def _infer_relations(groups: List[Dict[str, Any]], features: Dict[str, Any], part_type: str = "") -> List[Dict[str, str]]:
    face_to_group: Dict[int, str] = {}
    role_by_group = {g["node_id"]: g["role"] for g in groups}
    for group in groups:
        for fid in group["face_ids"]:
            face_to_group[int(fid)] = group["node_id"]

    adj = np.asarray(features["face_adjacency"], dtype=int)
    rels = set()
    for i in range(adj.shape[0]):
        for j in range(i + 1, adj.shape[1]):
            if adj[i, j] <= 0:
                continue
            gi = face_to_group.get(i)
            gj = face_to_group.get(j)
            if not gi or not gj or gi == gj:
                continue
            ri = role_by_group.get(gi, "")
            rj = role_by_group.get(gj, "")
            rel_type = "attached_to"
            if "transition" in {ri, rj}:
                rel_type = "smooth_connected"
            if ri == "hole":
                rel_type = "hole_of"
            elif rj == "hole":
                gi, gj = gj, gi
                rel_type = "hole_of"
            elif ri == "cutout":
                rel_type = "cutout_of"
            elif rj == "cutout":
                gi, gj = gj, gi
                rel_type = "cutout_of"
            elif ri == "runout":
                rel_type = "runout_of"
            elif rj == "runout":
                gi, gj = gj, gi
                rel_type = "runout_of"
            elif "boundary" in {ri, rj}:
                rel_type = "boundary_of"
            rels.add((gi, gj, rel_type))

    stiffeners = [g for g in groups if g["role"] == "stiffener"]
    for idx in range(1, len(stiffeners)):
        rels.add((stiffeners[idx]["node_id"], stiffeners[idx - 1]["node_id"], "parallel_to"))
    if len(stiffeners) >= 2:
        centers = [_group_centroid(g, features) for g in stiffeners]
        mid_x = 0.5 * (features["global_min"][0] + features["global_max"][0])
        for i in range(len(stiffeners)):
            for j in range(i + 1, len(stiffeners)):
                if abs((centers[i][0] - mid_x) + (centers[j][0] - mid_x)) <= 0.08 * features["global_dims"][0]:
                    rels.add((stiffeners[i]["node_id"], stiffeners[j]["node_id"], "symmetric_to"))

    has_panel = any(g["node_id"] == "panel_0" for g in groups)
    panel_like = part_type not in {"tapered_c_channel", "tapered_hat_stiffener"}
    for group in groups:
        if group["role"] in {"hole", "cutout", "boundary"}:
            target = "panel_0" if has_panel else "web_0"
            rel_type = "hole_of" if group["role"] == "hole" else "cutout_of" if group["role"] == "cutout" else "boundary_of"
            rels.add((group["node_id"], target, rel_type))
        if has_panel and panel_like and group["role"] in {"stiffener", "runout"}:
            rels.add((group["node_id"], "panel_0", "attached_to"))
    web_groups = [g for g in groups if g["role"] == "web"]
    flange_groups = [g for g in groups if g["role"] == "flange"]
    cap_groups = [g for g in groups if g["role"] == "cap"]
    if not (has_panel and panel_like) and web_groups:
        web_id = web_groups[0]["node_id"]
        for group in flange_groups:
            rels.add((group["node_id"], web_id, "attached_to"))
        for group in cap_groups:
            rels.add((group["node_id"], web_id, "attached_to"))
        if len(flange_groups) >= 2:
            rels.add((flange_groups[0]["node_id"], flange_groups[1]["node_id"], "parallel_to"))

    return [{"source": s, "target": t, "type": r} for s, t, r in sorted(rels)]


def _estimate_parameters(groups: List[Dict[str, Any]], features: Dict[str, Any]) -> Dict[str, float]:
    gd = np.asarray(features["global_dims"], dtype=float)
    params = {key: 0.0 for key in PARAMETER_KEYS}
    params["length"] = round(float(gd[2]), 3)
    params["width"] = round(float(gd[0]), 3)
    params["height"] = round(float(gd[1]), 3)

    panel_faces = [f for f in features["face_features"] if f["face_id"] in {fid for g in groups if g["role"] == "panel" for fid in g["face_ids"]}]
    thin_dims = [min(f["dims"]) for f in panel_faces if min(f["dims"]) > 1e-4]
    params["thickness"] = round(float(np.median(thin_dims)), 3) if thin_dims else round(float(min(gd)), 3)

    stiff_groups = [g for g in groups if g["role"] == "stiffener"]
    params["rib_count"] = float(len(stiff_groups))
    if stiff_groups:
        widths = []
        heights = []
        for group in stiff_groups:
            boxes = [features["face_features"][fid]["bbox"] for fid in group["face_ids"] if fid < len(features["face_features"])]
            if boxes:
                arr = np.asarray(boxes, dtype=float)
                mn = np.min(arr[:, :3], axis=0)
                mx = np.max(arr[:, 3:], axis=0)
                widths.append(mx[0] - mn[0])
                heights.append(mx[1] - mn[1])
        if widths:
            params["rib_width"] = round(float(np.median(widths)), 3)
        if heights:
            params["rib_height"] = round(float(np.median(heights)), 3)

    hole_groups = [g for g in groups if g["role"] == "hole"]
    cutout_groups = [g for g in groups if g["role"] == "cutout"]
    params["hole_count"] = float(len(hole_groups) + len(cutout_groups))
    if hole_groups:
        diameters = []
        for group in hole_groups:
            boxes = [features["face_features"][fid]["bbox"] for fid in group["face_ids"] if fid < len(features["face_features"])]
            if boxes:
                arr = np.asarray(boxes, dtype=float)
                mn = np.min(arr[:, :3], axis=0)
                mx = np.max(arr[:, 3:], axis=0)
                diameters.append(0.5 * ((mx[0] - mn[0]) + (mx[2] - mn[2])))
        if diameters:
            params["hole_radius"] = round(float(np.median(diameters) / 2.0), 3)
    if cutout_groups:
        widths = []
        heights = []
        for group in cutout_groups:
            boxes = [features["face_features"][fid]["bbox"] for fid in group["face_ids"] if fid < len(features["face_features"])]
            if boxes:
                arr = np.asarray(boxes, dtype=float)
                mn = np.min(arr[:, :3], axis=0)
                mx = np.max(arr[:, 3:], axis=0)
                widths.append(mx[0] - mn[0])
                heights.append(mx[2] - mn[2])
        if widths:
            params["hole_width"] = round(float(np.median(widths)), 3)
        if heights:
            params["hole_height"] = round(float(np.median(heights)), 3)

    flange_groups = [g for g in groups if g["role"] == "flange"]
    if flange_groups:
        spans = []
        for group in flange_groups:
            boxes = [features["face_features"][fid]["bbox"] for fid in group["face_ids"] if fid < len(features["face_features"])]
            if boxes:
                arr = np.asarray(boxes, dtype=float)
                mn = np.min(arr[:, :3], axis=0)
                mx = np.max(arr[:, 3:], axis=0)
                spans.append(mx[0] - mn[0])
        if spans:
            params["flange_width"] = round(float(np.median(spans)), 3)
            params["taper_ratio"] = round(float((max(spans) - min(spans)) / max(max(spans), 1e-6)), 3)

    panel_centroids_y = [f["centroid"][1] for f in features["face_features"] if f["face_id"] in {fid for g in groups if g["role"] == "panel" for fid in g["face_ids"]}]
    if panel_centroids_y:
        sag = max(panel_centroids_y) - min(panel_centroids_y)
        if sag > 0.5:
            params["curvature_radius"] = round(float((gd[0] ** 2) / max(8.0 * sag, 1e-6)), 3)

    runout_groups = [g for g in groups if g["role"] == "runout"]
    if runout_groups:
        z_spans = []
        for group in runout_groups:
            boxes = [features["face_features"][fid]["bbox"] for fid in group["face_ids"] if fid < len(features["face_features"])]
            if boxes:
                arr = np.asarray(boxes, dtype=float)
                mn = np.min(arr[:, :3], axis=0)
                mx = np.max(arr[:, 3:], axis=0)
                z_spans.append(mx[2] - mn[2])
        if z_spans:
            params["runout_length"] = round(float(np.median(z_spans)), 3)
    return params


def _infer_mechanisms(groups: List[Dict[str, Any]], params: Dict[str, float], part_type: str) -> List[str]:
    roles = Counter(g["role"] for g in groups)
    mechanisms = set()
    if roles["hole"] > 0:
        mechanisms.update(["hole", "inner_loop"])
    if roles["cutout"] > 0:
        mechanisms.update(["cutout", "inner_loop"])
    if roles["stiffener"] > 0:
        mechanisms.add("stiffener")
    if roles["stiffener"] > 1:
        mechanisms.add("multi_stiffener")
    if roles["runout"] > 0 or "runout" in part_type:
        mechanisms.add("runout")
    if params.get("taper_ratio", 0.0) > 0.05 or "tapered" in part_type:
        mechanisms.add("taper")
    if params.get("curvature_radius", 0.0) > 0.0 or "curved" in part_type:
        mechanisms.add("curved_surface")
    if "asymmetric" in part_type:
        mechanisms.add("asymmetric_layout")
    return sorted(mechanisms)


def infer_sample_semantics(json_data: Dict[str, Any], pkl_data: Dict[str, Any]) -> Dict[str, Any]:
    part_type = str(json_data.get("part_type", pkl_data.get("part_type", "")))
    features = extract_semantic_features(pkl_data)
    roles = _classify_faces(features, part_type)
    groups = _group_faces(roles, features, part_type)
    nodes = [{"id": g["node_id"], "type": g["role"], "face_ids": g["face_ids"]} for g in groups]
    relations = _infer_relations(groups, features, part_type)
    params = _estimate_parameters(groups, features)
    mechanisms = _infer_mechanisms(groups, params, part_type)
    return {
        "uid": json_data.get("uid", pkl_data.get("uid", "")),
        "part_type": part_type,
        "inferred_face_roles": {str(fid): role for fid, role in sorted(roles.items())},
        "inferred_face_groups": groups,
        "inferred_config_graph": {"nodes": nodes, "relations": relations},
        "inferred_parameters": params,
        "inferred_topology_mechanisms": mechanisms,
        "feature_summary": {
            "face_count": features["face_count"],
            "edge_count": features["edge_count"],
            "vertex_count": features["vertex_count"],
            "global_dims": features["global_dims"],
        },
    }


def infer_semantics(workdir: str) -> Dict[str, Any]:
    dirs = ensure_workdir(workdir)
    dataset_dir = dirs["enhanced_dataset"]
    parsed_dir = dirs["enhanced_parsed"]
    inferred_dir = dirs["inferred_semantics"]
    reports_dir = dirs["reports"]

    uids = sorted(set(scan_uid_files(dataset_dir, ".json")) & set(scan_uid_files(parsed_dir, ".pkl")))
    face_group_records = []
    graph_records = []
    parameter_records = []
    failures = []
    role_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    hole_count = 0
    stiffener_count_sum = 0
    unassigned = 0
    total_faces = 0
    node_counts = []
    relation_count_values = []

    for uid in uids:
        try:
            jd = read_json(os.path.join(dataset_dir, f"{uid}.json"))
            pd = _load_pkl(os.path.join(parsed_dir, f"{uid}.pkl"))
            result = infer_sample_semantics(jd, pd)
            roles = result["inferred_face_roles"]
            role_counts.update(roles.values())
            unassigned += sum(1 for role in roles.values() if role == "unassigned")
            total_faces += len(roles)
            graph = result["inferred_config_graph"]
            relation_counts.update(rel["type"] for rel in graph["relations"])
            hole_count += int(result["inferred_parameters"].get("hole_count", 0))
            stiffener_count_sum += int(result["inferred_parameters"].get("rib_count", 0))
            node_counts.append(len(graph["nodes"]))
            relation_count_values.append(len(graph["relations"]))
            face_group_records.append(
                {
                    "uid": uid,
                    "part_type": result["part_type"],
                    "inferred_face_roles": result["inferred_face_roles"],
                    "inferred_face_groups": result["inferred_face_groups"],
                }
            )
            graph_records.append(
                {
                    "uid": uid,
                    "part_type": result["part_type"],
                    "inferred_config_graph": graph,
                    "inferred_topology_mechanisms": result["inferred_topology_mechanisms"],
                }
            )
            parameter_records.append(
                {
                    "uid": uid,
                    "part_type": result["part_type"],
                    "inferred_parameters": result["inferred_parameters"],
                    "feature_summary": result["feature_summary"],
                }
            )
        except Exception as exc:
            failures.append({"uid": uid, "error": str(exc)})

    write_jsonl(os.path.join(inferred_dir, "inferred_face_groups.jsonl"), face_group_records)
    write_jsonl(os.path.join(inferred_dir, "inferred_config_graphs.jsonl"), graph_records)
    write_jsonl(os.path.join(inferred_dir, "inferred_parameters.jsonl"), parameter_records)

    success = len(face_group_records)
    avg_nodes = sum(node_counts) / max(len(node_counts), 1)
    avg_rels = sum(relation_count_values) / max(len(relation_count_values), 1)
    report = [
        "Innovation1 v2 Weak Semantic Parser Report",
        "=" * 72,
        f"Input paired samples: {len(uids)}",
        f"Success: {success}",
        f"Failures: {len(failures)}",
        f"Unassigned face ratio: {unassigned / max(total_faces, 1):.4f}",
        "",
        "Face role counts:",
    ]
    for role in FACE_ROLES:
        report.append(f"  - {role}: {role_counts.get(role, 0)}")
    report.extend(["", "Relation type counts:"])
    for rel_type, count in sorted(relation_counts.items()):
        report.append(f"  - {rel_type}: {count}")
    report.extend(
        [
            "",
            f"Hole/cutout detected count: {hole_count}",
            f"Accumulated stiffener count: {stiffener_count_sum}",
            f"Average inferred node count: {avg_nodes:.3f}",
            f"Average inferred relation count: {avg_rels:.3f}",
            "",
            "Boundary statement:",
            "  inferred_Gc is a rule-based weak parse from B-Rep geometry/topology features; it is not human annotation.",
            "  weak_aligned_face_groups are not copied into the inferred result.",
        ]
    )
    if failures:
        report.extend(["", "Failure details:"])
        for item in failures[:50]:
            report.append(f"  - {item['uid']}: {item['error']}")
    write_text(os.path.join(reports_dir, "semantic_parser_report.txt"), report)

    return {
        "success": success,
        "failures": failures,
        "face_group_records": face_group_records,
        "graph_records": graph_records,
        "parameter_records": parameter_records,
    }
