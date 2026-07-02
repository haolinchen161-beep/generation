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
        end_like = near_z_end and dz <= end_face_max

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

        if part_type == "tapered_c_channel" and not end_like and long_z:
            if dx <= max(0.18 * gd[0], thin_floor) and dy >= 0.30 * gd[1]:
                roles[fid] = "web"
                continue
            if dy <= max(0.18 * gd[1], thin_floor) and dx >= 0.18 * gd[0]:
                roles[fid] = "flange"
                continue
            if area <= small_area or face["aspect_ratio"] > 24.0 or (dx <= 0.20 * gd[0] and dy <= 0.20 * gd[1]):
                roles[fid] = "transition"
                continue

        if "hat" in part_type:
            mid_x = 0.5 * (gmin[0] + gmax[0])
            near_top = abs(bbox[4] - gmax[1]) <= tol[1] or c[1] > gmin[1] + 0.70 * gd[1]
            centered_x = abs(c[0] - mid_x) <= 0.25 * gd[0]
            cap_wide = dx >= 0.15 * gd[0]
            if near_top and centered_x and long_z and cap_wide:
                roles[fid] = "cap"
                continue

        if part_type == "curved_stiffened_panel":
            high_above_skin = c[1] > gmin[1] + 0.55 * gd[1]
            if high_above_skin and long_z and narrow_x:
                roles[fid] = "stiffener"
                continue

        if "curved" in part_type:
            large_panel_like = wide_x and long_z and area >= large_area
            near_skin_zone = c[1] <= gmin[1] + 0.55 * gd[1]
            if large_panel_like and not end_like and near_skin_zone:
                roles[fid] = "panel"
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


def _cluster_opening_objects(face_ids: Sequence[int], features: Dict[str, Any]) -> List[List[int]]:
    ids = sorted({int(fid) for fid in face_ids})
    if not ids:
        return []
    face_by_id = {int(f["face_id"]): f for f in features["face_features"]}
    gd = np.asarray(features["global_dims"], dtype=float)
    diag_xz = float(np.linalg.norm(gd[[0, 2]])) if gd.size >= 3 else 1.0

    def xz_box(fid: int) -> Tuple[float, float, float, float]:
        box = np.asarray(face_by_id[fid]["bbox"], dtype=float)
        return float(box[0]), float(box[2]), float(box[3]), float(box[5])

    def xz_center(fid: int) -> np.ndarray:
        c = np.asarray(face_by_id[fid]["centroid"], dtype=float)
        return c[[0, 2]]

    def overlap_1d(a0: float, a1: float, b0: float, b1: float, extra: float) -> bool:
        return max(a0, b0) <= min(a1, b1) + extra

    adjacency = {fid: set() for fid in ids}
    boxes = {fid: xz_box(fid) for fid in ids}
    centers = {fid: xz_center(fid) for fid in ids}
    for idx, a in enumerate(ids):
        ax0, az0, ax1, az1 = boxes[a]
        a_extent = max(ax1 - ax0, az1 - az0, 1e-6)
        for b in ids[idx + 1:]:
            bx0, bz0, bx1, bz1 = boxes[b]
            b_extent = max(bx1 - bx0, bz1 - bz0, 1e-6)
            local_pad = max(0.08 * min(a_extent, b_extent), 1e-6)
            bbox_touch = overlap_1d(ax0, ax1, bx0, bx1, local_pad) and overlap_1d(az0, az1, bz0, bz1, local_pad)
            center_limit = max(1e-6, min(0.015 * diag_xz, 0.70 * (a_extent + b_extent)))
            center_close = float(np.linalg.norm(centers[a] - centers[b])) <= center_limit
            if bbox_touch or center_close:
                adjacency[a].add(b)
                adjacency[b].add(a)

    components: List[List[int]] = []
    seen = set()
    for seed in ids:
        if seed in seen:
            continue
        stack = [seed]
        seen.add(seed)
        comp: List[int] = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in sorted(adjacency[cur]):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
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

    opening_ids = [fid for fid, role in roles.items() if role in {"hole", "cutout"}]
    opening_counts = Counter()
    for cluster in _cluster_opening_objects(opening_ids, features):
        role = "cutout" if any(roles.get(fid) == "cutout" for fid in cluster) else "hole"
        idx = opening_counts[role]
        opening_counts[role] += 1
        add_group(role, cluster, f"{role}_{idx}")

    for role, axes, scale in [
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


def _group_bbox(group: Dict[str, Any], features: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray] | None:
    face_by_id = {int(f["face_id"]): f for f in features["face_features"]}
    boxes = [face_by_id[int(fid)]["bbox"] for fid in group.get("face_ids", []) if int(fid) in face_by_id]
    if not boxes:
        return None
    arr = np.asarray(boxes, dtype=float)
    return np.min(arr[:, :3], axis=0), np.max(arr[:, 3:], axis=0)


def _nearest_group(source: Dict[str, Any], candidates: Sequence[Dict[str, Any]], features: Dict[str, Any]) -> Dict[str, Any] | None:
    if not candidates:
        return None
    c0 = _group_centroid(source, features)
    return min(candidates, key=lambda item: float(np.linalg.norm(c0 - _group_centroid(item, features))))


def _count_x_tracks(groups: Sequence[Dict[str, Any]], features: Dict[str, Any]) -> int:
    if not groups:
        return 0
    gd = np.asarray(features["global_dims"], dtype=float)
    threshold = max(0.08 * float(gd[0]), 1e-6)
    xs = sorted(float(_group_centroid(group, features)[0]) for group in groups)
    tracks: List[List[float]] = []
    for x in xs:
        for track in tracks:
            if abs(x - float(np.mean(track))) <= threshold:
                track.append(x)
                break
        else:
            tracks.append([x])
    return len(tracks)


def _infer_rib_tracks_from_high_faces(role_map: Dict[int, str], features: Dict[str, Any], part_type: str) -> List[List[int]]:
    if part_type not in {"curved_stiffened_panel", "stiffener_runout_panel"}:
        return []
    faces = features["face_features"]
    if not faces:
        return []
    gd = np.asarray(features["global_dims"], dtype=float)
    gmin = np.asarray(features["global_min"], dtype=float)

    panel_ys = [
        float(face["centroid"][1])
        for face in faces
        if role_map.get(int(face["face_id"])) == "panel"
    ]
    if panel_ys:
        skin_y_max = float(np.percentile(panel_ys, 80))
    else:
        skin_y_max = float(gmin[1] + 0.45 * gd[1])
    margin = max(0.05 * float(gd[1]), 1e-6)

    candidates: List[int] = []
    allowed_roles = {"stiffener", "transition", "runout", "flange", "unassigned"}
    for face in faces:
        fid = int(face["face_id"])
        role = role_map.get(fid, "unassigned")
        if role in {"boundary", "hole", "cutout", "panel"}:
            continue
        if role not in allowed_roles:
            continue
        dims = np.asarray(face["dims"], dtype=float)
        c = np.asarray(face["centroid"], dtype=float)
        long_z = float(dims[2]) >= 0.35 * float(gd[2])
        high_above_skin = float(c[1]) > skin_y_max + margin
        if high_above_skin and long_z:
            candidates.append(fid)

    if not candidates:
        return []
    face_by_id = {int(face["face_id"]): face for face in faces}
    candidates = sorted(candidates, key=lambda fid: float(face_by_id[fid]["centroid"][0]))
    if part_type == "curved_stiffened_panel":
        # A curved rib is represented by several side/cap/transition faces spread in x.
        # A slightly wider x-window keeps those faces as one structural rib track.
        cluster_radius = max(0.14 * float(gd[0]), 1e-6)
        min_cluster_size = 1
    else:
        # Runout count is primarily taken from explicit runout groups; this track
        # estimate remains a conservative diagnostic for high transition faces.
        cluster_radius = max(0.18 * float(gd[0]), 1e-6)
        min_cluster_size = 2
    clusters: List[List[int]] = []
    for fid in candidates:
        x = float(face_by_id[fid]["centroid"][0])
        placed = False
        for cluster in clusters:
            xs = [float(face_by_id[item]["centroid"][0]) for item in cluster]
            if abs(x - float(np.mean(xs))) <= cluster_radius:
                cluster.append(fid)
                placed = True
                break
        if not placed:
            clusters.append([fid])
    return [sorted(cluster) for cluster in clusters if len(cluster) >= min_cluster_size]


def _infer_relations(groups: List[Dict[str, Any]], features: Dict[str, Any], part_type: str = "") -> Tuple[List[Dict[str, str]], int]:
    face_to_group: Dict[int, str] = {}
    role_by_group = {g["node_id"]: g["role"] for g in groups}
    for group in groups:
        for fid in group["face_ids"]:
            face_to_group[int(fid)] = group["node_id"]

    adj = np.asarray(features["face_adjacency"], dtype=int)
    rels = set()
    by_role: Dict[str, List[Dict[str, Any]]] = {}
    for group in groups:
        by_role.setdefault(str(group["role"]), []).append(group)

    stiffeners = by_role.get("stiffener", [])
    panels = by_role.get("panel", [])
    webs = by_role.get("web", [])
    flanges = by_role.get("flange", [])
    caps = by_role.get("cap", [])
    transitions = by_role.get("transition", [])
    holes = by_role.get("hole", [])
    cutouts = by_role.get("cutout", [])
    boundaries = by_role.get("boundary", [])
    runouts = by_role.get("runout", [])

    main_target = panels[0]["node_id"] if panels else webs[0]["node_id"] if webs else None
    for group in boundaries:
        if main_target:
            rels.add((group["node_id"], main_target, "boundary_of"))
    for group in holes:
        if main_target:
            rels.add((group["node_id"], main_target, "hole_of"))
    for group in cutouts:
        if main_target:
            rels.add((group["node_id"], main_target, "cutout_of"))
    for group in stiffeners:
        if panels:
            rels.add((group["node_id"], panels[0]["node_id"], "attached_to"))
    for group in runouts:
        nearest_stiff = _nearest_group(group, stiffeners, features)
        if nearest_stiff is not None:
            rels.add((group["node_id"], nearest_stiff["node_id"], "runout_of"))
        elif panels:
            rels.add((group["node_id"], panels[0]["node_id"], "runout_of"))
        if panels:
            rels.add((group["node_id"], panels[0]["node_id"], "attached_to"))

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
    if not (has_panel and panel_like) and webs:
        web_id = webs[0]["node_id"]
        for group in flanges:
            rels.add((group["node_id"], web_id, "attached_to"))
        for group in caps:
            rels.add((group["node_id"], web_id, "attached_to"))
        if len(flanges) >= 2:
            rels.add((flanges[0]["node_id"], flanges[1]["node_id"], "parallel_to"))

    structural_roles = {"panel", "web", "flange", "cap", "stiffener", "runout"}
    for transition in transitions:
        neighbor_counts: Counter[str] = Counter()
        for fid in transition.get("face_ids", []):
            if int(fid) >= adj.shape[0]:
                continue
            for nb in np.where(adj[int(fid)] > 0)[0].tolist():
                group_id = face_to_group.get(int(nb))
                if not group_id or group_id == transition["node_id"]:
                    continue
                if role_by_group.get(group_id, "") in structural_roles:
                    neighbor_counts[group_id] += 1
        if not neighbor_counts:
            fallback_candidates = [g for role in sorted(structural_roles) for g in by_role.get(role, [])]
            nearest = sorted(fallback_candidates, key=lambda item: float(np.linalg.norm(_group_centroid(transition, features) - _group_centroid(item, features))))[:2]
            for item in nearest:
                rels.add((transition["node_id"], item["node_id"], "smooth_connected"))
        else:
            for group_id, _count in neighbor_counts.most_common(2):
                rels.add((transition["node_id"], group_id, "smooth_connected"))

    valid_nodes = set(role_by_group)
    invalid_relation_count = sum(1 for s, t, _r in rels if s not in valid_nodes or t not in valid_nodes)
    rels = {(s, t, r) for s, t, r in rels if s in valid_nodes and t in valid_nodes}
    return [{"source": s, "target": t, "type": r} for s, t, r in sorted(rels)], invalid_relation_count


def _estimate_parameters(groups: List[Dict[str, Any]], features: Dict[str, Any], part_type: str = "") -> Tuple[Dict[str, float], str]:
    gd = np.asarray(features["global_dims"], dtype=float)
    params = {key: 0.0 for key in PARAMETER_KEYS}
    params["length"] = round(float(gd[2]), 3)
    params["width"] = round(float(gd[0]), 3)
    params["height"] = round(float(gd[1]), 3)

    face_by_id = {int(f["face_id"]): f for f in features["face_features"]}
    beam_like = part_type in {"tapered_c_channel", "tapered_hat_stiffener"}
    if beam_like:
        thin_candidates = []
        for group in groups:
            if group["role"] not in {"web", "flange", "cap"}:
                continue
            for fid in group.get("face_ids", []):
                face = face_by_id.get(int(fid))
                if not face:
                    continue
                dims = [float(v) for v in face.get("dims", []) if float(v) > 1e-4]
                if dims:
                    thin_candidates.append(min(dims))
        if thin_candidates:
            params["thickness"] = round(float(np.median(thin_candidates)), 3)
            thickness_estimation_source = "beam_web_flange_cap_thin_dim"
        else:
            params["thickness"] = round(float(min(gd)), 3)
            thickness_estimation_source = "fallback_global_min_dim"
    else:
        panel_faces = [f for f in features["face_features"] if f["face_id"] in {fid for g in groups if g["role"] == "panel" for fid in g["face_ids"]}]
        thin_dims = [min(f["dims"]) for f in panel_faces if min(f["dims"]) > 1e-4]
        if thin_dims:
            params["thickness"] = round(float(np.median(thin_dims)), 3)
            thickness_estimation_source = "panel_face_thin_dim"
        else:
            params["thickness"] = round(float(min(gd)), 3)
            thickness_estimation_source = "fallback_global_min_dim"

    stiff_groups = [g for g in groups if g["role"] == "stiffener"]
    runout_groups = [g for g in groups if g["role"] == "runout"]
    group_role_map = {
        int(fid): str(group["role"])
        for group in groups
        for fid in group.get("face_ids", [])
    }
    high_rib_tracks = _infer_rib_tracks_from_high_faces(group_role_map, features, part_type)
    runout_track_count = _count_x_tracks(runout_groups, features)
    if part_type == "curved_stiffened_panel":
        params["rib_count"] = float(len(high_rib_tracks) if high_rib_tracks else len(stiff_groups))
    elif part_type == "stiffener_runout_panel":
        params["rib_count"] = float(max(len(stiff_groups), runout_track_count))
    else:
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
    elif runout_groups:
        widths = []
        heights = []
        for group in runout_groups:
            bbox_pair = _group_bbox(group, features)
            if bbox_pair is None:
                continue
            mn, mx = bbox_pair
            widths.append(float(mx[0] - mn[0]))
            heights.append(float(mx[1] - mn[1]))
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
            bbox_pair = _group_bbox(group, features)
            if bbox_pair is not None:
                mn, mx = bbox_pair
                spans.append(mx[0] - mn[0])
        if spans:
            params["flange_width"] = round(float(np.median(spans)), 3)
            params["taper_ratio"] = round(float((max(spans) - min(spans)) / max(max(spans), 1e-6)), 3)

    transition_groups = [g for g in groups if g["role"] == "transition"]
    if transition_groups:
        transition_spans = []
        for group in transition_groups:
            bbox_pair = _group_bbox(group, features)
            if bbox_pair is None:
                continue
            mn, mx = bbox_pair
            dims = mx - mn
            local_dims = [float(v) for v in dims[:2] if float(v) > 1e-6]
            if local_dims:
                transition_spans.append(min(local_dims))
        if transition_spans:
            transition_radius = round(float(np.median(transition_spans)), 3)
            if part_type in {"tapered_c_channel", "tapered_hat_stiffener"}:
                params["fillet_radius"] = transition_radius
            elif "stiffened_panel" in part_type or "runout" in part_type:
                params["root_fillet_radius"] = transition_radius
            elif "rectangular_cutout" in part_type:
                params["cutout_corner_radius"] = transition_radius

    if part_type in {"tapered_c_channel", "tapered_hat_stiffener"}:
        boundary_groups = [g for g in groups if g["role"] == "boundary"]
        start_boxes = [_group_bbox(g, features) for g in boundary_groups if "start" in g.get("node_id", "")]
        end_boxes = [_group_bbox(g, features) for g in boundary_groups if "end" in g.get("node_id", "")]

        def section_span(boxes: Sequence[Tuple[np.ndarray, np.ndarray] | None], axis: int, fallback: float) -> float:
            valid = [box for box in boxes if box is not None]
            if not valid:
                return float(fallback)
            spans = [float(mx[axis] - mn[axis]) for mn, mx in valid]
            return float(np.median(spans)) if spans else float(fallback)

        h_start = section_span(start_boxes, 1, gd[1])
        h_end = section_span(end_boxes, 1, gd[1])
        params["height_start"] = round(h_start, 3)
        params["height_end"] = round(h_end, 3)
        if h_start > 1e-6:
            params["taper_ratio"] = round(float(h_end / h_start), 3)

        if part_type == "tapered_hat_stiffener":
            cap_groups = [g for g in groups if g["role"] == "cap"]
            cap_spans = []
            for group in cap_groups:
                bbox_pair = _group_bbox(group, features)
                if bbox_pair is not None:
                    mn, mx = bbox_pair
                    cap_spans.append(float(mx[0] - mn[0]))
            cap_width = float(np.median(cap_spans)) if cap_spans else max(float(gd[0]) * 0.35, 0.0)
            params["cap_width"] = round(cap_width, 3)
            start_total = section_span(start_boxes, 0, gd[0])
            end_total = section_span(end_boxes, 0, gd[0])
            params["flange_width_start"] = round(max(0.0, 0.5 * (start_total - cap_width)), 3)
            params["flange_width_end"] = round(max(0.0, 0.5 * (end_total - cap_width)), 3)
        else:
            start_total = section_span(start_boxes, 0, params.get("flange_width", gd[0]))
            end_total = section_span(end_boxes, 0, params.get("flange_width", gd[0]))
            half_t = 0.5 * float(params.get("thickness", 0.0))
            params["flange_width_start"] = round(max(0.0, float(start_total) - half_t), 3)
            params["flange_width_end"] = round(max(0.0, float(end_total) - half_t), 3)

    panel_centroids_y = [f["centroid"][1] for f in features["face_features"] if f["face_id"] in {fid for g in groups if g["role"] == "panel" for fid in g["face_ids"]}]
    if panel_centroids_y:
        sag = max(panel_centroids_y) - min(panel_centroids_y)
        if sag > 0.5:
            params["curvature_radius"] = round(float((gd[0] ** 2) / max(8.0 * sag, 1e-6)), 3)

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
    return params, thickness_estimation_source


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
    relations, invalid_relation_count = _infer_relations(groups, features, part_type)
    params, thickness_source = _estimate_parameters(groups, features, part_type)
    mechanisms = _infer_mechanisms(groups, params, part_type)
    return {
        "uid": json_data.get("uid", pkl_data.get("uid", "")),
        "part_type": part_type,
        "inferred_face_roles": {str(fid): role for fid, role in sorted(roles.items())},
        "inferred_face_groups": groups,
        "inferred_config_graph": {"nodes": nodes, "relations": relations},
        "inferred_parameters": params,
        "inferred_topology_mechanisms": mechanisms,
        "inference_diagnostics": {
            "invalid_relation_count": invalid_relation_count,
            "surface_metadata_order_verified": bool(features.get("surface_metadata_order_verified", True)),
            "thickness_estimation_source": thickness_source,
        },
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
    thickness_source_counts: Counter[str] = Counter()
    hole_count = 0
    stiffener_count_sum = 0
    unassigned = 0
    total_faces = 0
    node_counts = []
    relation_count_values = []
    invalid_relation_count = 0

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
            invalid_relation_count += int(result.get("inference_diagnostics", {}).get("invalid_relation_count", 0))
            thickness_source_counts.update([str(result.get("inference_diagnostics", {}).get("thickness_estimation_source", "unknown"))])
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
                    "inference_diagnostics": result["inference_diagnostics"],
                }
            )
            parameter_records.append(
                {
                    "uid": uid,
                    "part_type": result["part_type"],
                    "inferred_parameters": result["inferred_parameters"],
                    "inference_diagnostics": result["inference_diagnostics"],
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
    report.extend(["", "Thickness estimation source counts:"])
    for source, count in sorted(thickness_source_counts.items()):
        report.append(f"  - {source}: {count}")
    report.extend(
        [
            "",
            f"Hole/cutout detected count: {hole_count}",
            f"Accumulated stiffener count: {stiffener_count_sum}",
            f"Average inferred node count: {avg_nodes:.3f}",
            f"Average inferred relation count: {avg_rels:.3f}",
            f"Filtered invalid inferred relations: {invalid_relation_count}",
            "",
            "Boundary statement:",
            "  inferred_Gc is a rule-based weak parse from B-Rep geometry/topology features; it is not human annotation.",
            "  weak_aligned_face_groups are not copied into the inferred result.",
            "  Curved panel roles use bbox/area weak rules when surface metadata order is unverified.",
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
