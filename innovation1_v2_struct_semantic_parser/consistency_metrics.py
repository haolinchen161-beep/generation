# -*- coding: utf-8 -*-
"""Consistency metrics between procedural_Gc and inferred_Gc."""

from __future__ import annotations

import math
import os
from collections import Counter, defaultdict
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from utils_io import PARAMETER_KEYS, ensure_workdir, read_json, read_jsonl, scan_uid_files, write_csv, write_text


PARAM_RANGES = {
    "length": 500.0,
    "width": 300.0,
    "thickness": 10.0,
    "height": 140.0,
    "flange_width": 120.0,
    "rib_width": 60.0,
    "rib_height": 120.0,
    "rib_count": 8.0,
    "fillet_radius": 30.0,
    "root_fillet_radius": 30.0,
    "cutout_corner_radius": 30.0,
    "hole_radius": 80.0,
    "hole_width": 120.0,
    "hole_height": 160.0,
    "hole_count": 6.0,
    "taper_ratio": 1.0,
    "curvature_radius": 1000.0,
    "sweep_angle": 40.0,
    "runout_length": 180.0,
    "notch_depth": 40.0,
    "offset_ratio": 0.5,
    "height_start": 150.0,
    "height_end": 150.0,
    "flange_width_start": 100.0,
    "flange_width_end": 100.0,
    "cap_width": 150.0,
}

CLASS_MEANINGFUL_PARAMS = {
    "panel_with_circular_cutout": ["length", "width", "thickness", "hole_radius", "hole_count"],
    "panel_with_rectangular_cutout": [
        "length",
        "width",
        "thickness",
        "hole_width",
        "hole_height",
        "cutout_corner_radius",
    ],
    "stiffened_panel_with_cutout": [
        "length",
        "width",
        "thickness",
        "height",
        "rib_width",
        "rib_height",
        "rib_count",
        "hole_radius",
        "hole_count",
        "root_fillet_radius",
    ],
    "multi_stiffened_panel": [
        "length",
        "width",
        "thickness",
        "height",
        "rib_width",
        "rib_height",
        "rib_count",
        "root_fillet_radius",
    ],
    "asymmetric_stiffened_panel": [
        "length",
        "width",
        "thickness",
        "height",
        "rib_width",
        "rib_height",
        "rib_count",
        "offset_ratio",
        "root_fillet_radius",
    ],
    "tapered_c_channel": [
        "length",
        "thickness",
        "height_start",
        "height_end",
        "flange_width_start",
        "flange_width_end",
        "fillet_radius",
    ],
    "tapered_hat_stiffener": [
        "length",
        "thickness",
        "height_start",
        "height_end",
        "flange_width_start",
        "flange_width_end",
        "cap_width",
        "fillet_radius",
    ],
    "curved_panel": ["length", "width", "thickness", "curvature_radius", "sweep_angle"],
    "curved_stiffened_panel": [
        "length",
        "width",
        "thickness",
        "height",
        "curvature_radius",
        "sweep_angle",
        "rib_width",
        "rib_height",
        "rib_count",
        "root_fillet_radius",
    ],
    "stiffener_runout_panel": [
        "length",
        "width",
        "thickness",
        "height",
        "rib_width",
        "rib_height",
        "rib_count",
        "runout_length",
        "root_fillet_radius",
    ],
}

DIMENSIONAL_PARAMS = {
    "length",
    "width",
    "thickness",
    "height",
    "flange_width",
    "rib_width",
    "rib_height",
    "fillet_radius",
    "root_fillet_radius",
    "cutout_corner_radius",
    "hole_radius",
    "hole_width",
    "hole_height",
    "curvature_radius",
    "runout_length",
    "notch_depth",
    "height_start",
    "height_end",
    "flange_width_start",
    "flange_width_end",
    "cap_width",
}


def _parameter_scale(params: Dict[str, Any], keys: Sequence[str]) -> float:
    candidates = [float(params.get("length", 0.0) or 0.0)]
    candidates.extend(float(params.get(key, 0.0) or 0.0) for key in keys if key in DIMENSIONAL_PARAMS)
    return max(max(abs(v) for v in candidates), 1e-6)


def _load_jsonl_map(path: str, key: str = "uid") -> Dict[str, Dict[str, Any]]:
    return {str(item.get(key)): item for item in read_jsonl(path)}


def _procedural_role_map(groups: Sequence[Dict[str, Any]]) -> Dict[int, str]:
    role_map: Dict[int, str] = {}
    for group in groups:
        role = str(group.get("role", "unassigned"))
        for fid in group.get("face_ids", []):
            role_map[int(fid)] = role
    return role_map


def _relation_type_counter(graph: Dict[str, Any]) -> Counter[str]:
    return Counter(str(rel.get("type", "")) for rel in graph.get("relations", []))


def _node_type_counter(graph: Dict[str, Any]) -> Counter[str]:
    return Counter(str(node.get("type", "")) for node in graph.get("nodes", []))


def _relation_triplet_counter(graph: Dict[str, Any]) -> Counter[Tuple[str, str, str]]:
    node_type = {str(node.get("id", "")): str(node.get("type", "")) for node in graph.get("nodes", [])}
    counter: Counter[Tuple[str, str, str]] = Counter()
    for rel in graph.get("relations", []):
        src = str(rel.get("source", rel.get("src", "")))
        dst = str(rel.get("target", rel.get("dst", "")))
        typ = str(rel.get("type", rel.get("relation", "")))
        counter[(node_type.get(src, "unknown"), node_type.get(dst, "unknown"), typ)] += 1
    return counter


def _counter_overlap_acc(a: Counter[str], b: Counter[str]) -> float:
    total = max(sum(a.values()), sum(b.values()), 1)
    overlap = sum(min(a[k], b[k]) for k in set(a) | set(b))
    return float(overlap / total)


def _weak_face_role_consistency(weak_roles: Dict[int, str], inferred_roles: Dict[str, str]) -> float:
    if not weak_roles:
        return 0.0
    correct = 0
    total = 0
    for fid, weak_role in weak_roles.items():
        inf_role = inferred_roles.get(str(fid), "unassigned")
        if inf_role == weak_role:
            correct += 1
        total += 1
    return float(correct / max(total, 1))


def _assign_ratio(inferred_roles: Dict[str, str]) -> float:
    if not inferred_roles:
        return 0.0
    assigned = sum(1 for role in inferred_roles.values() if role != "unassigned")
    return float(assigned / len(inferred_roles))


def _weak_face_group_iou(weak_groups: Sequence[Dict[str, Any]], inf_groups: Sequence[Dict[str, Any]]) -> float:
    scores = []
    for pg in weak_groups:
        pset = set(int(x) for x in pg.get("face_ids", []))
        if not pset:
            continue
        role = str(pg.get("role", ""))
        best = 0.0
        for ig in inf_groups:
            if str(ig.get("role", "")) != role:
                continue
            iset = set(int(x) for x in ig.get("face_ids", []))
            if not iset:
                continue
            inter = len(pset & iset)
            union = len(pset | iset)
            best = max(best, inter / max(union, 1))
        scores.append(best)
    return float(mean(scores)) if scores else 0.0


def _parameter_l1_scale_normalized(proc_params: Dict[str, Any], inf_params: Dict[str, Any], part_type: str) -> float:
    vals = []
    keys = CLASS_MEANINGFUL_PARAMS.get(part_type, PARAMETER_KEYS)
    proc_scale = _parameter_scale(proc_params, keys)
    inf_scale = _parameter_scale(inf_params, keys)
    for key in keys:
        pa = float(proc_params.get(key, 0.0) or 0.0)
        ib = float(inf_params.get(key, 0.0) or 0.0)
        if key in DIMENSIONAL_PARAMS:
            vals.append(abs(pa / proc_scale - ib / inf_scale))
        else:
            vals.append(abs(pa - ib) / PARAM_RANGES.get(key, 1.0))
    return float(mean(vals)) if vals else 0.0


def _parameter_l1_abs_mm(proc_params: Dict[str, Any], inf_params: Dict[str, Any], part_type: str) -> float:
    vals = []
    keys = CLASS_MEANINGFUL_PARAMS.get(part_type, PARAMETER_KEYS)
    for key in keys:
        pa = float(proc_params.get(key, 0.0) or 0.0)
        ib = float(inf_params.get(key, 0.0) or 0.0)
        vals.append(abs(pa - ib) / PARAM_RANGES.get(key, 1.0))
    return float(mean(vals)) if vals else 0.0


def _hole_detection_acc(proc_params: Dict[str, Any], inf_params: Dict[str, Any]) -> float:
    proc_count = int(round(float(proc_params.get("hole_count", 0.0) or 0.0)))
    inf_count = int(round(float(inf_params.get("hole_count", 0.0) or 0.0)))
    return 1.0 if proc_count == inf_count else 0.0


def _stiffener_count_acc(proc_params: Dict[str, Any], inf_params: Dict[str, Any], part_type: str) -> float:
    if part_type not in {
        "stiffened_panel_with_cutout",
        "multi_stiffened_panel",
        "asymmetric_stiffened_panel",
        "curved_stiffened_panel",
        "stiffener_runout_panel",
    }:
        return 1.0
    proc_count = int(round(float(proc_params.get("rib_count", 0.0) or 0.0)))
    inf_count = int(round(float(inf_params.get("rib_count", 0.0) or 0.0)))
    return 1.0 if proc_count == inf_count else 0.0


def _mechanism_acc(proc_mechs: Sequence[str], inf_mechs: Sequence[str]) -> float:
    a = set(proc_mechs)
    b = set(inf_mechs)
    if not a and not b:
        return 1.0
    return float(len(a & b) / max(len(a | b), 1))


def _failure_reasons(row: Dict[str, Any], proc_mechs: Sequence[str]) -> List[str]:
    reasons = []
    if ("hole" in proc_mechs or "cutout" in proc_mechs or "inner_loop" in proc_mechs) and row["hole_detection_acc"] < 1.0:
        reasons.append("hole boundary/cutout loop mis-detected")
    if row["weak_face_role_consistency"] < 0.65:
        reasons.append("transition or panel/web/flange face role confusion")
    if row["stiffener_count_acc"] < 1.0:
        reasons.append("stiffener grouping/count mismatch")
    if "curved_surface" in proc_mechs and row["weak_face_role_consistency"] < 0.8:
        reasons.append("curved surface normal/bbox judgement unstable")
    if row["assign_ratio"] < 0.85:
        reasons.append("boundary or small faces left unassigned")
    return reasons or ["no dominant failure"]


def _average_rows(rows: Sequence[Dict[str, Any]], part_type: str) -> Dict[str, Any]:
    metrics = [
        "weak_face_role_consistency",
        "weak_face_group_iou",
        "assign_ratio",
        "node_type_count_consistency",
        "relation_type_count_consistency",
        "relation_triplet_overlap",
        "parameter_l1_scale_normalized",
        "parameter_l1_abs_mm",
        "hole_detection_acc",
        "stiffener_count_acc",
        "topology_mechanism_acc",
    ]
    out: Dict[str, Any] = {"dataset_family": "enhanced", "part_type": part_type, "sample_count": len(rows)}
    for key in metrics:
        vals = [float(r[key]) for r in rows]
        out[key] = round(mean(vals), 5) if vals else 0.0
    reason_counter: Counter[str] = Counter()
    for row in rows:
        for reason in str(row.get("failure_reasons", "")).split(";"):
            reason = reason.strip()
            if reason:
                reason_counter[reason] += 1
    out["major_failure_reasons"] = "; ".join(f"{k}({v})" for k, v in reason_counter.most_common(4))
    return out


def evaluate_consistency(workdir: str) -> Dict[str, Any]:
    dirs = ensure_workdir(workdir)
    dataset_dir = dirs["enhanced_dataset"]
    inferred_dir = dirs["inferred_semantics"]
    reports_dir = dirs["reports"]

    face_map = _load_jsonl_map(os.path.join(inferred_dir, "inferred_face_groups.jsonl"))
    graph_map = _load_jsonl_map(os.path.join(inferred_dir, "inferred_config_graphs.jsonl"))
    param_map = _load_jsonl_map(os.path.join(inferred_dir, "inferred_parameters.jsonl"))
    uids = scan_uid_files(dataset_dir, ".json")
    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for uid in uids:
        if uid not in face_map or uid not in graph_map or uid not in param_map:
            skipped.append(uid)
            continue
        jd = read_json(os.path.join(dataset_dir, f"{uid}.json"))
        proc_graph = jd.get("procedural_config_graph", jd.get("configuration_graph", {}))
        weak_groups = jd.get("weak_aligned_face_groups", [])
        proc_params = jd.get("parameters", {})
        proc_mechs = jd.get("topology_mechanisms", [])

        inf_roles = face_map[uid].get("inferred_face_roles", {})
        inf_groups = face_map[uid].get("inferred_face_groups", [])
        inf_graph = graph_map[uid].get("inferred_config_graph", {})
        inf_params = param_map[uid].get("inferred_parameters", {})
        inf_mechs = graph_map[uid].get("inferred_topology_mechanisms", [])

        weak_roles = _procedural_role_map(weak_groups)
        row = {
            "uid": uid,
            "dataset_family": "enhanced",
            "part_type": jd.get("part_type", "unknown"),
            "weak_face_role_consistency": _weak_face_role_consistency(weak_roles, inf_roles),
            "weak_face_group_iou": _weak_face_group_iou(weak_groups, inf_groups),
            "assign_ratio": _assign_ratio(inf_roles),
            "node_type_count_consistency": _counter_overlap_acc(_node_type_counter(proc_graph), _node_type_counter(inf_graph)),
            "relation_type_count_consistency": _counter_overlap_acc(_relation_type_counter(proc_graph), _relation_type_counter(inf_graph)),
            "relation_triplet_overlap": _counter_overlap_acc(_relation_triplet_counter(proc_graph), _relation_triplet_counter(inf_graph)),
            "parameter_l1_scale_normalized": _parameter_l1_scale_normalized(proc_params, inf_params, str(jd.get("part_type", "unknown"))),
            "parameter_l1_abs_mm": _parameter_l1_abs_mm(proc_params, inf_params, str(jd.get("part_type", "unknown"))),
            "hole_detection_acc": _hole_detection_acc(proc_params, inf_params),
            "stiffener_count_acc": _stiffener_count_acc(proc_params, inf_params, str(jd.get("part_type", "unknown"))),
            "topology_mechanism_acc": _mechanism_acc(proc_mechs, inf_mechs),
        }
        row["failure_reasons"] = "; ".join(_failure_reasons(row, proc_mechs))
        rows.append(row)

    fieldnames = [
        "uid",
        "dataset_family",
        "part_type",
        "weak_face_role_consistency",
        "weak_face_group_iou",
        "assign_ratio",
        "node_type_count_consistency",
        "relation_type_count_consistency",
        "relation_triplet_overlap",
        "parameter_l1_scale_normalized",
        "parameter_l1_abs_mm",
        "hole_detection_acc",
        "stiffener_count_acc",
        "topology_mechanism_acc",
        "failure_reasons",
    ]
    write_csv(os.path.join(reports_dir, "auxiliary", "procedural_vs_inferred_consistency.csv"), rows, fieldnames)

    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row["part_type"])].append(row)
    summary_rows = [_average_rows(rows, "ALL")] if rows else []
    for part_type in sorted(by_type):
        summary_rows.append(_average_rows(by_type[part_type], part_type))

    report = [
        "Innovation1 v2 Procedural_Gc vs Inferred_Gc Consistency Report",
        "=" * 72,
        f"Evaluated samples: {len(rows)}",
        f"Skipped samples without inferred semantics: {len(skipped)}",
        "",
        "Important interpretation:",
        "  inferred_Gc is a weak rule-based parse from B-Rep geometry/topology, not human ground truth.",
        "  procedural_Gc is the structural supervision label synchronized during procedural enhanced generation.",
        "  Face-level consistency is measured against weak_aligned_face_groups, not human/manual face-level ground truth.",
        "  parameter_l1_scale_normalized is the main scale-aware comparison for DTG-standardized coordinates.",
        "  parameter_l1_abs_mm is only a diagnostic absolute-size comparison when parsed coordinates preserve STEP mm scale.",
        "",
        "Summary by part_type:",
    ]
    for item in summary_rows:
        report.append(
            "  - {part_type}: n={sample_count}, weak_role_cons={weak_face_role_consistency:.4f}, weak_group_iou={weak_face_group_iou:.4f}, "
            "assign={assign_ratio:.4f}, node_count_cons={node_type_count_consistency:.4f}, rel_count_cons={relation_type_count_consistency:.4f}, "
            "rel_triplet={relation_triplet_overlap:.4f}, "
            "param_l1_scale={parameter_l1_scale_normalized:.4f}, param_l1_abs_mm={parameter_l1_abs_mm:.4f}, "
            "hole_acc={hole_detection_acc:.4f}, stiffener_acc={stiffener_count_acc:.4f}, "
            "mech_acc={topology_mechanism_acc:.4f}".format(**item)
        )
        if item.get("major_failure_reasons"):
            report.append(f"    main failures: {item['major_failure_reasons']}")
    if skipped:
        report.extend(["", "Skipped samples:"])
        for uid in skipped[:100]:
            report.append(f"  - {uid}")
    report.extend(
        [
            "",
            "Failure modes tracked:",
            "  hole boundary mis-detection; transition face confusion; stiffener grouping failure;",
            "  curved surface normal instability; boundary face unassignment.",
        ]
    )
    write_text(os.path.join(reports_dir, "semantic_consistency_report.txt"), report)
    return {"rows": rows, "summary_rows": summary_rows, "skipped": skipped}
