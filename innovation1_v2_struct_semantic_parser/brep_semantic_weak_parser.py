# -*- coding: utf-8 -*-
"""STEP-to-PKL parser and procedural face-group alignment for innovation1 v2."""

from __future__ import annotations

import os
import pickle
import sys
import warnings
import io
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from graph_inference import _classify_faces, _group_faces
from semantic_feature_extractor import extract_semantic_features
from utils_io import (
    MAX_DIM_LIMITS,
    build_tensor_schema,
    ensure_workdir,
    make_data_splits,
    read_json,
    scan_uid_files,
    summarize_numeric,
    write_csv,
    write_json,
    write_jsonl,
    write_text,
)

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*deprecated function.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from occwl.io import load_step
    from data_process.brep_process import bspline_fitting_local, count_fef_adj, parse_solid

    DTG_AVAILABLE = True
except Exception:  # pragma: no cover - runtime environment report
    DTG_AVAILABLE = False

try:
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.GeomAbs import GeomAbs_Plane
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX
    from OCC.Core.TopExp import TopExp_Explorer, topexp
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape
    from OCC.Core.TopoDS import topods

    OCC_PARSE_AVAILABLE = True
except Exception:  # pragma: no cover
    OCC_PARSE_AVAILABLE = False


def _cleanup_parsed_dir(parsed_dir: str) -> None:
    root = Path(parsed_dir)
    for suffix in ("*.pkl", "*.json", "*.jsonl", "*.csv"):
        for path in root.glob(suffix):
            path.unlink()


def _bbox_for_shape(shape: Any) -> np.ndarray:
    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    try:
        xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    except Exception:
        return np.zeros(6, dtype=np.float32)
    return np.asarray([xmin, ymin, zmin, xmax, ymax, zmax], dtype=np.float32)


def _grid_from_bbox(bbox: np.ndarray, n: int = 32) -> np.ndarray:
    mn = bbox[:3].astype(float)
    mx = bbox[3:].astype(float)
    dims = np.maximum(mx - mn, 0.0)
    normal_axis = int(np.argmin(dims))
    axes = [axis for axis in range(3) if axis != normal_axis]
    a_vals = np.linspace(mn[axes[0]], mx[axes[0]], n)
    b_vals = np.linspace(mn[axes[1]], mx[axes[1]], n)
    grid = np.zeros((n, n, 3), dtype=np.float32)
    const = 0.5 * (mn[normal_axis] + mx[normal_axis])
    for i, av in enumerate(a_vals):
        for j, bv in enumerate(b_vals):
            p = np.zeros(3, dtype=np.float32)
            p[normal_axis] = const
            p[axes[0]] = av
            p[axes[1]] = bv
            grid[i, j] = p
    return grid


def _edge_points_from_bbox(bbox: np.ndarray, n: int = 32) -> np.ndarray:
    p0 = bbox[:3].astype(float)
    p1 = bbox[3:].astype(float)
    if np.linalg.norm(p1 - p0) < 1e-8:
        return np.repeat(p0[None, :], n, axis=0).astype(np.float32)
    t = np.linspace(0.0, 1.0, n)[:, None]
    return ((1 - t) * p0 + t * p1).astype(np.float32)


def _indexed_shapes(shape: Any, shape_type: int) -> Tuple[TopTools_IndexedMapOfShape, List[Any]]:
    indexed = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, shape_type, indexed)
    size = indexed.Size() if hasattr(indexed, "Size") else indexed.Extent()
    items = [indexed.FindKey(i + 1) for i in range(size)]
    return indexed, items


def _surface_metadata_from_shape(shape: Any, expected_face_count: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    _, faces = _indexed_shapes(shape, TopAbs_FACE)
    surface_types: List[int] = []
    curvature_proxy: List[float] = []
    try:
        plane_type = int(GeomAbs_Plane)
    except Exception:
        plane_type = 0
    for face in faces:
        try:
            adaptor = BRepAdaptor_Surface(topods.Face(face), True)
            surf_type = int(adaptor.GetType())
            surface_types.append(surf_type)
            curvature_proxy.append(0.0 if surf_type == plane_type else 1.0)
        except Exception:
            surface_types.append(0)
            curvature_proxy.append(0.0)
    if expected_face_count is not None:
        n = int(expected_face_count)
        if len(surface_types) < n:
            surface_types.extend([0] * (n - len(surface_types)))
            curvature_proxy.extend([0.0] * (n - len(curvature_proxy)))
        surface_types = surface_types[:n]
        curvature_proxy = curvature_proxy[:n]
    return np.asarray(surface_types, dtype=np.int64), np.asarray(curvature_proxy, dtype=np.float32)


def _surface_metadata_from_step(step_path: str, expected_face_count: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    if not OCC_PARSE_AVAILABLE:
        n = int(expected_face_count or 0)
        return np.zeros((n,), dtype=np.int64), np.zeros((n,), dtype=np.float32)
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_path)
    if status != 1:
        n = int(expected_face_count or 0)
        return np.zeros((n,), dtype=np.int64), np.zeros((n,), dtype=np.float32)
    reader.TransferRoots()
    return _surface_metadata_from_shape(reader.OneShape(), expected_face_count)


def _attach_surface_metadata(data: Dict[str, Any], step_path: str) -> Dict[str, Any]:
    face_count = int(data.get("face_count", 0))
    try:
        surface_types, curvature_proxy = _surface_metadata_from_step(step_path, face_count)
        data["face_surface_type"] = surface_types
        data["face_curvature_proxy"] = curvature_proxy
    except Exception:
        data["face_surface_type"] = np.zeros((face_count,), dtype=np.int64)
        data["face_curvature_proxy"] = np.zeros((face_count,), dtype=np.float32)
    return data


def _parse_step_occ_fallback(step_path: str) -> Dict[str, Any]:
    if not OCC_PARSE_AVAILABLE:
        raise RuntimeError("pythonOCC fallback parser is not available.")
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_path)
    if status != 1:
        raise RuntimeError(f"STEPControl_Reader.ReadFile failed with status {status}")
    reader.TransferRoots()
    shape = reader.OneShape()

    face_map, faces = _indexed_shapes(shape, TopAbs_FACE)
    edge_map, edges = _indexed_shapes(shape, TopAbs_EDGE)
    vertex_map, vertices = _indexed_shapes(shape, TopAbs_VERTEX)
    face_count = len(faces)
    edge_count = len(edges)
    vertex_count = len(vertices)

    face_bboxes = np.stack([_bbox_for_shape(face) for face in faces], axis=0) if faces else np.zeros((0, 6), dtype=np.float32)
    edge_bboxes = np.stack([_bbox_for_shape(edge) for edge in edges], axis=0) if edges else np.zeros((0, 6), dtype=np.float32)
    surface_types, curvature_proxy = _surface_metadata_from_shape(shape, face_count)
    face_wcs = np.stack([_grid_from_bbox(bbox) for bbox in face_bboxes], axis=0) if face_count else np.zeros((0, 32, 32, 3), dtype=np.float32)
    edge_wcs = np.stack([_edge_points_from_bbox(bbox) for bbox in edge_bboxes], axis=0) if edge_count else np.zeros((0, 32, 3), dtype=np.float32)

    vert_wcs = []
    for vertex_shape in vertices:
        try:
            pnt = BRep_Tool.Pnt(topods.Vertex(vertex_shape))
            vert_wcs.append([pnt.X(), pnt.Y(), pnt.Z()])
        except Exception:
            bbox = _bbox_for_shape(vertex_shape)
            vert_wcs.append((0.5 * (bbox[:3] + bbox[3:])).tolist())
    vert_wcs_arr = np.asarray(vert_wcs, dtype=np.float32) if vert_wcs else np.zeros((0, 3), dtype=np.float32)

    face_edge_adj: List[List[int]] = []
    edge_faces: List[List[int]] = [[] for _ in range(edge_count)]
    for fidx, face in enumerate(faces):
        edge_ids: List[int] = []
        explorer = TopExp_Explorer(face, TopAbs_EDGE)
        while explorer.More():
            edge_idx = edge_map.FindIndex(explorer.Current()) - 1
            if edge_idx >= 0 and edge_idx not in edge_ids:
                edge_ids.append(edge_idx)
                edge_faces[edge_idx].append(fidx)
            explorer.Next()
        face_edge_adj.append(edge_ids)

    edge_face_adj = -np.ones((edge_count, 2), dtype=np.int64)
    for eidx, face_ids in enumerate(edge_faces):
        for col, fid in enumerate(face_ids[:2]):
            edge_face_adj[eidx, col] = int(fid)

    edge_vert_adj = -np.ones((edge_count, 2), dtype=np.int64)
    for eidx, edge in enumerate(edges):
        vids: List[int] = []
        explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
        while explorer.More():
            vid = vertex_map.FindIndex(explorer.Current()) - 1
            if vid >= 0 and vid not in vids:
                vids.append(vid)
            explorer.Next()
        for col, vid in enumerate(vids[:2]):
            edge_vert_adj[eidx, col] = int(vid)

    return {
        "face_bbox_wcs": face_bboxes,
        "face_surface_type": surface_types,
        "face_curvature_proxy": curvature_proxy,
        "edge_bbox_wcs": edge_bboxes,
        "vert_wcs": vert_wcs_arr,
        "face_wcs": face_wcs,
        "edge_wcs": edge_wcs,
        "edgeFace_adj": edge_face_adj,
        "edgeVert_adj": edge_vert_adj,
        "faceEdge_adj": face_edge_adj,
        "face_count": face_count,
        "edge_count": edge_count,
        "vertex_count": vertex_count,
        "parser_backend": "pythonocc_fallback",
    }


def _ensure_minimal_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    if "face_bbox_wcs" not in data and "face_wcs" in data:
        face_wcs = np.asarray(data["face_wcs"], dtype=np.float32)
        data["face_bbox_wcs"] = np.asarray(
            [np.concatenate([np.min(face.reshape(-1, 3), axis=0), np.max(face.reshape(-1, 3), axis=0)]) for face in face_wcs],
            dtype=np.float32,
        )
    if "edge_bbox_wcs" not in data and "edge_wcs" in data:
        edge_wcs = np.asarray(data["edge_wcs"], dtype=np.float32)
        data["edge_bbox_wcs"] = np.asarray(
            [np.concatenate([np.min(edge.reshape(-1, 3), axis=0), np.max(edge.reshape(-1, 3), axis=0)]) for edge in edge_wcs],
            dtype=np.float32,
        )
    if "face_wcs" not in data:
        bboxes = np.asarray(data.get("face_bbox_wcs", np.zeros((0, 6))), dtype=np.float32)
        data["face_wcs"] = np.stack([_grid_from_bbox(bbox) for bbox in bboxes], axis=0) if len(bboxes) else np.zeros((0, 32, 32, 3), dtype=np.float32)
    if "edge_wcs" not in data:
        bboxes = np.asarray(data.get("edge_bbox_wcs", np.zeros((0, 6))), dtype=np.float32)
        data["edge_wcs"] = np.stack([_edge_points_from_bbox(bbox) for bbox in bboxes], axis=0) if len(bboxes) else np.zeros((0, 32, 3), dtype=np.float32)
    data["face_count"] = int(data.get("face_count", len(data.get("face_bbox_wcs", []))))
    data["edge_count"] = int(data.get("edge_count", len(data.get("edge_bbox_wcs", []))))
    data["vertex_count"] = int(data.get("vertex_count", len(data.get("vert_wcs", []))))
    if "face_surface_type" not in data:
        data["face_surface_type"] = np.zeros((data["face_count"],), dtype=np.int64)
    if "face_curvature_proxy" not in data:
        data["face_curvature_proxy"] = np.zeros((data["face_count"],), dtype=np.float32)
    return data


def _parse_step_dtg(step_path: str) -> Dict[str, Any]:
    if not DTG_AVAILABLE:
        raise RuntimeError("DTG parser is not available.")
    with warnings.catch_warnings(), redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        solids = load_step(step_path)
        if len(solids) != 1:
            raise ValueError(f"STEP must contain exactly one solid, found {len(solids)}")
        data = parse_solid(solids[0])
        if data is None:
            raise ValueError("DTG parse_solid returned None")
        if "faceEdge_adj" in data:
            try:
                data["fef_adj"] = count_fef_adj(data["faceEdge_adj"])
            except Exception:
                pass
        try:
            data = bspline_fitting_local(data)
        except Exception:
            pass
    data["parser_backend"] = "dtg_occwl"
    return _ensure_minimal_fields(data)


def _parse_step(step_path: str) -> Dict[str, Any]:
    errors = []
    try:
        data = _parse_step_dtg(step_path)
        return _attach_surface_metadata(data, step_path)
    except Exception as exc:
        errors.append(f"DTG: {exc}")
    try:
        data = _parse_step_occ_fallback(step_path)
        data["parser_warning"] = " | ".join(errors)
        return _ensure_minimal_fields(data)
    except Exception as exc:
        errors.append(f"pythonOCC fallback: {exc}")
        raise RuntimeError(" ; ".join(errors))


def _align_procedural_face_groups(json_data: Dict[str, Any], pkl_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    features = extract_semantic_features(pkl_data)
    part_type = str(json_data.get("part_type", ""))
    roles = _classify_faces(features, part_type)
    weak_groups = _group_faces(roles, features)
    role_groups: Dict[str, List[Dict[str, Any]]] = {}
    for group in weak_groups:
        role_groups.setdefault(group["role"], []).append(group)

    aligned = []
    role_cursors = {role: 0 for role in role_groups}
    for node in json_data.get("configuration_graph", {}).get("nodes", []):
        role = str(node.get("type", "unassigned"))
        node_id = str(node.get("id", role))
        face_ids: List[int] = []
        if role == "panel":
            face_ids = sorted({fid for group in role_groups.get(role, []) for fid in group["face_ids"]})
        elif role == "boundary" and node_id.endswith("start"):
            starts = [group for group in role_groups.get("boundary", []) if "start" in group["node_id"]]
            face_ids = sorted({fid for group in starts for fid in group["face_ids"]})
        elif role == "boundary" and node_id.endswith("end"):
            ends = [group for group in role_groups.get("boundary", []) if "end" in group["node_id"]]
            face_ids = sorted({fid for group in ends for fid in group["face_ids"]})
        else:
            groups = role_groups.get(role, [])
            cursor = role_cursors.get(role, 0)
            if groups:
                face_ids = list(groups[min(cursor, len(groups) - 1)]["face_ids"])
                role_cursors[role] = cursor + 1
        aligned.append(
            {
                "node_id": node_id,
                "role": role,
                "face_ids": sorted(int(x) for x in face_ids),
                "assignment_status": "aligned_by_v2_geometry_rule" if face_ids else "empty_or_not_detected",
            }
        )
    return aligned


def parse_enhanced_dataset(workdir: str, seed: int = 42) -> Dict[str, Any]:
    dirs = ensure_workdir(workdir)
    dataset_dir = dirs["enhanced_dataset"]
    parsed_dir = dirs["enhanced_parsed"]
    reports_dir = dirs["reports"]
    _cleanup_parsed_dir(parsed_dir)

    uids = scan_uid_files(dataset_dir, ".step")
    records: List[Dict[str, Any]] = []
    face_group_records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    parsed_uids: List[str] = []

    for uid in uids:
        step_path = os.path.join(dataset_dir, f"{uid}.step")
        json_path = os.path.join(dataset_dir, f"{uid}.json")
        pkl_path = os.path.join(parsed_dir, f"{uid}.pkl")
        part_type = "unknown"
        try:
            jd = read_json(json_path)
            part_type = str(jd.get("part_type", "unknown"))
            data = _parse_step(step_path)
            data["uid"] = uid
            data["part_type"] = part_type
            data["parameters"] = jd.get("parameters", {})
            data["configuration_graph"] = jd.get("configuration_graph", {})
            data["source_step"] = f"{uid}.step"
            data["source_stl"] = f"{uid}.stl"
            data["source_json"] = f"{uid}.json"
            data = _ensure_minimal_fields(data)

            aligned_groups = _align_procedural_face_groups(jd, data)
            jd["procedural_face_groups"] = aligned_groups
            jd["procedural_face_group_alignment_note"] = (
                "Face ids are assigned during parse_enhanced by geometry-rule alignment to procedural Gc nodes."
            )
            write_json(json_path, jd)
            face_group_records.append({"uid": uid, "part_type": part_type, "face_groups": aligned_groups})

            with open(pkl_path, "wb") as f:
                pickle.dump(data, f)
            parsed_uids.append(uid)
            face_count = int(data["face_count"])
            edge_count = int(data["edge_count"])
            vertex_count = int(data["vertex_count"])
            records.append(
                {
                    "uid": uid,
                    "part_type": part_type,
                    "parse_status": "SUCCESS",
                    "parser_backend": data.get("parser_backend", "unknown"),
                    "face_count": face_count,
                    "edge_count": edge_count,
                    "vertex_count": vertex_count,
                    "exceeds_max_faces_30": int(face_count > MAX_DIM_LIMITS["max_faces"]),
                    "exceeds_max_edges_68": int(edge_count > MAX_DIM_LIMITS["max_edges"]),
                    "exceeds_max_vertices_40": int(vertex_count > MAX_DIM_LIMITS["max_vertices"]),
                    "error": data.get("parser_warning", ""),
                }
            )
        except Exception as exc:
            failures.append({"uid": uid, "part_type": part_type, "error": str(exc)})
            records.append(
                {
                    "uid": uid,
                    "part_type": part_type,
                    "parse_status": "FAILED",
                    "parser_backend": "none",
                    "face_count": 0,
                    "edge_count": 0,
                    "vertex_count": 0,
                    "exceeds_max_faces_30": 0,
                    "exceeds_max_edges_68": 0,
                    "exceeds_max_vertices_40": 0,
                    "error": str(exc),
                }
            )

    write_jsonl(os.path.join(parsed_dir, "face_group_index.jsonl"), face_group_records)
    write_json(os.path.join(parsed_dir, "tensor_schema.json"), build_tensor_schema())
    write_csv(os.path.join(parsed_dir, "data_splits.csv"), make_data_splits(parsed_uids, seed), ["uid", "split"])
    write_csv(
        os.path.join(reports_dir, "auxiliary", "enhanced_brep_field_stats.csv"),
        records,
        [
            "uid",
            "part_type",
            "parse_status",
            "parser_backend",
            "face_count",
            "edge_count",
            "vertex_count",
            "exceeds_max_faces_30",
            "exceeds_max_edges_68",
            "exceeds_max_vertices_40",
            "error",
        ],
    )
    write_csv(
        os.path.join(parsed_dir, "parse_manifest.csv"),
        records,
        [
            "uid",
            "part_type",
            "parse_status",
            "parser_backend",
            "face_count",
            "edge_count",
            "vertex_count",
            "error",
        ],
    )

    good = [r for r in records if r["parse_status"] == "SUCCESS"]
    lite_records = [
        r
        for r in good
        if not (r["exceeds_max_faces_30"] or r["exceeds_max_edges_68"] or r["exceeds_max_vertices_40"])
    ]
    lite_uids = [str(r["uid"]) for r in lite_records]
    lite_uid_set = set(lite_uids)
    lite_face_group_records = [r for r in face_group_records if str(r.get("uid")) in lite_uid_set]
    write_csv(
        os.path.join(reports_dir, "auxiliary", "enhanced_lite_manifest.csv"),
        lite_records,
        [
            "uid",
            "part_type",
            "parse_status",
            "parser_backend",
            "face_count",
            "edge_count",
            "vertex_count",
            "exceeds_max_faces_30",
            "exceeds_max_edges_68",
            "exceeds_max_vertices_40",
            "error",
        ],
    )
    write_text(os.path.join(parsed_dir, "enhanced_lite_uids.txt"), lite_uids)
    write_csv(os.path.join(parsed_dir, "data_splits_lite.csv"), make_data_splits(lite_uids, seed), ["uid", "split"])
    write_jsonl(os.path.join(parsed_dir, "face_group_index_lite.jsonl"), lite_face_group_records)

    faces = [float(r["face_count"]) for r in good]
    edges = [float(r["edge_count"]) for r in good]
    verts = [float(r["vertex_count"]) for r in good]
    over_limit = [
        r
        for r in good
        if r["exceeds_max_faces_30"] or r["exceeds_max_edges_68"] or r["exceeds_max_vertices_40"]
    ]
    face_stats = summarize_numeric(faces)
    edge_stats = summarize_numeric(edges)
    vert_stats = summarize_numeric(verts)
    report = [
        "Innovation1 v2 Enhanced B-Rep Parse Report",
        "=" * 72,
        f"STEP files scanned: {len(uids)}",
        f"Parse success: {len(good)}",
        f"Parse failures: {len(failures)}",
        f"Success rate: {len(good) / max(len(uids), 1):.4f}",
        "",
        f"Face count min/mean/max: {face_stats['min']} / {face_stats['mean']} / {face_stats['max']}",
        f"Edge count min/mean/max: {edge_stats['min']} / {edge_stats['mean']} / {edge_stats['max']}",
        f"Vertex count min/mean/max: {vert_stats['min']} / {vert_stats['mean']} / {vert_stats['max']}",
        "",
        "Innovation2 current limits: max_faces=30, max_edges=68, max_vertices=40",
        f"Samples exceeding at least one current limit: {len(over_limit)}",
        f"Enhanced-lite samples within current limits: {len(lite_records)}",
    ]
    if over_limit:
        report.append("Exceeded samples:")
        for item in over_limit[:100]:
            report.append(
                f"  - {item['uid']}: faces={item['face_count']}, edges={item['edge_count']}, vertices={item['vertex_count']}"
            )
    if failures:
        report.extend(["", "Parse failures:"])
        for item in failures[:100]:
            report.append(f"  - {item['uid']}: {item['error']}")
    report.extend(
        [
            "",
            "Parser policy:",
            "  DTG/occwl parse_solid is tried first.",
            "  If DTG parsing fails, a pythonOCC compatibility parser extracts bbox/grid/adjaency fields.",
            "  Errors and dimension-limit exceedances are reported explicitly.",
            "  enhanced_lite_uids.txt, data_splits_lite.csv and face_group_index_lite.jsonl provide a current-limit subset for innovation2.",
        ]
    )
    write_text(os.path.join(reports_dir, "enhanced_parse_report.txt"), report)
    return {"records": records, "failures": failures, "parsed_uids": parsed_uids, "lite_uids": lite_uids}
