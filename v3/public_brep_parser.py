# -*- coding: utf-8 -*-
"""Parse public no-semantic STEP B-Reps into DTG-compatible tensors."""

from __future__ import annotations

import io
import os
import pickle
import sys
import warnings
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from utils_io import (
    clean_dir,
    detect_source_name,
    ensure_workdir,
    scan_step_files,
    short_hash,
    timestamp,
    uid_from_step,
    write_csv,
    write_text,
)

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from occwl.io import load_step
    from data_process.brep_process import bspline_fitting_local, count_fef_adj, parse_solid

    DTG_AVAILABLE = True
except Exception:
    DTG_AVAILABLE = False

try:
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.GeomAbs import GeomAbs_Plane
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID, TopAbs_VERTEX
    from OCC.Core.TopExp import TopExp_Explorer, topexp
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape
    from OCC.Core.TopoDS import topods

    OCC_PARSE_AVAILABLE = True
except Exception:
    OCC_PARSE_AVAILABLE = False


MAX_FACES = 70


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
    return ((1.0 - t) * p0 + t * p1).astype(np.float32)


def _indexed_shapes(shape: Any, shape_type: int) -> Tuple[TopTools_IndexedMapOfShape, List[Any]]:
    indexed = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, shape_type, indexed)
    size = indexed.Size() if hasattr(indexed, "Size") else indexed.Extent()
    items = [indexed.FindKey(i + 1) for i in range(size)]
    return indexed, items


def _surface_metadata_from_shape(shape: Any, expected_face_count: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    _, faces = _indexed_shapes(shape, TopAbs_FACE)
    try:
        plane_type = int(GeomAbs_Plane)
    except Exception:
        plane_type = 0
    surface_types: List[int] = []
    curvature_proxy: List[float] = []
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
    if reader.ReadFile(step_path) != 1:
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
        data["surface_metadata_order_verified"] = str(data.get("parser_backend", "")) != "dtg_occwl"
    except Exception:
        data["face_surface_type"] = np.zeros((face_count,), dtype=np.int64)
        data["face_curvature_proxy"] = np.zeros((face_count,), dtype=np.float32)
        data["surface_metadata_order_verified"] = False
    return data


def _parse_step_dtg(step_path: str) -> Dict[str, Any]:
    if not DTG_AVAILABLE:
        raise RuntimeError("DTG parser is not available.")
    with warnings.catch_warnings(), redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        solids = load_step(step_path)
        if len(solids) != 1:
            raise ValueError(f"single_solid_required_found_{len(solids)}")
        data = parse_solid(solids[0])
        if data is None:
            raise ValueError("parse_solid_returned_none")
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
    data["geometry_sampling_quality"] = "true_or_dtg_sampling"
    data["surface_metadata_order_verified"] = False
    return _ensure_minimal_fields(data)


def _parse_step_occ_fallback(step_path: str) -> Dict[str, Any]:
    if not OCC_PARSE_AVAILABLE:
        raise RuntimeError("pythonOCC fallback parser is not available.")
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_path)
    if status != 1:
        raise RuntimeError(f"STEPControl_Reader_failed_status_{status}")
    reader.TransferRoots()
    shape = reader.OneShape()

    _, solids = _indexed_shapes(shape, TopAbs_SOLID)
    if len(solids) != 1:
        raise ValueError(f"single_solid_required_found_{len(solids)}")

    face_map, faces = _indexed_shapes(shape, TopAbs_FACE)
    edge_map, edges = _indexed_shapes(shape, TopAbs_EDGE)
    vertex_map, vertices = _indexed_shapes(shape, TopAbs_VERTEX)
    face_count, edge_count, vertex_count = len(faces), len(edges), len(vertices)

    face_bboxes = np.stack([_bbox_for_shape(face) for face in faces], axis=0) if faces else np.zeros((0, 6), dtype=np.float32)
    edge_bboxes = np.stack([_bbox_for_shape(edge) for edge in edges], axis=0) if edges else np.zeros((0, 6), dtype=np.float32)
    face_wcs = np.stack([_grid_from_bbox(bbox) for bbox in face_bboxes], axis=0) if face_count else np.zeros((0, 32, 32, 3), dtype=np.float32)
    edge_wcs = np.stack([_edge_points_from_bbox(bbox) for bbox in edge_bboxes], axis=0) if edge_count else np.zeros((0, 32, 3), dtype=np.float32)
    surface_types, curvature_proxy = _surface_metadata_from_shape(shape, face_count)

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
        "edge_bbox_wcs": edge_bboxes,
        "vert_wcs": vert_wcs_arr,
        "face_wcs": face_wcs,
        "edge_wcs": edge_wcs,
        "edgeFace_adj": edge_face_adj,
        "edgeVert_adj": edge_vert_adj,
        "faceEdge_adj": face_edge_adj,
        "face_surface_type": surface_types,
        "face_curvature_proxy": curvature_proxy,
        "face_count": face_count,
        "edge_count": edge_count,
        "vertex_count": vertex_count,
        "parser_backend": "pythonocc_fallback",
        "geometry_sampling_quality": "bbox_fallback_sampling",
        "surface_metadata_order_verified": True,
    }


def _ensure_minimal_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    if "face_bbox_wcs" not in data and "face_wcs" in data:
        face_wcs = np.asarray(data["face_wcs"], dtype=np.float32)
        data["face_bbox_wcs"] = np.asarray([np.concatenate([np.min(face.reshape(-1, 3), axis=0), np.max(face.reshape(-1, 3), axis=0)]) for face in face_wcs], dtype=np.float32)
    if "edge_bbox_wcs" not in data and "edge_wcs" in data:
        edge_wcs = np.asarray(data["edge_wcs"], dtype=np.float32)
        data["edge_bbox_wcs"] = np.asarray([np.concatenate([np.min(edge.reshape(-1, 3), axis=0), np.max(edge.reshape(-1, 3), axis=0)]) for edge in edge_wcs], dtype=np.float32)
    if "face_wcs" not in data:
        bboxes = np.asarray(data.get("face_bbox_wcs", np.zeros((0, 6))), dtype=np.float32)
        data["face_wcs"] = np.stack([_grid_from_bbox(bbox) for bbox in bboxes], axis=0) if len(bboxes) else np.zeros((0, 32, 32, 3), dtype=np.float32)
    if "edge_wcs" not in data:
        bboxes = np.asarray(data.get("edge_bbox_wcs", np.zeros((0, 6))), dtype=np.float32)
        data["edge_wcs"] = np.stack([_edge_points_from_bbox(bbox) for bbox in bboxes], axis=0) if len(bboxes) else np.zeros((0, 32, 3), dtype=np.float32)
    data["face_count"] = int(data.get("face_count", len(data.get("face_bbox_wcs", []))))
    data["edge_count"] = int(data.get("edge_count", len(data.get("edge_bbox_wcs", []))))
    data["vertex_count"] = int(data.get("vertex_count", len(data.get("vert_wcs", []))))
    data.setdefault("face_surface_type", np.zeros((data["face_count"],), dtype=np.int64))
    data.setdefault("face_curvature_proxy", np.zeros((data["face_count"],), dtype=np.float32))
    data.setdefault("parser_backend", "unknown")
    data.setdefault("geometry_sampling_quality", "true_or_dtg_sampling" if data.get("parser_backend") == "dtg_occwl" else "bbox_fallback_sampling")
    data.setdefault("surface_metadata_order_verified", str(data.get("parser_backend", "")) != "dtg_occwl")
    return data


def _parse_step(step_path: str) -> Dict[str, Any]:
    errors = []
    try:
        data = _parse_step_dtg(step_path)
        return _attach_surface_metadata(data, step_path)
    except Exception as exc:
        errors.append(f"DTG:{exc}")
    try:
        data = _parse_step_occ_fallback(step_path)
        data["parser_warning"] = " | ".join(errors)
        return _ensure_minimal_fields(data)
    except Exception as exc:
        errors.append(f"pythonOCC:{exc}")
        raise RuntimeError(" ; ".join(errors))


def _has_required_fields(data: Dict[str, Any]) -> Tuple[bool, str]:
    required = ["face_bbox_wcs", "edge_bbox_wcs", "vert_wcs", "face_wcs", "edge_wcs", "edgeFace_adj", "edgeVert_adj", "faceEdge_adj"]
    for key in required:
        if key not in data:
            return False, f"missing_{key}"
    arrays = ["face_bbox_wcs", "edge_bbox_wcs", "vert_wcs", "face_wcs", "edge_wcs", "edgeFace_adj", "edgeVert_adj"]
    for key in arrays:
        arr = np.asarray(data.get(key))
        if arr.size and not np.all(np.isfinite(arr)):
            return False, f"invalid_tensor_{key}"
    face_count = int(data.get("face_count", 0))
    if face_count <= 0:
        return False, "zero_face_count"
    if face_count > MAX_FACES:
        return False, "face_over_limit"
    bbox = np.asarray(data.get("face_bbox_wcs"), dtype=np.float32)
    if bbox.size == 0:
        return False, "empty_bbox"
    global_dims = np.max(bbox[:, 3:], axis=0) - np.min(bbox[:, :3], axis=0)
    if float(np.max(global_dims)) <= 1e-6:
        return False, "bbox_scale_too_small"
    return True, ""


def _structure_signature(data: Dict[str, Any]) -> str:
    face_bbox = np.asarray(data.get("face_bbox_wcs", np.zeros((0, 6))), dtype=np.float32)
    edge_face = np.asarray(data.get("edgeFace_adj", np.zeros((0, 2))), dtype=np.int64)
    if face_bbox.size:
        dims = np.maximum(face_bbox[:, 3:] - face_bbox[:, :3], 0.0)
        scale = max(float(np.max(dims)), 1e-8)
        norm_dims = np.round(np.sort(dims / scale, axis=1), 3)
        dim_sig = sorted(tuple(row.tolist()) for row in norm_dims[: min(len(norm_dims), 40)])
    else:
        dim_sig = []
    return short_hash(
        {
            "counts": [int(data.get("face_count", 0)), int(data.get("edge_count", 0)), int(data.get("vertex_count", 0))],
            "face_dims": dim_sig,
            "edge_face_degree_hist": np.bincount(np.maximum(edge_face.reshape(-1), 0), minlength=max(int(data.get("face_count", 0)), 1)).astype(int).tolist()[:80],
        }
    )


def parse_public_dataset(source_dir: str, workdir: str, max_files: int = 0) -> Dict[str, Any]:
    dirs = ensure_workdir(workdir)
    parsed_dir = dirs["parsed_public"]
    reports_dir = dirs["reports"]
    clean_dir(parsed_dir, ["*.pkl", "*.json", "*.jsonl", "*.csv"])

    step_files = scan_step_files(source_dir, max_files=max_files)
    manifest: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    duplicates: List[Dict[str, Any]] = []
    seen_signatures: Dict[str, str] = {}

    for index, step_path in enumerate(step_files):
        uid = uid_from_step(step_path, source_dir)
        source = detect_source_name(step_path)
        base = {"uid": uid, "source": source, "step_path": step_path, "index": index}
        try:
            data = _parse_step(step_path)
            data = _ensure_minimal_fields(data)
            ok, reason = _has_required_fields(data)
            if not ok:
                item = dict(base)
                item.update({"reject_reason": reason})
                rejected.append(item)
                continue
            sig = _structure_signature(data)
            if sig in seen_signatures:
                item = dict(base)
                item.update({"duplicate_of": seen_signatures[sig], "structure_signature": sig})
                duplicates.append(item)
                continue
            seen_signatures[sig] = uid
            data["uid"] = uid
            data["source"] = source
            data["source_step"] = step_path
            data["structure_signature"] = sig
            with open(os.path.join(parsed_dir, f"{uid}.pkl"), "wb") as f:
                pickle.dump(data, f)
            manifest.append(
                {
                    **base,
                    "parse_status": "SUCCESS",
                    "parser_backend": data.get("parser_backend", "unknown"),
                    "geometry_sampling_quality": data.get("geometry_sampling_quality", "unknown"),
                    "face_count": int(data.get("face_count", 0)),
                    "edge_count": int(data.get("edge_count", 0)),
                    "vertex_count": int(data.get("vertex_count", 0)),
                    "structure_signature": sig,
                    "error": data.get("parser_warning", ""),
                }
            )
        except Exception as exc:
            item = dict(base)
            text = str(exc)
            if "single_solid_required" in text:
                reason = "single_solid_reject"
            else:
                reason = "parse_failed"
            item.update({"reject_reason": reason, "error": text[:500]})
            rejected.append(item)

    write_csv(
        os.path.join(reports_dir, "public_parse_manifest.csv"),
        manifest,
        ["uid", "source", "step_path", "index", "parse_status", "parser_backend", "geometry_sampling_quality", "face_count", "edge_count", "vertex_count", "structure_signature", "error"],
    )
    write_csv(os.path.join(reports_dir, "rejected_manifest.csv"), rejected, ["uid", "source", "step_path", "index", "reject_reason", "error"])
    write_csv(os.path.join(reports_dir, "duplicate_manifest.csv"), duplicates, ["uid", "source", "step_path", "index", "duplicate_of", "structure_signature"])

    reason_counts: Dict[str, int] = {}
    for row in rejected:
        reason_counts[str(row.get("reject_reason", "unknown"))] = reason_counts.get(str(row.get("reject_reason", "unknown")), 0) + 1
    report = [
        "Innovation1 v3 Public B-Rep Parse Report",
        "=" * 72,
        f"Time: {timestamp()}",
        f"Source dir: {source_dir}",
        f"Total STEP files scanned: {len(step_files)}",
        f"Parse success kept: {len(manifest)}",
        f"Rejected: {len(rejected)}",
        f"Duplicates removed: {len(duplicates)}",
        "",
        "Reject reasons:",
    ]
    for reason, count in sorted(reason_counts.items()):
        report.append(f"  - {reason}: {count}")
    report.extend(
        [
            "",
            "Boundary statement:",
            "  Only stable single-solid B-Reps are kept. Cleaning is not claimed as the innovation.",
            "  Downstream labels are algorithm-extracted motifs, not manual truth.",
        ]
    )
    write_text(os.path.join(reports_dir, "public_parse_report.txt"), report)
    return {"manifest": manifest, "rejected": rejected, "duplicates": duplicates, "step_files": step_files}

