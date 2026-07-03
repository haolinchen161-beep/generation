# -*- coding: utf-8 -*-
"""STEP loading for public semantics-free B-Rep datasets."""

from __future__ import annotations

import io
import os
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:  # pragma: no cover - supports both script and package execution
    from .brep_cleaner import check_dtg_train_compatible, ensure_minimal_fields, validate_brep
    from .utils_io import (
        ensure_workdir,
        make_uid,
        normalize_path,
        scan_step_files,
        summarize_numeric,
        timestamp,
        write_csv,
        write_json,
        write_pickle,
        write_text,
    )
except ImportError:  # pragma: no cover
    from brep_cleaner import check_dtg_train_compatible, ensure_minimal_fields, validate_brep
    from utils_io import (
        ensure_workdir,
        make_uid,
        normalize_path,
        scan_step_files,
        summarize_numeric,
        timestamp,
        write_csv,
        write_json,
        write_pickle,
        write_text,
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class BrepParseError(RuntimeError):
    """Parse failure with an audit-friendly reason code."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


def _bbox_for_shape(shape: Any) -> np.ndarray:
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib

    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    try:
        xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    except Exception:
        return np.zeros(6, dtype=np.float32)
    return np.asarray([xmin, ymin, zmin, xmax, ymax, zmax], dtype=np.float32)


def _grid_from_bbox(bbox: np.ndarray, n: int = 32) -> np.ndarray:
    mn = np.asarray(bbox[:3], dtype=np.float32)
    mx = np.asarray(bbox[3:], dtype=np.float32)
    dims = np.maximum(mx - mn, 0.0)
    normal_axis = int(np.argmin(dims)) if dims.size else 2
    axes = [axis for axis in range(3) if axis != normal_axis]
    a_vals = np.linspace(float(mn[axes[0]]), float(mx[axes[0]]), n, dtype=np.float32)
    b_vals = np.linspace(float(mn[axes[1]]), float(mx[axes[1]]), n, dtype=np.float32)
    grid = np.zeros((n, n, 3), dtype=np.float32)
    const = float(0.5 * (mn[normal_axis] + mx[normal_axis]))
    for i, av in enumerate(a_vals):
        for j, bv in enumerate(b_vals):
            p = np.zeros(3, dtype=np.float32)
            p[normal_axis] = const
            p[axes[0]] = av
            p[axes[1]] = bv
            grid[i, j] = p
    return grid


def _edge_points_from_bbox(bbox: np.ndarray, n: int = 32) -> np.ndarray:
    p0 = np.asarray(bbox[:3], dtype=np.float32)
    p1 = np.asarray(bbox[3:], dtype=np.float32)
    if float(np.linalg.norm(p1 - p0)) < 1e-8:
        return np.repeat(p0[None, :], n, axis=0).astype(np.float32)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    return ((1.0 - t) * p0 + t * p1).astype(np.float32)


def _indexed_shapes(shape: Any, shape_type: int) -> Tuple[Any, List[Any]]:
    from OCC.Core.TopExp import topexp
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape

    indexed = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, shape_type, indexed)
    size = indexed.Size() if hasattr(indexed, "Size") else indexed.Extent()
    items = [indexed.FindKey(i + 1) for i in range(size)]
    return indexed, items


def _surface_metadata_from_shape(shape: Any, expected_face_count: int) -> Tuple[np.ndarray, np.ndarray]:
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs import GeomAbs_Plane
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopoDS import topods

    _, faces = _indexed_shapes(shape, TopAbs_FACE)
    surface_types: List[int] = []
    curvature_proxy: List[float] = []
    plane_type = int(GeomAbs_Plane)
    for face in faces[:expected_face_count]:
        try:
            adaptor = BRepAdaptor_Surface(topods.Face(face), True)
            surf_type = int(adaptor.GetType())
            surface_types.append(surf_type)
            curvature_proxy.append(0.0 if surf_type == plane_type else 1.0)
        except Exception:
            surface_types.append(0)
            curvature_proxy.append(0.0)
    if len(surface_types) < expected_face_count:
        surface_types.extend([0] * (expected_face_count - len(surface_types)))
        curvature_proxy.extend([0.0] * (expected_face_count - len(curvature_proxy)))
    return np.asarray(surface_types, dtype=np.int64), np.asarray(curvature_proxy, dtype=np.float32)


def _parse_step_occ_fallback(step_path: str) -> Dict[str, Any]:
    try:
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID, TopAbs_VERTEX
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopoDS import topods
    except Exception as exc:
        raise BrepParseError("pythonocc_unavailable", str(exc)) from exc

    reader = STEPControl_Reader()
    status = reader.ReadFile(step_path)
    if status != 1:
        raise BrepParseError("step_read_failed", f"STEPControl status={status}")
    reader.TransferRoots()
    shape = reader.OneShape()

    _, solids = _indexed_shapes(shape, TopAbs_SOLID)
    if len(solids) != 1:
        raise BrepParseError("not_single_solid", f"solid_count={len(solids)}")
    solid = solids[0]

    face_map, faces = _indexed_shapes(solid, TopAbs_FACE)
    edge_map, edges = _indexed_shapes(solid, TopAbs_EDGE)
    vertex_map, vertices = _indexed_shapes(solid, TopAbs_VERTEX)
    face_count = len(faces)
    edge_count = len(edges)
    vertex_count = len(vertices)

    face_bboxes = np.stack([_bbox_for_shape(face) for face in faces], axis=0) if faces else np.zeros((0, 6), dtype=np.float32)
    edge_bboxes = np.stack([_bbox_for_shape(edge) for edge in edges], axis=0) if edges else np.zeros((0, 6), dtype=np.float32)
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
                if fidx not in edge_faces[edge_idx]:
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

    surface_types, curvature_proxy = _surface_metadata_from_shape(solid, face_count)
    return ensure_minimal_fields(
        {
            "face_bbox_wcs": face_bboxes,
            "edge_bbox_wcs": edge_bboxes,
            "face_wcs": face_wcs,
            "edge_wcs": edge_wcs,
            "vert_wcs": vert_wcs_arr,
            "edgeFace_adj": edge_face_adj,
            "edgeVert_adj": edge_vert_adj,
            "faceEdge_adj": face_edge_adj,
            "face_surface_type": surface_types,
            "face_curvature_proxy": curvature_proxy,
            "face_count": face_count,
            "edge_count": edge_count,
            "vertex_count": vertex_count,
            "solid_count": 1,
            "parser_backend": "pythonocc_fallback",
            "geometry_sampling_quality": "bbox_fallback_sampling",
            "surface_metadata_order_verified": True,
        }
    )


def _parse_step_dtg(step_path: str) -> Dict[str, Any]:
    try:
        from occwl.io import load_step
        from data_process.brep_process import bspline_fitting_local, count_fef_adj, parse_solid
    except Exception as exc:
        raise BrepParseError("dtg_occwl_unavailable", str(exc)) from exc

    with warnings.catch_warnings(), redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        solids = load_step(step_path)
        if len(solids) != 1:
            raise BrepParseError("not_single_solid", f"solid_count={len(solids)}")
        data = parse_solid(solids[0])
        if data is None:
            raise BrepParseError("dtg_parse_solid_returned_none", "possibly over face limit or unsupported topology")
        try:
            data["fef_adj"] = count_fef_adj(data["faceEdge_adj"])
        except Exception:
            pass
        try:
            data = bspline_fitting_local(data)
        except Exception:
            pass
    data["solid_count"] = 1
    data["parser_backend"] = "dtg_occwl"
    data["geometry_sampling_quality"] = "true_or_dtg_sampling"
    data["surface_metadata_order_verified"] = False
    return ensure_minimal_fields(data)


def parse_step_file(step_path: str) -> Dict[str, Any]:
    errors: List[str] = []
    try:
        return _parse_step_dtg(step_path)
    except BrepParseError as exc:
        if exc.reason == "not_single_solid":
            raise
        errors.append(f"DTG:{exc}")
    except Exception as exc:
        errors.append(f"DTG:{exc}")

    try:
        data = _parse_step_occ_fallback(step_path)
        if errors:
            data["parser_warning"] = " | ".join(errors)
        return data
    except BrepParseError as exc:
        errors.append(f"pythonOCC:{exc}")
        if exc.reason == "not_single_solid":
            raise
        reason = "parser_unavailable" if any("unavailable" in item for item in errors) else "parse_failed"
        raise BrepParseError(reason, " ; ".join(errors)) from exc
    except Exception as exc:
        errors.append(f"pythonOCC:{exc}")
        raise BrepParseError("parse_failed", " ; ".join(errors)) from exc


def _manifest_fieldnames() -> List[str]:
    return [
        "uid",
        "source",
        "step_path",
        "pkl_path",
        "parse_status",
        "parser_backend",
        "geometry_sampling_quality",
        "face_count",
        "edge_count",
        "vertex_count",
        "global_scale",
        "dtg_train_compatible",
        "dtg_filter_reason",
        "error",
    ]


def _rejected_fieldnames() -> List[str]:
    return [
        "uid",
        "source",
        "step_path",
        "stage",
        "reject_reason",
        "parser_backend",
        "geometry_sampling_quality",
        "face_count",
        "edge_count",
        "vertex_count",
        "global_scale",
        "dtg_train_compatible",
        "dtg_filter_reason",
        "error",
    ]


def _write_parse_report(
    workdir: str,
    scanned_count: int,
    clean_rows: List[Dict[str, Any]],
    rejected_rows: List[Dict[str, Any]],
    parse_success_count: int,
    max_faces: int,
) -> Dict[str, Any]:
    dirs = ensure_workdir(workdir)
    parse_failures = [r for r in rejected_rows if r.get("stage") == "parse" and r.get("reject_reason") != "not_single_solid"]
    single_entity_filtered = [r for r in rejected_rows if r.get("reject_reason") == "not_single_solid"]
    face_limit_filtered = [r for r in rejected_rows if "face_count_over_limit" in str(r.get("reject_reason"))]
    faces = [float(r.get("face_count", 0)) for r in clean_rows]
    edges = [float(r.get("edge_count", 0)) for r in clean_rows]
    verts = [float(r.get("vertex_count", 0)) for r in clean_rows]
    backend_counts: Dict[str, int] = {}
    quality_counts: Dict[str, int] = {}
    for row in clean_rows:
        backend = str(row.get("parser_backend", "unknown"))
        quality = str(row.get("geometry_sampling_quality", "unknown"))
        backend_counts[backend] = backend_counts.get(backend, 0) + 1
        quality_counts[quality] = quality_counts.get(quality, 0) + 1
    dtg_train_compatible_count = sum(int(row.get("dtg_train_compatible", 0)) for row in clean_rows)

    summary = {
        "scan_step_count": scanned_count,
        "parse_success_count": parse_success_count,
        "parse_failure_count": len(parse_failures),
        "single_entity_filter_count": len(single_entity_filtered),
        "canonical_face_count_max": int(max_faces),
        "canonical_face_count_policy": "DTG backend counts faces after parse_solid closed-face and closed-edge splitting; pythonOCC fallback counts TopAbs_FACE and is marked low-quality.",
        "face_count_over_limit_filter_count": len(face_limit_filtered),
        "clean_sample_count": len(clean_rows),
        "dtg_train_compatible_count": dtg_train_compatible_count,
        "backend_counts": backend_counts,
        "geometry_sampling_quality_counts": quality_counts,
    }
    write_json(os.path.join(dirs["reports"], "parse_summary.json"), summary)

    report = [
        "Innovation1 v3 Clean ABC/DeepCAD Parse Report",
        "=" * 72,
        f"Report time: {timestamp()}",
        "",
        f"Scanned STEP files: {scanned_count}",
        f"Parse success before clean filters: {parse_success_count}",
        f"Parse failures: {len(parse_failures)}",
        f"Single-solid filter rejects: {len(single_entity_filtered)}",
        f"canonical face_count > {max_faces} rejects: {len(face_limit_filtered)}",
        f"Final clean samples: {len(clean_rows)}",
        f"DTG-train-compatible clean samples: {dtg_train_compatible_count}",
        "",
        f"Face count min/mean/max: {summarize_numeric(faces)}",
        f"Edge count min/mean/max: {summarize_numeric(edges)}",
        f"Vertex count min/mean/max: {summarize_numeric(verts)}",
        "",
        "Parser backend counts:",
    ]
    for key, value in sorted(backend_counts.items()):
        report.append(f"  - {key}: {value}")
    report.append("")
    report.append("Geometry sampling quality counts:")
    for key, value in sorted(quality_counts.items()):
        report.append(f"  - {key}: {value}")
    report.extend(
        [
            "",
            "Clean policy:",
            "  - Keep exactly one solid.",
            f"  - Keep canonical face_count <= {max_faces}.",
            "  - canonical face_count is counted after occwl/DTG parse_solid closed-face and closed-edge splitting when the DTG backend succeeds.",
            "  - pythonOCC fallback counts TopAbs_FACE directly, is marked bbox_fallback_sampling, and is not motif-ready unless it also passes DTG-compatible checks.",
            "  - Require nonzero edge_count and vertex_count.",
            "  - Require constructible edgeFace_adj, edgeVert_adj and faceEdge_adj.",
            "  - Require finite face_wcs, edge_wcs, vert_wcs and bbox fields.",
            "  - Require global bbox scale > 1e-6.",
            "  - Reject parse failures directly without topology repair or multi-entity splitting.",
        ]
    )
    if rejected_rows:
        report.extend(["", "Rejected examples:"])
        for row in rejected_rows[:40]:
            report.append(f"  - {row.get('uid')}: {row.get('reject_reason')} ({row.get('error', '')})")
    report_path = os.path.join(dirs["reports"], "clean_parse_report.txt")
    write_text(report_path, report)
    return summary


def parse_abc_dataset(
    step_root: str,
    workdir: str,
    source: str = "abc",
    limit: int = 0,
    max_faces: int = 70,
) -> Dict[str, Any]:
    dirs = ensure_workdir(workdir)
    step_files = scan_step_files(step_root, limit=limit)
    clean_rows: List[Dict[str, Any]] = []
    rejected_rows: List[Dict[str, Any]] = []
    parse_success_count = 0

    for idx, step_path in enumerate(step_files, start=1):
        uid = make_uid(step_path, step_root)
        pkl_path = os.path.join(dirs["parsed"], f"{uid}.pkl")
        try:
            data = parse_step_file(step_path)
            parse_success_count += 1
            data["uid"] = uid
            data["source"] = source
            data["source_step"] = normalize_path(step_path)
            try:
                data["source_step_rel"] = str(Path(step_path).resolve().relative_to(Path(step_root).resolve())).replace("\\", "/")
            except Exception:
                data["source_step_rel"] = normalize_path(step_path)
            data = ensure_minimal_fields(data)
            ok, reason, stats = validate_brep(data, max_faces=max_faces)
            dtg_ok, dtg_reason, _ = check_dtg_train_compatible(data)
            data["dtg_train_compatible"] = int(dtg_ok)
            data["dtg_filter_reason"] = dtg_reason
            if not ok:
                rejected_rows.append(
                    {
                        "uid": uid,
                        "source": source,
                        "step_path": normalize_path(step_path),
                        "stage": "filter",
                        "reject_reason": reason,
                        "parser_backend": stats.get("parser_backend", data.get("parser_backend", "unknown")),
                        "geometry_sampling_quality": stats.get("geometry_sampling_quality", data.get("geometry_sampling_quality", "unknown")),
                        "face_count": stats.get("face_count", data.get("face_count", 0)),
                        "edge_count": stats.get("edge_count", data.get("edge_count", 0)),
                        "vertex_count": stats.get("vertex_count", data.get("vertex_count", 0)),
                        "global_scale": stats.get("global_scale", ""),
                        "dtg_train_compatible": int(dtg_ok),
                        "dtg_filter_reason": dtg_reason,
                        "error": data.get("parser_warning", ""),
                    }
                )
                continue
            write_pickle(pkl_path, data)
            clean_rows.append(
                {
                    "uid": uid,
                    "source": source,
                    "step_path": normalize_path(step_path),
                    "pkl_path": normalize_path(pkl_path),
                    "parse_status": "SUCCESS",
                    "parser_backend": data.get("parser_backend", "unknown"),
                    "geometry_sampling_quality": data.get("geometry_sampling_quality", "unknown"),
                    "face_count": int(data.get("face_count", 0)),
                    "edge_count": int(data.get("edge_count", 0)),
                    "vertex_count": int(data.get("vertex_count", 0)),
                    "global_scale": float(np.max(np.asarray(data["global_bbox"])[3:] - np.asarray(data["global_bbox"])[:3])),
                    "dtg_train_compatible": int(dtg_ok),
                    "dtg_filter_reason": dtg_reason,
                    "error": data.get("parser_warning", ""),
                }
            )
        except BrepParseError as exc:
            reject_stage = "filter" if exc.reason == "not_single_solid" else "parse"
            rejected_rows.append(
                {
                    "uid": uid,
                    "source": source,
                    "step_path": normalize_path(step_path),
                    "stage": reject_stage,
                    "reject_reason": exc.reason,
                    "parser_backend": "none",
                    "geometry_sampling_quality": "none",
                    "face_count": 0,
                    "edge_count": 0,
                    "vertex_count": 0,
                    "global_scale": "",
                    "dtg_train_compatible": 0,
                    "dtg_filter_reason": "parse_rejected",
                    "error": exc.detail or str(exc),
                }
            )
        except Exception as exc:
            rejected_rows.append(
                {
                    "uid": uid,
                    "source": source,
                    "step_path": normalize_path(step_path),
                    "stage": "parse",
                    "reject_reason": "parse_failed",
                    "parser_backend": "none",
                    "geometry_sampling_quality": "none",
                    "face_count": 0,
                    "edge_count": 0,
                    "vertex_count": 0,
                    "global_scale": "",
                    "dtg_train_compatible": 0,
                    "dtg_filter_reason": "parse_rejected",
                    "error": str(exc),
                }
            )
        if idx % 250 == 0:
            print(f"[parse_abc] processed {idx}/{len(step_files)}; clean={len(clean_rows)} rejected={len(rejected_rows)}")

    clean_manifest = os.path.join(dirs["parsed"], "clean_manifest.csv")
    rejected_manifest = os.path.join(dirs["parsed"], "rejected_manifest.csv")
    write_csv(clean_manifest, clean_rows, _manifest_fieldnames())
    write_csv(rejected_manifest, rejected_rows, _rejected_fieldnames())
    write_csv(os.path.join(dirs["reports"], "clean_manifest.csv"), clean_rows, _manifest_fieldnames())
    write_csv(os.path.join(dirs["reports"], "rejected_manifest.csv"), rejected_rows, _rejected_fieldnames())
    summary = _write_parse_report(workdir, len(step_files), clean_rows, rejected_rows, parse_success_count, max_faces)
    return {
        "step_root": normalize_path(step_root),
        "clean_manifest": clean_manifest,
        "rejected_manifest": rejected_manifest,
        "records": clean_rows,
        "rejected": rejected_rows,
        "summary": summary,
    }
