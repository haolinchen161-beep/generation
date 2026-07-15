# -*- coding: utf-8 -*-
"""公共无语义 B-Rep 数据集的 STEP 解析与清洗入口。"""

from __future__ import annotations

import io
import os
import sys
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
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
        read_pickle,
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
        read_pickle,
        write_text,
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class BrepParseError(RuntimeError):
    """带审计原因码的解析失败异常。"""

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


def _surface_metadata_from_shape(shape: Any, expected_face_count: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCC.Core.TopoDS import topods
    from OCC.Core.BRepLProp import BRepLProp_SLProps

    _, faces = _indexed_shapes(shape, TopAbs_FACE)
    surface_types: List[int] = []
    curvature_proxy: List[float] = []
    analytical_normals: List[List[float]] = []
    cylinder_radii: List[float] = []
    cylinder_axes: List[List[float]] = []
    cylinder_locations: List[List[float]] = []
    
    mean_curvatures: List[float] = []
    max_curvatures: List[float] = []
    var_curvatures: List[float] = []
    gaussian_signs: List[int] = []

    plane_type = int(GeomAbs_Plane)
    cyl_type = int(GeomAbs_Cylinder)
    for face in faces[:expected_face_count]:
        try:
            face_obj = topods.Face(face)
            adaptor = BRepAdaptor_Surface(face_obj, True)
            surf_type = int(adaptor.GetType())
            surface_types.append(surf_type)
            curvature_proxy.append(0.0 if surf_type == plane_type else 1.0)

            normal = [0.0, 0.0, 0.0]
            radius = 0.0
            axis = [0.0, 0.0, 0.0]
            location = [0.0, 0.0, 0.0]
            if surf_type == plane_type:
                gp_dir = adaptor.Plane().Position().Direction()
                normal = [gp_dir.X(), gp_dir.Y(), gp_dir.Z()]
            elif surf_type == cyl_type:
                gp_cyl = adaptor.Cylinder()
                gp_dir = gp_cyl.Position().Direction()
                normal = [gp_dir.X(), gp_dir.Y(), gp_dir.Z()]
                radius = float(gp_cyl.Radius())
                axis = [gp_dir.X(), gp_dir.Y(), gp_dir.Z()]
                gp_loc = gp_cyl.Location()
                location = [gp_loc.X(), gp_loc.Y(), gp_loc.Z()]
            
            if face_obj.Orientation() == TopAbs_REVERSED:
                normal = [-val for val in normal]
                
            # 计算微分几何曲率特征
            mean_curvs = []
            max_curvs = []
            gaussian_signs_sample = []
            try:
                from OCC.Core.BRepTools import breptools
                u_min, u_max, v_min, v_max = breptools.UVBounds(face_obj)
                
                u_samples = np.linspace(u_min, u_max, 5)
                v_samples = np.linspace(v_min, v_max, 5)
                
                for u_val in u_samples:
                    for v_val in v_samples:
                        try:
                            props = BRepLProp_SLProps(adaptor, u_val, v_val, 2, 1e-6)
                            if props.IsCurvatureDefined():
                                k1 = props.MaxCurvature()
                                k2 = props.MinCurvature()
                                mean_curvs.append(0.5 * (abs(k1) + abs(k2)))
                                max_curvs.append(max(abs(k1), abs(k2)))
                                g_curv = k1 * k2
                                g_sign = 0
                                if g_curv > 1e-4:
                                    g_sign = 1
                                elif g_curv < -1e-4:
                                    g_sign = -1
                                gaussian_signs_sample.append(g_sign)
                        except Exception:
                            pass
            except Exception:
                pass
                
            if mean_curvs:
                mean_c_avg = float(np.mean(mean_curvs))
                max_c_avg = float(np.mean(max_curvs))
                var_c = float(np.var(mean_curvs))
                g_sign_avg = int(np.round(np.mean(gaussian_signs_sample)))
            else:
                mean_c_avg = 0.0
                max_c_avg = 0.0
                var_c = 0.0
                g_sign_avg = 0
                
            mean_curvatures.append(mean_c_avg)
            max_curvatures.append(max_c_avg)
            var_curvatures.append(var_c)
            gaussian_signs.append(g_sign_avg)
                
            analytical_normals.append(normal)
            cylinder_radii.append(radius)
            cylinder_axes.append(axis)
            cylinder_locations.append(location)
        except Exception:
            surface_types.append(0)
            curvature_proxy.append(0.0)
            analytical_normals.append([0.0, 0.0, 0.0])
            cylinder_radii.append(0.0)
            cylinder_axes.append([0.0, 0.0, 0.0])
            cylinder_locations.append([0.0, 0.0, 0.0])
            mean_curvatures.append(0.0)
            max_curvatures.append(0.0)
            var_curvatures.append(0.0)
            gaussian_signs.append(0)
    while len(surface_types) < expected_face_count:
        surface_types.append(0)
        curvature_proxy.append(0.0)
        analytical_normals.append([0.0, 0.0, 0.0])
        cylinder_radii.append(0.0)
        cylinder_axes.append([0.0, 0.0, 0.0])
        cylinder_locations.append([0.0, 0.0, 0.0])
        mean_curvatures.append(0.0)
        max_curvatures.append(0.0)
        var_curvatures.append(0.0)
        gaussian_signs.append(0)
    return (
        np.asarray(surface_types, dtype=np.int64),
        np.asarray(curvature_proxy, dtype=np.float32),
        np.asarray(analytical_normals, dtype=np.float32),
        np.asarray(cylinder_radii, dtype=np.float32),
        np.asarray(cylinder_axes, dtype=np.float32),
        np.asarray(cylinder_locations, dtype=np.float32),
        np.asarray(mean_curvatures, dtype=np.float32),
        np.asarray(max_curvatures, dtype=np.float32),
        np.asarray(var_curvatures, dtype=np.float32),
        np.asarray(gaussian_signs, dtype=np.int32)
    )


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

    edge_face_adj = edge_faces

    edge_vert_adj = []
    for eidx, edge in enumerate(edges):
        vids: List[int] = []
        explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
        while explorer.More():
            vid = vertex_map.FindIndex(explorer.Current()) - 1
            if vid >= 0 and vid not in vids:
                vids.append(vid)
            explorer.Next()
        edge_vert_adj.append(vids)

    (
        surface_types,
        curvature_proxy,
        analytical_normals,
        cylinder_radii,
        cylinder_axes,
        cylinder_locations,
        mean_curvatures,
        max_curvatures,
        var_curvatures,
        gaussian_signs,
    ) = _surface_metadata_from_shape(solid, face_count)
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
            "face_analytical_normals": analytical_normals,
            "face_cylinder_radius": cylinder_radii,
            "face_cylinder_axis": cylinder_axes,
            "face_cylinder_location": cylinder_locations,
            "face_mean_curvature": mean_curvatures,
            "face_max_curvature": max_curvatures,
            "face_var_curvature": var_curvatures,
            "face_gaussian_sign": gaussian_signs,
            "face_count": face_count,
            "edge_count": edge_count,
            "vertex_count": vertex_count,
            "solid_count": 1,
            "parser_backend": "pythonocc_fallback",
            "geometry_sampling_quality": "bbox_fallback_sampling",
            "surface_metadata_order_verified": True,
        }
    )


def _extract_surface_types_from_split_solid(split_solid: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.TopoDS import topods
    from occwl.entity_mapper import EntityMapper
    mapper = EntityMapper(split_solid)
    
    face_dict = {}
    for face in split_solid.faces():
        face_idx = mapper.face_index(face)
        face_dict[face_idx] = face
        
    dict_new = {}
    max_idx = max(face_dict.keys()) if face_dict else 0
    skipped_indices = set(range(max_idx)) - set(face_dict.keys())
    for idx, face in face_dict.items():
        skips = sum(1 for x in skipped_indices if x < idx)
        idx_new = idx - skips
        dict_new[idx_new] = face
        
    from OCC.Core.BRepLProp import BRepLProp_SLProps

    surface_types = []
    curvature_proxy = []
    analytical_normals = []
    cylinder_radii = []
    cylinder_axes = []
    cylinder_locations = []
    
    mean_curvatures = []
    max_curvatures = []
    var_curvatures = []
    gaussian_signs = []

    face_count = len(dict_new)
    for i in range(face_count):
        face = dict_new[i]
        try:
            enum_val = int(face.surface_type_enum())
            surface_types.append(enum_val)
            curvature_proxy.append(0.0 if enum_val == 0 else 1.0)
            
            normal = [0.0, 0.0, 0.0]
            radius = 0.0
            axis = [0.0, 0.0, 0.0]
            location = [0.0, 0.0, 0.0]
            
            topo_face = topods.Face(face.topods_shape())
            adaptor = BRepAdaptor_Surface(topo_face, True)
            
            if enum_val == 0 or enum_val == 1:
                if enum_val == 0:
                    gp_dir = adaptor.Plane().Position().Direction()
                else:
                    gp_cyl = adaptor.Cylinder()
                    gp_dir = gp_cyl.Position().Direction()
                    radius = float(gp_cyl.Radius())
                    axis = [gp_dir.X(), gp_dir.Y(), gp_dir.Z()]
                    gp_loc = gp_cyl.Location()
                    location = [gp_loc.X(), gp_loc.Y(), gp_loc.Z()]
                normal = [gp_dir.X(), gp_dir.Y(), gp_dir.Z()]
                from OCC.Core.TopAbs import TopAbs_REVERSED
                if topo_face.Orientation() == TopAbs_REVERSED:
                    normal = [-val for val in normal]
            
            # 计算微分几何曲率特征
            mean_curvs = []
            max_curvs = []
            gaussian_signs_sample = []
            try:
                box = face.uv_bounds()
                u_min, v_min = box.min_point()
                u_max, v_max = box.max_point()
                
                u_samples = np.linspace(u_min, u_max, 5)
                v_samples = np.linspace(v_min, v_max, 5)
                
                for u_val in u_samples:
                    for v_val in v_samples:
                        try:
                            props = BRepLProp_SLProps(adaptor, u_val, v_val, 2, 1e-6)
                            if props.IsCurvatureDefined():
                                k1 = props.MaxCurvature()
                                k2 = props.MinCurvature()
                                mean_curvs.append(0.5 * (abs(k1) + abs(k2)))
                                max_curvs.append(max(abs(k1), abs(k2)))
                                g_curv = k1 * k2
                                g_sign = 0
                                if g_curv > 1e-4:
                                    g_sign = 1
                                elif g_curv < -1e-4:
                                    g_sign = -1
                                gaussian_signs_sample.append(g_sign)
                        except Exception:
                            pass
            except Exception:
                pass
                
            if mean_curvs:
                mean_c_avg = float(np.mean(mean_curvs))
                max_c_avg = float(np.mean(max_curvs))
                var_c = float(np.var(mean_curvs))
                g_sign_avg = int(np.round(np.mean(gaussian_signs_sample)))
            else:
                mean_c_avg = 0.0
                max_c_avg = 0.0
                var_c = 0.0
                g_sign_avg = 0
                
            mean_curvatures.append(mean_c_avg)
            max_curvatures.append(max_c_avg)
            var_curvatures.append(var_c)
            gaussian_signs.append(g_sign_avg)
            
            analytical_normals.append(normal)
            cylinder_radii.append(radius)
            cylinder_axes.append(axis)
            cylinder_locations.append(location)
        except Exception:
            surface_types.append(0)
            curvature_proxy.append(0.0)
            analytical_normals.append([0.0, 0.0, 0.0])
            cylinder_radii.append(0.0)
            cylinder_axes.append([0.0, 0.0, 0.0])
            cylinder_locations.append([0.0, 0.0, 0.0])
            mean_curvatures.append(0.0)
            max_curvatures.append(0.0)
            var_curvatures.append(0.0)
            gaussian_signs.append(0)
            
    return (
        np.asarray(surface_types, dtype=np.int64),
        np.asarray(curvature_proxy, dtype=np.float32),
        np.asarray(analytical_normals, dtype=np.float32),
        np.asarray(cylinder_radii, dtype=np.float32),
        np.asarray(cylinder_axes, dtype=np.float32),
        np.asarray(cylinder_locations, dtype=np.float32),
        np.asarray(mean_curvatures, dtype=np.float32),
        np.asarray(max_curvatures, dtype=np.float32),
        np.asarray(var_curvatures, dtype=np.float32),
        np.asarray(gaussian_signs, dtype=np.int32)
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
        
        # 先对 solids[0] 进行 split，提取面类型和曲率代理，保证与 parse_solid 完全一致
        split_solid = solids[0].split_all_closed_faces(num_splits=2)
        split_solid = split_solid.split_all_closed_edges(num_splits=2)
        (
            surface_types,
            curvature_proxy,
            analytical_normals,
            cylinder_radii,
            cylinder_axes,
            cylinder_locations,
            mean_curvatures,
            max_curvatures,
            var_curvatures,
            gaussian_signs,
        ) = _extract_surface_types_from_split_solid(split_solid)

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
        
        # 计算从原始 split_solid 到归一化 data 坐标系的缩放与平移变换
        parsed_bboxes = data["face_bbox_wcs"]
        if parsed_bboxes.size > 0:
            norm_min = np.min(parsed_bboxes[:, :3], axis=0)
            norm_max = np.max(parsed_bboxes[:, 3:], axis=0)
            norm_center = 0.5 * (norm_min + norm_max)
            norm_scale = float(np.max(norm_max - norm_min))

            orig_min = np.array([1e9, 1e9, 1e9], dtype=np.float32)
            orig_max = np.array([-1e9, -1e9, -1e9], dtype=np.float32)
            for face in split_solid.faces():
                box = face.box()
                orig_min = np.minimum(orig_min, box.min_point())
                orig_max = np.maximum(orig_max, box.max_point())
            orig_center = 0.5 * (orig_min + orig_max)
            orig_scale = float(np.max(orig_max - orig_min))

            s = norm_scale / max(orig_scale, 1e-8)
            T = norm_center - s * orig_center

            # 应用缩放平移
            cylinder_radii = [float(r * s) for r in cylinder_radii]
            cylinder_locations = [(np.asarray(loc, dtype=np.float32) * s + T).tolist() for loc in cylinder_locations]
            mean_curvatures = [float(k / s) for k in mean_curvatures]
            max_curvatures = [float(k / s) for k in max_curvatures]
            var_curvatures = [float(v / (s**2)) for v in var_curvatures]

        data["face_surface_type"] = surface_types
        data["face_curvature_proxy"] = curvature_proxy
        data["face_analytical_normals"] = analytical_normals
        data["face_cylinder_radius"] = cylinder_radii
        data["face_cylinder_axis"] = cylinder_axes
        data["face_cylinder_location"] = cylinder_locations
        data["face_mean_curvature"] = mean_curvatures
        data["face_max_curvature"] = max_curvatures
        data["face_var_curvature"] = var_curvatures
        data["face_gaussian_sign"] = gaussian_signs
        
        # 增加防御性校验：验证 split_solid 与 parse_solid 两套路径的拓扑面片顺序是否完全吻合
        try:
            from occwl.entity_mapper import EntityMapper
            mapper = EntityMapper(split_solid)
            face_dict = {}
            for face in split_solid.faces():
                face_idx = mapper.face_index(face)
                face_dict[face_idx] = face
            
            dim_ratios = []
            parsed_bboxes = data["face_bbox_wcs"]
            sorted_keys = sorted(face_dict.keys())
            for i in range(len(sorted_keys)):
                face_obj = face_dict[sorted_keys[i]]
                box = face_obj.box()
                min_p = box.min_point()
                max_p = box.max_point()
                split_dims = max_p - min_p
                
                parsed_bbox = parsed_bboxes[i]
                parsed_dims = parsed_bbox[3:] - parsed_bbox[:3]
                
                nz = (split_dims > 1e-4)
                if np.any(nz):
                    ratio = parsed_dims[nz] / split_dims[nz]
                    dim_ratios.extend(ratio.tolist())
                    
            if len(dim_ratios) > 0:
                std_ratio = float(np.std(dim_ratios))
                if std_ratio > 1e-2:
                    raise BrepParseError("face_order_mismatch", f"检测到面片遍历顺序发生错位，缩放标准差为 {std_ratio:.6f}")
                else:
                    data["surface_metadata_order_verified"] = True
            else:
                data["surface_metadata_order_verified"] = False
        except BrepParseError:
            raise
        except Exception:
            data["surface_metadata_order_verified"] = False

    data["solid_count"] = 1
    data["parser_backend"] = "dtg_occwl"
    data["geometry_sampling_quality"] = "true_or_dtg_sampling"
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
    num_workers: int,
    unresolved_rows: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    dirs = ensure_workdir(workdir)
    unresolved_rows = list(unresolved_rows or [])
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
        "parse_num_workers": int(num_workers),
        "parse_success_count": parse_success_count,
        "parse_failure_count": len(parse_failures),
        "single_entity_filter_count": len(single_entity_filtered),
        "canonical_face_count_max": int(max_faces),
        "canonical_face_count_policy": "DTG 后端在 parse_solid 拆分 closed faces / closed edges 后统计规范 face_count；pythonOCC fallback 直接统计 TopAbs_FACE，并标记为低质量采样。",
        "face_count_over_limit_filter_count": len(face_limit_filtered),
        "clean_sample_count": len(clean_rows),
        "unresolved_without_pkl_count": len(unresolved_rows),
        "dtg_train_compatible_count": dtg_train_compatible_count,
        "backend_counts": backend_counts,
        "geometry_sampling_quality_counts": quality_counts,
    }
    write_json(os.path.join(dirs["reports"], "parse_summary.json"), summary)

    report = [
        "Innovation1 v3 ABC/DeepCAD 清洗解析报告",
        "=" * 72,
        f"报告时间：{timestamp()}",
        "",
        f"扫描 STEP 文件数：{scanned_count}",
        f"解析进程数：{num_workers}",
        f"清洗前解析成功数：{parse_success_count}",
        f"解析失败数：{len(parse_failures)}",
        f"非单 solid 过滤数：{len(single_entity_filtered)}",
        f"规范 face_count > {max_faces} 过滤数：{len(face_limit_filtered)}",
        f"最终 clean 样本数：{len(clean_rows)}",
        f"未重建样本数（无 pkl，可能是过滤/失败/被中断）：{len(unresolved_rows)}",
        f"DTG 训练兼容 clean 样本数：{dtg_train_compatible_count}",
        "",
        f"Face count 最小/均值/最大：{summarize_numeric(faces)}",
        f"Edge count 最小/均值/最大：{summarize_numeric(edges)}",
        f"Vertex count 最小/均值/最大：{summarize_numeric(verts)}",
        "",
        "解析后端统计：",
    ]
    for key, value in sorted(backend_counts.items()):
        report.append(f"  - {key}: {value}")
    report.append("")
    report.append("几何采样质量统计：")
    for key, value in sorted(quality_counts.items()):
        report.append(f"  - {key}: {value}")
    report.extend(
        [
            "",
            "清洗策略：",
            "  - 只保留单 solid。",
            f"  - 只保留规范 face_count <= {max_faces} 的样本。",
            "  - DTG 后端成功时，face_count 采用 occwl/DTG parse_solid 拆分 closed faces / closed edges 后的规范计数。",
            "  - pythonOCC fallback 直接统计 TopAbs_FACE，标记为 bbox_fallback_sampling；除非同时通过 DTG 兼容检查，否则默认不作为 motif-ready 样本。",
            "  - 要求 edge_count 与 vertex_count 非零。",
            "  - 要求 edgeFace_adj、edgeVert_adj、faceEdge_adj 可构造。",
            "  - 要求 face_wcs、edge_wcs、vert_wcs 与 bbox 字段均为有限值。",
            "  - 要求全局 bbox 尺度 > 1e-6。",
            "  - 解析失败直接拒绝，不做拓扑修复或多实体拆分。",
        ]
    )
    if rejected_rows:
        report.extend(["", "拒绝样本示例："])
        for row in rejected_rows[:40]:
            report.append(f"  - {row.get('uid')}: {row.get('reject_reason')} ({row.get('error', '')})")
    if unresolved_rows:
        report.extend(
            [
                "",
                "未重建样本说明：",
                "  - 这些 STEP 没有对应 pkl。本报告不再重新解析 STEP，因此不能区分它们是解析失败、过滤失败，还是上次中断前尚未完成。",
                "  - 如需完整 rejected 细分，需要重新运行 parse_abc；如只继续后续 motif 抽取，可直接使用当前 clean_manifest.csv。",
            ]
        )
    report_path = os.path.join(dirs["reports"], "clean_parse_report.txt")
    write_text(report_path, report)
    return summary


def _parse_one_step_record(
    idx: int,
    step_path: str,
    step_root: str,
    parsed_dir: str,
    source: str,
    max_faces: int,
) -> Dict[str, Any]:
    uid = make_uid(step_path, step_root)
    pkl_path = os.path.join(parsed_dir, f"{uid}.pkl")
    try:
        data = parse_step_file(step_path)
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
            return {
                "idx": idx,
                "parse_success": 1,
                "clean": None,
                "rejected": {
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
                },
            }
        write_pickle(pkl_path, data)
        return {
            "idx": idx,
            "parse_success": 1,
            "clean": {
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
            },
            "rejected": None,
        }
    except BrepParseError as exc:
        reject_stage = "filter" if exc.reason == "not_single_solid" else "parse"
        return {
            "idx": idx,
            "parse_success": 0,
            "clean": None,
            "rejected": {
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
            },
        }
    except Exception as exc:
        return {
            "idx": idx,
            "parse_success": 0,
            "clean": None,
            "rejected": {
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
            },
        }


def _future_reject_row(step_path: str, step_root: str, source: str, error: str) -> Dict[str, Any]:
    uid = make_uid(step_path, step_root)
    return {
        "uid": uid,
        "source": source,
        "step_path": normalize_path(step_path),
        "stage": "parse",
        "reject_reason": "parse_worker_failed",
        "parser_backend": "none",
        "geometry_sampling_quality": "none",
        "face_count": 0,
        "edge_count": 0,
        "vertex_count": 0,
        "global_scale": "",
        "dtg_train_compatible": 0,
        "dtg_filter_reason": "parse_rejected",
        "error": error,
    }


def _result_from_existing_pkl(
    idx: int,
    step_path: str,
    step_root: str,
    pkl_path: str,
    source: str,
    max_faces: int,
) -> Dict[str, Any]:
    uid = make_uid(step_path, step_root)
    try:
        data = read_pickle(pkl_path)
        data = ensure_minimal_fields(data)
        ok, reason, stats = validate_brep(data, max_faces=max_faces)
        dtg_ok, dtg_reason, _ = check_dtg_train_compatible(data)
        if not ok:
            return {
                "idx": idx,
                "parse_success": 1,
                "clean": None,
                "rejected": {
                    "uid": uid,
                    "source": source,
                    "step_path": normalize_path(step_path),
                    "stage": "filter",
                    "reject_reason": f"existing_pkl_{reason}",
                    "parser_backend": stats.get("parser_backend", data.get("parser_backend", "unknown")),
                    "geometry_sampling_quality": stats.get("geometry_sampling_quality", data.get("geometry_sampling_quality", "unknown")),
                    "face_count": stats.get("face_count", data.get("face_count", 0)),
                    "edge_count": stats.get("edge_count", data.get("edge_count", 0)),
                    "vertex_count": stats.get("vertex_count", data.get("vertex_count", 0)),
                    "global_scale": stats.get("global_scale", ""),
                    "dtg_train_compatible": int(dtg_ok),
                    "dtg_filter_reason": dtg_reason,
                    "error": "已有 pkl 未通过当前清洗规则",
                },
            }
        global_bbox = np.asarray(data["global_bbox"], dtype=np.float32)
        global_scale = float(np.max(global_bbox[3:] - global_bbox[:3]))
        return {
            "idx": idx,
            "parse_success": 1,
            "clean": {
                "uid": uid,
                "source": source,
                "step_path": normalize_path(step_path),
                "pkl_path": normalize_path(pkl_path),
                "parse_status": "SUCCESS_RESUMED",
                "parser_backend": data.get("parser_backend", "unknown"),
                "geometry_sampling_quality": data.get("geometry_sampling_quality", "unknown"),
                "face_count": int(data.get("face_count", 0)),
                "edge_count": int(data.get("edge_count", 0)),
                "vertex_count": int(data.get("vertex_count", 0)),
                "global_scale": global_scale,
                "dtg_train_compatible": int(dtg_ok),
                "dtg_filter_reason": dtg_reason,
                "error": data.get("parser_warning", ""),
            },
            "rejected": None,
        }
    except Exception as exc:
        return {
            "idx": idx,
            "parse_success": 0,
            "clean": None,
            "rejected": {
                "uid": uid,
                "source": source,
                "step_path": normalize_path(step_path),
                "stage": "parse",
                "reject_reason": "existing_pkl_read_failed",
                "parser_backend": "none",
                "geometry_sampling_quality": "none",
                "face_count": 0,
                "edge_count": 0,
                "vertex_count": 0,
                "global_scale": "",
                "dtg_train_compatible": 0,
                "dtg_filter_reason": "parse_rejected",
                "error": str(exc),
            },
        }


def _timeout_reject_result(idx: int, step_path: str, step_root: str, source: str, timeout_sec: int) -> Dict[str, Any]:
    return {
        "idx": idx,
        "parse_success": 0,
        "clean": None,
        "rejected": {
            **_future_reject_row(step_path, step_root, source, f"单个 STEP 解析超过 {timeout_sec} 秒，已跳过。"),
            "reject_reason": "parse_timeout",
        },
    }


def _terminate_executor_workers(executor: ProcessPoolExecutor) -> None:
    processes = getattr(executor, "_processes", {}) or {}
    for proc in list(processes.values()):
        try:
            proc.terminate()
        except Exception:
            pass


def parse_abc_dataset(
    step_root: str,
    workdir: str,
    source: str = "abc",
    limit: int = 0,
    max_faces: int = 70,
    num_workers: int = 1,
    task_timeout_sec: int = 900,
) -> Dict[str, Any]:
    dirs = ensure_workdir(workdir)
    step_files = scan_step_files(step_root, limit=limit)
    raw_results: List[Dict[str, Any]] = []
    parse_success_count = 0
    pending_steps: List[Tuple[int, str]] = []

    for idx, step_path in enumerate(step_files, start=1):
        uid = make_uid(step_path, step_root)
        pkl_path = os.path.join(dirs["parsed"], f"{uid}.pkl")
        if os.path.exists(pkl_path):
            result = _result_from_existing_pkl(idx, step_path, step_root, pkl_path, source, max_faces)
            raw_results.append(result)
            parse_success_count += int(result.get("parse_success", 0))
        else:
            pending_steps.append((idx, step_path))

    if raw_results:
        clean_count = sum(1 for item in raw_results if item.get("clean"))
        rejected_count = sum(1 for item in raw_results if item.get("rejected"))
        print(
            f"[parse_abc] 断点续跑：复用已有 pkl {len(raw_results)} 个；"
            f"clean={clean_count} rejected={rejected_count}；待解析={len(pending_steps)}"
        )

    worker_count = max(1, int(num_workers or 1))
    worker_count = min(worker_count, max(len(pending_steps), 1))
    timeout_sec = max(60, int(task_timeout_sec or 900))

    if worker_count <= 1:
        for done_idx, (idx, step_path) in enumerate(pending_steps, start=1):
            result = _parse_one_step_record(idx, step_path, step_root, dirs["parsed"], source, max_faces)
            raw_results.append(result)
            parse_success_count += int(result.get("parse_success", 0))
            if done_idx % 250 == 0 or done_idx == len(pending_steps):
                clean_count = sum(1 for item in raw_results if item.get("clean"))
                rejected_count = sum(1 for item in raw_results if item.get("rejected"))
                print(f"[parse_abc] 已处理 {len(raw_results)}/{len(step_files)}；clean={clean_count} rejected={rejected_count}")
    else:
        print(
            f"[parse_abc] 已启用并行解析：进程数={worker_count}，"
            f"待解析文件数={len(pending_steps)}，单任务超时={timeout_sec}s"
        )
        queue = list(pending_steps)
        completed_pending = 0
        while queue:
            active: Dict[Any, Tuple[int, str, float]] = {}
            executor = ProcessPoolExecutor(max_workers=worker_count)
            last_progress = time.time()
            timed_out_batch = False
            try:
                while queue or active:
                    while queue and len(active) < worker_count:
                        idx, step_path = queue.pop(0)
                        future = executor.submit(_parse_one_step_record, idx, step_path, step_root, dirs["parsed"], source, max_faces)
                        active[future] = (idx, step_path, time.time())
                    if not active:
                        break
                    done, _ = wait(active.keys(), timeout=5.0, return_when=FIRST_COMPLETED)
                    now = time.time()
                    if not done:
                        stalled = [(future, meta) for future, meta in active.items() if now - meta[2] >= timeout_sec]
                        if stalled:
                            timed_out_items = list(active.items())
                            for future, (idx, step_path, _) in timed_out_items:
                                raw_results.append(_timeout_reject_result(idx, step_path, step_root, source, timeout_sec))
                                active.pop(future, None)
                                completed_pending += 1
                            _terminate_executor_workers(executor)
                            timed_out_batch = True
                            print(
                                f"[parse_abc] 检测到 worker 超时，当前在途 {len(timed_out_items)} 个 STEP 已记入 rejected，并重启 worker 池。"
                            )
                            break
                        if now - last_progress >= min(timeout_sec, 300):
                            print(f"[parse_abc] 等待 worker 返回中：已处理 {len(raw_results)}/{len(step_files)}")
                            last_progress = now
                        continue

                    for future in done:
                        idx, step_path, _ = active.pop(future)
                        try:
                            result = future.result()
                        except Exception as exc:
                            result = {
                                "idx": idx,
                                "parse_success": 0,
                                "clean": None,
                                "rejected": _future_reject_row(step_path, step_root, source, str(exc)),
                            }
                        raw_results.append(result)
                        parse_success_count += int(result.get("parse_success", 0))
                        completed_pending += 1
                    last_progress = time.time()
                    if completed_pending % 250 == 0 or len(raw_results) == len(step_files):
                        clean_count = sum(1 for item in raw_results if item.get("clean"))
                        rejected_count = sum(1 for item in raw_results if item.get("rejected"))
                        print(
                            f"[parse_abc] 已处理 {len(raw_results)}/{len(step_files)}；"
                            f"clean={clean_count} rejected={rejected_count}"
                        )
            finally:
                if timed_out_batch:
                    try:
                        executor.shutdown(wait=True, cancel_futures=True)
                    except TypeError:
                        executor.shutdown(wait=True)
                    except Exception:
                        pass
                else:
                    try:
                        executor.shutdown(wait=True, cancel_futures=True)
                    except TypeError:
                        executor.shutdown(wait=True)

    raw_results = sorted(raw_results, key=lambda item: int(item.get("idx", 0)))
    clean_rows = [item["clean"] for item in raw_results if item.get("clean")]
    rejected_rows = [item["rejected"] for item in raw_results if item.get("rejected")]

    clean_manifest = os.path.join(dirs["parsed"], "clean_manifest.csv")
    rejected_manifest = os.path.join(dirs["parsed"], "rejected_manifest.csv")
    write_csv(clean_manifest, clean_rows, _manifest_fieldnames())
    write_csv(rejected_manifest, rejected_rows, _rejected_fieldnames())
    write_csv(os.path.join(dirs["reports"], "clean_manifest.csv"), clean_rows, _manifest_fieldnames())
    write_csv(os.path.join(dirs["reports"], "rejected_manifest.csv"), rejected_rows, _rejected_fieldnames())
    summary = _write_parse_report(workdir, len(step_files), clean_rows, rejected_rows, parse_success_count, max_faces, worker_count)
    return {
        "step_root": normalize_path(step_root),
        "clean_manifest": clean_manifest,
        "rejected_manifest": rejected_manifest,
        "records": clean_rows,
        "rejected": rejected_rows,
        "summary": summary,
    }


def rebuild_manifest_from_existing_pkl(
    step_root: str,
    workdir: str,
    source: str = "abc",
    limit: int = 0,
    max_faces: int = 70,
) -> Dict[str, Any]:
    """只根据已有 pkl 重建 clean/rejected manifest，不重新解析 STEP。"""
    dirs = ensure_workdir(workdir)
    step_files = scan_step_files(step_root, limit=limit)
    raw_results: List[Dict[str, Any]] = []
    unresolved_rows: List[Dict[str, Any]] = []
    parse_success_count = 0

    for idx, step_path in enumerate(step_files, start=1):
        uid = make_uid(step_path, step_root)
        pkl_path = os.path.join(dirs["parsed"], f"{uid}.pkl")
        if not os.path.exists(pkl_path):
            alt_uid = make_uid(step_path, None)
            alt_pkl_path = os.path.join(dirs["parsed"], f"{alt_uid}.pkl")
            if os.path.exists(alt_pkl_path):
                uid = alt_uid
                pkl_path = alt_pkl_path
        if os.path.exists(pkl_path):
            result = _result_from_existing_pkl(idx, step_path, step_root, pkl_path, source, max_faces)
            raw_results.append(result)
            parse_success_count += int(result.get("parse_success", 0))
        else:
            unresolved_rows.append(
                {
                    "uid": uid,
                    "source": source,
                    "step_path": normalize_path(step_path),
                    "stage": "unresolved",
                    "reject_reason": "no_existing_pkl",
                    "parser_backend": "unknown",
                    "geometry_sampling_quality": "unknown",
                    "face_count": 0,
                    "edge_count": 0,
                    "vertex_count": 0,
                    "global_scale": "",
                    "dtg_train_compatible": 0,
                    "dtg_filter_reason": "not_reconstructed",
                    "error": "未重新解析 STEP；无法判断是否过滤失败、解析失败或中断未完成。",
                }
            )
        if idx % 1000 == 0:
            clean_count = sum(1 for item in raw_results if item.get("clean"))
            print(f"[rebuild_manifest] 已扫描 {idx}/{len(step_files)}；existing_clean={clean_count}")

    raw_results = sorted(raw_results, key=lambda item: int(item.get("idx", 0)))
    clean_rows = [item["clean"] for item in raw_results if item.get("clean")]
    rejected_rows = [item["rejected"] for item in raw_results if item.get("rejected")]

    clean_manifest = os.path.join(dirs["parsed"], "clean_manifest.csv")
    rejected_manifest = os.path.join(dirs["parsed"], "rejected_manifest.csv")
    unresolved_manifest = os.path.join(dirs["parsed"], "unresolved_manifest.csv")
    write_csv(clean_manifest, clean_rows, _manifest_fieldnames())
    write_csv(rejected_manifest, rejected_rows, _rejected_fieldnames())
    write_csv(unresolved_manifest, unresolved_rows, _rejected_fieldnames())
    write_csv(os.path.join(dirs["reports"], "clean_manifest.csv"), clean_rows, _manifest_fieldnames())
    write_csv(os.path.join(dirs["reports"], "rejected_manifest.csv"), rejected_rows, _rejected_fieldnames())
    write_csv(os.path.join(dirs["reports"], "unresolved_manifest.csv"), unresolved_rows, _rejected_fieldnames())
    summary = _write_parse_report(
        workdir,
        len(step_files),
        clean_rows,
        rejected_rows,
        parse_success_count,
        max_faces,
        0,
        unresolved_rows=unresolved_rows,
    )
    summary["manifest_rebuilt_from_existing_pkl"] = True
    summary["unresolved_manifest"] = unresolved_manifest
    write_json(os.path.join(dirs["reports"], "parse_summary.json"), summary)
    return {
        "step_root": normalize_path(step_root),
        "clean_manifest": clean_manifest,
        "rejected_manifest": rejected_manifest,
        "unresolved_manifest": unresolved_manifest,
        "records": clean_rows,
        "rejected": rejected_rows,
        "unresolved": unresolved_rows,
        "summary": summary,
    }
