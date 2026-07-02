# -*- coding: utf-8 -*-
"""Enhanced procedural dataset generator for innovation1 v2."""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from utils_io import (
    ENHANCED_PART_TYPES,
    NODE_TYPES,
    PARAMETER_KEYS,
    RELATION_TYPES,
    blank_parameters,
    count_by_key,
    ensure_workdir,
    write_csv,
    write_json,
    write_text,
)

try:
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
    from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeFillet
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakePrism
    from OCC.Core.GC import GC_MakeArcOfCircle
    from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Vec
    from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer
    from OCC.Core.StlAPI import StlAPI_Writer
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_SOLID
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods

    OCC_AVAILABLE = True
except Exception:  # pragma: no cover - reported at runtime
    OCC_AVAILABLE = False


def _box(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float):
    eps = 1e-3
    lo = (min(x0, x1), min(y0, y1), min(z0, z1))
    hi = (max(x0, x1), max(y0, y1), max(z0, z1))
    hi = tuple(max(hi[i], lo[i] + eps) for i in range(3))
    return BRepPrimAPI_MakeBox(gp_Pnt(*lo), gp_Pnt(*hi)).Shape()


def _unify(shape: Any):
    try:
        unify = ShapeUpgrade_UnifySameDomain(shape, True, True, True)
        unify.Build()
        return unify.Shape()
    except Exception:
        return shape


def _fuse_all(shapes: Sequence[Any]):
    if not shapes:
        raise ValueError("No shapes to fuse.")
    shape = shapes[0]
    for other in shapes[1:]:
        shape = BRepAlgoAPI_Fuse(shape, other).Shape()
    return _unify(shape)


def _cut_all(shape: Any, cutters: Sequence[Any]):
    result = shape
    for cutter in cutters:
        result = BRepAlgoAPI_Cut(result, cutter).Shape()
    return _unify(result)


def _wire_from_points(points: Sequence[Tuple[float, float, float]]):
    wire = BRepBuilderAPI_MakeWire()
    pnts = [gp_Pnt(float(x), float(y), float(z)) for x, y, z in points]
    for idx in range(len(pnts)):
        wire.Add(BRepBuilderAPI_MakeEdge(pnts[idx], pnts[(idx + 1) % len(pnts)]).Edge())
    return wire.Wire()


def _same_point(a: Tuple[float, float, float], b: Tuple[float, float, float], tol: float = 1e-7) -> bool:
    return math.dist(a, b) <= tol


def _add_line_edge(wire: Any, p0: Tuple[float, float, float], p1: Tuple[float, float, float]) -> None:
    if _same_point(p0, p1):
        return
    wire.Add(BRepBuilderAPI_MakeEdge(gp_Pnt(*p0), gp_Pnt(*p1)).Edge())


def _add_arc_edge_3p(
    wire: Any,
    p0: Tuple[float, float, float],
    pm: Tuple[float, float, float],
    p1: Tuple[float, float, float],
) -> None:
    if _same_point(p0, p1):
        return
    try:
        wire.Add(BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(gp_Pnt(*p0), gp_Pnt(*pm), gp_Pnt(*p1)).Value()).Edge())
    except Exception:
        _add_line_edge(wire, p0, p1)


def _section_face_from_wire(wire: Any):
    return BRepBuilderAPI_MakeFace(wire.Wire()).Face()


def _loft_sections(sections: Sequence[Sequence[Tuple[float, float, float]]]):
    loft = BRepOffsetAPI_ThruSections(True, False, 1e-6)
    for section in sections:
        loft.AddWire(_wire_from_points(section))
    loft.Build()
    return _unify(loft.Shape())


def _loft_wires(wires: Sequence[Any]):
    loft = BRepOffsetAPI_ThruSections(True, False, 1e-6)
    for wire in wires:
        loft.AddWire(wire)
    loft.Build()
    return _unify(loft.Shape())


def _loft_between_sections(section0: Sequence[Tuple[float, float, float]], section1: Sequence[Tuple[float, float, float]]):
    return _loft_sections([section0, section1])


def _loft_between_wires(wire0: Any, wire1: Any):
    return _loft_wires([wire0, wire1])


def _solid_count(shape: Any) -> int:
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    count = 0
    while explorer.More():
        count += 1
        explorer.Next()
    return count


def _apply_transition_fillets(shape: Any, params: Dict[str, float]) -> Tuple[Any, bool, str]:
    thickness = float(params.get("thickness", 0.0) or 0.0)
    requested = float(params.get("fillet_radius", 0.0) or 0.0)
    radius = min(requested, max(0.35, 0.25 * thickness))
    if radius <= 0.2:
        return shape, False, "fillet radius too small"
    try:
        maker = BRepFilletAPI_MakeFillet(shape)
        explorer = TopExp_Explorer(shape, TopAbs_EDGE)
        edge_count = 0
        while explorer.More():
            maker.Add(radius, topods.Edge(explorer.Current()))
            edge_count += 1
            explorer.Next()
        if edge_count == 0:
            return shape, False, "no edges for fillet"
        maker.Build()
        new_shape = _unify(maker.Shape())
        if _solid_count(new_shape) != 1:
            return shape, False, "fillet did not preserve single-solid topology"
        return new_shape, True, f"applied radius={radius:.3f} on {edge_count} edges"
    except Exception as exc:
        return shape, False, str(exc)


def _save_step(shape: Any, filepath: str) -> None:
    solids = _solid_count(shape)
    if solids != 1:
        raise RuntimeError(f"Generated shape must contain exactly one solid, found {solids}")
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(filepath)
    if status != 1:
        raise RuntimeError(f"OCC STEP write failed with status {status}")


def _save_stl(shape: Any, filepath: str) -> None:
    mesh = BRepMesh_IncrementalMesh(shape, 0.5)
    mesh.Perform()
    writer = StlAPI_Writer()
    writer.Write(shape, filepath)


def _cylindrical_hole(x: float, z: float, radius: float, thickness: float, cut_height: float | None = None):
    cut_height = cut_height or (3.0 * thickness)
    axis = gp_Ax2(gp_Pnt(x, -thickness, z), gp_Dir(0, 1, 0))
    return BRepPrimAPI_MakeCylinder(axis, radius, cut_height).Shape()


def _rounded_rect_cutout(
    x: float,
    z: float,
    hole_width: float,
    hole_height: float,
    thickness: float,
    corner_radius: float,
):
    r = min(max(corner_radius, 0.1), 0.45 * min(hole_width, hole_height))
    x0, x1 = x - hole_width / 2.0, x + hole_width / 2.0
    z0, z1 = z - hole_height / 2.0, z + hole_height / 2.0
    y = -thickness
    wire = BRepBuilderAPI_MakeWire()

    def p(px: float, pz: float) -> gp_Pnt:
        return gp_Pnt(px, y, pz)

    line_pairs = [
        (p(x0 + r, z0), p(x1 - r, z0)),
        (p(x1, z0 + r), p(x1, z1 - r)),
        (p(x1 - r, z1), p(x0 + r, z1)),
        (p(x0, z1 - r), p(x0, z0 + r)),
    ]
    arc_triplets = [
        (p(x1 - r, z0), p(x1, z0), p(x1, z0 + r)),
        (p(x1, z1 - r), p(x1, z1), p(x1 - r, z1)),
        (p(x0 + r, z1), p(x0, z1), p(x0, z1 - r)),
        (p(x0, z0 + r), p(x0, z0), p(x0 + r, z0)),
    ]
    for idx in range(4):
        wire.Add(BRepBuilderAPI_MakeEdge(line_pairs[idx][0], line_pairs[idx][1]).Edge())
        wire.Add(BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(*arc_triplets[idx]).Value()).Edge())
    face = BRepBuilderAPI_MakeFace(wire.Wire()).Face()
    return BRepPrimAPI_MakePrism(face, gp_Vec(0, 3.0 * thickness, 0)).Shape()


def _rib_centers(
    width: float,
    rib_count: int,
    asymmetric: bool,
    rng: random.Random,
    rib_width: float = 0.0,
    offset_ratio: float = 0.0,
) -> List[float]:
    if rib_count <= 0:
        return []
    if not asymmetric:
        return [-width / 2.0 + (i + 1) * width / (rib_count + 1) for i in range(rib_count)]
    min_spacing = max(1.5 * rib_width, 1.0)
    margin = max(rib_width, width * 0.12)
    usable = max(width - 2.0 * margin, min_spacing * max(rib_count - 1, 1))
    centers = [
        -width / 2.0 + margin + i * usable / max(rib_count - 1, 1)
        for i in range(rib_count)
    ]
    skew = offset_ratio * 0.25 * width
    for i in range(rib_count):
        direction = -1.0 if i % 2 == 0 else 1.0
        centers[i] += skew + direction * rng.uniform(0.04 * width, 0.10 * width)
        centers[i] = min(max(centers[i], -width / 2.0 + margin), width / 2.0 - margin)
    centers = sorted(centers)
    for i in range(1, len(centers)):
        if centers[i] - centers[i - 1] < min_spacing:
            centers[i] = centers[i - 1] + min_spacing
    overflow = centers[-1] - (width / 2.0 - margin)
    if overflow > 0:
        centers = [c - overflow for c in centers]
    return centers


def _runout_rib_shape(
    center_x: float,
    rib_width: float,
    base_y: float,
    rib_height: float,
    length: float,
    runout_length: float,
):
    x0 = center_x - rib_width / 2.0
    x1 = center_x + rib_width / 2.0
    small_h = max(0.08 * rib_height, 1.0)
    full_h = rib_height
    z0 = 0.0
    z1 = min(max(runout_length, 1.0), 0.45 * length)
    z2 = max(length - z1, z1 + 1.0)
    z3 = length

    def section(z: float, height: float):
        return [
            (x0, base_y, z),
            (x1, base_y, z),
            (x1, base_y + height, z),
            (x0, base_y + height, z),
        ]

    return _loft_sections(
        [
            section(z0, small_h),
            section(z1, full_h),
            section(z2, full_h),
            section(z3, small_h),
        ]
    )


def _safe_root_radius(root_radius: float, rib_width: float, rib_height: float, spacing: float | None = None) -> float:
    limits = [0.42 * rib_width, 0.42 * rib_height]
    if spacing is not None and spacing > rib_width:
        limits.append(0.35 * (spacing - rib_width))
    limit = max(0.0, min(limits)) if limits else 0.0
    return max(0.0, min(float(root_radius), limit))


def _ribbed_panel_wire(
    width: float,
    thickness: float,
    centers: Sequence[float],
    rib_width: float,
    rib_heights: Sequence[float],
    root_radius: float,
    z: float,
):
    left_bound = -width / 2.0
    right_bound = width / 2.0
    ribs = sorted(
        [
            (float(cx), max(float(h), 0.0))
            for cx, h in zip(centers, rib_heights)
            if h > 0.2 and left_bound < cx < right_bound
        ],
        key=lambda item: item[0],
    )
    if ribs:
        gaps = [ribs[i + 1][0] - ribs[i][0] for i in range(len(ribs) - 1)]
        spacing = min(gaps) if gaps else None
        root_radius = _safe_root_radius(root_radius, rib_width, min(h for _, h in ribs), spacing)
    else:
        root_radius = 0.0

    wire = BRepBuilderAPI_MakeWire()
    p = (left_bound, 0.0, z)
    _add_line_edge(wire, p, (right_bound, 0.0, z))
    p = (right_bound, 0.0, z)
    _add_line_edge(wire, p, (right_bound, thickness, z))
    p = (right_bound, thickness, z)

    for cx, rib_h in reversed(ribs):
        x0 = cx - rib_width / 2.0
        x1 = cx + rib_width / 2.0
        r = _safe_root_radius(root_radius, rib_width, rib_h)
        y_root = thickness + min(r, max(rib_h - 0.2, 0.0))
        _add_line_edge(wire, p, (min(right_bound, x1 + r), thickness, z))
        if r > 0.2 and x1 + r <= right_bound + 1e-6 and y_root < thickness + rib_h:
            _add_arc_edge_3p(
                wire,
                (x1 + r, thickness, z),
                (x1 + r - 0.707107 * r, thickness + r - 0.707107 * r, z),
                (x1, thickness + r, z),
            )
            p = (x1, thickness + r, z)
        else:
            _add_line_edge(wire, p, (x1, thickness, z))
            p = (x1, thickness, z)
        _add_line_edge(wire, p, (x1, thickness + rib_h, z))
        p = (x1, thickness + rib_h, z)
        _add_line_edge(wire, p, (x0, thickness + rib_h, z))
        p = (x0, thickness + rib_h, z)
        if r > 0.2 and x0 - r >= left_bound - 1e-6 and y_root < thickness + rib_h:
            _add_line_edge(wire, p, (x0, thickness + r, z))
            _add_arc_edge_3p(
                wire,
                (x0, thickness + r, z),
                (x0 - r + 0.707107 * r, thickness + r - 0.707107 * r, z),
                (x0 - r, thickness, z),
            )
            p = (x0 - r, thickness, z)
        else:
            _add_line_edge(wire, p, (x0, thickness, z))
            p = (x0, thickness, z)

    _add_line_edge(wire, p, (left_bound, thickness, z))
    p = (left_bound, thickness, z)
    _add_line_edge(wire, p, (left_bound, 0.0, z))
    return wire.Wire()


def _ribbed_panel_prism(
    length: float,
    width: float,
    thickness: float,
    centers: Sequence[float],
    rib_width: float,
    rib_heights: Sequence[float],
    root_radius: float,
):
    wire = _ribbed_panel_wire(width, thickness, centers, rib_width, rib_heights, root_radius, 0.0)
    face = BRepBuilderAPI_MakeFace(wire).Face()
    return _unify(BRepPrimAPI_MakePrism(face, gp_Vec(0, 0, length)).Shape())


def _ribbed_panel_loft_runout(
    length: float,
    width: float,
    thickness: float,
    centers: Sequence[float],
    rib_width: float,
    rib_height: float,
    root_radius: float,
    runout_length: float,
):
    z1 = min(max(runout_length, 1.0), 0.45 * length)
    z2 = max(length - z1, z1 + 1.0)
    toe_h = max(0.12 * rib_height, min(root_radius + 0.5 * thickness, 0.35 * rib_height), 1.0)
    sections = [
        _ribbed_panel_wire(width, thickness, centers, rib_width, [toe_h] * len(centers), root_radius, 0.0),
        _ribbed_panel_wire(width, thickness, centers, rib_width, [rib_height] * len(centers), root_radius, z1),
        _ribbed_panel_wire(width, thickness, centers, rib_width, [rib_height] * len(centers), root_radius, z2),
        _ribbed_panel_wire(width, thickness, centers, rib_width, [toe_h] * len(centers), root_radius, length),
    ]
    return _loft_wires(sections)


def _panel_with_ribs(params: Dict[str, float], rng: random.Random, asymmetric: bool = False, runout: bool = False):
    length = params["length"]
    width = params["width"]
    thickness = params["thickness"]
    rib_count = int(params["rib_count"])
    rib_width = params["rib_width"]
    rib_height = params["rib_height"]
    centers = _rib_centers(width, rib_count, asymmetric, rng, rib_width=rib_width, offset_ratio=params.get("offset_ratio", 0.0))
    root_radius = params.get("root_fillet_radius", 0.0) or params.get("fillet_radius", 0.0)
    if not centers:
        return _box(-width / 2, width / 2, 0, thickness, 0, length)
    if runout:
        runout_len = max(params["runout_length"], 0.12 * length)
        return _ribbed_panel_loft_runout(length, width, thickness, centers, rib_width, rib_height, root_radius, runout_len)
    rib_heights: List[float] = []
    for _idx, _cx in enumerate(centers):
        height_scale = rng.uniform(0.75, 1.15) if asymmetric else 1.0
        rib_heights.append(rib_height * height_scale)
    return _ribbed_panel_prism(length, width, thickness, centers, rib_width, rib_heights, root_radius)


def _arc_point(radius: float, theta: float, z: float, center_radius: float | None = None) -> Tuple[float, float, float]:
    center_radius = radius if center_radius is None else center_radius
    return (radius * math.sin(theta), radius * math.cos(theta) - center_radius, z)


def _curved_panel_shell(length: float, width: float, thickness: float, radius: float, sweep_angle: float = 0.0):
    radius = max(radius, width * 1.5)
    if sweep_angle > 0.0:
        half_angle = math.radians(min(max(sweep_angle, 5.0), 35.0)) / 2.0
    else:
        half_angle = min(width / (2.0 * radius), math.radians(28.0))
    inner_r = radius
    outer_r = radius + thickness

    outer_l = gp_Pnt(*_arc_point(outer_r, -half_angle, 0.0, inner_r))
    outer_m = gp_Pnt(*_arc_point(outer_r, 0.0, 0.0, inner_r))
    outer_rp = gp_Pnt(*_arc_point(outer_r, half_angle, 0.0, inner_r))
    inner_rp = gp_Pnt(*_arc_point(inner_r, half_angle, 0.0, inner_r))
    inner_m = gp_Pnt(*_arc_point(inner_r, 0.0, 0.0, inner_r))
    inner_l = gp_Pnt(*_arc_point(inner_r, -half_angle, 0.0, inner_r))

    wire = BRepBuilderAPI_MakeWire()
    wire.Add(BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(outer_l, outer_m, outer_rp).Value()).Edge())
    wire.Add(BRepBuilderAPI_MakeEdge(outer_rp, inner_rp).Edge())
    wire.Add(BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(inner_rp, inner_m, inner_l).Value()).Edge())
    wire.Add(BRepBuilderAPI_MakeEdge(inner_l, outer_l).Edge())
    face = BRepBuilderAPI_MakeFace(wire.Wire()).Face()
    return _unify(BRepPrimAPI_MakePrism(face, gp_Vec(0, 0, length)).Shape())


def _add_arc_edge(wire: Any, radius: float, theta0: float, theta1: float, center_radius: float | None = None) -> None:
    if abs(theta1 - theta0) < 1e-6:
        return
    p0 = gp_Pnt(*_arc_point(radius, theta0, 0.0, center_radius))
    pm = gp_Pnt(*_arc_point(radius, 0.5 * (theta0 + theta1), 0.0, center_radius))
    p1 = gp_Pnt(*_arc_point(radius, theta1, 0.0, center_radius))
    wire.Add(BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(p0, pm, p1).Value()).Edge())


def _curved_stiffened_panel_shell(params: Dict[str, float], rng: random.Random):
    length = params["length"]
    width = params["width"]
    thickness = params["thickness"]
    radius = max(params["curvature_radius"], width * 1.5)
    outer_r = radius + thickness
    half_angle = math.radians(min(max(params.get("sweep_angle", 0.0), 5.0), 35.0)) / 2.0
    rib_w = params["rib_width"]
    rib_h = params["rib_height"]
    root_r = max(params.get("root_fillet_radius", 0.0) or params.get("fillet_radius", 0.0), 0.0)
    centers = _rib_centers(width, int(params["rib_count"]), False, rng, rib_width=rib_w)

    intervals = []
    for cx in centers:
        center_theta = math.asin(max(min(cx / outer_r, 0.95), -0.95))
        half_w = max(rib_w / (2.0 * outer_r), math.radians(0.25))
        a = max(-half_angle, center_theta - half_w)
        b = min(half_angle, center_theta + half_w)
        if b > a:
            intervals.append((a, b))
    intervals.sort()

    merged = []
    for item in intervals:
        if not merged or item[0] > merged[-1][1]:
            merged.append([item[0], item[1]])
        else:
            merged[-1][1] = max(merged[-1][1], item[1])

    wire = BRepBuilderAPI_MakeWire()
    current = -half_angle
    for a, b in merged:
        root_theta = min(root_r / max(outer_r, 1e-6), 0.25 * (b - a))
        trans_a = min(a, max(current, a - root_theta))
        trans_b = max(b, min(half_angle, b + root_theta))
        _add_arc_edge(wire, outer_r, current, trans_a, radius)
        outer_a = gp_Pnt(*_arc_point(outer_r, trans_a, 0.0, radius))
        root_a = gp_Pnt(*_arc_point(outer_r + min(root_r, rib_h * 0.4), a, 0.0, radius))
        top_a = gp_Pnt(*_arc_point(outer_r + rib_h, a, 0.0, radius))
        top_b = gp_Pnt(*_arc_point(outer_r + rib_h, b, 0.0, radius))
        root_b = gp_Pnt(*_arc_point(outer_r + min(root_r, rib_h * 0.4), b, 0.0, radius))
        outer_b = gp_Pnt(*_arc_point(outer_r, trans_b, 0.0, radius))
        wire.Add(BRepBuilderAPI_MakeEdge(outer_a, root_a).Edge())
        wire.Add(BRepBuilderAPI_MakeEdge(root_a, top_a).Edge())
        _add_arc_edge(wire, outer_r + rib_h, a, b, radius)
        wire.Add(BRepBuilderAPI_MakeEdge(top_b, root_b).Edge())
        wire.Add(BRepBuilderAPI_MakeEdge(root_b, outer_b).Edge())
        current = trans_b
    _add_arc_edge(wire, outer_r, current, half_angle, radius)

    outer_right = gp_Pnt(*_arc_point(outer_r, half_angle, 0.0, radius))
    inner_right = gp_Pnt(*_arc_point(radius, half_angle, 0.0, radius))
    inner_left = gp_Pnt(*_arc_point(radius, -half_angle, 0.0, radius))
    outer_left = gp_Pnt(*_arc_point(outer_r, -half_angle, 0.0, radius))
    wire.Add(BRepBuilderAPI_MakeEdge(outer_right, inner_right).Edge())
    _add_arc_edge(wire, radius, half_angle, -half_angle, radius)
    wire.Add(BRepBuilderAPI_MakeEdge(inner_left, outer_left).Edge())

    face = BRepBuilderAPI_MakeFace(wire.Wire()).Face()
    return _unify(BRepPrimAPI_MakePrism(face, gp_Vec(0, 0, length)).Shape())


def _curved_panel(params: Dict[str, float], rng: random.Random, stiffened: bool = False):
    length = params["length"]
    width = params["width"]
    thickness = params["thickness"]
    radius = max(params["curvature_radius"], width * 1.5)
    if stiffened:
        return _curved_stiffened_panel_shell(params, rng)
    return _curved_panel_shell(length, width, thickness, radius, params.get("sweep_angle", 0.0))


def _c_channel_wire(h: float, f: float, t: float, radius: float, z: float):
    xv = t / 2.0
    r = min(max(radius, 0.0), max(0.0, f - xv - 0.5), max(0.0, 0.42 * (h - 2.0 * t)))
    wire = BRepBuilderAPI_MakeWire()
    p = (-t / 2.0, 0.0, z)
    for q in [(f, 0.0, z), (f, t, z)]:
        _add_line_edge(wire, p, q)
        p = q
    if r > 0.2:
        q = (xv + r, t, z)
        _add_line_edge(wire, p, q)
        _add_arc_edge_3p(
            wire,
            q,
            (xv + r - 0.707107 * r, t + r - 0.707107 * r, z),
            (xv, t + r, z),
        )
        p = (xv, t + r, z)
        q = (xv, h - t - r, z)
        _add_line_edge(wire, p, q)
        _add_arc_edge_3p(
            wire,
            q,
            (xv + r - 0.707107 * r, h - t - r + 0.707107 * r, z),
            (xv + r, h - t, z),
        )
        p = (xv + r, h - t, z)
    else:
        for q in [(xv, t, z), (xv, h - t, z)]:
            _add_line_edge(wire, p, q)
            p = q
    for q in [(f, h - t, z), (f, h, z), (-t / 2.0, h, z), (-t / 2.0, 0.0, z)]:
        _add_line_edge(wire, p, q)
        p = q
    return wire.Wire()


def _tapered_c_channel(params: Dict[str, float]):
    length = params["length"]
    thickness = params["thickness"]
    h0 = params.get("height_start", params["height"])
    h1 = params.get("height_end", params["height"])
    f0 = params.get("flange_width_start", params["flange_width"])
    f1 = params.get("flange_width_end", params["flange_width"])
    radius = params.get("fillet_radius", 0.0)
    return _loft_between_wires(
        _c_channel_wire(h0, f0, thickness, radius, 0.0),
        _c_channel_wire(h1, f1, thickness, radius, length),
    )


def _hat_wire(h: float, f: float, cap_width: float, t: float, radius: float, z: float):
    left = -cap_width / 2.0
    right = cap_width / 2.0
    r = min(
        max(radius, 0.0),
        max(0.0, f - t - 0.5),
        max(0.0, 0.32 * h),
        max(0.0, 0.20 * cap_width),
    )
    wire = BRepBuilderAPI_MakeWire()
    p = (left - f, 0.0, z)
    for q in [(right + f, 0.0, z), (right + f, t, z)]:
        _add_line_edge(wire, p, q)
        p = q
    if r > 0.2:
        q = (right + t + r, t, z)
        _add_line_edge(wire, p, q)
        _add_arc_edge_3p(
            wire,
            q,
            (right + t + r - 0.707107 * r, t + r - 0.707107 * r, z),
            (right + t, t + r, z),
        )
        p = (right + t, t + r, z)
        q = (right + t, h - r, z)
        _add_line_edge(wire, p, q)
        _add_arc_edge_3p(
            wire,
            q,
            (right + t - r + 0.707107 * r, h - r + 0.707107 * r, z),
            (right + t - r, h, z),
        )
        p = (right + t - r, h, z)
        q = (left - t + r, h, z)
        _add_line_edge(wire, p, q)
        _add_arc_edge_3p(
            wire,
            q,
            (left - t + r - 0.707107 * r, h - r + 0.707107 * r, z),
            (left - t, h - r, z),
        )
        p = (left - t, h - r, z)
        q = (left - t, t + r, z)
        _add_line_edge(wire, p, q)
        _add_arc_edge_3p(
            wire,
            q,
            (left - t - r + 0.707107 * r, t + r - 0.707107 * r, z),
            (left - t - r, t, z),
        )
        p = (left - t - r, t, z)
    else:
        for q in [(right + t, t, z), (right + t, h, z), (left - t, h, z), (left - t, t, z)]:
            _add_line_edge(wire, p, q)
            p = q
    for q in [(left - f, t, z), (left - f, 0.0, z)]:
        _add_line_edge(wire, p, q)
        p = q
    return wire.Wire()


def _tapered_hat(params: Dict[str, float]):
    length = params["length"]
    cap_width = params.get("cap_width", params["width"])
    thickness = params["thickness"]
    return _loft_between_wires(
        _hat_wire(
            params.get("height_start", params["height"]),
            params.get("flange_width_start", params["flange_width"]),
            cap_width,
            thickness,
            params.get("fillet_radius", 0.0),
            0.0,
        ),
        _hat_wire(
            params.get("height_end", params["height"]),
            params.get("flange_width_end", params["flange_width"]),
            cap_width,
            thickness,
            params.get("fillet_radius", 0.0),
            length,
        ),
    )


def _sample_parameters(part_type: str, rng: random.Random) -> Dict[str, float]:
    p = blank_parameters()

    def ru(a: float, b: float, ndigits: int = 1) -> float:
        return round(rng.uniform(a, b), ndigits)

    def ri(a: int, b: int) -> int:
        return int(rng.randint(a, b))

    def set_fillet(local_min: float, allow_zero: bool = False, max_mult: float = 8.0) -> None:
        t = p["thickness"]
        if allow_zero and rng.random() < 0.25:
            p["fillet_radius"] = 0.0
            return
        lo = max(1.5 * t, 3.0)
        hi = min(max_mult * t, 0.25 * max(local_min, lo + 1.0))
        if hi < lo:
            hi = lo
        p["fillet_radius"] = round(rng.uniform(lo, hi), 2)

    def set_root_fillet(local_min: float) -> None:
        t = p["thickness"]
        lo = max(2.0 * t, 3.0)
        hi = min(5.0 * t, 0.25 * max(local_min, lo + 1.0))
        if hi < lo:
            hi = lo
        p["root_fillet_radius"] = round(rng.uniform(lo, hi), 2)

    def set_cutout_corner(local_min: float) -> None:
        t = p["thickness"]
        lo = max(2.0 * t, 3.0)
        hi = min(6.0 * t, 0.20 * max(local_min, lo + 1.0))
        if hi < lo:
            hi = lo
        p["cutout_corner_radius"] = round(rng.uniform(lo, hi), 2)
        p["fillet_radius"] = p["cutout_corner_radius"]

    def set_ribs(count_min: int, count_max: int, width_range: Tuple[float, float], height_range: Tuple[float, float]) -> None:
        p["rib_count"] = float(ri(count_min, count_max))
        spacing = p["width"] / (int(p["rib_count"]) + 1)
        rib_w_max = min(width_range[1], 0.48 * spacing)
        p["rib_width"] = ru(width_range[0], max(width_range[0], rib_w_max), 1)
        p["rib_height"] = ru(height_range[0], height_range[1], 1)

    def set_curve(width_range: Tuple[float, float], radius_range: Tuple[float, float], sweep_range: Tuple[float, float]) -> None:
        for _ in range(100):
            sweep = rng.uniform(sweep_range[0], sweep_range[1])
            radius = rng.uniform(radius_range[0], radius_range[1])
            width = 2.0 * radius * math.sin(math.radians(sweep) / 2.0)
            if width_range[0] <= width <= width_range[1] and radius >= 1.5 * width:
                p["width"] = round(width, 1)
                p["curvature_radius"] = round(radius, 1)
                p["sweep_angle"] = round(sweep, 2)
                return
        p["width"] = ru(width_range[0], width_range[1], 1)
        p["curvature_radius"] = round(max(radius_range[0], 1.5 * p["width"]), 1)
        p["sweep_angle"] = round(math.degrees(2.0 * math.asin(min(p["width"] / (2.0 * p["curvature_radius"]), 0.95))), 2)

    if part_type == "panel_with_circular_cutout":
        p["length"] = ru(300.0, 1000.0)
        p["width"] = ru(150.0, 600.0)
        p["thickness"] = ru(1.5, 4.5, 2)
        max_r = min(70.0, 0.18 * min(p["length"], p["width"]))
        min_r = max(10.0, 4.0 * p["thickness"])
        p["hole_radius"] = ru(min_r, max(min_r, max_r), 1)
        usable_len = max(0.0, p["length"] - 5.0 * p["hole_radius"])
        max_count = max(1, min(4, int(usable_len // (3.0 * p["hole_radius"])) + 1))
        p["hole_count"] = float(ri(1, max_count))
        p["fillet_radius"] = 0.0

    elif part_type == "panel_with_rectangular_cutout":
        p["length"] = ru(300.0, 1000.0)
        p["width"] = ru(150.0, 600.0)
        p["thickness"] = ru(1.5, 4.5, 2)
        p["hole_width"] = ru(30.0, min(220.0, 0.35 * p["length"]), 1)
        p["hole_height"] = ru(25.0, min(160.0, 0.35 * p["width"]), 1)
        p["hole_count"] = 1.0
        set_cutout_corner(min(p["hole_width"], p["hole_height"]))

    elif part_type == "stiffened_panel_with_cutout":
        p["length"] = ru(400.0, 1200.0)
        p["width"] = ru(200.0, 800.0)
        p["thickness"] = ru(1.8, 5.0, 2)
        set_ribs(1, 4, (12.0, 60.0), (20.0, min(100.0, 0.35 * p["width"])))
        max_r = min(60.0, 0.15 * min(p["length"], p["width"]))
        min_r = max(10.0, 4.0 * p["thickness"])
        p["hole_radius"] = ru(min_r, max(min_r, max_r), 1)
        p["hole_count"] = float(ri(1, 3))
        set_fillet(min(p["rib_width"], p["rib_height"], p["width"]), max_mult=8.0)
        set_root_fillet(min(p["rib_width"], p["rib_height"]))

    elif part_type == "multi_stiffened_panel":
        p["length"] = ru(400.0, 1200.0)
        p["width"] = ru(250.0, 900.0)
        p["thickness"] = ru(1.8, 5.0, 2)
        set_ribs(2, 6, (12.0, 70.0), (20.0, 120.0))
        p["flange_width"] = ru(15.0, 80.0)
        set_fillet(min(p["rib_width"], p["rib_height"], p["flange_width"]), max_mult=8.0)
        set_root_fillet(min(p["rib_width"], p["rib_height"]))

    elif part_type == "asymmetric_stiffened_panel":
        p["length"] = ru(400.0, 1200.0)
        p["width"] = ru(250.0, 900.0)
        p["thickness"] = ru(1.8, 5.0, 2)
        set_ribs(2, 5, (12.0, 65.0), (20.0, 110.0))
        p["offset_ratio"] = round(rng.uniform(-0.25, 0.25), 3)
        set_fillet(min(p["rib_width"], p["rib_height"]), max_mult=8.0)
        set_root_fillet(min(p["rib_width"], p["rib_height"]))

    elif part_type == "tapered_c_channel":
        p["length"] = ru(300.0, 1000.0)
        p["thickness"] = ru(1.8, 5.0, 2)
        p["height_start"] = ru(40.0, 150.0)
        p["taper_ratio"] = round(rng.uniform(0.6, 1.4), 3)
        p["height_end"] = round(max(25.0, min(140.0, p["height_start"] * p["taper_ratio"])), 1)
        p["flange_width_start"] = ru(20.0, 100.0)
        ratio_f = rng.uniform(0.6, 1.4)
        p["flange_width_end"] = round(max(15.0, min(90.0, p["flange_width_start"] * ratio_f)), 1)
        p["height"] = round(0.5 * (p["height_start"] + p["height_end"]), 1)
        p["flange_width"] = round(0.5 * (p["flange_width_start"] + p["flange_width_end"]), 1)
        p["width"] = round(max(p["flange_width_start"], p["flange_width_end"]) + p["thickness"], 1)
        set_fillet(min(p["height_end"], p["flange_width_end"]), max_mult=8.0)

    elif part_type == "tapered_hat_stiffener":
        p["length"] = ru(300.0, 1000.0)
        p["thickness"] = ru(1.8, 5.0, 2)
        p["width"] = ru(80.0, 300.0)
        p["height_start"] = ru(30.0, 120.0)
        p["taper_ratio"] = round(rng.uniform(0.6, 1.4), 3)
        p["height_end"] = round(max(20.0, min(110.0, p["height_start"] * p["taper_ratio"])), 1)
        p["flange_width_start"] = ru(15.0, 80.0)
        ratio_f = rng.uniform(0.6, 1.4)
        p["flange_width_end"] = round(max(12.0, min(70.0, p["flange_width_start"] * ratio_f)), 1)
        p["cap_width"] = ru(max(25.0, 1.2 * p["thickness"]), min(140.0, p["width"]))
        p["height"] = round(0.5 * (p["height_start"] + p["height_end"]), 1)
        p["flange_width"] = round(0.5 * (p["flange_width_start"] + p["flange_width_end"]), 1)
        set_fillet(min(p["height_end"], p["flange_width_end"], p["cap_width"]), max_mult=8.0)

    elif part_type == "curved_panel":
        p["length"] = ru(300.0, 1200.0)
        p["thickness"] = ru(1.5, 5.0, 2)
        set_curve((150.0, 800.0), (600.0, 4000.0), (5.0, 35.0))
        set_fillet(p["width"], allow_zero=True, max_mult=5.0)

    elif part_type == "curved_stiffened_panel":
        p["length"] = ru(400.0, 1200.0)
        p["thickness"] = ru(1.8, 5.0, 2)
        set_curve((200.0, 800.0), (800.0, 5000.0), (5.0, 35.0))
        set_ribs(1, 4, (12.0, 60.0), (20.0, min(90.0, 0.25 * p["curvature_radius"])))
        set_fillet(min(p["rib_width"], p["rib_height"], p["width"]), max_mult=8.0)
        set_root_fillet(min(p["rib_width"], p["rib_height"]))

    elif part_type == "stiffener_runout_panel":
        p["length"] = ru(500.0, 1200.0)
        p["width"] = ru(200.0, 800.0)
        p["thickness"] = ru(1.8, 5.0, 2)
        p["rib_count"] = float(ri(1, 3))
        spacing = p["width"] / (int(p["rib_count"]) + 1)
        p["rib_width"] = ru(12.0, min(70.0, 0.48 * spacing), 1)
        max_runout = min(300.0, 0.35 * p["length"])
        max_rib_h = min(120.0, max_runout / 3.0)
        p["rib_height"] = ru(25.0, max(25.0, max_rib_h), 1)
        p["runout_length"] = ru(max(60.0, 3.0 * p["rib_height"]), max_runout, 1)
        p["notch_depth"] = ru(5.0, min(60.0, 0.5 * p["rib_height"]), 1)
        set_fillet(min(p["rib_width"], p["rib_height"]), max_mult=8.0)
        set_root_fillet(min(p["rib_width"], p["rib_height"]))

    else:
        raise ValueError(f"Unsupported enhanced part type: {part_type}")

    if p["height"] > 0:
        max_t = 0.08 * min(p["width"] if p["width"] > 0 else 1e9, p["height"])
        if max_t > 0 and p["thickness"] >= max_t:
            p["thickness"] = round(max(1.5, 0.92 * max_t), 2)
    feasible_t_limits: List[float] = []
    if p["fillet_radius"] > 0.0:
        vals = [p["width"], p["height"], p["height_end"], p["flange_width"], p["flange_width_end"], p["rib_width"], p["hole_width"], p["hole_height"]]
        local_vals = [v for v in vals if v > 0.0]
        if local_vals:
            feasible_t_limits.append(min(local_vals) / 6.1)
    if p["root_fillet_radius"] > 0.0:
        local_vals = [v for v in [p["rib_width"], p["rib_height"]] if v > 0.0]
        if local_vals:
            feasible_t_limits.append(min(local_vals) / 8.1)
    if p["cutout_corner_radius"] > 0.0:
        local_vals = [v for v in [p["hole_width"], p["hole_height"]] if v > 0.0]
        if local_vals:
            feasible_t_limits.append(min(local_vals) / 10.1)
    if feasible_t_limits:
        p["thickness"] = round(max(1.5, min(p["thickness"], min(feasible_t_limits))), 2)
    t = p["thickness"]
    if p["fillet_radius"] > 0.0:
        local = min(v for v in [p["width"], p["height"], p["height_end"], p["flange_width"], p["flange_width_end"], p["rib_width"], p["hole_width"], p["hole_height"]] if v > 0.0)
        p["fillet_radius"] = round(min(max(p["fillet_radius"], 1.5 * t, 3.0), 8.0 * t, 0.25 * local), 2)
    if p["root_fillet_radius"] > 0.0:
        local = min(v for v in [p["rib_width"], p["rib_height"]] if v > 0.0)
        p["root_fillet_radius"] = round(min(max(p["root_fillet_radius"], 2.0 * t, 3.0), 5.0 * t, 0.25 * local), 2)
    if p["cutout_corner_radius"] > 0.0:
        local = min(v for v in [p["hole_width"], p["hole_height"]] if v > 0.0)
        p["cutout_corner_radius"] = round(min(max(p["cutout_corner_radius"], 2.0 * t, 3.0), 6.0 * t, 0.20 * local), 2)
        p["fillet_radius"] = p["cutout_corner_radius"]
    return {key: float(p.get(key, 0.0)) for key in PARAMETER_KEYS}


def _z_positions_with_spacing(length: float, count: int, margin: float, spacing: float) -> List[float]:
    if count <= 0:
        return []
    usable0 = margin
    usable1 = length - margin
    if count == 1:
        return [0.5 * (usable0 + usable1)]
    required = spacing * (count - 1)
    available = max(usable1 - usable0, required)
    start = 0.5 * (usable0 + usable1 - required)
    return [start + i * spacing for i in range(count)]


def _circular_hole_positions(
    params: Dict[str, float],
    rng: random.Random,
    avoid_x_centers: Sequence[float] | None = None,
    avoid_width: float = 0.0,
) -> List[Tuple[float, float]]:
    count = int(params.get("hole_count", 0))
    if count <= 0:
        return []
    length = params["length"]
    width = params["width"]
    radius = params["hole_radius"]
    margin = 2.5 * radius
    spacing = 3.0 * radius
    z_values = _z_positions_with_spacing(length, count, margin, spacing)

    x_candidates = [0.0]
    if avoid_x_centers:
        blocked = sorted(float(x) for x in avoid_x_centers)
        left = -width / 2.0 + margin
        right = width / 2.0 - margin
        boundaries = [left] + blocked + [right]
        x_candidates = []
        for idx in range(len(boundaries) - 1):
            a = boundaries[idx]
            b = boundaries[idx + 1]
            if idx > 0:
                a += 0.5 * avoid_width + radius
            if idx < len(boundaries) - 2:
                b -= 0.5 * avoid_width + radius
            if b - a >= 2.0 * radius:
                x_candidates.append(0.5 * (a + b))
        if not x_candidates:
            x_candidates = [0.0]
    return [(x_candidates[i % len(x_candidates)], z_values[i]) for i in range(count)]


def _rect_cutout_positions(params: Dict[str, float], rng: random.Random) -> List[Tuple[float, float]]:
    length = params["length"]
    width = params["width"]
    x_margin = 0.15 * width + 0.5 * params["hole_width"]
    z_margin = 0.15 * length + 0.5 * params["hole_height"]
    x = rng.uniform(-width / 2.0 + x_margin, width / 2.0 - x_margin) if width > 2.0 * x_margin else 0.0
    z = rng.uniform(z_margin, length - z_margin) if length > 2.0 * z_margin else 0.5 * length
    return [(x, z)]


def _validate_parameter_constraints(part_type: str, params: Dict[str, float]) -> Dict[str, Any]:
    notes: List[str] = []

    def fail(msg: str) -> None:
        notes.append(msg)

    length = params.get("length", 0.0)
    width = params.get("width", 0.0)
    height = params.get("height", 0.0)
    thickness = params.get("thickness", 0.0)
    local_height = height or params.get("height_end", 0.0) or params.get("height_start", 0.0)

    if not (250.0 <= length <= 1200.0):
        fail(f"length outside global range: {length:.3f}")
    if width > 0.0 and not (80.0 <= width <= 900.0):
        if part_type not in {"tapered_c_channel"}:
            fail(f"width outside global range: {width:.3f}")
    if not (1.5 <= thickness <= 5.0):
        fail(f"thickness outside global range: {thickness:.3f}")
    thin_ref = min(v for v in [width, local_height] if v > 0.0) if any(v > 0.0 for v in [width, local_height]) else 0.0
    if thin_ref > 0.0 and thickness >= 0.08 * thin_ref:
        fail(f"thickness violates thin-wall ratio: t={thickness:.3f}, ref={thin_ref:.3f}")

    fillet = params.get("fillet_radius", 0.0)
    if fillet > 0.0:
        if fillet < 1.5 * thickness:
            fail(f"fillet_radius < 1.5t: r={fillet:.3f}, t={thickness:.3f}")
        if fillet > 8.0 * thickness:
            fail(f"fillet_radius > 8t: r={fillet:.3f}, t={thickness:.3f}")
        local_dims = [
            params.get("height", 0.0),
            params.get("height_end", 0.0),
            params.get("flange_width", 0.0),
            params.get("flange_width_end", 0.0),
            params.get("rib_width", 0.0),
            params.get("hole_width", 0.0),
            params.get("hole_height", 0.0),
        ]
        local_dims = [v for v in local_dims if v > 0.0]
        if local_dims and fillet > 0.25 * min(local_dims):
            fail(f"fillet_radius exceeds 0.25 local min: r={fillet:.3f}, local={min(local_dims):.3f}")

    root_fillet = params.get("root_fillet_radius", 0.0)
    if root_fillet > 0.0:
        if root_fillet < 2.0 * thickness:
            fail(f"root_fillet_radius < 2t: r={root_fillet:.3f}, t={thickness:.3f}")
        if root_fillet > 5.0 * thickness:
            fail(f"root_fillet_radius > 5t: r={root_fillet:.3f}, t={thickness:.3f}")
        root_dims = [params.get("rib_width", 0.0), params.get("rib_height", 0.0)]
        root_dims = [v for v in root_dims if v > 0.0]
        if root_dims and root_fillet > 0.25 * min(root_dims):
            fail(f"root_fillet_radius exceeds 0.25 root local min: r={root_fillet:.3f}, local={min(root_dims):.3f}")

    cutout_corner = params.get("cutout_corner_radius", 0.0)
    if cutout_corner > 0.0:
        if cutout_corner < 2.0 * thickness:
            fail(f"cutout_corner_radius < 2t: r={cutout_corner:.3f}, t={thickness:.3f}")
        if cutout_corner > 6.0 * thickness:
            fail(f"cutout_corner_radius > 6t: r={cutout_corner:.3f}, t={thickness:.3f}")
        if params.get("hole_width", 0.0) > 0.0 and params.get("hole_height", 0.0) > 0.0:
            local = min(params["hole_width"], params["hole_height"])
            if cutout_corner > 0.20 * local:
                fail(f"cutout_corner_radius exceeds 0.2 cutout min: r={cutout_corner:.3f}, local={local:.3f}")

    hole_radius = params.get("hole_radius", 0.0)
    hole_count = int(params.get("hole_count", 0.0))
    if hole_radius > 0.0:
        if hole_radius < 4.0 * thickness:
            fail(f"hole_radius < 4t: r={hole_radius:.3f}, t={thickness:.3f}")
        if width > 0.0 and width < 5.0 * hole_radius:
            fail(f"width cannot satisfy 2.5R side margin: width={width:.3f}, R={hole_radius:.3f}")
        if length < 5.0 * hole_radius:
            fail(f"length cannot satisfy 2.5R end margin: length={length:.3f}, R={hole_radius:.3f}")
        if hole_count > 1 and length < 5.0 * hole_radius + 3.0 * hole_radius * (hole_count - 1):
            fail(f"multi-hole spacing capacity insufficient: count={hole_count}, length={length:.3f}, R={hole_radius:.3f}")

    rib_count = int(params.get("rib_count", 0.0))
    rib_width = params.get("rib_width", 0.0)
    rib_height = params.get("rib_height", 0.0)
    if rib_count > 0 and width > 0.0 and rib_width > 0.0:
        spacing = width / (rib_count + 1)
        if spacing < 2.0 * rib_width:
            fail(f"rib_spacing < 2*rib_width: spacing={spacing:.3f}, rib_width={rib_width:.3f}")

    runout = params.get("runout_length", 0.0)
    if runout > 0.0:
        if runout < 3.0 * rib_height:
            fail(f"runout_length < 3*rib_height: runout={runout:.3f}, rib_height={rib_height:.3f}")
        if runout > 0.35 * length:
            fail(f"runout_length > 0.35*length: runout={runout:.3f}, length={length:.3f}")

    if part_type.startswith("tapered_"):
        if params.get("height_end", 0.0) < 20.0:
            fail(f"height_end below minimum: {params.get('height_end', 0.0):.3f}")
        if params.get("flange_width_end", 0.0) < 12.0:
            fail(f"flange_width_end below minimum: {params.get('flange_width_end', 0.0):.3f}")

    if "curved" in part_type:
        curv = params.get("curvature_radius", 0.0)
        if curv < 1.5 * width:
            fail(f"curvature_radius < 1.5*width: R={curv:.3f}, width={width:.3f}")

    return {
        "valid": len(notes) == 0,
        "notes": notes,
    }


def _build_graph(part_type: str, params: Dict[str, float], mechanisms: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    nodes: List[Dict[str, Any]] = [{"id": "panel_0", "type": "panel"}]
    relations: List[Dict[str, Any]] = []

    def add_node(node_id: str, node_type: str) -> None:
        if node_type not in NODE_TYPES:
            raise ValueError(f"Unsupported node type: {node_type}")
        nodes.append({"id": node_id, "type": node_type})

    def add_rel(source: str, target: str, rel_type: str) -> None:
        if rel_type not in RELATION_TYPES:
            raise ValueError(f"Unsupported relation type: {rel_type}")
        relations.append({"source": source, "target": target, "type": rel_type})

    if part_type in {"tapered_c_channel", "tapered_hat_stiffener"}:
        add_node("web_0", "web")
        add_rel("web_0", "panel_0", "connected_to")
        for i in range(2):
            add_node(f"flange_{i}", "flange")
            add_rel(f"flange_{i}", "web_0", "attached_to")
            add_rel(f"flange_{i}", "web_0", "parallel_to")
        for i in range(2):
            add_node(f"transition_web_flange_{i}", "transition")
            add_rel(f"transition_web_flange_{i}", "web_0", "smooth_connected")
            add_rel(f"transition_web_flange_{i}", f"flange_{i}", "smooth_connected")
        if part_type == "tapered_hat_stiffener":
            add_node("transition_web_cap_0", "transition")
            add_rel("transition_web_cap_0", "web_0", "smooth_connected")
            add_node("transition_web_cap_1", "transition")
            add_rel("transition_web_cap_1", "web_0", "smooth_connected")

    if part_type in {
        "stiffened_panel_with_cutout",
        "multi_stiffened_panel",
        "asymmetric_stiffened_panel",
        "curved_stiffened_panel",
        "stiffener_runout_panel",
    }:
        for i in range(int(params["rib_count"])):
            add_node(f"stiffener_{i}", "stiffener")
            add_rel(f"stiffener_{i}", "panel_0", "attached_to")
            if "root_fillet_transition" in mechanisms:
                add_node(f"transition_root_{i}", "transition")
                add_rel(f"transition_root_{i}", f"stiffener_{i}", "smooth_connected")
                add_rel(f"transition_root_{i}", "panel_0", "smooth_connected")
            if i > 0:
                add_rel(f"stiffener_{i}", f"stiffener_{i-1}", "parallel_to")

    if part_type in {"panel_with_circular_cutout", "stiffened_panel_with_cutout"}:
        for i in range(int(params["hole_count"])):
            add_node(f"hole_{i}", "hole")
            add_rel(f"hole_{i}", "panel_0", "hole_of")

    if part_type == "panel_with_rectangular_cutout":
        for i in range(int(params["hole_count"])):
            add_node(f"cutout_{i}", "cutout")
            add_rel(f"cutout_{i}", "panel_0", "cutout_of")

    if part_type == "stiffener_runout_panel":
        for i in range(int(params["rib_count"])):
            add_node(f"runout_{i}", "runout")
            add_rel(f"runout_{i}", f"stiffener_{i}", "runout_of")
            add_rel(f"runout_{i}", f"stiffener_{i}", "smooth_connected")
            add_rel(f"runout_{i}", "panel_0", "attached_to")
            add_rel(f"runout_{i}", "panel_0", "smooth_connected")

    for side in ["start", "end"]:
        add_node(f"boundary_{side}", "boundary")
        add_rel(f"boundary_{side}", "panel_0", "boundary_of")

    if "curved_surface" in mechanisms:
        add_node("transition_curvature_0", "transition")
        add_rel("transition_curvature_0", "panel_0", "smooth_connected")
    if "fillet_transition" in mechanisms:
        for node in [n["id"] for n in nodes if n["type"] in {"cutout", "hole"}]:
            tid = f"transition_{node}"
            add_node(tid, "transition")
            add_rel(tid, node, "smooth_connected")
            add_rel(tid, "panel_0", "smooth_connected")

    return {"nodes": nodes, "relations": relations}


def _face_groups_from_graph(graph: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    return [
        {
            "node_id": node["id"],
            "role": node["type"],
            "face_ids": [],
            "assignment_status": "pending_parse_alignment",
        }
        for node in graph.get("nodes", [])
    ]


def _build_shape(part_type: str, params: Dict[str, float], rng: random.Random):
    length = params["length"]
    width = params["width"]
    thickness = params["thickness"]
    mechanisms: List[str] = []

    if part_type == "panel_with_circular_cutout":
        mechanisms = ["hole", "inner_loop"]
        base = _box(-width / 2, width / 2, 0, thickness, 0, length)
        cutters = [_cylindrical_hole(x, z, params["hole_radius"], thickness) for x, z in _circular_hole_positions(params, rng)]
        shape = _cut_all(base, cutters)
    elif part_type == "panel_with_rectangular_cutout":
        mechanisms = ["cutout", "inner_loop", "fillet_transition", "cutout_corner_fillet"]
        base = _box(-width / 2, width / 2, 0, thickness, 0, length)
        cutters = [
            _rounded_rect_cutout(
                x,
                z,
                params["hole_width"],
                params["hole_height"],
                thickness,
                params.get("cutout_corner_radius", params["fillet_radius"]),
            )
            for x, z in _rect_cutout_positions(params, rng)
        ]
        shape = _cut_all(base, cutters)
    elif part_type == "stiffened_panel_with_cutout":
        mechanisms = ["hole", "inner_loop", "stiffener", "root_fillet_transition"]
        if int(params["rib_count"]) > 1:
            mechanisms.append("multi_stiffener")
        panel = _panel_with_ribs(params, rng)
        cut_height = thickness + params["rib_height"] + 2.0 * thickness
        rib_centers = _rib_centers(width, int(params["rib_count"]), False, rng, rib_width=params["rib_width"])
        cutters = [
            _cylindrical_hole(x, z, params["hole_radius"], thickness, cut_height=cut_height)
            for x, z in _circular_hole_positions(params, rng, avoid_x_centers=rib_centers, avoid_width=params["rib_width"])
        ]
        shape = _cut_all(panel, cutters)
    elif part_type == "multi_stiffened_panel":
        mechanisms = ["stiffener", "multi_stiffener", "root_fillet_transition"]
        shape = _panel_with_ribs(params, rng)
    elif part_type == "asymmetric_stiffened_panel":
        mechanisms = ["stiffener", "multi_stiffener", "asymmetric_layout", "root_fillet_transition"]
        shape = _panel_with_ribs(params, rng, asymmetric=True)
    elif part_type == "tapered_c_channel":
        mechanisms = ["taper", "flange", "web", "fillet_transition", "web_flange_fillet"]
        shape = _tapered_c_channel(params)
    elif part_type == "tapered_hat_stiffener":
        mechanisms = ["taper", "flange", "web", "fillet_transition", "web_flange_fillet", "web_cap_fillet"]
        shape = _tapered_hat(params)
    elif part_type == "curved_panel":
        mechanisms = ["curved_surface"]
        shape = _curved_panel(params, rng, stiffened=False)
    elif part_type == "curved_stiffened_panel":
        mechanisms = ["curved_surface", "stiffener", "multi_stiffener", "root_fillet_transition"]
        shape = _curved_panel(params, rng, stiffened=True)
    elif part_type == "stiffener_runout_panel":
        mechanisms = ["stiffener", "runout", "root_fillet_transition"]
        shape = _panel_with_ribs(params, rng, runout=True)
    else:
        raise ValueError(f"Unsupported enhanced part type: {part_type}")
    if part_type == "panel_with_rectangular_cutout":
        return shape, mechanisms, {
            "fillet_applied": True,
            "fillet_note": f"explicit rounded-rectangle cutout corner radius={params.get('cutout_corner_radius', params['fillet_radius']):.3f}",
        }

    fillet_safe_types = set()
    if part_type in fillet_safe_types:
        shape, fillet_applied, fillet_note = _apply_transition_fillets(shape, params)
    else:
        fillet_applied = False
        if "root_fillet_transition" in mechanisms or "web_flange_fillet" in mechanisms or "web_cap_fillet" in mechanisms:
            fillet_applied = True
            fillet_note = "explicit section-level root/web-flange/web-cap transition geometry; no global all-edge fillet"
        else:
            fillet_note = "automatic OCC fillet skipped; no required structural transition for this part type"
    if fillet_applied and "fillet_transition" not in mechanisms:
        mechanisms.append("fillet_transition")
    return shape, mechanisms, {"fillet_applied": fillet_applied, "fillet_note": fillet_note}


def _cleanup_dataset_dir(dataset_dir: str) -> None:
    root = Path(dataset_dir)
    for suffix in ("*.step", "*.stl", "*.json"):
        for path in root.glob(suffix):
            path.unlink()


def generate_enhanced_dataset(
    workdir: str,
    num_per_type: int = 50,
    seed: int = 42,
    part_types: Sequence[str] | None = None,
) -> Dict[str, Any]:
    if not OCC_AVAILABLE:
        raise RuntimeError("pythonOCC is not available; enhanced STEP/STL generation cannot run.")

    dirs = ensure_workdir(workdir)
    dataset_dir = dirs["enhanced_dataset"]
    reports_dir = dirs["reports"]
    _cleanup_dataset_dir(dataset_dir)

    rng = random.Random(seed)
    part_types = list(part_types or ENHANCED_PART_TYPES)
    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for part_type in part_types:
        for idx in range(num_per_type):
            uid = f"{part_type}_{idx:04d}"
            step_path = os.path.join(dataset_dir, f"{uid}.step")
            stl_path = os.path.join(dataset_dir, f"{uid}.stl")
            json_path = os.path.join(dataset_dir, f"{uid}.json")
            params = _sample_parameters(part_type, rng)
            try:
                constraint_audit = _validate_parameter_constraints(part_type, params)
                if not constraint_audit["valid"]:
                    raise ValueError("parameter constraint violation: " + "; ".join(constraint_audit["notes"]))
                shape, mechanisms, geometry_build = _build_shape(part_type, params, rng)
                graph = _build_graph(part_type, params, mechanisms)
                face_groups = _face_groups_from_graph(graph)
                _save_step(shape, step_path)
                _save_stl(shape, stl_path)
                sample = {
                    "uid": uid,
                    "part_type": part_type,
                    "source": "procedural_enhanced",
                    "parameters": params,
                    "configuration_graph": graph,
                    "procedural_face_groups": face_groups,
                    "topology_mechanisms": mechanisms,
                    "parameter_constraint_audit": constraint_audit,
                    "schema": {
                        "parameter_keys": PARAMETER_KEYS,
                        "node_types": NODE_TYPES,
                        "relation_types": RELATION_TYPES,
                    },
                    "geometry_note": (
                        "Curved panels use circular-arc swept solids; straight stiffened panels use section-level "
                        "root fillet arcs; tapered C/hat parts use section-level web-flange/web-cap arcs; "
                        "runout ribs use lofted closed sections."
                    ),
                    "geometry_build": geometry_build,
                }
                write_json(json_path, sample)
                records.append(
                    {
                        "uid": uid,
                        "part_type": part_type,
                        "step_success": 1,
                        "stl_success": 1,
                        "json_success": 1,
                        "node_count": len(graph["nodes"]),
                        "relation_count": len(graph["relations"]),
                        "topology_mechanisms": ";".join(mechanisms),
                        "constraint_valid": int(constraint_audit["valid"]),
                        "constraint_notes": "; ".join(constraint_audit["notes"]),
                        "error": "",
                    }
                )
            except Exception as exc:
                failures.append({"uid": uid, "part_type": part_type, "error": str(exc)})
                records.append(
                    {
                        "uid": uid,
                        "part_type": part_type,
                        "step_success": 0,
                        "stl_success": 0,
                        "json_success": 0,
                        "node_count": 0,
                        "relation_count": 0,
                        "topology_mechanisms": "",
                        "constraint_valid": 0,
                        "constraint_notes": "",
                        "error": str(exc),
                    }
                )

    manifest_path = os.path.join(reports_dir, "auxiliary", "enhanced_manifest.csv")
    write_csv(
        manifest_path,
        records,
        [
            "uid",
            "part_type",
            "step_success",
            "stl_success",
            "json_success",
            "node_count",
            "relation_count",
            "topology_mechanisms",
            "constraint_valid",
            "constraint_notes",
            "error",
        ],
    )

    type_counts = count_by_key([r for r in records if r["json_success"]], "part_type")
    mechanism_counts: Dict[str, int] = {}
    for rec in records:
        for mech in str(rec.get("topology_mechanisms", "")).split(";"):
            if mech:
                mechanism_counts[mech] = mechanism_counts.get(mech, 0) + 1
    good = [r for r in records if r["json_success"]]
    constraint_valid_count = sum(int(r.get("constraint_valid", 0)) for r in good)
    avg_nodes = sum(float(r["node_count"]) for r in good) / max(len(good), 1)
    avg_relations = sum(float(r["relation_count"]) for r in good) / max(len(good), 1)
    report_lines = [
        "Innovation1 v2 Enhanced Dataset Report",
        "=" * 72,
        f"Total requested samples: {len(records)}",
        f"Generated JSON samples: {sum(int(r['json_success']) for r in records)}",
        f"STEP success: {sum(int(r['step_success']) for r in records)}",
        f"STL success: {sum(int(r['stl_success']) for r in records)}",
        f"Failures: {len(failures)}",
        f"Parameter constraint valid samples: {constraint_valid_count} / {len(good)}",
        "",
        "Per-type sample counts:",
    ]
    for part_type in part_types:
        report_lines.append(f"  - {part_type}: {type_counts.get(part_type, 0)}")
    report_lines.extend(["", "Topology mechanism counts:"])
    for mech, count in sorted(mechanism_counts.items()):
        report_lines.append(f"  - {mech}: {count}")
    report_lines.extend(
        [
            "",
            f"Average node count: {avg_nodes:.3f}",
            f"Average relation count: {avg_relations:.3f}",
            "",
            "Modeling note:",
            "  Curved panels are circular-arc swept solids; straight stiffened panels use explicit section-level root fillet arcs.",
            "  Tapered C/hat parts use section-level web-flange/web-cap arcs; runout ribs use lofted closed-section transition solids.",
            "  Rectangular cutouts use rounded-corner cutting wires; global all-edge fillets are intentionally avoided.",
            "  Boolean cut/fuse outputs are checked to contain exactly one solid before STEP export.",
            "  They are intended to strengthen structural-topology supervision, not to claim exhaustive aerospace composite CAD coverage.",
        ]
    )
    if failures:
        report_lines.extend(["", "Failure details:"])
        for item in failures[:50]:
            report_lines.append(f"  - {item['uid']}: {item['error']}")

    report_path = os.path.join(reports_dir, "enhanced_dataset_report.txt")
    write_text(report_path, report_lines)

    return {
        "records": records,
        "failures": failures,
        "manifest_path": manifest_path,
        "report_path": report_path,
        "dataset_dir": dataset_dir,
    }
