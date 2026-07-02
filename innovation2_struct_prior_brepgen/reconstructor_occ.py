# -*- coding: utf-8 -*-
"""
程序名称：reconstructor_occ.py
程序功能：
    本程序基于 OpenCascade (pythonOCC) 几何造型内核实现结构约束下的 CAD 重建器 (ConstructiveBRepReconstructor)。
    它负责将模型预测出的归一化参数向量 P*，经由边界修补与合理性修正逻辑，还原为真实尺寸，
    并实例化生成对应的 3D B-Rep (STEP) 和三角网格面 (STL) 模型。
    本程序是第二创新点“模型预测参数 -> 物理实体模型”落地重建的核心通路。

主要模块功能：
    1. repair_parameters: 针对模型预测出的 9 维参数进行航空规范性与几何拓扑合理性修复，
       防止出现自相交、负值或超出加工物理极限的几何尺寸。
    2. 核心几何生成器 (OCC Generators):
       - create_l_angle: 重建 L 型梁几何。
       - create_c_channel: 重建 C 型梁几何。
       - create_z_beam: 重建 Z 型梁几何。
       - create_hat_stiffener: 重建帽型加筋梁几何。
       - create_stiffened_panel: 重建加筋蒙皮壁板几何。
    3. reconstruct_brep_occ: 主调度重建接口，将参数转换并写入目标 step、stl 和 json 文件。

使用方法：
    由主运行程序 run_innovation2.py 内部导入并调用。
"""

import os
import sys
import numpy as np

# OpenCascade 建模标准导入
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakePrism
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.GC import GC_MakeArcOfCircle
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh

def intersect_lines(p1, n1, p2, n2):
    """求两条二维射线的交点 (帽型件截面偏置计算辅助)"""
    c1 = p1[0]*n1[0] + p1[1]*n1[1]
    c2 = p2[0]*n2[0] + p2[1]*n2[1]
    det = n1[0]*n2[1] - n1[1]*n2[0]
    if abs(det) < 1e-6:
        return p1
    x = (c1*n2[1] - c2*n1[1]) / det
    y = (n1[0]*c2 - n2[0]*c1) / det
    return np.array([x, y])

def repair_parameters(part_type, raw_params):
    """
    根据航空复材薄壁件物理制造与几何拓扑约束，对模型预测出的原始参数进行合理性修复。
    输入:
        part_type: 零件类别字符串
        raw_params: 原始预测的参数字典，形如 {"length": ..., "thickness": ...}
    返回:
        repaired_params: 修复后的参数字典
        num_repairs: 发生修复的参数个数
        repair_notes: 修复内容文字注记
    """
    repaired = raw_params.copy()
    num_repairs = 0
    repair_notes = []

    def check_and_clip(key, min_val, max_val):
        nonlocal num_repairs
        val = repaired.get(key, 0.0)
        clipped = clip_val(val, min_val, max_val)
        if abs(clipped - val) > 1e-3:
            repaired[key] = clipped
            num_repairs += 1
            repair_notes.append(f"{key}: {val:.2f} -> {clipped:.2f} (clipped)")

    # 1. 执行边界硬性裁剪
    check_and_clip("thickness", 1.8, 3.5)
    check_and_clip("length", 120.0, 500.0)
    check_and_clip("width", 30.0, 220.0)
    check_and_clip("height", 20.0, 120.0)
    check_and_clip("flange_width", 15.0, 80.0)
    check_and_clip("rib_width", 8.0, 50.0)
    check_and_clip("rib_height", 10.0, 100.0)

    # 筋条数限制为离散整数且在 [0, 5]
    r_count = raw_params.get("rib_count", 0)
    repaired_rc = int(np.clip(round(r_count), 0, 5))
    # 对于除 stiffened_panel 以外的零件，rib_count 强制设为 0
    if part_type != "stiffened_panel":
        repaired_rc = 0

    if abs(repaired_rc - r_count) > 1e-3:
        repaired["rib_count"] = repaired_rc
        num_repairs += 1
        repair_notes.append(f"rib_count: {r_count:.2f} -> {repaired_rc} (discrete constraint)")

    # 2. 关系几何约束修复 (圆角半径与厚度、高度、翻边宽度的关联约束)
    t = repaired["thickness"]
    r = raw_params.get("fillet_radius", 1.5 * t)

    # 最小圆角必须为 1.5t (防复材应力集中断裂)
    min_r = 1.5 * t
    
    # 最大圆角约束，防止圆角过大导致壁板几何自相交
    max_r = 0.25 * 50.0 # 默认安全值
    if part_type == "l_angle":
        max_r = 0.25 * min(repaired["width"], repaired["height"])
    elif part_type == "c_channel":
        max_r = 0.25 * min(repaired["flange_width"], repaired["height"])
    elif part_type == "z_beam":
        max_r = 0.25 * min(repaired["flange_width"], repaired["height"])
    elif part_type == "hat_stiffener":
        max_r = 0.25 * min(repaired["width"], repaired["height"], repaired["flange_width"])
    elif part_type == "stiffened_panel":
        max_r = 0.25 * min(repaired["width"], repaired["rib_width"])

    r_clipped = np.clip(r, min_r, max_r)
    if abs(r_clipped - r) > 1e-3:
        repaired["fillet_radius"] = round(r_clipped, 2)
        num_repairs += 1
        repair_notes.append(f"fillet_radius: {r:.2f} -> {r_clipped:.2f} (geometric consistency)")

    # 特殊处理：Z-beam 的 width 逻辑上等于 thickness
    if part_type == "z_beam":
        repaired["width"] = t
    # 特殊处理：L-angle 的 flange_width 为 0
    if part_type == "l_angle":
        repaired["flange_width"] = 0.0

    return repaired, num_repairs, "; ".join(repair_notes)

def clip_val(val, min_v, max_v):
    return max(min_v, min(val, max_v))

# ----------------- 下述为 OCC 重建构件逻辑 -----------------

def create_l_angle(length, width, height, thickness, fillet_radius):
    W, H, t, r = width, height, thickness, fillet_radius
    r = min(r, W - t - 0.5, H - t - 0.5)
    r = max(r, 0.1)
        
    edges = []
    edges.append(BRepBuilderAPI_MakeEdge(gp_Pnt(0, H, 0), gp_Pnt(0, 0, 0)).Edge())
    edges.append(BRepBuilderAPI_MakeEdge(gp_Pnt(0, 0, 0), gp_Pnt(W, 0, 0)).Edge())
    edges.append(BRepBuilderAPI_MakeEdge(gp_Pnt(W, 0, 0), gp_Pnt(W, t, 0)).Edge())
    edges.append(BRepBuilderAPI_MakeEdge(gp_Pnt(W, t, 0), gp_Pnt(t+r, t, 0)).Edge())
    
    p1 = gp_Pnt(t+r, t, 0)
    p2 = gp_Pnt(t, t+r, 0)
    pm = gp_Pnt(t+r - r * 0.7071, t+r - r * 0.7071, 0)
    arc = GC_MakeArcOfCircle(p1, pm, p2).Value()
    edges.append(BRepBuilderAPI_MakeEdge(arc).Edge())
    
    edges.append(BRepBuilderAPI_MakeEdge(gp_Pnt(t, t+r, 0), gp_Pnt(t, H, 0)).Edge())
    edges.append(BRepBuilderAPI_MakeEdge(gp_Pnt(t, H, 0), gp_Pnt(0, H, 0)).Edge())
    
    wire = BRepBuilderAPI_MakeWire()
    for e in edges:
        wire.Add(e)
    wire = wire.Wire()
    
    face = BRepBuilderAPI_MakeFace(wire).Face()
    prism = BRepPrimAPI_MakePrism(face, gp_Vec(0, 0, length))
    return prism.Shape()

def create_c_channel(length, width, height, thickness, fillet_radius):
    W, H, t, r = width, height, thickness, fillet_radius
    r = min(r, W/2 - t - 0.5, H - t - 0.5)
    r = max(r, 0.1)
        
    edges = []
    edges.append(BRepBuilderAPI_MakeEdge(gp_Pnt(W, H, 0), gp_Pnt(W, 0, 0)).Edge())
    edges.append(BRepBuilderAPI_MakeEdge(gp_Pnt(W, 0, 0), gp_Pnt(0, 0, 0)).Edge())
    edges.append(BRepBuilderAPI_MakeEdge(gp_Pnt(0, 0, 0), gp_Pnt(0, H, 0)).Edge())
    edges.append(BRepBuilderAPI_MakeEdge(gp_Pnt(0, H, 0), gp_Pnt(t, H, 0)).Edge())
    edges.append(BRepBuilderAPI_MakeEdge(gp_Pnt(t, H, 0), gp_Pnt(t, t+r, 0)).Edge())
    
    p1 = gp_Pnt(t, t+r, 0)
    p2 = gp_Pnt(t+r, t, 0)
    pm = gp_Pnt(t+r - r * 0.7071, t+r - r * 0.7071, 0)
    arc1 = GC_MakeArcOfCircle(p1, pm, p2).Value()
    edges.append(BRepBuilderAPI_MakeEdge(arc1).Edge())
    
    edges.append(BRepBuilderAPI_MakeEdge(gp_Pnt(t+r, t, 0), gp_Pnt(W - t - r, t, 0)).Edge())
    
    p3 = gp_Pnt(W - t - r, t, 0)
    p4 = gp_Pnt(W - t, t+r, 0)
    pm2 = gp_Pnt(W - t - r + r * 0.7071, t+r - r * 0.7071, 0)
    arc2 = GC_MakeArcOfCircle(p3, pm2, p4).Value()
    edges.append(BRepBuilderAPI_MakeEdge(arc2).Edge())
    
    edges.append(BRepBuilderAPI_MakeEdge(gp_Pnt(W - t, t+r, 0), gp_Pnt(W - t, H, 0)).Edge())
    edges.append(BRepBuilderAPI_MakeEdge(gp_Pnt(W - t, H, 0), gp_Pnt(W, H, 0)).Edge())
    
    wire = BRepBuilderAPI_MakeWire()
    for e in edges:
        wire.Add(e)
    wire = wire.Wire()
    
    face = BRepBuilderAPI_MakeFace(wire).Face()
    prism = BRepPrimAPI_MakePrism(face, gp_Vec(0, 0, length))
    return prism.Shape()

def create_z_beam(length, width, height, thickness, fillet_radius, flange_width):
    # Z梁宽度参数内部设定为 t
    W, H, t, r, F = width, height, thickness, fillet_radius, flange_width
    r = min(r, F - 0.5, H - 2*t - 0.5)
    r = max(r, 0.1)
        
    edges = []
    edges.append(BRepBuilderAPI_MakeEdge(gp_Pnt(-F, t, 0), gp_Pnt(-F, 0, 0)).Edge())
    edges.append(BRepBuilderAPI_MakeEdge(gp_Pnt(-F, 0, 0), gp_Pnt(t, 0, 0)).Edge())
    edges.append(BRepBuilderAPI_MakeEdge(gp_Pnt(t, 0, 0), gp_Pnt(t, H - t - r, 0)).Edge())
    
    p1 = gp_Pnt(t, H - t - r, 0)
    p2 = gp_Pnt(t + r, H - t, 0)
    pm1 = gp_Pnt(t + r - r * 0.7071, H - t - r + r * 0.7071, 0)
    arc1 = GC_MakeArcOfCircle(p1, pm1, p2).Value()
    edges.append(BRepBuilderAPI_MakeEdge(arc1).Edge())
    
    edges.append(BRepBuilderAPI_MakeEdge(gp_Pnt(t + r, H - t, 0), gp_Pnt(t + F, H - t, 0)).Edge())
    edges.append(BRepBuilderAPI_MakeEdge(gp_Pnt(t + F, H - t, 0), gp_Pnt(t + F, H, 0)).Edge())
    edges.append(BRepBuilderAPI_MakeEdge(gp_Pnt(t + F, H, 0), gp_Pnt(0, H, 0)).Edge())
    edges.append(BRepBuilderAPI_MakeEdge(gp_Pnt(0, H, 0), gp_Pnt(0, t + r, 0)).Edge())
    
    p3 = gp_Pnt(0, t + r, 0)
    p4 = gp_Pnt(-r, t, 0)
    pm2 = gp_Pnt(-r + r * 0.7071, t + r - r * 0.7071, 0)
    arc2 = GC_MakeArcOfCircle(p3, pm2, p4).Value()
    edges.append(BRepBuilderAPI_MakeEdge(arc2).Edge())
    
    edges.append(BRepBuilderAPI_MakeEdge(gp_Pnt(-r, t, 0), gp_Pnt(-F, t, 0)).Edge())
    
    wire = BRepBuilderAPI_MakeWire()
    for e in edges:
        wire.Add(e)
    wire = wire.Wire()
    
    face = BRepBuilderAPI_MakeFace(wire).Face()
    prism = BRepPrimAPI_MakePrism(face, gp_Vec(0, 0, length))
    return prism.Shape()

def create_hat_stiffener(length, width, height, thickness, flange_width, fillet_radius, beta_deg=80.0):
    C, H, t, F = width, height, thickness, flange_width
    beta = np.radians(beta_deg)
    dx = H / np.tan(beta)
    
    v0 = np.array([-C/2 - dx - F, 0])
    v1 = np.array([-C/2 - dx, 0])
    v2 = np.array([-C/2, H])
    v3 = np.array([C/2, H])
    v4 = np.array([C/2 + dx, 0])
    v5 = np.array([C/2 + dx + F, 0])
    
    p0 = v0 + np.array([0, t])
    n0 = np.array([0, 1])
    
    dir1 = (v2 - v1) / np.linalg.norm(v2 - v1)
    n1_perp = np.array([dir1[1], -dir1[0]])
    if n1_perp[0] < 0:
        n1_perp = -n1_perp
    p1 = v1 + t * n1_perp
    n1 = np.array([-dir1[1], dir1[0]])
    
    p2 = v2 + np.array([0, -t])
    n2 = np.array([0, 1])
    
    dir3 = (v4 - v3) / np.linalg.norm(v4 - v3)
    n3_perp = np.array([dir3[1], -dir3[0]])
    if n3_perp[0] > 0:
        n3_perp = -n3_perp
    p3 = v3 + t * n3_perp
    n3 = np.array([-dir3[1], dir3[0]])
    
    p4 = v4 + np.array([0, t])
    n4 = np.array([0, 1])
    
    i1 = intersect_lines(p0, n0, p1, n1)
    i2 = intersect_lines(p1, n1, p2, n2)
    i3 = intersect_lines(p2, n2, p3, n3)
    i4 = intersect_lines(p3, n3, p4, n4)
    
    poly = [
        gp_Pnt(v0[0], v0[1], 0),
        gp_Pnt(v5[0], v5[1], 0),
        gp_Pnt(v5[0], t, 0),
        gp_Pnt(i4[0], i4[1], 0),
        gp_Pnt(i3[0], i3[1], 0),
        gp_Pnt(i2[0], i2[1], 0),
        gp_Pnt(i1[0], i1[1], 0),
        gp_Pnt(v0[0], t, 0)
    ]
    
    edges = []
    for idx in range(len(poly)):
        edges.append(BRepBuilderAPI_MakeEdge(poly[idx], poly[(idx+1)%len(poly)]).Edge())
        
    wire = BRepBuilderAPI_MakeWire()
    for e in edges:
        wire.Add(e)
    wire = wire.Wire()
    
    face = BRepBuilderAPI_MakeFace(wire).Face()
    prism = BRepPrimAPI_MakePrism(face, gp_Vec(0, 0, length))
    return prism.Shape()

def create_stiffened_panel(length, width, thickness, rib_count, rib_width, rib_height, fillet_radius):
    W, t, L, n, Rw, Rh = width, thickness, length, int(rib_count), rib_width, rib_height
    
    base_panel = BRepPrimAPI_MakeBox(
        gp_Pnt(-W/2, 0, 0),
        gp_Pnt(W/2, t, L)
    ).Shape()
    
    if n == 0:
        return base_panel
        
    shape = base_panel
    for i in range(n):
        x_center = -W/2 + (i + 1) * (W / (n + 1))
        rib = BRepPrimAPI_MakeBox(
            gp_Pnt(x_center - Rw/2, t, 0),
            gp_Pnt(x_center + Rw/2, t + Rh, L)
        ).Shape()
        shape = BRepAlgoAPI_Fuse(shape, rib).Shape()
        
    return shape

def save_step(shape, filepath):
    """保存几何对象为 STEP 实体文件"""
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(filepath)
    if status != 1:
        raise RuntimeError(f"OCC STEP write failed: status {status}")

def save_stl(shape, filepath):
    """保存网格化三角表面为 STL 文件"""
    mesh = BRepMesh_IncrementalMesh(shape, 0.1)
    mesh.Perform()
    writer = StlAPI_Writer()
    writer.Write(shape, filepath)

def reconstruct_brep_occ(part_type, parameters, output_step, output_stl):
    """
    主几何重建调度函数。根据零件类型实例化不同的生成器，并完成 STEP 与 STL 保存。
    """
    length = parameters["length"]
    width = parameters["width"]
    height = parameters["height"]
    thickness = parameters["thickness"]
    fillet_radius = parameters["fillet_radius"]
    flange_width = parameters["flange_width"]
    rib_width = parameters["rib_width"]
    rib_height = parameters["rib_height"]
    rib_count = int(parameters["rib_count"])
    
    # 针对帽型件获取其 beta 参数，未预测则默认 80.0
    beta = parameters.get("beta", 80.0)

    if part_type == "l_angle":
        shape = create_l_angle(length, width, height, thickness, fillet_radius)
    elif part_type == "c_channel":
        shape = create_c_channel(length, width, height, thickness, fillet_radius)
    elif part_type == "z_beam":
        shape = create_z_beam(length, width, height, thickness, fillet_radius, flange_width)
    elif part_type == "hat_stiffener":
        shape = create_hat_stiffener(length, width, height, thickness, flange_width, fillet_radius, beta)
    elif part_type == "stiffened_panel":
        shape = create_stiffened_panel(length, width, thickness, rib_count, rib_width, rib_height, fillet_radius)
    else:
        raise ValueError(f"Unknown part type {part_type}")

    save_step(shape, output_step)
    save_step_ok = os.path.exists(output_step)
    
    save_stl(shape, output_stl)
    save_stl_ok = os.path.exists(output_stl)
    
    return save_step_ok and save_stl_ok
