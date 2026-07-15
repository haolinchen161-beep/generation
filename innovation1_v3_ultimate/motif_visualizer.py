# -*- coding: utf-8 -*-
"""弱结构基元图 M 的可视化工具。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

try:  # pragma: no cover
    from .motif_feature_extractor import extract_motif_features
    from .utils_io import ensure_workdir, read_csv, read_jsonl, read_pickle, write_csv
except ImportError:  # pragma: no cover
    from motif_feature_extractor import extract_motif_features
    from utils_io import ensure_workdir, read_csv, read_jsonl, read_pickle, write_csv


NODE_COLORS = {
    "face_group": "#DCDFE2",       # 高级太空灰
    "sheet_region": "#4A90E2", # 莫兰迪蔚蓝
    "thin_wall_pair": "#E06C75",   # 柔和珊瑚红
    "loop_or_hole": "#9B5DE5",     # 薰衣草淡紫
    "transition_group": "#F3A712", # 优雅杏黄
    "repeated_feature": "#52B788", # 莫兰迪薄荷绿
    "boundary_group": "#A88F80",   # 哑光砂褐
}

NODE_PRIORITY = [
    "thin_wall_pair",
    "repeated_feature",
    "loop_or_hole",
    "transition_group",
    "sheet_region",
    "boundary_group",
    "face_group",
]

NODE_LABEL_ABBR = {
    "face_group": "fg",
    "sheet_region": "sheet",
    "thin_wall_pair": "wall",
    "loop_or_hole": "loop",
    "transition_group": "trans",
    "repeated_feature": "repeat",
    "boundary_group": "bound",
}

RELATION_VIS_PRIORITY = {
    "thin_wall_pair": 0,
    "has_member": 1,
    "hosted_by": 2,
    "repeated_with": 3,
    "bounded_by": 4,
    "opposite_to": 5,
    "coplanar_with": 6,
    "orthogonal_to": 7,
    "parallel_to": 8,
    "smooth_connected": 9,
    "adjacent_to": 90,
    "embedded_in": 99,
}

RELATION_COLORS = {
    "adjacent_to": "#A0AAB5",      # 拓扑相邻浅灰蓝
    "parallel_to": "#4682B4",      # 钢蓝平行线
    "opposite_to": "#D16D6A",      # 珊瑚红相对线
    "orthogonal_to": "#E29578",    # 暖杏正交线
    "coplanar_with": "#5A9EAD",    # 莫兰迪蓝绿共面线
    "smooth_connected": "#8FBC8F", # 优雅灰绿平滑线
    "embedded_in": "#BAC1C8",      # 浅银灰支撑线
    "repeated_with": "#52B788",    # 莫兰迪薄荷绿重复线
    "bounded_by": "#B5838D",       # 灰玫瑰有界线
    "thin_wall_pair": "#D16D6A",   # 珊瑚红
    "has_member": "#52B788",      # 薄荷绿
    "hosted_by": "#B5838D",       # 灰玫瑰
}

RELATION_LINESTYLES = {
    "adjacent_to": ":",
    "parallel_to": "--",
    "opposite_to": "-",
    "orthogonal_to": "-.",
    "coplanar_with": "--",
    "smooth_connected": ":",
    "embedded_in": ":",
    "repeated_with": "-",
    "bounded_by": "-.",
    "thin_wall_pair": "-",
    "has_member": "-",
    "hosted_by": "-.",
}

STRUCTURAL_RELATION_TYPES = {
    "parallel_to",
    "opposite_to",
    "orthogonal_to",
    "coplanar_with",
    "smooth_connected",
    "repeated_with",
    "bounded_by",
    "thin_wall_pair",
    "has_member",
    "hosted_by",
}


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    return plt


def _short_uid(uid: Any, max_len: int = 38) -> str:
    text = str(uid)
    if len(text) <= max_len:
        return text
    return "..." + text[-max_len:]


def _bbox_edges(bbox: Sequence[float]) -> List[tuple[np.ndarray, np.ndarray]]:
    b = np.asarray(bbox, dtype=np.float32)
    mn = b[:3]
    mx = b[3:]
    pts = np.asarray(
        [
            [mn[0], mn[1], mn[2]],
            [mx[0], mn[1], mn[2]],
            [mx[0], mx[1], mn[2]],
            [mn[0], mx[1], mn[2]],
            [mn[0], mn[1], mx[2]],
            [mx[0], mn[1], mx[2]],
            [mx[0], mx[1], mx[2]],
            [mn[0], mx[1], mx[2]],
        ],
        dtype=np.float32,
    )
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    return [(pts[i], pts[j]) for i, j in pairs]


def _set_equal_axes(ax: Any, bboxes: Sequence[Sequence[float]]) -> None:
    if not bboxes:
        return
    arr = np.asarray(bboxes, dtype=np.float32)
    mn = np.min(arr[:, :3], axis=0)
    mx = np.max(arr[:, 3:], axis=0)
    center = 0.5 * (mn + mx)
    radius = float(np.max(mx - mn)) * 0.55 + 1e-6
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def _draw_bbox(ax: Any, bbox: Sequence[float], color: str, alpha: float = 0.75, linewidth: float = 0.8) -> None:
    for p0, p1 in _bbox_edges(bbox):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=color, alpha=alpha, linewidth=linewidth)


def _face_samples(parsed_data: Dict[str, Any]) -> np.ndarray:
    arr = np.asarray(parsed_data.get("face_wcs", []), dtype=np.float32)
    if arr.ndim == 4 and arr.shape[-1] == 3:
        return arr
    return np.zeros((0, 0, 0, 3), dtype=np.float32)


def _draw_face_surface(
    ax: Any,
    samples: np.ndarray,
    color: str,
    alpha: float = 0.18,
    max_grid: int = 18,
) -> bool:
    pts = np.asarray(samples, dtype=np.float32)
    if pts.ndim != 3 or pts.shape[-1] != 3 or pts.size == 0 or not np.all(np.isfinite(pts)):
        return False
    flat = pts.reshape(-1, 3)
    if float(np.max(np.ptp(flat, axis=0))) <= 1e-8:
        return False
    step_u = max(1, int(np.ceil(pts.shape[0] / max_grid)))
    step_v = max(1, int(np.ceil(pts.shape[1] / max_grid)))
    grid = pts[::step_u, ::step_v, :]
    ax.plot_surface(
        grid[:, :, 0],
        grid[:, :, 1],
        grid[:, :, 2],
        color=color,
        alpha=alpha,
        linewidth=0,
        antialiased=True,
        shade=True, # 开启三维光影遮罩，让曲面极富立体感和质感
    )
    return True


def _face_color_map(graph: Dict[str, Any]) -> Dict[int, str]:
    result: Dict[int, str] = {}
    nodes = graph.get("motif_nodes", [])
    for node_type in NODE_PRIORITY:
        for node in nodes:
            if node.get("type") != node_type:
                continue
            color = NODE_COLORS.get(node_type, "#7f8c8d")
            for fid in node.get("face_ids", []):
                result.setdefault(int(fid), color)
    return result


def _compact_face_ids(face_ids: Sequence[Any], max_items: int = 5) -> str:
    vals = [str(int(fid)) for fid in face_ids]
    if len(vals) <= max_items:
        return ",".join(vals)
    return ",".join(vals[:max_items]) + f"+{len(vals) - max_items}"


def _node_visual_label(node: Dict[str, Any], expanded: bool) -> str:
    node_id = str(node.get("id", ""))
    if not expanded:
        return node_id
    node_type = str(node.get("type", ""))
    abbr = NODE_LABEL_ABBR.get(node_type, node_type[:6])
    faces = _compact_face_ids(node.get("face_ids", []), max_items=4)
    return f"{node_id}\n{abbr}\nf{faces}"


def _non_base_nodes(graph: Dict[str, Any], max_nodes: int = 16) -> List[Dict[str, Any]]:
    nodes = [node for node in graph.get("motif_nodes", []) if node.get("type") != "face_group"]
    return sorted(
        nodes,
        key=lambda node: (
            NODE_PRIORITY.index(str(node.get("type"))) if str(node.get("type")) in NODE_PRIORITY else 99,
            -float(node.get("confidence", 0.0)),
            str(node.get("id")),
        ),
    )[:max_nodes]


def _node_centroid_from_faces(node: Dict[str, Any], face_features: Sequence[Dict[str, Any]]) -> np.ndarray:
    face_by_id = {int(face["face_id"]): face for face in face_features}
    centroids = []
    for fid in node.get("face_ids", []):
        face = face_by_id.get(int(fid))
        if face is not None:
            centroids.append(np.asarray(face["centroid"], dtype=np.float32))
    if centroids:
        return np.mean(np.stack(centroids, axis=0), axis=0)
    return _node_centroid(node)


def _relation_face_text(rel: Dict[str, Any], node_by_id: Dict[str, Dict[str, Any]]) -> str:
    src = node_by_id.get(str(rel.get("source")), {})
    dst = node_by_id.get(str(rel.get("target")), {})
    return f"f{_compact_face_ids(src.get('face_ids', []))}->f{_compact_face_ids(dst.get('face_ids', []))}"


def _simplify_graph_for_visualization(
    graph: Dict[str, Any],
    max_nodes: int = 48,
    max_edges: int = 54,
    paper_vis: bool = True,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """在保留结构主线的同时压缩可视化图，避免边过密。

    JSON 图保留训练和审计所需证据；PNG 默认展示高层 motif prior，
    不展示所有两两 face evidence。
    """
    all_nodes = list(graph.get("motif_nodes", []))
    all_relations = list(graph.get("motif_relations", []))
    if paper_vis:
        all_relations = [rel for rel in all_relations if str(rel.get("type")) in STRUCTURAL_RELATION_TYPES]
    node_by_id = {str(node.get("id")): node for node in all_nodes}

    non_base = [node for node in all_nodes if node.get("type") != "face_group"]
    non_base = sorted(
        non_base,
        key=lambda node: (
            NODE_PRIORITY.index(str(node.get("type"))) if str(node.get("type")) in NODE_PRIORITY else 99,
            -float(node.get("confidence", 0.0)),
            str(node.get("id")),
        ),
    )

    visible_ids = {str(node.get("id")) for node in non_base[:max_nodes]}
    support_ids: List[str] = []
    for node in non_base:
        features = node.get("features", {})
        for key in ["base_face_group_id"]:
            value = features.get(key)
            if value and str(value) in node_by_id:
                support_ids.append(str(value))
        for key in ["base_face_group_ids", "member_face_group_ids"]:
            for value in features.get(key, []) or []:
                if str(value) in node_by_id:
                    support_ids.append(str(value))

    for node_id in support_ids:
        if len(visible_ids) >= max_nodes:
            break
        visible_ids.add(node_id)

    if not visible_ids:
        visible_ids = {str(node.get("id")) for node in all_nodes[:max_nodes]}

    motif_edges: List[Dict[str, Any]] = []
    support_edges: List[Dict[str, Any]] = []
    structural_edges: List[Dict[str, Any]] = []
    adjacency_edges: List[Dict[str, Any]] = []
    for rel in all_relations:
        src = str(rel.get("source"))
        dst = str(rel.get("target"))
        typ = str(rel.get("type"))
        if src not in visible_ids or dst not in visible_ids:
            continue
        if typ in {"repeated_with", "bounded_by"}:
            motif_edges.append(rel)
        elif typ == "embedded_in":
            support_edges.append(rel)
        elif typ == "adjacent_to":
            adjacency_edges.append(rel)
        else:
            structural_edges.append(rel)

    structural_edges = sorted(
        structural_edges,
        key=lambda rel: (
            RELATION_VIS_PRIORITY.get(str(rel.get("type")), 99),
            -float(rel.get("confidence", 0.0)),
        ),
    )
    adjacency_edges = sorted(adjacency_edges, key=lambda rel: -float(rel.get("confidence", 0.0)))

    support_edges = sorted(support_edges, key=lambda rel: -float(rel.get("confidence", 0.0)))
    compact_support_edges: List[Dict[str, Any]] = []
    support_targets: set[str] = set()
    support_budget = max(4, max_edges // 4)
    for rel in support_edges:
        target = str(rel.get("target"))
        if target in support_targets:
            continue
        compact_support_edges.append(rel)
        support_targets.add(target)
        if len(compact_support_edges) >= support_budget:
            break

    selected_edges = motif_edges + structural_edges[: max(0, max_edges - len(motif_edges))]
    if len(selected_edges) < max_edges:
        selected_edges += compact_support_edges[: max_edges - len(selected_edges)]
    if len(selected_edges) < max_edges:
        selected_edges += adjacency_edges[: max_edges - len(selected_edges)]

    edge_nodes = {str(rel.get("source")) for rel in selected_edges} | {str(rel.get("target")) for rel in selected_edges}
    visible_ids |= edge_nodes
    visible_nodes = [node_by_id[node_id] for node_id in node_by_id if node_id in visible_ids]
    return visible_nodes, selected_edges[:max_edges]


def _select_overlay_relations(graph: Dict[str, Any], max_edges: int = 36, paper_vis: bool = True) -> List[Dict[str, Any]]:
    if paper_vis:
        relations = [rel for rel in graph.get("motif_relations", []) if str(rel.get("type")) in STRUCTURAL_RELATION_TYPES]
    else:
        relations = list(graph.get("motif_relations", []))
    important = [rel for rel in relations if str(rel.get("type")) in STRUCTURAL_RELATION_TYPES]
    support = [] if paper_vis else [rel for rel in relations if str(rel.get("type")) in {"embedded_in", "adjacent_to"}]
    important = sorted(
        important,
        key=lambda rel: (
            RELATION_VIS_PRIORITY.get(str(rel.get("type")), 99),
            -float(rel.get("confidence", 0.0)),
        ),
    )
    support = sorted(support, key=lambda rel: -float(rel.get("confidence", 0.0)))
    selected = important[:max_edges]
    if len(selected) < max_edges:
        selected += support[: max_edges - len(selected)]
    return selected[:max_edges]


def _node_centroid(node: Dict[str, Any]) -> np.ndarray:
    features = node.get("features", {})
    centroid = features.get("centroid")
    if centroid is not None:
        return np.asarray(centroid, dtype=np.float32)
    bbox = np.asarray(features.get("bbox", [0, 0, 0, 0, 0, 0]), dtype=np.float32)
    if bbox.size == 6:
        return 0.5 * (bbox[:3] + bbox[3:])
    return np.zeros(3, dtype=np.float32)


def plot_faces(parsed_data: Dict[str, Any], output_path: str) -> None:
    plt = _mpl()
    features = extract_motif_features(parsed_data)
    face_features = features.get("face_features", [])
    samples = _face_samples(parsed_data)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    bboxes = []
    for face in face_features:
        fid = int(face["face_id"])
        bbox = face["bbox"]
        bboxes.append(bbox)
        if fid < samples.shape[0]:
            _draw_face_surface(ax, samples[fid], "#aeb8c2", alpha=0.20)
        _draw_bbox(ax, bbox, "#4f5b66", alpha=0.45, linewidth=0.7)
        ctr = np.asarray(face["centroid"], dtype=np.float32)
        ax.scatter([ctr[0]], [ctr[1]], [ctr[2]], color="#111111", s=10)
        if len(face_features) <= 70:
            ax.text(ctr[0], ctr[1], ctr[2], f"f{fid}", fontsize=6, color="#111111")
    ax.set_title(f"{_short_uid(parsed_data.get('uid', ''))}：面采样 / bbox / face id", fontsize=10)
    _set_equal_axes(ax, bboxes)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_motif_overlay(parsed_data: Dict[str, Any], graph: Dict[str, Any], output_path: str, paper_vis: bool = True) -> None:
    plt = _mpl()
    features = extract_motif_features(parsed_data)
    face_features = features.get("face_features", [])
    samples = _face_samples(parsed_data)
    face_colors = _face_color_map(graph)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    bboxes = []
    node_by_id = {str(node.get("id")): node for node in graph.get("motif_nodes", [])}
    overlay_relations = _select_overlay_relations(graph, paper_vis=paper_vis)
    highlighted_face_ids = set()
    for rel in overlay_relations:
        for end_key in ["source", "target"]:
            node = node_by_id.get(str(rel.get(end_key)))
            if node:
                highlighted_face_ids.update(int(fid) for fid in node.get("face_ids", []))
    for node in graph.get("motif_nodes", []):
        if node.get("type") != "face_group":
            highlighted_face_ids.update(int(fid) for fid in node.get("face_ids", []))

    for face in face_features:
        fid = int(face["face_id"])
        bbox = face["bbox"]
        bboxes.append(bbox)
        color = face_colors.get(fid, "#b7bec7")
        surface_alpha = 0.32 if fid in highlighted_face_ids else 0.13
        line_alpha = 0.88 if fid in highlighted_face_ids else 0.30
        linewidth = 1.2 if fid in highlighted_face_ids else 0.45
        if fid < samples.shape[0]:
            _draw_face_surface(ax, samples[fid], color, alpha=surface_alpha)
        _draw_bbox(ax, bbox, color, alpha=line_alpha, linewidth=linewidth)
        ctr = np.asarray(face["centroid"], dtype=np.float32)
        ax.scatter([ctr[0]], [ctr[1]], [ctr[2]], color=color, s=18 if fid in highlighted_face_ids else 8, alpha=0.95)
        if len(face_features) <= 30 or fid in highlighted_face_ids:
            ax.text(ctr[0], ctr[1], ctr[2], f"f{fid}", fontsize=6, color="#111111")
    for rel in overlay_relations:
        src = node_by_id.get(str(rel.get("source")))
        dst = node_by_id.get(str(rel.get("target")))
        if not src or not dst:
            continue
        p0 = _node_centroid(src)
        p1 = _node_centroid(dst)
        if float(np.linalg.norm(p1 - p0)) <= 1e-8:
            continue
        rel_type = str(rel.get("type"))
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            [p0[2], p1[2]],
            color=RELATION_COLORS.get(rel_type, "#333333"),
            linestyle=RELATION_LINESTYLES.get(rel_type, "-"),
            linewidth=1.7,
            alpha=0.9,
        )
    graph_symbol = "S" if graph.get("graph_view") == "distilled_motif_prior" else "M"
    ax.set_title(
        f"{_short_uid(graph.get('uid', ''))} {'论文图' if paper_vis else '调试图'}：{graph_symbol} 在实体上 + {len(overlay_relations)} 条结构关系",
        fontsize=9,
    )
    _set_equal_axes(ax, bboxes)
    legend_handles = []
    from matplotlib.lines import Line2D

    for typ, color in NODE_COLORS.items():
        if any(node.get("type") == typ for node in graph.get("motif_nodes", [])):
            legend_handles.append(Line2D([0], [0], color=color, lw=2, label=typ))
    for typ, color in RELATION_COLORS.items():
        if any(str(rel.get("type")) == typ for rel in overlay_relations):
            legend_handles.append(
                Line2D([0], [0], color=color, lw=1.8, linestyle=RELATION_LINESTYLES.get(typ, "-"), label=f"rel:{typ}")
            )
    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_motif_on_solid(parsed_data: Dict[str, Any], graph: Dict[str, Any], output_path: str, paper_vis: bool = True) -> None:
    """把 M 直接绘制到规范化 B-Rep face 采样上。"""
    plt = _mpl()
    from matplotlib.lines import Line2D

    features = extract_motif_features(parsed_data)
    face_features = features.get("face_features", [])
    samples = _face_samples(parsed_data)
    face_colors = _face_color_map(graph)
    node_by_id = {str(node.get("id")): node for node in graph.get("motif_nodes", [])}
    motif_nodes = _non_base_nodes(graph)
    key_relations = _select_overlay_relations(graph, max_edges=18, paper_vis=paper_vis)
    graph_symbol = "S" if graph.get("graph_view") == "distilled_motif_prior" else "M"

    fig = plt.figure(figsize=(12.5, 7.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.15], wspace=0.02)
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    ax.set_axis_off() # 隐藏四周刻度尺与背景灰面板，营造纯净的悬空模型视口
    ax_text = fig.add_subplot(gs[0, 1])
    ax_text.axis("off")

    bboxes = []
    motif_face_ids = {int(fid) for node in motif_nodes for fid in node.get("face_ids", [])}
    relation_face_ids = set()
    for rel in key_relations:
        for key in ["source", "target"]:
            node = node_by_id.get(str(rel.get(key)))
            if node:
                relation_face_ids.update(int(fid) for fid in node.get("face_ids", []))
    highlighted_face_ids = motif_face_ids | relation_face_ids

    # 第一层：绘制半透明实体虚影，帮助判断 motif 在零件上的位置。
    for face in face_features:
        fid = int(face["face_id"])
        bbox = face["bbox"]
        bboxes.append(bbox)
        if fid < samples.shape[0]:
            _draw_face_surface(ax, samples[fid], "#d8dee5", alpha=0.10, max_grid=14)
        _draw_bbox(ax, bbox, "#9aa3ad", alpha=0.20, linewidth=0.45)

    # 第二层：突出承载 motif 的 face，并叠加节点与关系。
    for face in face_features:
        fid = int(face["face_id"])
        if fid not in highlighted_face_ids:
            continue
        bbox = face["bbox"]
        color = face_colors.get(fid, "#7f8c8d")
        if fid < samples.shape[0]:
            _draw_face_surface(ax, samples[fid], color, alpha=0.42, max_grid=18)
        _draw_bbox(ax, bbox, color, alpha=0.95, linewidth=1.45)
        ctr = np.asarray(face["centroid"], dtype=np.float32)
        ax.scatter([ctr[0]], [ctr[1]], [ctr[2]], color="#1f2328", s=18, alpha=0.95)
        ax.text(ctr[0], ctr[1], ctr[2], f"f{fid}", fontsize=7, color="#111111")

    for node in motif_nodes:
        center = _node_centroid_from_faces(node, face_features)
        node_type = str(node.get("type", ""))
        color = NODE_COLORS.get(node_type, "#7f8c8d")
        label = _node_visual_label(node, expanded=True).replace("\n", " ")
        ax.scatter([center[0]], [center[1]], [center[2]], color=color, edgecolor="#111111", s=42, alpha=0.98)
        ax.text(center[0], center[1], center[2], label, fontsize=7, color="#111111")

    for rel in key_relations:
        src = node_by_id.get(str(rel.get("source")))
        dst = node_by_id.get(str(rel.get("target")))
        if not src or not dst:
            continue
        p0 = _node_centroid_from_faces(src, face_features)
        p1 = _node_centroid_from_faces(dst, face_features)
        if float(np.linalg.norm(p1 - p0)) <= 1e-8:
            continue
        rel_type = str(rel.get("type"))
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            [p0[2], p1[2]],
            color=RELATION_COLORS.get(rel_type, "#333333"),
            linestyle=RELATION_LINESTYLES.get(rel_type, "-"),
            linewidth=2.2,
            alpha=0.95,
        )

    _set_equal_axes(ax, bboxes)
    ax.set_title(f"{_short_uid(graph.get('uid', ''))}：{graph_symbol} 在实体上的表现", fontsize=10)

    handles = []
    present_types = {str(node.get("type")) for node in motif_nodes}
    for typ in NODE_PRIORITY:
        if typ in present_types:
            handles.append(Line2D([0], [0], color=NODE_COLORS.get(typ, "#7f8c8d"), lw=4, label=typ))
    present_rel_types = {str(rel.get("type")) for rel in key_relations}
    for typ in RELATION_COLORS:
        if typ in present_rel_types:
            handles.append(
                Line2D([0], [0], color=RELATION_COLORS[typ], lw=2, linestyle=RELATION_LINESTYLES.get(typ, "-"), label=f"rel:{typ}")
            )
    if handles:
        ax.legend(handles=handles, fontsize=7, loc="upper left")

    lines = [
        f"{'论文' if paper_vis else '调试'}：{graph_symbol} 在实体上",
        f"face 数：{graph.get('num_faces', 0)}",
        f"motif node 数：{len(graph.get('motif_nodes', []))}",
        f"relation 数：{len(graph.get('motif_relations', []))}",
        "",
        "Vm 对应 face：",
    ]
    for node in motif_nodes[:12]:
        abbr = NODE_LABEL_ABBR.get(str(node.get("type")), str(node.get("type"))[:6])
        lines.append(
            f"{node.get('id')} {abbr:<6} f{_compact_face_ids(node.get('face_ids', []))}  c={float(node.get('confidence', 0.0)):.2f}"
        )
    if len(motif_nodes) > 12:
        lines.append(f"... 另有 {len(motif_nodes) - 12} 个 nodes")
    lines.extend(["", "Em 对应 face："])
    for rel in key_relations[:14]:
        lines.append(
            f"{str(rel.get('type')):<13} {_relation_face_text(rel, node_by_id)}  c={float(rel.get('confidence', 0.0)):.2f}"
        )
    if len(key_relations) > 14:
        lines.append(f"... 另有 {len(key_relations) - 14} 条 rels")

    ax_text.text(
        0.02,
        0.98,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=8,
        family="monospace",
        color="#1f2328",
    )

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_motif_graph(graph: Dict[str, Any], output_path: str, paper_vis: bool = True) -> None:
    plt = _mpl()
    try:
        import networkx as nx
    except Exception:
        nx = None

    nodes, relations = _simplify_graph_for_visualization(graph, paper_vis=paper_vis)
    fig, ax = plt.subplots(figsize=(9, 7))
    if not nodes:
        ax.text(0.5, 0.5, "空 motif graph", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    if nx is not None:
        g = nx.Graph()
        for node in nodes:
            g.add_node(node["id"], node_type=node.get("type", "face_group"))
        for rel in relations:
            g.add_edge(rel["source"], rel["target"], relation_type=rel.get("type", ""), confidence=rel.get("confidence", 0.0))
        pos = nx.spring_layout(g, seed=7, k=0.65)
        for typ, color in NODE_COLORS.items():
            nodelist = [n for n, attrs in g.nodes(data=True) if attrs.get("node_type") == typ]
            if nodelist:
                nx.draw_networkx_nodes(
                    g, pos, nodelist=nodelist, node_color=color,
                    node_size=380, alpha=0.92, edgecolors="#2E3440", linewidths=1.0, ax=ax, label=typ
                )
        for rel_type, color in RELATION_COLORS.items():
            edgelist = [(u, v) for u, v, attrs in g.edges(data=True) if attrs.get("relation_type") == rel_type]
            if edgelist:
                nx.draw_networkx_edges(
                    g,
                    pos,
                    edgelist=edgelist,
                    edge_color=color,
                    style=RELATION_LINESTYLES.get(rel_type, "-"),
                    alpha=0.85,
                    width=1.2,
                    ax=ax,
                    label=f"rel:{rel_type}",
                    connectionstyle="arc3,rad=0.08", # 使用优雅平滑的贝塞尔微弧线替换生硬直折线
                )
        expanded_labels = len(nodes) <= 18
        labels = {node["id"]: _node_visual_label(node, expanded=expanded_labels) for node in nodes}
        nx.draw_networkx_labels(g, pos, labels=labels, font_size=6 if expanded_labels else 7, ax=ax)
        
        # 隐藏 2D 拓扑图四周的默认边框线
        for spine in ["top", "right", "bottom", "left"]:
            ax.spines[spine].set_visible(False)
    else:
        angles = np.linspace(0.0, 2.0 * np.pi, len(nodes), endpoint=False)
        pos = {node["id"]: np.asarray([np.cos(a), np.sin(a)]) for node, a in zip(nodes, angles)}
        for rel in relations:
            a = pos.get(rel["source"])
            b = pos.get(rel["target"])
            if a is not None and b is not None:
                rel_type = str(rel.get("type"))
                ax.plot(
                    [a[0], b[0]],
                    [a[1], b[1]],
                    color=RELATION_COLORS.get(rel_type, "#78838f"),
                    linestyle=RELATION_LINESTYLES.get(rel_type, "-"),
                    alpha=0.72,
                    linewidth=1.0,
                )
        for node in nodes:
            p = pos[node["id"]]
            color = NODE_COLORS.get(node.get("type"), "#7f8c8d")
            ax.scatter([p[0]], [p[1]], color=color, s=180)
            ax.text(
                p[0],
                p[1],
                _node_visual_label(node, expanded=len(nodes) <= 18),
                fontsize=6 if len(nodes) <= 18 else 7,
                ha="center",
                va="center",
            )
    total_nodes = len(graph.get("motif_nodes", []))
    total_edges = len(graph.get("motif_relations", []))
    ax.set_title(
        f"{_short_uid(graph.get('uid', ''))} motif graph 视图：{len(nodes)}/{total_nodes} nodes，{len(relations)}/{total_edges} edges",
        fontsize=9,
    )
    ax.axis("off")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_motif_legend(graph: Dict[str, Any], output_path: str) -> None:
    rows: List[Dict[str, Any]] = []
    node_by_id = {str(node.get("id")): node for node in graph.get("motif_nodes", [])}
    for node in graph.get("motif_nodes", []):
        features = node.get("features", {}) or {}
        rows.append(
            {
                "record": "node",
                "id": node.get("id", ""),
                "type": node.get("type", ""),
                "face_ids": " ".join(str(int(fid)) for fid in node.get("face_ids", [])),
                "confidence": round(float(node.get("confidence", 0.0)), 6),
                "source": "",
                "target": "",
                "relation_type": "",
                "source_face_ids": "",
                "target_face_ids": "",
                "face_relation": "",
                "summary": f"centroid={features.get('centroid', '')}; area={features.get('relative_area_sum', '')}",
            }
        )
    for rel in graph.get("motif_relations", []):
        src_node = node_by_id.get(str(rel.get("source")), {})
        dst_node = node_by_id.get(str(rel.get("target")), {})
        src_faces = src_node.get("face_ids", [])
        dst_faces = dst_node.get("face_ids", [])
        src_face_text = _compact_face_ids(src_faces)
        dst_face_text = _compact_face_ids(dst_faces)
        rows.append(
            {
                "record": "relation",
                "id": "",
                "type": "",
                "face_ids": "",
                "confidence": round(float(rel.get("confidence", 0.0)), 6),
                "source": rel.get("source", ""),
                "target": rel.get("target", ""),
                "relation_type": rel.get("type", ""),
                "source_face_ids": src_face_text,
                "target_face_ids": dst_face_text,
                "face_relation": f"f{src_face_text} -> f{dst_face_text}",
                "summary": str(rel.get("evidence", ""))[:240],
            }
        )
    write_csv(
        output_path,
        rows,
        [
            "record",
            "id",
            "type",
            "face_ids",
            "confidence",
            "source",
            "target",
            "relation_type",
            "source_face_ids",
            "target_face_ids",
            "face_relation",
            "summary",
        ],
    )


def _cleanup_generated_visualizations(visualization_dir: str) -> Dict[str, int]:
    """渲染新子集前清理旧的可视化 PNG 和 legend。"""
    root = Path(visualization_dir)
    removed = 0
    for pattern in ["*.png", "*_motif_legend.csv"]:
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            path.unlink()
            removed += 1
    return {"removed_stale_visualizations": removed}


def visualize_motifs(
    workdir: str,
    count: int = 10,
    uids: Sequence[str] | None = None,
    ready_only: bool = True,
    paper_vis: bool = True,
) -> Dict[str, Any]:
    dirs = ensure_workdir(workdir)
    cleanup_info = _cleanup_generated_visualizations(dirs["visualizations"])
    prior_ready_index = os.path.join(dirs["motif_graphs"], "motif_prior_index_ready.jsonl")
    prior_index = os.path.join(dirs["motif_graphs"], "motif_prior_index.jsonl")
    ready_index = os.path.join(dirs["motif_graphs"], "motif_graph_index_ready.jsonl")
    full_index = os.path.join(dirs["motif_graphs"], "motif_graph_index.jsonl")
    if paper_vis and ready_only and Path(prior_ready_index).exists():
        index_path = prior_ready_index
    elif paper_vis and Path(prior_index).exists():
        index_path = prior_index
    elif ready_only and Path(ready_index).exists():
        index_path = ready_index
    else:
        index_path = full_index
    is_prior_view = "motif_prior" in Path(index_path).name
    graphs = read_jsonl(index_path)
    if uids:
        uid_set = {str(uid) for uid in uids}
        graphs = [graph for graph in graphs if str(graph.get("uid")) in uid_set]
    else:
        graphs = graphs[: max(int(count), 0)]

    manifest_rows = read_csv(os.path.join(dirs["parsed"], "clean_manifest.csv"))
    pkl_by_uid = {str(row.get("uid")): str(row.get("pkl_path")) for row in manifest_rows}
    outputs: List[Dict[str, str]] = []
    for graph in graphs:
        uid = str(graph.get("uid", ""))
        pkl_path = pkl_by_uid.get(uid, os.path.join(dirs["parsed"], f"{uid}.pkl"))
        if not Path(pkl_path).exists():
            continue
        parsed_data = read_pickle(pkl_path)
        faces_path = os.path.join(dirs["visualizations"], f"{uid}_faces.png")
        if is_prior_view:
            overlay_path = os.path.join(dirs["visualizations"], f"{uid}_motif_prior_overlay.png")
            solid_path = os.path.join(dirs["visualizations"], f"{uid}_motif_prior_on_solid.png")
            graph_path = os.path.join(dirs["visualizations"], f"{uid}_motif_prior_graph.png")
            legend_path = os.path.join(dirs["visualizations"], f"{uid}_motif_prior_legend.csv")
        else:
            overlay_path = os.path.join(dirs["visualizations"], f"{uid}_motif_overlay.png")
            solid_path = os.path.join(dirs["visualizations"], f"{uid}_motif_on_solid.png")
            graph_path = os.path.join(dirs["visualizations"], f"{uid}_motif_graph.png")
            legend_path = os.path.join(dirs["visualizations"], f"{uid}_motif_legend.csv")
        plot_faces(parsed_data, faces_path)
        plot_motif_overlay(parsed_data, graph, overlay_path, paper_vis=paper_vis)
        plot_motif_on_solid(parsed_data, graph, solid_path, paper_vis=paper_vis)
        plot_motif_graph(graph, graph_path, paper_vis=paper_vis)
        write_motif_legend(graph, legend_path)
        outputs.append(
            {
                "uid": uid,
                "faces": faces_path,
                "motif_overlay": overlay_path,
                "motif_on_solid": solid_path,
                "motif_graph": graph_path,
                "motif_legend": legend_path,
            }
        )
    return {
        "visualized": len(outputs),
        "outputs": outputs,
        "index_path": index_path,
        "ready_only": ready_only,
        "paper_vis": paper_vis,
        "prior_view": is_prior_view,
        **cleanup_info,
    }
