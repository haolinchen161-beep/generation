# -*- coding: utf-8 -*-
"""Readable motif-graph visualization for innovation1 v3.

This replaces the previous opaque CAD-looking screenshots with diagnostic figures
that explicitly show:
  1) motif node bounding boxes and face ids,
  2) relation lines between motif centroids,
  3) a compact 2D graph/table of node-relation correspondence.

It visualizes algorithm-extracted motif candidates, not human semantic labels.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from utils_io import ensure_workdir, read_jsonl, write_csv, write_text, timestamp

NODE_COLORS = {
    "sheet_like_group": "tab:blue",
    "face_group": "tab:gray",
    "thin_wall_pair": "tab:orange",
    "loop_or_hole": "tab:red",
    "transition_group": "tab:green",
    "repeated_feature": "tab:purple",
    "boundary_group": "tab:brown",
}

RELATION_COLORS = {
    "parallel_to": "tab:blue",
    "opposite_to": "tab:orange",
    "orthogonal_to": "tab:green",
    "coplanar_with": "tab:cyan",
    "adjacent_to": "0.55",
    "smooth_connected": "tab:olive",
    "embedded_in": "tab:red",
    "repeated_with": "tab:purple",
    "bounded_by": "tab:brown",
}

IMPORTANT_RELATIONS = [
    "thin_wall_pair",
    "opposite_to",
    "parallel_to",
    "orthogonal_to",
    "embedded_in",
    "repeated_with",
    "smooth_connected",
    "coplanar_with",
    "adjacent_to",
    "bounded_by",
]


def _as_bbox(node: Dict[str, Any]) -> np.ndarray:
    box = np.asarray(node.get("properties", {}).get("bbox", [0, 0, 0, 0, 0, 0]), dtype=float)
    if box.shape[0] != 6 or not np.all(np.isfinite(box)):
        return np.zeros(6, dtype=float)
    return box


def _centroid(node: Dict[str, Any]) -> np.ndarray:
    c = np.asarray(node.get("properties", {}).get("centroid", []), dtype=float)
    if c.shape[0] == 3 and np.all(np.isfinite(c)):
        return c
    box = _as_bbox(node)
    return 0.5 * (box[:3] + box[3:])


def _bbox_edges(box: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    mn, mx = box[:3], box[3:]
    if np.linalg.norm(mx - mn) <= 1e-12:
        return []
    x0, y0, z0 = mn.tolist()
    x1, y1, z1 = mx.tolist()
    pts = [
        np.array([x0, y0, z0]),
        np.array([x1, y0, z0]),
        np.array([x1, y1, z0]),
        np.array([x0, y1, z0]),
        np.array([x0, y0, z1]),
        np.array([x1, y0, z1]),
        np.array([x1, y1, z1]),
        np.array([x0, y1, z1]),
    ]
    idx = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    return [(pts[i], pts[j]) for i, j in idx]


def _relation_priority(rel: Dict[str, Any]) -> Tuple[int, float]:
    typ = str(rel.get("type", ""))
    try:
        priority = IMPORTANT_RELATIONS.index(typ)
    except ValueError:
        priority = len(IMPORTANT_RELATIONS)
    return priority, -float(rel.get("score", 0.0))


def _select_graphs(graphs: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    def score(graph: Dict[str, Any]) -> Tuple[int, int, int, int]:
        summary = graph.get("motif_summary", {})
        rel_types = {str(r.get("type")) for r in graph.get("motif_relations", [])}
        return (
            int(bool(summary.get("motif_rich"))),
            len(rel_types),
            len(graph.get("motif_relations", [])),
            len(graph.get("motif_nodes", [])),
        )

    ranked = sorted(graphs, key=score, reverse=True)
    return ranked[: max(int(limit), 0)]


def _set_equal_3d(ax: Any, nodes: Sequence[Dict[str, Any]]) -> None:
    boxes = np.asarray([_as_bbox(node) for node in nodes if np.linalg.norm(_as_bbox(node)[3:] - _as_bbox(node)[:3]) > 1e-12], dtype=float)
    if boxes.size == 0:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        return
    mn = np.min(boxes[:, :3], axis=0)
    mx = np.max(boxes[:, 3:], axis=0)
    center = 0.5 * (mn + mx)
    radius = max(float(np.max(mx - mn)), 1e-6) * 0.58
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _draw_3d_motif(ax: Any, graph: Dict[str, Any]) -> None:
    nodes = graph.get("motif_nodes", [])
    node_map = {str(node.get("id")): node for node in nodes}

    draw_nodes = nodes[:24]
    for node in draw_nodes:
        box = _as_bbox(node)
        typ = str(node.get("type", "face_group"))
        color = NODE_COLORS.get(typ, "0.5")
        segments = _bbox_edges(box)
        if not segments:
            continue
        collection = Line3DCollection(segments, colors=color, linewidths=1.8, alpha=0.9)
        ax.add_collection3d(collection)
        c = _centroid(node)
        label = f"{node.get('id')}\n{typ}\nf={','.join(str(x) for x in node.get('face_ids', [])[:5])}"
        if len(node.get("face_ids", [])) > 5:
            label += ",..."
        ax.text(c[0], c[1], c[2], label, fontsize=6, color=color)

    relations = sorted(graph.get("motif_relations", []), key=_relation_priority)[:18]
    for rel in relations:
        src = node_map.get(str(rel.get("source")))
        dst = node_map.get(str(rel.get("target")))
        if not src or not dst:
            continue
        p0 = _centroid(src)
        p1 = _centroid(dst)
        typ = str(rel.get("type", ""))
        color = RELATION_COLORS.get(typ, "0.25")
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=color, linewidth=1.0, alpha=0.75)
        mid = 0.5 * (p0 + p1)
        ax.text(mid[0], mid[1], mid[2], typ, fontsize=5, color=color)

    _set_equal_3d(ax, draw_nodes)
    ax.set_title("3D motif boxes + relation lines")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=24, azim=-55)


def _draw_2d_graph(ax: Any, graph: Dict[str, Any]) -> None:
    nodes = graph.get("motif_nodes", [])[:18]
    relations = sorted(graph.get("motif_relations", []), key=_relation_priority)[:24]
    node_ids = [str(n.get("id")) for n in nodes]
    node_map = {str(n.get("id")): n for n in nodes}
    n = max(len(nodes), 1)
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    pos = {node_ids[i]: np.array([math.cos(angles[i]), math.sin(angles[i])]) for i in range(len(node_ids))}

    for rel in relations:
        src = str(rel.get("source"))
        dst = str(rel.get("target"))
        if src not in pos or dst not in pos:
            continue
        p0, p1 = pos[src], pos[dst]
        typ = str(rel.get("type", ""))
        color = RELATION_COLORS.get(typ, "0.4")
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, linewidth=0.9, alpha=0.8)
        mid = 0.5 * (p0 + p1)
        ax.text(mid[0], mid[1], typ.replace("_to", ""), fontsize=6, color=color, ha="center", va="center")

    for node_id in node_ids:
        node = node_map[node_id]
        typ = str(node.get("type", "face_group"))
        color = NODE_COLORS.get(typ, "0.5")
        p = pos[node_id]
        ax.scatter([p[0]], [p[1]], s=220, color=color, edgecolor="black", zorder=3)
        face_text = ",".join(str(x) for x in node.get("face_ids", [])[:4])
        if len(node.get("face_ids", [])) > 4:
            face_text += ",..."
        ax.text(p[0], p[1], f"{node_id}\n{typ}\nf:{face_text}", fontsize=6, ha="center", va="center", color="white", zorder=4)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Motif graph: node id / type / face ids / relation labels")


def _draw_text_panel(ax: Any, graph: Dict[str, Any]) -> None:
    ax.axis("off")
    node_counter: Dict[str, int] = {}
    rel_counter: Dict[str, int] = {}
    for node in graph.get("motif_nodes", []):
        typ = str(node.get("type", "unknown"))
        node_counter[typ] = node_counter.get(typ, 0) + 1
    for rel in graph.get("motif_relations", []):
        typ = str(rel.get("type", "unknown"))
        rel_counter[typ] = rel_counter.get(typ, 0) + 1

    lines = [
        f"uid: {graph.get('uid', '')}",
        f"faces / edges / verts: {graph.get('num_faces', 0)} / {graph.get('num_edges', 0)} / {graph.get('num_vertices', 0)}",
        f"motif nodes: {len(graph.get('motif_nodes', []))}",
        f"motif relations: {len(graph.get('motif_relations', []))}",
        "",
        "Node counts:",
    ]
    for key, value in sorted(node_counter.items()):
        lines.append(f"  {key}: {value}")
    lines.extend(["", "Relation counts:"])
    for key, value in sorted(rel_counter.items()):
        lines.append(f"  {key}: {value}")
    lines.extend(["", "Top relation evidence:"])
    node_map = {str(n.get("id")): n for n in graph.get("motif_nodes", [])}
    for rel in sorted(graph.get("motif_relations", []), key=_relation_priority)[:10]:
        src = node_map.get(str(rel.get("source")), {})
        dst = node_map.get(str(rel.get("target")), {})
        lines.append(
            f"  {rel.get('type')} {rel.get('score', 0):.2f}: "
            f"{rel.get('source')}[{src.get('type','')},f={src.get('face_ids', [])[:4]}] -> "
            f"{rel.get('target')}[{dst.get('type','')},f={dst.get('face_ids', [])[:4]}]"
        )
    ax.text(0.0, 1.0, "\n".join(lines), ha="left", va="top", fontsize=8, family="monospace")
    ax.set_title("Machine-readable correspondence summary")


def _safe_filename(uid: str) -> str:
    keep = []
    for ch in uid:
        keep.append(ch if ch.isalnum() or ch in "-_" else "_")
    return "".join(keep)[:120] or "sample"


def _plot_graph(graph: Dict[str, Any], out_path: str) -> None:
    fig = plt.figure(figsize=(19, 7), dpi=150)
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    _draw_3d_motif(ax1, graph)
    _draw_2d_graph(ax2, graph)
    _draw_text_panel(ax3, graph)
    fig.suptitle("Innovation1 v3: algorithm-extracted B-Rep motif graph M", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def visualize_motif(workdir: str, num_visualizations: int = 20) -> Dict[str, Any]:
    dirs = ensure_workdir(workdir)
    motif_path = os.path.join(dirs["motif_graphs"], "motif_graphs.jsonl")
    graphs = read_jsonl(motif_path)
    selected = _select_graphs(graphs, int(num_visualizations))
    written: List[str] = []
    rows: List[Dict[str, Any]] = []

    for graph in selected:
        uid = str(graph.get("uid", "sample"))
        out_path = os.path.join(dirs["visualizations"], f"{_safe_filename(uid)}__motif_debug.png")
        try:
            _plot_graph(graph, out_path)
            written.append(out_path)
            rows.append(
                {
                    "uid": uid,
                    "path": out_path,
                    "num_nodes": len(graph.get("motif_nodes", [])),
                    "num_relations": len(graph.get("motif_relations", [])),
                    "motif_rich": int(bool(graph.get("motif_summary", {}).get("motif_rich"))),
                }
            )
        except Exception as exc:
            rows.append({"uid": uid, "path": "", "num_nodes": 0, "num_relations": 0, "motif_rich": 0, "error": str(exc)})

    write_csv(
        os.path.join(dirs["reports"], "motif_visualization_manifest.csv"),
        rows,
        ["uid", "path", "num_nodes", "num_relations", "motif_rich", "error"],
    )
    report = [
        "Innovation1 v3 Motif Visualization Report",
        "=" * 72,
        f"Time: {timestamp()}",
        f"Input motif graphs: {len(graphs)}",
        f"Requested visualizations: {num_visualizations}",
        f"Written PNG files: {len(written)}",
        "",
        "Each PNG contains:",
        "  - 3D motif bbox view with node ids, motif types and face ids.",
        "  - 2D motif graph with relation labels.",
        "  - Text evidence panel listing node/relation counts and top relation correspondences.",
    ]
    write_text(os.path.join(dirs["reports"], "motif_visualization_report.txt"), report)
    return {"selected_count": len(selected), "written": written, "rows": rows}
