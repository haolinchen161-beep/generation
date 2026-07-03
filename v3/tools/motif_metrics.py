# -*- coding: utf-8 -*-
"""Metrics and audit reports for innovation1 v3 motif graphs.

The metrics here deliberately evaluate algorithm-extracted weak structural motifs,
not engineering truth labels.  They are used to check whether the v3 pipeline
actually creates an inspectable structure prior layer M=(Vm, Em, Pm).
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List

import numpy as np

from utils_io import ensure_workdir, read_jsonl, timestamp, write_csv, write_text

NODE_TYPES = [
    "face_group",
    "sheet_like_group",
    "thin_wall_pair",
    "loop_or_hole",
    "transition_group",
    "repeated_feature",
    "boundary_group",
]

RELATION_TYPES = [
    "adjacent_to",
    "parallel_to",
    "opposite_to",
    "orthogonal_to",
    "coplanar_with",
    "smooth_connected",
    "embedded_in",
    "repeated_with",
    "bounded_by",
]


def _counter_values(counter: Counter, keys: Iterable[str], prefix: str) -> Dict[str, int]:
    return {f"{prefix}_{key}": int(counter.get(key, 0)) for key in keys}


def _node_by_id(graph: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(node.get("id")): node for node in graph.get("motif_nodes", [])}


def _relation_evidence_rows(graphs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for graph in graphs:
        uid = str(graph.get("uid", ""))
        nodes = _node_by_id(graph)
        for rel in graph.get("motif_relations", []):
            src = nodes.get(str(rel.get("source")), {})
            dst = nodes.get(str(rel.get("target")), {})
            rows.append(
                {
                    "uid": uid,
                    "relation_type": str(rel.get("type", "")),
                    "score": float(rel.get("score", 0.0)),
                    "source": str(rel.get("source", "")),
                    "source_type": str(src.get("type", "")),
                    "source_faces": ";".join(str(x) for x in src.get("face_ids", [])),
                    "target": str(rel.get("target", "")),
                    "target_type": str(dst.get("type", "")),
                    "target_faces": ";".join(str(x) for x in dst.get("face_ids", [])),
                }
            )
    return rows


def evaluate_motif(workdir: str) -> Dict[str, Any]:
    dirs = ensure_workdir(workdir)
    motif_path = os.path.join(dirs["motif_graphs"], "motif_graphs.jsonl")
    graphs = read_jsonl(motif_path)

    per_sample_rows: List[Dict[str, Any]] = []
    total_node_counter: Counter = Counter()
    total_rel_counter: Counter = Counter()
    rich_count = 0
    has_any_relation = 0
    motif_node_counts: List[int] = []
    motif_rel_counts: List[int] = []

    for graph in graphs:
        node_counter = Counter(str(node.get("type", "unknown")) for node in graph.get("motif_nodes", []))
        rel_counter = Counter(str(rel.get("type", "unknown")) for rel in graph.get("motif_relations", []))
        total_node_counter.update(node_counter)
        total_rel_counter.update(rel_counter)
        n_nodes = len(graph.get("motif_nodes", []))
        n_rel = len(graph.get("motif_relations", []))
        motif_node_counts.append(n_nodes)
        motif_rel_counts.append(n_rel)
        if graph.get("motif_summary", {}).get("motif_rich"):
            rich_count += 1
        if n_rel > 0:
            has_any_relation += 1
        row: Dict[str, Any] = {
            "uid": graph.get("uid", ""),
            "source": graph.get("source", ""),
            "num_faces": int(graph.get("num_faces", 0)),
            "num_edges": int(graph.get("num_edges", 0)),
            "num_vertices": int(graph.get("num_vertices", 0)),
            "num_motif_nodes": n_nodes,
            "num_motif_relations": n_rel,
            "motif_rich": int(bool(graph.get("motif_summary", {}).get("motif_rich"))),
            "has_parallel": int(bool(graph.get("motif_summary", {}).get("has_parallel"))),
            "has_thin_wall": int(bool(graph.get("motif_summary", {}).get("has_thin_wall"))),
            "has_loop_or_hole": int(bool(graph.get("motif_summary", {}).get("has_loop_or_hole"))),
            "has_transition": int(bool(graph.get("motif_summary", {}).get("has_transition"))),
            "has_repeated_feature": int(bool(graph.get("motif_summary", {}).get("has_repeated_feature"))),
        }
        row.update(_counter_values(node_counter, NODE_TYPES, "node"))
        row.update(_counter_values(rel_counter, RELATION_TYPES, "rel"))
        per_sample_rows.append(row)

    fieldnames = [
        "uid",
        "source",
        "num_faces",
        "num_edges",
        "num_vertices",
        "num_motif_nodes",
        "num_motif_relations",
        "motif_rich",
        "has_parallel",
        "has_thin_wall",
        "has_loop_or_hole",
        "has_transition",
        "has_repeated_feature",
    ] + [f"node_{x}" for x in NODE_TYPES] + [f"rel_{x}" for x in RELATION_TYPES]
    write_csv(os.path.join(dirs["reports"], "motif_stats.csv"), per_sample_rows, fieldnames)

    evidence_rows = _relation_evidence_rows(graphs)
    write_csv(
        os.path.join(dirs["reports"], "motif_relation_evidence.csv"),
        evidence_rows,
        [
            "uid",
            "relation_type",
            "score",
            "source",
            "source_type",
            "source_faces",
            "target",
            "target_type",
            "target_faces",
        ],
    )

    graph_count = len(graphs)
    stats: Dict[str, Any] = {
        "graph_count": graph_count,
        "motif_rich_count": rich_count,
        "has_any_relation_count": has_any_relation,
        "avg_motif_nodes": float(np.mean(motif_node_counts)) if motif_node_counts else 0.0,
        "avg_motif_relations": float(np.mean(motif_rel_counts)) if motif_rel_counts else 0.0,
        "relation_evidence_rows": len(evidence_rows),
    }
    for node_type in NODE_TYPES:
        stats[f"node_{node_type}_count"] = int(total_node_counter.get(node_type, 0))
    for rel_type in RELATION_TYPES:
        stats[f"rel_{rel_type}_count"] = int(total_rel_counter.get(rel_type, 0))

    report = [
        "Innovation1 v3 Motif Metrics Report",
        "=" * 72,
        f"Time: {timestamp()}",
        f"Motif graphs: {graph_count}",
        f"Motif-rich graphs: {rich_count}",
        f"Graphs with at least one relation: {has_any_relation}",
        f"Average motif nodes: {stats['avg_motif_nodes']:.3f}",
        f"Average motif relations: {stats['avg_motif_relations']:.3f}",
        f"Relation evidence rows: {len(evidence_rows)}",
        "",
        "Node totals:",
    ]
    for node_type in NODE_TYPES:
        report.append(f"  - {node_type}: {total_node_counter.get(node_type, 0)}")
    report.extend(["", "Relation totals:"])
    for rel_type in RELATION_TYPES:
        report.append(f"  - {rel_type}: {total_rel_counter.get(rel_type, 0)}")
    report.extend(
        [
            "",
            "Boundary statement:",
            "  These metrics check algorithm-extracted weak structural motifs.",
            "  They are not manual CAD semantic annotation accuracy.",
            "  Use motif_relation_evidence.csv to inspect the actual node-face/relation mapping.",
        ]
    )
    write_text(os.path.join(dirs["reports"], "motif_metrics_report.txt"), report)
    return {"graphs": graphs, "rows": per_sample_rows, "evidence_rows": evidence_rows, "stats": stats}


def write_audit_report(workdir: str, parse_res: Dict[str, Any], motif_res: Dict[str, Any], metrics_res: Dict[str, Any]) -> str:
    dirs = ensure_workdir(workdir)
    stats = metrics_res.get("stats", {})
    manifest = parse_res.get("manifest", [])
    rejected = parse_res.get("rejected", [])
    duplicates = parse_res.get("duplicates", [])
    graphs = motif_res.get("graphs", [])
    failures = motif_res.get("failures", [])

    report = [
        "Innovation1 v3 Full Audit Report",
        "=" * 72,
        f"Time: {timestamp()}",
        "",
        "1. Method position",
        "  - v3 is not a CAD renderer and not an aerospace-label parser.",
        "  - v3 extracts a weak B-Rep structural motif graph M=(Vm, Em, Pm).",
        "  - Motif labels are algorithm-extracted candidates, not manual truth.",
        "",
        "2. Public B-Rep parsing",
        f"  - Total scanned: {len(parse_res.get('step_files', []))}",
        f"  - Kept parsed samples: {len(manifest)}",
        f"  - Rejected samples: {len(rejected)}",
        f"  - Duplicate samples removed: {len(duplicates)}",
        "",
        "3. Motif extraction",
        f"  - Motif graph success: {len(graphs)}",
        f"  - Motif graph failures: {len(failures)}",
        f"  - Motif-rich count: {stats.get('motif_rich_count', 0)}",
        f"  - Average motif nodes: {stats.get('avg_motif_nodes', 0.0):.3f}",
        f"  - Average motif relations: {stats.get('avg_motif_relations', 0.0):.3f}",
        f"  - Relation evidence rows: {stats.get('relation_evidence_rows', 0)}",
        "",
        "4. Key relation totals",
    ]
    for rel_type in RELATION_TYPES:
        report.append(f"  - {rel_type}: {stats.get(f'rel_{rel_type}_count', 0)}")
    report.extend(
        [
            "",
            "5. Required deliverables",
            "  - outputs/motif_graphs/motif_graphs.jsonl",
            "  - outputs/reports/motif_stats.csv",
            "  - outputs/reports/motif_relation_evidence.csv",
            "  - outputs/reports/motif_metrics_report.txt",
            "  - outputs/visualizations/*__motif_debug.png",
            "",
            "6. Current limitation",
            "  - If the visualization shows only shaded CAD-like silhouettes, it is not enough.",
            "  - The required visualization must show node ids, face ids, motif types and relation labels.",
        ]
    )
    path = os.path.join(dirs["reports"], "innovation1_v3_audit_report.txt")
    write_text(path, report)
    return path
