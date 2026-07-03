# -*- coding: utf-8 -*-
"""Metrics and audit reports for innovation1 v3 motif graph extraction."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence

try:  # pragma: no cover
    from .utils_io import (
        NODE_TYPES,
        RELATION_TYPES,
        count_by_type,
        ensure_workdir,
        read_csv,
        read_json,
        read_jsonl,
        summarize_numeric,
        timestamp,
        write_csv,
        write_json,
        write_text,
    )
except ImportError:  # pragma: no cover
    from utils_io import (
        NODE_TYPES,
        RELATION_TYPES,
        count_by_type,
        ensure_workdir,
        read_csv,
        read_json,
        read_jsonl,
        summarize_numeric,
        timestamp,
        write_csv,
        write_json,
        write_text,
    )


LIMITATION_LINES = [
    "M is a weak structural motif prior, not manually annotated semantic ground truth.",
    "loop_or_hole is an internal-closure candidate, not a true engineering hole label.",
    "transition_group is a geometric-topological connector candidate and does not guarantee true fillet semantics.",
    "Public ABC/DeepCAD data are not forcibly mapped to aerospace composite semantics.",
]


def _safe_read_json(path: str) -> Dict[str, Any]:
    try:
        return read_json(path)
    except Exception:
        return {}


def _parse_summary_from_outputs(workdir: str) -> Dict[str, Any]:
    dirs = ensure_workdir(workdir)
    summary = _safe_read_json(os.path.join(dirs["reports"], "parse_summary.json"))
    if summary:
        return summary
    clean_rows = read_csv(os.path.join(dirs["parsed"], "clean_manifest.csv"))
    rejected_rows = read_csv(os.path.join(dirs["parsed"], "rejected_manifest.csv"))
    return {
        "scan_step_count": len(clean_rows) + len(rejected_rows),
        "parse_success_count": len(clean_rows) + sum(1 for r in rejected_rows if r.get("stage") == "filter"),
        "parse_failure_count": sum(1 for r in rejected_rows if r.get("stage") == "parse" and r.get("reject_reason") != "not_single_solid"),
        "single_entity_filter_count": sum(1 for r in rejected_rows if r.get("reject_reason") == "not_single_solid"),
        "face_count_over_limit_filter_count": sum(1 for r in rejected_rows if "face_count_over_limit" in str(r.get("reject_reason"))),
        "clean_sample_count": len(clean_rows),
    }


def _motif_rich_count(graphs: Sequence[Dict[str, Any]]) -> int:
    count = 0
    for graph in graphs:
        types = {
            str(node.get("type"))
            for node in graph.get("motif_nodes", [])
            if str(node.get("type")) not in {"face_group", "boundary_group"}
        }
        if len(types) >= 2:
            count += 1
    return count


def _aggregate_graphs(graphs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    node_counts = {typ: 0 for typ in NODE_TYPES}
    node_sample_counts = {typ: 0 for typ in NODE_TYPES}
    relation_counts = {typ: 0 for typ in RELATION_TYPES}
    relation_sample_counts = {typ: 0 for typ in RELATION_TYPES}
    motif_stats_totals: Dict[str, int] = {
        "num_parallel_pairs": 0,
        "num_opposite_pairs": 0,
        "num_orthogonal_pairs": 0,
        "num_coplanar_pairs": 0,
        "num_thin_wall_pairs": 0,
        "num_loop_candidates": 0,
        "num_transition_groups": 0,
        "num_repeated_features": 0,
    }
    face_counts: List[float] = []
    node_counts_per_graph: List[float] = []
    relation_counts_per_graph: List[float] = []
    motif_ready_rows: List[Dict[str, Any]] = []

    for graph in graphs:
        face_counts.append(float(graph.get("num_faces", 0)))
        node_counts_per_graph.append(float(len(graph.get("motif_nodes", []))))
        relation_counts_per_graph.append(float(len(graph.get("motif_relations", []))))
        present_node_types = set()
        present_relation_types = set()
        for node in graph.get("motif_nodes", []):
            typ = str(node.get("type", "unknown"))
            if typ in node_counts:
                node_counts[typ] += 1
                present_node_types.add(typ)
        for rel in graph.get("motif_relations", []):
            typ = str(rel.get("type", "unknown"))
            if typ in relation_counts:
                relation_counts[typ] += 1
                present_relation_types.add(typ)
        for typ in present_node_types:
            node_sample_counts[typ] += 1
        for typ in present_relation_types:
            relation_sample_counts[typ] += 1
        stats = graph.get("motif_stats", {})
        for key in motif_stats_totals:
            motif_stats_totals[key] += int(stats.get(key, 0))
        quality = graph.get("motif_quality", {})
        motif_ready_rows.append(
            {
                "uid": graph.get("uid", ""),
                "source": graph.get("source", ""),
                "num_faces": graph.get("num_faces", 0),
                "num_nodes": len(graph.get("motif_nodes", [])),
                "num_relations": len(graph.get("motif_relations", [])),
                "geometry_sampling_quality": graph.get("geometry_sampling_quality", ""),
                "dtg_train_compatible": int(bool(graph.get("dtg_train_compatible", False))),
                "dtg_filter_reason": graph.get("dtg_filter_reason", ""),
                "motif_ready": int(bool(quality.get("motif_ready", False))),
                "motif_quality_grade": quality.get("motif_quality_grade", "unknown"),
                "motif_quality_score": quality.get("motif_quality_score", 0.0),
                "non_base_motif_types": ";".join(quality.get("non_base_motif_types", [])),
                "core_motif_types": ";".join(quality.get("core_motif_types", [])),
                "key_relation_types": ";".join(quality.get("key_relation_types", [])),
                "quality_reasons": ";".join(quality.get("quality_reasons", [])),
            }
        )

    node_rows = [
        {
            "node_type": typ,
            "count": node_counts[typ],
            "sample_count": node_sample_counts[typ],
            "mean_per_success": round(node_counts[typ] / max(len(graphs), 1), 6),
        }
        for typ in NODE_TYPES
    ]
    relation_rows = [
        {
            "relation_type": typ,
            "count": relation_counts[typ],
            "sample_count": relation_sample_counts[typ],
            "mean_per_success": round(relation_counts[typ] / max(len(graphs), 1), 6),
        }
        for typ in RELATION_TYPES
    ]
    return {
        "node_counts": node_counts,
        "node_sample_counts": node_sample_counts,
        "relation_counts": relation_counts,
        "relation_sample_counts": relation_sample_counts,
        "motif_stats_totals": motif_stats_totals,
        "node_rows": node_rows,
        "relation_rows": relation_rows,
        "face_count_stats": summarize_numeric(face_counts),
        "node_count_stats": summarize_numeric(node_counts_per_graph),
        "relation_count_stats": summarize_numeric(relation_counts_per_graph),
        "motif_rich_sample_count": _motif_rich_count(graphs),
        "motif_ready_rows": motif_ready_rows,
        "motif_ready_count": sum(int(row["motif_ready"]) for row in motif_ready_rows),
        "motif_quality_grade_counts": count_by_type(motif_ready_rows, "motif_quality_grade"),
    }


def write_motif_reports(workdir: str, graphs: Sequence[Dict[str, Any]] | None = None, failures: Sequence[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    dirs = ensure_workdir(workdir)
    if graphs is None:
        graphs = read_jsonl(os.path.join(dirs["motif_graphs"], "motif_graph_index.jsonl"))
    if failures is None:
        failures = []
    graphs = list(graphs)
    failures = list(failures)
    parse_summary = _parse_summary_from_outputs(workdir)
    aggregate = _aggregate_graphs(graphs)

    node_stats_path = os.path.join(dirs["reports"], "motif_node_stats.csv")
    relation_stats_path = os.path.join(dirs["reports"], "motif_relation_stats.csv")
    motif_ready_path = os.path.join(dirs["reports"], "motif_ready_manifest.csv")
    write_csv(node_stats_path, aggregate["node_rows"], ["node_type", "count", "sample_count", "mean_per_success"])
    write_csv(relation_stats_path, aggregate["relation_rows"], ["relation_type", "count", "sample_count", "mean_per_success"])
    write_csv(
        motif_ready_path,
        aggregate["motif_ready_rows"],
        [
            "uid",
            "source",
            "num_faces",
            "num_nodes",
            "num_relations",
            "geometry_sampling_quality",
            "dtg_train_compatible",
            "dtg_filter_reason",
            "motif_ready",
            "motif_quality_grade",
            "motif_quality_score",
            "non_base_motif_types",
            "core_motif_types",
            "key_relation_types",
            "quality_reasons",
        ],
    )
    if failures:
        write_csv(os.path.join(dirs["reports"], "motif_failure_manifest.csv"), list(failures))

    extraction_report = [
        "Innovation1 v3 Motif Extraction Report",
        "=" * 72,
        f"Report time: {timestamp()}",
        "",
        f"Clean parsed samples: {parse_summary.get('clean_sample_count', 0)}",
        f"DTG-train-compatible clean samples: {parse_summary.get('dtg_train_compatible_count', 0)}",
        f"Motif graph generation success: {len(graphs)}",
        f"Motif graph generation failures: {len(failures)}",
        f"Motif-ready samples for training: {aggregate['motif_ready_count']}",
        f"Motif-rich samples with >=2 non-base motif types: {aggregate['motif_rich_sample_count']}",
        "",
        f"Face count min/mean/max: {aggregate['face_count_stats']}",
        f"Motif node count min/mean/max: {aggregate['node_count_stats']}",
        f"Motif relation count min/mean/max: {aggregate['relation_count_stats']}",
        "",
        "Motif node counts:",
    ]
    for row in aggregate["node_rows"]:
        extraction_report.append(f"  - {row['node_type']}: {row['count']} in {row['sample_count']} sample(s)")
    extraction_report.extend(["", "Motif relation counts:"])
    for row in aggregate["relation_rows"]:
        extraction_report.append(f"  - {row['relation_type']}: {row['count']} in {row['sample_count']} sample(s)")
    extraction_report.extend(["", "Face-level evidence totals:"])
    for key, value in aggregate["motif_stats_totals"].items():
        extraction_report.append(f"  - {key}: {value}")
    extraction_report.extend(["", "Motif-ready quality grades:"])
    for grade, value in sorted(aggregate["motif_quality_grade_counts"].items()):
        extraction_report.append(f"  - {grade}: {value}")
    extraction_report.extend(
        [
            "",
            "Interpretation:",
            "  Motif nodes are weak structural hypernodes over B-Rep face ids.",
            "  Motif relations are sparsified structural priors for training, not all pairwise face evidence.",
            "  Face-level evidence totals are reported for audit only and are not exported as dense training edges.",
            "  motif_ready_manifest.csv marks the strict high-quality subset; only high-grade samples enter motif_graph_index_ready.jsonl by default.",
            "  The output motif_prior block is intended as the interface for the later motif-guided neural generator.",
            "",
            "Current limitations:",
        ]
    )
    for line in LIMITATION_LINES:
        extraction_report.append(f"  - {line}")
    extraction_report_path = os.path.join(dirs["reports"], "motif_extraction_report.txt")
    write_text(extraction_report_path, extraction_report)

    audit_report = [
        "Innovation1 v3 Full Audit Report",
        "=" * 72,
        f"Audit time: {timestamp()}",
        "",
        "Research objective:",
        "  Extract a weak structural motif graph M=(Vm,Em,Pm) from semantics-free public B-Rep data.",
        "  M is designed as a structure-prior interface for later M -> face group -> topology -> geometry generation.",
        "",
        "1. STEP parse and clean summary",
        f"  - Scanned STEP files: {parse_summary.get('scan_step_count', 0)}",
        f"  - Parse success before clean filters: {parse_summary.get('parse_success_count', 0)}",
        f"  - Parse failures: {parse_summary.get('parse_failure_count', 0)}",
        f"  - Single-solid filter rejects: {parse_summary.get('single_entity_filter_count', 0)}",
        f"  - canonical face_count > {parse_summary.get('canonical_face_count_max', 50)} rejects: {parse_summary.get('face_count_over_limit_filter_count', 0)}",
        f"  - Final clean samples: {parse_summary.get('clean_sample_count', 0)}",
        f"  - DTG-train-compatible clean samples: {parse_summary.get('dtg_train_compatible_count', 0)}",
        f"  - canonical counting policy: {parse_summary.get('canonical_face_count_policy', 'DTG backend counts faces after parse_solid closed-face and closed-edge splitting.')}",
        "",
        "2. Motif graph extraction summary",
        f"  - Motif graph generation success: {len(graphs)}",
        f"  - Motif graph generation failures: {len(failures)}",
        f"  - Motif-ready samples for training: {aggregate['motif_ready_count']}",
        f"  - Motif-rich samples with >=2 non-base motif types: {aggregate['motif_rich_sample_count']}",
        "",
        "3. Motif node coverage",
    ]
    for row in aggregate["node_rows"]:
        audit_report.append(f"  - {row['node_type']}: {row['count']} total, {row['sample_count']} samples")
    audit_report.extend(["", "4. Motif relation coverage"])
    for row in aggregate["relation_rows"]:
        audit_report.append(f"  - {row['relation_type']}: {row['count']} total, {row['sample_count']} samples")
    audit_report.extend(
        [
            "",
            "5. Method position relative to DTG",
            "  - DTG directly models B-Rep topology/geometry distributions.",
            "  - v3 inserts a weak motif graph prior before face-group/topology/geometry generation.",
            "  - The first innovation is the semantics-free motif prior representation, not aerospace semantic classification.",
            "  - The output is intentionally compatible with a later motif-conditioned neural generator.",
            "",
            "6. Current limitations",
        ]
    )
    for line in LIMITATION_LINES:
        audit_report.append(f"  - {line}")
    audit_report.extend(
        [
            "",
            "Generated artifacts:",
            "  - outputs/parsed/clean_manifest.csv",
            "  - outputs/reports/rejected_manifest.csv",
            "  - outputs/motif_graphs/motif_graph_index.jsonl",
            "  - outputs/motif_graphs/motif_graph_index_ready.jsonl",
            "  - outputs/reports/motif_node_stats.csv",
            "  - outputs/reports/motif_relation_stats.csv",
            "  - outputs/reports/motif_ready_manifest.csv",
            "  - outputs/reports/motif_extraction_report.txt",
            "  - outputs/reports/method_summary.md",
        ]
    )
    audit_report_path = os.path.join(dirs["reports"], "innovation1_v3_audit_report.txt")
    write_text(audit_report_path, audit_report)

    summary = {
        "parse_summary": parse_summary,
        "motif_graph_success_count": len(graphs),
        "motif_graph_failure_count": len(failures),
        "node_counts": aggregate["node_counts"],
        "relation_counts": aggregate["relation_counts"],
        "motif_rich_sample_count": aggregate["motif_rich_sample_count"],
        "node_stats_csv": node_stats_path,
        "relation_stats_csv": relation_stats_path,
        "motif_ready_manifest": motif_ready_path,
        "motif_extraction_report": extraction_report_path,
        "audit_report": audit_report_path,
    }
    write_json(os.path.join(dirs["reports"], "motif_audit_summary.json"), summary)
    return summary
