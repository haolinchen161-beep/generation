# -*- coding: utf-8 -*-
"""创新点一 v3 结构基元图抽取的统计与审计报告。"""

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
    "M_raw 与 S 都是弱结构候选，不是人工标注语义真值。",
    "loop_or_hole 是内环或局部闭合候选，不是真实工程孔洞标签。",
    "transition_group 是几何拓扑连接候选，不保证真实圆角或过渡语义。",
    "公共 ABC/DeepCAD 数据不强行映射为航空复材语义。",
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
    role_counts = {"structural": 0, "support": 0, "topology_support": 0}
    role_sample_counts = {"structural": 0, "support": 0, "topology_support": 0}
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
        present_roles = set()
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
            role = str(rel.get("relation_role", "structural" if typ not in {"embedded_in", "adjacent_to"} else ("support" if typ == "embedded_in" else "topology_support")))
            if role in role_counts:
                role_counts[role] += 1
                present_roles.add(role)
        for typ in present_node_types:
            node_sample_counts[typ] += 1
        for typ in present_relation_types:
            relation_sample_counts[typ] += 1
        for role in present_roles:
            role_sample_counts[role] += 1
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
                "num_structural_relations": int(stats.get("num_structural_relations", 0)),
                "num_support_relations": int(stats.get("num_support_relations", 0)),
                "structural_relation_density": stats.get("structural_relation_density", 0.0),
                "support_relation_density": stats.get("support_relation_density", 0.0),
                "embedded_in_per_node": stats.get("embedded_in_per_node", 0.0),
                "embedded_in_per_sample": stats.get("embedded_in_per_sample", 0),
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
    role_rows = [
        {
            "relation_role": role,
            "count": role_counts[role],
            "sample_count": role_sample_counts[role],
            "mean_per_success": round(role_counts[role] / max(len(graphs), 1), 6),
        }
        for role in ["structural", "support", "topology_support"]
    ]
    return {
        "node_counts": node_counts,
        "node_sample_counts": node_sample_counts,
        "relation_counts": relation_counts,
        "relation_sample_counts": relation_sample_counts,
        "relation_role_counts": role_counts,
        "relation_role_sample_counts": role_sample_counts,
        "motif_stats_totals": motif_stats_totals,
        "node_rows": node_rows,
        "relation_rows": relation_rows,
        "relation_role_rows": role_rows,
        "face_count_stats": summarize_numeric(face_counts),
        "node_count_stats": summarize_numeric(node_counts_per_graph),
        "relation_count_stats": summarize_numeric(relation_counts_per_graph),
        "motif_rich_sample_count": _motif_rich_count(graphs),
        "motif_ready_rows": motif_ready_rows,
        "motif_ready_count": sum(int(row["motif_ready"]) for row in motif_ready_rows),
        "motif_quality_grade_counts": count_by_type(motif_ready_rows, "motif_quality_grade"),
    }


def _aggregate_prior_graphs(prior_graphs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    node_counts: List[float] = []
    relation_counts: List[float] = []
    compression_values: List[float] = []
    coverage_values: List[float] = []
    ready_count = 0
    rows: List[Dict[str, Any]] = []
    node_type_counts = {typ: 0 for typ in NODE_TYPES}
    relation_type_counts = {typ: 0 for typ in RELATION_TYPES}
    for graph in prior_graphs:
        stats = graph.get("motif_stats", {}) or {}
        nodes = graph.get("motif_nodes", [])
        relations = graph.get("motif_relations", [])
        node_count = int(stats.get("num_prior_nodes", len(nodes)))
        relation_count = int(stats.get("num_prior_relations", len(relations)))
        node_counts.append(float(node_count))
        relation_counts.append(float(relation_count))
        compression_values.append(float(stats.get("prior_compression_ratio", 0.0)))
        coverage_values.append(float(stats.get("prior_coverage_face_ratio", 0.0)))
        prior_ready = bool(graph.get("motif_quality", {}).get("motif_prior_ready", stats.get("prior_ready", False)))
        ready_count += int(prior_ready)
        for node in nodes:
            typ = str(node.get("type", "unknown"))
            if typ in node_type_counts:
                node_type_counts[typ] += 1
        for rel in relations:
            typ = str(rel.get("type", "unknown"))
            if typ in relation_type_counts:
                relation_type_counts[typ] += 1
        rows.append(
            {
                "uid": graph.get("uid", ""),
                "source": graph.get("source", ""),
                "num_faces": graph.get("num_faces", 0),
                "num_prior_nodes": node_count,
                "num_prior_relations": relation_count,
                "prior_node_density": stats.get("prior_node_density", 0.0),
                "prior_relation_density": stats.get("prior_relation_density", 0.0),
                "prior_retention_ratio": stats.get("prior_retention_ratio", stats.get("prior_compression_ratio", 0.0)),
                "prior_reduction_ratio": stats.get("prior_reduction_ratio", 1.0 - float(stats.get("prior_compression_ratio", 0.0))),
                "prior_compression_ratio": stats.get("prior_compression_ratio", 0.0),
                "prior_node_face_reduction_ratio": stats.get("prior_node_face_reduction_ratio", 1.0 - float(stats.get("prior_node_density", 0.0))),
                "prior_coverage_faces": stats.get("prior_coverage_faces", 0),
                "prior_coverage_face_ratio": stats.get("prior_coverage_face_ratio", 0.0),
                "prior_motif_types": ";".join(stats.get("prior_motif_types", [])),
                "prior_relation_types": ";".join(stats.get("prior_relation_types", [])),
                "motif_prior_ready": int(prior_ready),
            }
        )
    return {
        "prior_rows": rows,
        "prior_graph_count": len(prior_graphs),
        "motif_prior_ready_count": ready_count,
        "prior_node_count_stats": summarize_numeric(node_counts),
        "prior_relation_count_stats": summarize_numeric(relation_counts),
        "prior_compression_stats": summarize_numeric(compression_values),
        "prior_coverage_stats": summarize_numeric(coverage_values),
        "prior_node_type_counts": node_type_counts,
        "prior_relation_type_counts": relation_type_counts,
    }


def write_motif_reports(
    workdir: str,
    graphs: Sequence[Dict[str, Any]] | None = None,
    failures: Sequence[Dict[str, Any]] | None = None,
    prior_graphs: Sequence[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    dirs = ensure_workdir(workdir)
    if graphs is None:
        structural_index = os.path.join(dirs["motif_graphs"], "motif_graph_index_structural.jsonl")
        full_index = os.path.join(dirs["motif_graphs"], "motif_graph_index.jsonl")
        graphs = read_jsonl(structural_index if os.path.exists(structural_index) else full_index)
    if prior_graphs is None:
        prior_index = os.path.join(dirs["motif_graphs"], "motif_prior_index.jsonl")
        prior_graphs = read_jsonl(prior_index) if os.path.exists(prior_index) else []
    if failures is None:
        failures = []
    graphs = list(graphs)
    prior_graphs = list(prior_graphs)
    failures = list(failures)
    parse_summary = _parse_summary_from_outputs(workdir)
    aggregate = _aggregate_graphs(graphs)
    prior_aggregate = _aggregate_prior_graphs(prior_graphs)

    node_stats_path = os.path.join(dirs["reports"], "motif_node_stats.csv")
    relation_stats_path = os.path.join(dirs["reports"], "motif_relation_stats.csv")
    relation_role_stats_path = os.path.join(dirs["reports"], "motif_relation_role_stats.csv")
    motif_ready_path = os.path.join(dirs["reports"], "motif_ready_manifest.csv")
    prior_stats_path = os.path.join(dirs["reports"], "motif_prior_stats.csv")
    write_csv(node_stats_path, aggregate["node_rows"], ["node_type", "count", "sample_count", "mean_per_success"])
    write_csv(relation_stats_path, aggregate["relation_rows"], ["relation_type", "count", "sample_count", "mean_per_success"])
    write_csv(relation_role_stats_path, aggregate["relation_role_rows"], ["relation_role", "count", "sample_count", "mean_per_success"])
    write_csv(
        prior_stats_path,
        prior_aggregate["prior_rows"],
        [
            "uid",
            "source",
            "num_faces",
            "num_prior_nodes",
            "num_prior_relations",
            "prior_node_density",
            "prior_relation_density",
            "prior_retention_ratio",
            "prior_reduction_ratio",
            "prior_compression_ratio",
            "prior_node_face_reduction_ratio",
            "prior_coverage_faces",
            "prior_coverage_face_ratio",
            "prior_motif_types",
            "prior_relation_types",
            "motif_prior_ready",
        ],
    )
    write_csv(
        motif_ready_path,
        aggregate["motif_ready_rows"],
        [
            "uid",
            "source",
            "num_faces",
            "num_nodes",
            "num_relations",
            "num_structural_relations",
            "num_support_relations",
            "structural_relation_density",
            "support_relation_density",
            "embedded_in_per_node",
            "embedded_in_per_sample",
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
        "Innovation1 v3 结构基元图抽取报告",
        "=" * 72,
        f"报告时间：{timestamp()}",
        "",
        f"Clean parsed 样本数：{parse_summary.get('clean_sample_count', 0)}",
        f"DTG 训练兼容 clean 样本数：{parse_summary.get('dtg_train_compatible_count', 0)}",
        f"Motif graph 生成成功数：{len(graphs)}",
        f"Motif graph 生成失败数：{len(failures)}",
        f"生成先验 S 样本数：{prior_aggregate['prior_graph_count']}",
        f"motif-prior-ready 样本数：{prior_aggregate['motif_prior_ready_count']}",
        f"用于训练的 motif-ready 样本数：{aggregate['motif_ready_count']}",
        f"至少含 2 类非基础 motif 的 motif-rich 样本数：{aggregate['motif_rich_sample_count']}",
        "",
        f"Face count 最小/均值/最大：{aggregate['face_count_stats']}",
        f"Motif node count 最小/均值/最大：{aggregate['node_count_stats']}",
        f"Motif relation count 最小/均值/最大：{aggregate['relation_count_stats']}",
        f"S prior node count 最小/均值/最大：{prior_aggregate['prior_node_count_stats']}",
        f"S prior relation count 最小/均值/最大：{prior_aggregate['prior_relation_count_stats']}",
        f"S prior compression ratio 最小/均值/最大：{prior_aggregate['prior_compression_stats']}",
        f"S prior face coverage ratio 最小/均值/最大：{prior_aggregate['prior_coverage_stats']}",
        "",
        "Motif node 类型统计：",
    ]
    for row in aggregate["node_rows"]:
        extraction_report.append(f"  - {row['node_type']}: {row['count']} 条，覆盖 {row['sample_count']} 个样本")
    extraction_report.extend(["", "Motif relation 类型统计："])
    for row in aggregate["relation_rows"]:
        extraction_report.append(f"  - {row['relation_type']}: {row['count']} 条，覆盖 {row['sample_count']} 个样本")
    extraction_report.extend(["", "Relation role 统计："])
    for row in aggregate["relation_role_rows"]:
        extraction_report.append(f"  - {row['relation_role']}: {row['count']} 条，覆盖 {row['sample_count']} 个样本")
    extraction_report.extend(["", "生成先验 S 的 node 类型统计："])
    for typ, value in prior_aggregate["prior_node_type_counts"].items():
        if value:
            extraction_report.append(f"  - {typ}: {value}")
    extraction_report.extend(["", "生成先验 S 的 relation 类型统计："])
    for typ, value in prior_aggregate["prior_relation_type_counts"].items():
        if value:
            extraction_report.append(f"  - {typ}: {value}")
    extraction_report.extend(["", "面级证据（face-level evidence）总量："])
    for key, value in aggregate["motif_stats_totals"].items():
        extraction_report.append(f"  - {key}: {value}")
    extraction_report.extend(["", "Motif-ready 质量等级统计："])
    for grade, value in sorted(aggregate["motif_quality_grade_counts"].items()):
        extraction_report.append(f"  - {grade}: {value}")
    extraction_report.extend(
        [
            "",
            "解释说明：",
            "  M_raw 是从 B-Rep 抽取的弱结构证据全集，用于审计、统计和监督信号来源。",
            "  S = D(M_raw) 是面向生成的稀疏结构先验，默认写入 motif_prior_index_ready.jsonl。",
            "  face_group、embedded_in、adjacent_to 等支撑信息保留在 M_raw 中，但不默认进入 S。",
            "  S 只保留对无条件生成更有价值的 sheet/thin-wall/repeat/loop/transition/boundary 等结构骨架。",
            "  面级证据（face-level evidence）总量仅用于审计，不作为密集训练边导出。",
            "  motif_prior_stats.csv 记录 S 的压缩率、覆盖率、节点密度和关系密度。",
            "",
            "当前局限：",
        ]
    )
    for line in LIMITATION_LINES:
        extraction_report.append(f"  - {line}")
    extraction_report_path = os.path.join(dirs["reports"], "motif_extraction_report.txt")
    write_text(extraction_report_path, extraction_report)

    audit_report = [
        "Innovation1 v3 全流程审计报告",
        "=" * 72,
        f"审计时间：{timestamp()}",
        "",
        "研究目标：",
        "  从无语义公共 B-Rep 数据中抽取弱结构证据图 M_raw，并蒸馏生成稀疏结构先验 S。",
        "  M_raw 用于审计和监督信号来源；S 作为后续 S -> face group -> topology -> geometry 生成链路的结构先验接口。",
        "",
        "1. STEP 解析与清洗汇总",
        f"  - 扫描 STEP 文件数：{parse_summary.get('scan_step_count', 0)}",
        f"  - 清洗前解析成功数：{parse_summary.get('parse_success_count', 0)}",
        f"  - 解析失败数：{parse_summary.get('parse_failure_count', 0)}",
        f"  - 非单 solid 过滤数：{parse_summary.get('single_entity_filter_count', 0)}",
        f"  - 规范 face_count > {parse_summary.get('canonical_face_count_max', 50)} 过滤数：{parse_summary.get('face_count_over_limit_filter_count', 0)}",
        f"  - 最终 clean 样本数：{parse_summary.get('clean_sample_count', 0)}",
        f"  - DTG 训练兼容 clean 样本数：{parse_summary.get('dtg_train_compatible_count', 0)}",
        f"  - 规范计数策略：{parse_summary.get('canonical_face_count_policy', 'DTG 后端在 parse_solid 拆分 closed faces / closed edges 后统计 face_count。')}",
        "",
        "2. Motif graph 抽取汇总",
        f"  - Motif graph 生成成功数：{len(graphs)}",
        f"  - Motif graph 生成失败数：{len(failures)}",
        f"  - 生成先验 S 样本数：{prior_aggregate['prior_graph_count']}",
        f"  - motif-prior-ready 样本数：{prior_aggregate['motif_prior_ready_count']}",
        f"  - 用于训练的 motif-ready 样本数：{aggregate['motif_ready_count']}",
        f"  - 至少含 2 类非基础 motif 的 motif-rich 样本数：{aggregate['motif_rich_sample_count']}",
        "",
        "3. Motif node 覆盖情况",
    ]
    for row in aggregate["node_rows"]:
        audit_report.append(f"  - {row['node_type']}: 共 {row['count']} 条，覆盖 {row['sample_count']} 个样本")
    audit_report.extend(["", "4. Motif relation 覆盖情况"])
    for row in aggregate["relation_rows"]:
        audit_report.append(f"  - {row['relation_type']}: 共 {row['count']} 条，覆盖 {row['sample_count']} 个样本")
    audit_report.extend(["", "4b. Relation role 覆盖情况"])
    for row in aggregate["relation_role_rows"]:
        audit_report.append(f"  - {row['relation_role']}: 共 {row['count']} 条，覆盖 {row['sample_count']} 个样本")
    audit_report.extend(
        [
            "",
            "4c. 生成先验 S 蒸馏统计",
            f"  - S prior node count 最小/均值/最大：{prior_aggregate['prior_node_count_stats']}",
            f"  - S prior relation count 最小/均值/最大：{prior_aggregate['prior_relation_count_stats']}",
            f"  - S prior compression ratio 最小/均值/最大：{prior_aggregate['prior_compression_stats']}",
            f"  - S prior face coverage ratio 最小/均值/最大：{prior_aggregate['prior_coverage_stats']}",
        ]
    )
    audit_report.extend(
        [
            "",
            "5. 相对 DTG 的方法定位",
            "  - DTG 直接建模 B-Rep topology / geometry 分布。",
            "  - v3 先抽取 M_raw，再蒸馏出 S，避免把全量局部关系直接喂给生成网络。",
            "  - 第一个创新点是无语义结构证据抽取与生成先验蒸馏，不是航空复材语义分类。",
            "  - 输出接口有意对齐后续无条件 motif-prior 层级 B-Rep 生成网络。",
            "",
            "6. 当前局限",
        ]
    )
    for line in LIMITATION_LINES:
        audit_report.append(f"  - {line}")
    audit_report.extend(
        [
            "",
            "生成文件：",
            "  - outputs/parsed/clean_manifest.csv",
            "  - outputs/reports/rejected_manifest.csv",
            "  - outputs/motif_graphs/motif_graph_index.jsonl",
            "  - outputs/motif_graphs/motif_graph_index_structural.jsonl",
            "  - outputs/motif_graphs/motif_graph_index_ready.jsonl",
            "  - outputs/motif_graphs/motif_prior_index.jsonl",
            "  - outputs/motif_graphs/motif_prior_index_ready.jsonl",
            "  - outputs/reports/motif_node_stats.csv",
            "  - outputs/reports/motif_relation_stats.csv",
            "  - outputs/reports/motif_prior_stats.csv",
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
        "relation_role_counts": aggregate["relation_role_counts"],
        "motif_rich_sample_count": aggregate["motif_rich_sample_count"],
        "motif_prior_graph_count": prior_aggregate["prior_graph_count"],
        "motif_prior_ready_count": prior_aggregate["motif_prior_ready_count"],
        "prior_node_count_stats": prior_aggregate["prior_node_count_stats"],
        "prior_relation_count_stats": prior_aggregate["prior_relation_count_stats"],
        "prior_compression_stats": prior_aggregate["prior_compression_stats"],
        "prior_coverage_stats": prior_aggregate["prior_coverage_stats"],
        "node_stats_csv": node_stats_path,
        "relation_stats_csv": relation_stats_path,
        "relation_role_stats_csv": relation_role_stats_path,
        "motif_prior_stats_csv": prior_stats_path,
        "motif_ready_manifest": motif_ready_path,
        "motif_extraction_report": extraction_report_path,
        "audit_report": audit_report_path,
    }
    write_json(os.path.join(dirs["reports"], "motif_audit_summary.json"), summary)
    return summary
