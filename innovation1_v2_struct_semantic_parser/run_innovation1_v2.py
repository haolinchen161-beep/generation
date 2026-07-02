# -*- coding: utf-8 -*-
"""Command entry for innovation1 v2 structural-semantic parser."""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.simplefilter("ignore")

from brep_semantic_weak_parser import parse_enhanced_dataset
from consistency_metrics import evaluate_consistency
from enhanced_dataset_generator import generate_enhanced_dataset
from graph_inference import infer_semantics
from utils_io import ensure_workdir, timestamp, write_text


def inspect_source(source_dir: str, workdir: str) -> Dict[str, Any]:
    dirs = ensure_workdir(workdir)
    src = Path(source_dir)
    reports_dir = dirs["reports"]
    files = sorted(path for path in src.rglob("*") if path.is_file())
    rel_files = [str(path.relative_to(src)).replace("\\", "/") for path in files]
    py_files = [item for item in rel_files if item.endswith(".py")]
    config_files = [item for item in rel_files if item.startswith("configs/")]
    doc_files = [item for item in rel_files if item.lower().endswith((".md", ".txt"))]

    keyword_hits: Dict[str, List[str]] = {
        "configuration_graph": [],
        "face_group": [],
        "tensor_schema": [],
        "edgeFace_adj": [],
        "face_bbox_wcs": [],
    }
    for path in files:
        if path.suffix.lower() not in {".py", ".md", ".yaml", ".yml", ".txt"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        rel = str(path.relative_to(src)).replace("\\", "/")
        for keyword in keyword_hits:
            if keyword in text:
                keyword_hits[keyword].append(rel)

    report = [
        "Innovation1 v2 Source Program Inspection Report",
        "=" * 72,
        f"Inspection time: {timestamp()}",
        f"Source directory: {src.resolve()}",
        "",
        "Policy:",
        "  This v2 reuses the ideas, interfaces, data schema and face-group alignment logic from innovation1.",
        "  It does not modify files under cfg_brepgen_v1/innovation1_core_programs.",
        "  It does not modify DTG/DTGBrepGen baseline source code.",
        "",
        f"Total source files found: {len(rel_files)}",
        "",
        "Core Python programs:",
    ]
    for item in py_files:
        report.append(f"  - {item}")
    report.extend(["", "Config files:"])
    for item in config_files:
        report.append(f"  - {item}")
    report.extend(["", "Documentation files:"])
    for item in doc_files:
        report.append(f"  - {item}")

    report.extend(
        [
            "",
            "Detected reusable interfaces:",
            "  - Gc=(Vc,Ec,P) / configuration_graph JSON interface",
            "  - procedural parameters P attached to each sample JSON",
            "  - DTG-compatible B-Rep PKL fields: face_wcs, edge_wcs, face_bbox_wcs, edge_bbox_wcs, edgeFace_adj, edgeVert_adj, faceEdge_adj, vert_wcs",
            "  - face group alignment index for mapping configuration nodes to B-Rep face ids",
            "  - tensor_schema/data_splits/weak_aligned_face_group_index output style for innovation2 training",
            "",
            "Keyword evidence:",
        ]
    )
    for keyword, paths in keyword_hits.items():
        report.append(f"  - {keyword}: {len(paths)} file(s)")
        for rel in paths[:8]:
            report.append(f"      * {rel}")

    report_path = os.path.join(reports_dir, "auxiliary", "source_program_inspection_report.txt")
    write_text(report_path, report)
    return {"source_dir": str(src), "files": rel_files, "report_path": report_path}


def audit_all(args: argparse.Namespace) -> Dict[str, Any]:
    dirs = ensure_workdir(args.workdir)
    inspect_res = inspect_source(args.source_dir, args.workdir)
    gen_res = generate_enhanced_dataset(args.workdir, args.num_per_type, args.seed)
    parse_res = parse_enhanced_dataset(args.workdir, args.seed)
    infer_res = infer_semantics(args.workdir)
    eval_res = evaluate_consistency(args.workdir)

    records = gen_res.get("records", [])
    generated = sum(int(r.get("json_success", 0)) for r in records)
    mechanism_counts: Dict[str, int] = {}
    for rec in records:
        for mech in str(rec.get("topology_mechanisms", "")).split(";"):
            if mech:
                mechanism_counts[mech] = mechanism_counts.get(mech, 0) + 1

    parsed_records = parse_res.get("records", [])
    parsed_success = sum(1 for r in parsed_records if r.get("parse_status") == "SUCCESS")
    lite_count = len(parse_res.get("lite_uids", []))
    inferred_success = int(infer_res.get("success", 0))
    summary_rows = eval_res.get("summary_rows", [])
    all_summary = summary_rows[0] if summary_rows else {}

    report = [
        "Innovation1 v2 Full Audit Report",
        "=" * 72,
        f"Audit time: {timestamp()}",
        "",
        "1. v2 additions over original innovation1",
        "  - Enhanced aerospace composite thin-wall stiffened part generator.",
        "  - B-Rep weak semantic parser that infers face roles, face groups, Gc and P from geometry/topology.",
        "  - procedural_Gc vs inferred_Gc consistency evaluation.",
        "  - Innovation2-ready enhanced_dataset, enhanced_parsed, inferred_semantics outputs.",
        "",
        "2. Enhanced dataset scale",
        f"  - Generated samples: {generated}",
        f"  - Requested per type: {args.num_per_type}",
        "",
        "3. Enhanced topology mechanism coverage",
    ]
    for mech, count in sorted(mechanism_counts.items()):
        report.append(f"  - {mech}: {count}")
    report.extend(
        [
            "",
            "4. B-Rep parse success",
            f"  - Parsed successfully: {parsed_success} / {len(parsed_records)}",
            f"  - Parse success rate: {parsed_success / max(len(parsed_records), 1):.4f}",
            f"  - Enhanced-lite within current innovation2 limits: {lite_count}",
            "",
            "5. Weak semantic parse success",
            f"  - Inferred successfully: {inferred_success} / {parsed_success}",
            f"  - Inference success rate: {inferred_success / max(parsed_success, 1):.4f}",
            "",
            "6. procedural_Gc vs inferred_Gc consistency",
        ]
    )
    if all_summary:
        report.extend(
            [
                f"  - weak_face_role_consistency: {all_summary.get('weak_face_role_consistency', 0.0)}",
                f"  - weak_face_group_iou: {all_summary.get('weak_face_group_iou', 0.0)}",
                f"  - assign_ratio: {all_summary.get('assign_ratio', 0.0)}",
                f"  - node_type_count_consistency: {all_summary.get('node_type_count_consistency', 0.0)}",
                f"  - relation_type_count_consistency: {all_summary.get('relation_type_count_consistency', 0.0)}",
                f"  - relation_triplet_overlap: {all_summary.get('relation_triplet_overlap', 0.0)}",
                f"  - parameter_l1: {all_summary.get('parameter_l1', 0.0)}",
                f"  - topology_mechanism_acc: {all_summary.get('topology_mechanism_acc', 0.0)}",
            ]
        )
    report.extend(
        [
            "",
            "7. Current limitations",
            "  - Enhanced samples are still procedural constructive CAD, not full industrial aircraft composite CAD.",
            "  - inferred_Gc is rule-based weak parsing, not manually annotated engineering truth.",
            "  - Curved panels use circular-arc swept solids; tapered C/hat and runout ribs use lofted closed sections.",
            "  - Main structural transitions are modeled explicitly: rib-root arcs, web-flange/web-cap arcs, rounded rectangular cutout corners and runout loft transitions.",
            "  - Global all-edge fillets are intentionally avoided to control face/edge growth and OCC robustness.",
            "  - Some enhanced samples may exceed innovation2 current max_faces/max_edges/max_vertices limits; see enhanced_parse_report.txt.",
            "",
            "8. Suggestions for innovation2 interface",
            "  - Use enhanced_parsed/tensor_schema.json, data_splits.csv and weak_aligned_face_group_index.jsonl as the training interface.",
            "  - Consider increasing model max_faces/max_edges/max_vertices or filtering samples listed as over-limit.",
            "  - Treat inferred_semantics as weak auxiliary supervision and procedural_Gc as synchronized procedural label.",
            "",
            "Auxiliary reports: outputs/reports/auxiliary/",
            "Source inspection report: outputs/reports/auxiliary/source_program_inspection_report.txt",
        ]
    )
    report_path = os.path.join(dirs["reports"], "innovation1_v2_audit_report.txt")
    write_text(report_path, report)
    return {
        "inspect": inspect_res,
        "generate": gen_res,
        "parse": parse_res,
        "infer": infer_res,
        "evaluate": eval_res,
        "report_path": report_path,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Innovation1 v2 structural semantic parser")
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "inspect_source",
            "generate_enhanced",
            "parse_enhanced",
            "infer_semantics",
            "evaluate_consistency",
            "audit_all",
        ],
    )
    parser.add_argument("--source_dir", default="cfg_brepgen_v1/innovation1_core_programs")
    parser.add_argument("--workdir", default="innovation1_v2_struct_semantic_parser")
    parser.add_argument("--num_per_type", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    ensure_workdir(args.workdir)

    if args.mode == "inspect_source":
        result = inspect_source(args.source_dir, args.workdir)
        print(f"[inspect_source] report: {result['report_path']}")
    elif args.mode == "generate_enhanced":
        result = generate_enhanced_dataset(args.workdir, args.num_per_type, args.seed)
        print(f"[generate_enhanced] report: {result['report_path']}")
    elif args.mode == "parse_enhanced":
        result = parse_enhanced_dataset(args.workdir, args.seed)
        print(f"[parse_enhanced] parsed: {len(result['parsed_uids'])}, failures: {len(result['failures'])}")
    elif args.mode == "infer_semantics":
        result = infer_semantics(args.workdir)
        print(f"[infer_semantics] success: {result['success']}, failures: {len(result['failures'])}")
    elif args.mode == "evaluate_consistency":
        result = evaluate_consistency(args.workdir)
        print(f"[evaluate_consistency] evaluated: {len(result['rows'])}, skipped: {len(result['skipped'])}")
    elif args.mode == "audit_all":
        result = audit_all(args)
        print(f"[audit_all] report: {result['report_path']}")
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
