# -*- coding: utf-8 -*-
"""Entry point for innovation1 v3 B-Rep motif graph extraction."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

from motif_graph_builder import build_motif_graphs
from public_brep_parser import parse_public_dataset
from tools.motif_metrics import evaluate_motif, write_audit_report
from tools.motif_visualizer import visualize_motif
from utils_io import ensure_workdir, scan_step_files, timestamp, write_text


def inspect_source(source_dir: str, workdir: str, max_files: int = 0) -> Dict[str, Any]:
    dirs = ensure_workdir(workdir)
    step_files = scan_step_files(source_dir, max_files=0)
    preview = step_files[: min(20, len(step_files))]
    report = [
        "Innovation1 v3 Source Inspection Report",
        "=" * 72,
        f"Time: {timestamp()}",
        f"Source dir: {source_dir}",
        f"Total STEP/STP files found: {len(step_files)}",
        f"Configured max_files for parse: {max_files if max_files else 'all'}",
        "",
        "Preview:",
    ]
    for path in preview:
        report.append(f"  - {path}")
    report.extend(
        [
            "",
            "Scope boundaries:",
            "  - Do not modify original DTG / DTGBrepGen source.",
            "  - Do not modify cfg_brepgen_v1/innovation1_core_programs.",
            "  - Do not modify innovation1_v2_struct_semantic_parser.",
            "  - Do not modify innovation2_struct_prior_brepgen.",
            "  - Public-data labels are algorithm-extracted motifs, not manual truth.",
        ]
    )
    path = os.path.join(dirs["reports"], "source_inspection_report.txt")
    write_text(path, report)
    return {"step_count": len(step_files), "report_path": path}


def audit_all(args: argparse.Namespace) -> Dict[str, Any]:
    inspect_res = inspect_source(args.source_dir, args.workdir, args.max_files)
    parse_res = parse_public_dataset(args.source_dir, args.workdir, args.max_files)
    motif_res = build_motif_graphs(args.workdir)
    vis_res = visualize_motif(args.workdir, args.num_visualizations)
    metrics_res = evaluate_motif(args.workdir)
    report_path = write_audit_report(args.workdir, parse_res, motif_res, metrics_res)
    return {
        "inspect": inspect_res,
        "parse": parse_res,
        "motif": motif_res,
        "visualize": vis_res,
        "metrics": metrics_res,
        "report_path": report_path,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Innovation1 v3 B-Rep motif graph extraction")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["inspect_source", "parse_public", "extract_motif", "visualize_motif", "evaluate_motif", "audit_all"],
    )
    parser.add_argument("--source_dir", default="data")
    parser.add_argument("--workdir", default="innovation1_v3_brep_motif_graph")
    parser.add_argument("--max_files", type=int, default=200, help="0 means parse all STEP files; default keeps the first 200 for manageable audit.")
    parser.add_argument("--num_visualizations", type=int, default=20)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    ensure_workdir(args.workdir)
    if args.mode == "inspect_source":
        result = inspect_source(args.source_dir, args.workdir, args.max_files)
        print(f"[inspect_source] step_count={result['step_count']}, report={result['report_path']}")
    elif args.mode == "parse_public":
        result = parse_public_dataset(args.source_dir, args.workdir, args.max_files)
        print(f"[parse_public] kept={len(result['manifest'])}, rejected={len(result['rejected'])}, duplicates={len(result['duplicates'])}")
    elif args.mode == "extract_motif":
        result = build_motif_graphs(args.workdir)
        print(f"[extract_motif] graphs={len(result['graphs'])}, failures={len(result['failures'])}")
    elif args.mode == "visualize_motif":
        result = visualize_motif(args.workdir, args.num_visualizations)
        print(f"[visualize_motif] selected={result['selected_count']}, png={len(result['written'])}")
    elif args.mode == "evaluate_motif":
        result = evaluate_motif(args.workdir)
        print(f"[evaluate_motif] stats={result['stats']}")
    elif args.mode == "audit_all":
        result = audit_all(args)
        print(f"[audit_all] report={result['report_path']}")
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
