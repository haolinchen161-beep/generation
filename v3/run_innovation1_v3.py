# -*- coding: utf-8 -*-
"""Entry point for innovation1 v3 B-Rep motif graph extraction.

v3 focuses on a readable, inspectable motif graph M=(Vm, Em, Pm), not on
CAD-style shaded screenshots.  The visualizer now exports diagnostic PNG files
that show motif node ids, face ids and relation labels explicitly.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import public_brep_parser as public_parser
from motif_graph_builder import build_motif_graphs
from tools.motif_metrics import evaluate_motif, write_audit_report
from tools.motif_visualizer import visualize_motif
from utils_io import ensure_workdir, scan_step_files, timestamp, write_text


def _resolve_source_dir(args: argparse.Namespace) -> str:
    # --step_root is kept as an alias because earlier commands used it.
    return str(args.step_root or args.source_dir)


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
            "",
            "Expected v3 outputs:",
            "  - outputs/motif_graphs/motif_graphs.jsonl",
            "  - outputs/reports/motif_relation_evidence.csv",
            "  - outputs/visualizations/*__motif_debug.png",
        ]
    )
    path = os.path.join(dirs["reports"], "source_inspection_report.txt")
    write_text(path, report)
    return {"step_count": len(step_files), "report_path": path}


def audit_all(args: argparse.Namespace) -> Dict[str, Any]:
    source_dir = _resolve_source_dir(args)
    public_parser.MAX_FACES = int(args.max_faces)
    inspect_res = inspect_source(source_dir, args.workdir, args.max_files)
    parse_res = public_parser.parse_public_dataset(source_dir, args.workdir, args.max_files)
    motif_res = build_motif_graphs(args.workdir)
    metrics_res = evaluate_motif(args.workdir)
    vis_res = visualize_motif(args.workdir, args.num_visualizations)
    report_path = write_audit_report(args.workdir, parse_res, motif_res, metrics_res)
    return {
        "inspect": inspect_res,
        "parse": parse_res,
        "motif": motif_res,
        "metrics": metrics_res,
        "visualize": vis_res,
        "report_path": report_path,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Innovation1 v3 B-Rep motif graph extraction")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["inspect_source", "parse_public", "extract_motif", "visualize_motif", "evaluate_motif", "audit_all"],
    )
    parser.add_argument("--source_dir", default="data", help="Folder containing ABC/DeepCAD STEP files.")
    parser.add_argument("--step_root", default="", help="Alias of --source_dir for previous command compatibility.")
    parser.add_argument("--workdir", default="v3")
    parser.add_argument("--max_files", type=int, default=200, help="0 means parse all STEP files; default keeps the first 200 for manageable audit.")
    parser.add_argument("--limit", type=int, default=-1, help="Alias of --max_files. If >=0, overrides --max_files.")
    parser.add_argument("--max_faces", type=int, default=70, help="Strict face-count cutoff for stable single-solid B-Reps.")
    parser.add_argument("--num_visualizations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42, help="Reserved for future randomized selection; current pipeline is deterministic.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if int(args.limit) >= 0:
        args.max_files = int(args.limit)
    source_dir = _resolve_source_dir(args)
    public_parser.MAX_FACES = int(args.max_faces)
    ensure_workdir(args.workdir)

    if args.mode == "inspect_source":
        result = inspect_source(source_dir, args.workdir, args.max_files)
        print(f"[inspect_source] step_count={result['step_count']}, report={result['report_path']}")
    elif args.mode == "parse_public":
        result = public_parser.parse_public_dataset(source_dir, args.workdir, args.max_files)
        print(f"[parse_public] kept={len(result['manifest'])}, rejected={len(result['rejected'])}, duplicates={len(result['duplicates'])}")
    elif args.mode == "extract_motif":
        result = build_motif_graphs(args.workdir)
        print(f"[extract_motif] graphs={len(result['graphs'])}, failures={len(result['failures'])}")
        metrics = evaluate_motif(args.workdir)
        print(f"[extract_motif] evidence_rows={metrics['stats'].get('relation_evidence_rows', 0)}")
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
