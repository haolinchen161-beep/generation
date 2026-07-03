# -*- coding: utf-8 -*-
"""Command entry for innovation1 v3 weak B-Rep motif graph extraction."""

from __future__ import annotations

import argparse
import os
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, List

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.simplefilter("ignore")

try:  # pragma: no cover
    from .brep_loader import parse_abc_dataset
    from .motif_graph_builder import extract_motif_graphs
    from .motif_metrics import write_motif_reports
    from .motif_visualizer import visualize_motifs
    from .utils_io import ensure_dir, ensure_workdir, timestamp, write_text
except ImportError:  # pragma: no cover
    from brep_loader import parse_abc_dataset
    from motif_graph_builder import extract_motif_graphs
    from motif_metrics import write_motif_reports
    from motif_visualizer import visualize_motifs
    from utils_io import ensure_dir, ensure_workdir, timestamp, write_text


DEFAULT_WORKDIR = str(Path(__file__).resolve().parent)
DEFAULT_STEP_ROOT = str(Path(__file__).resolve().parent / "data")


def copy_method_summary(workdir: str) -> str:
    dirs = ensure_workdir(workdir)
    src = Path(workdir) / "method_summary.md"
    dst = Path(dirs["reports"]) / "method_summary.md"
    if src.exists():
        ensure_dir(dst.parent)
        shutil.copyfile(src, dst)
    else:
        write_text(
            dst,
            [
                "# Innovation1 v3 Method Summary",
                "",
                f"Generated at {timestamp()}.",
                "",
                "This module extracts a weak structural motif graph M from semantics-free B-Rep data.",
            ],
        )
    return str(dst)


def parse_abc(args: argparse.Namespace) -> Dict[str, Any]:
    return parse_abc_dataset(
        step_root=args.step_root,
        workdir=args.workdir,
        source=args.source,
        limit=args.limit,
        max_faces=args.max_faces,
    )


def extract_motif(args: argparse.Namespace) -> Dict[str, Any]:
    result = extract_motif_graphs(args.workdir)
    report = write_motif_reports(args.workdir, result.get("graphs", []), result.get("failures", []))
    copy_method_summary(args.workdir)
    result["report"] = report
    return result


def visualize_motif(args: argparse.Namespace) -> Dict[str, Any]:
    uid_list: List[str] | None = None
    if args.uids:
        uid_list = [item.strip() for item in str(args.uids).split(",") if item.strip()]
    return visualize_motifs(args.workdir, count=args.vis_count, uids=uid_list, ready_only=not args.visualize_all_clean)


def audit_all(args: argparse.Namespace) -> Dict[str, Any]:
    parse_result = parse_abc(args)
    motif_result = extract_motif_graphs(args.workdir)
    report = write_motif_reports(args.workdir, motif_result.get("graphs", []), motif_result.get("failures", []))
    method_summary = copy_method_summary(args.workdir)
    vis_result = visualize_motifs(args.workdir, count=args.vis_count, ready_only=not args.visualize_all_clean)
    return {
        "parse": parse_result,
        "motif": motif_result,
        "report": report,
        "method_summary": method_summary,
        "visualization": vis_result,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Innovation1 v3 weak B-Rep motif graph extraction")
    parser.add_argument("--mode", required=True, choices=["parse_abc", "extract_motif", "visualize_motif", "audit_all"])
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR)
    parser.add_argument("--step_root", default=DEFAULT_STEP_ROOT)
    parser.add_argument("--source", default="abc", choices=["abc", "deepcad", "self"])
    parser.add_argument("--limit", type=int, default=0, help="Optional STEP limit for smoke tests; 0 means all files.")
    parser.add_argument("--max_faces", type=int, default=50, help="Default follows stricter ABC/DTG training-style filtering; use 70 only for parse audit.")
    parser.add_argument("--vis_count", type=int, default=12)
    parser.add_argument("--uids", default="", help="Comma-separated uid list for visualize_motif.")
    parser.add_argument("--visualize_all_clean", action="store_true", help="Visualize all clean graphs instead of the default motif-ready subset.")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    ensure_workdir(args.workdir)

    if args.mode == "parse_abc":
        result = parse_abc(args)
        print(
            "[parse_abc] "
            f"clean={len(result.get('records', []))}, rejected={len(result.get('rejected', []))}, "
            f"manifest={result.get('clean_manifest')}"
        )
    elif args.mode == "extract_motif":
        result = extract_motif(args)
        print(
            "[extract_motif] "
            f"success={len(result.get('graphs', []))}, failures={len(result.get('failures', []))}, "
            f"index={result.get('motif_graph_index')}"
        )
    elif args.mode == "visualize_motif":
        result = visualize_motif(args)
        print(f"[visualize_motif] visualized={result.get('visualized', 0)}, index={result.get('index_path', '')}")
    elif args.mode == "audit_all":
        result = audit_all(args)
        report_path = result.get("report", {}).get("audit_report", "")
        print(f"[audit_all] report={report_path}")
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
