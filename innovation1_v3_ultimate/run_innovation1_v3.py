# -*- coding: utf-8 -*-
"""创新点一 v3：无语义 B-Rep 弱结构基元图抽取入口。"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.simplefilter("ignore")

try:  # pragma: no cover
    from .brep_loader import parse_abc_dataset, rebuild_manifest_from_existing_pkl
    from .motif_graph_builder import extract_motif_graphs
    from .motif_metrics import write_motif_reports
    from .motif_visualizer import visualize_motifs
    from .utils_io import ensure_dir, ensure_workdir, timestamp, write_text
except ImportError:  # pragma: no cover
    from brep_loader import parse_abc_dataset, rebuild_manifest_from_existing_pkl
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
        write_text(dst, src.read_text(encoding="utf-8-sig", errors="ignore"))
    else:
        write_text(
            dst,
            [
                "# Innovation1 v3 Method Summary",
                "",
                f"生成时间：{timestamp()}。",
                "",
                "本模块从无语义公共 B-Rep 数据中抽取弱结构基元图 M。",
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
        num_workers=args.num_workers,
        task_timeout_sec=args.task_timeout_sec,
    )


def rebuild_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    return rebuild_manifest_from_existing_pkl(
        step_root=args.step_root,
        workdir=args.workdir,
        source=args.source,
        limit=args.limit,
        max_faces=args.max_faces,
    )


def extract_motif(args: argparse.Namespace) -> Dict[str, Any]:
    result = extract_motif_graphs(args.workdir)
    report_graphs = result.get("graphs", [])
    report = write_motif_reports(
        args.workdir,
        report_graphs,
        result.get("failures", []),
        result.get("prior_graphs", []),
    )
    copy_method_summary(args.workdir)
    result["report"] = report
    return result


def visualize_motif(args: argparse.Namespace) -> Dict[str, Any]:
    uid_list: List[str] | None = None
    if args.uids:
        uid_list = [item.strip() for item in str(args.uids).split(",") if item.strip()]
    return visualize_motifs(
        args.workdir,
        count=args.vis_count,
        uids=uid_list,
        ready_only=not args.visualize_all_clean,
        paper_vis=not args.debug_visualization,
    )


def audit_all(args: argparse.Namespace) -> Dict[str, Any]:
    parse_result = parse_abc(args)
    motif_result = extract_motif_graphs(args.workdir)
    report_graphs = motif_result.get("graphs", [])
    report = write_motif_reports(
        args.workdir,
        report_graphs,
        motif_result.get("failures", []),
        motif_result.get("prior_graphs", []),
    )
    method_summary = copy_method_summary(args.workdir)
    vis_result = visualize_motifs(
        args.workdir,
        count=args.vis_count,
        ready_only=not args.visualize_all_clean,
        paper_vis=not args.debug_visualization,
    )
    return {
        "parse": parse_result,
        "motif": motif_result,
        "report": report,
        "method_summary": method_summary,
        "visualization": vis_result,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="创新点一 v3：无语义 B-Rep 弱结构基元图抽取")
    parser.add_argument("--mode", required=True, choices=["parse_abc", "rebuild_manifest", "extract_motif", "visualize_motif", "audit_all"])
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR)
    parser.add_argument("--step_root", default=DEFAULT_STEP_ROOT)
    parser.add_argument("--source", default="abc", choices=["abc", "deepcad", "self"])
    parser.add_argument("--limit", type=int, default=0, help="调试用 STEP 数量上限；0 表示扫描全部文件。")
    parser.add_argument("--max_faces", type=int, default=50, help="默认采用比 DTG 更严格的结构先验训练过滤；仅做解析审计时可设为 70。")
    parser.add_argument("--num_workers", type=int, default=1, help="STEP 并行解析进程数；1 表示串行，大规模 ABC 可尝试 4-8。")
    parser.add_argument("--task_timeout_sec", type=int, default=900, help="单个 STEP 解析超时秒数；超时样本记入 rejected_manifest，默认 900。")
    parser.add_argument("--vis_count", type=int, default=12)
    parser.add_argument("--uids", default="", help="visualize_motif 指定样本 uid，多个 uid 用英文逗号分隔。")
    parser.add_argument("--visualize_all_clean", action="store_true", help="可视化全部 clean 图；默认只可视化 motif-ready 子集。")
    parser.add_argument("--debug_visualization", action="store_true", help="调试图显示 support/topology-support 关系；默认论文图只显示 structural relations。")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    ensure_workdir(args.workdir)

    if args.mode == "parse_abc":
        result = parse_abc(args)
        print(
            "[parse_abc] "
            f"clean样本={len(result.get('records', []))}, rejected样本={len(result.get('rejected', []))}, "
            f"清单={result.get('clean_manifest')}"
        )
    elif args.mode == "rebuild_manifest":
        result = rebuild_manifest(args)
        print(
            "[rebuild_manifest] "
            f"clean样本={len(result.get('records', []))}, "
            f"rejected样本={len(result.get('rejected', []))}, "
            f"unresolved样本={len(result.get('unresolved', []))}, "
            f"清单={result.get('clean_manifest')}"
        )
    elif args.mode == "extract_motif":
        result = extract_motif(args)
        print(
            "[extract_motif] "
            f"成功={len(result.get('graphs', []))}, 失败={len(result.get('failures', []))}, "
            f"prior_ready={len(result.get('prior_ready_graphs', []))}, "
            f"索引={result.get('motif_graph_index')}, prior={result.get('motif_prior_index')}"
        )
    elif args.mode == "visualize_motif":
        result = visualize_motif(args)
        print(
            f"[visualize_motif] 已可视化={result.get('visualized', 0)}, "
            f"paper_vis={result.get('paper_vis', True)}, index={result.get('index_path', '')}"
        )
    elif args.mode == "audit_all":
        result = audit_all(args)
        report_path = result.get("report", {}).get("audit_report", "")
        print(f"[audit_all] 审计报告={report_path}")
    else:
        raise ValueError(f"不支持的 mode：{args.mode}")


if __name__ == "__main__":
    main()
