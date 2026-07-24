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
    from .brep_loader import parse_abc_dataset
    from .motif_graph_builder import build_compact_dataset_from_steps, extract_motif_graphs
    from .motif_visualizer import visualize_motifs
    from .utils_io import ensure_dir, ensure_workdir, timestamp, write_text
except ImportError:  # pragma: no cover
    from brep_loader import parse_abc_dataset
    from motif_graph_builder import build_compact_dataset_from_steps, extract_motif_graphs
    from motif_visualizer import visualize_motifs
    from utils_io import ensure_dir, ensure_workdir, timestamp, write_text


DEFAULT_WORKDIR = str(Path(__file__).resolve().parent / "output_deepcad")
DEFAULT_STEP_ROOT = str(Path(__file__).resolve().parent / "deepcad_data" / "cad_step")
DEFAULT_SPLIT_FILE = str(Path(__file__).resolve().parents[1] / "deepcad_data_split_6bit.pkl")


def copy_method_summary(workdir: str) -> str:
    dirs = ensure_workdir(workdir)
    src = Path(__file__).resolve().parent / "method_summary.md"
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


def extract_motif(args: argparse.Namespace) -> Dict[str, Any]:
    result = extract_motif_graphs(
        args.workdir,
        limit=args.limit,
        resume=not args.restart,
        debug_samples=args.debug_samples,
    )
    copy_method_summary(args.workdir)
    return result


def build_compact(args: argparse.Namespace) -> Dict[str, Any]:
    result = build_compact_dataset_from_steps(
        step_root=args.step_root,
        workdir=args.workdir,
        source=args.source,
        limit=args.limit,
        max_faces=args.max_faces,
        num_workers=args.num_workers,
        task_timeout_sec=args.task_timeout_sec,
        resume=not args.restart,
        split_file=args.split_file,
    )
    copy_method_summary(args.workdir)
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


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="创新点一 v3：无语义 B-Rep 弱结构基元图抽取")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["build_compact", "parse_abc", "extract_motif", "visualize_motif"],
    )
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR)
    parser.add_argument("--step_root", default=DEFAULT_STEP_ROOT)
    parser.add_argument("--split_file", default=DEFAULT_SPLIT_FILE, help="DTG 6-bit train/val/test 划分；第二步严格继承，不重新划分。")
    parser.add_argument("--source", default="deepcad", choices=["abc", "deepcad", "self"])
    parser.add_argument("--limit", type=int, default=0, help="调试用 STEP 数量上限；0 表示扫描全部文件。")
    parser.add_argument("--max_faces", type=int, default=30, help="与 DTG DeepCAD 拓扑上限一致；创新点1和2全流程默认只保留不超过30面的零件。")
    parser.add_argument("--num_workers", type=int, default=4, help="STEP 并行解析进程数；正式 DeepCAD 构建默认使用 4。")
    parser.add_argument("--task_timeout_sec", type=int, default=900, help="单个 STEP 解析超时秒数；超时样本记入 rejected_manifest，默认 900。")
    parser.add_argument("--vis_count", type=int, default=12)
    parser.add_argument("--uids", default="", help="visualize_motif 指定样本 uid，多个 uid 用英文逗号分隔。")
    parser.add_argument("--visualize_all_clean", action="store_true", help="可视化全部 clean 图；默认只可视化 motif-ready 子集。")
    parser.add_argument("--debug_visualization", action="store_true", help="调试图显示 support/topology-support 关系；默认论文图只显示 structural relations。")
    parser.add_argument("--debug_samples", type=int, default=0, help="诊断时为前 N 个新样本保存完整四层证据；默认不保存。")
    parser.add_argument("--restart", action="store_true", help="删除紧凑数据集后重新构建；默认按 UID 和 STEP 相对路径断点续跑。")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    ensure_workdir(args.workdir)

    if args.mode == "build_compact":
        result = build_compact(args)
        print(
            "[build_compact] "
            f"总记录={result.get('records', 0)}, 本次新增={result.get('new_records', 0)}, "
            f"本次拒绝={result.get('rejected_in_current_run', 0)}, 数据集={result.get('dataset', '')}"
        )
    elif args.mode == "parse_abc":
        result = parse_abc(args)
        print(
            "[parse_abc] "
            f"clean样本={len(result.get('records', []))}, rejected样本={len(result.get('rejected', []))}, "
            f"清单={result.get('clean_manifest')}"
        )
    elif args.mode == "extract_motif":
        result = extract_motif(args)
        print(
            "[extract_motif] "
            f"总记录={result.get('records', 0)}, 本次新增={result.get('new_records', 0)}, "
            f"本次失败={result.get('failures_in_current_run', 0)}, 数据集={result.get('dataset', '')}"
        )
    elif args.mode == "visualize_motif":
        result = visualize_motif(args)
        print(
            f"[visualize_motif] 已可视化={result.get('visualized', 0)}, "
            f"paper_vis={result.get('paper_vis', True)}, index={result.get('index_path', '')}"
        )
    else:
        raise ValueError(f"不支持的 mode：{args.mode}")


if __name__ == "__main__":
    main()
