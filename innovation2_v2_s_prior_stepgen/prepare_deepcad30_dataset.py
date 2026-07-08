"""准备 DeepCAD-30 S-ready 数据子集。

本脚本只读取 innovation1_v3 的 DeepCAD 输出，并把创新点二 v2 需要的最小数据
复制到当前新目录内。它不会修改 DTG 原源码，也不会修改 innovation1_v3 的输出。
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


NODE_TYPES = [
    "sheet_like_group",
    "thin_wall_pair",
    "loop_or_hole",
    "transition_group",
    "repeated_feature",
    "boundary_group",
]

RELATION_TYPES = [
    "parallel_to",
    "opposite_to",
    "orthogonal_to",
    "coplanar_with",
    "repeated_with",
    "bounded_by",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_clean_manifest(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["uid"]: row for row in csv.DictReader(f)}


def pkl_path_for_uid(source_parsed_dir: Path, manifest_row: Dict[str, str], uid: str) -> Path:
    local = source_parsed_dir / f"{uid}.pkl"
    if local.exists():
        return local
    fallback = manifest_row.get("pkl_path", "")
    if fallback:
        return Path(fallback)
    return local


def max_edges_per_face(parsed: Dict[str, Any]) -> int:
    face_edge_adj = parsed.get("faceEdge_adj", [])
    return max([len(item) for item in face_edge_adj] or [0])


def sorted_float_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0, "mean": 0, "p95": 0, "p99": 0, "max": 0}
    values = sorted(values)

    def q(p: float) -> float:
        idx = min(len(values) - 1, int(round((len(values) - 1) * p)))
        return float(values[idx])

    return {
        "min": float(values[0]),
        "mean": float(sum(values) / len(values)),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": float(values[-1]),
    }


def split_uids(uids: List[str], seed: int) -> Dict[str, List[str]]:
    shuffled = list(uids)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * 0.9)
    n_val = int(n * 0.05)
    return {
        "train": sorted(shuffled[:n_train]),
        "val": sorted(shuffled[n_train : n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val :]),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare(args: argparse.Namespace) -> Dict[str, Any]:
    root = repo_root()
    workdir = Path(__file__).resolve().parent
    source_root = root / "innovation1_v3_brep_motif_graph" / "output_deepcad" / "outputs"
    source_parsed_dir = source_root / "parsed"
    source_prior_jsonl = source_root / "motif_graphs" / "motif_prior_index_ready.jsonl"
    source_graph_jsonl = source_root / "motif_graphs" / "motif_graph_index_ready.jsonl"
    source_manifest = source_root / "reports" / "clean_manifest.csv"

    dataset_dir = ensure_dir(workdir / "data" / args.dataset_name)
    parsed_out = ensure_dir(dataset_dir / "parsed")
    split_dir = ensure_dir(dataset_dir / "splits")
    reports_dir = ensure_dir(dataset_dir / "reports")

    manifest = read_clean_manifest(source_manifest)
    selected_prior_rows: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []
    selected_uids = set()
    rejected_counter = Counter()
    stats_values: Dict[str, List[float]] = {
        "num_faces": [],
        "num_edges": [],
        "num_vertices": [],
        "num_s_nodes": [],
        "num_s_relations": [],
        "max_edges_per_face": [],
    }
    node_counter = Counter()
    relation_counter = Counter()

    scanned = 0
    for row in read_jsonl(source_prior_jsonl):
        scanned += 1
        uid = row["uid"]
        if int(row.get("num_faces", 999999)) > args.max_faces:
            rejected_counter["face_count_over_limit"] += 1
            continue
        manifest_row = manifest.get(uid)
        if not manifest_row:
            rejected_counter["missing_clean_manifest"] += 1
            continue
        pkl_path = pkl_path_for_uid(source_parsed_dir, manifest_row, uid)
        if not pkl_path.exists():
            rejected_counter["missing_pkl"] += 1
            continue
        try:
            with pkl_path.open("rb") as f:
                parsed = pickle.load(f)
        except Exception:
            rejected_counter["bad_pkl"] += 1
            continue
        max_epf = max_edges_per_face(parsed)
        if max_epf > args.max_edges_per_face:
            rejected_counter["max_edges_per_face_over_limit"] += 1
            continue

        dst_pkl = parsed_out / f"{uid}.pkl"
        if args.copy_pkl and (args.overwrite or not dst_pkl.exists()):
            shutil.copy2(pkl_path, dst_pkl)

        selected_uids.add(uid)
        selected_prior_rows.append(row)

        motif_nodes = row.get("motif_nodes", [])
        motif_relations = row.get("motif_relations", [])
        node_counter.update([item.get("type", "unknown") for item in motif_nodes])
        relation_counter.update([item.get("type", "unknown") for item in motif_relations])

        stats_values["num_faces"].append(float(row.get("num_faces", 0)))
        stats_values["num_edges"].append(float(row.get("num_edges", 0)))
        stats_values["num_vertices"].append(float(row.get("num_vertices", 0)))
        stats_values["num_s_nodes"].append(float(len(motif_nodes)))
        stats_values["num_s_relations"].append(float(len(motif_relations)))
        stats_values["max_edges_per_face"].append(float(max_epf))

        manifest_rows.append(
            {
                "uid": uid,
                "source": row.get("source", "deepcad"),
                "local_pkl": str((Path("data") / args.dataset_name / "parsed" / f"{uid}.pkl").as_posix()),
                "source_step_path": manifest_row.get("step_path", ""),
                "source_pkl_path": manifest_row.get("pkl_path", ""),
                "num_faces": row.get("num_faces", ""),
                "num_edges": row.get("num_edges", ""),
                "num_vertices": row.get("num_vertices", ""),
                "num_s_nodes": len(motif_nodes),
                "num_s_relations": len(motif_relations),
                "max_edges_per_face": max_epf,
                "parser_backend": row.get("parser_backend", ""),
                "geometry_sampling_quality": row.get("geometry_sampling_quality", ""),
                "dtg_train_compatible": row.get("dtg_train_compatible", ""),
                "dtg_filter_reason": row.get("dtg_filter_reason", ""),
            }
        )

        if len(selected_prior_rows) % 500 == 0:
            print(f"已选择 {len(selected_prior_rows)} 个样本...")

    selected_uids_sorted = sorted(selected_uids)
    graph_rows = [row for row in read_jsonl(source_graph_jsonl) if row.get("uid") in selected_uids]
    splits = split_uids(selected_uids_sorted, args.seed)

    write_jsonl(dataset_dir / "motif_prior_index_ready.jsonl", selected_prior_rows)
    write_jsonl(dataset_dir / "motif_graph_index_ready.jsonl", graph_rows)
    write_csv(
        dataset_dir / "manifest.csv",
        sorted(manifest_rows, key=lambda item: item["uid"]),
        [
            "uid",
            "source",
            "local_pkl",
            "source_step_path",
            "source_pkl_path",
            "num_faces",
            "num_edges",
            "num_vertices",
            "num_s_nodes",
            "num_s_relations",
            "max_edges_per_face",
            "parser_backend",
            "geometry_sampling_quality",
            "dtg_train_compatible",
            "dtg_filter_reason",
        ],
    )

    for name, uids in splits.items():
        with (split_dir / f"{name}_uids.txt").open("w", encoding="utf-8") as f:
            f.write("\n".join(uids) + "\n")

    split_paths = {
        name: [str((workdir / "data" / args.dataset_name / "parsed" / f"{uid}.pkl").resolve()) for uid in uids]
        for name, uids in splits.items()
    }
    with (split_dir / "deepcad30_local_paths.pkl").open("wb") as f:
        pickle.dump(split_paths, f)
    with (split_dir / "deepcad30_uids_split.json").open("w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)

    summary = {
        "dataset_name": args.dataset_name,
        "source_prior_jsonl": str(source_prior_jsonl.as_posix()),
        "source_graph_jsonl": str(source_graph_jsonl.as_posix()),
        "source_clean_manifest": str(source_manifest.as_posix()),
        "policy": {
            "purpose": "先跑通无条件 S-prior -> DTG backbone -> STEP 生成流程",
            "max_faces": args.max_faces,
            "max_edges_per_face": args.max_edges_per_face,
            "copy_pkl": args.copy_pkl,
            "ablation": "暂不进行消融对照",
        },
        "scanned_prior_ready_samples": scanned,
        "selected_samples": len(selected_uids_sorted),
        "rejected": dict(rejected_counter),
        "split_counts": {name: len(uids) for name, uids in splits.items()},
        "stats": {key: sorted_float_stats(values) for key, values in stats_values.items()},
        "node_type_counts": {key: int(node_counter.get(key, 0)) for key in NODE_TYPES},
        "relation_type_counts": {key: int(relation_counter.get(key, 0)) for key in RELATION_TYPES},
        "dtg_backbone_compatible_config": {
            "max_face": 30,
            "max_edge": 20,
            "edge_classes": 5,
            "max_num_edge": 274,
            "max_vert": 186,
            "max_vertFace": 12,
            "max_seq_length": 280,
            "max_num_edge_topo": 108,
            "checkpoint_hint": "checkpoints_base/deepcad",
        },
    }
    with (dataset_dir / "dataset_stats.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    report_lines = [
        "创新点2 v2 DeepCAD-30 S-ready 数据准备报告",
        "=" * 72,
        f"源 prior-ready 样本数：{scanned}",
        f"最终选择样本数：{len(selected_uids_sorted)}",
        f"筛选条件：face_count <= {args.max_faces}, max_edges_per_face <= {args.max_edges_per_face}",
        f"训练/验证/测试划分：{summary['split_counts']}",
        "",
        "拒绝原因：",
    ]
    for key, value in sorted(rejected_counter.items()):
        report_lines.append(f"  - {key}: {value}")
    report_lines += [
        "",
        "核心统计：",
    ]
    for key, value in summary["stats"].items():
        report_lines.append(f"  - {key}: {value}")
    report_lines += [
        "",
        "说明：",
        "  - 当前阶段不做消融对照，目标是先跑通无条件 S 采样、DTG backbone 调用和 STEP 输出。",
        "  - 本目录是独立工作区；外部 DTG 源码、innovation1_v3 输出均未被修改。",
        "  - copied parsed pkl 是后续训练 S adapter 和调用几何/拓扑目标的本地数据入口。",
    ]
    report_text = "\n".join(report_lines) + "\n"
    (reports_dir / "prepare_dataset_report.txt").write_text(report_text, encoding="utf-8")
    print(report_text)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备创新点2 v2 的 DeepCAD-30 S-ready 本地数据集")
    parser.add_argument("--dataset_name", default="deepcad30_s_ready")
    parser.add_argument("--max_faces", type=int, default=30)
    parser.add_argument("--max_edges_per_face", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260708)
    copy_group = parser.add_mutually_exclusive_group()
    copy_group.add_argument("--copy_pkl", dest="copy_pkl", action="store_true", default=True)
    copy_group.add_argument("--no_copy_pkl", dest="copy_pkl", action="store_false")
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
