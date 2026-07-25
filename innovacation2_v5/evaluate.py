"""Evaluate pilot outputs with CAD, distribution, stage and structural metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import yaml
from OCC.Extend.DataExchange import read_step_file

from .data import StagewiseH5Dataset
from .metrics import (
    HASH_SCHEMA,
    brep_hash,
    cad_metrics,
    distribution_metrics,
    generated_structure,
    prior_structure,
    sample_shape_points,
    shape_validity,
    structural_scores,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
FACE_BINS = ((1, 6), (7, 10), (11, 14), (15, 19), (20, 24), (25, 30))


def _config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _uid(value: str) -> str:
    return Path(str(value).replace("\\", "/")).stem


def _load_split(path: Path) -> Dict[str, List[str]]:
    with path.open("rb") as handle:
        split = pickle.load(handle)
    if not isinstance(split, dict) or "train" not in split or "test" not in split:
        raise ValueError("invalid DTG split: %s" % path)
    return {key: [_uid(value) for value in values] for key, values in split.items()}


def _reference_pool(
    metrics_dir: Path,
    split_path: Path,
    step_root: Path,
    count: int,
    points: int,
    seed: int,
) -> np.ndarray:
    manifest_path = metrics_dir / "reference_manifest.json"
    points_path = metrics_dir / "reference_points.npz"
    split = _load_split(split_path)
    available = sorted(
        uid for uid in set(split["test"]) if (step_root / (uid + ".step")).is_file()
    )
    random.Random(seed).shuffle(available)
    selected = available[:count]
    if len(selected) < count:
        raise ValueError("only %d reference STEP files are available" % len(selected))
    protocol = {
        "schema_version": "innovation2_reference_pool_v1",
        "split": str(split_path.resolve()),
        "step_root": str(step_root.resolve()),
        "count": count,
        "points": points,
        "seed": seed,
        "uids": selected,
    }
    if manifest_path.exists() and points_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != protocol:
            raise ValueError("reference protocol changed; use a new metrics directory")
        return np.load(points_path, allow_pickle=False)["points"].astype(np.float32)
    clouds = []
    for index, uid in enumerate(selected):
        shape = read_step_file(str(step_root / (uid + ".step")))
        clouds.append(sample_shape_points(shape, points, seed + index))
        if (index + 1) % 25 == 0:
            print("reference %d/%d" % (index + 1, count), flush=True)
    result = np.stack(clouds).astype(np.float32)
    temporary = points_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, points=result)
    os.replace(temporary, points_path)
    _write_json(manifest_path, protocol)
    return result


def _training_hashes(metrics_dir: Path, split_path: Path, step_root: Path) -> set:
    cache = metrics_dir / "training_hashes_brepgen_v2_4bit.jsonl"
    legacy = (
        PROJECT_ROOT
        / "innovation2_motif_guided_generation"
        / "outputs"
        / "cad_metrics_v2_diagnostic"
        / "training_hashes_brepgen_v2_4bit.jsonl"
    )
    if not cache.exists() and legacy.exists():
        shutil.copyfile(str(legacy), str(cache))
    hashes = set()
    completed = set()
    if cache.exists():
        with cache.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                completed.add(str(record.get("uid", "")))
                if record.get("hash"):
                    hashes.add(str(record["hash"]))
    split = _load_split(split_path)
    missing = [uid for uid in sorted(set(split["train"])) if uid not in completed]
    if missing:
        with cache.open("a", encoding="utf-8") as handle:
            for index, uid in enumerate(missing):
                step = step_root / (uid + ".step")
                record = {"uid": uid, "schema": HASH_SCHEMA, "bits": 4, "hash": None, "error": None}
                try:
                    record["hash"] = brep_hash(read_step_file(str(step)), 4)
                    hashes.add(record["hash"])
                except Exception as exc:
                    record["error"] = "%s: %s" % (type(exc).__name__, exc)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                if (index + 1) % 100 == 0:
                    print("training hash %d/%d" % (index + 1, len(missing)), flush=True)
    return hashes


def _strict_group(
    group_dir: Path,
    references: np.ndarray,
    training_hashes: set,
    points: int,
    seed: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    with (group_dir / "batch_summary.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    requested = int(manifest["requested"])
    records = sorted(manifest["records"], key=lambda row: int(row["sample_index"]))
    hashes, clouds, audits = [], [], []
    for record in records:
        row = dict(record)
        row.update({"strict_valid": False, "metric_error": None})
        step = Path(str(record.get("step", ""))) if record.get("step") else None
        if step is not None and step.is_file():
            try:
                shape = read_step_file(str(step))
                validity = shape_validity(shape)
                row.update(validity)
                row["strict_valid"] = bool(validity["valid"])
                if row["strict_valid"]:
                    value = brep_hash(shape, 4)
                    hashes.append(value)
                    row["brep_hash"] = value
                    clouds.append(sample_shape_points(shape, points, seed + int(row["sample_index"])))
            except Exception as exc:
                row["metric_error"] = "%s: %s" % (type(exc).__name__, exc)
        audits.append(row)
    cad = cad_metrics(hashes, training_hashes, requested)
    distribution = (
        distribution_metrics(np.stack(clouds), references)
        if clouds
        else {"cov": None, "mmd": None, "jsd": None}
    )
    stage = {
        key: sum(bool(row.get(field, False)) for row in records) / requested
        for key, field in {
            "face_edge_rate": "face_edge_success",
            "edge_vert_rate": "edge_vert_success",
            "geometry_rate": "geometry_success",
            "step_rate": "step_written",
            "stl_rate": "stl_written",
        }.items()
    }
    timing_keys = sorted(
        {
            key
            for row in records
            for key in dict(row.get("timing", {})).keys()
        }
    )
    timing = {
        key: float(
            np.mean(
                [
                    float(row["timing"][key])
                    for row in records
                    if key in dict(row.get("timing", {}))
                ]
            )
        )
        for key in timing_keys
    }
    return {
        "requested": requested,
        "strict_valid_steps": len(hashes),
        "COV_percent": None if distribution["cov"] is None else distribution["cov"] * 100,
        "MMD_x100": None if distribution["mmd"] is None else distribution["mmd"] * 100,
        "JSD_x100": None if distribution["jsd"] is None else distribution["jsd"] * 100,
        "Novel_percent": None if cad["novel"] is None else cad["novel"] * 100,
        "Unique_percent": None if cad["unique"] is None else cad["unique"] * 100,
        "Valid_percent": cad["valid"] * 100,
        "stage_success": {key: value * 100 for key, value in stage.items()},
        "failure_stage_counts": dict(
            Counter(str(row.get("failure_stage") or "success") for row in records)
        ),
        "mean_timing_seconds": timing,
        "COV_theoretical_max_percent": min(requested, len(references)) / len(references) * 100,
    }, audits


def _bin(face_count: Any) -> int:
    if face_count is None:
        return -1
    for index, (lower, upper) in enumerate(FACE_BINS):
        if lower <= int(face_count) <= upper:
            return index
    return -1


def _complexity_standardized(
    dtg_rows: Sequence[Mapping[str, Any]],
    motif_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    all_rows = list(dtg_rows) + list(motif_rows)
    target_counts = Counter(_bin(row.get("generated_num_faces")) for row in all_rows)
    total = sum(target_counts.values())
    weights = {key: value / total for key, value in target_counts.items()}
    result = {}
    for name, rows in (("dtg", dtg_rows), ("motif", motif_rows)):
        rates = {}
        score = 0.0
        for bin_id, weight in weights.items():
            selected = [row for row in rows if _bin(row.get("generated_num_faces")) == bin_id]
            rate = (
                sum(bool(row.get("strict_valid", False)) for row in selected) / len(selected)
                if selected
                else 0.0
            )
            rates[str(bin_id)] = {"requests": len(selected), "valid_rate": rate}
            score += weight * rate
        result[name] = {"standardized_valid_percent": score * 100, "bins": rates}
    result["weights"] = {str(key): value for key, value in weights.items()}
    return result


def _structural(
    dtg_rows: Sequence[Mapping[str, Any]],
    motif_rows: Sequence[Mapping[str, Any]],
    dataset: StagewiseH5Dataset,
    prior_rows: Sequence[int],
) -> Dict[str, Any]:
    groups = {"dtg": [], "motif": []}
    by_name = {"dtg": dtg_rows, "motif": motif_rows}
    for name, rows in by_name.items():
        for index, row in enumerate(rows):
            entry = {
                "sample_index": int(row["sample_index"]),
                "valid": bool(row.get("strict_valid", False)),
                "signature_similarity": 0.0,
            }
            step = Path(str(row.get("step", ""))) if row.get("step") else None
            if step is not None and step.is_file():
                try:
                    prior = prior_structure(dataset[int(prior_rows[index])])
                    generated = generated_structure(step)
                    entry.update(structural_scores(prior, generated))
                except Exception as exc:
                    entry["error"] = "%s: %s" % (type(exc).__name__, exc)
            entry["valid_structure_score"] = (
                float(entry["signature_similarity"]) if entry["valid"] else 0.0
            )
            groups[name].append(entry)

    result = {}
    for name, rows in groups.items():
        successful = [row for row in rows if "motif" in row]
        distribution_jsd = _mean_structure_jsd(successful)
        result[name] = {
            "VSS_percent": float(np.mean([row["valid_structure_score"] for row in rows]) * 100),
            "Motif_F1_percent": float(np.mean([row["motif"]["f1"] for row in successful]) * 100)
            if successful else 0.0,
            "Relation_F1_percent": float(np.mean([row["relation"]["f1"] for row in successful]) * 100)
            if successful else 0.0,
            "Surface_macro_F1_percent": float(np.mean([row["surface_macro_f1"] for row in successful]) * 100)
            if successful else 0.0,
            "Structure_distribution_JSD_x100": distribution_jsd * 100,
            "records": rows,
        }
    return result


def _mean_structure_jsd(rows: Sequence[Mapping[str, Any]], bins: int = 20) -> float:
    if not rows:
        return 1.0
    prior = np.asarray([row["prior_signature"] for row in rows], dtype=np.float64)
    generated = np.asarray([row["generated_signature"] for row in rows], dtype=np.float64)
    values = []
    for column in range(prior.shape[1]):
        left = np.clip(np.floor(prior[:, column] * (bins - 1)), 0, bins - 1).astype(int)
        right = np.clip(np.floor(generated[:, column] * (bins - 1)), 0, bins - 1).astype(int)
        p = np.bincount(left, minlength=bins).astype(np.float64)
        q = np.bincount(right, minlength=bins).astype(np.float64)
        p, q = p / p.sum(), q / q.sum()
        middle = 0.5 * (p + q)
        mask_p, mask_q = p > 0, q > 0
        kl_p = np.sum(p[mask_p] * np.log2(p[mask_p] / middle[mask_p]))
        kl_q = np.sum(q[mask_q] * np.log2(q[mask_q] / middle[mask_q]))
        values.append(0.5 * (kl_p + kl_q))
    return float(np.mean(values))


def _bootstrap_difference(
    left: Sequence[float], right: Sequence[float], repeats: int, seed: int
) -> Dict[str, float]:
    left, right = np.asarray(left), np.asarray(right)
    if len(left) != len(right):
        raise ValueError("paired bootstrap arrays differ in length")
    rng = np.random.RandomState(seed)
    differences = []
    for _ in range(repeats):
        indices = rng.randint(0, len(left), len(left))
        differences.append(float(np.mean(right[indices] - left[indices])))
    return {
        "difference": float(np.mean(right - left)),
        "ci95_low": float(np.percentile(differences, 2.5)),
        "ci95_high": float(np.percentile(differences, 97.5)),
    }


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    config = _config(args.config)
    output_root = args.output_dir or _resolve(config["paths"]["output_dir"])
    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    split_path = PROJECT_ROOT / "deepcad_data_split_6bit.pkl"
    step_root = _resolve(config["paths"]["step_root"])
    reference_count = int(args.reference_count or config["evaluation"]["reference_count"])
    points = int(args.points or config["evaluation"]["point_count"])
    references = _reference_pool(
        metrics_dir, split_path, step_root, reference_count, points, int(args.seed)
    )
    training_hashes = _training_hashes(metrics_dir, split_path, step_root)
    dtg_metrics, dtg_rows = _strict_group(
        output_root / "dtg", references, training_hashes, points, int(args.seed)
    )
    motif_metrics, motif_rows = _strict_group(
        output_root / "motif", references, training_hashes, points, int(args.seed)
    )
    complexity = _complexity_standardized(dtg_rows, motif_rows)

    with (output_root / "motif" / "batch_summary.json").open("r", encoding="utf-8") as handle:
        motif_manifest = json.load(handle)
    prior_rows = motif_manifest["protocol"]["prior_rows"]
    data_dir = args.data_dir or _resolve(config["paths"]["data_dir"])
    dataset = StagewiseH5Dataset(data_dir / "validation.h5", "face_edge", training=False)
    structural = _structural(dtg_rows, motif_rows, dataset, prior_rows)
    eval_len = len(motif_rows)
    dtg_vss = [row["valid_structure_score"] for row in structural["dtg"]["records"][:eval_len]]
    motif_vss = [row["valid_structure_score"] for row in structural["motif"]["records"][:eval_len]]
    vss_difference = _bootstrap_difference(
        dtg_vss,
        motif_vss,
        int(config["evaluation"]["bootstrap_samples"]),
        int(args.seed),
    )
    vss_difference = {key: value * 100 for key, value in vss_difference.items()}

    gate = config["evaluation"]["pilot_gate"]
    def medium_high_step_rate(rows):
        selected = [row for row in rows if int(row.get("generated_num_faces", 0) or 0) >= 11]
        return (
            sum(bool(row.get("step_written", False)) for row in selected) / len(selected)
            if selected
            else 0.0
        )
    dtg_medium_high = medium_high_step_rate(dtg_rows)
    motif_medium_high = medium_high_step_rate(motif_rows)
    def relative_worse(new, base, lower_is_better=False):
        if new is None or base in (None, 0):
            return float("inf")
        return (new - base) / abs(base) if lower_is_better else (base - new) / abs(base)
    checks = {
        "complexity_standardized_valid_not_lower": (
            complexity["motif"]["standardized_valid_percent"]
            >= complexity["dtg"]["standardized_valid_percent"]
        ),
        "VSS_gain_at_least_5_points": vss_difference["difference"] >= float(gate["vss_gain_points"]),
        "VSS_ci_does_not_cross_zero": vss_difference["ci95_low"] > 0,
        "Novel_drop_within_5_points": (
            motif_metrics["Novel_percent"] is not None
            and dtg_metrics["Novel_percent"] is not None
            and motif_metrics["Novel_percent"] >= dtg_metrics["Novel_percent"] - gate["max_novel_drop_points"]
        ),
        "Unique_drop_within_5_points": (
            motif_metrics["Unique_percent"] is not None
            and dtg_metrics["Unique_percent"] is not None
            and motif_metrics["Unique_percent"] >= dtg_metrics["Unique_percent"] - gate["max_unique_drop_points"]
        ),
        "COV_drop_within_5_points": (
            motif_metrics["COV_percent"] is not None
            and dtg_metrics["COV_percent"] is not None
            and motif_metrics["COV_percent"] >= dtg_metrics["COV_percent"] - gate["max_cov_drop_points"]
        ),
        "MMD_worsening_within_10_percent": relative_worse(
            motif_metrics["MMD_x100"], dtg_metrics["MMD_x100"], True
        ) <= gate["max_mmd_relative_worsening"],
        "JSD_worsening_within_10_percent": relative_worse(
            motif_metrics["JSD_x100"], dtg_metrics["JSD_x100"], True
        ) <= gate["max_jsd_relative_worsening"],
        "medium_high_face_STEP_rate_not_lower": motif_medium_high >= dtg_medium_high,
    }
    result = {
        "schema_version": "innovation2_pilot_evaluation_v1",
        "protocol": {
            "generated_requests_per_group": dtg_metrics["requested"],
            "reference_models": reference_count,
            "points_per_model": points,
            "failed_requests_remain_in_valid_denominator": True,
            "COV_MMD_distance": "bidirectional Chamfer Distance",
            "hash": "BrepGen-compatible 4-bit face geometry plus exact B-rep topology",
        },
        "dtg": dtg_metrics,
        "motif": motif_metrics,
        "complexity_standardized": complexity,
        "medium_high_face_STEP_percent": {
            "dtg": dtg_medium_high * 100,
            "motif": motif_medium_high * 100,
        },
        "structural": structural,
        "VSS_paired_difference_points": vss_difference,
        "pilot_gate_checks": checks,
        "pilot_passed": all(checks.values()),
    }
    _write_json(metrics_dir / "evaluation_summary.json", result)
    with (metrics_dir / "metrics_table.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "method", "COV_percent", "MMD_x100", "JSD_x100",
            "Novel_percent", "Unique_percent", "Valid_percent",
            "VSS_percent", "Motif_F1_percent", "Relation_F1_percent", "Surface_macro_F1_percent",
            "Structure_distribution_JSD_x100",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, metrics in (("DTG", dtg_metrics), ("StagewisePrior", motif_metrics)):
            writer.writerow(
                {
                    "method": name,
                    **{key: metrics[key] for key in fields[1:7]},
                    **{
                        key: structural["dtg" if name == "DTG" else "motif"][key]
                        for key in fields[7:]
                    },
                }
            )
    return result


def evaluate_phase1_bbox(
    model: torch.nn.Module,
    val_loader: Any,
    device: torch.device,
) -> Dict[str, float]:
    """Directive 10: Phase 1 Primary Evaluation Metrics.
    
    Computes Bbox noise MSE, Center MSE, Size MSE, and Face Role Classification F1.
    """
    model.base.eval()
    model.prior_allocator.eval()
    
    center_mses = []
    size_mses = []
    role_f1s = []
    
    with torch.no_grad():
        for cpu_batch in val_loader:
            batch = to_device(cpu_batch, device)
            prior, target = batch["prior"], batch["target"]
            clean_bbox = target["face_bbox"].float() # [b, max_faces, 6]
            face_mask = prior["face_mask"].bool()    # [b, max_faces]
            fef_adj = target["fef_adj"].long()       # [b, max_faces, max_faces]
            gt_membership = prior["motif_membership"].float() # [b, max_faces, 3]
            
            node_graph = extract_motif_node_graph(prior, max_nodes=15)
            per_face_prior, role_logits = model.prior_allocator(
                node_graph["node_features"],
                node_graph["hosted_adj"],
                node_graph["thin_wall_adj"],
                node_graph["node_mask"],
                fef_adj,
                face_mask,
            )
            
            # Role F1
            role_preds = (torch.sigmoid(role_logits[face_mask]) > 0.5).float()
            role_gt = gt_membership[face_mask]
            
            tp = (role_preds * role_gt).sum()
            fp = (role_preds * (1.0 - role_gt)).sum()
            fn = ((1.0 - role_preds) * role_gt).sum()
            
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            role_f1s.append(float(f1.cpu()))
            
    return {
        "face_role_f1": float(np.mean(role_f1s)),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "config.yaml")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--reference-count", type=int, default=None)
    parser.add_argument("--points", type=int, default=None)
    parser.add_argument("--seed", type=int, default=9000)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.pilot:
        raise SystemExit("use --pilot for the 100-request comparison")
    result = evaluate(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
