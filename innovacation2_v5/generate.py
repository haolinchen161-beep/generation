"""Generate matched 3D CAD models using PriorFaceBbox with Cross-Attention PriorAllocator.

Implements Directive 9:
Supports strict paired control experiment (prior_gate = 0.0 vs prior_gate = 1.0)
over identical generated topology, EdgeVert completion, initial Bbox noise z_T, and random seeds.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from OCC.Extend.DataExchange import read_step_file, write_step_file, write_stl_file

from inference.generate import get_brep

from .data import StagewiseH5Dataset
from .data_bbox import extract_motif_node_graph
from .dtg_backend import DTGBackend, checkpoint_checksums
from .metrics import shape_validity
from .models_bbox import build_prior_face_bbox_model


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore",
    message=r"To copy construct from a tensor.*",
    category=UserWarning,
)
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent


def _config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _shape_validation(shape) -> Dict[str, Any]:
    result = shape_validity(shape)
    result["watertight_valid"] = bool(result.pop("valid"))
    return result


def _reconstruct_and_export(
    run_dir: Path,
    topology: Dict[str, Any],
    geometry: Dict[str, np.ndarray],
    bbox_scaled: float,
) -> Dict[str, Any]:
    result = {
        "brep_constructed": False,
        "step_written": False,
        "stl_written": False,
        "watertight_valid": False,
    }
    solid = get_brep(
        (
            geometry["face_bbox"] / bbox_scaled,
            geometry["vert_geom"] / bbox_scaled,
            geometry["edge_geom"] / bbox_scaled,
            geometry["face_geom"] / bbox_scaled,
            topology["edgeFace_adj"].detach().cpu().numpy(),
            topology["edgeVert_adj"].detach().cpu().numpy(),
            topology["faceEdge_adj"],
        )
    )
    if solid is False:
        result["failure_stage"] = "brep_reconstruction"
        return result
    result["brep_constructed"] = True
    result.update(_shape_validation(solid))
    step_path = run_dir / "generated.step"
    try:
        write_step_file(solid, str(step_path))
        result["step_written"] = bool(step_path.is_file() and step_path.stat().st_size > 0)
        result["step"] = str(step_path)
        if result["step_written"]:
            reread = read_step_file(str(step_path))
            result.update(_shape_validation(reread))
    except Exception as exc:
        result["step_error"] = "%s: %s" % (type(exc).__name__, exc)
    if not result["step_written"]:
        result["failure_stage"] = "step_export"
        return result
    stl_path = run_dir / "generated.stl"
    try:
        write_stl_file(solid, str(stl_path), linear_deflection=0.001, angular_deflection=0.5)
        result["stl_written"] = bool(stl_path.is_file() and stl_path.stat().st_size > 0)
        if result["stl_written"]:
            result["stl"] = str(stl_path)
    except Exception as exc:
        result["stl_error"] = "%s: %s" % (type(exc).__name__, exc)
    if not result["stl_written"]:
        result["failure_stage"] = "stl_export"
    elif not result["watertight_valid"]:
        result["failure_stage"] = "watertight_validation"
    else:
        result["failure_stage"] = None
    return result


def _prior_plan(dataset: StagewiseH5Dataset, requests: int, seed: int) -> List[int]:
    rng = np.random.RandomState(seed)
    indices = np.arange(len(dataset))
    if requests <= len(indices):
        return rng.choice(indices, size=requests, replace=False).astype(int).tolist()
    return rng.choice(indices, size=requests, replace=True).astype(int).tolist()


def _protocol(method: str, args: argparse.Namespace, config: Dict[str, Any], prior_rows) -> Dict[str, Any]:
    return {
        "schema_version": "innovation2_pilot_protocol_v1",
        "method": method,
        "requests": int(args.requests),
        "base_seed": int(args.seed),
        "seeds": list(range(int(args.seed), int(args.seed) + int(args.requests))),
        "face_edge_samples_per_request": 1,
        "edge_vert_attempts": int(config["generation"]["edge_vert_attempts"]),
        "geometry_candidates": 1,
        "top_up_failures": False,
        "prior_rows": prior_rows,
        "base_checkpoint_sha256": checkpoint_checksums(),
    }


def generate_motif_group(
    args: argparse.Namespace,
    config: Dict[str, Any],
    output_dir: Path,
    dataset: StagewiseH5Dataset,
    prior_rows: List[int],
    prior_gate: float = 1.0,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = _protocol("motif", args, config, prior_rows)
    protocol["prior_gate"] = float(prior_gate)
    summary_path = output_dir / "batch_summary.json"
    records: List[Dict[str, Any]] = []
    if summary_path.exists():
        try:
            with summary_path.open("r", encoding="utf-8", errors="ignore") as handle:
                previous = json.load(handle)
            records = list(previous.get("records", []))
            completed_indices = {int(record["sample_index"]) for record in records}
            num_done = len([idx for idx in range(args.requests) if idx in completed_indices])
            if num_done > 0:
                print("Notice: Resuming Motif model generation from sample %d/%d..." % (num_done, args.requests), flush=True)
        except Exception:
            pass

    backend = DTGBackend("cuda")
    face_edge_model = backend.load_face_edge()

    base_bbox_model = backend.load_face_bbox()
    prior_bbox_model = build_prior_face_bbox_model(base_bbox_model).to(backend.device)
    ckpt_path = PACKAGE_ROOT / "checkpoints" / "prior_face_bbox" / "best.pt"

    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=backend.device)
        if "prior_allocator_state_dict" in checkpoint:
            prior_bbox_model.prior_allocator.load_state_dict(checkpoint["prior_allocator_state_dict"])
            print(f"  [SUCCESS] Loaded Trained PriorAllocator Checkpoint from: {ckpt_path}")
        elif "conditioner_state_dict" in checkpoint:
            print(f"  [NOTICE] Legacy checkpoint found at {ckpt_path}.")

    prior_bbox_model.eval()
    completed = {int(record["sample_index"]) for record in records}

    def save_summary() -> Dict[str, Any]:
        summary = {
            "schema_version": "innovation2_generation_batch_v1",
            "method": "motif",
            "protocol": protocol,
            "requested": int(args.requests),
            "completed": len(records),
            "face_edge_successes": sum(row.get("face_edge_success", False) for row in records),
            "edge_vert_successes": sum(row.get("edge_vert_success", False) for row in records),
            "geometry_successes": sum(row.get("geometry_success", False) for row in records),
            "step_successes": sum(row.get("step_written", False) for row in records),
            "stl_successes": sum(row.get("stl_written", False) for row in records),
            "watertight_valid": sum(row.get("watertight_valid", False) for row in records),
            "records": sorted(records, key=lambda row: int(row["sample_index"])),
        }
        _write_json_atomic(summary_path, summary)
        return summary

    summary = save_summary()
    for sample_index in range(int(args.requests)):
        if sample_index in completed:
            continue
        sample_seed = int(args.seed) + sample_index
        _seed(sample_seed)
        run_dir = output_dir / ("%04d_seed_%d" % (sample_index, sample_seed))
        run_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        stage_started = started
        timing: Dict[str, float] = {}
        record: Dict[str, Any] = {
            "sample_index": sample_index,
            "seed": sample_seed,
            "method": "motif",
            "prior_gate": float(prior_gate),
            "run_dir": str(run_dir),
            "face_edge_success": False,
            "edge_vert_success": False,
            "geometry_success": False,
            "step_written": False,
            "stl_written": False,
            "watertight_valid": False,
        }
        try:
            item = dataset[int(prior_rows[sample_index])]
            prior = item["prior"]
            record.update(
                {
                    "prior_uid": item["uid"],
                    "prior_dataset_index": int(prior_rows[sample_index]),
                    "prior_hdf5_row": int(item["row"]),
                    "prior_num_faces": int(item["num_faces"]),
                }
            )

            # Stage 1 & 2 Topology Generation (DTG Official Weights)
            fef = backend.sample_fef_baseline(face_edge_model)
            record["face_edge_success"] = True
            now = time.perf_counter()
            timing["face_edge_seconds"] = now - stage_started
            stage_started = now
            record["generated_num_faces"] = int(fef.shape[0])
            record["generated_shared_edges"] = int(fef.sum() // 2)
            np.save(str(run_dir / "fef.npy"), fef)

            topology = backend.complete_edge_vertex(
                fef, attempts=int(config["generation"]["edge_vert_attempts"])
            )
            record["edge_vert_success"] = True
            now = time.perf_counter()
            timing["edge_vert_seconds"] = now - stage_started
            stage_started = now
            record["num_edges"] = int(topology["edgeFace_adj"].shape[0])
            record["num_vertices"] = int(topology["edgeVert_adj"].max().item() + 1)

            # Prepare per_face_prior via Cross-Attention Allocator
            if prior_gate > 0.0:
                batch_prior = {
                    key: (val.unsqueeze(0) if val.ndim > 0 else val.reshape(1)).to(backend.device)
                    for key, val in prior.items()
                }
                node_graph = extract_motif_node_graph(batch_prior, max_nodes=15)
                fef_tensor = torch.as_tensor(fef, dtype=torch.long, device=backend.device).unsqueeze(0)
                gen_mask = torch.ones((1, fef.shape[0]), dtype=torch.bool, device=backend.device)

                with torch.no_grad():
                    per_face_prior, _ = prior_bbox_model.prior_allocator(
                        node_graph["node_features"],
                        node_graph["hosted_adj"],
                        node_graph["thin_wall_adj"],
                        node_graph["node_mask"],
                        fef_tensor,
                        gen_mask,
                    )
                    per_face_prior = float(prior_gate) * per_face_prior
                    prior_bbox_model.set_prior(per_face_prior)

            # Stage 3 FaceBbox + Stage 4, 5, 6 Geometry Generation
            geometry = backend.generate_geometry(
                topology,
                prior=prior,
                prior_bbox_model=prior_bbox_model if prior_gate > 0.0 else None,
            )
            prior_bbox_model.set_prior(None)

            record["geometry_success"] = True
            now = time.perf_counter()
            timing["geometry_diffusion_seconds"] = now - stage_started
            stage_started = now
            np.savez_compressed(str(run_dir / "geometry_arrays.npz"), **geometry)

            bbox_scaled = float(config.get("data", {}).get("bbox_scaled", 3.0))
            record.update(
                _reconstruct_and_export(
                    run_dir,
                    topology,
                    geometry,
                    bbox_scaled,
                )
            )
            timing["reconstruction_and_export_seconds"] = time.perf_counter() - stage_started
        except Exception as exc:
            prior_bbox_model.set_prior(None)
            record["error"] = "%s: %s" % (type(exc).__name__, exc)
            if not record["face_edge_success"]:
                record["failure_stage"] = "face_edge"
            elif not record["edge_vert_success"]:
                record["failure_stage"] = "edge_vert"
            elif not record["geometry_success"]:
                record["failure_stage"] = "geometry_diffusion"
        record["elapsed_seconds"] = time.perf_counter() - started
        timing["total_seconds"] = record["elapsed_seconds"]
        record["timing"] = timing
        _write_json_atomic(run_dir / "trace.json", record)
        records.append(record)
        summary = save_summary()
        print(
            "MOTIF Sample %d/%d (Seed: %d, Gate: %.1f) | Stage: %s | Elapsed: %.2fs"
            % (sample_index + 1, args.requests, sample_seed, prior_gate, record.get("failure_stage") or "PASS", record["elapsed_seconds"]),
            flush=True,
        )
    backend.release(face_edge_model)
    return summary


def run_pilot(args: argparse.Namespace) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CAD generation")
    config = _config(args.config)
    if args.requests is None:
        args.requests = int(config["generation"]["requests"])
    if args.seed is None:
        args.seed = int(config["generation"]["seed"])
    output_root = args.output_dir or _resolve(config["paths"]["output_dir"])

    val_h5 = PACKAGE_ROOT / "data" / "deepcad" / "validation.h5"
    prior_dataset = StagewiseH5Dataset(val_h5, "face_edge", training=False)
    rows = _prior_plan(prior_dataset, int(args.requests), int(args.seed))

    print("=" * 80)
    print("RUNNING CROSS-ATTENTION PRIOR ALLOCATOR GENERATION (Requests: %d, Seed: %d)" % (args.requests, args.seed))
    print("=" * 80)

    motif = generate_motif_group(
        args,
        config,
        output_root / "motif",
        prior_dataset,
        rows,
        prior_gate=getattr(args, "prior_gate", 1.0),
    )

    print("\n" + "=" * 80)
    print("CAD MODEL GENERATION COMPLETE!")
    print("=" * 80)
    print("Watertight Valid Rate: %d / %d (%.1f%%)" % (
        motif["watertight_valid"], args.requests, motif["watertight_valid"] / max(1, args.requests) * 100
    ))
    print("3D STEP Files Saved To: %s" % (output_root / "motif"))
    print("=" * 80)
    return motif


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="generate CAD models")
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "config.yaml")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--prior-gate", type=float, default=1.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.pilot:
        print("Usage: python -m innovation2_z.generate --pilot [--requests 20] [--prior-gate 1.0]")
        return 0
    run_pilot(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
