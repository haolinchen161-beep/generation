"""Generate matched 3D CAD models using PriorFaceBbox (Strict Paired Control Benchmark v6.1).

Implements Core Directives 8, 9, 10:
- MAX_MOTIF_NODES = 96.
- Strict Checkpoint Schema Verification (schema_version == "prior_face_bbox_v6", max_nodes == 96).
- Rejects candidate topologies if face_count_gap > 2 after 10 attempts (no fallback candidate!).
- Per-sample try/except/finally architecture with atomic summary saving for interrupt recovery.
- Maintains 100% paired record symmetry between dtg (prior_gate=0) and motif (prior_gate=1).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from OCC.Extend.DataExchange import read_step_file, write_step_file, write_stl_file

from inference.generate import get_brep

from .data import StagewiseH5Dataset
from .data_bbox import extract_motif_node_graph, MAX_MOTIF_NODES
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


def generate_paired_benchmark(
    args: argparse.Namespace,
    config: Dict[str, Any],
    output_root: Path,
    dataset: StagewiseH5Dataset,
    prior_rows: List[int],
) -> Dict[str, Any]:
    dtg_dir = output_root / "dtg"
    motif_dir = output_root / "motif"
    dtg_dir.mkdir(parents=True, exist_ok=True)
    motif_dir.mkdir(parents=True, exist_ok=True)

    backend = DTGBackend("cuda")
    face_edge_model = backend.load_face_edge()
    base_bbox_model = backend.load_face_bbox()

    # Directive 10: Strict Checkpoint Verification!
    ckpt_path = PACKAGE_ROOT / "checkpoints" / "prior_face_bbox" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Error: Trained PriorAllocator checkpoint NOT found at {ckpt_path}! Train first via python -m innovation2_z.train_prior_bbox.")

    checkpoint = torch.load(ckpt_path, map_location=backend.device)

    if "prior_allocator_state_dict" not in checkpoint:
        raise KeyError(f"Error: Checkpoint at {ckpt_path} does not contain 'prior_allocator_state_dict'!")

    if checkpoint.get("schema_version") != "prior_face_bbox_v6":
        raise ValueError(f"Error: Incompatible checkpoint schema version '{checkpoint.get('schema_version')}'!")

    if int(checkpoint.get("max_nodes", -1)) != MAX_MOTIF_NODES:
        raise ValueError(f"Error: Checkpoint max_nodes ({checkpoint.get('max_nodes')}) mismatch with current MAX_MOTIF_NODES ({MAX_MOTIF_NODES})!")

    prior_bbox_model = build_prior_face_bbox_model(base_bbox_model).to(backend.device)
    prior_bbox_model.prior_allocator.load_state_dict(checkpoint["prior_allocator_state_dict"], strict=True)
    print(f"  [SUCCESS] Loaded Verified PriorAllocator v6.1 Checkpoint from: {ckpt_path}")

    prior_bbox_model.eval()

    dtg_records: List[Dict[str, Any]] = []
    motif_records: List[Dict[str, Any]] = []

    # Restore existing records for resume
    dtg_summary_path = dtg_dir / "batch_summary.json"
    motif_summary_path = motif_dir / "batch_summary.json"

    completed_set = set()
    if dtg_summary_path.exists() and motif_summary_path.exists():
        try:
            with dtg_summary_path.open("r", encoding="utf-8") as h:
                dtg_records = json.load(h).get("records", [])
            with motif_summary_path.open("r", encoding="utf-8") as h:
                motif_records = json.load(h).get("records", [])
            completed_set = {rec["sample_index"] for rec in dtg_records}
        except Exception:
            dtg_records, motif_records, completed_set = [], [], set()

    protocol = {
        "prior_rows": prior_rows,
        "base_seed": int(args.seed),
        "requests": int(args.requests),
        "paired": True,
    }

    def save_summaries():
        _write_json_atomic(dtg_summary_path, {
            "schema_version": "innovation2_generation_batch_v1",
            "method": "dtg",
            "protocol": protocol,
            "requested": int(args.requests),
            "completed": len(dtg_records),
            "records": dtg_records,
        })
        _write_json_atomic(motif_summary_path, {
            "schema_version": "innovation2_generation_batch_v1",
            "method": "motif",
            "protocol": protocol,
            "requested": int(args.requests),
            "completed": len(motif_records),
            "records": motif_records,
        })

    bbox_scaled = float(config.get("data", {}).get("bbox_scaled", 3.0))

    for sample_index in range(int(args.requests)):
        if sample_index in completed_set:
            continue

        sample_seed = int(args.seed) + sample_index
        _seed(sample_seed)

        dtg_run_dir = dtg_dir / ("%04d_seed_%d" % (sample_index, sample_seed))
        motif_run_dir = motif_dir / ("%04d_seed_%d" % (sample_index, sample_seed))
        dtg_run_dir.mkdir(parents=True, exist_ok=True)
        motif_run_dir.mkdir(parents=True, exist_ok=True)

        item = dataset[int(prior_rows[sample_index])]
        prior = item["prior"]
        prior_faces = int(item["num_faces"])

        # Base record structure
        rec_base = {
            "sample_index": sample_index,
            "seed": sample_seed,
            "method": "dtg",
            "prior_gate": 0.0,
            "prior_uid": item["uid"],
            "prior_dataset_index": int(prior_rows[sample_index]),
            "prior_hdf5_row": int(item["row"]),
            "prior_num_faces": prior_faces,
            "generated_num_faces": None,
            "face_count_gap": None,
            "face_edge_success": False,
            "edge_vert_success": False,
            "geometry_success": False,
            "step_written": False,
            "stl_written": False,
            "watertight_valid": False,
        }
        rec_motif = {
            "sample_index": sample_index,
            "seed": sample_seed,
            "method": "motif",
            "prior_gate": 1.0,
            "prior_uid": item["uid"],
            "prior_dataset_index": int(prior_rows[sample_index]),
            "prior_hdf5_row": int(item["row"]),
            "prior_num_faces": prior_faces,
            "generated_num_faces": None,
            "face_count_gap": None,
            "face_edge_success": False,
            "edge_vert_success": False,
            "geometry_success": False,
            "step_written": False,
            "stl_written": False,
            "watertight_valid": False,
        }

        # Directive 9: Try / Except / Finally Architecture
        try:
            # Stage 1: FaceEdge sampling with face_count_gap <= 2 restriction
            fef = None
            face_gap = 0
            for attempt in range(10):
                fef_candidate = backend.sample_fef_baseline(face_edge_model)
                gen_faces = int(fef_candidate.shape[0])
                face_gap = abs(gen_faces - prior_faces)
                if face_gap <= 2:
                    fef = fef_candidate
                    break

            # Directive 8: If 10 attempts fail, reject sample and continue!
            if fef is None:
                rec_base["failure_stage"] = "compatible_face_count_topology"
                rec_base["error"] = "no topology with face_count_gap <= 2 after 10 attempts"
                rec_motif["failure_stage"] = "compatible_face_count_topology"
                rec_motif["error"] = "no topology with face_count_gap <= 2 after 10 attempts"
                continue

            rec_base["face_edge_success"] = True
            rec_motif["face_edge_success"] = True
            rec_base["generated_num_faces"] = int(fef.shape[0])
            rec_motif["generated_num_faces"] = int(fef.shape[0])
            rec_base["face_count_gap"] = face_gap
            rec_motif["face_count_gap"] = face_gap

            # Stage 2: EdgeVert Completion
            topology = backend.complete_edge_vertex(
                fef, attempts=int(config["generation"]["edge_vert_attempts"])
            )
            rec_base["edge_vert_success"] = True
            rec_motif["edge_vert_success"] = True

            # Save RNG states before geometry generation
            torch_state = torch.get_rng_state()
            cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            np_state = np.random.get_state()
            py_state = random.getstate()

            # --- Branch 1: Baseline (prior_gate = 0.0) ---
            prior_bbox_model.set_prior(None)
            geom_base = backend.generate_geometry(topology, prior=prior, prior_bbox_model=prior_bbox_model)
            rec_base["geometry_success"] = True
            res_base = _reconstruct_and_export(dtg_run_dir, topology, geom_base, bbox_scaled)
            rec_base.update(res_base)
            _write_json_atomic(dtg_run_dir / "trace.json", rec_base)

            # --- Branch 2: Our PriorAllocator (prior_gate = 1.0) ---
            # Restore EXACT RNG states
            torch.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state(cuda_state)
            np.random.set_state(np_state)
            random.setstate(py_state)

            batch_prior = {
                key: (val.unsqueeze(0) if val.ndim > 0 else val.reshape(1)).to(backend.device)
                for key, val in prior.items()
            }
            node_graph = extract_motif_node_graph(batch_prior, max_nodes=MAX_MOTIF_NODES)
            fef_tensor = torch.as_tensor(fef, dtype=torch.long, device=backend.device).unsqueeze(0)
            gen_mask = torch.ones((1, fef.shape[0]), dtype=torch.bool, device=backend.device)

            with torch.no_grad():
                per_face_prior, _, _ = prior_bbox_model.prior_allocator(
                    node_graph["node_features"],
                    node_graph["hosted_adj"],
                    node_graph["thin_wall_adj"],
                    node_graph["node_mask"],
                    node_graph["node_roles"],
                    fef_tensor,
                    gen_mask,
                )
                prior_bbox_model.set_prior(per_face_prior)

            geom_motif = backend.generate_geometry(topology, prior=prior, prior_bbox_model=prior_bbox_model)
            rec_motif["geometry_success"] = True
            res_motif = _reconstruct_and_export(motif_run_dir, topology, geom_motif, bbox_scaled)
            rec_motif.update(res_motif)
            _write_json_atomic(motif_run_dir / "trace.json", rec_motif)

            print(
                "PAIRED SAMPLE %d/%d (Seed: %d) | Gap: %d | DTG STEP: %s | Motif STEP: %s"
                % (
                    sample_index + 1,
                    args.requests,
                    sample_seed,
                    face_gap,
                    rec_base.get("step_written", False),
                    rec_motif.get("step_written", False),
                ),
                flush=True,
            )
        except Exception as exc:
            rec_base["error"] = "%s: %s" % (type(exc).__name__, exc)
            rec_motif["error"] = "%s: %s" % (type(exc).__name__, exc)
        finally:
            # Directive 9: ALWAYS clear prior and update atomic summaries
            prior_bbox_model.set_prior(None)
            dtg_records.append(rec_base)
            motif_records.append(rec_motif)
            save_summaries()

    backend.release(face_edge_model)
    return {"dtg": dtg_records, "motif": motif_records}


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
    print("RUNNING PAIRED BENCHMARK GENERATION V6.1 (Requests: %d, Seed: %d)" % (args.requests, args.seed))
    print("=" * 80)

    result = generate_paired_benchmark(args, config, output_root, prior_dataset, rows)
    print("=" * 80)
    print("PAIRED BENCHMARK COMPLETE! Results written to %s" % output_root)
    print("=" * 80)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="generate CAD models")
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "config.yaml")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--seed", type=int, default=9000)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.pilot:
        print("Usage: python -m innovation2_z.generate --pilot [--requests 20]")
        return 0
    run_pilot(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
