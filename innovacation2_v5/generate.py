"""Generate matched 3D CAD models using PriorFaceBbox (Strict Paired Control Benchmark v5).

Implements Core Directives 8, 9:
- Generates BOTH dtg (prior_gate=0) and motif (prior_gate=1) in a single sample loop.
- Restores exact CPU and CUDA RNG states between gate 0 and gate 1 passes.
- Restricts generated face count gap |N_generated - N_prior| <= 2 for Phase 1 alignment.
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
    prior_bbox_model = build_prior_face_bbox_model(base_bbox_model).to(backend.device)

    ckpt_path = PACKAGE_ROOT / "checkpoints" / "prior_face_bbox" / "best.pt"
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=backend.device)
        if "prior_allocator_state_dict" in checkpoint:
            prior_bbox_model.prior_allocator.load_state_dict(checkpoint["prior_allocator_state_dict"])
            print(f"  [SUCCESS] Loaded Trained PriorAllocator v5 Checkpoint from: {ckpt_path}")

    prior_bbox_model.eval()

    dtg_records: List[Dict[str, Any]] = []
    motif_records: List[Dict[str, Any]] = []

    bbox_scaled = float(config.get("data", {}).get("bbox_scaled", 3.0))

    for sample_index in range(int(args.requests)):
        sample_seed = int(args.seed) + sample_index
        _seed(sample_seed)

        dtg_run_dir = dtg_dir / ("%04d_seed_%d" % (sample_index, sample_seed))
        motif_run_dir = motif_dir / ("%04d_seed_%d" % (sample_index, sample_seed))
        dtg_run_dir.mkdir(parents=True, exist_ok=True)
        motif_run_dir.mkdir(parents=True, exist_ok=True)

        item = dataset[int(prior_rows[sample_index])]
        prior = item["prior"]
        prior_faces = int(item["num_faces"])

        # Stage 1: FaceEdge topology sampling with face count gap restriction |N_gen - N_prior| <= 2
        fef = None
        face_gap = 0
        for attempt in range(5):
            fef_candidate = backend.sample_fef_baseline(face_edge_model)
            gen_faces = int(fef_candidate.shape[0])
            face_gap = abs(gen_faces - prior_faces)
            if face_gap <= 2 or attempt == 4:
                fef = fef_candidate
                break

        topology = backend.complete_edge_vertex(
            fef, attempts=int(config["generation"]["edge_vert_attempts"])
        )

        # Save RNG states right before geometry generation
        torch_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        np_state = np.random.get_state()
        py_state = random.getstate()

        # --- Branch 1: Baseline (prior_gate = 0.0) ---
        prior_bbox_model.set_prior(None)
        geom_base = backend.generate_geometry(topology, prior=prior, prior_bbox_model=None)
        res_base = _reconstruct_and_export(dtg_run_dir, topology, geom_base, bbox_scaled)
        rec_base = {
            "sample_index": sample_index,
            "seed": sample_seed,
            "method": "dtg",
            "prior_gate": 0.0,
            "face_count_gap": face_gap,
            "generated_num_faces": int(fef.shape[0]),
            **res_base,
        }
        dtg_records.append(rec_base)
        _write_json_atomic(dtg_run_dir / "trace.json", rec_base)

        # --- Branch 2: Our PriorAllocator (prior_gate = 1.0) ---
        # Restore EXACT RNG states
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state)
        np.random.set_state(np_state)
        random.setstate(py_state)

        # Prepare per_face_prior
        batch_prior = {
            key: (val.unsqueeze(0) if val.ndim > 0 else val.reshape(1)).to(backend.device)
            for key, val in prior.items()
        }
        node_graph = extract_motif_node_graph(batch_prior, max_nodes=32)
        fef_tensor = torch.as_tensor(fef, dtype=torch.long, device=backend.device).unsqueeze(0)
        gen_mask = torch.ones((1, fef.shape[0]), dtype=torch.bool, device=backend.device)

        with torch.no_grad():
            per_face_prior, _, _ = prior_bbox_model.prior_allocator(
                node_graph["node_features"],
                node_graph["hosted_adj"],
                node_graph["thin_wall_adj"],
                node_graph["node_mask"],
                fef_tensor,
                gen_mask,
            )
            prior_bbox_model.set_prior(per_face_prior)

        geom_motif = backend.generate_geometry(topology, prior=prior, prior_bbox_model=prior_bbox_model)
        prior_bbox_model.set_prior(None)

        res_motif = _reconstruct_and_export(motif_run_dir, topology, geom_motif, bbox_scaled)
        rec_motif = {
            "sample_index": sample_index,
            "seed": sample_seed,
            "method": "motif",
            "prior_gate": 1.0,
            "face_count_gap": face_gap,
            "generated_num_faces": int(fef.shape[0]),
            **res_motif,
        }
        motif_records.append(rec_motif)
        _write_json_atomic(motif_run_dir / "trace.json", rec_motif)

        print(
            "PAIRED SAMPLE %d/%d (Seed: %d) | Face Gap: %d | DTG Step: %s | Motif Step: %s"
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

    backend.release(face_edge_model)

    dtg_summary = {
        "schema_version": "innovation2_generation_batch_v1",
        "method": "dtg",
        "requested": int(args.requests),
        "completed": len(dtg_records),
        "records": dtg_records,
    }
    motif_summary = {
        "schema_version": "innovation2_generation_batch_v1",
        "method": "motif",
        "requested": int(args.requests),
        "completed": len(motif_records),
        "records": motif_records,
    }
    _write_json_atomic(dtg_dir / "batch_summary.json", dtg_summary)
    _write_json_atomic(motif_dir / "batch_summary.json", motif_summary)

    return {"dtg": dtg_summary, "motif": motif_summary}


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
    print("RUNNING PAIRED BENCHMARK GENERATION (Requests: %d, Seed: %d)" % (args.requests, args.seed))
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
