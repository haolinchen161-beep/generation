"""Generate matched 3D CAD models using PriorFaceBbox (Strict Paired Control Benchmark v6.1 Final).

Implements Core Directives:
- Independent try/except blocks for shared topology, DTG geometry, and Motif geometry (prevents cross-branch error pollution).
- Paired intersection set resume (dtg_indices & motif_indices).
- Protocol validation assertion (prior_rows, seed, requests, checkpoint metadata, max_nodes, sha256).
- Geometry arrays (npz/npy) & shared topology (fef, edgeFace, edgeVert) saving.
- Complete diagnostic record and protocol metadata fields.
"""

from __future__ import annotations

import argparse
import hashlib
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


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_geometry_arrays(run_dir: Path, geometry: Dict[str, np.ndarray]) -> None:
    """Directive 4: Save compressed geometry arrays (npz) and face_bbox.npy."""
    np.savez_compressed(
        run_dir / "geometry_arrays.npz",
        face_bbox=np.asarray(geometry["face_bbox"]),
        vert_geom=np.asarray(geometry["vert_geom"]),
        edge_geom=np.asarray(geometry["edge_geom"]),
        face_geom=np.asarray(geometry["face_geom"]),
    )
    np.save(run_dir / "face_bbox.npy", np.asarray(geometry["face_bbox"]))


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

    ckpt_path = PACKAGE_ROOT / "checkpoints" / "prior_face_bbox" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Error: Trained PriorAllocator checkpoint NOT found at {ckpt_path}! Train first via python -m innovation2_z.train_prior_bbox.")

    checkpoint = torch.load(ckpt_path, map_location=backend.device)

    if "prior_allocator_state_dict" not in checkpoint:
        raise KeyError(f"Error: Checkpoint at {ckpt_path} does not contain 'prior_allocator_state_dict'!")

    ckpt_schema = checkpoint.get("schema_version", "unknown")
    ckpt_epoch = int(checkpoint.get("epoch", -1))
    ckpt_margin = float(checkpoint.get("causal_margin", checkpoint.get("best_causal_margin", 0.0)))
    ckpt_sha256 = _file_sha256(ckpt_path)

    if ckpt_schema != "prior_face_bbox_v6_1" and ckpt_schema != "prior_face_bbox_v6":
        raise ValueError(f"Error: Incompatible checkpoint schema version '{ckpt_schema}'!")

    if int(checkpoint.get("max_nodes", -1)) != MAX_MOTIF_NODES:
        raise ValueError(f"Error: Checkpoint max_nodes ({checkpoint.get('max_nodes')}) mismatch with current MAX_MOTIF_NODES ({MAX_MOTIF_NODES})!")

    prior_bbox_model = build_prior_face_bbox_model(base_bbox_model).to(backend.device)
    prior_bbox_model.prior_allocator.load_state_dict(checkpoint["prior_allocator_state_dict"], strict=True)
    print(f"  [SUCCESS] Loaded Verified PriorAllocator Checkpoint (Schema: {ckpt_schema}, Epoch: {ckpt_epoch}, Margin: {ckpt_margin:+.4f})")

    prior_bbox_model.eval()

    expected_protocol = {
        "prior_rows": prior_rows,
        "base_seed": int(args.seed),
        "requests": int(args.requests),
        "paired": True,
        "checkpoint_schema": ckpt_schema,
        "checkpoint_epoch": ckpt_epoch,
        "checkpoint_causal_margin": ckpt_margin,
        "max_nodes": MAX_MOTIF_NODES,
        "checkpoint_sha256": ckpt_sha256,
    }

    dtg_summary_path = dtg_dir / "batch_summary.json"
    motif_summary_path = motif_dir / "batch_summary.json"

    dtg_records: List[Dict[str, Any]] = []
    motif_records: List[Dict[str, Any]] = []
    completed_set = set()

    # Directive 1 & 2: Resume with Intersection Set & Protocol Validation Assertion!
    if dtg_summary_path.exists() and motif_summary_path.exists():
        try:
            with dtg_summary_path.open("r", encoding="utf-8") as h:
                old_dtg_sum = json.load(h)
                dtg_raw = old_dtg_sum.get("records", [])
                old_proto = old_dtg_sum.get("protocol", {})

            with motif_summary_path.open("r", encoding="utf-8") as h:
                old_motif_sum = json.load(h)
                motif_raw = old_motif_sum.get("records", [])

            # Directive 2: Assert Protocol Compatibility
            for k in ["prior_rows", "base_seed", "requests", "paired"]:
                if old_proto.get(k) != expected_protocol[k]:
                    raise RuntimeError(f"Existing output protocol field '{k}' ({old_proto.get(k)}) does not match current run ({expected_protocol[k]})! Clear output dir first.")

            dtg_indices = {int(r["sample_index"]) for r in dtg_raw}
            motif_indices = {int(r["sample_index"]) for r in motif_raw}

            # Directive 1: Paired Intersection Set
            completed_set = dtg_indices & motif_indices
            dtg_records = [r for r in dtg_raw if int(r["sample_index"]) in completed_set]
            motif_records = [r for r in motif_raw if int(r["sample_index"]) in completed_set]
            print(f"  [RESUME] Found {len(completed_set)} fully paired finished samples in output directory.")
        except Exception as exc:
            if "does not match current run" in str(exc):
                raise exc
            dtg_records, motif_records, completed_set = [], [], set()

    def save_summaries():
        _write_json_atomic(dtg_summary_path, {
            "schema_version": "innovation2_generation_batch_v1",
            "method": "dtg",
            "protocol": expected_protocol,
            "requested": int(args.requests),
            "completed": len(dtg_records),
            "records": dtg_records,
        })
        _write_json_atomic(motif_summary_path, {
            "schema_version": "innovation2_generation_batch_v1",
            "method": "motif",
            "protocol": expected_protocol,
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

        # Extract Diagnostic Prior Counts (Directive 7)
        batch_prior_cpu = {
            key: (val.unsqueeze(0) if val.ndim > 0 else val.reshape(1))
            for key, val in prior.items()
        }
        node_graph_diag = extract_motif_node_graph(batch_prior_cpu, max_nodes=MAX_MOTIF_NODES)
        sheet_node_cnt = int(node_graph_diag["sheet_node_count"][0].item())
        hole_node_cnt = int(node_graph_diag["hole_node_count"][0].item())
        repeat_node_cnt = int(node_graph_diag["repeat_node_count"][0].item())
        has_struct_prior = bool(node_graph_diag["has_structural_prior"][0].item())

        rec_base = {
            "sample_index": sample_index,
            "seed": sample_seed,
            "method": "dtg",
            "prior_gate": 0.0,
            "prior_uid": item["uid"],
            "prior_dataset_index": int(prior_rows[sample_index]),
            "prior_hdf5_row": int(item["row"]),
            "prior_num_faces": prior_faces,
            "sheet_node_count": sheet_node_cnt,
            "hole_node_count": hole_node_cnt,
            "repeat_node_count": repeat_node_cnt,
            "has_structural_prior": has_struct_prior,
            "generated_num_faces": None,
            "face_count_gap": None,
            "generated_shared_edges": None,
            "generated_mean_degree": None,
            "generated_max_degree": None,
            "face_edge_success": False,
            "edge_vert_success": False,
            "geometry_success": False,
            "step_written": False,
            "stl_written": False,
            "watertight_valid": False,
            "failure_stage": None,
            "error": None,
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
            "sheet_node_count": sheet_node_cnt,
            "hole_node_count": hole_node_cnt,
            "repeat_node_count": repeat_node_cnt,
            "has_structural_prior": has_struct_prior,
            "generated_num_faces": None,
            "face_count_gap": None,
            "generated_shared_edges": None,
            "generated_mean_degree": None,
            "generated_max_degree": None,
            "face_edge_success": False,
            "edge_vert_success": False,
            "geometry_success": False,
            "step_written": False,
            "stl_written": False,
            "watertight_valid": False,
            "failure_stage": None,
            "error": None,
        }

        # Directive 3: Independent Try / Except Blocks!
        # --- Stage A: Shared Topology Sampling & EdgeVert ---
        try:
            fef = None
            face_gap = 0
            for attempt in range(10):
                fef_candidate = backend.sample_fef_baseline(face_edge_model)
                gen_faces = int(fef_candidate.shape[0])
                face_gap = abs(gen_faces - prior_faces)
                if face_gap <= 2:
                    fef = fef_candidate
                    break

            if fef is None:
                rec_base["failure_stage"] = "compatible_face_count_topology"
                rec_base["error"] = "no topology with face_count_gap <= 2 after 10 attempts"
                rec_motif["failure_stage"] = "compatible_face_count_topology"
                rec_motif["error"] = "no topology with face_count_gap <= 2 after 10 attempts"
                dtg_records.append(rec_base)
                motif_records.append(rec_motif)
                save_summaries()
                continue

            rec_base["face_edge_success"] = True
            rec_motif["face_edge_success"] = True
            gen_faces_cnt = int(fef.shape[0])
            rec_base["generated_num_faces"] = gen_faces_cnt
            rec_motif["generated_num_faces"] = gen_faces_cnt
            rec_base["face_count_gap"] = face_gap
            rec_motif["face_count_gap"] = face_gap

            fef_binary = (fef > 0).astype(int)
            shared_edges = int(fef_binary.sum() // 2)
            degrees = fef_binary.sum(axis=-1)
            mean_deg = float(np.mean(degrees)) if len(degrees) > 0 else 0.0
            max_deg = int(np.max(degrees)) if len(degrees) > 0 else 0

            rec_base["generated_shared_edges"] = shared_edges
            rec_motif["generated_shared_edges"] = shared_edges
            rec_base["generated_mean_degree"] = mean_deg
            rec_motif["generated_mean_degree"] = mean_deg
            rec_base["generated_max_degree"] = max_deg
            rec_motif["generated_max_degree"] = max_deg

            topology = backend.complete_edge_vertex(
                fef, attempts=int(config["generation"]["edge_vert_attempts"])
            )
            rec_base["edge_vert_success"] = True
            rec_motif["edge_vert_success"] = True

            # Directive 5: Save Shared Topology Files in both directories
            np.save(dtg_run_dir / "fef.npy", fef)
            np.save(dtg_run_dir / "edge_face.npy", topology["edgeFace_adj"].detach().cpu().numpy())
            np.save(dtg_run_dir / "edge_vert.npy", topology["edgeVert_adj"].detach().cpu().numpy())

            np.save(motif_run_dir / "fef.npy", fef)
            np.save(motif_run_dir / "edge_face.npy", topology["edgeFace_adj"].detach().cpu().numpy())
            np.save(motif_run_dir / "edge_vert.npy", topology["edgeVert_adj"].detach().cpu().numpy())

        except Exception as exc:
            rec_base["failure_stage"] = "shared_topology"
            rec_base["error"] = "%s: %s" % (type(exc).__name__, exc)
            rec_motif["failure_stage"] = "shared_topology"
            rec_motif["error"] = "%s: %s" % (type(exc).__name__, exc)
            dtg_records.append(rec_base)
            motif_records.append(rec_motif)
            save_summaries()
            continue

        # Save RNG states before geometry generation
        torch_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        np_state = np.random.get_state()
        py_state = random.getstate()

        # Directive 3: --- Stage B: DTG Geometry Branch (Isolated try/except) ---
        try:
            prior_bbox_model.set_prior(None)
            geom_base = backend.generate_geometry(topology, prior=prior, prior_bbox_model=prior_bbox_model)
            rec_base["geometry_success"] = True
            save_geometry_arrays(dtg_run_dir, geom_base)
            res_base = _reconstruct_and_export(dtg_run_dir, topology, geom_base, bbox_scaled)
            rec_base.update(res_base)
            _write_json_atomic(dtg_run_dir / "trace.json", rec_base)
        except Exception as exc:
            rec_base["failure_stage"] = "dtg_geometry"
            rec_base["error"] = "%s: %s" % (type(exc).__name__, exc)

        # Directive 3: --- Stage C: Motif Geometry Branch (Isolated try/except) ---
        try:
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
            save_geometry_arrays(motif_run_dir, geom_motif)
            res_motif = _reconstruct_and_export(motif_run_dir, topology, geom_motif, bbox_scaled)
            rec_motif.update(res_motif)
            _write_json_atomic(motif_run_dir / "trace.json", rec_motif)
        except Exception as exc:
            rec_motif["failure_stage"] = "motif_geometry"
            rec_motif["error"] = "%s: %s" % (type(exc).__name__, exc)
        finally:
            # Directive 6: ALWAYS clear prior and update atomic summaries
            prior_bbox_model.set_prior(None)
            dtg_records.append(rec_base)
            motif_records.append(rec_motif)
            save_summaries()

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
    print("RUNNING PAIRED BENCHMARK GENERATION V6.1 FINAL (Requests: %d, Seed: %d)" % (args.requests, args.seed))
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
