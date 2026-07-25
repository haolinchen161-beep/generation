"""Build resumable, split HDF5 files from eligible Innovation-1 STEP records.

The new directory owns a compact UID/split allow-list (no duplicated geometry
or prior payload).  Every target and analytical face attribute is parsed again
from the original STEP so face ordering stays identical across prior, topology
and geometry targets.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from innovation1_v3_brep_motif_graph.brep_cleaner import check_dtg_train_compatible
from innovation1_v3_brep_motif_graph.brep_loader import parse_step_file

from innovation2_stagewise_prior_generation.data import inspect_hdf5


MOTIF_TYPES = ("sheet_region", "loop_or_hole", "repeated_feature")
RELATION_TYPES = ("hosted_by", "thin_wall_pair")
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_ALIASES = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}
SURFACE_COUNT = 6


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _read_eligible(path: Path, limit: int = 0) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    rows: List[Dict[str, Any]] = []
    seen = set()
    counts = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            uid = str(record.get("uid", ""))
            split = SPLIT_ALIASES.get(str(record.get("split", "")))
            if not uid or split is None:
                raise ValueError("invalid eligible record at line %d" % line_number)
            if uid in seen:
                raise ValueError("duplicate eligible UID: %s" % uid)
            seen.add(uid)
            rows.append({"uid": uid, "split": split})
            counts[split] += 1
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise ValueError("eligible index is empty: %s" % path)
    return rows, dict(counts)


def _create_dataset(
    group: h5py.Group,
    name: str,
    shape: Tuple[int, ...],
    dtype,
    fillvalue=0,
    compression: bool = True,
) -> h5py.Dataset:
    chunks = (min(64, max(1, shape[0])),) + shape[1:]
    kwargs = {
        "shape": shape,
        "dtype": dtype,
        "chunks": chunks,
        "fillvalue": fillvalue,
    }
    if compression and len(shape) > 1:
        kwargs.update({"compression": "gzip", "compression_opts": 1, "shuffle": True})
    return group.create_dataset(name, **kwargs)


def _initialize_h5(path: Path, rows: Sequence[Dict[str, Any]], limits: Mapping[str, int]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    count = len(rows)
    max_faces = int(limits["max_faces"])
    max_edges = int(limits["max_edges"])
    max_vertices = int(limits["max_vertices"])
    max_face_edges = int(limits["max_face_edges"])
    max_vertex_faces = int(limits["max_vertex_faces"])
    with h5py.File(path, "w", libver="latest") as handle:
        handle.attrs["schema_version"] = "innovation2_stagewise_prior_h5_v1"
        meta = handle.create_group("meta")
        prior = handle.create_group("prior")
        target = handle.create_group("target")

        meta.create_dataset("uid", data=np.asarray([row["uid"].encode("ascii") for row in rows], dtype="S32"))
        meta.create_dataset("written", shape=(count,), dtype=np.bool_, fillvalue=False)
        meta.create_dataset("rejected", shape=(count,), dtype=np.bool_, fillvalue=False)
        meta.create_dataset("num_faces", shape=(count,), dtype=np.uint8, fillvalue=0)
        meta.create_dataset("num_edges", shape=(count,), dtype=np.uint16, fillvalue=0)
        meta.create_dataset("num_vertices", shape=(count,), dtype=np.uint16, fillvalue=0)

        _create_dataset(prior, "surface_type", (count, max_faces), np.uint8, fillvalue=SURFACE_COUNT)
        _create_dataset(prior, "surface_confidence", (count, max_faces), np.float16)
        _create_dataset(prior, "motif_membership", (count, max_faces, 3), np.float16)
        _create_dataset(prior, "motif_confidence", (count, max_faces, 3), np.float16)
        _create_dataset(prior, "motif_instance", (count, max_faces, 3), np.uint8)
        _create_dataset(prior, "motif_counts", (count, 3), np.uint8)
        _create_dataset(prior, "relation_counts", (count, 2), np.uint8)
        _create_dataset(prior, "pair_relations", (count, max_faces, max_faces, 2), np.float16)
        _create_dataset(prior, "face_edge_cont", (count, max_faces, 4), np.float16)
        _create_dataset(prior, "face_bbox_bins", (count, max_faces, 6), np.uint8)
        # Five non-exact layout channels; six dequantized bins are prepended by data.py.
        _create_dataset(prior, "face_bbox_cont", (count, max_faces, 5), np.float16)
        _create_dataset(prior, "face_geom_cont", (count, max_faces, 14), np.float16)
        _create_dataset(prior, "face_mask", (count, max_faces), np.bool_)

        _create_dataset(target, "fef_adj", (count, max_faces, max_faces), np.int8)
        _create_dataset(target, "face_bbox", (count, max_faces, 6), np.float32)
        _create_dataset(target, "face_ctrl", (count, max_faces, 48), np.float32)
        _create_dataset(target, "edge_ctrl", (count, max_edges, 12), np.float32)
        _create_dataset(target, "vert_coords", (count, max_vertices, 3), np.float32)
        _create_dataset(target, "edge_face", (count, max_edges, 2), np.int16, fillvalue=-1)
        _create_dataset(target, "edge_vert", (count, max_edges, 2), np.int16, fillvalue=-1)
        _create_dataset(target, "face_edge", (count, max_faces, max_face_edges), np.int16, fillvalue=-1)
        _create_dataset(target, "face_edge_count", (count, max_faces), np.uint8)
        _create_dataset(target, "vert_face", (count, max_vertices, max_vertex_faces), np.int16, fillvalue=-1)
        _create_dataset(target, "vert_face_count", (count, max_vertices), np.uint8)
        _create_dataset(target, "edge_mask", (count, max_edges), np.bool_)
        _create_dataset(target, "vert_mask", (count, max_vertices), np.bool_)
        handle.flush()


def _normalize_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.divide(values, np.maximum(norm, 1e-8), out=np.zeros_like(values), where=norm > 1e-8)


def _surface_ids(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.int64)
    result = np.full(raw.shape, 5, dtype=np.uint8)
    for value in range(5):
        result[raw == value] = value
    return result


def _safe_array(data: Mapping[str, Any], key: str, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
    value = np.asarray(data.get(key), dtype=dtype)
    if value.shape != shape:
        raise ValueError("%s shape %s != %s" % (key, value.shape, shape))
    if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
        raise ValueError("%s contains non-finite values" % key)
    return value


def _motif_prior(
    record: Mapping[str, Any], num_faces: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    membership = np.zeros((num_faces, 3), dtype=np.float32)
    confidence = np.zeros((num_faces, 3), dtype=np.float32)
    motif_sizes = np.zeros((num_faces, 3), dtype=np.float32)
    instances = np.zeros((num_faces, 3), dtype=np.uint8)
    motif_counts = np.zeros(3, dtype=np.uint8)
    node_by_id: Dict[str, Mapping[str, Any]] = {}
    local_ids = Counter()
    for node in record.get("motif_nodes", []) or []:
        motif_type = str(node.get("type", ""))
        if motif_type not in MOTIF_TYPES:
            continue
        node_by_id[str(node.get("id", ""))] = node
        channel = MOTIF_TYPES.index(motif_type)
        local_ids[channel] += 1
        motif_counts[channel] += 1
        face_ids = sorted({int(face_id) for face_id in node.get("face_ids", [])})
        if not face_ids or min(face_ids) < 0 or max(face_ids) >= num_faces:
            raise ValueError("motif face ID is outside original face order")
        score = float(np.clip(node.get("confidence", 0.0), 0.0, 1.0))
        size = float(len(face_ids) / max(num_faces, 1))
        for face_id in face_ids:
            previous = confidence[face_id, channel]
            membership[face_id, channel] = 1.0
            confidence[face_id, channel] = max(confidence[face_id, channel], score)
            motif_sizes[face_id, channel] = max(motif_sizes[face_id, channel], size)
            if instances[face_id, channel] == 0 or score >= previous:
                instances[face_id, channel] = min(local_ids[channel], 255)

    relations = np.zeros((num_faces, num_faces, 2), dtype=np.float32)
    relation_counts = np.zeros(2, dtype=np.uint8)
    for relation in record.get("motif_relations", []) or []:
        relation_type = str(relation.get("type", ""))
        if relation_type not in RELATION_TYPES:
            continue
        source = node_by_id.get(str(relation.get("source", "")))
        target = node_by_id.get(str(relation.get("target", "")))
        if source is None or target is None:
            continue
        channel = RELATION_TYPES.index(relation_type)
        relation_counts[channel] = min(int(relation_counts[channel]) + 1, 255)
        score = float(np.clip(relation.get("confidence", 0.0), 0.0, 1.0))
        for left in source.get("face_ids", []):
            for right in target.get("face_ids", []):
                i, j = int(left), int(right)
                if i == j:
                    continue
                relations[i, j, channel] = max(relations[i, j, channel], score)
                relations[j, i, channel] = max(relations[j, i, channel], score)
    return membership, confidence, relations, instances, motif_counts, relation_counts


def _pack_record(uid: str, step_path: str, compact: Mapping[str, Any], limits: Mapping[str, int]) -> Dict[str, Any]:
    parsed = parse_step_file(step_path)
    compatible, reason, _ = check_dtg_train_compatible(parsed, dataset="deepcad")
    if not compatible:
        raise ValueError(reason)

    nf = int(parsed["face_count"])
    ne = int(parsed["edge_count"])
    nv = int(parsed["vertex_count"])
    if int(compact.get("num_faces", -1)) != nf:
        raise ValueError("compact_step_face_count_mismatch")
    max_faces = int(limits["max_faces"])
    max_edges = int(limits["max_edges"])
    max_vertices = int(limits["max_vertices"])
    max_face_edges = int(limits["max_face_edges"])
    max_vertex_faces = int(limits["max_vertex_faces"])
    bbox_scaled = float(limits.get("bbox_scaled", 3.0))

    membership, motif_conf, pair_rel, motif_instance, motif_counts, relation_counts = _motif_prior(compact, nf)
    bbox = _safe_array(parsed, "face_bbox_wcs", (nf, 6))
    face_ctrl = _safe_array(parsed, "face_ctrs", (nf, 16, 3)).reshape(nf, 48)
    edge_ctrl = _safe_array(parsed, "edge_ctrs", (ne, 4, 3)).reshape(ne, 12)
    vertices = _safe_array(parsed, "vert_wcs", (nv, 3))
    fef = _safe_array(parsed, "fef_adj", (nf, nf), dtype=np.int64)
    edge_face = _safe_array(parsed, "edgeFace_adj", (ne, 2), dtype=np.int64)
    edge_vert = _safe_array(parsed, "edgeVert_adj", (ne, 2), dtype=np.int64)

    face_mask = np.zeros(max_faces, dtype=bool)
    face_mask[:nf] = True
    edge_mask = np.zeros(max_edges, dtype=bool)
    edge_mask[:ne] = True
    vert_mask = np.zeros(max_vertices, dtype=bool)
    vert_mask[:nv] = True

    raw_surface = _safe_array(parsed, "face_surface_type", (nf,), dtype=np.int64)
    surface = _surface_ids(raw_surface)
    surface_conf = np.where(raw_surface >= 0, 1.0, 0.0).astype(np.float32)
    normals = _normalize_vector(_safe_array(parsed, "face_analytical_normals", (nf, 3)))
    cyl_axis = _normalize_vector(_safe_array(parsed, "face_cylinder_axis", (nf, 3)))
    layout_direction = np.where(
        (np.linalg.norm(normals, axis=-1) > 1e-6)[:, None],
        normals,
        cyl_axis,
    )
    areas = np.maximum(_safe_array(parsed, "face_area", (nf,)), 0.0)
    relative_area = areas / max(float(areas.sum()), 1e-8)
    dims = np.maximum(bbox[:, 3:] - bbox[:, :3], 0.0)
    part_min = bbox[:, :3].min(axis=0)
    part_max = bbox[:, 3:].max(axis=0)
    part_dims = np.maximum(part_max - part_min, 1e-8)
    part_scale = max(float(part_dims.max()), 1e-8)
    centers = 0.5 * (bbox[:, :3] + bbox[:, 3:])
    center_unit = np.clip((centers - part_min) / part_dims, 0.0, 1.0)
    size_unit = np.clip(dims / part_scale, 0.0, 1.0)
    # Layout is a deliberately coarse 2-bit prior, not the six-valued DTG bbox
    # regression target.  Values are stored on 0/5/10/15 so data.py can keep a
    # stable [0, 1] representation while exact centers/sizes remain target-only.
    bbox_bins = (
        np.rint(np.concatenate([center_unit, size_unit], axis=-1) * 3.0) * 5.0
    ).astype(np.uint8)
    thinness = dims.min(axis=-1) / np.maximum(dims.max(axis=-1), 1e-8)

    node_conf = motif_conf.max(axis=-1)
    motif_size = np.zeros(nf, dtype=np.float32)
    for node in compact.get("motif_nodes", []) or []:
        ids = [int(face_id) for face_id in node.get("face_ids", [])]
        size = len(set(ids)) / max(nf, 1)
        for face_id in ids:
            if 0 <= face_id < nf:
                motif_size[face_id] = max(motif_size[face_id], size)
    coverage = np.full(nf, float(np.any(membership > 0, axis=-1).mean()), dtype=np.float32)
    face_edge_cont = np.stack([node_conf, motif_size, coverage, relative_area], axis=-1)
    face_bbox_cont = np.concatenate(
        [relative_area[:, None], thinness[:, None], layout_direction],
        axis=-1,
    )

    mean_curv = _safe_array(parsed, "face_mean_curvature", (nf,))
    max_curv = _safe_array(parsed, "face_max_curvature", (nf,))
    var_curv = _safe_array(parsed, "face_var_curvature", (nf,))
    gaussian = _safe_array(parsed, "face_gaussian_sign", (nf,))
    cyl_radius = np.maximum(_safe_array(parsed, "face_cylinder_radius", (nf,)), 0.0) / part_scale
    normal_valid = (np.linalg.norm(normals, axis=-1) > 1e-6).astype(np.float32)
    cylinder_valid = (surface == 1).astype(np.float32)
    curvature_valid = (raw_surface >= 0).astype(np.float32)
    face_geom_cont = np.concatenate(
        [
            normals,
            mean_curv[:, None],
            max_curv[:, None],
            var_curv[:, None],
            gaussian[:, None],
            cyl_radius[:, None],
            cyl_axis,
            normal_valid[:, None],
            cylinder_valid[:, None],
            curvature_valid[:, None],
        ],
        axis=-1,
    )
    if face_geom_cont.shape[1] != 14:
        raise AssertionError("face geometry feature width drift")

    prior = {
        "surface_type": np.full(max_faces, SURFACE_COUNT, dtype=np.uint8),
        "surface_confidence": np.zeros(max_faces, dtype=np.float16),
        "motif_membership": np.zeros((max_faces, 3), dtype=np.float16),
        "motif_confidence": np.zeros((max_faces, 3), dtype=np.float16),
        "motif_instance": np.zeros((max_faces, 3), dtype=np.uint8),
        "motif_counts": motif_counts,
        "relation_counts": relation_counts,
        "pair_relations": np.zeros((max_faces, max_faces, 2), dtype=np.float16),
        "face_edge_cont": np.zeros((max_faces, 4), dtype=np.float16),
        "face_bbox_bins": np.zeros((max_faces, 6), dtype=np.uint8),
        "face_bbox_cont": np.zeros((max_faces, 5), dtype=np.float16),
        "face_geom_cont": np.zeros((max_faces, 14), dtype=np.float16),
        "face_mask": face_mask,
    }
    prior["surface_type"][:nf] = surface
    prior["surface_confidence"][:nf] = surface_conf.astype(np.float16)
    prior["motif_membership"][:nf] = membership.astype(np.float16)
    prior["motif_confidence"][:nf] = motif_conf.astype(np.float16)
    prior["motif_instance"][:nf] = motif_instance
    prior["pair_relations"][:nf, :nf] = pair_rel.astype(np.float16)
    prior["face_edge_cont"][:nf] = np.clip(face_edge_cont, -1e4, 1e4).astype(np.float16)
    prior["face_bbox_bins"][:nf] = bbox_bins
    prior["face_bbox_cont"][:nf] = np.clip(face_bbox_cont, -1e4, 1e4).astype(np.float16)
    prior["face_geom_cont"][:nf] = np.clip(face_geom_cont, -1e4, 1e4).astype(np.float16)

    face_edge = np.full((max_faces, max_face_edges), -1, dtype=np.int16)
    face_edge_count = np.zeros(max_faces, dtype=np.uint8)
    for face_id, values in enumerate(parsed["faceEdge_adj"]):
        values = np.asarray(values, dtype=np.int16).reshape(-1)
        face_edge_count[face_id] = len(values)
        face_edge[face_id, : len(values)] = values
    vert_face = np.full((max_vertices, max_vertex_faces), -1, dtype=np.int16)
    vert_face_count = np.zeros(max_vertices, dtype=np.uint8)
    for vert_id, values in enumerate(parsed["vertFace_adj"]):
        values = np.asarray(values, dtype=np.int16).reshape(-1)
        vert_face_count[vert_id] = len(values)
        vert_face[vert_id, : len(values)] = values

    target = {
        "fef_adj": np.zeros((max_faces, max_faces), dtype=np.int8),
        "face_bbox": np.zeros((max_faces, 6), dtype=np.float32),
        "face_ctrl": np.zeros((max_faces, 48), dtype=np.float32),
        "edge_ctrl": np.zeros((max_edges, 12), dtype=np.float32),
        "vert_coords": np.zeros((max_vertices, 3), dtype=np.float32),
        "edge_face": np.full((max_edges, 2), -1, dtype=np.int16),
        "edge_vert": np.full((max_edges, 2), -1, dtype=np.int16),
        "face_edge": face_edge,
        "face_edge_count": face_edge_count,
        "vert_face": vert_face,
        "vert_face_count": vert_face_count,
        "edge_mask": edge_mask,
        "vert_mask": vert_mask,
    }
    target["fef_adj"][:nf, :nf] = fef.astype(np.int8)
    # DTG geometry datasets train in the configured x3 coordinate space; the
    # generator divides by the same value before B-rep reconstruction.
    target["face_bbox"][:nf] = bbox * bbox_scaled
    target["face_ctrl"][:nf] = face_ctrl * bbox_scaled
    target["edge_ctrl"][:ne] = edge_ctrl * bbox_scaled
    target["vert_coords"][:nv] = vertices * bbox_scaled
    target["edge_face"][:ne] = edge_face.astype(np.int16)
    target["edge_vert"][:ne] = edge_vert.astype(np.int16)
    return {"uid": uid, "num_faces": nf, "num_edges": ne, "num_vertices": nv, "prior": prior, "target": target}


def _worker(payload: Tuple[str, str, Dict[str, Any], Dict[str, int]]) -> Tuple[str, Optional[Dict[str, Any]], str]:
    uid, path, compact, limits = payload
    try:
        return uid, _pack_record(uid, path, compact, limits), ""
    except Exception as exc:
        return uid, None, "%s: %s" % (type(exc).__name__, exc)


def _write_row(handle: h5py.File, row: int, packed: Mapping[str, Any]) -> None:
    expected_uid = handle["meta/uid"][row]
    expected_uid = expected_uid.decode("ascii") if isinstance(expected_uid, bytes) else str(expected_uid)
    if expected_uid != packed["uid"]:
        raise AssertionError("row UID mismatch: %s != %s" % (expected_uid, packed["uid"]))
    for key, value in packed["prior"].items():
        handle["prior"][key][row] = value
    for key, value in packed["target"].items():
        handle["target"][key][row] = value
    handle["meta/num_faces"][row] = packed["num_faces"]
    handle["meta/num_edges"][row] = packed["num_edges"]
    handle["meta/num_vertices"][row] = packed["num_vertices"]
    handle["meta/written"][row] = True


def _append_rejection(path: Path, row: Mapping[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["uid", "split", "reason"])
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _iter_compact(path: Path, eligible: set) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if str(record.get("uid", "")) in eligible:
                yield record


def _continuous_stats(path: Path, keys: Sequence[str]) -> Dict[str, Dict[str, List[float]]]:
    result = {}
    with h5py.File(path, "r") as handle:
        written = np.asarray(handle["meta/written"], dtype=bool)
        mask = np.asarray(handle["prior/face_mask"])[written].astype(bool)
        for key in keys:
            values = np.asarray(handle["prior/%s" % key])[written].astype(np.float64)
            flat = values[mask]
            mean = np.mean(flat, axis=0) if len(flat) else np.zeros(values.shape[-1])
            std = np.std(flat, axis=0) if len(flat) else np.ones(values.shape[-1])
            result[key] = {
                "mean": mean.astype(float).tolist(),
                "std": np.maximum(std, 1e-6).astype(float).tolist(),
            }
    return result


def _prior_statistics(path: Path) -> Dict[str, Any]:
    motif = np.zeros(3, dtype=np.int64)
    relation = np.zeros(3, dtype=np.int64)
    surface = np.zeros(SURFACE_COUNT, dtype=np.int64)
    face_histogram = Counter()
    with h5py.File(path, "r") as handle:
        written = np.flatnonzero(np.asarray(handle["meta/written"], dtype=bool))
        for start in range(0, len(written), 512):
            rows = written[start : start + 512]
            counts = np.asarray(handle["prior/motif_counts"][rows], dtype=np.int64)
            motif += counts.sum(axis=0)
            stored_relations = np.asarray(handle["prior/relation_counts"][rows], dtype=np.int64)
            relation[1:] += stored_relations.sum(axis=0)
            membership = np.asarray(handle["prior/motif_membership"][rows], dtype=np.float32)
            mask = np.asarray(handle["prior/face_mask"][rows], dtype=bool)
            relation[0] += int(np.sum((membership > 0) & mask[..., None]))
            types = np.asarray(handle["prior/surface_type"][rows], dtype=np.int64)
            surface += np.bincount(types[mask], minlength=SURFACE_COUNT)[:SURFACE_COUNT]
            for value in np.asarray(handle["meta/num_faces"][rows], dtype=np.int64):
                face_histogram[int(value)] += 1
    return {
        "motif_instances": dict(zip(MOTIF_TYPES, motif.astype(int).tolist())),
        "relation_instances": dict(
            zip(("embedded_in", *RELATION_TYPES), relation.astype(int).tolist())
        ),
        "surface_faces": dict(
            zip(
                ("plane", "cylinder", "cone", "sphere", "torus", "freeform_or_other"),
                surface.astype(int).tolist(),
            )
        ),
        "face_count_histogram": {
            str(key): int(value) for key, value in sorted(face_histogram.items())
        },
    }


def _repair_layout_directions(path: Path) -> int:
    """Backfill cylinder axes into old/resumed layout rows without reparsing STEP."""
    repaired = 0
    with h5py.File(path, "r+") as handle:
        written = np.flatnonzero(np.asarray(handle["meta/written"], dtype=bool))
        for start in range(0, len(written), 512):
            rows = written[start : start + 512]
            bbox_cont = np.asarray(handle["prior/face_bbox_cont"][rows], dtype=np.float32)
            geom_cont = np.asarray(handle["prior/face_geom_cont"][rows], dtype=np.float32)
            face_mask = np.asarray(handle["prior/face_mask"][rows], dtype=bool)
            current = bbox_cont[..., 2:5]
            cylinder_axis = geom_cont[..., 8:11]
            use_axis = (
                face_mask
                & (np.linalg.norm(current, axis=-1) <= 1e-6)
                & (np.linalg.norm(cylinder_axis, axis=-1) > 1e-6)
            )
            if np.any(use_axis):
                current[use_axis] = cylinder_axis[use_axis]
                bbox_cont[..., 2:5] = current
                handle["prior/face_bbox_cont"][rows] = bbox_cont.astype(np.float16)
                repaired += int(use_axis.sum())
        handle.flush()
    return repaired


def _coarsen_layout_bins(path: Path) -> int:
    """Project resumed 4-bit layout rows to the final 2-bit prior alphabet."""
    changed = 0
    with h5py.File(path, "r+") as handle:
        written = np.flatnonzero(np.asarray(handle["meta/written"], dtype=bool))
        for start in range(0, len(written), 512):
            rows = written[start : start + 512]
            values = np.asarray(handle["prior/face_bbox_bins"][rows], dtype=np.uint8)
            coarse = (np.rint(values.astype(np.float32) / 5.0) * 5.0).astype(np.uint8)
            difference = values != coarse
            if np.any(difference):
                handle["prior/face_bbox_bins"][rows] = coarse
                changed += int(difference.sum())
        handle.flush()
    return changed


def build(args: argparse.Namespace) -> Dict[str, Any]:
    config = _load_config(args.config)
    data_config = dict(config["data"])
    eligible_path = args.eligible_index or _resolve(config["paths"]["eligible_index"])
    compact_path = args.compact_prior or _resolve(config["paths"]["compact_prior"])
    step_root = args.step_root or _resolve(config["paths"]["step_root"])
    output_dir = args.output_dir or _resolve(config["paths"]["data_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, source_counts = _read_eligible(eligible_path, args.limit)
    by_split = {split: [row for row in rows if row["split"] == split] for split in SPLIT_NAMES}
    locations: Dict[str, Tuple[str, int]] = {}
    handles: Dict[str, h5py.File] = {}
    try:
        for split, split_rows in by_split.items():
            path = output_dir / ("%s.h5" % split)
            _initialize_h5(path, split_rows, data_config)
            handles[split] = h5py.File(path, "r+", libver="latest")
            handles[split].attrs["split"] = split
            handles[split]["meta"].attrs["original_dtg_split"] = split
            if "rejected" not in handles[split]["meta"]:
                handles[split]["meta"].create_dataset(
                    "rejected",
                    shape=(len(split_rows),),
                    dtype=np.bool_,
                    fillvalue=False,
                )
            for index, row in enumerate(split_rows):
                locations[row["uid"]] = (split, index)

        pending = {
            uid
            for uid, (split, index) in locations.items()
            if not bool(handles[split]["meta/written"][index])
        }
        rejected_path = output_dir / "rejected.csv"
        if not rejected_path.exists():
            with rejected_path.open("w", encoding="utf-8", newline="") as handle:
                csv.DictWriter(handle, fieldnames=["uid", "split", "reason"]).writeheader()
        rejected_before = set()
        if rejected_path.exists():
            with rejected_path.open("r", encoding="utf-8", newline="") as handle:
                rejected_before = {str(row["uid"]) for row in csv.DictReader(handle)}
        for uid in rejected_before:
            if uid in locations:
                split, row = locations[uid]
                handles[split]["meta/rejected"][row] = True
        pending -= rejected_before
        if not pending:
            print("all requested rows are already written or rejected", flush=True)

        processed = 0
        accepted = 0
        rejected = 0
        missing_step = 0
        submitted = {}
        workers = max(1, int(args.workers))
        max_pending = max(workers * 2, 1)

        def consume(result):
            nonlocal processed, accepted, rejected
            uid, packed, reason = result
            split, row = locations[uid]
            processed += 1
            if packed is None:
                _append_rejection(rejected_path, {"uid": uid, "split": split, "reason": reason})
                handles[split]["meta/rejected"][row] = True
                rejected += 1
            else:
                _write_row(handles[split], row, packed)
                accepted += 1
            if processed % max(1, int(args.flush_every)) == 0:
                for handle in handles.values():
                    handle.flush()
                print(
                    "processed=%d accepted=%d rejected=%d remaining=%d"
                    % (processed, accepted, rejected, max(0, len(pending) - processed)),
                    flush=True,
                )

        executor = ProcessPoolExecutor(max_workers=workers)
        try:
            for compact in _iter_compact(compact_path, pending):
                uid = str(compact["uid"])
                step_rel = str(compact.get("source_step_rel", "cad_step/%s.step" % uid))
                step_path = step_root / Path(step_rel).name
                if not step_path.is_file():
                    split, _ = locations[uid]
                    _append_rejection(
                        rejected_path,
                        {"uid": uid, "split": split, "reason": "missing_step:%s" % step_path},
                    )
                    handles[split]["meta/rejected"][locations[uid][1]] = True
                    missing_step += 1
                    processed += 1
                    continue
                payload = (uid, str(step_path), compact, data_config)
                while len(submitted) >= max_pending:
                    done, _ = wait(submitted, return_when=FIRST_COMPLETED)
                    for future in done:
                        fallback_payload = submitted.pop(future)
                        try:
                            consume(future.result())
                        except BrokenProcessPool:
                            # A native OCC crash invalidates the pool.  Record only
                            # the affected in-flight UIDs, then resume safely later.
                            for other, item in list(submitted.items()):
                                other.cancel()
                                split, _ = locations[item[0]]
                                _append_rejection(
                                    rejected_path,
                                    {"uid": item[0], "split": split, "reason": "native_worker_crash"},
                                )
                                handles[split]["meta/rejected"][locations[item[0]][1]] = True
                                rejected += 1
                                processed += 1
                            submitted.clear()
                            split, _ = locations[fallback_payload[0]]
                            _append_rejection(
                                rejected_path,
                                {"uid": fallback_payload[0], "split": split, "reason": "native_worker_crash"},
                            )
                            handles[split]["meta/rejected"][locations[fallback_payload[0]][1]] = True
                            rejected += 1
                            processed += 1
                            executor.shutdown(wait=False)
                            executor = ProcessPoolExecutor(max_workers=workers)
                            break
                    else:
                        continue
                    break
                submitted[executor.submit(_worker, payload)] = payload

            while submitted:
                done, _ = wait(submitted, return_when=FIRST_COMPLETED)
                for future in done:
                    payload = submitted.pop(future)
                    try:
                        consume(future.result())
                    except BrokenProcessPool:
                        affected = [payload] + list(submitted.values())
                        for item in affected:
                            split, _ = locations[item[0]]
                            _append_rejection(
                                rejected_path,
                                {"uid": item[0], "split": split, "reason": "native_worker_crash"},
                            )
                            handles[split]["meta/rejected"][locations[item[0]][1]] = True
                            rejected += 1
                            processed += 1
                        submitted.clear()
                        break
        finally:
            executor.shutdown(wait=True)
        for handle in handles.values():
            handle.flush()
    finally:
        for handle in handles.values():
            handle.close()

    layout_axis_repairs = {
        split: _repair_layout_directions(output_dir / ("%s.h5" % split))
        for split in SPLIT_NAMES
    }
    layout_bin_repairs = {
        split: _coarsen_layout_bins(output_dir / ("%s.h5" % split))
        for split in SPLIT_NAMES
    }
    audits = {split: inspect_hdf5(output_dir / ("%s.h5" % split)) for split in SPLIT_NAMES}
    for split, audit in audits.items():
        if audit["capacity"] != int(source_counts.get(split, 0)):
            raise AssertionError("%s HDF5 capacity does not match eligible split" % split)
        if audit["original_dtg_split"] != split:
            raise AssertionError("%s HDF5 lost its original DTG split label" % split)
        if not audit["complete"]:
            raise AssertionError("%s HDF5 still contains unfinished rows" % split)
        if audit["duplicate_uids"] or audit["finite_failures"]:
            raise AssertionError("%s HDF5 failed UID/finite-value audit" % split)
        if not set(audit["layout_prior_alphabet"]).issubset({0, 5, 10, 15}):
            raise AssertionError("%s HDF5 contains non-coarse bbox prior values" % split)
    normalization = _continuous_stats(
        output_dir / "train.h5",
        ("face_edge_cont", "face_bbox_cont", "face_geom_cont"),
    )
    prior_statistics = {
        split: _prior_statistics(output_dir / ("%s.h5" % split))
        for split in SPLIT_NAMES
    }
    summary = {
        "schema_version": "innovation2_stagewise_dataset_summary_v1",
        "created_at_unix": time.time(),
        "eligible_index": str(eligible_path),
        "compact_prior": str(compact_path),
        "step_root": str(step_root),
        "eligible_records": len(rows),
        "source_split_counts": source_counts,
        "limits": data_config,
        "formal_motif_types": list(MOTIF_TYPES),
        "formal_relation_types": ["embedded_in", *RELATION_TYPES],
        "surface_types": config["data"]["surface_types"],
        "normalization": normalization,
        "prior_statistics": prior_statistics,
        "splits": audits,
        "rejected_csv": str(output_dir / "rejected.csv"),
        "target_leakage_check": "passed",
        "layout_prior_quantization": "2-bit values stored as 0/5/10/15; exact bbox is target-only",
        "layout_direction_axis_repairs": layout_axis_repairs,
        "layout_bin_repairs": layout_bin_repairs,
    }
    summary_path = output_dir / "dataset_summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, summary_path)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "config.yaml")
    parser.add_argument("--eligible-index", type=Path, default=None)
    parser.add_argument("--compact-prior", type=Path, default=None)
    parser.add_argument("--step-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--flush-every", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="build only the first N eligible records for interface tests")
    return parser


def main() -> int:
    result = build(_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
