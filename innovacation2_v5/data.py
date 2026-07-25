"""HDF5 datasets, augmentation and leakage checks for the three adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


PRIOR_KEYS = {
    "surface_type",
    "surface_confidence",
    "motif_membership",
    "motif_confidence",
    "motif_instance",
    "motif_counts",
    "relation_counts",
    "pair_relations",
    "face_edge_cont",
    "face_bbox_bins",
    "face_bbox_cont",
    "face_geom_cont",
    "face_mask",
}
TARGET_KEYS = {
    "fef_adj",
    "face_bbox",
    "face_ctrl",
    "edge_ctrl",
    "vert_coords",
    "edge_face",
    "edge_vert",
    "face_edge",
    "face_edge_count",
    "vert_face",
    "vert_face_count",
    "edge_mask",
    "vert_mask",
}
FORBIDDEN_PRIOR_TOKENS = (
    "fef",
    "adj",
    "target",
    "control_point",
    "face_ctrl",
    "edge_ctrl",
    "vert_coords",
)
STAGE_TARGET_KEYS = {
    "face_edge": {"fef_adj"},
    "face_bbox": {"fef_adj", "face_bbox"},
    "face_geom": {
        "face_bbox",
        "face_ctrl",
        "edge_ctrl",
        "vert_coords",
        "edge_vert",
        "face_edge",
        "face_edge_count",
    },
}
INTEGER_TARGET_KEYS = {
    "fef_adj",
    "edge_face",
    "edge_vert",
    "face_edge",
    "face_edge_count",
    "vert_face",
    "vert_face_count",
}


def assert_no_target_leakage(path: Path) -> None:
    with h5py.File(path, "r") as handle:
        if set(handle.keys()) != {"meta", "prior", "target"}:
            raise AssertionError("%s must contain exactly meta/prior/target groups" % path)
        prior_keys = set(handle["prior"].keys())
        target_keys = set(handle["target"].keys())
        unknown = prior_keys - PRIOR_KEYS
        if unknown:
            raise AssertionError("unknown prior fields: %s" % sorted(unknown))
        missing = TARGET_KEYS - target_keys
        if missing:
            raise AssertionError("missing target fields: %s" % sorted(missing))
        for key in prior_keys:
            lowered = key.lower()
            if any(token in lowered for token in FORBIDDEN_PRIOR_TOKENS):
                raise AssertionError("target-like field leaked into prior: %s" % key)


def _decode_uid(value) -> str:
    if isinstance(value, bytes):
        return value.decode("ascii")
    return str(value)


class StagewiseH5Dataset(Dataset):
    """Lazy per-process HDF5 reader safe with DataLoader workers."""

    def __init__(
        self,
        path: Path,
        stage: str,
        training: bool = False,
        condition_dropout: float = 0.0,
        bbox_bins: int = 16,
        bbox_prior_levels: int = 4,
    ):
        self.path = Path(path)
        self.stage = stage
        if self.stage not in STAGE_TARGET_KEYS:
            raise ValueError("stage must be face_edge, face_bbox or face_geom")
        self.training = bool(training)
        self.condition_dropout = float(condition_dropout)
        self.bbox_bins = int(bbox_bins)
        self.bbox_prior_levels = int(bbox_prior_levels)
        self._handle: Optional[h5py.File] = None
        summary_path = self.path.parent / "dataset_summary.json"
        self.normalization = {}
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as summary_handle:
                self.normalization = json.load(summary_handle).get("normalization", {})
        with h5py.File(self.path, "r") as handle:
            written = np.asarray(handle["meta/written"], dtype=bool)
            self.indices = np.flatnonzero(written).astype(np.int64)
        if not len(self.indices):
            raise ValueError("HDF5 split contains no completed rows: %s" % self.path)

    def __len__(self) -> int:
        return int(len(self.indices))

    def _h5(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r", swmr=True)
        return self._handle

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handle"] = None
        return state

    def __del__(self):
        self.close()

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass
            self._handle = None

    def __getitem__(self, item: int) -> Dict[str, object]:
        row = int(self.indices[item])
        handle = self._h5()
        prior = {
            key: torch.from_numpy(np.asarray(dataset[row]))
            for key, dataset in handle["prior"].items()
        }
        target = {
            key: torch.from_numpy(np.asarray(handle["target"][key][row]))
            for key in STAGE_TARGET_KEYS[self.stage]
        }
        for key in INTEGER_TARGET_KEYS & target.keys():
            target[key] = target[key].long()
        prior["face_mask"] = prior["face_mask"].bool()

        bins = prior.pop("face_bbox_bins").to(torch.float32)
        if self.training:
            coarse_step = max(
                (self.bbox_bins - 1) / max(self.bbox_prior_levels - 1, 1),
                1.0,
            )
            bins = bins + torch.empty_like(bins).uniform_(
                -0.5 * coarse_step,
                0.5 * coarse_step,
            )
        bins = (bins / max(self.bbox_bins - 1, 1)).clamp(0.0, 1.0)

        for key in ("motif_membership", "motif_confidence", "pair_relations", "face_edge_cont", "face_geom_cont"):
            prior[key] = prior[key].float()
        prior["motif_instance"] = prior["motif_instance"].long()
        prior["motif_counts"] = prior["motif_counts"].long()
        prior["relation_counts"] = prior["relation_counts"].long()
        for key in ("face_edge_cont", "face_bbox_cont", "face_geom_cont"):
            stats = self.normalization.get(key)
            if stats:
                mean = torch.tensor(stats["mean"], dtype=torch.float32)
                std = torch.tensor(stats["std"], dtype=torch.float32).clamp_min(1e-6)
                prior[key] = (prior[key].float() - mean) / std
        prior["face_bbox_cont"] = torch.cat([bins, prior["face_bbox_cont"].float()], dim=-1)
        prior["surface_type"] = prior["surface_type"].long()

        if self.training and self.condition_dropout > 0.0:
            drop_all = bool(torch.rand(()) < self.condition_dropout)
            if drop_all:
                prior["surface_type"].fill_(6)
                prior["surface_confidence"].zero_()
                prior["motif_membership"].zero_()
                prior["motif_confidence"].zero_()
                prior["motif_instance"].zero_()
                prior["motif_counts"].zero_()
                prior["relation_counts"].zero_()
                prior["pair_relations"].zero_()
                for key in ("face_edge_cont", "face_bbox_cont", "face_geom_cont"):
                    prior[key].zero_()
            else:
                for key in ("face_edge_cont", "face_bbox_cont", "face_geom_cont"):
                    keep = torch.rand(prior[key].shape[-1]) >= self.condition_dropout
                    prior[key] *= keep.to(prior[key].dtype)

        item = {
            "row": row,
            "uid": _decode_uid(handle["meta/uid"][row]),
            "prior": prior,
            "target": target,
            "num_faces": int(handle["meta/num_faces"][row]),
            "num_edges": int(handle["meta/num_edges"][row]),
            "num_vertices": int(handle["meta/num_vertices"][row]),
        }
        if self.stage == "face_edge":
            item = sort_face_edge_slots(item)
        return item


def collate_stagewise(items: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "row": torch.tensor([int(item["row"]) for item in items], dtype=torch.long),
        "uid": [str(item["uid"]) for item in items],
        "num_faces": torch.tensor([int(item["num_faces"]) for item in items], dtype=torch.long),
        "num_edges": torch.tensor([int(item["num_edges"]) for item in items], dtype=torch.long),
        "num_vertices": torch.tensor([int(item["num_vertices"]) for item in items], dtype=torch.long),
        "prior": {
            key: torch.stack([item["prior"][key] for item in items])
            for key in items[0]["prior"]
        },
        "target": {
            key: torch.stack([item["target"][key] for item in items])
            for key in items[0]["target"]
        },
    }


def permute_prior_slots(
    prior: Dict[str, torch.Tensor],
    permutation: torch.Tensor,
    count: int,
) -> Dict[str, torch.Tensor]:
    """Synchronously permute the valid face slots of one prior."""
    permutation = permutation.long()
    if permutation.shape != (count,) or set(permutation.tolist()) != set(range(count)):
        raise ValueError("permutation must contain each valid face exactly once")
    full = torch.arange(prior["face_mask"].shape[0], dtype=torch.long)
    full[:count] = permutation.cpu()
    result = {key: value.clone() for key, value in prior.items()}
    face_keys = (
        "surface_type",
        "surface_confidence",
        "motif_membership",
        "motif_confidence",
        "motif_instance",
        "face_edge_cont",
        "face_bbox_cont",
        "face_geom_cont",
        "face_mask",
    )
    for key in face_keys:
        if key in prior:
            result[key] = prior[key][full]
    if "pair_relations" in prior:
        result["pair_relations"] = prior["pair_relations"][full][:, full]
    return result


def sort_face_edge_slots(item: Dict[str, object]) -> Dict[str, object]:
    """Match DTG FaceEdge/EdgeVert's stable ascending face-degree convention."""
    count = int(item["num_faces"])
    fef = item["target"]["fef_adj"]
    degrees = fef[:count, :count].sum(dim=1).cpu().numpy()
    permutation = torch.from_numpy(np.argsort(degrees, kind="stable").astype(np.int64))
    result = dict(item)
    result["prior"] = permute_prior_slots(item["prior"], permutation, count)
    full = torch.arange(fef.shape[0], dtype=torch.long)
    full[:count] = permutation
    result["target"] = dict(item["target"])
    result["target"]["fef_adj"] = fef[full][:, full]
    return result


def permute_face_slots(item: Dict[str, object], permutation: torch.Tensor) -> Dict[str, object]:
    """Apply one face permutation to every aligned prior/target field.

    This is used by tests and optional augmentation.  ``permutation`` gives
    new-to-old indices for the valid faces; padding remains after valid faces.
    """
    count = int(item["num_faces"])
    permutation = permutation.long()
    if permutation.shape != (count,) or set(permutation.tolist()) != set(range(count)):
        raise ValueError("permutation must contain each valid face exactly once")
    full = torch.arange(item["prior"]["face_mask"].shape[0], dtype=torch.long)
    full[:count] = permutation
    old_to_new = torch.arange(len(full), dtype=torch.long)
    old_to_new[permutation] = torch.arange(count, dtype=torch.long)
    result = {
        **item,
        "prior": {key: value.clone() for key, value in item["prior"].items()},
        "target": {key: value.clone() for key, value in item["target"].items()},
    }
    for key in (
        "surface_type",
        "surface_confidence",
        "motif_membership",
        "motif_confidence",
        "motif_instance",
        "face_edge_cont",
        "face_bbox_cont",
        "face_geom_cont",
        "face_mask",
    ):
        result["prior"][key] = item["prior"][key][full]
    result["prior"]["pair_relations"] = item["prior"]["pair_relations"][full][:, full]
    for key in ("face_bbox", "face_ctrl", "face_edge", "face_edge_count"):
        result["target"][key] = item["target"][key][full]
    result["target"]["fef_adj"] = item["target"]["fef_adj"][full][:, full]
    edge_face = result["target"]["edge_face"]
    valid_edge_face = edge_face >= 0
    edge_face[valid_edge_face] = old_to_new[edge_face[valid_edge_face]]
    vert_face = result["target"]["vert_face"]
    valid_vert_face = vert_face >= 0
    vert_face[valid_vert_face] = old_to_new[vert_face[valid_vert_face]]
    return result


def to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    result = dict(batch)
    result["row"] = batch["row"].to(device)
    result["num_faces"] = batch["num_faces"].to(device)
    result["num_edges"] = batch["num_edges"].to(device)
    result["num_vertices"] = batch["num_vertices"].to(device)
    result["prior"] = {key: value.to(device) for key, value in batch["prior"].items()}
    result["target"] = {key: value.to(device) for key, value in batch["target"].items()}
    return result


def inspect_hdf5(path: Path) -> Dict[str, object]:
    assert_no_target_leakage(path)
    with h5py.File(path, "r") as handle:
        written = np.asarray(handle["meta/written"], dtype=bool)
        rejected = (
            np.asarray(handle["meta/rejected"], dtype=bool)
            if "rejected" in handle["meta"]
            else np.zeros_like(written)
        )
        uids = [_decode_uid(value) for value in handle["meta/uid"][written]]
        layout_values = np.asarray(handle["prior/face_bbox_bins"])[written].astype(np.uint8)
        layout_mask = np.asarray(handle["prior/face_mask"])[written].astype(bool)
        layout_alphabet = (
            sorted(int(value) for value in np.unique(layout_values[layout_mask]))
            if np.any(layout_mask)
            else []
        )
        finite_failures: List[str] = []
        for group_name in ("prior", "target"):
            for key, dataset in handle[group_name].items():
                if np.issubdtype(dataset.dtype, np.floating):
                    indices = np.flatnonzero(written)
                    for start in range(0, len(indices), 512):
                        values = np.asarray(dataset[indices[start : start + 512]])
                        if not np.isfinite(values).all():
                            finite_failures.append("%s/%s" % (group_name, key))
                            break
        return {
            "path": str(path),
            "original_dtg_split": str(handle["meta"].attrs.get("original_dtg_split", "")),
            "capacity": int(len(written)),
            "written": int(written.sum()),
            "rejected": int(rejected.sum()),
            "unique_uids": int(len(set(uids))),
            "duplicate_uids": int(len(uids) - len(set(uids))),
            "finite_failures": sorted(set(finite_failures)),
            "layout_prior_alphabet": layout_alphabet,
            "complete": bool(np.all(written | rejected)),
        }


def load_summary(data_dir: Path) -> Mapping[str, object]:
    with (Path(data_dir) / "dataset_summary.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)
