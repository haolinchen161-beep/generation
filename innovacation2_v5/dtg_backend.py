"""Read-only DTGBrepGen backend with explicit stage-wise residual hooks."""

from __future__ import annotations

import gc
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from diffusers import DDPMScheduler, PNDMScheduler

from inference.generate import get_edgeGeom, get_faceBbox, get_faceGeom, get_vertGeom
from model import (
    EdgeGeomTransformer,
    EdgeVertModel,
    FaceBboxTransformer,
    FaceEdgeModel,
    FaceGeomTransformer,
    VertGeomTransformer,
)
from topology.topoGenerate import SeqGenerator
from topology.transfer import faceVert_from_edgeVert, face_vert_trans
from utils import calculate_y, sort_bbox_multi


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints_base" / "deepcad"


class _SeqAdapter:
    def __init__(self, model: EdgeVertModel):
        self.model = model

    def sample(self, topo_seq, seq_mask, mask, class_label, point_data=None):
        return self.model.sample(topo_seq, seq_mask, mask, class_label, point_data=None)

    def parameters(self):
        return self.model.parameters()


def checkpoint_checksums() -> Dict[str, str]:
    result = {}
    for path in sorted(CHECKPOINT_ROOT.rglob("*.pt")):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result[str(path.relative_to(PROJECT_ROOT))] = digest.hexdigest()
    return result


class DTGBackend:
    """Load one large frozen stage at a time to fit a 4-GB GPU."""

    def __init__(self, device: str = "cuda"):
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for full DTG generation")
        self.device = torch.device(device)
        with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)["deepcad"]

    def _load_state(self, model: torch.nn.Module, relative_path: str) -> torch.nn.Module:
        state = torch.load(str(CHECKPOINT_ROOT / relative_path), map_location="cpu")
        model.load_state_dict(state, strict=True)
        model.requires_grad_(False)
        return model.to(self.device).eval()

    @staticmethod
    def release(model: torch.nn.Module) -> None:
        model.to("cpu")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_face_edge(self) -> FaceEdgeModel:
        network = self.config["FaceEdgeModel"]
        return self._load_state(
            FaceEdgeModel(
                nf=self.config["max_face"],
                d_model=network["d_model"],
                nhead=network["nhead"],
                n_layers=network["n_layers"],
                num_categories=self.config["edge_classes"],
                use_cf=False,
                use_pc=False,
            ),
            "topo_faceEdge/epoch_2000.pt",
        )

    def load_edge_vert(self) -> EdgeVertModel:
        network = self.config["EdgeVertModel"]
        return self._load_state(
            EdgeVertModel(
                max_num_edge=self.config["max_num_edge_topo"],
                max_seq_length=self.config["max_seq_length"],
                edge_classes=self.config["edge_classes"],
                max_face=self.config["max_face"],
                max_edge=self.config["max_edge"],
                d_model=network["d_model"],
                n_layers=network["n_layers"],
                use_cf=False,
                use_pc=False,
            ),
            "topo_edgeVert/epoch_1000.pt",
        )

    def load_face_bbox(self) -> FaceBboxTransformer:
        network = self.config["FaceBboxModel"]
        return self._load_state(
            FaceBboxTransformer(
                n_layers=network["n_layers"],
                hidden_mlp_dims=network["hidden_mlp_dims"],
                hidden_dims=network["hidden_dims"],
                edge_classes=self.config["edge_classes"],
                act_fn_in=torch.nn.ReLU(),
                act_fn_out=torch.nn.ReLU(),
                use_cf=False,
                use_pc=False,
            ),
            "geom_faceBbox/epoch_3000.pt",
        )

    def load_vert_geom(self) -> VertGeomTransformer:
        network = self.config["VertGeomModel"]
        return self._load_state(
            VertGeomTransformer(
                n_layers=network["n_layers"],
                hidden_mlp_dims=network["hidden_mlp_dims"],
                hidden_dims=network["hidden_dims"],
                act_fn_in=torch.nn.ReLU(),
                act_fn_out=torch.nn.ReLU(),
                use_cf=False,
                use_pc=False,
            ),
            "geom_vertGeom/epoch_3000.pt",
        )

    def load_edge_geom(self) -> EdgeGeomTransformer:
        network = self.config["EdgeGeomModel"]
        return self._load_state(
            EdgeGeomTransformer(
                n_layers=network["n_layers"],
                edge_geom_dim=network["edge_geom_dim"],
                d_model=network["d_model"],
                nhead=network["nhead"],
                use_cf=False,
                use_pc=False,
            ),
            "geom_edgeGeom/epoch_3000.pt",
        )

    def load_face_geom(self) -> FaceGeomTransformer:
        network = self.config["FaceGeomModel"]
        return self._load_state(
            FaceGeomTransformer(
                n_layers=network["n_layers"],
                face_geom_dim=network["face_geom_dim"],
                d_model=network["d_model"],
                nhead=network["nhead"],
                use_cf=False,
                use_pc=False,
            ),
            "geom_faceGeom/epoch_3000.pt",
        )

    @staticmethod
    def schedulers() -> Tuple[PNDMScheduler, DDPMScheduler]:
        pndm = PNDMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            prediction_type="epsilon",
            beta_start=0.0001,
            beta_end=0.02,
        )
        ddpm = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            prediction_type="epsilon",
            beta_start=0.0001,
            beta_end=0.02,
            clip_sample=True,
            clip_sample_range=3,
        )
        return pndm, ddpm

    @torch.no_grad()
    def sample_fef_baseline(self, model: FaceEdgeModel) -> np.ndarray:
        matrix = model.sample(num_samples=1, class_label=None, point_data=None)[0]
        active = torch.any(matrix != 0, dim=1)
        matrix = matrix[active][:, active]
        if matrix.numel() == 0:
            raise RuntimeError("DTG FaceEdge generated an empty topology")
        degree = matrix.sum(dim=1)
        order = torch.argsort(degree)
        return matrix[order][:, order].cpu().numpy().astype(np.int64)

    @torch.no_grad()
    def sample_fef_guided(
        self,
        model: FaceEdgeModel,
        adapter: FaceEdgeAdapter,
        prior: Dict[str, torch.Tensor],
        gate: float = 1.0,
    ) -> np.ndarray:
        if float(gate) == 0.0:
            return self.sample_fef_baseline(model)
        face_count = int(prior["face_mask"].sum().item())
        if not 1 <= face_count <= int(self.config["max_face"]):
            raise ValueError("prior face count is outside DTG range")
        batch_prior = {
            key: (value.unsqueeze(0) if value.ndim > 0 else value.reshape(1))
            .to(self.device)
            for key, value in prior.items()
        }
        z = torch.randn(1, model.d_model, device=self.device)
        generated = torch.full(
            (1, 1),
            model.num_categories,
            dtype=torch.long,
            device=self.device,
        )
        pair_index = torch.triu_indices(model.nf, model.nf, offset=1, device=self.device)
        for position in range(model.seq_len):
            base_logits = model.decode(z, generated, None, None)[:, -1:, :]
            pair = pair_index[:, position : position + 1]
            i, j = int(pair[0, 0]), int(pair[1, 0])
            if i >= face_count or j >= face_count:
                next_token = torch.zeros(1, dtype=torch.long, device=self.device)
            else:
                delta = adapter(batch_prior, base_logits, pair)
                logits = base_logits + float(gate) * delta
                next_token = torch.distributions.Categorical(logits=logits[:, 0]).sample()
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=1)
        matrix = model.sequence_to_matrix(generated[:, 1:])[0, :face_count, :face_count]
        active = torch.any(matrix != 0, dim=1)
        matrix = matrix[active][:, active]
        if not torch.any(matrix != 0):
            raise RuntimeError("guided FaceEdge generated an empty topology")
        return matrix.cpu().numpy().astype(np.int64)

    def complete_edge_vertex(self, fef: np.ndarray, attempts: int = 10) -> Dict[str, Any]:
        fef_tensor = torch.as_tensor(fef, dtype=torch.long, device=self.device)
        if torch.any(fef_tensor.sum(dim=1) == 0):
            raise ValueError("FaceEdge produced one or more isolated real-face slots")
        indices = torch.triu(fef_tensor, diagonal=1).nonzero(as_tuple=False)
        if not len(indices):
            raise ValueError("face adjacency contains no shared edges")
        multiplicity = fef_tensor[indices[:, 0], indices[:, 1]]
        edge_face = indices.repeat_interleave(multiplicity, dim=0)
        model = self.load_edge_vert()
        if fef_tensor.sum(dim=1).max() > model.max_edge:
            raise ValueError(
                "face degree %d exceeds per-face edge limit %d"
                % (fef_tensor.sum(dim=1).max().item(), model.max_edge)
            )
        try:
            if edge_face.shape[0] > model.max_num_edge:
                raise ValueError(
                    "topology has %d edges, limit is %d" % (edge_face.shape[0], model.max_num_edge)
                )
            share_id = calculate_y(edge_face)
            model.save_cache(
                edgeFace_adj=edge_face.unsqueeze(0),
                edge_mask=torch.ones((1, edge_face.shape[0]), device=self.device, dtype=torch.bool),
                share_id=share_id,
                class_label=None,
                point_data=None,
            )
            generator = None
            adapter = _SeqAdapter(model)
            for _ in range(max(1, int(attempts))):
                candidate = SeqGenerator(edge_face.cpu().numpy())
                if candidate.generate(adapter, class_label=None):
                    generator = candidate
                    break
            model.clear_cache()
            if generator is None:
                raise RuntimeError("DTG EdgeVert failed to complete closed face loops")
            edge_vert = torch.as_tensor(generator.edgeVert_adj, dtype=torch.long, device=self.device)
            face_edge = generator.faceEdge_adj
            face_vert = faceVert_from_edgeVert(face_edge, generator.edgeVert_adj)
            vert_face = face_vert_trans(faceVert_adj=face_vert)
            return {
                "fef_adj": fef_tensor,
                "edgeFace_adj": edge_face,
                "edgeVert_adj": edge_vert,
                "faceEdge_adj": face_edge,
                "vertFace_adj": vert_face,
            }
        finally:
            self.release(model)

    def generate_geometry(
        self,
        topology: Dict[str, Any],
        prior: Optional[Dict[str, torch.Tensor]] = None,
        prior_bbox_model: Optional[torch.nn.Module] = None,
    ) -> Dict[str, np.ndarray]:
        fef_list = [topology["fef_adj"]]
        edge_face_list = [topology["edgeFace_adj"]]
        edge_vert_list = [topology["edgeVert_adj"]]
        face_edge_list = [topology["faceEdge_adj"]]
        vert_face_list = [topology["vertFace_adj"]]

        pndm, ddpm = self.schedulers()
        model = prior_bbox_model if prior_bbox_model is not None else self.load_face_bbox()
        face_bbox_tensor, face_mask = get_faceBbox(fef_list, model, pndm, ddpm, None, None)
        face_bbox = [sort_bbox_multi(values[mask]) for values, mask in zip(face_bbox_tensor, face_mask)]
        if prior_bbox_model is None:
            self.release(model)

        pndm, ddpm = self.schedulers()
        model = self.load_vert_geom()
        vert_tensor, vert_mask = get_vertGeom(
            face_bbox, vert_face_list, edge_vert_list, model, pndm, ddpm, None, None
        )
        vert_geom = [values[mask] for values, mask in zip(vert_tensor, vert_mask)]
        self.release(model)

        pndm, ddpm = self.schedulers()
        model = self.load_edge_geom()
        edge_tensor, edge_mask = get_edgeGeom(
            face_bbox,
            vert_geom,
            edge_face_list,
            edge_vert_list,
            model,
            pndm,
            ddpm,
            None,
            None,
        )
        edge_geom = [values[mask] for values, mask in zip(edge_tensor, edge_mask)]
        self.release(model)

        pndm, ddpm = self.schedulers()
        model = self.load_face_geom()
        face_tensor, generated_face_mask = get_faceGeom(
            face_bbox,
            vert_geom,
            edge_geom,
            face_edge_list,
            edge_face_list,
            edge_vert_list,
            model,
            pndm,
            ddpm,
            None,
            None,
        )
        face_geom = [values[mask] for values, mask in zip(face_tensor, generated_face_mask)]
        self.release(model)
        return {
            "face_bbox": face_bbox[0].float().cpu().numpy(),
            "vert_geom": vert_geom[0].float().cpu().numpy(),
            "edge_geom": edge_geom[0].float().cpu().numpy(),
            "face_geom": face_geom[0].float().cpu().numpy(),
        }
