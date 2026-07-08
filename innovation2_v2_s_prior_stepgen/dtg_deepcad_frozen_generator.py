"""冻结 DTG-DeepCAD backbone 的本地模型加载封装。

本文件只 import 原 DTG 模块和权重，不修改原源码。路线2使用该封装加载
DTG-DeepCAD 的 edge-vertex 与几何扩散模块，再由 S-conditioned adapter 提供
face bbox / face-edge topology。
"""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from diffusers import DDPMScheduler, PNDMScheduler

from utils_io import ensure_dir


def _insert_repo_path(repo_root: Path) -> None:
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


class FrozenDeepCADDTGGenerator:
    """加载冻结 DTG-DeepCAD 六权重，供路线2下游几何生成调用。"""

    def __init__(
        self,
        repo_root: Path,
        output_dir: Path,
        checkpoints_dir: Optional[Path] = None,
        device: Optional[str] = None,
        disable_point_condition: bool = True,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.output_dir = ensure_dir(Path(output_dir))
        self.checkpoints_dir = Path(checkpoints_dir or self.repo_root / "checkpoints_base" / "deepcad").resolve()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.disable_point_condition = bool(disable_point_condition)
        _insert_repo_path(self.repo_root)
        self.args = self._build_args()
        self.models_loaded = False

    def _build_args(self) -> Namespace:
        with (self.repo_root / "config.yaml").open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f).get("deepcad", {})
        # 先跑通无条件生成：不使用点云条件。权重仍来自 DTG-DeepCAD。
        if self.disable_point_condition:
            config["use_pc"] = False
        config["use_cf"] = False
        config["faceGeom_path"] = str(self.checkpoints_dir / "geom_faceGeom" / "epoch_3000.pt")
        config["edgeGeom_path"] = str(self.checkpoints_dir / "geom_edgeGeom" / "epoch_3000.pt")
        config["vertGeom_path"] = str(self.checkpoints_dir / "geom_vertGeom" / "epoch_3000.pt")
        config["faceBbox_path"] = str(self.checkpoints_dir / "geom_faceBbox" / "epoch_3000.pt")
        config["faceEdge_path"] = str(self.checkpoints_dir / "topo_faceEdge" / "epoch_2000.pt")
        config["edgeVert_path"] = str(self.checkpoints_dir / "topo_edgeVert" / "epoch_1000.pt")
        config["save_folder"] = str(self.output_dir / "steps")
        config["name"] = "deepcad"
        config["parallel"] = False
        return Namespace(**config)

    def _load_state(self, model: torch.nn.Module, path: str, strict: bool = False) -> Dict[str, Any]:
        state = torch.load(path, map_location=self.device)
        result = model.load_state_dict(state, strict=strict)
        return {
            "path": path,
            "missing_keys": list(getattr(result, "missing_keys", [])),
            "unexpected_keys": list(getattr(result, "unexpected_keys", [])),
        }

    def load_models(self) -> None:
        if self.models_loaded:
            return
        _insert_repo_path(self.repo_root)
        from model import (  # noqa: WPS433
            EdgeGeomTransformer,
            EdgeVertModel,
            FaceBboxTransformer,
            FaceEdgeModel,
            FaceGeomTransformer,
            VertGeomTransformer,
        )

        args = self.args
        device = self.device

        self.faceEdge_model = FaceEdgeModel(
            nf=args.max_face,
            d_model=args.FaceEdgeModel["d_model"],
            nhead=args.FaceEdgeModel["nhead"],
            n_layers=args.FaceEdgeModel["n_layers"],
            num_categories=args.edge_classes,
            use_cf=args.use_cf,
            use_pc=args.use_pc,
        )
        self._load_state(self.faceEdge_model, args.faceEdge_path, strict=False)
        self.faceEdge_model = self.faceEdge_model.to(device).eval()

        self.edgeVert_model = EdgeVertModel(
            max_num_edge=args.max_num_edge_topo,
            max_seq_length=args.max_seq_length,
            edge_classes=args.edge_classes,
            max_face=args.max_face,
            max_edge=args.max_edge,
            d_model=args.EdgeVertModel["d_model"],
            n_layers=args.EdgeVertModel["n_layers"],
            use_cf=args.use_cf,
            use_pc=args.use_pc,
        )
        self._load_state(self.edgeVert_model, args.edgeVert_path, strict=False)
        self.edgeVert_model = self.edgeVert_model.to(device).eval()

        self.faceBbox_model = FaceBboxTransformer(
            n_layers=args.FaceBboxModel["n_layers"],
            hidden_mlp_dims=args.FaceBboxModel["hidden_mlp_dims"],
            hidden_dims=args.FaceBboxModel["hidden_dims"],
            edge_classes=args.edge_classes,
            act_fn_in=torch.nn.ReLU(),
            act_fn_out=torch.nn.ReLU(),
            use_cf=args.use_cf,
            use_pc=args.use_pc,
        )
        self._load_state(self.faceBbox_model, args.faceBbox_path, strict=False)
        self.faceBbox_model = self.faceBbox_model.to(device).eval()

        self.vertGeom_model = VertGeomTransformer(
            n_layers=args.VertGeomModel["n_layers"],
            hidden_mlp_dims=args.VertGeomModel["hidden_mlp_dims"],
            hidden_dims=args.VertGeomModel["hidden_dims"],
            act_fn_in=torch.nn.ReLU(),
            act_fn_out=torch.nn.ReLU(),
            use_cf=args.use_cf,
            use_pc=args.use_pc,
        )
        self._load_state(self.vertGeom_model, args.vertGeom_path, strict=False)
        self.vertGeom_model = self.vertGeom_model.to(device).eval()

        self.edgeGeom_model = EdgeGeomTransformer(
            n_layers=args.EdgeGeomModel["n_layers"],
            edge_geom_dim=args.EdgeGeomModel["edge_geom_dim"],
            d_model=args.EdgeGeomModel["d_model"],
            nhead=args.EdgeGeomModel["nhead"],
            use_cf=args.use_cf,
            use_pc=args.use_pc,
        )
        self._load_state(self.edgeGeom_model, args.edgeGeom_path, strict=False)
        self.edgeGeom_model = self.edgeGeom_model.to(device).eval()

        self.faceGeom_model = FaceGeomTransformer(
            n_layers=args.FaceGeomModel["n_layers"],
            face_geom_dim=args.FaceGeomModel["face_geom_dim"],
            d_model=args.FaceGeomModel["d_model"],
            nhead=args.FaceGeomModel["nhead"],
            use_cf=args.use_cf,
            use_pc=args.use_pc,
        )
        self._load_state(self.faceGeom_model, args.faceGeom_path, strict=False)
        self.faceGeom_model = self.faceGeom_model.to(device).eval()

        self.pndm_scheduler = PNDMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            prediction_type="epsilon",
            beta_start=0.0001,
            beta_end=0.02,
        )
        self.ddpm_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            prediction_type="epsilon",
            beta_start=0.0001,
            beta_end=0.02,
            clip_sample=True,
            clip_sample_range=3,
        )
        self.models_loaded = True
