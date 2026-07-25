"""Training script for PriorFaceBbox (Cross-Attention Face Role Allocator).

Trains the unified PriorAllocator module while keeping official FaceBbox base weights 100% frozen.
Implements Directives 3, 6, 7 & 8:
  - Exact DDPMScheduler parameter alignment (beta_schedule="linear").
  - Base model in eval() mode (model.base.eval()), allocator in train() mode (model.prior_allocator.train()).
  - Auxiliary role supervision loss (L_total = L_noise + 0.1 * L_role).
  - Unified prior_allocator checkpoint saving.
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from innovation2_z.data import StagewiseH5Dataset, collate_stagewise, to_device
from innovation2_z.dtg_backend import DTGBackend
from innovation2_z.data_bbox import extract_motif_node_graph
from innovation2_z.models_bbox import build_prior_face_bbox_model


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    scheduler: DDPMScheduler,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    is_training: bool = True,
    max_batches: Optional[int] = None,
) -> Tuple[float, float, float]:
    # Directive 6: Keep base in eval mode, allocator in train mode!
    model.base.eval()
    if is_training:
        model.prior_allocator.train()
    else:
        model.prior_allocator.eval()

    losses_total = []
    losses_noise = []
    losses_role = []

    for b_idx, cpu_batch in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break
        batch = to_device(cpu_batch, device)
        prior, target = batch["prior"], batch["target"]

        clean_bbox = target["face_bbox"].float()  # [b, max_faces, 6]
        face_mask = prior["face_mask"].bool()     # [b, max_faces]
        fef_adj = target["fef_adj"].long()        # [b, max_faces, max_faces]
        gt_membership = prior["motif_membership"].float()  # [b, max_faces, 3]

        batch_size = clean_bbox.shape[0]
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=device).long()

        noise = torch.randn_like(clean_bbox)
        noisy_bbox = scheduler.add_noise(clean_bbox, noise, timesteps)

        # Extract Motif Node Graph
        node_graph = extract_motif_node_graph(prior, max_nodes=15)
        node_features = node_graph["node_features"]
        hosted_adj = node_graph["hosted_adj"]
        thin_wall_adj = node_graph["thin_wall_adj"]
        node_mask = node_graph["node_mask"]

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            # Compute per_face_prior and auxiliary role_logits
            per_face_prior, role_logits = model.prior_allocator(
                node_features,
                hosted_adj,
                thin_wall_adj,
                node_mask,
                fef_adj,
                face_mask,
            )

            # Set prior in GuidedMLPInX
            model.set_prior(per_face_prior)

            pred_noise = model(
                noisy_bbox,
                fef_adj,
                face_mask,
                class_label=None,
                point_data=None,
                timesteps=timesteps.unsqueeze(-1),
            )
            model.set_prior(None)

            # 1. Bbox noise prediction MSE loss
            l_noise = F.mse_loss(pred_noise[face_mask], noise[face_mask])

            # 2. Directive 3: Auxiliary Role Classification Loss (BCE against gt_membership)
            l_role = F.binary_cross_entropy_with_logits(role_logits[face_mask], gt_membership[face_mask])

            # Combined Loss
            l_total = l_noise + 0.1 * l_role

            if is_training:
                l_total.backward()
                nn.utils.clip_grad_norm_(model.prior_allocator.parameters(), 1.0)
                optimizer.step()

        losses_total.append(float(l_total.detach().cpu()))
        losses_noise.append(float(l_noise.detach().cpu()))
        losses_role.append(float(l_role.detach().cpu()))

    return float(np.mean(losses_total)), float(np.mean(losses_noise)), float(np.mean(losses_role))


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"TRAINING CROSS-ATTENTION PRIOR ALLOCATOR (Device: {device})")
    print("=" * 80)

    train_h5 = PACKAGE_ROOT / "data" / "deepcad" / "train.h5"
    val_h5 = PACKAGE_ROOT / "data" / "deepcad" / "validation.h5"

    if not train_h5.exists() or not val_h5.exists():
        print("Error: HDF5 datasets not found!")
        return 1

    train_ds = StagewiseH5Dataset(train_h5, "face_bbox", training=True)
    val_ds = StagewiseH5Dataset(val_h5, "face_bbox", training=False)

    train_loader = DataLoader(
        train_ds, batch_size=64, shuffle=True, collate_fn=collate_stagewise, pin_memory=True, num_workers=2, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=64, shuffle=False, collate_fn=collate_stagewise, pin_memory=True, num_workers=2, persistent_workers=True
    )

    # Load official pretrained FaceBbox base model
    backend = DTGBackend(str(device))
    base_model = backend.load_face_bbox()

    # Build PriorFaceBboxModel wrapper
    model = build_prior_face_bbox_model(base_model).to(device)

    # Directive 6: Freeze all base model parameters
    for param in model.base.parameters():
        param.requires_grad = False
    for param in model.prior_allocator.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.prior_allocator.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.base.parameters())
    print(f"Model Parameters: Trainable (PriorAllocator) = {trainable_params:,}, Frozen (Base) = {frozen_params:,}")

    # Directive 7: Optimizer covers complete prior_allocator
    optimizer = torch.optim.AdamW(model.prior_allocator.parameters(), lr=1e-3, weight_decay=1e-4)

    # Directive 8: Exact DDPMScheduler Parameter Alignment with DTG Official!
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
        beta_start=0.0001,
        beta_end=0.02,
        prediction_type="epsilon",
        clip_sample=False,
    )

    ckpt_dir = PACKAGE_ROOT / "checkpoints" / "prior_face_bbox"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="run only 2 batches for fast 5-second debug")
    args = parser.parse_args()

    max_b = 2 if args.debug else None
    num_epochs = 1 if args.debug else 10

    for epoch in range(1, num_epochs + 1):
        train_tot, train_n, train_r = run_epoch(model, train_loader, scheduler, optimizer, device, is_training=True, max_batches=max_b)
        val_tot, val_n, val_r = run_epoch(model, val_loader, scheduler, optimizer, device, is_training=False, max_batches=max_b)

        print(
            f"Epoch {epoch:02d}/{num_epochs:02d} | Train Total: {train_tot:.6f} (Noise: {train_n:.6f}, Role: {train_r:.6f}) | "
            f"Val Total: {val_tot:.6f} (Noise: {val_n:.6f}, Role: {val_r:.6f})"
        )

        if val_tot < best_val_loss:
            best_val_loss = val_tot
            ckpt_path = ckpt_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "prior_allocator_state_dict": model.prior_allocator.state_dict(),
                    "val_loss": val_tot,
                    "val_noise_loss": val_n,
                    "val_role_loss": val_r,
                },
                ckpt_path,
            )
            print(f"  --> Saved Best Checkpoint to {ckpt_path} (Val Total Loss: {val_tot:.6f})")

    print("=" * 80)
    print(f"TRAINING COMPLETE! Best Validation Total Loss: {best_val_loss:.6f}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
