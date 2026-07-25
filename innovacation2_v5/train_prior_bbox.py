"""Training script for PriorFaceBbox (Cross-Attention Face Role & Instance Allocator v5).

Implements Core Directives 6, 7, 8:
- Loss = loss_noise + 0.1 * loss_role + 0.1 * loss_assignment.
- 3-Way Validation: val_noise_baseline (prior off), val_noise_prior (correct prior), val_noise_shuffled (shuffled prior).
- Saves checkpoint based on val_noise_prior / noise_gain (val_noise_baseline - val_noise_prior).
"""

from __future__ import annotations

import argparse
import os
import sys
import json
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
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
) -> Dict[str, float]:
    model.base.eval()
    if is_training:
        model.prior_allocator.train()
    else:
        model.prior_allocator.eval()

    losses_total = []
    losses_noise = []
    losses_role = []
    losses_asgn = []

    # Validation ablation counters
    val_noise_baseline = []
    val_noise_prior = []
    val_noise_shuffled = []

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

        # Extract Motif Node Graph & Assignment Target
        node_graph = extract_motif_node_graph(prior, max_nodes=32)
        node_features = node_graph["node_features"]
        hosted_adj = node_graph["hosted_adj"]
        thin_wall_adj = node_graph["thin_wall_adj"]
        node_mask = node_graph["node_mask"]
        asgn_target = node_graph["assignment_target"]

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            per_face_prior, role_logits, assignment_logits = model.prior_allocator(
                node_features,
                hosted_adj,
                thin_wall_adj,
                node_mask,
                fef_adj,
                face_mask,
            )

            # Prior pass
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

            # 1. Bbox noise MSE loss
            l_noise = F.mse_loss(pred_noise[face_mask], noise[face_mask])

            # 2. Auxiliary Role Classification Loss
            l_role = F.binary_cross_entropy_with_logits(role_logits[face_mask], gt_membership[face_mask])

            # 3. Auxiliary Instance Assignment Loss
            l_asgn = F.binary_cross_entropy_with_logits(assignment_logits[face_mask], asgn_target[face_mask])

            # Total Loss
            l_total = l_noise + 0.1 * l_role + 0.1 * l_asgn

            if is_training:
                l_total.backward()
                nn.utils.clip_grad_norm_(model.prior_allocator.parameters(), 1.0)
                optimizer.step()
            else:
                # 3-Way Ablation Evaluation during validation
                val_noise_prior.append(float(l_noise.detach().cpu()))

                # Baseline pass (prior off)
                model.set_prior(None)
                pred_base = model(
                    noisy_bbox,
                    fef_adj,
                    face_mask,
                    class_label=None,
                    point_data=None,
                    timesteps=timesteps.unsqueeze(-1),
                )
                l_base = F.mse_loss(pred_base[face_mask], noise[face_mask])
                val_noise_baseline.append(float(l_base.detach().cpu()))

                # Shuffled prior pass
                if batch_size > 1:
                    perm = torch.randperm(batch_size, device=device)
                    shuffled_prior = per_face_prior[perm]
                    model.set_prior(shuffled_prior)
                    pred_shuf = model(
                        noisy_bbox,
                        fef_adj,
                        face_mask,
                        class_label=None,
                        point_data=None,
                        timesteps=timesteps.unsqueeze(-1),
                    )
                    model.set_prior(None)
                    l_shuf = F.mse_loss(pred_shuf[face_mask], noise[face_mask])
                    val_noise_shuffled.append(float(l_shuf.detach().cpu()))
                else:
                    val_noise_shuffled.append(float(l_base.detach().cpu()))

        losses_total.append(float(l_total.detach().cpu()))
        losses_noise.append(float(l_noise.detach().cpu()))
        losses_role.append(float(l_role.detach().cpu()))
        losses_asgn.append(float(l_asgn.detach().cpu()))

    result = {
        "total": float(np.mean(losses_total)),
        "noise": float(np.mean(losses_noise)),
        "role": float(np.mean(losses_role)),
        "asgn": float(np.mean(losses_asgn)),
    }
    if not is_training and len(val_noise_baseline) > 0:
        result["val_noise_baseline"] = float(np.mean(val_noise_baseline))
        result["val_noise_prior"] = float(np.mean(val_noise_prior))
        result["val_noise_shuffled"] = float(np.mean(val_noise_shuffled))
        result["noise_gain"] = result["val_noise_baseline"] - result["val_noise_prior"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="run only 2 batches for fast 5-second debug")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"TRAINING CROSS-ATTENTION PRIOR ALLOCATOR V5 (Device: {device})")
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

    backend = DTGBackend(str(device))
    base_model = backend.load_face_bbox()

    model = build_prior_face_bbox_model(base_model).to(device)

    for param in model.base.parameters():
        param.requires_grad = False
    for param in model.prior_allocator.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.prior_allocator.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.base.parameters())
    print(f"Model Parameters: Trainable (PriorAllocator v5) = {trainable_params:,}, Frozen (Base) = {frozen_params:,}")

    optimizer = torch.optim.AdamW(model.prior_allocator.parameters(), lr=1e-3, weight_decay=1e-4)

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
    best_val_noise_prior = float("inf")

    max_b = 2 if args.debug else None
    num_epochs = 1 if args.debug else 10

    for epoch in range(1, num_epochs + 1):
        tr = run_epoch(model, train_loader, scheduler, optimizer, device, is_training=True, max_batches=max_b)
        va = run_epoch(model, val_loader, scheduler, optimizer, device, is_training=False, max_batches=max_b)

        print(
            f"Epoch {epoch:02d}/{num_epochs:02d} | Train Total: {tr['total']:.6f} (Noise: {tr['noise']:.6f}, Role: {tr['role']:.6f}, Asgn: {tr['asgn']:.6f}) | "
            f"Val Noise (Base: {va.get('val_noise_baseline', 0):.6f}, Prior: {va.get('val_noise_prior', 0):.6f}, Shuf: {va.get('val_noise_shuffled', 0):.6f})"
        )

        curr_val_noise = va.get("val_noise_prior", va["total"])
        if curr_val_noise < best_val_noise_prior:
            best_val_noise_prior = curr_val_noise
            ckpt_path = ckpt_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "prior_allocator_state_dict": model.prior_allocator.state_dict(),
                    "val_noise_prior": curr_val_noise,
                    "val_noise_baseline": va.get("val_noise_baseline", 0),
                    "val_noise_shuffled": va.get("val_noise_shuffled", 0),
                },
                ckpt_path,
            )
            print(f"  --> Saved Best Checkpoint based on val_noise_prior: {curr_val_noise:.6f}")

    print("=" * 80)
    print(f"TRAINING COMPLETE! Best Validation Noise MSE: {best_val_noise_prior:.6f}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
