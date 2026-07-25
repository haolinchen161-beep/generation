"""Training script for PriorFaceBbox (Cross-Attention Face Role & Instance Allocator v6.1).

Implements Core Directives 3, 6, 7, 8, 11:
- Decoupled Loss: l_total = l_noise + 0.1 * l_role + 0.05 * l_host + 0.05 * l_local.
- l_host: Masked Soft-Target CrossEntropy over host_node_mask.
- l_local: Pos-Weighted BCE over local_node_mask.
- Deterministic Validation: torch.Generator(manual_seed=20260725 + b_idx).
- Cyclic Roll Shuffled Validation: torch.roll(shifts=1).
- Causal Margin Checkpoint Guard: saves best.pt ONLY when causal_margin = min(gain_vs_base, gain_vs_shuffle) > 0.
- Saves debug.pt for --debug, last.pt for recovery.
- Optimizer lr = 3e-4.
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
from innovation2_z.data_bbox import extract_motif_node_graph, MAX_MOTIF_NODES
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
    losses_host = []
    losses_local = []

    val_noise_baseline = []
    val_noise_prior = []
    val_noise_shuffled = []

    role_f1_sheet = []
    role_f1_hole = []
    role_f1_repeat = []

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

        # Directive 6: Deterministic Validation Seed Generator
        if is_training:
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
            noise = torch.randn_like(clean_bbox)
        else:
            gen = torch.Generator(device=device).manual_seed(20260725 + b_idx)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), generator=gen, device=device).long()
            noise = torch.randn(clean_bbox.shape, generator=gen, device=device, dtype=clean_bbox.dtype)

        noisy_bbox = scheduler.add_noise(clean_bbox, noise, timesteps)

        # Extract Motif Node Graph
        node_graph = extract_motif_node_graph(prior, max_nodes=MAX_MOTIF_NODES)
        node_features = node_graph["node_features"]
        hosted_adj = node_graph["hosted_adj"]
        thin_wall_adj = node_graph["thin_wall_adj"]
        node_mask = node_graph["node_mask"]
        node_roles = node_graph["node_roles"]
        asgn_target = node_graph["assignment_target"]

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            per_face_prior, role_logits, assignment_logits = model.prior_allocator(
                node_features,
                hosted_adj,
                thin_wall_adj,
                node_mask,
                node_roles,
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

            # 3. Directive 3: Decoupled Host Loss (Masked Soft-Target CrossEntropy)
            valid_node_mask = node_mask.bool()
            host_node_mask = valid_node_mask & ((node_roles == -1) | (node_roles == 0))
            host_mask_exp = host_node_mask.unsqueeze(1).expand(-1, face_mask.shape[1], -1)

            host_logits = assignment_logits.masked_fill(~host_mask_exp, -1e4)
            host_target = asgn_target * host_mask_exp.float()
            host_target_norm = host_target / host_target.sum(dim=-1, keepdim=True).clamp_min(1.0)
            host_log_probs = F.log_softmax(host_logits, dim=-1)
            host_loss_per_face = -(host_target_norm * host_log_probs).sum(dim=-1)
            l_host = host_loss_per_face[face_mask].mean()

            # 4. Directive 3: Decoupled Local Loss (Pos-Weighted BCE)
            local_node_mask = valid_node_mask & ((node_roles == 1) | (node_roles == 2))
            local_pair_mask = face_mask.unsqueeze(-1) & local_node_mask.unsqueeze(1)

            if local_pair_mask.any():
                local_vals = asgn_target[local_pair_mask]
                pos_cnt = local_vals.sum()
                neg_cnt = local_vals.numel() - pos_cnt
                pos_weight = (neg_cnt / pos_cnt.clamp_min(1.0)).clamp(1.0, 20.0)

                l_local = F.binary_cross_entropy_with_logits(
                    assignment_logits[local_pair_mask],
                    asgn_target[local_pair_mask],
                    pos_weight=pos_weight,
                )
            else:
                l_local = assignment_logits.sum() * 0.0

            # Combined Loss
            l_total = l_noise + 0.1 * l_role + 0.05 * l_host + 0.05 * l_local

            if is_training:
                l_total.backward()
                nn.utils.clip_grad_norm_(model.prior_allocator.parameters(), 1.0)
                optimizer.step()
            else:
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

                # Directive 6: Cyclic Roll Shuffled Validation
                if batch_size > 1:
                    perm = torch.roll(torch.arange(batch_size, device=device), shifts=1)
                    shuf_features = node_features[perm]
                    shuf_hosted = hosted_adj[perm]
                    shuf_thin = thin_wall_adj[perm]
                    shuf_nmask = node_mask[perm]
                    shuf_nroles = node_roles[perm]

                    shuf_per_face_prior, _, _ = model.prior_allocator(
                        shuf_features,
                        shuf_hosted,
                        shuf_thin,
                        shuf_nmask,
                        shuf_nroles,
                        fef_adj,
                        face_mask,
                    )
                    model.set_prior(shuf_per_face_prior)
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

                # Directive 11: Compute Role F1 per category
                role_preds = (torch.sigmoid(role_logits[face_mask]) > 0.5).float()
                role_gt = gt_membership[face_mask]
                for ch, lst in enumerate([role_f1_sheet, role_f1_hole, role_f1_repeat]):
                    p_ch = role_preds[:, ch]
                    g_ch = role_gt[:, ch]
                    tp = (p_ch * g_ch).sum()
                    fp = (p_ch * (1 - g_ch)).sum()
                    fn = ((1 - p_ch) * g_ch).sum()
                    f1 = (2 * tp / (2 * tp + fp + fn + 1e-8)).item()
                    lst.append(f1)

        losses_total.append(float(l_total.detach().cpu()))
        losses_noise.append(float(l_noise.detach().cpu()))
        losses_role.append(float(l_role.detach().cpu()))
        losses_host.append(float(l_host.detach().cpu()))
        losses_local.append(float(l_local.detach().cpu()))

    result = {
        "total": float(np.mean(losses_total)),
        "noise": float(np.mean(losses_noise)),
        "role": float(np.mean(losses_role)),
        "host": float(np.mean(losses_host)),
        "local": float(np.mean(losses_local)),
    }
    if not is_training and len(val_noise_baseline) > 0:
        result["val_noise_baseline"] = float(np.mean(val_noise_baseline))
        result["val_noise_prior"] = float(np.mean(val_noise_prior))
        result["val_noise_shuffled"] = float(np.mean(val_noise_shuffled))
        result["gain_vs_base"] = result["val_noise_baseline"] - result["val_noise_prior"]
        result["gain_vs_shuffle"] = result["val_noise_shuffled"] - result["val_noise_prior"]
        result["causal_margin"] = min(result["gain_vs_base"], result["gain_vs_shuffle"])
        result["role_f1_sheet"] = float(np.mean(role_f1_sheet))
        result["role_f1_hole"] = float(np.mean(role_f1_hole))
        result["role_f1_repeat"] = float(np.mean(role_f1_repeat))
        result["role_macro_f1"] = float(np.mean([result["role_f1_sheet"], result["role_f1_hole"], result["role_f1_repeat"]]))

    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="run only 2 batches for fast 5-second debug")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"TRAINING CROSS-ATTENTION PRIOR ALLOCATOR V6.1 (Device: {device})")
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
    print(f"Model Parameters: Trainable (PriorAllocator v6.1) = {trainable_params:,}, Frozen (Base) = {frozen_params:,}")

    # Directive 11: Learning Rate set to 3e-4
    optimizer = torch.optim.AdamW(model.prior_allocator.parameters(), lr=3e-4, weight_decay=1e-4)

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
    best_causal_margin = -float("inf")

    max_b = 2 if args.debug else None
    num_epochs = 1 if args.debug else 10

    for epoch in range(1, num_epochs + 1):
        tr = run_epoch(model, train_loader, scheduler, optimizer, device, is_training=True, max_batches=max_b)
        va = run_epoch(model, val_loader, scheduler, optimizer, device, is_training=False, max_batches=max_b)

        val_prior = va.get("val_noise_prior", va["noise"])
        val_base = va.get("val_noise_baseline", 0.0)
        val_shuf = va.get("val_noise_shuffled", 0.0)
        causal_margin = va.get("causal_margin", 0.0)

        print(
            f"Epoch {epoch:02d}/{num_epochs:02d} | Train Tot: {tr['total']:.4f} (Noise: {tr['noise']:.4f}, Host: {tr['host']:.4f}, Local: {tr['local']:.4f}) | "
            f"Val Noise (Base: {val_base:.4f}, Prior: {val_prior:.4f}, Shuf: {val_shuf:.4f}, Margin: {causal_margin:+.4f}) | "
            f"Role Macro F1: {va.get('role_macro_f1', 0):.4f}"
        )

        # Save last.pt every epoch for interrupt recovery
        last_path = ckpt_dir / "last.pt"
        ckpt_payload = {
            "schema_version": "prior_face_bbox_v6",
            "max_nodes": MAX_MOTIF_NODES,
            "epoch": epoch,
            "prior_allocator_state_dict": model.prior_allocator.state_dict(),
            "val_noise_prior": val_prior,
            "val_noise_baseline": val_base,
            "val_noise_shuffled": val_shuf,
            "gain_vs_base": va.get("gain_vs_base", 0.0),
            "gain_vs_shuffle": va.get("gain_vs_shuffle", 0.0),
            "causal_margin": causal_margin,
            "role_macro_f1": va.get("role_macro_f1", 0.0),
        }
        torch.save(ckpt_payload, last_path)

        # Directive 7: Save best.pt ONLY when causal_margin > 0 and improves over best!
        if args.debug:
            debug_path = ckpt_dir / "debug.pt"
            torch.save(ckpt_payload, debug_path)
            print(f"  --> Saved Debug Checkpoint to {debug_path}")
        elif causal_margin > 0 and causal_margin > best_causal_margin:
            best_causal_margin = causal_margin
            best_path = ckpt_dir / "best.pt"
            torch.save(ckpt_payload, best_path)
            print(f"  --> Saved Best Checkpoint based on causal_margin ({causal_margin:+.6f}) to {best_path}")

    print("=" * 80)
    print(f"TRAINING COMPLETE! Best Causal Margin: {best_causal_margin:+.6f}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
