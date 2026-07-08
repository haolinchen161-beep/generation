"""训练路线2的 S-conditioned 宏观适配器。

该模型学习 S -> face bbox / face-edge topology。训练好后，sample_route2_step.py
会把它作为无条件 STEP 生成流程的宏观结构生成器。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from route2_adapter_models import SRoute2MacroAdapter, face_feature_dim, s_feature_dim, sort_minmax_bbox_tensor
from route2_dataset import Route2MacroDataset
from utils_io import ensure_dir, workdir_from_file, write_csv


def get_args() -> argparse.Namespace:
    workdir = workdir_from_file()
    parser = argparse.ArgumentParser(description="训练路线2 S-conditioned 宏观适配器")
    parser.add_argument("--dataset_dir", default=str(workdir / "data" / "deepcad30_s_ready"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_faces", type=int, default=30)
    parser.add_argument("--edge_classes", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--edge_zero_weight", type=float, default=1.0, help="topology loss 中无边类别的权重")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true", help="从 checkpoints/route2_faceaware_adapter/last.pt 继续训练")
    return parser.parse_args()


def _loss_batch(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    edge_classes: int,
    edge_zero_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    face_mask = batch["face_mask"].bool()
    pred_bbox = sort_minmax_bbox_tensor(outputs["bbox"])
    bbox_loss = F.mse_loss(pred_bbox[face_mask], batch["bbox"][face_mask])
    count_loss = F.cross_entropy(outputs["count_logits"], batch["num_faces"])

    adj_logits = outputs["adj_logits"]
    adj_target = batch["adj"]
    bsz, max_faces = adj_target.shape[0], adj_target.shape[1]
    upper = torch.triu(torch.ones(max_faces, max_faces, dtype=torch.bool, device=adj_target.device), diagonal=1)
    valid_pair = upper.unsqueeze(0).expand(bsz, -1, -1)
    valid_faces = face_mask.unsqueeze(1) & face_mask.unsqueeze(2)
    valid_pair = valid_pair & valid_faces
    logits_flat = adj_logits[valid_pair]
    target_flat = adj_target[valid_pair]
    weights = torch.ones(edge_classes, device=adj_logits.device)
    weights[0] = float(edge_zero_weight)
    adj_loss = F.cross_entropy(logits_flat, target_flat, weight=weights)
    total = bbox_loss + 0.6 * adj_loss + 0.2 * count_loss
    return {"loss": total, "bbox_loss": bbox_loss, "adj_loss": adj_loss, "count_loss": count_loss}


def _run_eval(
    model: SRoute2MacroAdapter,
    loader: DataLoader,
    device: torch.device,
    edge_classes: int,
    edge_zero_weight: float,
) -> Dict[str, float]:
    model.eval()
    rows: List[Dict[str, float]] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(batch["s_vec"], batch["face_features"])
            losses = _loss_batch(outputs, batch, edge_classes, edge_zero_weight=edge_zero_weight)
            rows.append({key: float(value.detach().cpu()) for key, value in losses.items()})
    model.train()
    return {key: sum(row[key] for row in rows) / max(len(rows), 1) for key in rows[0].keys()} if rows else {}


def main() -> None:
    args = get_args()
    device = torch.device(args.device)
    workdir = workdir_from_file()
    ckpt_dir = ensure_dir(workdir / "checkpoints" / "route2_faceaware_adapter")
    reports_dir = ensure_dir(workdir / "outputs" / "reports")

    train_ds = Route2MacroDataset(Path(args.dataset_dir), "train", max_faces=args.max_faces, edge_classes=args.edge_classes)
    val_ds = Route2MacroDataset(Path(args.dataset_dir), "val", max_faces=args.max_faces, edge_classes=args.edge_classes)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SRoute2MacroAdapter(
        s_dim=s_feature_dim(),
        max_faces=args.max_faces,
        edge_classes=args.edge_classes,
        hidden_dim=args.hidden_dim,
        face_dim=face_feature_dim(),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history: List[Dict[str, float]] = []
    best_val = float("inf")
    start_epoch = 1
    last_ckpt = ckpt_dir / "last.pt"
    best_ckpt = ckpt_dir / "best.pt"
    if args.resume and last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("val_loss", float("inf")))
        if best_ckpt.exists():
            best_state = torch.load(best_ckpt, map_location=device)
            best_val = min(best_val, float(best_state.get("val_loss", best_val)))
        print(f"从 {last_ckpt} 继续训练，起始 epoch={start_epoch}, 当前 best_val={best_val:.6f}")

    end_epoch = start_epoch + int(args.epochs) - 1
    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        running = []
        progress = tqdm(train_loader, desc=f"route2 adapter epoch {epoch}")
        for batch in progress:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(batch["s_vec"], batch["face_features"])
            losses = _loss_batch(outputs, batch, args.edge_classes, edge_zero_weight=args.edge_zero_weight)
            optimizer.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            row = {key: float(value.detach().cpu()) for key, value in losses.items()}
            running.append(row)
            progress.set_postfix(loss=f"{row['loss']:.4f}", bbox=f"{row['bbox_loss']:.4f}", adj=f"{row['adj_loss']:.4f}")

        train_mean = {f"train_{key}": sum(row[key] for row in running) / max(len(running), 1) for key in running[0].keys()}
        val_mean_raw = _run_eval(model, val_loader, device, args.edge_classes, args.edge_zero_weight)
        val_mean = {f"val_{key}": value for key, value in val_mean_raw.items()}
        log_row = {"epoch": epoch, **train_mean, **val_mean}
        history.append(log_row)
        print(log_row)

        ckpt = {
            "model_state": model.state_dict(),
            "config": {
                "s_dim": s_feature_dim(),
                "max_faces": args.max_faces,
                "edge_classes": args.edge_classes,
                "hidden_dim": args.hidden_dim,
                "face_dim": face_feature_dim(),
                "model_type": "faceaware",
                "edge_zero_weight": args.edge_zero_weight,
            },
            "epoch": epoch,
            "val_loss": val_mean.get("val_loss", 0.0),
        }
        torch.save(ckpt, ckpt_dir / "last.pt")
        if val_mean.get("val_loss", float("inf")) < best_val:
            best_val = val_mean["val_loss"]
            torch.save(ckpt, ckpt_dir / "best.pt")

    write_csv(reports_dir / "route2_adapter_train_log.csv", history)
    report = {
        "dataset_dir": args.dataset_dir,
        "epochs": args.epochs,
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "batch_size": args.batch_size,
        "edge_zero_weight": args.edge_zero_weight,
        "best_val_loss": best_val,
        "checkpoint": str((ckpt_dir / "best.pt").resolve()),
        "route": "route2_faceaware_s_conditioned_macro_adapter",
    }
    (reports_dir / "route2_adapter_train_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
