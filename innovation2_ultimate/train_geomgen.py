# -*- coding: utf-8 -*-
"""
模块名称：train_geomgen.py
功能描述：自研的微观几何与拓扑生成器 (CustomGeomGenNet) 本地联合训练主程序。
          - 权重物理输出：直接物理隔离保存在 innovation2_spg_brep_geomgen/checkpoints/geomgen/ 目录下。
          - 本地物理日志：每轮 Epoch 结束后追加记录 LR、Train Loss、Val Face MSE、Val Edge MSE 以及拓扑连接验证精度。
          - 联合损失优化：
            1. 面片几何特征回归 Loss (MSE, 仅对真实面片计算)；
            2. 边线几何特征回归 Loss (MSE, 仅对真实线段计算)；
            3. 面-线连接拓扑分类 Loss (BCEWithLogits, 动态掩模遮蔽，只对有效面线区域惩罚)；
            4. 自研弹性属从几何围栏 Loss (Belonging Fence Loss)，在中后期提供定位拉回惩罚。
使用方法：
    F:\pytorch_cuda12\python.exe innovation2_spg_brep_geomgen/train_geomgen.py --train_epochs 50
"""

import os
import sys
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# 确保自研模块可以正确 import 本地 dataset 与 models_geomgen
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import MotifPriorDataset
from models_geomgen import CustomGeomGenNet, compute_belonging_fence_loss


def get_args():
    """
    配置并解析自研几何与拓扑生成网络的训练命令行参数。
    """
    parser = argparse.ArgumentParser(description="MotifPriorBRepGen - 微观几何与拓扑联合生成训练入口")
    
    # 基础路径参数
    parser.add_argument('--manifest', type=str, 
                        default='innovation1_v3_brep_motif_graph/outputs/motif_graphs/motif_prior_index_ready.jsonl',
                        help='ready 样本清单路径')
    
    # 训练超参数
    parser.add_argument('--batch_size', type=int, default=16, help='训练批大小 (4G显存设为16最佳)')
    parser.add_argument('--train_epochs', type=int, default=50, help='训练总轮数 (50轮可实现高精度收敛)')
    parser.add_argument('--lr', type=float, default=2e-4, help='学习率')
    parser.add_argument('--save_epochs', type=int, default=10, help='每隔多少轮保存一次备份 checkpoint')
    parser.add_argument('--test_epochs', type=int, default=1, help='每隔多少轮进行一次验证集测试 (每轮都验证)')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader 线程数')
    parser.add_argument('--gpu', type=int, default=0, help='使用的 GPU 物理编号')
    
    # Loss 权衡权重
    parser.add_argument('--fence_weight', type=float, default=1e-1, help='自研弹性几何围栏损失的权重')
    parser.add_argument('--topo_pos_weight', type=float, default=12.0,
                        help='面-边真实连接的 BCE 正样本权重，缓解拓扑矩阵稀疏导致的全零退化')
    parser.add_argument('--face_latent_weight', type=float, default=5.0,
                        help='面片 VAE latent 回归损失权重')
    parser.add_argument('--edge_latent_weight', type=float, default=5.0,
                        help='边线 VAE latent 回归损失权重')
    parser.add_argument('--face_bbox_weight', type=float, default=5.0,
                        help='面片 BBox 空间回归损失权重')
    parser.add_argument('--topo_weight', type=float, default=5.0,
                        help='edgeFace 拓扑 BCE 损失权重')
    
    args = parser.parse_args()
    
    # 物理隔离 geomgen 的权重输出目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    args.save_dir = os.path.join(base_dir, "checkpoints", "geomgen")
    return args


def train_geomgen(args):
    """
    主训练流程：执行自研几何回归与拓扑连通分类联合训练。
    """
    # 1. 创建隔离权重目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 2. 建立本地物理日志文件路径
    log_file_path = os.path.join(args.save_dir, "geomgen_train.log")
    with open(log_file_path, "w", encoding="utf-8") as lf:
        lf.write("=== MotifPriorBRepGen 几何与拓扑生成网络联合训练日志 ===\n")
        lf.write("Epoch | LR | Train Loss | Val Face MSE | Val Edge MSE | Val Topo BCE | Val Topo F1 | Best Topo F1 | Best Thresh\n")
    print(f"[几何生成网络训练] 本地训练日志将实时记录在：{log_file_path}")
    
    # 3. 设置计算设备
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[几何生成网络训练] 当前计算物理设备：{device}")
    
    # 4. 实例化自研数据集 (MAX_NODES=32, MAX_FACES=64, MAX_EDGES=160 对齐版)
    # 会自动执行面/线特征的 VAE 离线前向缓存，保证 DataLoader 的物理飞速载入
    print("[几何生成网络训练] 正在加载并编码 1717 个 ready 数据集中的微观几何与拓扑矩阵...")
    train_dataset = MotifPriorDataset(args.manifest, is_train=True, train_ratio=0.9, 
                                      max_nodes=32, max_faces=64, max_edges=160, device=device)
    val_dataset = MotifPriorDataset(args.manifest, is_train=False, train_ratio=0.9, 
                                    max_nodes=32, max_faces=64, max_edges=160, device=device)
    
    # 5. 构建 DataLoader (安全线程 0)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers)
    
    # 6. 实例化 CustomGeomGenNet 网络模型
    model = CustomGeomGenNet(max_nodes=32, max_faces=64, max_edges=160)
    model = model.to(device)
    
    # 7. 初始化优化器、余弦退火衰减器与 Loss 算子
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epochs, eta_min=1e-6)
    
    criterion_mse = nn.MSELoss(reduction='none')
    criterion_bce = nn.BCEWithLogitsLoss(reduction='none')
    
    # 8. 主训练循环
    best_val_face_mse = float('inf')
    print(f"[几何生成网络训练] 开始执行 {args.train_epochs} 轮微观特征还原与拓扑连通的多任务拟合...")
    
    for epoch in range(1, args.train_epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches = 0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.train_epochs}")
        for batch_data in progress:
            optimizer.zero_grad()
            
            # 读取特征并迁移至 GPU
            # 1. 骨架输入
            node_types = batch_data[0].to(device)
            node_bboxes = batch_data[1].to(device)
            adj_matrix = batch_data[2].to(device)
            # 2. 生成目标
            face_latents = batch_data[4].to(device)
            edge_latents = batch_data[5].to(device)
            edge_face_adj = batch_data[6].to(device)
            # 3. 掩模遮罩与物理围栏
            face_masks = batch_data[7].to(device)
            edge_masks = batch_data[8].to(device)
            face_belong_matrix = batch_data[9].to(device)
            face_bbox_wcs = batch_data[10].to(device)
            
            # 模型前向传播
            pred_face_latents, pred_edge_latents, pred_edge_face_logits, pred_face_bboxes = model(
                node_types, node_bboxes, adj_matrix
            )
            
            # Loss 1: 面片 VAE 几何隐特征 MSE Loss
            loss_f_raw = criterion_mse(pred_face_latents, face_latents) # (B, 64, 64)
            loss_f_masked = loss_f_raw * face_masks.unsqueeze(-1)
            loss_face = loss_f_masked.sum() / (face_masks.sum() * 64.0 + 1e-6)
            
            # Loss 2: 边线 VAE 几何隐特征 MSE Loss
            loss_e_raw = criterion_mse(pred_edge_latents, edge_latents) # (B, 160, 16)
            loss_e_masked = loss_e_raw * edge_masks.unsqueeze(-1)
            loss_edge = loss_e_masked.sum() / (edge_masks.sum() * 16.0 + 1e-6)
            
            # Loss 2b: 面片 BBox 空间尺寸回归 MSE Loss（强监督，防止属从围栏退化坍塌）
            loss_fb_raw = criterion_mse(pred_face_bboxes, face_bbox_wcs) # (B, 64, 6)
            loss_fb_masked = loss_fb_raw * face_masks.unsqueeze(-1)
            loss_face_bbox = loss_fb_masked.sum() / (face_masks.sum() * 6.0 + 1e-6)
            
            # Loss 3: 面-线连接拓扑 BCE Loss (动态屏蔽非真实存在面线)
            # shape: pred_edge_face_logits (B, 160, 64)，edge_face_adj (B, 160, 64)
            loss_topo_raw = criterion_bce(pred_edge_face_logits, edge_face_adj)
            # 构建有效的面线交点掩模 (B, 160, 64)
            topo_mask = edge_masks.unsqueeze(-1) * face_masks.unsqueeze(1)
            topo_weight = torch.where(
                edge_face_adj > 0.5,
                torch.full_like(edge_face_adj, args.topo_pos_weight),
                torch.ones_like(edge_face_adj)
            )
            loss_topo_masked = loss_topo_raw * topo_mask * topo_weight
            loss_topo = loss_topo_masked.sum() / ((topo_mask * topo_weight).sum() + 1e-6)
            
            # Loss 4: 自研弹性几何属从围栏 Loss (使用预测的面 bbox 与真实面群 BBox 计算)
            loss_fence = compute_belonging_fence_loss(pred_face_bboxes, face_belong_matrix, node_bboxes, face_masks)
            
            # 多任务总损失聚合：几何 latent、面片 BBox 与拓扑共同优化，避免单一拓扑任务压过几何。
            total_loss = (
                loss_face * args.face_latent_weight + 
                loss_edge * args.edge_latent_weight + 
                loss_face_bbox * args.face_bbox_weight + 
                loss_topo * args.topo_weight + 
                args.fence_weight * loss_fence
            )
            
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            batches += 1
            
            progress.set_postfix({
                "Loss": f"{total_loss.item():.4f}", 
                "Topo": f"{loss_topo.item():.4f}",
                "BBox": f"{loss_face_bbox.item():.4f}",
                "Fence": f"{loss_fence.item():.4f}"
            })
            
        mean_train_loss = epoch_loss / batches
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # 9. 执行验证集测试评估 (每一轮都进行验证)
        val_face_mse = 0.0
        val_edge_mse = 0.0
        val_topo_bce = 0.0
        val_topo_f1 = 0.0
        best_f1 = 0.0
        best_thresh = 0.50
        
        if epoch % args.test_epochs == 0:
            model.eval()
            f_mse_sum = 0.0
            e_mse_sum = 0.0
            t_bce_sum = 0.0
            val_batches = 0
            
            all_logits = []
            all_targets = []
            all_masks = []
            
            with torch.no_grad():
                for batch_data in val_loader:
                    # 骨架输入
                    node_types = batch_data[0].to(device)
                    node_bboxes = batch_data[1].to(device)
                    adj_matrix = batch_data[2].to(device)
                    # 生成目标
                    face_latents = batch_data[4].to(device)
                    edge_latents = batch_data[5].to(device)
                    edge_face_adj = batch_data[6].to(device)
                    # 掩模遮罩
                    face_masks = batch_data[7].to(device)
                    edge_masks = batch_data[8].to(device)
                    
                    pred_face_latents, pred_edge_latents, pred_edge_face_logits, _ = model(
                        node_types, node_bboxes, adj_matrix
                    )
                    
                    # 1. 评估面片几何回归 MSE
                    loss_f_raw = criterion_mse(pred_face_latents, face_latents)
                    loss_f_masked = loss_f_raw * face_masks.unsqueeze(-1)
                    val_face_mse_batch = loss_f_masked.sum() / (face_masks.sum() * 64.0 + 1e-6)
                    f_mse_sum += val_face_mse_batch.item()
                    
                    # 2. 评估边线几何回归 MSE
                    loss_e_raw = criterion_mse(pred_edge_latents, edge_latents)
                    loss_e_masked = loss_e_raw * edge_masks.unsqueeze(-1)
                    val_edge_mse_batch = loss_e_masked.sum() / (edge_masks.sum() * 16.0 + 1e-6)
                    e_mse_sum += val_edge_mse_batch.item()
                    
                    # 3. 评估连接拓扑 BCE
                    loss_topo_raw = criterion_bce(pred_edge_face_logits, edge_face_adj)
                    topo_mask = edge_masks.unsqueeze(-1) * face_masks.unsqueeze(1)
                    val_topo_bce_batch = (loss_topo_raw * topo_mask).sum() / (topo_mask.sum() + 1e-6)
                    t_bce_sum += val_topo_bce_batch.item()
                    
                    # 收集以进行多阈值搜索
                    all_logits.append(pred_edge_face_logits.cpu())
                    all_targets.append(edge_face_adj.cpu())
                    all_masks.append(topo_mask.cpu())
                    
                    val_batches += 1
                    
            val_face_mse = f_mse_sum / val_batches
            val_edge_mse = e_mse_sum / val_batches
            val_topo_bce = t_bce_sum / val_batches
            
            # 拼接验证集所有批次的预测与标签
            logits = torch.cat(all_logits, dim=0)
            targets = torch.cat(all_targets, dim=0)
            masks = torch.cat(all_masks, dim=0)
            
            # 仅提取有效面线对进行 F1 统计
            valid_logits = logits[masks > 0.5].numpy()
            valid_targets = targets[masks > 0.5].numpy()
            probs = 1.0 / (1.0 + np.exp(-valid_logits))
            
            # 1) 计算默认 0.50 阈值下的 F1 分数
            preds_50 = (probs > 0.50).astype(float)
            tp_50 = ((preds_50 == 1.0) & (valid_targets == 1.0)).sum()
            fp_50 = ((preds_50 == 1.0) & (valid_targets == 0.0)).sum()
            fn_50 = ((preds_50 == 0.0) & (valid_targets == 1.0)).sum()
            precision_50 = tp_50 / (tp_50 + fp_50 + 1e-6)
            recall_50 = tp_50 / (tp_50 + fn_50 + 1e-6)
            val_topo_f1 = 2 * precision_50 * recall_50 / (precision_50 + recall_50 + 1e-6)
            
            # 2) 动态扫描最佳 Threshold 以适配 BCE pos_weight 引起的概率偏移
            for thresh in np.arange(0.1, 0.95, 0.05):
                preds = (probs > thresh).astype(float)
                tp = ((preds == 1.0) & (valid_targets == 1.0)).sum()
                fp = ((preds == 1.0) & (valid_targets == 0.0)).sum()
                fn = ((preds == 0.0) & (valid_targets == 1.0)).sum()
                precision = tp / (tp + fp + 1e-6)
                recall = tp / (tp + fn + 1e-6)
                f1 = 2 * precision * recall / (precision + recall + 1e-6)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
            
            print(f"-> Epoch {epoch} 验证结束 | Val Face MSE: {val_face_mse:.6f} | Val Edge MSE: {val_edge_mse:.6f} | Val Topo BCE: {val_topo_bce:.6f} | "
                  f"Topo F1 (0.50): {val_topo_f1 * 100:.2f}% | Best F1: {best_f1 * 100:.2f}% (Thresh: {best_thresh:.2f})")
            
            # 物理保存最优面几何特征还原模型
            if val_face_mse < best_val_face_mse:
                best_val_face_mse = val_face_mse
                best_path = os.path.join(args.save_dir, "geomgen_net.pth")
                torch.save(model.state_dict(), best_path)
                print(f"🌟 发现更优模型！验证集最低面片几何 MSE: {best_val_face_mse:.6f}，已保存至：{best_path}")
                
        # 10. 将本轮指标实时写入本地物理日志文件
        with open(log_file_path, "a", encoding="utf-8") as lf:
            lf.write(
                f"{epoch:02d} | {current_lr:.2e} | {mean_train_loss:.6f} | {val_face_mse:.6f} | "
                f"{val_edge_mse:.6f} | {val_topo_bce:.6f} | {val_topo_f1:.4f} | {best_f1:.4f} | {best_thresh:.2f}\n"
            )
            
        # 11. 保存物理备份 checkpoint
        if epoch % args.save_epochs == 0:
            save_path = os.path.join(args.save_dir, f"geomgen_epoch_{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"[Checkpoint] 已保存备份权重至：{save_path}")
            
    print(f"🎉 联合几何与拓扑生成训练完成！历史最优 Face MSE: {best_val_face_mse:.6f}")


if __name__ == "__main__":
    train_geomgen(get_args())
