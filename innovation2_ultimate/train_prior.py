# -*- coding: utf-8 -*-
"""
模块名称：train_prior.py
功能描述：自研的先验骨架图与面群布局网络 (CustomPriorNet) 本地联合训练程序。
          - 权重物理输出：直接物理隔离保存在 innovation2_spg_brep_geomgen/checkpoints/prior/ 目录下。
          - 本地物理日志：每轮 Epoch 结束后追加记录 LR、Train Loss、Train KL、Val BBox MSE 以及物理约束 Loss 值。
          - 联合损失优化：
            1. 节点类型分类 Loss (CrossEntropy, 忽略 padding 索引 7)；
            2. 边关系分类 Loss (CrossEntropy, 忽略无边索引 6)；
            3. BBox 布局回归 Loss (MSE, 仅对真实节点算误差)；
            4. VAE 隐空间 KL Loss (正规 Sum-Mean VAE 公式)；
            5. 自研平行/共面物理约束 Loss (Physical Prior Loss)，在前中期提供几何对齐惩罚。
使用方法：
    F:\pytorch_cuda12\python.exe innovation2_spg_brep_geomgen/train_prior.py --train_epochs 50
"""

import os
import sys
import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# 确保自研模块可以正确 import 本地 dataset 与 models_prior
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import MotifPriorDataset
from models_prior import CustomPriorNet, compute_physical_prior_loss


def get_args():
    """
    配置并解析自研先验网络的训练命令行参数。
    """
    parser = argparse.ArgumentParser(description="MotifPriorBRepGen - 先验图与布局联合训练入口")
    
    # 基础路径参数
    parser.add_argument('--manifest', type=str, 
                        default='innovation1_v3_brep_motif_graph/outputs/motif_graphs/motif_prior_index_ready.jsonl',
                        help='ready 样本清单路径')
    
    # 训练超参数
    parser.add_argument('--batch_size', type=int, default=16, help='训练批大小 (4G显存设为16最佳)')
    parser.add_argument('--train_epochs', type=int, default=50, help='训练总轮数 (一般50轮即可完全收敛)')
    parser.add_argument('--lr', type=float, default=2e-4, help='学习率')
    parser.add_argument('--save_epochs', type=int, default=10, help='每隔多少轮保存一次备份 checkpoint')
    parser.add_argument('--test_epochs', type=int, default=1, help='每隔多少轮进行一次验证集测试 (每轮都验证)')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader 线程数')
    parser.add_argument('--gpu', type=int, default=0, help='使用的 GPU 物理编号')
    
    # Loss 权衡权重
    parser.add_argument('--kl_weight', type=float, default=1e-3, help='VAE KL 正则项的平衡权重')
    parser.add_argument('--phys_weight', type=float, default=1e-1, help='自研平行/共面物理先验约束项的平衡权重')
    parser.add_argument('--edge_neg_weight', type=float, default=0.05,
                        help='无连接边的弱监督权重，防止无条件采样时生成过密图')
    
    args = parser.parse_args()
    
    # 物理隔离 prior 的权重输出目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    args.save_dir = os.path.join(base_dir, "checkpoints", "prior")
    return args


def train_prior(args):
    """
    主训练流程：执行自研先验图与布局回归联合训练。
    """
    # 1. 创建隔离权重目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 2. 建立本地物理日志文件路径
    log_file_path = os.path.join(args.save_dir, "prior_train.log")
    with open(log_file_path, "w", encoding="utf-8") as lf:
        lf.write("=== MotifPriorBRepGen 骨架图与布局网络联合训练日志 ===\n")
        lf.write(
            "Epoch | LR | Train Loss | Train KL | Train Phys | Val BBox MSE | "
            "Val Node Acc | Val Edge Acc | Val Edge Pos Acc | Val Edge Neg Acc\n"
        )
    print(f"[先验网络训练] 本地训练日志将实时记录在：{log_file_path}")
    
    # 3. 设置计算设备
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[先验网络训练] 当前计算物理设备：{device}")
    
    # 4. 实例化自研数据集 (MAX_NODES = 32 稠密张量对齐版)
    print("[先验网络训练] 正在加载 1717 个 ready 骨架图与几何包围盒数据...")
    train_dataset = MotifPriorDataset(
        args.manifest, is_train=True, train_ratio=0.9, max_nodes=32,
        include_geometry_targets=False
    )
    val_dataset = MotifPriorDataset(
        args.manifest, is_train=False, train_ratio=0.9, max_nodes=32,
        include_geometry_targets=False
    )
    
    # 5. 构建 DataLoader (安全线程 0)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers)
    
    # 6. 实例化 CustomPriorNet 网络模型
    model = CustomPriorNet(latent_dim=64, num_node_classes=8, num_edge_classes=7, max_nodes=32)
    model = model.to(device)
    
    # 7. 初始化优化器、余弦退火衰减器与基本 Loss 算子
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epochs, eta_min=1e-6)
    
    # 节点分类 Loss 忽略填充值 7；边关系保留“无连接=6”的弱监督，避免生成图过密。
    criterion_node = nn.CrossEntropyLoss(ignore_index=7)
    criterion_edge = nn.CrossEntropyLoss(reduction='none')
    criterion_bbox = nn.MSELoss(reduction='none') # 动态遮掩后求 mean
    
    # 8. 主训练循环
    best_val_mse = float('inf')
    print(f"[先验网络训练] 开始执行 {args.train_epochs} 轮自研骨架与物理布局的多任务拟合...")
    
    for epoch in range(1, args.train_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_kl = 0.0
        epoch_phys = 0.0
        batches = 0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.train_epochs}")
        for batch_data in progress:
            optimizer.zero_grad()
            
            # 读取特征并迁移至 GPU
            node_types = batch_data[0].to(device)
            target_bboxes = batch_data[1].to(device)
            adj_matrix = batch_data[2].to(device)
            node_masks = batch_data[3].to(device)
            
            # 模型前向传播
            pred_node_logits, pred_node_bboxes, pred_edge_logits, mu, logvar = model(
                node_types, target_bboxes, adj_matrix
            )
            
            # Loss 1: 节点类别预测 Loss
            # shape: pred_node_logits (B, 16, 8) -> permute 为 (B, 8, 16)；node_types (B, 16)
            loss_node = criterion_node(pred_node_logits.permute(0, 2, 1), node_types)
            
            # Loss 2: 边关系预测 Loss
            # shape: pred_edge_logits (B, 16, 16, 7) -> permute 为 (B, 7, 16, 16)；adj_matrix (B, 16, 16)
            loss_edge_raw = criterion_edge(pred_edge_logits.permute(0, 3, 1, 2), adj_matrix)
            real_pair_mask = (node_masks.unsqueeze(1) > 0.5) & (node_masks.unsqueeze(2) > 0.5)
            eye_mask = torch.eye(adj_matrix.shape[1], dtype=torch.bool, device=device).unsqueeze(0)
            edge_eval_mask = real_pair_mask & (~eye_mask)
            positive_edge_mask = edge_eval_mask & (adj_matrix != 6)
            negative_edge_mask = edge_eval_mask & (adj_matrix == 6)
            edge_weights = positive_edge_mask.float() + negative_edge_mask.float() * args.edge_neg_weight
            loss_edge = (loss_edge_raw * edge_weights).sum() / (edge_weights.sum() + 1e-6)
            
            # Loss 3: BBox 布局回归 Loss (仅在真实节点上计算误差)
            # shape: pred_node_bboxes (B, 16, 6)，target_bboxes (B, 16, 6)
            loss_bbox_raw = criterion_bbox(pred_node_bboxes, target_bboxes) # (B, 16, 6)
            loss_bbox_masked = loss_bbox_raw * node_masks.unsqueeze(-1)    # 仅对存在实体节点的 BBox 惩罚
            loss_bbox = loss_bbox_masked.sum() / (node_masks.sum() * 6.0 + 1e-6)
            
            # Loss 4: VAE 隐空间 KL 散度 Loss
            loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            
            # Loss 5: 自研平行/共面物理先验约束 Loss
            loss_phys = compute_physical_prior_loss(pred_node_bboxes, target_bboxes, adj_matrix, node_masks)
            
            # 多任务总损失聚合
            total_loss = (
                loss_node + 
                loss_edge + 
                loss_bbox * 5.0 +               # 放大 BBox 回归误差尺度，强制回归精度
                args.kl_weight * loss_kl + 
                args.phys_weight * loss_phys     # 注入物理先验约束惩罚
            )
            
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            # 累加统计
            epoch_loss += total_loss.item()
            epoch_kl += loss_kl.item()
            epoch_phys += loss_phys.item()
            batches += 1
            
            progress.set_postfix({
                "Loss": f"{total_loss.item():.4f}", 
                "BBox": f"{loss_bbox.item():.4f}",
                "Phys": f"{loss_phys.item():.4f}"
            })
            
        mean_train_loss = epoch_loss / batches
        mean_train_kl = epoch_kl / batches
        mean_train_phys = epoch_phys / batches
        
        # 每一个 epoch 结束后更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # 9. 执行验证集测试评估 (每一轮都进行验证)
        mean_val_mse = 0.0
        val_node_acc = 0.0
        val_edge_acc = 0.0
        val_edge_pos_acc = 0.0
        val_edge_neg_acc = 0.0
        
        if epoch % args.test_epochs == 0:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            total_nodes_evaluated = 0
            correct_nodes = 0
            
            total_edges_evaluated = 0
            correct_edges = 0
            total_pos_edges = 0
            correct_pos_edges = 0
            total_neg_edges = 0
            correct_neg_edges = 0
            
            with torch.no_grad():
                for batch_data in val_loader:
                    node_types = batch_data[0].to(device)
                    target_bboxes = batch_data[1].to(device)
                    adj_matrix = batch_data[2].to(device)
                    node_masks = batch_data[3].to(device)
                    
                    pred_node_logits, pred_node_bboxes, pred_edge_logits, _, _ = model(
                        node_types, target_bboxes, adj_matrix
                    )
                    
                    # 验证集 BBox 回归 MSE 评估
                    loss_bbox_raw = criterion_bbox(pred_node_bboxes, target_bboxes)
                    loss_bbox_masked = loss_bbox_raw * node_masks.unsqueeze(-1)
                    loss_bbox = loss_bbox_masked.sum() / (node_masks.sum() * 6.0 + 1e-6)
                    
                    val_loss += loss_bbox.item()
                    val_batches += 1
                    
                    # 验证集节点类别准确率 (Node Acc)
                    pred_node_classes = torch.argmax(pred_node_logits, dim=-1) # (B, 16)
                    # 忽略填充类别 7
                    node_eval_mask = (node_types != 7)
                    correct_nodes += (pred_node_classes[node_eval_mask] == node_types[node_eval_mask]).sum().item()
                    total_nodes_evaluated += node_eval_mask.sum().item()
                    
                    # 验证集边关系准确率 (Edge Acc)
                    pred_edge_classes = torch.argmax(pred_edge_logits, dim=-1) # (B, 16, 16)
                    real_pair_mask = (node_masks.unsqueeze(1) > 0.5) & (node_masks.unsqueeze(2) > 0.5)
                    eye_mask = torch.eye(adj_matrix.shape[1], dtype=torch.bool, device=device).unsqueeze(0)
                    edge_eval_mask = real_pair_mask & (~eye_mask)
                    correct_edges += (pred_edge_classes[edge_eval_mask] == adj_matrix[edge_eval_mask]).sum().item()
                    total_edges_evaluated += edge_eval_mask.sum().item()

                    pos_edge_mask = edge_eval_mask & (adj_matrix != 6)
                    neg_edge_mask = edge_eval_mask & (adj_matrix == 6)
                    correct_pos_edges += (pred_edge_classes[pos_edge_mask] == adj_matrix[pos_edge_mask]).sum().item()
                    total_pos_edges += pos_edge_mask.sum().item()
                    correct_neg_edges += (pred_edge_classes[neg_edge_mask] == adj_matrix[neg_edge_mask]).sum().item()
                    total_neg_edges += neg_edge_mask.sum().item()
                    
            mean_val_mse = val_loss / val_batches
            val_node_acc = correct_nodes / (total_nodes_evaluated + 1e-6)
            val_edge_acc = correct_edges / (total_edges_evaluated + 1e-6)
            val_edge_pos_acc = correct_pos_edges / (total_pos_edges + 1e-6)
            val_edge_neg_acc = correct_neg_edges / (total_neg_edges + 1e-6)
            
            print(
                f"-> Epoch {epoch} 验证结束 | 验证集 BBox MSE: {mean_val_mse:.6f} | "
                f"节点准确率: {val_node_acc * 100:.2f}% | 边关系准确率: {val_edge_acc * 100:.2f}% | "
                f"结构边准确率: {val_edge_pos_acc * 100:.2f}% | 无连接准确率: {val_edge_neg_acc * 100:.2f}%"
            )
            
            # 保存最优权重
            if mean_val_mse < best_val_mse:
                best_val_mse = mean_val_mse
                best_path = os.path.join(args.save_dir, "prior_net.pth")
                torch.save(model.state_dict(), best_path)
                print(f"🌟 发现更优模型！验证集最低 BBox MSE: {best_val_mse:.6f}，已保存至：{best_path}")
                
        # 10. 将本轮指标实时写入本地物理日志文件
        with open(log_file_path, "a", encoding="utf-8") as lf:
            lf.write(
                f"{epoch:02d} | {current_lr:.2e} | {mean_train_loss:.6f} | {mean_train_kl:.4f} | "
                f"{mean_train_phys:.6f} | {mean_val_mse:.6f} | {val_node_acc:.4f} | "
                f"{val_edge_acc:.4f} | {val_edge_pos_acc:.4f} | {val_edge_neg_acc:.4f}\n"
            )
            
        # 11. 保存物理备份 checkpoint
        if epoch % args.save_epochs == 0:
            save_path = os.path.join(args.save_dir, f"prior_epoch_{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"[Checkpoint] 已保存备份权重至：{save_path}")
            
    print(f"🎉 训练完成！历史最优验证集 BBox MSE: {best_val_mse:.6f}")


if __name__ == "__main__":
    train_prior(get_args())
