# -*- coding: utf-8 -*-
"""
模块名称：train_vae.py
功能描述：自研的几何自编码器 (Face VAE 与 Edge VAE) 本地自监督预训练主程序。
          【思前想后、数据第一重构版】：
          1. 物理目录彻底隔离（face/ 与 edge/）：
             面片与线曲线的权重和日志将彻底物理隔离，分别保存在 checkpoints/face/ 和 checkpoints/edge/ 目录下。
          2. 隐维度按数据量精准匹配（防止退化）：
             - 二维面片 (3, 32, 32 = 3072维) 使用 64 维隐空间；
             - 一维边界线 (3, 32 = 96维) 降为 16 维隐空间（防止 64 维过剩引发的 KL 散度退化，迫使模型拟合几何特征）。
          3. Tanh 激活函数物理实锤：通过实测数据确证面/线 NCS 归一化极值在 [-1.0, 1.0] 闭区间，使用 Tanh 绝对吻合。
          4. 动态学习率：使用 CosineAnnealingLR，在 25 轮中从 2e-4 余弦衰减至 1e-6，精细捕捉几何倒角。
使用方法：
    重新训练面 VAE：F:\pytorch_cuda12\python.exe innovation2_spg_brep_geomgen/train_vae.py --option face --train_epochs 25
    重新训练线 VAE：F:\pytorch_cuda12\python.exe innovation2_spg_brep_geomgen/train_vae.py --option edge --train_epochs 25
"""

import os
import sys
import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# 确保自研模块可以正确 import 本地 dataset
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import FaceVaeDataset, EdgeVaeDataset


class CustomFaceVAE(nn.Module):
    """
    自研的二维面片几何变分自编码器 (Custom Face VAE)。
    """
    def __init__(self, latent_dim=64):
        super(CustomFaceVAE, self).__init__()
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # (B, 32, 16, 16)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # (B, 64, 8, 8)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # (B, 128, 4, 4)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten()                                          # (B, 2048)
        )
        
        self.fc_mu = nn.Linear(2048, latent_dim)
        self.fc_var = nn.Linear(2048, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 2048)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (B, 64, 8, 8)
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # (B, 32, 16, 16)
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),    # (B, 3, 32, 32)
            nn.Tanh()  # 100% 对齐数据源 [-1.0, 1.0] 的极值归一化范围
        )

    def encode(self, x):
        features = self.encoder(x)
        mu = self.fc_mu(features)
        logvar = self.fc_var(features)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x_flat = self.decoder_input(z)
        x_reshaped = x_flat.view(-1, 128, 4, 4)
        return self.decoder(x_reshaped)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


class CustomEdgeVAE(nn.Module):
    """
    自研的一维边界线几何变分自编码器 (Custom Edge VAE)。
    """
    def __init__(self, latent_dim=16): # 针对边界线特征降为 16 维，防止 KL 退化
        super(CustomEdgeVAE, self).__init__()
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=3, stride=2, padding=1),  # (B, 32, 16)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1), # (B, 64, 8)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1), # (B, 128, 4)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten()                                          # (B, 512)
        )
        
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_var = nn.Linear(512, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 512)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),  # (B, 64, 8)
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),   # (B, 32, 16)
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(32, 3, kernel_size=4, stride=2, padding=1),    # (B, 3, 32)
            nn.Tanh()
        )

    def encode(self, x):
        features = self.encoder(x)
        mu = self.fc_mu(features)
        logvar = self.fc_var(features)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x_flat = self.decoder_input(z)
        x_reshaped = x_flat.view(-1, 128, 4)
        return self.decoder(x_reshaped)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


def get_args():
    """
    配置并解析自研 VAE 的训练命令行参数。
    """
    parser = argparse.ArgumentParser(description="MotifPriorBRepGen - VAE 预训练入口")
    
    # 基础路径参数
    parser.add_argument('--manifest', type=str, 
                        default='innovation1_v3_brep_motif_graph/outputs/motif_graphs/motif_prior_index_ready.jsonl',
                        help='ready 样本清单路径')
    parser.add_argument('--option', type=str, choices=['face', 'edge'], default='face',
                        help='选择训练对象：face (面自编码器) 或 edge (线自编码器)')
    
    # 训练超参数
    parser.add_argument('--batch_size', type=int, default=16, help='训练批大小')
    parser.add_argument('--train_epochs', type=int, default=25, help='训练总轮数')
    parser.add_argument('--lr', type=float, default=2e-4, help='学习率')
    parser.add_argument('--save_epochs', type=int, default=5, help='每隔多少轮保存一次备份 checkpoint')
    parser.add_argument('--test_epochs', type=int, default=1, help='每隔多少轮进行一次验证集测试 (每轮都验证)')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader 线程数')
    parser.add_argument('--gpu', type=int, default=0, help='使用的 GPU 物理编号')
    
    args = parser.parse_args()
    
    # 划分物理子目录隔离权重
    base_dir = os.path.dirname(os.path.abspath(__file__))
    args.save_dir = os.path.join(base_dir, "checkpoints", args.option)
    return args


def train_vae(args):
    """
    主训练流程：执行自监督面/线 VAE 训练。
    """
    # 1. 创建物理子目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 2. 建立本地物理日志文件路径
    log_file_path = os.path.join(args.save_dir, "vae_train.log")
    with open(log_file_path, "w", encoding="utf-8") as lf:
        lf.write(f"=== {args.option.upper()} VAE 自研训练物理日志 ===\n")
        lf.write("Epoch | LR | Train MSE | Train KL | Val MSE\n")
    print(f"[VAE 预训练] 本地训练日志将实时记录在：{log_file_path}")
    
    # 3. 设置计算设备
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[VAE 预训练] 当前计算物理设备：{device}")
    
    # 4. 根据 option 载入自研模型与 Dataset，配置最适合数据的隐维度 (latent_dim)
    if args.option == 'face':
        print("[VAE 预训练] 正在加载面片几何数据 (CustomFaceVAE)...")
        train_dataset = FaceVaeDataset(args.manifest, is_train=True, train_ratio=0.9, aug=True)
        val_dataset = FaceVaeDataset(args.manifest, is_train=False, train_ratio=0.9, aug=False)
        model = CustomFaceVAE(latent_dim=64) # 3072维 -> 64维压缩
    else:
        print("[VAE 预训练] 正在加载边界线几何数据 (CustomEdgeVAE)...")
        train_dataset = EdgeVaeDataset(args.manifest, is_train=True, train_ratio=0.9, aug=True)
        val_dataset = EdgeVaeDataset(args.manifest, is_train=False, train_ratio=0.9, aug=False)
        model = CustomEdgeVAE(latent_dim=16) # 96维 -> 16维压缩，思前想后适配线数据，防隐空间坍塌
        
    model = model.to(device)
    
    # 5. 构建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers)
    
    # 6. 初始化优化器、余弦退火衰减器与损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epochs, eta_min=1e-6)
    loss_fn = nn.MSELoss()
    
    # 7. 主训练循环
    iters = 0
    best_val_mse = float('inf')
    print(f"[VAE 预训练] 开始执行 {args.train_epochs} 轮自监督拟合...")
    
    for epoch in range(1, args.train_epochs + 1):
        model.train()
        epoch_mse = 0.0
        epoch_kl = 0.0
        batches = 0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.train_epochs}")
        for data_batch in progress:
            optimizer.zero_grad()
            
            # 数据形状整理
            if args.option == 'face':
                x = data_batch.to(device).permute(0, 3, 1, 2)  # (B, 3, 32, 32)
            else:
                x = data_batch.to(device).permute(0, 2, 1)     # (B, 3, 32)
                
            # 前向传播
            recon_x, mu, logvar = model(x)
            
            # 损失计算
            mse_loss = loss_fn(recon_x, x)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            total_loss = mse_loss + 1e-3 * kl_loss
                
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            # 累加统计
            epoch_mse += mse_loss.item()
            epoch_kl += kl_loss.item()
            batches += 1
            iters += 1
            
            progress.set_postfix({"MSE": f"{mse_loss.item():.6f}", "KL": f"{kl_loss.item():.4f}"})
                
        mean_train_mse = epoch_mse / batches
        mean_train_kl = epoch_kl / batches
        
        # 每一个 epoch 结束后更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # 8. 执行验证集测试评估 (每一轮都进行验证)
        mean_val_mse = 0.0
        if epoch % args.test_epochs == 0:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for data_batch in val_loader:
                    if args.option == 'face':
                        x = data_batch.to(device).permute(0, 3, 1, 2)
                    else:
                        x = data_batch.to(device).permute(0, 2, 1)
                        
                    recon_x, mu, logvar = model(x)
                    mse_loss = loss_fn(recon_x, x)
                    val_loss += mse_loss.item()
                    val_batches += 1
                    
            mean_val_mse = val_loss / val_batches
            print(f"-> Epoch {epoch} 验证结束 | 验证集均方误差 MSE: {mean_val_mse:.6f}")
                
            # 保存历史最优权重 (分别保存在各自的子目录下，例如 checkpoints/face/face_vae.pth)
            if mean_val_mse < best_val_mse:
                best_val_mse = mean_val_mse
                best_path = os.path.join(args.save_dir, f"{args.option}_vae.pth")
                torch.save(model.state_dict(), best_path)
                print(f"🌟 发现更优模型！验证集最低 MSE: {best_val_mse:.6f}，已保存至：{best_path}")
        
        # 9. 将本轮指标实时写入各自的物理日志文件中
        with open(log_file_path, "a", encoding="utf-8") as lf:
            lf.write(f"{epoch:02d} | {current_lr:.2e} | {mean_train_mse:.6f} | {mean_train_kl:.4f} | {mean_val_mse:.6f}\n")
            
        # 10. 定期保存备份权重
        if epoch % args.save_epochs == 0:
            save_path = os.path.join(args.save_dir, f"{args.option}_epoch_{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"[Checkpoint] 已保存备份权重至：{save_path}")
            
    print(f"🎉 训练完成！历史最优验证集 MSE: {best_val_mse:.6f}")


if __name__ == "__main__":
    train_vae(get_args())
