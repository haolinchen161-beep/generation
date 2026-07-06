# -*- coding: utf-8 -*-
"""
模块名称：models_geomgen.py
功能描述：自研的微观拓扑与几何联合生成模型 (CustomGeomGenNet)。
          - 100% 独立自主设计，摒弃老基线 DTG 随机无约束的 Diffusion 几何生成。
          - 采用【先验条件注意力 Transformer (Prior-Conditional Transformer)】：
            1. 使用 Prior Graph Encoder 提取已生成的骨架先验图特征 S*；
            2. 使用多路 Cross-Attention，分别引导面片几何预测 (Face Queries) 与线几何预测 (Edge Queries)；
            3. 通过两两关联特征映射，直接预测面与线之间的微观拓扑硬规则连接图 (edgeFace_adj)。
          - 引入【自研几何属从围栏 Loss (Elastic Belonging Fence Loss)】：
            面片本属于某个面群（Motif 节点）。模型在回归预测面片中心位置时，必须受其所属面群 BBox 的物理边界限缩。
            我们通过 Relu 激活函数构建弹性几何围栏损失，一旦预测的面片中心偏离了面群包围盒的真实空间范围，即施加惩罚，
            从根本上解决了盲目 Diffusion 带来的几何散乱与实体缝合率低下的学术难题！
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomGeomGenNet(nn.Module):
    """
    自研的 B-Rep 微观几何与拓扑生成器联合模型。
    - 输入：
      - 骨架先验节点类型 node_types (B, 32)
      - 骨架先验空间布局 node_bboxes (B, 32, 6)
      - 骨架先验边连接 adj_matrix (B, 32, 32)
      - 节点 masks (B, 32)
    - 输出：
      - 预测面片几何特征向量 (B, 64, 64) -> 64个面片，每个 64 维面 VAE Latent
      - 预测边线几何特征向量 (B, 160, 16) -> 160条线，每个 16 维线 VAE Latent
      - 预测面与线的拓扑连接概率图 (B, 160, 64) -> 面-线邻接矩阵
      - 预测面片自身的 6D BBox 空间排布 (B, 64, 6) -> 供属从围栏约束计算
    """
    def __init__(self, max_nodes=32, max_faces=64, max_edges=160):
        super(CustomGeomGenNet, self).__init__()
        self.max_nodes = max_nodes
        self.max_faces = max_faces
        self.max_edges = max_edges
        
        # 1. 骨架先验图嵌入与融合层 (Prior Encoder)
        self.node_embed = nn.Embedding(8, 128) # 6类Motif + padding 7
        self.edge_embed = nn.Embedding(7, 128) # 6类先验边 + padding 6
        
        self.enc_node_proj = nn.Linear(128 + 6, 128) # 合并类型嵌入与 BBox 坐标
        self.enc_edge_proj = nn.Linear(128 * max_nodes, 128)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=512, batch_first=True)
        self.prior_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 2. 微观面片几何特征预测层 (Face Decoder)
        # 可学习的面查询向量 (Face Queries)
        self.face_queries = nn.Parameter(torch.randn(1, max_faces, 256))
        
        decoder_layer_face = nn.TransformerDecoderLayer(d_model=256, nhead=4, dim_feedforward=512, batch_first=True)
        self.face_decoder = nn.TransformerDecoder(decoder_layer_face, num_layers=3)
        
        # 面特征输出投影头。VAE latent 实际可超过 [-1, 1]，这里不能用 Tanh 截断。
        self.face_latent_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64)
        )
        
        # 面自包围盒预测投影头：中心 [-1, 1]，尺度 [0, 1]，避免负尺寸。
        self.face_bbox_center_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 3),
            nn.Tanh()
        )
        self.face_bbox_scale_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )
        
        # 3. 微观边线几何特征预测层 (Edge Decoder)
        self.edge_queries = nn.Parameter(torch.randn(1, max_edges, 256))
        
        decoder_layer_edge = nn.TransformerDecoderLayer(d_model=256, nhead=4, dim_feedforward=512, batch_first=True)
        self.edge_decoder = nn.TransformerDecoder(decoder_layer_edge, num_layers=3)
        
        # 线特征输出投影头。保持线性输出，匹配 Edge VAE latent 的真实范围。
        self.edge_latent_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 16)
        )
        
        # 4. 微观面与线连接拓扑生成头 (Topology Predictor)
        # 利用面和线的隐特征，通过双线性投影矩阵生成二分类的 edgeFace_adj 连通性
        self.tp_proj_edge = nn.Linear(256, 128)
        self.tp_proj_face = nn.Linear(256, 128)
        
        self.tp_classifier = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1) # 输出 1 维对数值，通过 Sigmoid 做拓扑连通概率
        )

        # 5. 面-边耦合消息传递层。用预测拓扑概率进行双向聚合，让几何和拓扑共同成形。
        self.face_coupler = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256)
        )
        self.edge_coupler = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256)
        )
        self.face_norm = nn.LayerNorm(256)
        self.edge_norm = nn.LayerNorm(256)
        self.coupling_steps = 2

    def _topology_logits(self, f_features, e_features):
        """根据当前面/边特征预测 edgeFace 连通 logits。"""
        ef_edge = self.tp_proj_edge(e_features).unsqueeze(2).repeat(1, 1, self.max_faces, 1)
        ef_face = self.tp_proj_face(f_features).unsqueeze(1).repeat(1, self.max_edges, 1, 1)
        ef_combined = torch.cat([ef_edge, ef_face], dim=-1)
        return self.tp_classifier(ef_combined).squeeze(-1)

    def forward(self, node_types, node_bboxes, adj_matrix):
        """
        前向传播：
        S* 骨架先验条件引导微观几何与拓扑生成。
        [耦合解码版]：不读取真实 edgeFace_adj，先预测粗拓扑，再用预测拓扑做面-边双向消息传递。
        """
        B = node_types.shape[0]
        
        # 1. 骨架层特征编码
        n_embed = self.node_embed(node_types) # (B, 32, 128)
        n_combined = torch.cat([n_embed, node_bboxes], dim=-1) # (B, 32, 128 + 6)
        n_feat = self.enc_node_proj(n_combined) # (B, 32, 128)
        
        e_embed = self.edge_embed(adj_matrix) # (B, 32, 32, 128)
        e_feat = e_embed.view(B, self.max_nodes, -1)
        e_feat = self.enc_edge_proj(e_feat) # (B, 32, 128)
        
        prior_combined = torch.cat([n_feat, e_feat], dim=-1) # (B, 32, 256)
        
        # 提取融合了全局布局与拓扑结构的先图记忆上下文 (Memory)
        prior_memory = self.prior_encoder(prior_combined) # (B, 32, 256)
        
        # 2. 初始面/边解码。查询槽位由数据集中的规范排序赋予稳定含义，不依赖真实拓扑标签。
        f_queries = self.face_queries.repeat(B, 1, 1)
        f_features = self.face_decoder(f_queries, prior_memory) # (B, 64, 256)

        e_queries = self.edge_queries.repeat(B, 1, 1)
        e_features = self.edge_decoder(e_queries, prior_memory) # (B, 160, 256)

        # 3. 拓扑-几何耦合：用模型自己的拓扑概率聚合对侧特征。
        pred_edge_face_logits = self._topology_logits(f_features, e_features)
        for _ in range(self.coupling_steps):
            edge_face_prob = torch.sigmoid(pred_edge_face_logits) # (B, E, F)

            face_degree = edge_face_prob.sum(dim=1).clamp_min(1e-6) # (B, F)
            edge_degree = edge_face_prob.sum(dim=2).clamp_min(1e-6) # (B, E)

            face_msg = torch.bmm(edge_face_prob.transpose(1, 2), e_features) / face_degree.unsqueeze(-1)
            edge_msg = torch.bmm(edge_face_prob, f_features) / edge_degree.unsqueeze(-1)

            f_features = self.face_norm(f_features + self.face_coupler(torch.cat([f_features, face_msg], dim=-1)))
            e_features = self.edge_norm(e_features + self.edge_coupler(torch.cat([e_features, edge_msg], dim=-1)))
            pred_edge_face_logits = self._topology_logits(f_features, e_features)

        pred_face_latents = self.face_latent_head(f_features) # (B, 64, 64) -> 面 VAE Latent
        pred_edge_latents = self.edge_latent_head(e_features) # (B, 160, 16) -> 线 VAE Latent

        pred_face_centers = self.face_bbox_center_head(f_features)
        pred_face_scales = self.face_bbox_scale_head(f_features)
        pred_face_bboxes = torch.cat([pred_face_centers, pred_face_scales], dim=-1)
        
        return pred_face_latents, pred_edge_latents, pred_edge_face_logits, pred_face_bboxes


def compute_belonging_fence_loss(pred_face_bboxes, target_face_belong_matrix, target_node_bboxes, face_masks):
    """
    自研的核心贡献：弹性属从几何围栏损失 (Elastic Belonging Fence Loss)。
    [向量化优化版]：完全通过广播机制代替 2048 次嵌套循环，极大提升训练速度。
    - 如果面片 f_i 属于 Motif 节点 s_k（belong_matrix[i, k] == 1.0）；
    - 我们计算面片中心 centroid_f 与它所属面群中心 centroid_s 的绝对距离；
    - 一旦距离超出了面群包围盒半径（scale_s / 2.0），则使用 Relu 激活函数强力惩罚；
    - 在中后期强行让生成的面片归位在自己所属的面群布局内部，实现面片的微观空间对齐！
    """
    # 提取中心与尺寸，并通过增加维度进行广播对齐
    # 注意：DTG 的 pred_face_bboxes 格式为 [xmin, ymin, zmin, xmax, ymax, zmax]
    # 我们需要将其转换为中心 (centroid) 和尺寸 (scale)
    pred_xmin = pred_face_bboxes[:, :, 0:3]
    pred_xmax = pred_face_bboxes[:, :, 3:6]
    pred_centroids_val = (pred_xmin + pred_xmax) / 2.0
    pred_scales_val = pred_xmax - pred_xmin
    
    pred_centroids = pred_centroids_val.unsqueeze(2)     # (B, F, 1, 3)
    pred_scales = pred_scales_val.unsqueeze(2)            # (B, F, 1, 3)
    
    target_centroids = target_node_bboxes[:, :, 0:3].unsqueeze(1)  # (B, 1, N, 3)
    target_scales = target_node_bboxes[:, :, 3:6].unsqueeze(1)     # (B, 1, N, 3)
    
    # 1. 计算面片中心与面群中心的物理距离偏差 (B, F, N, 3)
    diff_centroids = torch.abs(pred_centroids - target_centroids)
    allowed_half_bounds = target_scales / 2.0                      # (B, 1, N, 3)
    out_of_bounds_error = F.relu(diff_centroids - allowed_half_bounds)
    
    # 2. 对尺寸进行约束，面片的尺寸不能超出面群的总尺寸 (B, F, N, 3)
    out_of_scale_error = F.relu(pred_scales - target_scales)
    
    # 将位置与尺寸偏差求和得到 (B, F, N) 的损失对矩阵
    total_error_matrix = (out_of_bounds_error + out_of_scale_error).sum(dim=-1)
    
    # 构造属从关系与面片存在掩膜：(B, F, N)
    belong_mask = target_face_belong_matrix * face_masks.unsqueeze(-1)
    
    # 计算有效对的平均损失
    masked_error = total_error_matrix * belong_mask
    loss_fence = masked_error.sum() / (belong_mask.sum() + 1e-6)
    
    return loss_fence
