# -*- coding: utf-8 -*-
"""
程序名称：models.py
程序功能：
    本程序定义了论文第二创新点核心的多模态生成模型：StructPriorBRepCVAE。
    该模型采用条件变分自编码器 (CVAE) 架构，以随机隐变量 z 和弱类别标签 c 作为生成条件。
    训练时，利用真实构型图先验与 B-Rep 全量几何拓扑张量，通过 Encoder 提取高维隐空间分布 (mu, logvar)；
    生成时，仅基于 z 采样与类别 embedding 经过多通道层级 Decoder 重建出构型图参数、面布局、拓扑邻接与低层点云几何。

主要模块功能：
    1. ConditionalEncoder: 融合类别、参数向量、构型图节点/关系类型、面/边/顶点包络框、邻接网络以及点云均值的多模态特征编码器。
    2. StructuralPriorDecoder: 重建生成 Gc* = (Vc*, Ec*, P*) 的结构先验解码器。
    3. FaceGroupLayoutDecoder: 对应第一创新点，由结构先验与隐空间特征生成面布局 (BBox) 与分配标签的解码器。
    4. BoundaryTopologyDecoder: 生成边-面邻接与边-点邻接拓扑连结关系的边界拓扑解码器。
    5. CurveSurfaceGeometryDecoder: 预测实体角点、边采样曲线点、面网格采样点的曲线曲面几何解码器。

使用方法：
    由 run_innovation2.py 载入实例化，支持 train 与 generate 模式的前向推理。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalEncoder(nn.Module):
    """
    多模态条件编码器 q(z | x, c)。
    将全量 B-Rep 实体与构型图先验特征编码到 64 维隐空间分布。
    """
    def __init__(self, latent_dim=64, class_dim=16):
        super().__init__()
        self.class_emb = nn.Embedding(5, class_dim)
        
        # 扁平化特征维数核算：
        # class_dim(16) + param(9) + config_nodes(9) + config_relations(18*3=54) 
        # + face_bbox(30*6=180) + edge_mask(68) + vert_mask(40) + edgeFace(68*2=136) + edgeVert(68*2=136)
        # + geometry_means(3+3+3=9)
        # 16 + 9 + 9 + 54 + 180 + 68 + 40 + 136 + 136 + 9 = 657
        self.encoder_mlp = nn.Sequential(
            nn.Linear(657, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x, c):
        B = c.shape[0]
        c_emb = self.class_emb(c) # [B, 16]
        
        # A. 提取构型先验扁平化
        param = x["parameter_vector"] # [B, 9]
        c_nodes = x["config_node_type_ids"].float() # [B, 9]
        
        # 提取关系特征
        r_src = x["config_relation_src"].float().unsqueeze(-1)
        r_dst = x["config_relation_dst"].float().unsqueeze(-1)
        r_type = x["config_relation_type_ids"].float().unsqueeze(-1)
        c_rels = torch.cat([r_src, r_dst, r_type], dim=-1).view(B, -1) # [B, 18*3]

        # B. 提取低层 B-Rep 拓扑扁平化
        f_bbox = x["face_bbox_norm"].view(B, -1) # [B, 180]
        e_mask = x["edge_valid_mask"] # [B, 68]
        v_mask = x["vertex_valid_mask"] # [B, 40]
        
        ef_adj = x["edgeFace_adj"].float().view(B, -1) # [B, 136]
        ev_adj = x["edgeVert_adj"].float().view(B, -1) # [B, 136]

        # C. 提取点云几何特征均值 (极大地压缩参数量并获取空间重心特征)
        f_mean = x["face_wcs_norm"].mean(dim=(1, 2, 3)) # [B, 3]
        e_mean = x["edge_wcs_norm"].mean(dim=(1, 2)) # [B, 3]
        v_mean = x["vert_wcs_norm"].mean(dim=1) # [B, 3]
        geom_means = torch.cat([f_mean, e_mean, v_mean], dim=-1) # [B, 9]

        # D. 拼接总多模态特征向量进行前向编码
        feat = torch.cat([
            c_emb, param, c_nodes, c_rels, f_bbox, e_mask, v_mask, ef_adj, ev_adj, geom_means
        ], dim=1)

        h = self.encoder_mlp(feat)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class StructuralPriorDecoder(nn.Module):
    """
    生成中间结构先验 Gc* = (Vc*, Ec*, P*) 的解码器。
    """
    def __init__(self, in_dim=80):
        super().__init__()
        # 输出 pred_parameter_vector: [9]
        self.fc_param = nn.Linear(in_dim, 9)
        
        # 节点先验分支：类型为 6 类
        self.fc_node_valid = nn.Linear(in_dim, 9)
        self.fc_node_type = nn.Linear(in_dim, 9 * 6)

        # 关系先验分支：类型为 6 类
        self.fc_rel_valid = nn.Linear(in_dim, 18)
        self.fc_rel_src = nn.Linear(in_dim, 18 * 9)
        self.fc_rel_dst = nn.Linear(in_dim, 18 * 9)
        self.fc_rel_type = nn.Linear(in_dim, 18 * 6)

    def forward(self, z_c):
        B = z_c.shape[0]
        pred_param = self.fc_param(z_c)

        pred_n_valid = self.fc_node_valid(z_c)
        pred_n_type = self.fc_node_type(z_c).view(B, 9, 6)

        pred_r_valid = self.fc_rel_valid(z_c)
        pred_r_src = self.fc_rel_src(z_c).view(B, 18, 9)
        pred_r_dst = self.fc_rel_dst(z_c).view(B, 18, 9)
        pred_r_type = self.fc_rel_type(z_c).view(B, 18, 6)

        return {
            "pred_parameter_vector": pred_param,
            "pred_node_valid_logits": pred_n_valid,
            "pred_node_type_logits": pred_n_type,
            "pred_relation_valid_logits": pred_r_valid,
            "pred_relation_src_logits": pred_r_src,
            "pred_relation_dst_logits": pred_r_dst,
            "pred_relation_type_logits": pred_r_type
        }

class FaceGroupLayoutDecoder(nn.Module):
    """
    生成三维面布局与对齐的解码器。
    """
    def __init__(self, in_dim=80):
        super().__init__()
        # 输出面 BBox 包络: [30, 6]
        self.fc_bbox = nn.Linear(in_dim, 30 * 6)
        # 面有效性掩膜: [30]
        self.fc_mask = nn.Linear(in_dim, 30)
        # 面物理角色: [30, 7] (6类 node_type + 1类 unassigned)
        self.fc_role = nn.Linear(in_dim, 30 * 7)
        # 面分配至构型节点的概率: [30, 10] (9类 config_node + 1类 unassigned)
        self.fc_node = nn.Linear(in_dim, 30 * 10)

    def forward(self, z_c):
        B = z_c.shape[0]
        pred_bbox = self.fc_bbox(z_c).view(B, 30, 6)
        pred_mask = self.fc_mask(z_c)
        pred_role = self.fc_role(z_c).view(B, 30, 7)
        pred_node = self.fc_node(z_c).view(B, 30, 10)

        return {
            "pred_face_bbox": pred_bbox,
            "pred_face_valid_logits": pred_mask,
            "pred_face_role_logits": pred_role,
            "pred_face_node_logits": pred_node
        }

class BoundaryTopologyDecoder(nn.Module):
    """
    生成 B-Rep 交线边、交线点掩膜以及邻接张量的边界拓扑解码器。
    """
    def __init__(self, in_dim=80):
        super().__init__()
        # 边掩膜: [68]
        self.fc_edge_mask = nn.Linear(in_dim, 68)
        # 顶点掩膜: [40]
        self.fc_vert_mask = nn.Linear(in_dim, 40)
        # 边-面邻接连接: [68, 2, 30]
        self.fc_edgeFace = nn.Linear(in_dim, 68 * 2 * 30)
        # 边-点邻接连接: [68, 2, 40]
        self.fc_edgeVert = nn.Linear(in_dim, 68 * 2 * 40)

    def forward(self, z_c):
        B = z_c.shape[0]
        pred_e_mask = self.fc_edge_mask(z_c)
        pred_v_mask = self.fc_vert_mask(z_c)
        pred_ef = self.fc_edgeFace(z_c).view(B, 68, 2, 30)
        pred_ev = self.fc_edgeVert(z_c).view(B, 68, 2, 40)

        return {
            "pred_edge_valid_logits": pred_e_mask,
            "pred_vertex_valid_logits": pred_v_mask,
            "pred_edgeFace_logits": pred_ef,
            "pred_edgeVert_logits": pred_ev
        }

class CurveSurfaceGeometryDecoder(nn.Module):
    """
    生成低层绝对几何坐标采样点云的几何解码器。
    """
    def __init__(self, in_dim=80):
        super().__init__()
        # 顶点 WCS 空间坐标: [40, 3]
        self.fc_vert = nn.Linear(in_dim, 40 * 3)
        # 边曲线 WCS 控制采样点: [68, 32, 3]
        self.fc_edge = nn.Linear(in_dim, 68 * 32 * 3)
        # 面网格 WCS 三维点云: [30, 32, 32, 3]
        self.fc_face = nn.Linear(in_dim, 30 * 32 * 32 * 3)

    def forward(self, z_c):
        B = z_c.shape[0]
        pred_v = self.fc_vert(z_c).view(B, 40, 3)
        pred_e = self.fc_edge(z_c).view(B, 68, 32, 3)
        pred_f = self.fc_face(z_c).view(B, 30, 32, 32, 3)

        return {
            "pred_vert_wcs": pred_v,
            "pred_edge_wcs": pred_e,
            "pred_face_wcs": pred_f
        }

class StructPriorBRepCVAE(nn.Module):
    """
    层级 CVAE 主网络结构。
    """
    def __init__(self, latent_dim=64, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.class_emb = nn.Embedding(5, 16)
        
        # 实例化编码器
        self.encoder = ConditionalEncoder(latent_dim=latent_dim, class_dim=16)

        # 拼接隐空间特征和类别特征后进入各个解码器 [64 + 16 = 80]
        self.struct_prior_decoder = StructuralPriorDecoder(in_dim=80)
        self.face_layout_decoder = FaceGroupLayoutDecoder(in_dim=80)
        self.boundary_topo_decoder = BoundaryTopologyDecoder(in_dim=80)
        self.geometry_decoder = CurveSurfaceGeometryDecoder(in_dim=80)

    def reparameterize(self, mu, logvar):
        """隐空间重参数化采样"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        """
        训练模式下的前向推理流程。
        """
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        
        c_emb = self.class_emb(c) # [B, 16]
        z_c = torch.cat([z, c_emb], dim=1) # [B, 80]

        outputs = {}
        outputs["mu"] = mu
        outputs["logvar"] = logvar
        outputs["z"] = z
        outputs.update(self.struct_prior_decoder(z_c))
        outputs.update(self.face_layout_decoder(z_c))
        outputs.update(self.boundary_topo_decoder(z_c))
        outputs.update(self.geometry_decoder(z_c))

        return outputs

    def generate(self, z, c):
        """
        生成/测试模式下的前向推理流程 (纯解码生成)。
        输入为标准高斯噪声 z 和类别标签 c，严禁输入任何真实几何/图拓扑。
        """
        B = z.shape[0]
        c_emb = self.class_emb(c) # [B, 16]
        z_c = torch.cat([z, c_emb], dim=1) # [B, 80]

        outputs = {}
        outputs["z"] = z
        outputs.update(self.struct_prior_decoder(z_c))
        outputs.update(self.face_layout_decoder(z_c))
        outputs.update(self.boundary_topo_decoder(z_c))
        outputs.update(self.geometry_decoder(z_c))

        return outputs
