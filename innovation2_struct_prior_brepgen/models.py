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

class BoundedParameterDecoder(nn.Module):
    """
    工程有界参数解码器。
    使用 Sigmoid 激活函数将 continuous parameters 映射到工程尺度边界，
    并将离散加筋数设计为 categorical 分类输出概率。
    """
    def __init__(self, in_dim=80):
        super().__init__()
        # 前 8 维连续参数的回归输出
        self.fc_param = nn.Linear(in_dim, 8)
        # 第 9 维离散加筋数（0-5）分类输出
        self.fc_rib_count = nn.Linear(in_dim, 6)

    def forward(self, z_c):
        B = z_c.shape[0]
        
        # A. 连续参数有界输出
        x_cont = self.fc_param(z_c)
        s_cont = torch.sigmoid(x_cont)

        # 取出单列进行对应区间线性映射
        length_phys = 120.0 + (500.0 - 120.0) * s_cont[:, 0]
        width_phys = 30.0 + (220.0 - 30.0) * s_cont[:, 1]
        thickness_phys = 1.8 + (3.5 - 1.8) * s_cont[:, 2]
        height_phys = 20.0 + (120.0 - 20.0) * s_cont[:, 3]
        flange_width_phys = 15.0 + (80.0 - 15.0) * s_cont[:, 4]
        rib_width_phys = 8.0 + (50.0 - 8.0) * s_cont[:, 5]
        rib_height_phys = 10.0 + (100.0 - 10.0) * s_cont[:, 6]

        # 动态圆角工程边界 fillet_radius: [1.5*t, 0.25*min(w, h, f_w)]
        lower_bound = 1.5 * thickness_phys
        # 确保 upper_bound 始终大于 lower_bound，避免区间退化
        min_dim = torch.minimum(
            torch.minimum(width_phys, height_phys), 
            flange_width_phys
        )
        upper_bound = torch.maximum(lower_bound + 0.5, 0.25 * min_dim)
        fillet_radius_phys = lower_bound + (upper_bound - lower_bound) * s_cont[:, 7]

        # B. 离散加筋数输出
        rib_count_logits = self.fc_rib_count(z_c) # [B, 6]
        # 在生成或推理时获取最可能的加筋数作为物理值
        rib_count_val = torch.argmax(rib_count_logits, dim=-1).float()

        # 组装 9 维物理参数向量 pred_parameter_vector_physical: [B, 9] (对齐 schema 字段顺序)
        pred_param_phys = torch.stack([
            length_phys, width_phys, thickness_phys, height_phys,
            flange_width_phys, rib_width_phys, rib_height_phys, rib_count_val, fillet_radius_phys
        ], dim=1)

        # 组装 9 维归一化参数向量 pred_parameter_vector_norm
        length_norm = length_phys / 200.0
        width_norm = width_phys / 200.0
        thickness_norm = thickness_phys / 3.0
        height_norm = height_phys / 50.0
        flange_width_norm = flange_width_phys / 50.0
        rib_width_norm = rib_width_phys / 50.0
        rib_height_norm = rib_height_phys / 50.0
        rib_count_norm = rib_count_val
        fillet_radius_norm = fillet_radius_phys / 3.0

        pred_param_norm = torch.stack([
            length_norm, width_norm, thickness_norm, height_norm,
            flange_width_norm, rib_width_norm, rib_height_norm, rib_count_norm, fillet_radius_norm
        ], dim=1)

        return {
            "pred_parameter_vector_norm": pred_param_norm,
            "pred_parameter_vector_physical": pred_param_phys,
            "pred_rib_count_logits": rib_count_logits
        }

class StructuralPriorEmbedding(nn.Module):
    """
    将生成的 Gc* = (Vc*, Ec*, P*) 嵌入映射为 128 维表示向量 struct_emb。
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        # 输入维度: P*(9) + n_val(9) + n_type(9*6=54) + r_val(18) + r_src(18*9=162) + r_dst(18*9=162) + r_type(18*6=108) = 522
        self.net = nn.Sequential(
            nn.Linear(522, hidden_dim),
            nn.ReLU()
        )

    def forward(self, prior_dict):
        B = prior_dict["pred_parameter_vector"].shape[0]
        p = prior_dict["pred_parameter_vector"]
        n_val = prior_dict["pred_node_valid_logits"]
        n_type = prior_dict["pred_node_type_logits"].view(B, -1)
        r_val = prior_dict["pred_relation_valid_logits"]
        r_src = prior_dict["pred_relation_src_logits"].view(B, -1)
        r_dst = prior_dict["pred_relation_dst_logits"].view(B, -1)
        r_type = prior_dict["pred_relation_type_logits"].view(B, -1)

        feat = torch.cat([p, n_val, n_type, r_val, r_src, r_dst, r_type], dim=-1)
        return self.net(feat)

class FaceLayoutEmbedding(nn.Module):
    """
    将生成的 Face BBox、Masks、Roles 和分配嵌入映射为 128 维表示向量 face_layout_emb。
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        # 输入维度: bbox(30*6=180) + mask(30) + role(30*7=210) + node_assign(30*10=300) = 720
        self.net = nn.Sequential(
            nn.Linear(720, hidden_dim),
            nn.ReLU()
        )

    def forward(self, layout_dict):
        B = layout_dict["pred_face_bbox"].shape[0]
        bbox = layout_dict["pred_face_bbox"].view(B, -1)
        mask = layout_dict["pred_face_valid_logits"]
        role = layout_dict["pred_face_role_logits"].view(B, -1)
        node = layout_dict["pred_face_node_logits"].view(B, -1)

        feat = torch.cat([bbox, mask, role, node], dim=-1)
        return self.net(feat)

class TopologyEmbedding(nn.Module):
    """
    将生成的 Edge/Vert Masks 以及邻接拓扑嵌入映射为 128 维表示向量 topology_emb。
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        # 输入维度: e_mask(68) + v_mask(40) + ef(68*2*30=4080) + ev(68*2*40=5440) = 9628
        self.net = nn.Sequential(
            nn.Linear(9628, hidden_dim),
            nn.ReLU()
        )

    def forward(self, topo_dict):
        B = topo_dict["pred_edge_valid_logits"].shape[0]
        e_mask = topo_dict["pred_edge_valid_logits"]
        v_mask = topo_dict["pred_vertex_valid_logits"]
        ef = topo_dict["pred_edgeFace_logits"].view(B, -1)
        ev = topo_dict["pred_edgeVert_logits"].view(B, -1)

        feat = torch.cat([e_mask, v_mask, ef, ev], dim=-1)
        return self.net(feat)

class StructuralPriorDecoder(nn.Module):
    """
    生成中间结构先验 Gc* = (Vc*, Ec*, P*) 的解码器。
    """
    def __init__(self, in_dim=80):
        super().__init__()
        # 使用工程有界参数解码器
        self.bounded_param_decoder = BoundedParameterDecoder(in_dim)
        
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
        bounded_res = self.bounded_param_decoder(z_c)

        pred_n_valid = self.fc_node_valid(z_c)
        pred_n_type = self.fc_node_type(z_c).view(B, 9, 6)

        pred_r_valid = self.fc_rel_valid(z_c)
        pred_r_src = self.fc_rel_src(z_c).view(B, 18, 9)
        pred_r_dst = self.fc_rel_dst(z_c).view(B, 18, 9)
        pred_r_type = self.fc_rel_type(z_c).view(B, 18, 6)

        return {
            "pred_parameter_vector": bounded_res["pred_parameter_vector_norm"],
            "pred_parameter_vector_physical": bounded_res["pred_parameter_vector_physical"],
            "pred_rib_count_logits": bounded_res["pred_rib_count_logits"],
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
    接收输入：z_c 拼接 struct_emb [in_dim = 80 + 128 = 208]。
    """
    def __init__(self, in_dim=208):
        super().__init__()
        # 输出面 BBox 包络: [30, 6]
        self.fc_bbox = nn.Linear(in_dim, 30 * 6)
        # 面有效性掩膜: [30]
        self.fc_mask = nn.Linear(in_dim, 30)
        # 面物理角色: [30, 7]
        self.fc_role = nn.Linear(in_dim, 30 * 7)
        # 面分配至构型节点的概率: [30, 10]
        self.fc_node = nn.Linear(in_dim, 30 * 10)

    def forward(self, feat_combined):
        B = feat_combined.shape[0]
        pred_bbox = self.fc_bbox(feat_combined).view(B, 30, 6)
        pred_mask = self.fc_mask(feat_combined)
        pred_role = self.fc_role(feat_combined).view(B, 30, 7)
        pred_node = self.fc_node(feat_combined).view(B, 30, 10)

        return {
            "pred_face_bbox": pred_bbox,
            "pred_face_valid_logits": pred_mask,
            "pred_face_role_logits": pred_role,
            "pred_face_node_logits": pred_node
        }

class BoundaryTopologyDecoder(nn.Module):
    """
    生成 B-Rep 交线边、交线点掩膜以及邻接张量的边界拓扑解码器。
    接收输入：z_c 拼接 struct_emb 拼接 face_layout_emb [in_dim = 80 + 128 + 128 = 336]。
    """
    def __init__(self, in_dim=336):
        super().__init__()
        # 边掩膜: [68]
        self.fc_edge_mask = nn.Linear(in_dim, 68)
        # 顶点掩膜: [40]
        self.fc_vert_mask = nn.Linear(in_dim, 40)
        # 边-面邻接连接: [68, 2, 30]
        self.fc_edgeFace = nn.Linear(in_dim, 68 * 2 * 30)
        # 边-点邻接连接: [68, 2, 40]
        self.fc_edgeVert = nn.Linear(in_dim, 68 * 2 * 40)

    def forward(self, feat_combined):
        B = feat_combined.shape[0]
        pred_e_mask = self.fc_edge_mask(feat_combined)
        pred_v_mask = self.fc_vert_mask(feat_combined)
        pred_ef = self.fc_edgeFace(feat_combined).view(B, 68, 2, 30)
        pred_ev = self.fc_edgeVert(feat_combined).view(B, 68, 2, 40)

        return {
            "pred_edge_valid_logits": pred_e_mask,
            "pred_vertex_valid_logits": pred_v_mask,
            "pred_edgeFace_logits": pred_ef,
            "pred_edgeVert_logits": pred_ev
        }

class CurveSurfaceGeometryDecoder(nn.Module):
    """
    生成低层绝对几何坐标采样点云的几何解码器。
    接收输入：z_c 拼接 struct_emb, face_layout_emb, topology_emb [in_dim = 80 + 128 + 128 + 128 = 464]。
    """
    def __init__(self, in_dim=464):
        super().__init__()
        # 顶点 WCS 空间坐标: [40, 3]
        self.fc_vert = nn.Linear(in_dim, 40 * 3)
        # 边曲线 WCS 控制采样点: [68, 32, 3]
        self.fc_edge = nn.Linear(in_dim, 68 * 32 * 3)
        # 面网格 WCS 三维点云: [30, 32, 32, 3]
        self.fc_face = nn.Linear(in_dim, 30 * 32 * 32 * 3)

    def forward(self, feat_combined):
        B = feat_combined.shape[0]
        pred_v = self.fc_vert(feat_combined).view(B, 40, 3)
        pred_e = self.fc_edge(feat_combined).view(B, 68, 32, 3)
        pred_f = self.fc_face(feat_combined).view(B, 30, 32, 32, 3)

        return {
            "pred_vert_wcs": pred_v,
            "pred_edge_wcs": pred_e,
            "pred_face_wcs": pred_f
        }

class StructPriorBRepCVAE(nn.Module):
    """
    升级后的层级 CVAE 主网络结构 (v2)。
    基于图-几何层级流式条件进行预测。
    """
    def __init__(self, latent_dim=64, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.class_emb = nn.Embedding(5, 16)
        
        # 实例化条件编码器
        self.encoder = ConditionalEncoder(latent_dim=latent_dim, class_dim=16)

        # 实例化三个 Embedding 中间表示连通模块
        self.struct_embedder = StructuralPriorEmbedding(hidden_dim=hidden_dim)
        self.face_layout_embedder = FaceLayoutEmbedding(hidden_dim=hidden_dim)
        self.topo_embedder = TopologyEmbedding(hidden_dim=hidden_dim)

        # 实例化具有层级依赖输入维度的四大 Decoders
        self.struct_prior_decoder = StructuralPriorDecoder(in_dim=80)                      # z_c: 80
        self.face_layout_decoder = FaceGroupLayoutDecoder(in_dim=80 + hidden_dim)          # z_c + struct: 208
        self.boundary_topo_decoder = BoundaryTopologyDecoder(in_dim=80 + 2 * hidden_dim)   # z_c + struct + layout: 336
        self.geometry_decoder = CurveSurfaceGeometryDecoder(in_dim=80 + 3 * hidden_dim)    # z_c + struct + layout + topo: 464

    def reparameterize(self, mu, logvar):
        """隐空间重参数化采样"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        """
        训练模式下的前向推理层级逻辑。
        """
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        
        c_emb = self.class_emb(c) # [B, 16]
        z_c = torch.cat([z, c_emb], dim=1) # [B, 80]

        outputs = {}
        outputs["mu"] = mu
        outputs["logvar"] = logvar
        outputs["z"] = z

        # 1. 结构先验解码并生成 struct_emb
        struct_res = self.struct_prior_decoder(z_c)
        outputs.update(struct_res)
        struct_emb = self.struct_embedder(struct_res)

        # 2. 拼接面布局输入并解码，生成 face_layout_emb
        feat_layout = torch.cat([z_c, struct_emb], dim=1)
        layout_res = self.face_layout_decoder(feat_layout)
        outputs.update(layout_res)
        face_layout_emb = self.face_layout_embedder(layout_res)

        # 3. 拼接边界拓扑输入并解码，生成 topology_emb
        feat_topo = torch.cat([z_c, struct_emb, face_layout_emb], dim=1)
        topo_res = self.boundary_topo_decoder(feat_topo)
        outputs.update(topo_res)
        topology_emb = self.topo_embedder(topo_res)

        # 4. 拼接几何点云输入并解码
        feat_geom = torch.cat([z_c, struct_emb, face_layout_emb, topology_emb], dim=1)
        geom_res = self.geometry_decoder(feat_geom)
        outputs.update(geom_res)

        return outputs

    def generate(self, z, c):
        """
        生成/测试模式下的前向推理层级逻辑 (纯随机解码生成)。
        输入为标准高斯噪声 z 和类别标签 c，严禁输入任何真实几何/图拓扑。
        """
        B = z.shape[0]
        c_emb = self.class_emb(c) # [B, 16]
        z_c = torch.cat([z, c_emb], dim=1) # [B, 80]

        outputs = {}
        outputs["z"] = z

        # 1. 结构先验解码并生成 struct_emb
        struct_res = self.struct_prior_decoder(z_c)
        outputs.update(struct_res)
        struct_emb = self.struct_embedder(struct_res)

        # 2. 拼接面布局输入并解码，生成 face_layout_emb
        feat_layout = torch.cat([z_c, struct_emb], dim=1)
        layout_res = self.face_layout_decoder(feat_layout)
        outputs.update(layout_res)
        face_layout_emb = self.face_layout_embedder(layout_res)

        # 3. 拼接边界拓扑输入并解码，生成 topology_emb
        feat_topo = torch.cat([z_c, struct_emb, face_layout_emb], dim=1)
        topo_res = self.boundary_topo_decoder(feat_topo)
        outputs.update(topo_res)
        topology_emb = self.topo_embedder(topo_res)

        # 4. 拼接几何点云输入并解码
        feat_geom = torch.cat([z_c, struct_emb, face_layout_emb, topology_emb], dim=1)
        geom_res = self.geometry_decoder(feat_geom)
        outputs.update(geom_res)

        return outputs
