# -*- coding: utf-8 -*-
"""
程序名称：losses.py
程序功能：
    本程序实现 StructPriorBRepCVAE 网络的完整损失计算函数 (compute_losses)。
    损失函数分为隐空间约束 (KL 散度)、结构先验生成损失、三维面布局损失、边界拓扑邻接分类损失以及低层点云几何拟合损失。
    支持在有效实体元素掩膜约束下计算 masked L1/SmoothL1/CrossEntropy，并且支持 KL 散度的线性 Warmup 预热机制。

主要模块功能：
    1. masked_cross_entropy: 带有掩膜的交叉熵计算辅助函数。
    2. masked_smooth_l1: 带有掩膜的 Smooth L1 回归损失计算辅助函数。
    3. compute_losses: 综合汇总 18 项子损失，并输出带权重的 L_total，同时返回各子项损失的数值指标用于日志记录。

使用方法：
    由 run_innovation2.py 在训练与验证迭代中导入调用。
"""

import torch
import torch.nn.functional as F

def masked_cross_entropy(logits, targets, mask):
    """
    带有有效掩膜过滤的多分类交叉熵损失计算。
    logits: [B, N, C] 或 [B, N, D, C] (最后一维为类别概率)
    targets: [B, N] 或 [B, N, D] (整数索引)
    mask: [B, N] 或 [B, N, D] (0.0/1.0 浮点型)
    """
    # 调整维度以符合 cross_entropy 的 C 维度要求
    # 如果是 3 维：[B, C, N]
    if len(logits.shape) == 3:
        logits = logits.transpose(1, 2)
    elif len(logits.shape) == 4:
        # 如果是 4 维：[B, C, N, D]
        logits = logits.permute(0, 3, 1, 2)
        
    loss_raw = F.cross_entropy(logits, targets, reduction='none')
    
    # 自动将 mask 维度升至与 loss_raw 一致以支持广播相乘
    diff_dims = len(loss_raw.shape) - len(mask.shape)
    mask_expanded = mask
    for _ in range(diff_dims):
        mask_expanded = mask_expanded.unsqueeze(-1)
        
    loss_masked = (loss_raw * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)
    return loss_masked

def masked_smooth_l1(pred, target, mask):
    """
    带有有效掩膜过滤的 Smooth L1 回归损失计算。
    """
    # 扩展 mask 形状与 target 完全一致
    # 假设 pred/target 形状为 [B, N, dim1, dim2...]
    # mask 形状为 [B, N]
    diff_dims = len(target.shape) - len(mask.shape)
    mask_expanded = mask
    for _ in range(diff_dims):
        mask_expanded = mask_expanded.unsqueeze(-1)
        
    loss_raw = F.smooth_l1_loss(pred, target, reduction='none')
    scale_factor = target.shape[2:].numel() if len(target.shape) > 2 else 1
    loss_masked = (loss_raw * mask_expanded).sum() / (mask.sum() * scale_factor + 1e-8)
    return loss_masked

def compute_losses(outputs, targets, epoch, warmup_epochs=10):
    """
    计算 CVAE 联合损失。
    """
    # 1. KL 散度与 Warmup 预热
    mu = outputs["mu"]
    logvar = outputs["logvar"]
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    
    if epoch <= warmup_epochs:
        kl_weight = 0.01 * (epoch / float(warmup_epochs))
    else:
        kl_weight = 0.01

    # 2. 结构先验损失 (参数、节点与关系)
    # 参数回归
    pred_param = outputs["pred_parameter_vector"]
    target_param = targets["parameter_vector"]
    loss_param = F.smooth_l1_loss(pred_param, target_param)

    # 节点有效性与节点类型
    pred_n_val = outputs["pred_node_valid_logits"]
    target_n_val = targets["config_node_valid"]
    loss_node_valid = F.binary_cross_entropy_with_logits(pred_n_val, target_n_val)

    pred_n_type = outputs["pred_node_type_logits"]
    target_n_type = targets["config_node_type_ids"]
    loss_node_type = masked_cross_entropy(pred_n_type, target_n_type, target_n_val)

    # 关系有效性、端点与关系类型
    pred_r_val = outputs["pred_relation_valid_logits"]
    target_r_val = targets["config_relation_valid"]
    loss_relation_valid = F.binary_cross_entropy_with_logits(pred_r_val, target_r_val)

    pred_r_src = outputs["pred_relation_src_logits"]
    target_r_src = targets["config_relation_src"]
    loss_relation_src = masked_cross_entropy(pred_r_src, target_r_src, target_r_val)

    pred_r_dst = outputs["pred_relation_dst_logits"]
    target_r_dst = targets["config_relation_dst"]
    loss_relation_dst = masked_cross_entropy(pred_r_dst, target_r_dst, target_r_val)

    pred_r_type = outputs["pred_relation_type_logits"]
    target_r_type = targets["config_relation_type_ids"]
    loss_relation_type = masked_cross_entropy(pred_r_type, target_r_type, target_r_val)

    # 3. 三维面布局损失
    # 面有效掩膜
    pred_f_mask = outputs["pred_face_valid_logits"]
    target_f_mask = targets["face_valid_mask"]
    loss_face_mask = F.binary_cross_entropy_with_logits(pred_f_mask, target_f_mask)

    # 面 BBox
    pred_f_bbox = outputs["pred_face_bbox"]
    target_f_bbox = targets["face_bbox_norm"]
    loss_face_bbox = masked_smooth_l1(pred_f_bbox, target_f_bbox, target_f_mask)

    # 面角色
    pred_f_role = outputs["pred_face_role_logits"]
    target_f_role = targets["face_role_label"]
    loss_face_role = masked_cross_entropy(pred_f_role, target_f_role, target_f_mask)

    # 面节点分配
    pred_f_node = outputs["pred_face_node_logits"]
    target_f_node = targets["face_node_assignment_label"]
    loss_face_node_assignment = masked_cross_entropy(pred_f_node, target_f_node, target_f_mask)

    # 4. 边界拓扑损失
    # 边/点有效性掩膜
    pred_e_mask = outputs["pred_edge_valid_logits"]
    target_e_mask = targets["edge_valid_mask"]
    loss_edge_mask = F.binary_cross_entropy_with_logits(pred_e_mask, target_e_mask)

    pred_v_mask = outputs["pred_vertex_valid_logits"]
    target_v_mask = targets["vertex_valid_mask"]
    loss_vertex_mask = F.binary_cross_entropy_with_logits(pred_v_mask, target_v_mask)

    # 边-面邻接与边-点邻接 (使用边有效掩膜过滤)
    pred_ef = outputs["pred_edgeFace_logits"]
    target_ef = targets["edgeFace_adj"]
    loss_edgeFace = masked_cross_entropy(pred_ef, target_ef, target_e_mask)

    pred_ev = outputs["pred_edgeVert_logits"]
    target_ev = targets["edgeVert_adj"]
    loss_edgeVert = masked_cross_entropy(pred_ev, target_ev, target_e_mask)

    # 5. 点云几何损失
    # 顶点几何
    pred_v_geom = outputs["pred_vert_wcs"]
    target_v_geom = targets["vert_wcs_norm"]
    loss_vert_geom = masked_smooth_l1(pred_v_geom, target_v_geom, target_v_mask)

    # 边几何
    pred_e_geom = outputs["pred_edge_wcs"]
    target_e_geom = targets["edge_wcs_norm"]
    loss_edge_geom = masked_smooth_l1(pred_e_geom, target_e_geom, target_e_mask)

    # 面几何
    pred_f_geom = outputs["pred_face_wcs"]
    target_f_geom = targets["face_wcs_norm"]
    loss_face_geom = masked_smooth_l1(pred_f_geom, target_f_geom, target_f_mask)

    # 6. 计算加权总损失
    total_loss = (
        kl_weight * kl_loss
        + 1.0 * loss_param
        + 0.5 * loss_node_valid
        + 0.5 * loss_node_type
        + 0.5 * loss_relation_valid
        + 0.5 * loss_relation_src
        + 0.5 * loss_relation_dst
        + 0.5 * loss_relation_type
        + 1.0 * loss_face_bbox
        + 0.5 * loss_face_mask
        + 0.5 * loss_face_role
        + 0.5 * loss_face_node_assignment
        + 1.0 * loss_edge_mask
        + 1.0 * loss_vertex_mask
        + 1.0 * loss_edgeFace
        + 1.0 * loss_edgeVert
        + 0.5 * loss_vert_geom
        + 1.0 * loss_edge_geom
        + 1.0 * loss_face_geom
    )

    return {
        "loss_total": total_loss,
        "loss_kl": kl_loss,
        "loss_param": loss_param,
        "loss_node_valid": loss_node_valid,
        "loss_node_type": loss_node_type,
        "loss_relation_valid": loss_relation_valid,
        "loss_relation_src": loss_relation_src,
        "loss_relation_dst": loss_relation_dst,
        "loss_relation_type": loss_relation_type,
        "loss_face_bbox": loss_face_bbox,
        "loss_face_mask": loss_face_mask,
        "loss_face_role": loss_face_role,
        "loss_face_node_assignment": loss_face_node_assignment,
        "loss_edge_mask": loss_edge_mask,
        "loss_vertex_mask": loss_vertex_mask,
        "loss_edgeFace": loss_edgeFace,
        "loss_edgeVert": loss_edgeVert,
        "loss_vert_geom": loss_vert_geom,
        "loss_edge_geom": loss_edge_geom,
        "loss_face_geom": loss_face_geom,
        "kl_weight": kl_weight
    }
