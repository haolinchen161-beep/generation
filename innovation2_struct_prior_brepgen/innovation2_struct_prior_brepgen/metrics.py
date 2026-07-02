# -*- coding: utf-8 -*-
"""
程序名称：metrics.py
程序功能：
    本程序实现 CVAE 网络生成指标与精度的评估模块。
    通过对预测的分类概率取 argmax、对有效掩膜应用阈值判定，
    计算出 15 类精细的评价指标（Accuracies 与 L1 误差），并将结果分类汇总输出为 CSV 数据记录。

主要评估指标：
    1. 参数 L1 误差 (param_l1)
    2. 构型节点/关系有效性分类精度 (node_valid_acc, relation_valid_acc)
    3. 构型节点/关系分类精度 (node_type_acc, relation_type_acc)
    4. 面边界框与有效性精度 (face_bbox_l1, face_mask_acc)
    5. 面角色与节点分配分类精度 (face_role_acc, face_node_assignment_acc)
    6. 边/顶点有效性分类精度 (edge_mask_acc, vertex_mask_acc)
    7. 边-面/边-点邻接拓扑分类精度 (edgeFace_acc, edgeVert_acc)
    8. 顶点/边/面空间几何点云 L1 误差 (vert_wcs_l1, edge_wcs_l1, face_wcs_l1)

使用方法：
    由 run_innovation2.py 导入调用，用于 Epoch 结束后的 Validation 评估以及最终的 Test 评估。
"""

import torch
import numpy as np
from parameter_schema import class_parameter_mask_tensor

def binary_acc(logits, targets):
    """计算二分类准确率"""
    preds = (logits > 0.0).float()
    correct = (preds == targets).float()
    return correct.mean().item()

def masked_class_acc(logits, targets, mask):
    """计算在掩膜为 1.0 的元素上的多分类准确率"""
    if mask.sum().item() < 1e-5:
        return 1.0
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == targets).float()
    return (correct * mask).sum().item() / (mask.sum().item() + 1e-8)

def masked_regression_l1(pred, target, mask):
    """计算在掩膜为 1.0 的元素上的 L1 误差"""
    if mask.sum().item() < 1e-5:
        return 0.0
    diff = torch.abs(pred - target)
    diff_dims = len(target.shape) - len(mask.shape)
    mask_expanded = mask
    for _ in range(diff_dims):
        mask_expanded = mask_expanded.unsqueeze(-1)
        
    scale_factor = target.shape[2:].numel() if len(target.shape) > 2 else 1
    l1_masked = (diff * mask_expanded).sum().item() / (mask.sum().item() * scale_factor + 1e-8)
    return l1_masked

def evaluate_metrics(model, dataloader, device):
    """
    对指定的 DataLoader 跑一遍模型推理，计算并返回所有评估指标的平均值。
    """
    model.eval()
    
    total_samples = 0
    sum_param_l1 = 0.0
    
    sum_n_val_acc = 0.0
    sum_n_type_acc = 0.0
    sum_r_val_acc = 0.0
    sum_r_type_acc = 0.0

    sum_f_bbox_l1 = 0.0
    sum_f_mask_acc = 0.0
    sum_f_role_acc = 0.0
    sum_f_node_acc = 0.0

    sum_e_mask_acc = 0.0
    sum_v_mask_acc = 0.0
    sum_ef_acc = 0.0
    sum_ev_acc = 0.0

    sum_v_geom_l1 = 0.0
    sum_e_geom_l1 = 0.0
    sum_f_geom_l1 = 0.0

    with torch.no_grad():
        for batch in dataloader:
            # 数据迁往计算设备
            targets = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            c = targets["part_type_id"]
            
            # 训练评估模式下通过 Encoder 预测分布并重采样 z
            outputs = model(targets, c)
            
            B = c.shape[0]
            total_samples += B

            # 1. 参数向量 L1（按类别只统计真实有意义的参数）
            param_mask = outputs.get("pred_parameter_mask")
            if param_mask is None:
                param_mask = class_parameter_mask_tensor(c, device=device, dtype=outputs["pred_parameter_vector"].dtype)
            diff_param = torch.abs(outputs["pred_parameter_vector"] - targets["parameter_vector"]) * param_mask
            param_l1_per_sample = diff_param.sum(dim=-1) / (param_mask.sum(dim=-1) + 1e-8)
            sum_param_l1 += param_l1_per_sample.sum().item()

            # 2. 构型节点/关系有效性
            sum_n_val_acc += binary_acc(outputs["pred_node_valid_logits"], targets["config_node_valid"]) * B
            sum_r_val_acc += binary_acc(outputs["pred_relation_valid_logits"], targets["config_relation_valid"]) * B

            # 3. 构型节点/关系类型
            for b in range(B):
                n_mask = targets["config_node_valid"][b]
                sum_n_type_acc += masked_class_acc(outputs["pred_node_type_logits"][b], targets["config_node_type_ids"][b], n_mask)
                
                r_mask = targets["config_relation_valid"][b]
                sum_r_type_acc += masked_class_acc(outputs["pred_relation_type_logits"][b], targets["config_relation_type_ids"][b], r_mask)

            # 4. 面布局
            sum_f_mask_acc += binary_acc(outputs["pred_face_valid_logits"], targets["face_valid_mask"]) * B
            for b in range(B):
                f_mask = targets["face_valid_mask"][b]
                sum_f_bbox_l1 += masked_regression_l1(outputs["pred_face_bbox"][b], targets["face_bbox_norm"][b], f_mask)
                sum_f_role_acc += masked_class_acc(outputs["pred_face_role_logits"][b], targets["face_role_label"][b], f_mask)
                sum_f_node_acc += masked_class_acc(outputs["pred_face_node_logits"][b], targets["face_node_assignment_label"][b], f_mask)

            # 5. 边/点掩膜
            sum_e_mask_acc += binary_acc(outputs["pred_edge_valid_logits"], targets["edge_valid_mask"]) * B
            sum_v_mask_acc += binary_acc(outputs["pred_vertex_valid_logits"], targets["vertex_valid_mask"]) * B

            # 6. 拓扑邻接
            for b in range(B):
                e_mask = targets["edge_valid_mask"][b]
                # 边-面邻接包含 2 个面，需要对最后一维进行多分类评估
                sum_ef_acc += masked_class_acc(outputs["pred_edgeFace_logits"][b], targets["edgeFace_adj"][b], e_mask.unsqueeze(-1).expand(-1, 2))
                sum_ev_acc += masked_class_acc(outputs["pred_edgeVert_logits"][b], targets["edgeVert_adj"][b], e_mask.unsqueeze(-1).expand(-1, 2))

            # 7. 几何点云
            for b in range(B):
                v_mask = targets["vertex_valid_mask"][b]
                sum_v_geom_l1 += masked_regression_l1(outputs["pred_vert_wcs"][b], targets["vert_wcs_norm"][b], v_mask)

                e_mask = targets["edge_valid_mask"][b]
                sum_e_geom_l1 += masked_regression_l1(outputs["pred_edge_wcs"][b], targets["edge_wcs_norm"][b], e_mask)

                f_mask = targets["face_valid_mask"][b]
                sum_f_geom_l1 += masked_regression_l1(outputs["pred_face_wcs"][b], targets["face_wcs_norm"][b], f_mask)

    # 汇总计算平均值
    n_batches_scaled = total_samples if total_samples > 0 else 1
    
    return {
        "param_l1": round(sum_param_l1 / n_batches_scaled, 4),
        "node_valid_acc": round(sum_n_val_acc / n_batches_scaled, 4),
        "node_type_acc": round(sum_n_type_acc / n_batches_scaled, 4),
        "relation_valid_acc": round(sum_r_val_acc / n_batches_scaled, 4),
        "relation_type_acc": round(sum_r_type_acc / n_batches_scaled, 4),
        "face_bbox_l1": round(sum_f_bbox_l1 / n_batches_scaled, 4),
        "face_mask_acc": round(sum_f_mask_acc / n_batches_scaled, 4),
        "face_role_acc": round(sum_f_role_acc / n_batches_scaled, 4),
        "face_node_assignment_acc": round(sum_f_node_acc / n_batches_scaled, 4),
        "edge_mask_acc": round(sum_e_mask_acc / n_batches_scaled, 4),
        "vertex_mask_acc": round(sum_v_mask_acc / n_batches_scaled, 4),
        "edgeFace_acc": round(sum_ef_acc / n_batches_scaled, 4),
        "edgeVert_acc": round(sum_ev_acc / n_batches_scaled, 4),
        "vert_wcs_l1": round(sum_v_geom_l1 / n_batches_scaled, 4),
        "edge_wcs_l1": round(sum_e_geom_l1 / n_batches_scaled, 4),
        "face_wcs_l1": round(sum_f_geom_l1 / n_batches_scaled, 4)
    }
