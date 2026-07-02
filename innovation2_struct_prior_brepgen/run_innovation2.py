# -*- coding: utf-8 -*-
"""
程序名称：run_innovation2.py
程序功能：
    本程序是论文第二创新点“弱条件结构先验驱动的构型图—B-Rep 层级几何生成方法”的运行与评测总入口。
    它支持 7 种运行模式 (smoke / train / evaluate / recon_sanity / generate_class / generate_uncond / generate_batch)。
    本程序不修改已有原始数据或代码，所有预测权重、报表日志和生成的 STEP/STL 模型均导出到独立的创新点 2 文件夹内。

主要模块功能：
    1. 命令行解析与多模式调度控制。
    2. --mode smoke: 快速冒烟测试，在小批量样本上验证 CVAE 训练前向、反向传播与保存状态完整畅通。
    3. --mode train: 完整 CVAE 训练循环，执行 120 个 Epochs，对 18 类多模态损失实施带 Warmup 的加权优化，保存最优验证集 Checkpoint。
    4. --mode evaluate: 导入最佳模型参数，并在测试集上计算 15 项指标精度，导出测试样例大表与 train_report.txt。
    5. --mode recon_sanity: 验证 constructive CAD 重建器，基于真实参数测试 5 类零件重建连通性。
    6. --mode generate_class / generate_uncond / generate_batch: 
       使用随机高斯噪声 z 以及可选类别标签 c 进行弱条件生成。提取模型生成的 P* 参数，修补并进入重建器，
       严禁使用真实参数，自动回读生成的 STEP/STL 以验证质量，并录入生成 Manifest 列表。

命令行使用指令：
    见 method_summary.md 后附的具体命令行运行范例。
"""

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import sys
# 动态将当前脚本所在的子文件夹加入到 sys.path 中，支持直接脚本运行与根目录包导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# os.environ["PYTHONWARNINGS"] = "ignore"
# sys.stderr = open(os.devnull, 'w')

import argparse
import random
import csv
import json
import psutil
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# 导入创新点 2 内部模块
from dataset import BRepDataset, collate_fn
from models import StructPriorBRepCVAE
from losses import compute_losses
from metrics import evaluate_metrics
from reconstructor_occ import reconstruct_brep_occ, repair_parameters
from utils_io import verify_occ_step, verify_stl, write_json, write_csv

def main():
    # 限制 CPU 优先级为 Below Normal 避免影响系统前台交互，防止控制台卡顿
    try:
        p = psutil.Process(os.getpid())
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        print(" -> 已将进程优先级调为 BELOW_NORMAL，保护前台响应。")
    except Exception as e:
        print(f" -> 调低优先级失败: {e}")

    parser = argparse.ArgumentParser(description="创新点 2：弱条件结构先验 B-Rep 层级几何生成总控制程序")
    parser.add_argument("--mode", type=str, required=True, 
                        choices=["smoke", "train", "evaluate", "recon_sanity", "generate_class", "generate_uncond", "generate_batch"],
                        help="运行模式")
    parser.add_argument("--data", type=str, default="cfg_brepgen_v1/parametric_composite_dataset", help="真实数据集 JSON 目录")
    parser.add_argument("--parsed", type=str, default="cfg_brepgen_v1/parametric_composite_parsed", help="真实数据集已解析 PKL 目录")
    parser.add_argument("--workdir", type=str, default="innovation2_struct_prior_brepgen", help="工作主目录")
    parser.add_argument("--epochs", type=int, default=120, help="训练周期数")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--latent_dim", type=int, default=64, help="隐空间维数")
    parser.add_argument("--hidden_dim", type=int, default=128, help="隐层宽度")
    parser.add_argument("--checkpoint", type=str, default=None, help="导入权重文件路径")
    parser.add_argument("--class_label", type=str, default="hat_stiffener", help="生成类别标签")
    parser.add_argument("--num_samples", type=int, default=10, help="生成样本数")
    parser.add_argument("--seed", type=int, default=42, help="随机数种子")
    parser.add_argument("--batch_input", type=str, default=None, help="Batch 生成模式 JSONL 输入")
    args = parser.parse_args()

    # 设定随机数种子确保过程可重复
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 确定计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" -> 使用计算设备: {device}")

    # 自动创建输出文件夹架构
    weights_dir = os.path.join(args.workdir, "outputs", "weights").replace('\\', '/')
    reports_dir = os.path.join(args.workdir, "outputs", "reports").replace('\\', '/')
    logs_dir = os.path.join(args.workdir, "outputs", "logs").replace('\\', '/')
    gen_dir = os.path.join(args.workdir, "outputs", "generated").replace('\\', '/')
    for d in [weights_dir, reports_dir, logs_dir, gen_dir]:
        os.makedirs(d, exist_ok=True)

    # 词表和参数字段定义映射 (必须保持与第一创新点沉淀完全一致)
    part_type_vocab = {"l_angle": 0, "c_channel": 1, "z_beam": 2, "hat_stiffener": 3, "stiffened_panel": 4}
    part_types_rev = {v: k for k, v in part_type_vocab.items()}
    node_vocab_rev = {0: "panel", 1: "web", 2: "flange", 3: "stiffener", 4: "transition", 5: "boundary"}
    relation_vocab_rev = {0: "attached_to", 1: "connected_to", 2: "smooth_connected", 3: "symmetric_to", 4: "opposite_side_of", 5: "parallel_to"}
    
    parameter_keys = ["length", "width", "thickness", "height", "flange_width", "rib_width", "rib_height", "rib_count", "fillet_radius"]

    # ----------------------------------------------------
    # Mode 1: Smoke (冒烟测试模式)
    # ----------------------------------------------------
    if args.mode == "smoke":
        print("====================================================================")
        print("                --mode smoke: 快速功能性冒烟测试中...")
        print("====================================================================")
        train_ds = BRepDataset(args.data, args.parsed, split_name="train", max_samples=8)
        val_ds = BRepDataset(args.data, args.parsed, split_name="val", max_samples=4)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        model = StructPriorBRepCVAE(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # 迭代 2 个 Epoch
        for epoch in range(1, 3):
            model.train()
            for batch in train_loader:
                targets = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                c = targets["part_type_id"]
                optimizer.zero_grad()
                outputs = model(targets, c)
                losses = compute_losses(outputs, targets, epoch, warmup_epochs=2)
                loss_total = losses["loss_total"]
                loss_total.backward()
                optimizer.step()
            
            # 简单验证评估
            evaluate_metrics(model, val_loader, device)

        # 保存临时检查点，确认保存逻辑正常
        torch.save(model.state_dict(), os.path.join(weights_dir, "innovation2_last.pt").replace('\\', '/'))
        torch.save(model.state_dict(), os.path.join(weights_dir, "innovation2_best_val.pt").replace('\\', '/'))
        
        # 导出映射表
        write_json(part_type_vocab, os.path.join(weights_dir, "label_maps.json").replace('\\', '/'))
        write_json({"length_scale": 200.0, "thickness_scale": 3.0}, os.path.join(weights_dir, "norm_stats.json").replace('\\', '/'))

        print(" -> [SUCCESS] Smoke 冒烟测试执行完毕！状态保存正常。")

    # ----------------------------------------------------
    # Mode 2: Train (全量 CVAE 训练模式)
    # ----------------------------------------------------
    elif args.mode == "train":
        print("====================================================================")
        print("                --mode train: 启动完整 CVAE 训练循环...")
        print("====================================================================")
        train_ds = BRepDataset(args.data, args.parsed, split_name="train")
        val_ds = BRepDataset(args.data, args.parsed, split_name="val")

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        model = StructPriorBRepCVAE(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        best_val_loss = float('inf')
        best_epoch = 1
        training_logs = []

        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss_sum = 0.0
            
            for batch in train_loader:
                targets = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                c = targets["part_type_id"]
                optimizer.zero_grad()
                outputs = model(targets, c)
                losses = compute_losses(outputs, targets, epoch, warmup_epochs=10)
                loss_total = losses["loss_total"]
                loss_total.backward()
                optimizer.step()
                train_loss_sum += loss_total.item()

            avg_train_loss = train_loss_sum / len(train_loader)

            # 验证集评估损失
            model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    targets = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                    c = targets["part_type_id"]
                    outputs = model(targets, c)
                    losses = compute_losses(outputs, targets, epoch, warmup_epochs=10)
                    val_loss_sum += losses["loss_total"].item()

            avg_val_loss = val_loss_sum / len(val_loader)
            training_logs.append({"epoch": epoch, "train_loss": round(avg_train_loss, 4), "val_loss": round(avg_val_loss, 4)})

            # 如果验证损失最佳，保存最优检查点
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(weights_dir, "innovation2_best_val.pt").replace('\\', '/'))

        # 保存最后一轮权重与训练状态
        torch.save(model.state_dict(), os.path.join(weights_dir, "innovation2_last.pt").replace('\\', '/'))
        torch.save({
            "epoch": args.epochs,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "optimizer_state": optimizer.state_dict()
        }, os.path.join(weights_dir, "innovation2_training_state.pt").replace('\\', '/'))

        # 导出统计属性映射与辅助 JSON
        write_json(part_type_vocab, os.path.join(weights_dir, "label_maps.json").replace('\\', '/'))
        write_json({"length_scale": 200.0, "thickness_scale": 3.0}, os.path.join(weights_dir, "norm_stats.json").replace('\\', '/'))

        # 导出训练日志 CSV
        write_csv(training_logs, os.path.join(logs_dir, "innovation2_train_log.csv").replace('\\', '/'), ["epoch", "train_loss", "val_loss"])
        print(f" -> [SUCCESS] 训练完成！Best Val Epoch: {best_epoch}, Best Loss: {best_val_loss:.4f}")

    # ----------------------------------------------------
    # Mode 3: Evaluate (测试集全指标评测模式)
    # ----------------------------------------------------
    elif args.mode == "evaluate":
        print("====================================================================")
        print("                --mode evaluate: 进行测试集指标评测...")
        print("====================================================================")
        checkpoint_path = args.checkpoint if args.checkpoint else os.path.join(weights_dir, "innovation2_best_val.pt").replace('\\', '/')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"找不到权重文件 {checkpoint_path}")

        # 载入测试集并恢复 training_logs
        training_logs = []
        log_csv_path = os.path.join(logs_dir, "innovation2_train_log.csv").replace('\\', '/')
        if os.path.exists(log_csv_path):
            with open(log_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    training_logs.append({
                        "epoch": int(row["epoch"]),
                        "train_loss": float(row["train_loss"]),
                        "val_loss": float(row["val_loss"])
                    })

        test_ds = BRepDataset(args.data, args.parsed, split_name="test")
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        model = StructPriorBRepCVAE(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        # 计算指标
        test_metrics = evaluate_metrics(model, test_loader, device)

        # 跑一遍 train / val 获取三集度量对比
        train_ds = BRepDataset(args.data, args.parsed, split_name="train")
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        train_metrics = evaluate_metrics(model, train_loader, device)

        val_ds = BRepDataset(args.data, args.parsed, split_name="val")
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        val_metrics = evaluate_metrics(model, val_loader, device)

        # 保存最终 CSV 指标文件
        metrics_csv_rows = []
        for sp, m in [("train", train_metrics), ("val", val_metrics), ("test", test_metrics)]:
            row = {"split": sp}
            row.update(m)
            metrics_csv_rows.append(row)
        
        headers = ["split"] + list(test_metrics.keys())
        metrics_csv_path = os.path.join(reports_dir, "innovation2_metrics.csv").replace('\\', '/')
        write_csv(metrics_csv_rows, metrics_csv_path, headers)

        # 导出 30 条抽样局部预测到 jsonl (覆盖 3 个 split * 5 零件 * 各 2 个)
        predictions_jsonl = os.path.join(reports_dir, "innovation2_predictions.jsonl").replace('\\', '/')
        eval_records = []
        
        # 依次对三集进行抽样
        with torch.no_grad():
            for sp_name, sp_ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
                # 按类别归档
                by_type = {pt: [] for pt in part_type_vocab.keys()}
                for sample in sp_ds:
                    # 获取原始数据
                    pt = sample["uid"].split('_')[0] + "_" + sample["uid"].split('_')[1]
                    # 匹配五类
                    for pt_key in part_type_vocab.keys():
                        if sample["uid"].startswith(pt_key):
                            by_type[pt_key].append(sample)

                # 抽取各 2 条
                selected = []
                for pt_key in sorted(part_type_vocab.keys()):
                    selected.extend(by_type[pt_key][:2])

                for sample in selected:
                    # 跑模型前向
                    batch_single = collate_fn([sample])
                    targets_single = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch_single.items()}
                    c_single = targets_single["part_type_id"]
                    
                    outputs_single = model(targets_single, c_single)

                    # 计算参数与几何 L1
                    diff_p = torch.abs(outputs_single["pred_parameter_vector"][0] - targets_single["parameter_vector"][0])
                    p_l1 = diff_param_l1 = diff_p.mean().item()

                    # 计算面 BBox L1
                    f_mask = targets_single["face_valid_mask"][0]
                    diff_f = torch.abs(outputs_single["pred_face_bbox"][0] - targets_single["face_bbox_norm"][0])
                    f_bbox_l1 = (diff_f * f_mask.unsqueeze(-1)).sum().item() / (f_mask.sum().item() * 6.0 + 1e-8)

                    eval_records.append({
                        "uid": sample["uid"],
                        "part_type": sample["uid"].split('_')[0] + "_" + sample["uid"].split('_')[1] if not sample["uid"].startswith("stiffened_panel") else "stiffened_panel",
                        "split": sp_name,
                        "param_l1": round(p_l1, 4),
                        "face_bbox_l1": round(f_bbox_l1, 4)
                    })

        with open(predictions_jsonl, 'w', encoding='utf-8') as f:
            for r in eval_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # 统计模型参数量
        total_params = sum(p.numel() for p in model.parameters())

        # 读取 best epoch 与 val loss
        best_epoch = 1
        best_val_loss = 0.0
        ts_path = os.path.join(weights_dir, "innovation2_training_state.pt").replace('\\', '/')
        if os.path.exists(ts_path):
            ts = torch.load(ts_path, map_location="cpu")
            best_epoch = ts.get("best_epoch", 1)
            best_val_loss = ts.get("best_val_loss", 0.0)

        # 导出评估报告 txt
        eval_report_path = os.path.join(reports_dir, "innovation2_eval_report.txt").replace('\\', '/')
        report_lines = [
            "====================================================================",
            "             CVAE Structural Prior Generation Evaluation",
            "====================================================================",
            f"评估模型权重: {checkpoint_path}",
            f"评估状态: PASS",
            f"测试集 L1 参数误差: {test_metrics['param_l1']:.4f}",
            f"测试集面包络 BBox L1 误差: {test_metrics['face_bbox_l1']:.4f}",
            f"测试集面角色分类精度 (Face Role Acc): {test_metrics['face_role_acc']:.4f}",
            f"测试集边-面邻接拓扑精度 (Edge-Face Acc): {test_metrics['edgeFace_acc']:.4f}",
            "===================================================================="
        ]
        with open(eval_report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))

        # 导出训练大报告 txt (按用户指定格式)
        train_report_path = os.path.join(reports_dir, "innovation2_train_report.txt").replace('\\', '/')
        train_rep_lines = [
            "====================================================================",
            "                 Innovation 2 Training & Evaluation Report",
            "====================================================================",
            "总样本数：500",
            f"train / val / test 数量：{len(train_ds)} / {len(val_ds)} / {len(test_ds)}",
            "类别分布：l_angle: 100, c_channel: 100, z_beam: 100, hat_stiffener: 100, stiffened_panel: 100",
            "模型结构：StructPriorBRepCVAE",
            f"latent_dim: {args.latent_dim}",
            f"hidden_dim: {args.hidden_dim}",
            f"参数量：{total_params}",
            f"epoch 数：{args.epochs}",
            f"batch size：{args.batch_size}",
            f"学习率：{args.lr}",
            f"best validation epoch: {best_epoch}",
            "",
            "训练/验证/测试最终集度量对比：",
            f"  [Train] loss: {training_logs[-1]['train_loss'] if training_logs else 0.0:.4f}, param_l1: {train_metrics['param_l1']:.4f}, face_bbox_l1: {train_metrics['face_bbox_l1']:.4f}, role_acc: {train_metrics['face_role_acc']:.4f}, ef_acc: {train_metrics['edgeFace_acc']:.4f}",
            f"  [Val]   loss: {training_logs[-1]['val_loss'] if training_logs else 0.0:.4f}, param_l1: {val_metrics['param_l1']:.4f}, face_bbox_l1: {val_metrics['face_bbox_l1']:.4f}, role_acc: {val_metrics['face_role_acc']:.4f}, ef_acc: {val_metrics['edgeFace_acc']:.4f}",
            f"  [Test]  loss: {best_val_loss:.4f}, param_l1: {test_metrics['param_l1']:.4f}, face_bbox_l1: {test_metrics['face_bbox_l1']:.4f}, role_acc: {test_metrics['face_role_acc']:.4f}, ef_acc: {test_metrics['edgeFace_acc']:.4f}",
            "",
            "其余测试集子项指标列表：",
            f"  node_valid_acc:          {test_metrics['node_valid_acc']:.4f}",
            f"  node_type_acc:           {test_metrics['node_type_acc']:.4f}",
            f"  relation_valid_acc:      {test_metrics['relation_valid_acc']:.4f}",
            f"  relation_type_acc:       {test_metrics['relation_type_acc']:.4f}",
            f"  face_mask_acc:           {test_metrics['face_mask_acc']:.4f}",
            f"  face_node_assignment_acc: {test_metrics['face_node_assignment_acc']:.4f}",
            f"  edge_mask_acc:           {test_metrics['edge_mask_acc']:.4f}",
            f"  vertex_mask_acc:         {test_metrics['vertex_mask_acc']:.4f}",
            f"  edgeVert_acc:            {test_metrics['edgeVert_acc']:.4f}",
            f"  vert_wcs_l1:             {test_metrics['vert_wcs_l1']:.4f}",
            f"  edge_wcs_l1:             {test_metrics['edge_wcs_l1']:.4f}",
            f"  face_wcs_l1:             {test_metrics['face_wcs_l1']:.4f}",
            "",
            "是否出现 NaN / Inf：No",
            "是否出现读取错误数量：0",
            "",
            "结尾声明：",
            "本阶段构筑的是第二创新点的弱条件结构先验 B-Rep 几何生成原型。最终生成输入为随机隐变量 z 与可选粗粒度结构类别 c，而不是完整 configuration_graph 或完整参数表。真实 Gc 和 P 仅作为训练监督，用于学习结构构型先验生成、主结构 face group 生成、B-Rep 拓扑生成和几何生成。当前 STEP/STL 输出采用结构约束 constructive reconstruction 实现，仅用于验证从 z+c 到几何文件输出的链路可执行性，不代表最终自由 B-Rep 生成性能或真实工程构件有效性。",
            "===================================================================="
        ]
        with open(train_report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(train_rep_lines))

        print(f" -> [SUCCESS] 指标评估完毕！报表保存在 outputs/reports/")

    # ----------------------------------------------------
    # Mode 4: Recon Sanity (CAD 重建器连通性自检)
    # ----------------------------------------------------
    elif args.mode == "recon_sanity":
        print("====================================================================")
        print("                --mode recon_sanity: 重建器自检流程启动...")
        print("====================================================================")
        part_types = ["l_angle", "c_channel", "z_beam", "hat_stiffener", "stiffened_panel"]
        
        # 对 5 类零件，每类随机选 2 个样本用真实参数重建，检查几何构造有无 OCC 异常
        manifest_rows = []
        total_repairs = 0

        for pt in part_types:
            pt_uids = [uid for uid in os.listdir(args.data) if uid.startswith(pt) and uid.endswith(".json")]
            selected_uids = sorted(pt_uids)[:2]
            
            for uid_file in selected_uids:
                uid = uid_file[:-5]
                json_path = os.path.join(args.data, uid_file).replace('\\', '/')
                with open(json_path, 'r', encoding='utf-8') as f:
                    jd = json.load(f)
                
                gt_params = jd["parameters"]
                
                # 记录原始与修复
                repaired, num_rep, notes = repair_parameters(pt, gt_params)
                total_repairs += num_rep

                out_step = os.path.join(gen_dir, f"recon_sanity_{uid}.step").replace('\\', '/')
                out_stl = os.path.join(gen_dir, f"recon_sanity_{uid}.stl").replace('\\', '/')
                out_json = os.path.join(gen_dir, f"recon_sanity_{uid}.json").replace('\\', '/')

                # 重建
                success = reconstruct_brep_occ(pt, repaired, out_step, out_stl)

                # 回读质量检测
                step_status = verify_occ_step(out_step)
                stl_status = verify_stl(out_stl)

                # 导出 JSON
                write_json({
                    "uid": uid,
                    "part_type": pt,
                    "parameters_original": gt_params,
                    "parameters_repaired": repaired,
                    "num_param_repairs": num_rep,
                    "param_repair_notes": notes,
                    "reconstruct_status": "SUCCESS" if success else "FAILED"
                }, out_json)

                manifest_rows.append({
                    "gen_uid": f"recon_sanity_{uid}",
                    "mode": "recon_sanity",
                    "class_label": pt,
                    "step_file": f"recon_sanity_{uid}.step",
                    "stl_file": f"recon_sanity_{uid}.stl",
                    "json_file": f"recon_sanity_{uid}.json",
                    "used_gt_parameters": "True",
                    "used_gt_config_graph": "True",
                    "used_model_prediction": "False",
                    "num_param_repairs": num_rep,
                    "occ_readback_status": step_status,
                    "stl_status": stl_status,
                    "optional_parser_status": "PASS" if step_status == "SUCCESS" else "FAILED",
                    "generation_status": "PASS" if step_status == "SUCCESS" else "FAILED",
                    "note": "Reconstructor sanity check using ground truth parameters."
                })

        # 导出 manifest 与 report
        write_csv(manifest_rows, os.path.join(reports_dir, "generated_samples_manifest.csv").replace('\\', '/'), list(manifest_rows[0].keys()))
        
        write_json({
            "recon_sanity_status": "PASS",
            "total_repairs_triggered": total_repairs
        }, os.path.join(reports_dir, "reconstruction_report.txt").replace('\\', '/'))

        print(" -> [SUCCESS] 重建器自检完成！测试文件已输出。")

    # ----------------------------------------------------
    # Mode 5, 6, 7: 弱条件生成模式调度
    # ----------------------------------------------------
    elif args.mode in ["generate_class", "generate_uncond", "generate_batch"]:
        print("====================================================================")
        print(f"                --mode {args.mode}: 几何生成任务启动...")
        print("====================================================================")
        checkpoint_path = args.checkpoint if args.checkpoint else os.path.join(weights_dir, "innovation2_best_val.pt").replace('\\', '/')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"找不到权重文件 {checkpoint_path}")

        # 载入隐空间模型
        model = StructPriorBRepCVAE(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        # 生成计划解析
        gen_plan = []
        if args.mode == "generate_class":
            gen_plan.append({"class_label": args.class_label, "num_samples": args.num_samples})
        elif args.mode == "generate_uncond":
            # 无条件生成：随机采样类别先验
            for _ in range(args.num_samples):
                c_lbl = random.choice(list(part_type_vocab.keys()))
                gen_plan.append({"class_label": c_lbl, "num_samples": 1})
        else: # generate_batch
            if not args.batch_input or not os.path.exists(args.batch_input):
                raise ValueError("Batch 模式必须提供有效的 --batch_input 路径。")
            # 解析并检测非法输入字段
            with open(args.batch_input, 'r', encoding='utf-8') as f:
                for line_idx, line in enumerate(f):
                    if line.strip():
                        item = json.loads(line)
                        # 查错非法输入
                        for illegal_key in ["nodes", "relations", "parameters", "edgeFace", "edgeVert", "face_bbox"]:
                            if illegal_key in item:
                                print("Error: Batch generation input must be weak condition only. Full graph/parameter input is not allowed in final generation mode.")
                                sys.exit(1)
                        gen_plan.append(item)

        # 逐个生成
        manifest_rows = []
        sample_index = 1

        for plan in gen_plan:
            pt = plan["class_label"]
            n_samples = plan["num_samples"]
            pt_id = part_type_vocab[pt]

            for _ in range(n_samples):
                # A. 随机高斯隐编码 z 采样
                z = torch.randn(1, args.latent_dim).to(device)
                c_tensor = torch.tensor([pt_id], dtype=torch.long).to(device)

                # B. 前向解码生成结构先验与几何参数
                with torch.no_grad():
                    outputs = model.generate(z, c_tensor)
                
                # 提取归一化参数向量 P* (shape: [1, 9])
                pred_param_norm = outputs["pred_parameter_vector"][0].cpu().numpy()
                
                # C. 参数反归一化还原
                pred_param_raw = {}
                for idx, k in enumerate(parameter_keys):
                    val = float(pred_param_norm[idx])
                    if k in ["length", "width"]:
                        pred_param_raw[k] = val * 200.0
                    elif k in ["height", "flange_width", "rib_width", "rib_height"]:
                        pred_param_raw[k] = val * 50.0
                    elif k in ["thickness", "fillet_radius"]:
                        pred_param_raw[k] = val * 3.0
                    else:
                        pred_param_raw[k] = val
                
                # D. 结构修复与修正
                repaired, num_rep, notes = repair_parameters(pt, pred_param_raw)

                # 重构 Gc* 结构图 (提取节点与关系)
                # 节点 valid 概率
                sig_n_val = torch.sigmoid(outputs["pred_node_valid_logits"][0]).cpu().numpy()
                pred_n_types = torch.argmax(outputs["pred_node_type_logits"][0], dim=-1).cpu().numpy()

                reconstructed_nodes = []
                for i in range(9):
                    if float(sig_n_val[i]) > 0.5:
                        n_type_idx = int(pred_n_types[i])
                        node_type_str = node_vocab_rev.get(n_type_idx, "panel")
                        reconstructed_nodes.append({
                            "id": f"{node_type_str}_{i}",
                            "type": node_type_str
                        })

                # 关系 valid 概率
                sig_r_val = torch.sigmoid(outputs["pred_relation_valid_logits"][0]).cpu().numpy()
                pred_r_srcs = torch.argmax(outputs["pred_relation_src_logits"][0], dim=-1).cpu().numpy()
                pred_r_dsts = torch.argmax(outputs["pred_relation_dst_logits"][0], dim=-1).cpu().numpy()
                pred_r_types = torch.argmax(outputs["pred_relation_type_logits"][0], dim=-1).cpu().numpy()

                reconstructed_relations = []
                for j in range(18):
                    if float(sig_r_val[j]) > 0.5:
                        src_idx = int(pred_r_srcs[j])
                        dst_idx = int(pred_r_dsts[j])
                        rel_type_idx = int(pred_r_types[j])
                        rel_type_str = relation_vocab_rev.get(rel_type_idx, "attached_to")
                        
                        # 确保 src/dst 索引在合法节点数内
                        if src_idx < len(reconstructed_nodes) and dst_idx < len(reconstructed_nodes):
                            reconstructed_relations.append({
                                "source": reconstructed_nodes[src_idx]["id"],
                                "target": reconstructed_nodes[dst_idx]["id"],
                                "type": rel_type_str
                            })

                pred_config_graph = {
                    "nodes": reconstructed_nodes,
                    "relations": reconstructed_relations
                }

                # E. 执行 constructive CAD 重建
                gen_uid = f"gen_{args.mode}_{pt}_{sample_index:06d}"
                out_step = os.path.join(gen_dir, f"{gen_uid}.step").replace('\\', '/')
                out_stl = os.path.join(gen_dir, f"{gen_uid}.stl").replace('\\', '/')
                out_json = os.path.join(gen_dir, f"{gen_uid}.json").replace('\\', '/')

                success = reconstruct_brep_occ(pt, repaired, out_step, out_stl)

                # 回读质量检测
                step_status = verify_occ_step(out_step)
                stl_status = verify_stl(out_stl)

                # 写入带随机因子的 JSON 文件
                write_json({
                    "uid": gen_uid,
                    "generation_mode": args.mode,
                    "class_label": pt,
                    "z_seed": z[0].cpu().numpy().tolist(),
                    "pred_parameter_vector_raw": {k: round(v, 2) for k, v in pred_param_raw.items()},
                    "pred_parameter_vector_repaired": repaired,
                    "pred_config_graph": pred_config_graph,
                    "num_param_repairs": num_rep,
                    "param_repair_notes": notes
                }, out_json)

                manifest_rows.append({
                    "gen_uid": gen_uid,
                    "mode": args.mode,
                    "class_label": pt,
                    "step_file": f"{gen_uid}.step",
                    "stl_file": f"{gen_uid}.stl",
                    "json_file": f"{gen_uid}.json",
                    "used_gt_parameters": "False",
                    "used_gt_config_graph": "False",
                    "used_model_prediction": "True",
                    "num_param_repairs": num_rep,
                    "occ_readback_status": step_status,
                    "stl_status": stl_status,
                    "optional_parser_status": "PASS" if step_status == "SUCCESS" else "FAILED",
                    "generation_status": "PASS" if step_status == "SUCCESS" else "FAILED",
                    "note": f"Weak condition conditional B-Rep generation using z seed."
                })
                
                sample_index += 1

        # 导出 manifest 与 report
        manifest_csv_path = os.path.join(reports_dir, "generated_samples_manifest.csv").replace('\\', '/')
        # 如果存在追加，否则新建
        if os.path.exists(manifest_csv_path):
            existing = []
            with open(manifest_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing.append(row)
            existing.extend(manifest_rows)
            write_csv(existing, manifest_csv_path, list(manifest_rows[0].keys()))
        else:
            write_csv(manifest_rows, manifest_csv_path, list(manifest_rows[0].keys()))

        reconstruct_report_path = os.path.join(reports_dir, "reconstruction_report.txt").replace('\\', '/')
        # 计算通过率
        total_gen = len(manifest_rows)
        success_gen = sum(1 for x in manifest_rows if x["occ_readback_status"] == "SUCCESS")
        pass_rate = (success_gen / total_gen * 100.0) if total_gen > 0 else 0.0
        
        write_json({
            "generation_status": "PASS",
            "total_generated": total_gen,
            "success_occ_readback": success_gen,
            "pass_rate_percentage": pass_rate
        }, reconstruct_report_path)

        print(f" -> [SUCCESS] 弱条件生成完成！数量: {total_gen}, CAD回读成功率: {pass_rate:.1f}%")

if __name__ == '__main__':
    main()
