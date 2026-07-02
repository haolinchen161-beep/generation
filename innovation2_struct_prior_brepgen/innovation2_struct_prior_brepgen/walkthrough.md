# 创新点 2 v2 升级工作总结

## 升级目标

将创新点 2 从初版"弱条件 CVAE + 参数预测 + constructive reconstruction"升级为：
**弱条件结构先验驱动的层级 B-Rep 生成与工程可行性重建方法 (v2)**

---

## 代码修改清单

### 1. [models.py](file:///F:/开题答辩/a中期答辩专用/DTGBrepGen-master/innovation2_struct_prior_brepgen/models.py)
- **BoundedParameterDecoder**：连续参数通过 Sigmoid 映射到工程有界区间 + 类别相关参数屏蔽 + 离散加筋数 6 分类交叉熵
- **StructuralPriorEmbedding / FaceLayoutEmbedding / TopologyEmbedding**：三级 128 维 Embedding 层
- **StructPriorBRepCVAE**：层级依赖前向 forward/generate，上级特征 cat 注入下级 Decoder

### 2. [losses.py](file:///F:/开题答辩/a中期答辩专用/DTGBrepGen-master/innovation2_struct_prior_brepgen/losses.py)
- 拆分连续 SmoothL1（8 维）与离散 CrossEntropy（rib_count）联合损失；参数损失按当前构件类别 mask 后计算

### 2a. [parameter_schema.py](file:///F:/开题答辩/a中期答辩专用/DTGBrepGen-master/innovation2_struct_prior_brepgen/parameter_schema.py)
- 统一维护 9 维参数顺序、5 类构件有效参数 mask、参数边界、融合字段和修复统计字段

### 3. [run_innovation2.py](file:///F:/开题答辩/a中期答辩专用/DTGBrepGen-master/innovation2_struct_prior_brepgen/run_innovation2.py)
- **validate_and_repair_pred_config_graph**：5 类构件拓扑合法性校验与模板修补
- **derive_parameters_from_face_layout**：面布局反推全局几何尺寸
- 生成循环中引入类别 mask 后的 $P_{\text{final}} = 0.8 P_{\text{decoder}} + 0.2 P_{\text{layout}}$ 融合
- 新增 `--mode evaluate_prior`：先验生成 + OCC 回读 + 多样性评测；每类样本数由 `--num_prior_samples` 控制
- `parameter_repair_stats.csv` 从 `repair_flags` 真实汇总，不再使用硬编码修补率
- 所有权重/报告输出使用 `_v2` 命名空间，不覆盖 v1 结果

### 4. [reconstructor_occ.py](file:///F:/开题答辩/a中期答辩专用/DTGBrepGen-master/innovation2_struct_prior_brepgen/reconstructor_occ.py)
- `repair_parameters()` 返回 `repaired, num_repairs, notes, repair_flags`
- 修补逻辑按类别屏蔽无意义参数，模板覆盖均进入统计

### 5. [method_summary.md](file:///F:/开题答辩/a中期答辩专用/DTGBrepGen-master/innovation2_struct_prior_brepgen/method_summary.md)
- 更新为 v2 版本层级架构与工程可行性重建方法学术描述

---

## 训练与验证结果

### 当前版本训练指标 (120 Epochs, Best Epoch 96)

| 集合 | Loss | Param L1 | BBox L1 | Role Acc | Edge-Face Acc |
|------|------|----------|---------|----------|---------------|
| Train | 0.1706 | 0.1926 | 0.3428 | 97.75% | 100% |
| Val | 0.2217 | 0.1978 | 0.3329 | 97.57% | 100% |
| **Test** | **0.1959** | **0.1907** | **0.3347** | **98.29%** | **100%** |

### 当前版本测试集完整子项指标

| 指标 | 值 |
|------|-----|
| node_valid_acc | 100% |
| node_type_acc | 100% |
| relation_valid_acc | 100% |
| relation_type_acc | 100% |
| face_mask_acc | 100% |
| face_node_assignment_acc | 94.51% |
| edge_mask_acc | 100% |
| vertex_mask_acc | 100% |
| edgeVert_acc | 100% |
| vert_wcs_l1 | 0.1582 |
| edge_wcs_l1 | 1.6256 |
| face_wcs_l1 | 1.3336 |

### 当前版本先验生成评测 (evaluate_prior, 5×50=250 样本)

| 类别 | 参数无修补率 | 平均参数修补数 | 图合法率 | OCC 回读率 | STL 成功率 | 一致性 L1 | 多样性距离 |
|------|------------|---------------|---------|-----------|-----------|----------|-----------|
| l_angle | 100% | 0.00 | 0% → 100% | 100% | 100% | 46.16 | 6.91 |
| c_channel | 100% | 0.00 | 100% | 100% | 100% | 58.64 | 10.27 |
| z_beam | 100% | 0.00 | 100% | 100% | 100% | 62.75 | 6.11 |
| hat_stiffener | 100% | 0.00 | 100% | 100% | 100% | 53.42 | 8.38 |
| stiffened_panel | 100% | 0.00 | 100% | 100% | 100% | 106.16 | 12.95 |

250 个先验生成样本均已写出 STEP/STL/JSON 并通过回读验证；逐字段参数修补率由 `parameter_repair_stats.csv` 真实统计，当前五类全部为 0。

> 论文中所有训练、测试、先验生成和 OCC 回读结论，均以当前代码重新生成的报告文件为准。

---

## 输出文件清单

```
innovation2_struct_prior_brepgen/outputs/
├── weights/
│   ├── innovation2_v2_best_val.pt          (204 MB, v2 最优权重)
│   ├── innovation2_v2_last.pt              (v2 最终 epoch 权重)
│   ├── innovation2_training_state.pt       (当前训练状态)
│   ├── label_maps.json                     (当前类别映射)
│   └── norm_stats.json                     (当前归一化配置)
├── reports/
│   ├── innovation2_v2_train_report.txt     (v2 训练大报告)
│   ├── innovation2_v2_eval_report.txt      (v2 评估报告)
│   ├── innovation2_metrics.csv             (train/val/test 指标 CSV)
│   ├── innovation2_predictions.jsonl       (抽样预测记录)
│   ├── prior_generation_report.txt         (当前先验生成质量报告)
│   ├── prior_generation_metrics.csv        (当前先验生成指标 CSV)
│   ├── parameter_repair_stats.csv          (当前参数修补统计)
│   ├── diversity_metrics.csv               (当前多样性指标)
│   └── generated_samples_manifest.csv      (当前生成样本审计大表)
├── logs/
│   ├── innovation2_v2_train_log.csv        (v2 训练日志)
│   └── runs/                               (当前 train/evaluate 标准输出日志)
└── generated/
    └── gen_prior_*.step / .stl / .json     (当前 250 个先验生成样本)
```
