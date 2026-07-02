# 创新点 2 v2 升级工作总结

## 升级目标

将创新点 2 从初版"弱条件 CVAE + 参数预测 + constructive reconstruction"升级为：
**弱条件结构先验驱动的层级 B-Rep 生成与工程可行性重建方法 (v2)**

---

## 代码修改清单

### 1. [models.py](file:///F:/开题答辩/a中期答辩专用/DTGBrepGen-master/innovation2_struct_prior_brepgen/models.py)
- **BoundedParameterDecoder**：连续参数通过 Sigmoid 映射到工程有界区间 + 离散加筋数 6 分类交叉熵
- **StructuralPriorEmbedding / FaceLayoutEmbedding / TopologyEmbedding**：三级 128 维 Embedding 层
- **StructPriorBRepCVAE**：层级依赖前向 forward/generate，上级特征 cat 注入下级 Decoder

### 2. [losses.py](file:///F:/开题答辩/a中期答辩专用/DTGBrepGen-master/innovation2_struct_prior_brepgen/losses.py)
- 拆分连续 SmoothL1（8 维）与离散 CrossEntropy（rib_count）联合损失

### 3. [run_innovation2.py](file:///F:/开题答辩/a中期答辩专用/DTGBrepGen-master/innovation2_struct_prior_brepgen/run_innovation2.py)
- **validate_and_repair_pred_config_graph**：5 类构件拓扑合法性校验与模板修补
- **derive_parameters_from_face_layout**：面布局反推全局几何尺寸
- 生成循环中引入 $P_{\text{final}} = 0.8 P_{\text{decoder}} + 0.2 P_{\text{layout}}$ 融合
- 新增 `--mode evaluate_prior`：250 个样本的先验生成 + OCC 回读 + 多样性评测
- 所有权重/报告输出使用 `_v2` 命名空间，不覆盖 v1 结果

### 4. [method_summary.md](file:///F:/开题答辩/a中期答辩专用/DTGBrepGen-master/innovation2_struct_prior_brepgen/method_summary.md)
- 更新为 v2 版本层级架构与工程可行性重建方法学术描述

---

## 训练与验证结果

### v2 训练指标 (120 Epochs, Best Epoch 116)

| 集合 | Loss | Param L1 | BBox L1 | Role Acc | Edge-Face Acc |
|------|------|----------|---------|----------|---------------|
| Train | 0.1943 | 0.1852 | 0.3493 | 97.81% | 100% |
| Val | 0.2234 | 0.1915 | 0.3458 | 97.69% | 100% |
| **Test** | **0.1898** | **0.1873** | **0.3544** | **98.70%** | **100%** |

### 测试集完整子项指标

| 指标 | 值 |
|------|-----|
| node_valid_acc | 100% |
| node_type_acc | 100% |
| relation_valid_acc | 100% |
| relation_type_acc | 100% |
| face_mask_acc | 100% |
| face_node_assignment_acc | 94.33% |
| edge_mask_acc | 100% |
| vertex_mask_acc | 100% |
| edgeVert_acc | 100% |

### 先验生成评测 (evaluate_prior, 5×50=250 样本)

| 类别 | 参数无修补率 | 图合法率 | OCC 回读率 | 一致性 L1 | 多样性距离 |
|------|------------|---------|-----------|----------|-----------|
| l_angle | 100% | 0% → 100% | **100%** | 30.60 | 6.04 |
| c_channel | 100% | 100% | **100%** | 46.33 | 8.46 |
| z_beam | 100% | 100% | **100%** | 39.13 | 7.93 |
| hat_stiffener | 100% | 100% | **100%** | 42.71 | 6.70 |
| stiffened_panel | 0% → 100% | 100% | **100%** | 65.50 | 9.88 |

> **250/250 样本全部通过 OCC 回读验证，CAD 实体化成功率 100%。**

---

## 输出文件清单

```
innovation2_struct_prior_brepgen/outputs/
├── weights/
│   ├── innovation2_v2_best_val.pt          (204 MB, v2 最优权重)
│   ├── innovation2_v2_last.pt              (v2 最终 epoch 权重)
│   ├── innovation2_v2_training_state.pt    (v2 训练状态)
│   ├── innovation2_best_val.pt             (v1 权重，保留不动)
│   └── ...
├── reports/
│   ├── innovation2_v2_train_report.txt     (v2 训练大报告)
│   ├── innovation2_v2_eval_report.txt      (v2 评估报告)
│   ├── prior_generation_report.txt         (先验生成质量报告)
│   ├── prior_generation_metrics.csv        (先验生成指标 CSV)
│   ├── parameter_repair_stats.csv          (参数修补统计)
│   ├── diversity_metrics.csv               (多样性指标)
│   └── generated_samples_manifest.csv      (生成样本审计大表)
├── logs/
│   ├── innovation2_v2_train_log.csv        (v2 训练日志)
│   └── innovation2_train_log.csv           (v1 日志，保留)
└── generated/
    └── gen_prior_*.step / .stl / .json     (250 个先验生成样本)
```