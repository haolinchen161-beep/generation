# 第二创新点：隐空间驱动的 B-Rep 层级几何生成方法收尾报告

## 🛠️ 创新点 2 原型实现与流程验证

(Innovation 2 Prototype & Pipeline Verification)

为了完整验证论文第二创新点“**弱条件结构先验驱动的构型图—B-Rep 层级几何生成方法**”，我们在独立文件夹 [innovation2_struct_prior_brepgen](file:///F:/开题答辩/a中期答辩专用/DTGBrepGen-master/innovation2_struct_prior_brepgen/) 中构筑并成功测试了首版全流程深度生成系统原型。

### 1. 算法与模型架构 (Models & Losses)
* **CVAE 弱条件隐空间模型** ([models.py](file:///F:/开题答辩/a中期答辩专用/DTGBrepGen-master/innovation2_struct_prior_brepgen/models.py))：
  * **最终生成输入**：仅使用 $z \sim \mathcal{N}(0, I)$ 64维高斯随机噪声与可选粗粒度结构类别标签 $c$ 作为解码源，严禁将完整结构图与真实参数表直接作为生成输入。
  * **结构先验中间层 ($G_c^*$)**：由 `StructuralPriorDecoder` 自适应生成内部节点、关系与尺度参数 $P^*$。
  * **层级式解码生成**：先后通过 `FaceGroupLayoutDecoder`（生成面包络）、`BoundaryTopologyDecoder`（生成邻接矩阵）及 `CurveSurfaceGeometryDecoder`（生成三维曲面/边线离散控制点）。
* **18项联合损失带 Warmup 优化** ([losses.py](file:///F:/开题答辩/a中期答辩专用/DTGBrepGen-master/innovation2_struct_prior_brepgen/losses.py))：
  * 对 CVAE 隐空间 KL 散度（实施前10个 Epoch 的 Warmup 预热）以及高维拓扑分类交叉熵与几何回归 L1 损失实施多通路带掩膜（Masked）的加权优化。

### 2. 重建器设计与参数修补逻辑 ([reconstructor_occ.py](file:///F:/开题答辩/a中期答辩专用/DTGBrepGen-master/innovation2_struct_prior_brepgen/reconstructor_occ.py))
* **CAD 重建闭环**：基于 OpenCascade 内核在 Z-beam、C-channel 等五类薄壁件的强几何约束下实现了 Constructive B-Rep 重建。
* **边界安全修复**：包含对 thickness ($[1.8, 3.5]$ mm) 及圆角半径 ($r \ge 1.5t$) 的航空加工工艺硬性修补，防止几何退化与自相交。

### 3. 数据集预加载与性能提速 ([dataset.py](file:///F:/开题答辩/a中期答辩专用/DTGBrepGen-master/innovation2_struct_prior_brepgen/dataset.py))
* **内存 I/O 优化**：将 PKL 与 JSON 大张量的磁盘读写上移至 `__init__` 进行一次性预加载缓存，成功消除每轮 Epoch 对 4.2 万个 PKL 文件的盘读开销，将冒烟测试前向耗时从两分钟压缩至 **9 秒**。

### 4. 七大模式功能完整性校验 ([run_innovation2.py](file:///F:/开题答辩/a中期答辩专用/DTGBrepGen-master/innovation2_struct_prior_brepgen/run_innovation2.py))
* **`smoke` 冒烟测试**：通过，状态正常导出。
* **`train` 训练状态**：完成 120 个 Epochs 训练，GPU 显存无 NaN/Inf 泄露，在 Epoch 104 达到最优验证集损失 `0.1861`，成功导出 pt Checkpoint。
* **`evaluate` 指标评测**：
  * 测试集参数回归 L1 = `0.1534`，面包络 L1 = `0.3813`，面角色识别 Acc = `0.9874`，边-面邻接拓扑 Acc = `1.0000`。
  * 导出测试指标大表 [innovation2_metrics.csv](file:///F:/开题答辩/a中期答辩专用/DTGBrepGen-master/innovation2_struct_prior_brepgen/outputs/reports/innovation2_metrics.csv) 与大报告 [innovation2_train_report.txt](file:///F:/开题答辩/a中期答辩专用/DTGBrepGen-master/innovation2_struct_prior_brepgen/outputs/reports/innovation2_train_report.txt)。
* **`recon_sanity` 自检**：通过，使用真实参数 100% 重建成功，OCC 回读率 `100.0%`。
* **`generate_class` / `generate_uncond` / `generate_batch` 条件与无条件生成**：
  * 通过，完全由随机隐变量 $z$ + 类别特征驱动，生成的 CAD STEP/STL 文件输出正常。
  * 采用 OpenCascade STEP 几何读回器进行 100% 回读质量监控，读回通过率达到 **`100.0%`**，成功生成清单并归档至 [generated_samples_manifest.csv](file:///F:/开题答辩/a中期答辩专用/DTGBrepGen-master/innovation2_struct_prior_brepgen/outputs/reports/generated_samples_manifest.csv)。

### 5. 学术结论边界声明
已在工作区内建立学术摘要文件 [method_summary.md](file:///F:/开题答辩/a中期答辩专用/DTGBrepGen-master/innovation2_struct_prior_brepgen/method_summary.md)。本阶段成果只用于验证“弱条件控制 -> 拓扑与参数生成 -> CAD 实体输出”这一层级几何生成数据管道的学术畅通性，不用于代表真实航空构件物理力学性能或最终生成算法的泛化性能。

## ⚠️ 学术归因与防范免责声明

1. **学术归因**：
   本阶段从 ABC 通用 CAD 中挖掘 composite-like 弱候选样本，结果仅作为结构构型先验建模的数据基础，不代表真实航空复材构件语义标注。
2. **数据集使用免责声明**：
   本数据集为参数化生成的航空复材薄壁加筋构件几何样本，用于结构构型图先验建模和 B-Rep 几何生成方法研究，不代表真实型号飞机结构设计，不能直接用于工程承载结构或适航认证。
3. **审计与核验免责声明**：
   本阶段仅验证参数化构型 JSON 与 DTG-compatible B-Rep PKL 是否能够被统一读取并整理为后续程序可消费的 batch 数据结构。该结果用于确认后续方法验证的数据读取流程完整性。由于样本由规则化参数程序生成，本阶段不用于证明数据集的工程真实性，也不用于证明生成模型性能。