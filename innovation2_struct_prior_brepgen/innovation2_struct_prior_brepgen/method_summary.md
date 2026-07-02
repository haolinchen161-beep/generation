# 创新点 2：弱条件结构先验驱动的层级 B-Rep 生成与工程可行性重建方法 (v2)

## 1. 方法原理与层级架构

针对航空复材薄壁加筋构件（如 L型长桁、C型梁、Z型梁、帽型筋、加筋壁板等），传统生成模型容易在底层 B-Rep 面几何及局部拓扑缝合上产生扭曲或不自洽。本方法提出一种**弱条件结构先验驱动的层级几何生成模型 (StructPriorBRepCVAE v2)**，采用层级依赖图谱与几何-拓扑协同传递的前向解码流：

该方法建立在基线源程序与创新点 1 形成的结构化构件参数、构型图和 B-Rep 张量表达基础之上，用于支撑论文第二创新点的模型设计与工程可行性验证。

```
                    z + c (64维隐变量 + 类别条件)
                              ↓
          结构先验解码器 (StructuralPriorDecoder) 
            - 预测有界连续参数、离散加筋数 (BoundedParameterDecoder)
            - 预测构型图节点 Vc*、边关系 Ec*
                              ↓
          [特征传递] cat(z_c, StructuralPriorEmbedding(128))
                              ↓
          面布局生成器 (FaceGroupLayoutDecoder) -> 预测 Face BBox & Role
                              ↓
          [特征传递] cat(z_c, FaceLayoutEmbedding(128))
                              ↓
          边界拓扑生成器 (BoundaryTopologyDecoder) -> 预测 Edge/Vertex Adj
                              ↓
          [特征传递] cat(z_c, TopologyEmbedding(128))
                              ↓
          几何曲线曲面生成器 (CurveSurfaceGeometryDecoder) -> 预测点云三维坐标
                              ↓
    -----------------------------------------------------------------
    [工程可行性重建工程闭环 (Engineering Feasibility Reconstruction)]
                              ↓
    1. 构型图拓扑合法性规则检测与强制图补齐 (validate_and_repair_pred_config_graph)
    2. 面布局特征尺度提取 (derive_parameters_from_face_layout)
    3. 参数跨度估计融合: P_final = 0.8 * P_decoder + 0.2 * P_layout
    4. 边界范围限制与物理修补 (repair_parameters)
                              ↓
            OpenCascade CAD 引擎几何缝合与实体化重建
                              ↓
                         STEP / STL 实体文件
```

## 2. v2 升级核心技术特征

### A. 真正的层级依赖解码传递
在 CVAE 解码阶段，后续 Decoder 的条件输入不仅包含随机隐状态 $z_c$，还通过高效全连接 Embedding 层层级联注入前一阶段预测的特征表达。即后续的局部特征（如面片、拓扑、点云几何）受到全局结构先验信息的强约束，从而抑制了生成漂移。

### B. 工程有界参数解码 (BoundedParameterDecoder)
* 摒弃传统的无边界直接数值回归，设计了物理空间映射与有界约束机制。
* 连续几何参数（长、宽、厚、高、翼缘宽、筋宽、筋高、圆角半径）通过 Sigmoid 激活函数映射到安全、合理的工程实际区间。
* 引入类别相关参数屏蔽 (class-aware parameter mask)：训练与生成阶段只对当前构件类别真实有意义的参数进行监督、融合与落地，避免无意义参数的 0 目标与有界解码下界产生不可消除误差。
* Fillet radius 圆角半径引入了与厚度和面宽度的动态联合有界公式：
  $$\text{fillet\_radius} \in [1.5 \times \text{thickness}, 0.25 \times \min(\text{width}, \text{height}, \text{flange\_width})]$$
* 离散参数（加筋数目 `rib_count`）通过独立的 6 分类交叉熵损失分支进行离散分类预测；该损失仅在 `stiffened_panel` 类别上启用。

### C. 工程可行性拓扑与几何融合机制
* **构型图修补**：对生成的先验构型图 $G_c^*$，执行基于力学与制造规范的合法性检查（如腹板数、缘条数、过渡圆角数是否完整）。当前 $G_c^*$ 参与结构合法性校验与生成诊断；OCC 落地重建仍采用类别模板与预测参数驱动。
* **面布局尺度一致性融合**：利用已生成的 Face Group Layout Bounding Box 极值估计零件的全局外廓尺度（Length, Width, Height, Flange Width, Stiffener Height），作为轻量尺度一致性约束，用于修正和诊断预测参数，并以 0.8 对 0.2 的比例与解码器直接预测的参数进行加权融合：
  $$P_{\text{final}} = 0.8 P_{\text{decoder}} + 0.2 P_{\text{layout}}$$
  以实现参数生成与局部面群几何自恰性的闭环协同。
* **参数修补审计**：`repair_parameters()` 返回逐字段 `repair_flags`，`parameter_repair_stats.csv` 由真实修补记录汇总生成，用于支撑论文中关于后处理依赖程度的定量分析。

## 3. 学术价值与科学对齐

本方法落实了第二创新点“弱条件结构先验驱动”与“工程可行性重建”的物理图景：
1. **输入弱条件**：仅需 $z+c$。真实构型图 $G_c$ 和参数 $P$ 仅作为多模态监督目标，不参与生成输入。
2. **中间先验表达**：输出包含显式的拓扑语义 $G_c^*$。
3. **闭环可制造性**：将深度学习概率生成产物通过拓扑校验、参数修复、几何一致性融合等机制进行校正。在当前规则化五类构件范围内，程序提供 OCC 回读与 STL 写出验证机制；生成结果是否全部通过以 `prior_generation_report.txt` 和 `generated_samples_manifest.csv` 为准。

## 4. 实验报告更新约定

当前代码版本的论文指标必须由重新训练与重新评估后的报告文件给出。训练指标以 `innovation2_v2_train_report.txt` 和 `innovation2_v2_eval_report.txt` 为准；先验生成、OCC 回读、多样性和参数修补统计分别以 `prior_generation_report.txt`、`generated_samples_manifest.csv`、`diversity_metrics.csv` 和 `parameter_repair_stats.csv` 为准。
