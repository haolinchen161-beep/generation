# 创新点二：基于骨架先验的条件 B-Rep 几何拓扑生成网络与消融验证

本目录存放研究工作第二个创新点“弱条件结构先验驱动的构型图—B-Rep 层级几何生成方法”的自研模型原型、训练脚本及验证子模块。所有程序均进行物理隔离，确保学术主权与实验独立性。

---

## 一、 学术定位与研究定位说明

### 1. 为什么要建立本目录的自研生成器原型？
在原版 DTG 生成网络中，拓扑和几何生成由 6 个庞大的扩散模型（UNet1D Diffusion）与重型 Transformer 模块共同组成。如果直接将先验 $S$ 注入该系统进行联合微调，会面临以下阻碍：
1. **训练成本极高**：在 1717 个样本的小数据子集上，原版 Diffusion 很难在有限资源下从零收敛，时间成本过长。
2. **无法做纯粹的消融对照**：原版大模型过多的自由度会掩盖结构先验 $S$ 的真实物理约束作用。

因此，本目录设计并实现了一套**自研轻量级条件生成器原型（Custom Conditional Generator Prototype）**。该原型完全避开了原版 DTG 的 Diffusion 几何生成，采用可学习的先验 Cross-Attention Transformer 进行微观面线预测，并在损失层引入自研的“属从几何围栏”，用作验证结构先验有效性的学术实验支架。

### 2. 本目录程序与“原版 DTG”的物理边界
*   **本目录外部脚本 (`train_prior.py` / `train_geomgen.py` / `models_geomgen.py`等)**：属于**自研条件生成器**的训练和定义。
*   **`frozen_dtg_s_validation/` 子目录**：在不改变原版预训练 DTG 模型权重的前提下，利用 $S$ 一致性打分对 DTG 生成的 1024 个随机候选池进行检索、过滤和重排序验证（检索式验证，对应 Stage 1）。
*   **`fair_training_comparison/` 子目录**：以本目录的 `CustomGeomGenNet` 为基础模型，在相同的 seed 和数据划分下运行 A（无S）、B（随机S）、C（仅节点）、D（完整S）四组 50 轮训练，用于定量分析先验各分支的独立增益（消融式验证，对应 Stage 3）。

---

## 二、 两阶段层级生成数据流架构

自研条件生成器原型采用自上而下的层级生成数据流：

1.  **阶段一（S 先验图生成）**：
    $$z_{prior} (64\text{ 维}) \xrightarrow{\text{CustomPriorNet}} S = \{\text{基元节点类型}, \text{面群 BBox 布局}, \text{稀疏结构边关系}\}$$
2.  **阶段二（微观几何与拓扑合成）**：
    $$S \xrightarrow{\text{CustomGeomGenNet}} \text{面片几何 Latents } (64 \times 64) + \text{边线几何 Latents } (160 \times 16) + \text{拓扑连接图 } (160 \times 64)$$
3.  **VAE 解码还原与实体组装**：
    $$\text{面/边 Latents} \xrightarrow{\text{VAE Decoders}} \text{三维几何点云} \xrightarrow{\text{拓扑边连通缝合}} \text{STEP 流形 solid}$$

---

## 三、 核心程序与算法原理对应

本目录下各核心脚本与论文中所写算法公式的严格对应如下：

### 1. [dataset.py](file:///f:/开题答辩/a中期答辩专用/DTGBrepGen-master/innovation2_spg_brep_geomgen/dataset.py) (MotifPriorDataset)
*   **确定性划分**：固定使用 `seed=42` 对 1717 个样本进行 Shuffle 并按 `train_ratio=0.9` 严格切分。
*   **实时 VAE 编码**：在 `__getitem__` 读取数据时，程序从本地 `.pkl` 提取面片点云 $32\times32\times3$ 和边线点云 $32\times3$，实时送入 `FaceVAE` 与 `EdgeVAE` 的 Encoder，产出 $64$ 维的面 VAE 均值 `mu_f` 与 $16$ 维的线 VAE 均值 `mu_e` 作为训练回归靶值。
*   **规范化槽位重排**：利用 `_build_canonical_face_order` 对面片槽位进行“面群归属 $\rightarrow$ 空间质心 $\rightarrow$ 包围盒尺寸”的规范化重排，防止 VAE latent 的 slot-wise MSE 发生退化。

### 2. [models_prior.py](file:///f:/开题答辩/a中期答辩专用/DTGBrepGen-master/innovation2_spg_brep_geomgen/models_prior.py) (CustomPriorNet)
*   **Transformer 解码**：使用标准 `TransformerDecoder`。将 64 维隐变量 `z_prior` 稠密线性扩展，两两节点向量通过双线性融合（`edge_rel_head`）输出 $32 \times 32 \times 7$ 的稀疏结构边邻接分类。

### 3. [models_geomgen.py](file:///f:/开题答辩/a中期答辩专用%5CDTGBrepGen-master%5Cinnovation2_spg_brep_geomgen%5Cmodels_geomgen.py) (CustomGeomGenNet)
*   **先验 Cross-Attention 交叉注意力**：
    使用 Prior Encoder 提取输入的骨架先验图 $S$ 特征。设置 64 个可学习的面查询向量（`face_queries`）与 160 个线查询向量（`edge_queries`），通过多头交叉注意力从 $S$ 中提取空间和语义指引。
*   **双线性拓扑预测头**：
    面特征与线特征通过双线性映射投影，输出 $160 \times 64$ 的 edgeFace 邻接连通对数（Logits）。
*   **弹性几何属从围栏 Loss (`compute_belonging_fence_loss`)**：
    限制预测的面片中心位置偏离其所属 Motif 节点的包围盒边界。一旦偏离，Relu 激活施加惩罚：
    $$L_{fence} = \sum_{f=1}^{N_{face}} \left( \text{Relu}(C_f - B^{motif}_{upper}) + \text{Relu}(B^{motif}_{lower} - C_f) \right) \cdot \text{mask}_f$$
    其中 $C_f$ 为面片预测中心，$B^{motif}$ 为该面片在 `face_belong_matrix` 中对应 Motif 节点的真实 bbox 上下界。

---

## 四、 权重与运行目录说明

### 1. 权重存储物理结构
所有训练结果存放于 [checkpoints/](file:///f:/开题答辩/a中期答辩专用/DTGBrepGen-master/innovation2_spg_brep_geomgen/checkpoints) 下，每个模块训练 50 epochs：
*   `checkpoints/prior/prior_net.pth`：已收敛的宏观骨架生成权重。
*   `checkpoints/geomgen/geomgen_net.pth`：已收敛的微观几何拓扑生成权重。
*   `checkpoints/face/face_vae.pth` / `checkpoints/edge/edge_vae.pth`：面片与边线自编码器权重。

### 五、 后续架构重构与端到端无条件生成集成计划

在目前的研究阶段，我们虽然打通了 **“结构先验生成”** 与 **“条件几何/拓扑生成”** 的分段训练，并完成了消融验证，但在实现**真正意义上的端到端无条件 CAD 生成与 OCC 实体缝合**上，仍需进行以下核心架构的补充与重构。这也是本研究下一步走向实用化、闭环化的最关键环节。

### 1. 终极端到端推理管线（End-to-End Inference Loop）
要实现“从无到有”的自动三维设计，需要编写独立的端到端生成脚本（如规划中的 `generate_samples.py`），将训练好的两个阶段模型在推理阶段进行级联（Chaining）：
$$\text{Noise } z_{prior} \xrightarrow{\text{PriorNet}} S^* \xrightarrow{\text{GeomGenNet}} \{\text{Face/Edge Latents } (Z_f, Z_e), \text{Topology Logits } P_{adj}\}$$

推理代码的核心逻辑应为：
1.  **采样骨架**：向 `CustomPriorNet` 输入高斯噪声 $z \sim \mathcal{N}(0, I)$，采样生成离散的 $S^* = \{\text{Nodes}, \text{BBoxes}, \text{Relations}\}$。
2.  **约束传播**：将 $S^*$ 送入 `CustomGeomGenNet` 的 Prior Encoder，利用 Cross-Attention 引导模型解码出微观几何 Latents 和拓扑矩阵。
3.  **拓扑截断**：对预测出的 $P_{adj}$ 邻接对数实施二值化截断（使用我们在训练日志中搜索到的 `Best Thresh = 0.70`），输出硬决策拓扑边集。

### 2. 基于 OCC 的几何拟合、修复与实体缝合（OCC Reconstruction & Repair）
由于 VAE 解码出的几何特征是离散的 $32\times32$ 点云，无法直接作为 CAD 几何使用，必须经过高精度的几何重建管道：
1.  **解析曲面拟合 (B-Spline Fitting)**：
    使用 OpenCASCADE (OCC) 库中的 `GeomAPI_PointsToBSpline` 或最小二乘拟合算法，将 VAE 解码出的面片点云拟合为解析的 $B$-样条曲面（B-Spline Surfaces），获取精准的解析参数。
2.  **相交边界裁剪 (Boundary Trimming)**：
    利用生成的 `edgeFace_adj` 拓扑关系，寻找相邻面片的相交线。调用 OCC 的拓扑求交工具，对拟合曲面的物理边界进行裁剪，形成有界曲面（`TopoDS_Face`）。
3.  **流形缝合与缝隙修复 (Sewing & Repair)**：
    使用 `BRepBuilderAPI_Sewing` 将裁剪后的有界曲面缝合为三维实体。由于生成误差，面与面接缝处会存在微小缝隙，重构代码需调用 OCC 的缝隙修复工具（`ShapeFix_Shape`），以指定的容差范围进行容差面缝合，最终导出合法的流形实体 `TopoDS_Solid`，保存为标准的 `.step` 物理格式。

### 3. 将骨架先验 $S$ 深度融入原版 DTG 生成网络的集成方案
作为研究的最终演进目标，若要将结构先验 $S$ 深度融入原版预训练 DTG 网络，而非使用轻量级替代模型，后续重构应采用 **Adapter 微调模式**：
*   **网络修改**：在原版 DTG 的 1D 扩散模型（Diffusion）和双向 Transformer 中引入 Cross-Attention 层。
*   **冻结训练**：完全冻结原版 DTG 庞大的无条件生成权重，只将我们抽取的 `S_embd` 作为 Context 注入 Cross-Attention。仅训练新增的 Adapter 权重。这样既能继承 DTG 大模型在大规模 CAD 数据集上的微观几何生成精度，又能使其接受我们结构先验的宏观控制，最终实现兼具高几何生成质量与强物理拓扑约束的端到端 CAD 设计工具。
*   **Stage 3 同子集消融训练**（在 `fair_training_comparison` 目录运行）：
    ```powershell
    python run_fair_training_comparison.py --train_epochs 50
    ```
