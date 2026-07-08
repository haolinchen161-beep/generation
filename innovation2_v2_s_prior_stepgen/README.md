# 创新点2 v2：S 先验驱动的无条件 STEP 生成说明书 (Academic Guide)

本目录实现的是本项目的第二核心创新点路线2：**基于 S 先验图骨架驱动的无条件三维 CAD 实体（STEP）生成**。它保留了高成功率和高物理连通性的稳定主线：

```text
无条件采样 S* (先验图骨架)
  -> 经过向量化 (s_vec + face_features) 接入自研宏观适配器
  -> face-aware S-conditioned adapter 预测面数、face bbox 以及 face-edge topology (面面拓扑连通)
  -> 级联至原版已训练好的冻结 DTG-DeepCAD edge-vertex 拓扑生成器与微观几何扩散模块
  -> 调用 OpenCASCADE 缝合输出合法的 .step 实体文件
```

该方法的最大特点是**不修改原版 DTG 底层源码**，不启用会破坏共享边界的局部几何硬修正，通过宏观适配器天然实现几何-拓扑数据流的同序处理，从根本上避开了索引错位崩溃的 Bug。

---

## 一、 核心方法与设计思想 (Methodology & Design)

1. **宏微解耦生成策略**：
   * **宏观布局（自研适配器）**：由我们提取的先验图 $S^*$ 决定零件的面数、包围盒位置和面片连通图，代表全局设计意图；
   * **微观还原（原版底座扩散）**：利用原版在 DeepCAD 上训练的扩散模型，利用标准高斯噪声还原顶点、边和面片的精确微观起伏与特征，保证生成的零部件具备几何唯一性与变异度。
2. **天然索引对齐**：
   * 原版底座模型的几何与拓扑预测器是在未排序的数据上训练的。
   * 本路线的自研适配器同样直接在未排序的真值（GT）上训练。因此数据在前后级传递时**处于相同的索引基准下**，无需任何复杂的拓扑重排排序，天然杜绝了错位崩溃。
3. **空间几何锚点自适应微调 (`bbox_anchor_blend`)**：
   * 适配器预测的 `face_bbox` 有时会存在一定偏差。本路线设计了**锚点微调机制**：利用输入先验图 $S^*$ 包含的 Motif 节点的空间位置包围盒（物理锚点），对 Adapter 预测出的包围盒进行按比例（如 `0.30`）的自适应加权拉回，极大地缩小了面片边缘在重构时的缝隙，从而将缝合成功率提升至质变级别。

---

## 二、 适配器数学模型 (Mathematical & Neural Network Models)

为了让 Transformer 读懂抽象的先验图 $S^*$ 并翻译为几何，我们设计了精细的特征工程向量化方案：

1. **先验全局特征向量 `s_vec` (维度：25)**：
   * 包含零件整体的面数、边数、点数归一化比例；
   * 包含 $S^*$ 蒸馏后保留的 6 类 Motif 节点（薄板、薄壁、孔洞等）和 6 类物理关系边（相对、平行、垂直等）的频次统计；
   * 包含所有 Motif 节点的置信度均值和标准差、薄壁平均厚度、以及长宽比分布特征。
2. **逐面感应条件特征 `face_features` (维度：`30 × 32`)**：
   * 对每个 Face 槽位（最大 30）计算局部 Token。
   * 特征包含：面片 ID 归一化比例、面片激活状态；
   * 面片所归属的所有 Motif 节点类型的叠加概率、对应 Motif 的空间包围盒中心与尺度代理、法向代理向量、面积代理、薄度代理及长宽比。
3. **网络架构设计**：
   * `SRoute2MacroAdapter` 包含一个全局特征提取支路（`global_trunk`）用来提取全局设计风格上下文。
   * 将 `face_features` 与全局上下文拼接后，通过带有位置编码（Positional Encoding）的 3 层面感应 Transformer 编码器（`face_encoder`）充分交互，学习空间相对布局。
   * 采用解耦的预测头输出：`count_head` 回归面片数量；`bbox_head` 回归 6 维包围盒；`adj_pair_head` 采用自注意力对称矩阵回归两两面片之间共享边数。

---

## 三、 数据准备 (Data Preparation)

本地数据目录：

```text
data/deepcad30_s_ready/
```

当前样本数：5365。

筛选条件（仅筛选中小复杂度的零件以保证高精度缝合）：

```text
motif_prior_ready
face_count <= 30
max_edges_per_face <= 20
```

### 数据集提取方法：
运行以下脚本，自动将 `innovation1_v3` 提取出的 6111 个 ready 数据集进行面数与度数过滤，并在本地建立 train/val/test 的物理划分索引：
```powershell
Set-Location 'F:\开题答辩\a中期答辩专用\DTGBrepGen-master'
& 'F:\pytorch_cuda12\python.exe' .\innovation2_v2_s_prior_stepgen\prepare_deepcad30_dataset.py
```

---

## 四、 训练 Adapter (Adapter Training)

已有 checkpoint 时不需要重训。若有数据集更新或需要重训时：

```powershell
Set-Location 'F:\开题答辩\a中期答辩专用\DTGBrepGen-master'

& 'F:\pytorch_cuda12\python.exe' .\innovation2_v2_s_prior_stepgen\train_route2_adapters.py `
  --epochs 80 `
  --batch_size 64 `
  --num_workers 0
```

输出：
*   **最佳模型权重**：`checkpoints/route2_faceaware_adapter/best.pt`
*   **训练日志曲线**：`outputs/reports/route2_adapter_train_log.csv`

---

## 五、 稳定生成 STEP (Stable STEP Generation)

先生成中等复杂度形状，保证 STEP 成功率和连通性：

```powershell
Set-Location 'F:\开题答辩\a中期答辩专用\DTGBrepGen-master'

& 'F:\pytorch_cuda12\python.exe' .\innovation2_v2_s_prior_stepgen\sample_route2_step.py `
  --target_success 10 `
  --max_trials 40 `
  --min_sample_faces 6 `
  --max_sample_faces 10 `
  --max_sample_edges 24 `
  --sample_prefix route2_stable `
  --adapter_ckpt .\innovation2_v2_s_prior_stepgen\checkpoints\route2_faceaware_adapter\best.pt `
  --checkpoints_dir .\checkpoints_base\deepcad `
  --bbox_anchor_blend 0.30 `
  --build_retries 4 `
  --min_face_degree 3
```

输出：
*   **生成的 STEP 实体目录**：`outputs/steps_route2/`
*   **生成结果元数据清单**：`outputs/reports/route2_stepgen_results.csv`
*   **学术生成汇总报告**：`outputs/reports/route2_stepgen_report.txt`

### 推理时的数据流动链路：
1. **采样骨架**：由先验采样器 `EmpiricalSPriorSampler` 在先验库中随机抓取一个具有代表性的 $S^*$ 逻辑骨架。
2. **宏观布局推导**：自研适配器依据 $S^*$ 预测面数 $N_f$、粗包围盒及邻接关系，并通过 Motif 几何锚点进行空间修正。
3. **线面稀疏投影**：邻接 logits 通过最大置信度和面度数下限剪枝过滤为 `edgeFace_adj`。
4. **冻结级联解算**：
   * 将包围盒与 `edgeFace_adj` 送入 `edgeVert` 模块循环 10 次试错生成边点拓扑。
   * 级联至 `vertGeom` -> `edgeGeom` -> `faceGeom` 三维扩散去噪网络，最终解码出每个实体的物理三维坐标点云。
5. **OpenCASCADE 缝合**：调用原版 `get_brep` 提取三维多项式曲线与曲面并执行裁剪、缝合操作，最终持久化写出 STEP 实体。

---

## 六、 脚本模块清单与功能说明 (Script Files Catalog)

| 文件名 | 类型 | 核心作用与学术定位 |
| :--- | :--- | :--- |
| **`prepare_deepcad30_dataset.py`** | 预处理脚本 | 对原始 ready 数据集进行面数（$\le 30$）和面度数（$\le 20$）过滤，进行 train/val 划分。 |
| **`route2_adapter_models.py`** | 网络定义 | 定义先验向量化工程（`s_vec`、`face_features`）和面感应 Transformer 适配器结构。 |
| **`route2_dataset.py`** | 数据流 | 路线2专属数据读取，执行三倍坐标缩放（`3.0`）及拓扑离散化。 |
| **`train_route2_adapters.py`** | 训练器 | 多任务端到端联合训练 Adapter，优化宏观包围盒与拓扑预测。 |
| **`s_prior_sampler.py`** | 采样器 | 经验采样器（Empirical Sampler），无条件采出 $S^*$ 先验逻辑骨架。 |
| **`dtg_deepcad_frozen_generator.py`**| 封装层 | 只读封装和动态载入 DTG 六大官方权重，构建冻结的几何扩散解算器。 |
| **`sample_route2_step.py`** | 生成总入口 | 路线2的无条件生成总调度器，实现 $S^*$ 先验到最终缝合 STEP 实体的全流程。 |
| **`utils_io.py`** | 工具类 | 隔离辅助 IO 读写、多进程写及 Numpy/Torch 安全序列化支持。 |

---

## 七、 约束与注意事项 (Constraints & Cautions)

*   **局部几何硬修正禁用说明**：不要开启任何局部几何硬修正；强行拼合点线会破坏相邻面片之间的共享边界，导致 B-Rep 面片在大尺度位移时分离，缝合率反而下降。
*   **局部缝隙局限性**：少数生成的形状会存在局部面片质量较差、微小缝隙的情况（导致 OpenCASCADE 缝合失败），这是扩散模型缺乏局部控制点的通病，需后续训练局部几何精炼网络（Geometry Refiner）进一步解决，但目前主线能够稳定产生可用 STEP。
