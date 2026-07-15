# Innovation 1 v3：无语义 B-Rep 弱结构证据抽取与生成先验蒸馏方法汇总

本模块对应研究工作的第一个核心创新点：针对 ABC、DeepCAD 等公共无语义 B-Rep CAD 数据集，通过设计无领域限制的几何与拓扑分析算法，从 STEP 文件中自动抽取弱结构基元证据图 $M_{raw}$，并进一步蒸馏出高压缩、强物理约束的稀疏结构先验图 $S$，为后续的 B-Rep 生成模型提供结构与空间物理引导。

---

## 一、 模块设计定位与架构

在公共 CAD 数据集（如 ABC 和 DeepCAD）中，模型不包含如航空复材或特定机械装配的领域语义标签（例如没有 `panel`、`stiffener`、`flange` 等）。本模块放弃了强行拟合领域特定语义的做法，而是提出**无领域限制的“弱结构基元”提取理论**，从纯几何拓扑角度提炼零件的通用几何设计意图。

本模块的核心数据处理链路为：
$$\text{STEP B-Rep Solid} \xrightarrow{\text{多后端解析与清洗}} \text{pkl 拓扑几何缓存} \xrightarrow{\text{证据提取}} M_{raw} \xrightarrow{\text{先验蒸馏}} S$$

1. **$M_{raw}$ (完整弱结构证据图)**：包含零件中所有的几何基元、面群、孔洞/圆角/薄壁候选以及面与面、群与群之间的拓扑、空间支撑、几何关系。作为审计、统计、可视化和可解释分析的证据全集。
2. **$S$ (生成先验图)**：由 $M_{raw}$ 经蒸馏函数 $D(M_{raw})$ 过滤得到。只保留核心结构节点与平行、相对、共面等**强结构边**，去除密集的支撑与局部拓扑边，作为生成网络学习的稀疏结构骨架。

---

## 二、 多解析器后端与预处理流程 (`brep_loader.py`)

本模块支持多进程并发处理 STEP 数据集，并设计了多级解析与采样机制，保证提取出的微观几何适用于三维生成神经网络：

1. **多解析后端**：
   * **主路径**：优先调用 `occwl` 库并结合原版 DTG 的 `parse_solid` 算法，执行缝合、闭合面/边（closed faces/edges）的规范化拆分，以获得精确的拓扑图连接关系。
   * **Fallback 路径**：在主路径失败时，若本地可用 `pythonOCC`，则自动调用 `_parse_step_occ_fallback` 方法。该方法利用 OpenCASCADE 拓扑资源（`TopExp_Explorer`）提取 `TopAbs_SOLID`（要求 `solid_count == 1`），以及 `TopAbs_FACE`、`TopAbs_EDGE`、`TopAbs_VERTEX`，以确保解析鲁棒性。从 Fallback 提取的样本打上 `bbox_fallback_sampling` 标签，默认不作为严格的 Motif 训练真值。
2. **微观特征网格化采样**：
   * **面片采样**：对于每个 Face，在其局部 UV 参数空间（或三维包围盒平面内）执行 $32 \times 32 \times 3$ 的稠密三维网格采样（`_grid_from_bbox`），以供面片几何 VAE 的 Encoder 进行编码。
   * **边线采样**：对于每个 Edge，沿曲线参数空间（或包围盒对角线）采样 32 个三维坐标点（`_edge_points_from_bbox`），供线几何 VAE 编码。

---

## 三、 数据严格清洗与拓扑校验算法 (`brep_cleaner.py`)

为了保证三维生成网络（如 Transformer 解码器 and VAE）训练时不出现梯度爆炸或维度退化，本模块在数据加载阶段实施了严格的 B-Rep 清洗和拓扑校验：

1. **B-Rep 实体基础校验 (`validate_brep`)**：
   * **单实体约束**：必须满足 `solid_count == 1`，排除多实体装配体和散乱曲面。
   * **面数阈值过滤**：面片数量必须在 $0 < N_{faces} \le 70$（默认 70，DTG 兼容检查为 50）范围内。
   * **不可为空校验**：`edge_count` 和 `vertex_count` 必须大于 0。
   * **拓扑矩阵可构造性**：`edgeFace_adj` 形状必须为 $(N_{edges}, 2)$（多于两个相邻面的交线会被拒绝）；`edgeVert_adj` 形状必须为 $(N_{edges}, 2)$。
   * **有限值校验**：所有坐标网格、包围盒（`face_bbox_wcs`, `edge_bbox_wcs`, `global_bbox`）必须全部为有限值（通过 `np.all(np.isfinite(...))` 检查）。
   * **尺度合理性**：全局包围盒的最大对角线跨度 `global_scale` 必须大于 $10^{-6}$，过滤无效微小零件。
2. **派生拓扑邻接计算**：
   * **面-边-面连接矩阵 (`fef_adj`)**：通过 `build_fef_adj` 算法，计算每一对面之间共享边的数量，形成 $N_{faces} \times N_{faces}$ 的连通强度矩阵。
   * **点-面邻接列表 (`vertFace_adj`)**：通过 `build_vert_face_adj`，计算每个顶点所对应的面片索引集合。
3. **DTG 训练兼容性校验 (`check_dtg_train_compatible`)**：
   * **度数限制**：单顶点相连的面片数量不可超过 15。
   * **简单回路校验**：每个 Face 的边列表与顶点列表中，边的数量必须等于去重后的顶点数量（即满足简单闭合环定理 $\text{edges} = \text{vertices}$，剔除自交或非流形面）。
   * **几何重叠校验**：利用 `_has_duplicate_bbox` 剔除包围盒完全重合的面片或边线，防止拓扑歧义。

---

## 四、 面级特征与弱结构证据提取 (`motif_feature_extractor.py`)

该脚本负责对清洗后的 B-Rep 零件提取面片级的几何拓扑特征，并计算两两面片之间的 6 类物理设计意图证据：

1. **面级特征提取**：
   * **质心 (Centroid)**：面片包围盒中点 $C_i = \frac{1}{2}(\text{min}_i + \text{max}_i)$。
   * **法向代理 (Normal Proxy)**：利用 `_normal_from_grid` 提取。优先取 UV 参数网格顶点的平均叉乘；若失败，退化为 SVD 奇异值分解（`_normal_from_pca`）取最小特征值对应的特征向量作为法向；再失败则退化为包围盒最短轴（`_normal_from_bbox`）。
   * **面积代理 (Area Proxy)**：面片 BBox 三维尺寸排序后，最大两个维度的乘积。
   * **曲率代理 (Curvature Proxy)**：通过面片拟合评估，平面记为 0.0，曲面记为 1.0。
   * **边界标记 (Boundary Flag)**：面片 bbox 是否接触零件全局边界（容差为全局尺度的 2%）。
2. **面对关系判定标准（阈值机制）**：
   * **`adjacent_to` (相邻)**：共享至少一条边（`face_adj[i, j] > 0`）。
   * **`parallel_to` (平行)**：仅限于平面—平面，法向点积绝对值 $|N_i \cdot N_j| \ge \cos(0.1^\circ) \approx 0.999998$（即数学上完全平行）。
   * **`coplanar_with` (共面)**：仅限于平面—平面，已满足平行，且平面精细投影距离满足：
     $$\text{plane\_distance} = |(P_j - P_i) \cdot N_i| \le \max(10^{-4}, 10^{-5} \times \text{global\_scale})$$
   * **`opposite_to` (相对)**：仅限于平面—平面，已满足平行，且平面投影有效间距满足 $\max(10^{-4}, 10^{-5} \times \text{global\_scale}) \le \text{effective\_gap} \le 0.15 \times \text{global\_scale}$，法向相反且双向对向投影积判定为实体材料填充。
   * **`orthogonal_to` (正交)**：仅限于平面—平面，法向点积绝对值 $|N_i \cdot N_j| \le \sin(12^\circ) \approx 0.208$。
   * **`smooth_connected` (平滑相连)**：已满足相邻，且法向夹角满足 $\ge \cos(18^\circ) \approx 0.951$，表示相切过渡。优先使用高精度共享边局部法向计算，在降级匹配中对非平面—平面限制退化，避免产生缝线处的过渡噪点。

---

## 五、 结构基元节点与关系图构建算法 (`motif_graph_builder.py`)

`motif_graph_builder.py` 是本创新点的算法核心，它将底层的面级关系通过图聚类与物理启发式规则聚合为宏观的结构基元图。

### 1. 结构基元节点 ($V_m$) 提取算法

本模块共定义并抽取了 7 类结构节点：

* **`face_group` (连通面组)**：
  * **算法**：对所有 Face 建立并查集（`UnionFind`），当且仅当两个面片相邻且（共面或平滑连接）时进行合并。将几何上连续且法向一致的面群聚类为单一面组节点。
* **`sheet_region` (薄板/板状区域)**：
  * **算法**：在 `face_group` 基础上，筛选相对面积较大（面积分位数 $\ge 70\%$，或面积 $\ge 0.06$ 且薄度 $\le 0.08$）的面群，代表零件的主体板结构。
* **`boundary_group` (零件外轮廓/边界组)**：
  * **算法**：面群中超过 $50\%$ 的面片接触到全局包围盒边界。
* **`thin_wall_pair` (薄壁对)**：
  * **算法**：在不同的 `face_group` 之间，寻找存在 `opposite_to` 关系的面对，要求它们面积比例 $\ge 0.62$，相对面积 $\ge 0.015$，且投影重叠率有效（`projection_overlap_valid == True`），重叠率满足 `overlap >= 0.50`，且其精细有效平面间距满足薄壁厚度截断：
    $$\text{effective\_gap} \le \text{thin\_gap\_cut} = \max(10^{-5}, \min(0.08 \times \text{global\_scale}, 0.22 \times \text{major\_span}))$$
* **`loop_or_hole` (局部有界闭合环/孔洞候选)**：
  * **算法**：连通面群不处于全局边界上，面数 $\le 8$，且向外相邻的外部面群数量 $\ge 3$（形成包围围栏）。
* **`transition_group` (几何过渡/圆角候选)**：
  * **算法**：面积小于截断值（默认 0.1），面片长宽比大（$\ge 3.0$）或呈高曲率，且至少连接了两个比其面积大 1.4 倍的外部面群。
* **`repeated_feature` (重复特征簇)**：
  * **算法**：对相对面积 $\le 0.35$ 且面积大于 0 的面群，通过特征特征签名匹配进行 complete-link 层次聚类。两个面群相似的判定条件为：
    $$\text{dims\_rel\_diff} \le 0.28 \land \text{area\_ratio} \ge 0.62 \land \text{normal\_absdot} \ge 0.92 \land \text{face\_count\_gap} \le 1 \land \text{degree\_gap} \le 2.5$$
  * **间距正则度计算 (`_spacing_regular_score`)**：对重复面群的质心矩阵执行 SVD 奇异值分解，提取第一主成分方差占比计算线性度 `linearity`。将质心投影到主成分方向并排序，计算相邻投影间距的变异系数 `spacing_cv`。最终输出正则排列得分：
    $$\text{regular\_score} = \text{linearity} \times (1.0 - \min(\text{spacing\_cv}, 1.0))$$

### 2. 结构关系边 ($E_m$) 链接算法

在结构节点建立后，关系边通过以下逻辑映射到图空间：

* **`embedded_in` (包含层级)**：根据成员面片 ID 的子集包含关系（`issubset`）建立，用以追溯微观面片隶属于哪个宏观 Motif。
* **`adjacent_to` (拓扑邻接支撑)**：面群节点之间在边界上共享边。
* **物理几何边 (`parallel_to`, `opposite_to`, `orthogonal_to`, `coplanar_with`, `smooth_connected`)**：由两个 Motif 节点内部的面片对关系通过加权置信度合并而得。
* **`repeated_with` (重复链式边)**：
  * **算法**：对 `repeated_feature` 的成员，若超过 2 个，则沿着它们的一维 SVD 主方向投影顺序进行链式单向相连；否则进行两两组合相连。
* **`bounded_by` (边界限制边)**：
  * **算法**：非边界基元节点与边界组 `boundary_group` 的成员面片在拓扑上相邻时建立连接。

---

## 六、 结构图与先验蒸馏策略 (`S = D(M_c); M_c = C(M_raw)`)

在构建完包含完整物理特征和支撑细节的 $M_{raw}$ 后，本模块设计了**蒸馏函数 (Distillation Function)**，将 $M_{raw}$ 蒸馏压缩为面向神经网络生成输入的高质量稀疏骨架 $S$。

### 1. 节点蒸馏过滤 (`_prior_node_keep`)
* 过滤掉低置信度的 Motif 节点。
* 丢弃 `face_group` 这一纯底层支撑节点（它只作为包含关系的媒介，在生成先验中是冗余的）。
* 保留 `sheet_region`、`thin_wall_pair`、`repeated_feature`、`boundary_group`，以及高置信度的 `loop_or_hole` ($\ge 0.58$) 和 `transition_group` ($\ge 0.60$)。

### 2. 关系边蒸馏过滤与剪枝 (`_prune_prior_relations`)
* **去除支撑关系**：彻底过滤掉拓扑支撑边 `adjacent_to` 和包含层次边 `embedded_in`（只在 $M_{raw}$ 中保留用于后处理追溯）。
* **稀疏性约束（按重要性优先级保留）**：
  * 只保留属于 `PRIOR_RELATION_TYPES` 的 6 类关系：`opposite_to` > `parallel_to` > `coplanar_with` > `orthogonal_to` > `repeated_with` > `bounded_by`。
  * 根据关系置信度（必须大于各自阈值，如 `parallel_to` $\ge 0.72$）降序排列。
  * 对每个节点所关联的边关系数量进行上限预算剪枝（Degree Budgeting），确保先验图的稀疏性，防止过度密集的完全图边干扰神经网络注意力权重的学习。

---

## 七、 模块输出文件与统计审计指标

### 1. 输出文件规范
模块处理数据后，在 `outputs/` 下输出以下物理文件：
* **`outputs/parsed/clean_manifest.csv`**：清洗后、所有拓扑与几何指标完全合法、DTG 兼容的零件清单。
* **`outputs/parsed/rejected_manifest.csv`**：记录所有被拒绝的 STEP 零件及具体的拒绝原因（如 `face_count_over_limit`、`not_single_solid` 等），用于数据审计。
* **`outputs/motif_graphs/motif_graph_index.jsonl`**：全量 $M_{raw}$ 索引文件，包含所有支撑和物理节点。
* **`outputs/motif_graphs/motif_prior_index_ready.jsonl`**：最终过滤清洗出的高品質先验图 $S$ 索引文件，作为创新点二的训练数据集输入。
* **`outputs/reports/motif_extraction_report.txt`**：自动生成的中文学术统计报告，包含压缩率、图密度等指标。

### 2. 统计审计指标定义 (`motif_metrics.py`)
在最终输出的审计报告中，设计了如下量化评估指标：
* **基元压缩率 (Node Compression Ratio)**：
  $$\text{Compression Ratio} = \frac{N_{faces} - N_{motif\_nodes}}{N_{faces}}$$
  用以评估结构先验对离散几何面片的概括和信息压缩能力（通常在 $30\% \sim 60\%$ 之间）。
* **先验图密度 (Graph Density)**：
  $$\text{Density} = \frac{2 \times |E_s|}{|V_s| \times (|V_s| - 1)}$$
  评估蒸馏后 $S$ 的稀疏程度，反映模型是否成功去除了冗余连接。
* **基元覆盖率 (Motif Coverage)**：
  $$\text{Coverage} = \frac{\left| \bigcup_{v \in V_s} \text{face\_ids}(v) \right|}{N_{faces}}$$
  零件中被结构 Motif 包含的微观面片比例，评估该零件结构表达的完整度。
