# 创新点 1 v2 方法总结

## 方法定义

本 v2 将创新点 1 定义为：

**航空复材薄壁加筋构件结构构型图先验表达与 B-Rep 弱语义解析方法。**

它在原创新点 1 的参数化构件生成、`Gc=(Vc,Ec,P)` 构型图表达、face group 对齐和 DTG-compatible B-Rep 数据接口基础上，进一步增强结构复杂度与 B-Rep 语义对齐可信度。

## 核心内容

1. **Gc=(Vc,Ec,P) 结构先验表达**  
   每个样本同步保存结构节点、结构关系和参数表，用于描述 `panel`、`web`、`flange`、`stiffener`、`transition`、`hole`、`cutout`、`runout` 等薄壁加筋构件语义单元。

2. **procedural_Gc 程序化结构监督标签**  
   enhanced 样本在生成 STEP/STL 几何的同时生成程序化构型图、参数、拓扑机制和待对齐 face group，作为后续创新点 2 可读取的结构监督数据。

3. **inferred_Gc B-Rep 几何弱语义解析**  
   从解析后的 B-Rep PKL 中提取 `face_bbox_wcs`、`face_wcs`、邻接关系、边界位置、尺寸比例、重复阵列结构、`face_surface_type` 和 `face_curvature_proxy` 等特征，规则推断 face role、face group、构型图和参数。

4. **procedural_Gc 与 inferred_Gc 一致性评估**  
   使用 face role accuracy、face group IoU、node/relation type accuracy、parameter L1、hole detection、stiffener count 和 topology mechanism accuracy 等指标，评估程序化标签与弱解析结果的一致性。

5. **面向创新点 2 的数据接口**  
   输出 `enhanced_dataset`、`enhanced_parsed`、`inferred_semantics`、`tensor_schema.json`、`data_splits.csv`、`face_group_index.jsonl`，并额外输出当前模型维度可直接使用的 `enhanced_lite` 子集索引。

## 增强样本范围

当前 enhanced dataset 聚焦航空复材薄壁加筋构件内部复杂机制，包括：

- circular / rectangular cutout
- stiffened panel with cutout
- multi-stiffened panel
- asymmetric stiffened panel
- tapered C-channel
- tapered hat stiffener
- curved panel
- curved stiffened panel
- stiffener runout panel
- root fillet / web-flange fillet / web-cap fillet / cutout-corner transition face

## 主连接圆角与过渡

本版本不再依赖不稳定的全边自动倒圆，而是采用选择性、结构语义相关的显式过渡建模：

- `stiffened_panel_with_cutout`、`multi_stiffened_panel`、`asymmetric_stiffened_panel`、`curved_stiffened_panel` 和 `stiffener_runout_panel` 在 stiffener 与 panel 连接根部加入 `root_fillet_radius` 或根部过渡面。
- `tapered_c_channel` 在 web-flange 折角处加入显式截面圆弧 transition。
- `tapered_hat_stiffener` 在 web-flange 和 web-cap 连接处加入显式截面圆弧 transition。
- `panel_with_rectangular_cutout` 使用圆角矩形切割 wire，形成 `cutout_corner_radius` 对应的角部 transition。
- `stiffener_runout_panel` 使用沿长度方向变化的闭合截面 loft 表示筋条终止过渡，避免简单截断。

这些 transition 在 `configuration_graph` 中对应 `transition` 节点和 `smooth_connected` 关系，用于支撑 B-Rep 弱语义解析。

## 参数范围与硬约束

所有 enhanced 样本采用 mm 单位，定位为航空复材薄壁加筋子构件 / 试件级几何，不代表真实整机部件规范。

主要范围：

- `length`: 250-1200
- `width`: 80-900
- `thickness`: 1.5-5.0
- `height`: 20-150
- `flange_width`: 12-100
- `rib_width`: 10-80
- `rib_height`: 15-120
- `rib_count`: 0-6
- `fillet_radius`: `max(1.5*thickness, 3.0)` 到 `min(8*thickness, 0.25*local_min_size)`
- `root_fillet_radius`: `2*thickness` 到 `min(5*thickness, 0.25*min(rib_width,rib_height))`
- `cutout_corner_radius`: `2*thickness` 到 `min(6*thickness, 0.2*min(hole_width,hole_height))`
- `hole_radius`: 8-80
- `hole_width`: 25-220
- `hole_height`: 20-160
- `hole_count`: 0-4
- `taper_ratio`: 0.55-1.45
- `curvature_radius`: 500-5000
- `runout_length`: 50-300
- `notch_depth`: 5-60

主要硬约束包括薄壁厚度比例、孔边距、孔间距、筋条间距、runout 长度、taper 后端部最小尺寸、曲率半径，以及 STEP/STL 单实体写出检查。失败样本必须写入报告，不静默跳过。

## 谨慎边界

- 当前 enhanced dataset 仍然是程序化生成样本，不等同于真实工业型号的完整人工标注数据集。
- `inferred_Gc` 是规则弱解析结果，不是真实工程人工语义标注。
- 本方法不声称覆盖全部航空复材构件，仅聚焦薄壁加筋构件中孔、开口、加筋、渐缩、曲面、runout 和主连接过渡等典型机制。
- face group 对齐用于构建可训练接口；一致性指标应结合 `semantic_consistency_report.txt` 中的失败原因解释，不应被写成“人工真值准确率”。
- enhanced 样本的 face / edge / vertex 数可能超过创新点 2 当前 `max_faces=30, max_edges=68, max_vertices=40`。后续训练应选择提高模型上限，或使用 `enhanced_lite_uids.txt`、`data_splits_lite.csv` 和 `face_group_index_lite.jsonl`。

## 主要输出

- `outputs/enhanced_dataset`: enhanced STEP/STL/JSON 样本。
- `outputs/enhanced_parsed`: DTG-compatible PKL、schema、split、face group index、enhanced_lite 子集。
- `outputs/inferred_semantics`: inferred face groups、config graphs、parameters。
- `outputs/reports`: 数据生成、B-Rep 解析、弱语义解析、一致性评估和总审计报告。
