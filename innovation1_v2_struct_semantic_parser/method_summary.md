# 创新点 1 v2 方法总结

## 方法定义

本 v2 将创新点 1 定义为：

**航空复材薄壁加筋构件结构构型图先验表达与 B-Rep 弱语义解析方法。**

它在原创新点 1 的参数化构件生成、`Gc=(Vc,Ec,P)` 构型图表达、face group 对齐和 DTG-compatible B-Rep 数据接口基础上，进一步增强结构复杂度与 B-Rep 语义对齐可信度。

## 核心内容

1. **Gc=(Vc,Ec,P) 结构先验表达**  
   每个样本同步保存结构节点、结构关系和参数表，用于描述 `panel`、`web`、`flange`、`cap`、`stiffener`、`transition`、`hole`、`cutout`、`runout` 等薄壁加筋构件语义单元。

2. **procedural_Gc 程序化结构监督标签**  
   enhanced 样本在生成 STEP/STL 几何的同时生成程序化构型图、参数和拓扑机制。正式字段为 `procedural_expected_face_groups`，只保留节点与角色预期，不提供由程序直接得到的 face ids；`procedural_face_groups` 仅作为兼容旧接口的别名，不应被理解为 face-level 真值。

3. **weak_aligned_face_groups 弱几何对齐**  
   `parse_enhanced` 阶段会基于 B-Rep 几何规则生成 `weak_aligned_face_groups`。该字段用于训练接口和一致性审计，但不是人工标注真值，也不是 procedural 生成器直接输出的 face-level 真值。

4. **inferred_Gc B-Rep 几何弱语义解析**  
   从解析后的 B-Rep PKL 中提取 `face_bbox_wcs`、`face_wcs`、邻接关系、边界位置、尺寸比例、重复阵列结构、`face_surface_type` 和 `face_curvature_proxy` 等特征，规则推断 face role、face group、构型图和参数。

5. **procedural_Gc 与 inferred_Gc 一致性评估**  
   node、relation、parameter、topology mechanism 主要用于 procedural_Gc 与 inferred_Gc 的结构一致性审计。face-level 指标命名为 `weak_face_role_consistency` 和 `weak_face_group_iou`，表示弱对齐结果与 inferred parser 的一致性，不表示人工真值准确率。参数一致性采用 class-aware meaningful parameter L1，只比较当前构件类型有实际几何意义的参数；报告同时输出 `parameter_l1_scale_normalized` 和 `parameter_l1_abs_mm`。其中前者是 DTG 坐标可能标准化情况下的主指标，后者仅在 `geometry_sampling_quality=bbox_fallback_sampling` 时输出，否则留空为 NA。

6. **面向创新点 2 的数据接口**  
   输出 `enhanced_dataset`、`enhanced_parsed`、`inferred_semantics`、`tensor_schema.json`、`data_splits.csv`、`weak_aligned_face_group_index.jsonl`，并额外输出当前模型维度可直接使用的 `enhanced_lite` 子集索引。

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

本版本不依赖不稳定的全边自动倒圆，而是采用选择性、结构语义相关的显式过渡建模：

- 加筋壁板类在 stiffener 与 panel 连接根部加入 `root_fillet_radius` 或根部过渡面。
- `tapered_c_channel` 以 `web_0` 为主结构节点，在 web-flange 折角处加入显式截面圆弧 transition。
- `tapered_hat_stiffener` 以 `web_0` 和 `cap_0` 表示帽型筋主结构，不强行使用 `panel_0`；web-flange 与 web-cap 连接处加入显式截面圆弧 transition。
- `panel_with_rectangular_cutout` 使用圆角矩形切割 wire，形成 `cutout_corner_radius` 对应的角部 transition。
- `stiffener_runout_panel` 使用沿长度方向变化的闭合截面 loft 表示筋条终止过渡，端部采用非退化 toe height 近似，避免零高度截面导致 OCC 退化，也避免简单截断。

## 参数范围与硬约束

所有 enhanced 样本采用 mm 单位，定位为航空复材薄壁加筋子构件 / 试件级几何，不代表真实整机部件规范。

主要范围包括 `length=250-1200`、`width=80-900`、`thickness=1.5-5.0`、`rib_count=0-6`、`hole_count=0-4`、`curvature_radius=500-5000` 等。圆角约束包括：

- `fillet_radius`: `max(1.5*thickness, 3.0)` 到 `min(8*thickness, 0.25*local_min_size)`
- `root_fillet_radius`: `2*thickness` 到 `min(5*thickness, 0.25*min(rib_width,rib_height))`
- `cutout_corner_radius`: `2*thickness` 到 `min(6*thickness, 0.2*min(hole_width,hole_height))`

主要硬约束包括薄壁厚度比例、孔边距、孔间距、筋条间距、runout 长度、taper 后端部最小尺寸、曲率半径，以及 STEP/STL 单实体写出检查。失败样本必须写入报告，不静默跳过。

加筋壁板类的 `height` 定义为 `thickness + rib_height`，与生成几何的整体高度保持一致；未在几何中实现的参数不应作为有效监督标签，例如当前 `notch_depth` 固定为 0。

当前 `tapered_hat_stiffener` 采用实体化薄壁截面代理：由闭合 hat 截面 loft 得到可稳定解析的 B-Rep 实体，用于算法验证和结构语义对齐，不代表真实复材帽型筋铺层厚度细节。后续若追求更高工程真实性，可改为 cap plate、web plates、flange plates 与 transition arcs 的组合式薄壁建模，或使用外轮廓/内轮廓构造薄壁闭合截面。

`stiffener_runout_panel` 的 runout 区域不是让筋条高度严格降为 0，而是使用非退化 toe height 到 full height 再回到 toe height 的渐变截面近似。这样可以表达筋条终止过渡区，同时避免 loft 端部退化。

弱参数解析会输出 `thickness_estimation_source` 诊断字段，用于区分 `panel_face_thin_dim`、`beam_web_flange_cap_thin_dim` 和 `fallback_global_min_dim`。其中 tapered 梁类的 thickness 来自 web/flange/cap face 薄尺度候选，仍属于弱估计，不作为工程精确测厚。

## B-Rep 解析质量

解析流程优先使用 DTG/occwl，若失败则使用 pythonOCC fallback 提取 bbox、拓扑邻接和基本 WCS 兼容字段。需要注意：

- `geometry_sampling_quality=true_or_dtg_sampling` 的样本可用于真实 `face_wcs/edge_wcs` 几何训练。
- `geometry_sampling_quality=bbox_fallback_sampling` 的样本中 `face_wcs/edge_wcs` 是 bbox 近似网格/线段，只适合 bbox、topology 和弱语义审计，不应作为真实曲面采样训练依据。
- 当 DTG/occwl 成功解析后，若再由 pythonOCC 重新读取 STEP 补充 `face_surface_type` / `face_curvature_proxy`，face 顺序可能未严格验证；此时 `surface_metadata_order_verified=false`，语义解析不应把该曲率代理作为强判据。
- 曲面 panel 的 role 识别在 surface metadata 顺序未验证时采用 bbox/面积/长宽比/skin-zone 位置联合弱规则，不声称是强曲率语义识别；曲面加筋区域仍可能存在 panel、transition 和 stiffener 的局部混淆。
- 曲面加筋板的 `rib_count` 采用简化局部坐标思想：先估计蒙皮 skin-zone，再对高于蒙皮且沿长度方向延伸的 rib candidate faces 按横向中心聚类为 rib tracks。`stiffener_runout_panel` 的 `rib_count` 优先来自 runout face group 的横向 track，而不是把所有高位过渡面都计为筋条。

## 谨慎边界

- 当前 enhanced dataset 仍然是程序化试件级简化几何，不等同于真实工业型号的完整工程 CAD。
- `inferred_Gc` 是规则弱解析结果，不是真实工程人工语义标注。
- `weak_aligned_face_groups` 是几何规则弱对齐结果，不是人工真值，也不应用来宣称 face-level 真实准确率。
- `procedural_expected_face_groups` 是程序生成时同步得到的结构节点预期；`procedural_face_groups` 只是兼容旧接口的空 face-id 别名。
- 本方法不声称覆盖全部航空复材构件，仅聚焦薄壁加筋构件中孔、开口、加筋、渐缩、曲面、runout 和主连接过渡等典型机制。
- enhanced 样本的 face / edge / vertex 数可能超过创新点 2 当前 `max_faces=30, max_edges=68, max_vertices=40`。后续训练应选择提高模型上限，或使用 `enhanced_lite_uids.txt`、`data_splits_lite.csv` 和 `weak_aligned_face_group_index_lite.jsonl`。

## 主要输出

- `outputs/enhanced_dataset`: enhanced STEP/STL/JSON 样本。
- `outputs/enhanced_parsed`: DTG-compatible PKL、schema、split、weak aligned face group index、enhanced_lite 子集。
- `outputs/inferred_semantics`: inferred face groups、config graphs、parameters。
- `outputs/reports`: 数据生成、B-Rep 解析、弱语义解析、一致性评估和总审计报告。
