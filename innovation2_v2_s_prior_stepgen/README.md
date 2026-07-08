# 创新点2 v2：S 先验驱动的无条件 STEP 生成

本目录只保留稳定主线：

```text
无条件采样 S*
  -> face-aware S-conditioned adapter 预测 face bbox / face-edge topology
  -> 冻结 DTG-DeepCAD edge-vertex 与几何模块
  -> 输出 STEP
```

不修改 DTG 原始源码，不启用会破坏共享边界的局部几何硬修正。

## 数据

本地数据目录：

```text
data/deepcad30_s_ready/
```

当前样本数：5365。

筛选条件：

```text
motif_prior_ready
face_count <= 30
max_edges_per_face <= 20
```

## 训练 Adapter

已有 checkpoint 时不需要重训。需要重训时：

```powershell
Set-Location 'F:\开题答辩\a中期答辩专用\DTGBrepGen-master'

& 'F:\pytorch_cuda12\python.exe' .\innovation2_v2_s_prior_stepgen\train_route2_adapters.py `
  --epochs 80 `
  --batch_size 64 `
  --num_workers 0
```

输出：

```text
checkpoints/route2_faceaware_adapter/best.pt
```

## 稳定生成 STEP

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

```text
outputs/steps_route2/
outputs/reports/route2_stepgen_results.csv
outputs/reports/route2_stepgen_report.txt
```

## 注意

- 不要开启任何局部几何硬修正；它会破坏共享边界，导致面片分离。
- 复杂自由曲面局部面质量需要后续几何 refiner 训练解决；当前主线先稳定生成可用 STEP。
