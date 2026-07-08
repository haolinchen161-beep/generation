"""路线2：S-conditioned 宏观适配器驱动的无条件 STEP 生成。

流程：
随机种子 -> 无条件采样 S* -> S adapter 生成 face bbox / face-edge topology
-> 冻结 DTG edge-vertex 与几何模块 -> STEP。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from dtg_deepcad_frozen_generator import FrozenDeepCADDTGGenerator
from route2_adapter_models import SRoute2MacroAdapter, sort_minmax_bbox_tensor, vectorize_s_faces, vectorize_s_prior
from s_prior_sampler import EmpiricalSPriorSampler
from utils_io import ensure_dir, repo_root_from_file, save_pickle, workdir_from_file, write_csv, write_jsonl


def get_args() -> argparse.Namespace:
    workdir = workdir_from_file()
    parser = argparse.ArgumentParser(description="路线2：S-conditioned adapter 无条件生成 STEP")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--dataset_dir", default=str(workdir / "data" / "deepcad30_s_ready"))
    parser.add_argument("--adapter_ckpt", default=str(workdir / "checkpoints" / "route2_faceaware_adapter" / "best.pt"))
    parser.add_argument("--checkpoints_dir", default=str(repo_root_from_file() / "checkpoints_base" / "deepcad"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_faces", type=int, default=30)
    parser.add_argument("--max_edge_per_face", type=int, default=20)
    parser.add_argument("--min_face_degree", type=int, default=3, help="路线2拓扑投影时每个 face 至少保留的邻接边数")
    parser.add_argument("--target_success", type=int, default=0, help="目标成功 STEP 数；0 表示只尝试 num_samples 个样本")
    parser.add_argument("--max_trials", type=int, default=0, help="target_success 模式下最多尝试的 S* 数量；0 表示自动设为 4 倍目标数")
    parser.add_argument("--min_sample_faces", type=int, default=2, help="S* 采样池的最小 face 数")
    parser.add_argument("--max_sample_faces", type=int, default=0, help="S* 采样池的最大 face 数；0 表示不额外限制")
    parser.add_argument("--max_sample_edges", type=int, default=0, help="S* 采样池的最大 edge 数；0 表示不额外限制")
    parser.add_argument("--clean_output", action="store_true", help="运行前清理 route2_s_uncond_* 旧输出，避免新旧样本混在一起")
    parser.add_argument("--sample_prefix", default="route2_s_uncond", help="输出样本名前缀，用于避免不同实验互相覆盖")
    parser.add_argument("--bbox_anchor_blend", type=float, default=0.35, help="用 S 中的 face/motif bbox 锚点修正 adapter bbox 的强度；0 表示关闭")
    parser.add_argument("--bbox_anchor_max_shift", type=float, default=0.75, help="bbox 锚点修正的单坐标最大位移，单位为 DTG 缩放坐标")
    parser.add_argument("--build_retries", type=int, default=2, help="每个 S* 的 STEP 构造最大尝试次数")
    parser.add_argument("--edge_classes", type=int, default=5)
    parser.add_argument("--use_pred_count", action="store_true", help="使用 adapter 预测面数；默认使用采样 S* 的 num_faces")
    return parser.parse_args()


def _load_adapter(path: Path, device: torch.device) -> SRoute2MacroAdapter:
    ckpt = torch.load(path, map_location=device)
    config = ckpt["config"]
    model = SRoute2MacroAdapter(
        s_dim=config["s_dim"],
        max_faces=config["max_faces"],
        edge_classes=config["edge_classes"],
        hidden_dim=config["hidden_dim"],
        face_dim=config.get("face_dim"),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _adjacency_to_edgeface(
    adj_logits: np.ndarray,
    nf: int,
    target_edge_count: int,
    max_edge_per_face: int,
    min_face_degree: int = 3,
) -> np.ndarray:
    """按 S 的 edge 数预算，把 adjacency logits 稀疏投影为 edgeFace_adj。"""
    logits = np.asarray(adj_logits, dtype=np.float64)[:nf, :nf]
    if logits.ndim != 3 or logits.shape[-1] < 2:
        return np.empty((0, 2), dtype=np.int64)
    logits = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / np.maximum(probs.sum(axis=-1, keepdims=True), 1e-12)

    # S 里有源样本的 num_edges，这是当前无条件采样 S* 后最可靠的拓扑规模先验。
    min_edges = int(np.ceil(max(nf, 2) * max(min_face_degree, 1) / 2.0))
    target = int(target_edge_count or round(2.4 * nf))
    target = max(target, min_edges)
    target = min(target, nf * max_edge_per_face // 2)

    candidates: List[Tuple[float, float, int, int, int]] = []
    for i in range(nf):
        for j in range(i + 1, nf):
            p0 = float(probs[i, j, 0])
            nonzero = probs[i, j, 1:]
            best_count = int(np.argmax(nonzero)) + 1
            p_nonzero = float(nonzero.sum())
            # 用 margin 排序，比直接 argmax 更能抑制“弱阳性边”造成的过密拓扑。
            score = p_nonzero - p0
            candidates.append((score, p_nonzero, i, j, best_count))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

    adj = np.zeros((nf, nf), dtype=np.int64)
    degree = np.zeros((nf,), dtype=np.int64)
    edge_total = 0

    def add_edge(i: int, j: int, count: int = 1) -> bool:
        nonlocal edge_total
        if i == j or degree[i] >= max_edge_per_face or degree[j] >= max_edge_per_face:
            return False
        if edge_total >= target:
            return False
        count = max(1, min(int(count), target - edge_total))
        count = min(count, max_edge_per_face - int(degree[i]), max_edge_per_face - int(degree[j]))
        if count <= 0:
            return False
        adj[i, j] += count
        adj[j, i] += count
        degree[i] += count
        degree[j] += count
        edge_total += count
        return True

    # 先补低度 face，再按全局置信度填充到 edge budget。
    for _score, _p, i, j, _best_count in candidates:
        if edge_total >= target:
            break
        if degree[i] < min_face_degree or degree[j] < min_face_degree:
            add_edge(i, j, 1)
        if np.all(degree >= min_face_degree):
            break

    for _score, _p, i, j, best_count in candidates:
        if edge_total >= target:
            break
        if adj[i, j] == 0:
            add_edge(i, j, 1)

    # 只有在 edge budget 仍未用完时，才允许少量多重 edge。
    for _score, _p, i, j, best_count in candidates:
        if edge_total >= target:
            break
        extra = max(0, best_count - int(adj[i, j]))
        if extra > 0:
            add_edge(i, j, extra)

    # 控制每个面连接边数，尽量保持 DTG DeepCAD 的 max_edge=20。
    degree = adj.sum(axis=1)
    guard = 0
    while degree.max(initial=0) > max_edge_per_face and guard < 1000:
        guard += 1
        face = int(np.argmax(degree))
        neighbors = np.where(adj[face] > 0)[0]
        if len(neighbors) == 0:
            break
        removable = [n for n in neighbors if adj[n].sum() > min_face_degree and abs(n - face) not in (1, nf - 1)]
        target_neighbors = removable or list(neighbors)
        n = max(target_neighbors, key=lambda item: adj[face, item])
        if adj[n].sum() <= min_face_degree:
            break
        adj[face, n] -= 1
        adj[n, face] -= 1
        degree = adj.sum(axis=1)

    edge_faces = []
    for i in range(nf):
        for j in range(i + 1, nf):
            count = int(adj[i, j])
            for _ in range(count):
                edge_faces.append([i, j])
    return np.asarray(edge_faces, dtype=np.int64)


def _bbox_anchor_from_s(
    s_prior: Dict[str, Any],
    nf: int,
    bbox_scaled: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """从 S 的 motif 节点中提取逐 face bbox 锚点。"""
    type_weights = {
        "face_group": 1.0,
        "sheet_like_group": 0.9,
        "transition_group": 0.75,
        "boundary_group": 0.55,
        "loop_or_hole": 0.45,
        "thin_wall_pair": 0.35,
        "repeated_feature": 0.3,
    }
    anchors = np.zeros((nf, 6), dtype=np.float32)
    weights = np.zeros((nf,), dtype=np.float32)
    for node in s_prior.get("motif_nodes", []):
        features = node.get("features", {})
        bbox = np.asarray(features.get("bbox", []), dtype=np.float32).reshape(-1)
        if bbox.size != 6 or not np.all(np.isfinite(bbox)):
            continue
        bbox = np.concatenate([np.minimum(bbox[:3], bbox[3:]), np.maximum(bbox[:3], bbox[3:])]).astype(np.float32)
        node_type = str(node.get("type", ""))
        weight = type_weights.get(node_type, 0.25) * float(node.get("confidence", 0.5) or 0.5)
        if weight <= 0.0:
            continue
        for face_id in node.get("face_ids", []):
            try:
                face_id = int(face_id)
            except Exception:
                continue
            if 0 <= face_id < nf:
                anchors[face_id] += bbox * float(bbox_scaled) * weight
                weights[face_id] += weight
    active = weights > 1e-6
    anchors[active] /= weights[active, None]
    # 多个 motif 支撑同一 face 时提高可信度，但最高不超过 1。
    confidence = np.clip(weights / 1.5, 0.0, 1.0)
    return anchors, confidence


def _refine_bbox_with_s_anchor(
    face_bbox: torch.Tensor,
    s_prior: Dict[str, Any],
    blend: float,
    max_shift: float,
    bbox_scaled: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """用 S 的 bbox 锚点做小幅几何修正，减少下游 BRep 构造失败。"""
    blend = float(np.clip(blend, 0.0, 1.0))
    if blend <= 0.0 or len(face_bbox) == 0:
        return face_bbox, {"bbox_anchor_enabled": False, "bbox_anchor_used_faces": 0}
    anchors_np, conf_np = _bbox_anchor_from_s(s_prior, len(face_bbox), bbox_scaled=bbox_scaled)
    active = conf_np > 1e-6
    if not np.any(active):
        return face_bbox, {"bbox_anchor_enabled": True, "bbox_anchor_used_faces": 0}

    anchors = torch.from_numpy(anchors_np).to(device=face_bbox.device, dtype=face_bbox.dtype)
    conf = torch.from_numpy(conf_np).to(device=face_bbox.device, dtype=face_bbox.dtype).unsqueeze(-1)
    effective = blend * conf
    delta = torch.clamp(anchors - face_bbox, min=-float(max_shift), max=float(max_shift))
    refined = face_bbox + effective * delta
    refined = sort_minmax_bbox_tensor(refined)
    return refined, {
        "bbox_anchor_enabled": True,
        "bbox_anchor_blend": blend,
        "bbox_anchor_max_shift": float(max_shift),
        "bbox_anchor_used_faces": int(np.count_nonzero(active)),
        "bbox_anchor_mean_confidence": float(conf_np[active].mean()) if np.any(active) else 0.0,
    }


def _generate_edge_vertices(generator: FrozenDeepCADDTGGenerator, edge_face_np: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
    from topology.topoGenerate import SeqGenerator  # noqa: WPS433
    from topology.transfer import faceVert_from_edgeVert, face_vert_trans, fef_from_faceEdge  # noqa: WPS433
    from utils import calculate_y  # noqa: WPS433

    device = generator.device
    ef_adj = torch.from_numpy(edge_face_np).long().to(device)
    if ef_adj.numel() == 0:
        return False, {"reason": "empty_edge_face"}
    share_id = calculate_y(ef_adj)
    generator.edgeVert_model.save_cache(
        edgeFace_adj=ef_adj.unsqueeze(0),
        edge_mask=torch.ones((1, ef_adj.shape[0]), device=device, dtype=torch.bool),
        share_id=share_id,
        class_label=None,
        point_data=None,
    )
    for _ in range(10):
        seq_generator = SeqGenerator(edge_face_np)
        if seq_generator.generate(generator.edgeVert_model, None):
            generator.edgeVert_model.clear_cache()
            ev_adj = seq_generator.edgeVert_adj
            fe_adj = seq_generator.faceEdge_adj
            fv_adj = faceVert_from_edgeVert(fe_adj, ev_adj)
            vf_adj = face_vert_trans(faceVert_adj=fv_adj)
            fef = fef_from_faceEdge(edgeFace_adj=edge_face_np)
            return True, {
                "edgeVert_adj": torch.from_numpy(ev_adj).long().to(device),
                "faceEdge_adj": fe_adj,
                "vertFace_adj": vf_adj,
                "fef_adj": torch.from_numpy(fef).long().to(device),
            }
    generator.edgeVert_model.clear_cache()
    return False, {"reason": "edge_vertex_generation_failed"}


def _build_one_step(
    generator: FrozenDeepCADDTGGenerator,
    sample_id: str,
    face_bbox: torch.Tensor,
    edge_face_np: np.ndarray,
    output_dir: Path,
) -> Dict[str, Any]:
    from inference.generate import get_brep, get_edgeGeom, get_faceGeom, get_vertGeom  # noqa: WPS433
    from OCC.Extend.DataExchange import write_step_file  # noqa: WPS433

    topo_ok, topo = _generate_edge_vertices(generator, edge_face_np)
    if not topo_ok:
        return {
            "sample_id": sample_id,
            "build_success": False,
            "face_count": int(len(face_bbox)),
            "edge_count": int(len(edge_face_np)),
            "edgeFace_adj": edge_face_np,
            "face_bbox": face_bbox.detach().cpu().numpy(),
            **topo,
        }

    device = generator.device
    face_bbox = face_bbox.to(device)
    edge_face = torch.from_numpy(edge_face_np).long().to(device)
    with torch.no_grad():
        vert_geom, vert_mask = get_vertGeom(
            [face_bbox],
            [topo["vertFace_adj"]],
            [topo["edgeVert_adj"]],
            generator.vertGeom_model,
            generator.pndm_scheduler,
            generator.ddpm_scheduler,
            class_label=None,
            point_data=None,
        )
        vert_geom = [k[l] for k, l in zip(vert_geom, vert_mask)]
        edge_geom, edge_mask = get_edgeGeom(
            [face_bbox],
            vert_geom,
            [edge_face],
            [topo["edgeVert_adj"]],
            generator.edgeGeom_model,
            generator.pndm_scheduler,
            generator.ddpm_scheduler,
            class_label=None,
            point_data=None,
        )
        edge_geom = [k[l] for k, l in zip(edge_geom, edge_mask)]
        face_geom, face_mask = get_faceGeom(
            [face_bbox],
            vert_geom,
            edge_geom,
            [topo["faceEdge_adj"]],
            [edge_face],
            [topo["edgeVert_adj"]],
            generator.faceGeom_model,
            generator.pndm_scheduler,
            generator.ddpm_scheduler,
            class_label=None,
            point_data=None,
        )
        face_geom = [k[l] for k, l in zip(face_geom, face_mask)]

    bbox_scaled = float(getattr(generator.args, "bbox_scaled", 3.0))
    face_bbox_np = face_bbox.detach().cpu().numpy() / bbox_scaled
    vert_geom_np = vert_geom[0].detach().cpu().numpy() / bbox_scaled
    edge_geom_np = edge_geom[0].detach().cpu().numpy() / bbox_scaled
    face_geom_np = face_geom[0].detach().cpu().numpy() / bbox_scaled
    edge_vert_np = topo["edgeVert_adj"].detach().cpu().numpy()
    step_path = output_dir / f"{sample_id}_route2.step"
    try:
        solid = get_brep(
            (
                face_bbox_np,
                vert_geom_np,
                edge_geom_np,
                face_geom_np,
                edge_face_np,
                edge_vert_np,
                topo["faceEdge_adj"],
            ),
            save_name=str(step_path.with_suffix("")),
        )
        if solid is not False:
            write_step_file(solid, str(step_path))
            success = True
        else:
            success = False
    except Exception as exc:
        return {"sample_id": sample_id, "build_success": False, "reason": repr(exc)}

    return {
        "sample_id": sample_id,
        "build_success": success,
        "reason": "" if success else "brep_build_failed",
        "step_path": str(step_path) if success else "",
        "face_count": int(len(face_bbox_np)),
        "edge_count": int(len(edge_face_np)),
        "vertex_count": int(len(vert_geom_np)),
        "face_bbox": face_bbox_np,
        "vert_geom": vert_geom_np,
        "edge_geom": edge_geom_np,
        "face_geom": face_geom_np,
        "edgeFace_adj": edge_face_np,
        "edgeVert_adj": edge_vert_np,
    }


def _clean_previous_outputs(step_dir: Path, pkl_dir: Path, samples_dir: Path, reports_dir: Path, sample_prefix: str) -> None:
    """清理本脚本生成的旧 route2 样本文件。"""
    for directory, patterns in [
        (step_dir, [f"{sample_prefix}_*_route2.step"]),
        (pkl_dir, [f"{sample_prefix}_*.pkl"]),
        (samples_dir, ["route2_sampled_s_priors.jsonl", "route2_generated_steps.jsonl"]),
        (reports_dir, ["route2_stepgen_results.csv", "route2_stepgen_report.txt"]),
    ]:
        directory = Path(directory)
        if not directory.exists():
            continue
        for pattern in patterns:
            for path in directory.glob(pattern):
                if path.is_file():
                    path.unlink()


def main() -> None:
    args = get_args()
    device = torch.device(args.device)
    workdir = workdir_from_file()
    dataset_dir = Path(args.dataset_dir)
    outputs_dir = ensure_dir(workdir / "outputs")
    samples_dir = ensure_dir(outputs_dir / "samples")
    step_dir = ensure_dir(outputs_dir / "steps_route2")
    reports_dir = ensure_dir(outputs_dir / "reports")
    pkl_dir = ensure_dir(outputs_dir / "route2_pkl")
    if args.clean_output:
        _clean_previous_outputs(
            step_dir=step_dir,
            pkl_dir=pkl_dir,
            samples_dir=samples_dir,
            reports_dir=reports_dir,
            sample_prefix=args.sample_prefix,
        )

    adapter = _load_adapter(Path(args.adapter_ckpt), device)
    sampled_s_path = samples_dir / "route2_sampled_s_priors.jsonl"
    target_success = max(int(args.target_success), 0)
    sample_attempts = int(args.num_samples)
    if target_success > 0:
        sample_attempts = int(args.max_trials) if int(args.max_trials) > 0 else max(target_success * 4, target_success)
    sampler = EmpiricalSPriorSampler(
        prior_jsonl=dataset_dir / "motif_prior_index_ready.jsonl",
        split_uids_path=dataset_dir / "splits" / "train_uids.txt",
        seed=args.seed,
        min_faces=args.min_sample_faces,
        max_faces=args.max_sample_faces if args.max_sample_faces > 0 else None,
        max_edges=args.max_sample_edges if args.max_sample_edges > 0 else None,
    )
    sampled_pool = sampler.sample(sample_attempts)

    generator = FrozenDeepCADDTGGenerator(
        repo_root=repo_root_from_file(),
        output_dir=outputs_dir / "route2_internal",
        checkpoints_dir=Path(args.checkpoints_dir),
        device=args.device,
        disable_point_condition=True,
    )
    generator.load_models()

    rows: List[Dict[str, Any]] = []
    attempted_s: List[Dict[str, Any]] = []
    success = 0
    for idx, s_prior in enumerate(sampled_pool):
        if target_success > 0 and success >= target_success:
            break
        attempted_s.append(s_prior)
        sample_id = f"{args.sample_prefix}_{idx:05d}"
        s_vec = torch.from_numpy(vectorize_s_prior(s_prior)).float().unsqueeze(0).to(device)
        face_features = torch.from_numpy(vectorize_s_faces(s_prior, args.max_faces)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            pred = adapter(s_vec, face_features)
            bbox = sort_minmax_bbox_tensor(pred["bbox"])[0]
            adj_logits = pred["adj_logits"][0].detach().cpu().numpy()
            pred_count = int(torch.argmax(pred["count_logits"][0]).item())
        nf = pred_count if args.use_pred_count else int(s_prior.get("num_faces", pred_count))
        nf = max(2, min(int(nf), args.max_faces))
        face_bbox = bbox[:nf].clamp(-3.5, 3.5)
        face_bbox, bbox_anchor_info = _refine_bbox_with_s_anchor(
            face_bbox,
            s_prior=s_prior,
            blend=args.bbox_anchor_blend,
            max_shift=args.bbox_anchor_max_shift,
            bbox_scaled=float(getattr(generator.args, "bbox_scaled", 3.0)),
        )
        face_bbox = face_bbox.clamp(-3.5, 3.5)
        edge_face_np = _adjacency_to_edgeface(
            adj_logits,
            nf,
            int(s_prior.get("num_edges", 0) or 0),
            args.max_edge_per_face,
            min_face_degree=args.min_face_degree,
        )
        result: Dict[str, Any] = {}
        build_attempts = max(int(args.build_retries), 1)
        for attempt_idx in range(build_attempts):
            result = _build_one_step(
                generator,
                sample_id,
                face_bbox,
                edge_face_np,
                step_dir,
            )
            result["build_attempts"] = attempt_idx + 1
            if result.get("build_success"):
                break
        result.update(
            {
                "sample_source_uid": s_prior.get("sample_source_uid", ""),
                "s_num_faces": int(s_prior.get("num_faces", 0)),
                "s_num_edges": int(s_prior.get("num_edges", 0)),
                "adapter_pred_count": pred_count,
                "used_face_count": nf,
                **bbox_anchor_info,
            }
        )
        save_pickle(pkl_dir / f"{sample_id}.pkl", result)
        rows.append(
            {
                "sample_id": result.get("sample_id", sample_id),
                "sample_source_uid": result.get("sample_source_uid", ""),
                "build_success": bool(result.get("build_success", False)),
                "reason": result.get("reason", ""),
                "step_path": result.get("step_path", ""),
                "s_num_faces": result.get("s_num_faces", 0),
                "s_num_edges": result.get("s_num_edges", 0),
                "adapter_pred_count": result.get("adapter_pred_count", 0),
                "used_face_count": result.get("used_face_count", 0),
                "build_attempts": result.get("build_attempts", 0),
                "bbox_anchor_used_faces": result.get("bbox_anchor_used_faces", 0),
                "bbox_anchor_mean_confidence": result.get("bbox_anchor_mean_confidence", 0.0),
                "face_count": result.get("face_count", 0),
                "edge_count": result.get("edge_count", 0),
                "vertex_count": result.get("vertex_count", 0),
                "pkl_path": str(pkl_dir / f"{sample_id}.pkl"),
            }
        )
        print(f"{sample_id}: build_success={result.get('build_success')} step={result.get('step_path', '')}")
        if result.get("build_success"):
            success += 1

    write_jsonl(sampled_s_path, attempted_s)
    write_jsonl(samples_dir / "route2_generated_steps.jsonl", rows)
    write_csv(reports_dir / "route2_stepgen_results.csv", rows)
    success = sum(1 for row in rows if row.get("build_success"))
    report = [
        "路线2：S-conditioned adapter 无条件 STEP 生成报告",
        "=" * 72,
        f"尝试 S* 数量：{len(attempted_s)}",
        f"STEP 构造成功数：{success}",
        f"STEP 构造成功率：{success / max(len(attempted_s), 1):.6f}",
        f"目标成功数：{target_success}",
        f"S* face 过滤：[{args.min_sample_faces}, {args.max_sample_faces if args.max_sample_faces > 0 else '不限'}]",
        f"S* edge 上限：{args.max_sample_edges if args.max_sample_edges > 0 else '不限'}",
        f"输出前缀：{args.sample_prefix}",
        f"bbox 锚点修正强度：{args.bbox_anchor_blend}",
        f"每个 S* 最大构造尝试：{max(int(args.build_retries), 1)}",
        "",
        "路线定义：",
        "  S* -> adapter(face bbox + face-edge topology) -> frozen DTG downstream geometry -> STEP",
        "",
        "注意：",
        "  - 当前流程直接使用 S-conditioned adapter 生成宏观结构。",
        "  - 需要先运行 train_route2_adapters.py 得到 adapter checkpoint。",
    ]
    report_path = reports_dir / "route2_stepgen_report.txt"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
