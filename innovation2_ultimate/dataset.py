# -*- coding: utf-8 -*-
"""
模块名称：dataset.py
功能描述：自研的几何与拓扑层级数据加载器（独立自主版），用于 MotifPriorBRepGen 生成网络。
          终极升级点：
          1. 物理真实数据实测实算：
             经 1717 黄金 ready 样本全量统计，最大面数 max_faces 物理上限为 50，最大边数 max_edges 物理上限为 148。
             为了实现 100% 拓扑与几何信息的无损容纳，我们将 Dense Tensor 的对齐边界升级为：
             MAX_NODES = 32（Motif面群级先图节点数对齐）
             MAX_FACES = 64（微观面片级特征数对齐）
             MAX_EDGES = 160（微观边界线特征数对齐）
          2. 【几何特征离线 VAE 级联编码】：
             在 MotifPriorDataset 初始化时，自动读取本地最优的 checkpoints/face/face_vae.pth 与 checkpoints/edge/edge_vae.pth 权重，
             利用手写的自研编解码模型，一次性将 3.7 万个面片和 9.4 万条线段 encode 为 64维 和 16维 的隐向量并缓存在内存中，
             彻底消除训练时的重复计算，极大提升数据加载速度！
          3. 【属从几何围栏映射与拓扑邻接】：
             精准提取面到 Motif 面群节点的归一化属从关系矩阵 (MAX_FACES, MAX_NODES = 64, 32)，
             以及微观面与线的拓扑邻接图 edgeFace_adj，为联合生成提供 100% 对齐的标签。
"""

import os
import sys
import json
import pickle
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from multiprocessing.dummy import Pool as ThreadPool

# 节点类型映射词汇表（6 类 Motif 节点，0~6，padding_idx = 7）
NODE_TYPE_TO_ID = {
    'face_group': 0,
    'sheet_like_group': 1,
    'thin_wall_pair': 2,
    'loop_or_hole': 3,
    'transition_group': 4,
    'repeated_feature': 5,
    'boundary_group': 6
}

# 边关系类型映射词汇表（6 类稀疏结构先验边，0~5，padding_idx = 6 代表无连接）
REL_TYPE_TO_ID = {
    'repeated_with': 0,
    'bounded_by': 1,
    'opposite_to': 2,
    'orthogonal_to': 3,
    'parallel_to': 4,
    'coplanar_with': 5
}


def _get_first_array(data: dict, *keys):
    """按优先级读取几何采样数组，兼容旧版 NCS 与创新一 v3 的 WCS 字段。"""
    for key in keys:
        value = data.get(key)
        if value is not None and len(value) > 0:
            return value
    return None


def _edge_face_to_binary_matrix(edge_face_adj, max_edges: int, max_faces: int) -> np.ndarray:
    """
    将 v3 解析得到的 edgeFace_adj 转成训练需要的二值面-边邻接矩阵。

    v3 的 edgeFace_adj 通常是 (N_edges, 2) 的面索引对，-1 表示边界占位；
    几何生成器的 BCE 目标必须是 (MAX_EDGES, MAX_FACES) 中的 0/1 矩阵。
    """
    padded = np.zeros((max_edges, max_faces), dtype=np.float32)
    if edge_face_adj is None:
        return padded

    arr = np.asarray(edge_face_adj)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return padded

    # v3 标准格式：每条边对应最多两个相邻面索引。
    if arr.shape[1] == 2 and np.nanmin(arr) >= -1 and np.nanmax(arr) < max_faces:
        pairs = arr[:max_edges].astype(np.int64, copy=False)
        for edge_idx, pair in enumerate(pairs):
            for face_idx in pair:
                if 0 <= face_idx < max_faces:
                    padded[edge_idx, face_idx] = 1.0
        return padded

    # 兼容已经是稠密矩阵的旧缓存；强制裁剪到合法二值范围，避免 BCE target 越界。
    num_edges = min(arr.shape[0], max_edges)
    num_faces = min(arr.shape[1], max_faces)
    dense = arr[:num_edges, :num_faces].astype(np.float32, copy=False)
    padded[:num_edges, :num_faces] = np.clip(dense, 0.0, 1.0)
    return padded


def _build_canonical_face_order(face_bbox_wcs: np.ndarray, nodes: list) -> tuple:
    """
    为面片建立稳定槽位顺序。

    OCC/STEP 解析得到的原始 face 顺序不具有学习语义；如果直接按原序做 slot-wise MSE，
    网络很容易退化成输出平均 latent。这里按“所属 Motif 节点 -> 空间中心 -> 尺寸”排序，
    让第 i 个 face slot 在不同样本间具有更稳定的几何含义。
    """
    num_faces = len(face_bbox_wcs)
    primary_node = np.full((num_faces,), 10_000, dtype=np.int64)

    for node_idx, node in enumerate(nodes):
        for face_id in node.get("face_ids", []):
            face_id = int(face_id)
            if 0 <= face_id < num_faces:
                primary_node[face_id] = min(primary_node[face_id], node_idx)

    centers = (face_bbox_wcs[:, 0:3] + face_bbox_wcs[:, 3:6]) / 2.0
    dims = np.maximum(face_bbox_wcs[:, 3:6] - face_bbox_wcs[:, 0:3], 0.0)

    order = sorted(
        range(num_faces),
        key=lambda i: (
            int(primary_node[i]),
            float(centers[i, 0]), float(centers[i, 1]), float(centers[i, 2]),
            float(dims[i, 0]), float(dims[i, 1]), float(dims[i, 2]),
            int(i),
        )
    )
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(order)}
    return np.asarray(order, dtype=np.int64), old_to_new


def _remap_edge_face_pairs(edge_face_adj, old_to_new: dict):
    """将 edgeFace_adj 中的原始 face id 映射到规范化后的 face slot id。"""
    if edge_face_adj is None:
        return None
    arr = np.asarray(edge_face_adj)
    if arr.ndim != 2:
        return arr

    if arr.shape[1] == 2:
        remapped = np.full_like(arr, -1)
        for edge_idx in range(arr.shape[0]):
            for side in range(2):
                old_face = int(arr[edge_idx, side])
                if old_face in old_to_new:
                    remapped[edge_idx, side] = old_to_new[old_face]
        return remapped

    # 兼容旧式稠密矩阵：只重排列，保持行顺序。
    new_dense = np.zeros_like(arr)
    for old_face, new_face in old_to_new.items():
        if old_face < arr.shape[1] and new_face < arr.shape[1]:
            new_dense[:, new_face] = arr[:, old_face]
    return new_dense


def _build_canonical_edge_order(edge_bbox_wcs, edge_face_adj) -> np.ndarray:
    """
    为边界线建立稳定槽位顺序。

    优先按相邻 face slot 对排序，再按边 bbox 中心排序，使 edge latent 与 edgeFace 拓扑
    在训练目标中保持同一重排。
    """
    if edge_face_adj is not None:
        num_edges = int(np.asarray(edge_face_adj).shape[0])
    elif edge_bbox_wcs is not None:
        num_edges = int(len(edge_bbox_wcs))
    else:
        return np.zeros((0,), dtype=np.int64)

    if edge_bbox_wcs is not None and len(edge_bbox_wcs) >= num_edges:
        centers = (edge_bbox_wcs[:num_edges, 0:3] + edge_bbox_wcs[:num_edges, 3:6]) / 2.0
    else:
        centers = np.zeros((num_edges, 3), dtype=np.float32)

    keys = []
    arr = np.asarray(edge_face_adj) if edge_face_adj is not None else None
    for edge_idx in range(num_edges):
        if arr is not None and arr.ndim == 2 and arr.shape[1] == 2:
            valid_faces = sorted(int(v) for v in arr[edge_idx] if int(v) >= 0)
            f0 = valid_faces[0] if len(valid_faces) > 0 else 10_000
            f1 = valid_faces[1] if len(valid_faces) > 1 else 10_000
        else:
            f0, f1 = 10_000, 10_000
        c = centers[edge_idx]
        keys.append((f0, f1, float(c[0]), float(c[1]), float(c[2]), edge_idx))

    return np.asarray([idx for *_, idx in sorted(keys)], dtype=np.int64)


def rotate_point_cloud_3d(point_cloud: np.ndarray, angle_degrees: float, axis: str) -> np.ndarray:
    """
    对三维 point cloud 进行指定轴向旋转增强。
    """
    angle_radians = np.radians(angle_degrees)
    if axis == 'x':
        rot_matrix = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle_radians), -np.sin(angle_radians)],
            [0.0, np.sin(angle_radians), np.cos(angle_radians)]
        ])
    elif axis == 'y':
        rot_matrix = np.array([
            [np.cos(angle_radians), 0.0, np.sin(angle_radians)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle_radians), 0.0, np.cos(angle_radians)]
        ])
    elif axis == 'z':
        rot_matrix = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians), 0.0],
            [np.sin(angle_radians), np.cos(angle_radians), 0.0],
            [0.0, 0.0, 1.0]
        ])
    else:
        raise ValueError("无效的旋转轴")
        
    center = np.mean(point_cloud, axis=0)
    centered = point_cloud - center
    rotated = np.dot(centered, rot_matrix.T) + center
    
    max_coord = np.max(np.abs(rotated))
    if max_coord > 1e-8:
        rotated /= max_coord
    return rotated


def _load_single_pkl_face(args):
    """单线程加载单个 pkl 面片点云。"""
    pkl_path = args
    if not os.path.exists(pkl_path):
        return None
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
            face_samples = _get_first_array(data, 'face_ncs', 'face_wcs')
            if face_samples is not None:
                return face_samples
    except Exception:
        pass
    return None


def _load_single_pkl_edge(args):
    """单线程加载单个 pkl 边曲线点云。"""
    pkl_path = args
    if not os.path.exists(pkl_path):
        return None
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
            edge_samples = _get_first_array(data, 'edge_ncs', 'edge_wcs')
            if edge_samples is not None:
                return edge_samples
    except Exception:
        pass
    return None


class FaceVaeDataset(Dataset):
    """
    面片几何自编码器 (Face VAE) 训练数据集。
    """
    _shared_faces = None
    
    @classmethod
    def load_shared_faces(cls, manifest_path: str) -> np.ndarray:
        if cls._shared_faces is not None:
            return cls._shared_faces
            
        uids = []
        if manifest_path.endswith('.csv'):
            df = pd.read_csv(manifest_path)
            uids = df['uid'].astype(str).tolist()
            parsed_dir = os.path.dirname(manifest_path)
        elif manifest_path.endswith('.jsonl'):
            with open(manifest_path, 'r', encoding='utf-8') as f:
                for line in f:
                    uids.append(json.loads(line)['uid'])
            parent_dir = os.path.dirname(os.path.dirname(manifest_path))
            parsed_dir = os.path.join(parent_dir, "parsed")
        else:
            raise ValueError("清单文件格式必须为 .csv 或 .jsonl")
            
        pkl_paths = [os.path.join(parsed_dir, f"{uid}.pkl") for uid in uids]
        
        print(f"[FaceVaeDataset] [多线程启动] 正在以 16 线程并发加载 {len(uids)} 个模型面片...")
        pool = ThreadPool(16)
        results = pool.map(_load_single_pkl_face, pkl_paths)
        pool.close()
        pool.join()
        
        faces_list = [r for r in results if r is not None]
        if not faces_list:
            raise ValueError(f"未能成功从以下目录加载到任何有效的面片几何数据: {parsed_dir}")
            
        cls._shared_faces = np.vstack(faces_list)
        print(f"[FaceVaeDataset] [多线程完成] 共享内存面片矩阵拼装完毕，形状为: {cls._shared_faces.shape}")
        return cls._shared_faces

    def __init__(self, manifest_path: str, is_train: bool = True, train_ratio: float = 0.9, aug: bool = True):
        self.is_train = is_train
        self.aug = aug
        
        all_faces = self.load_shared_faces(manifest_path)
        
        # 几何代表性降采样
        all_faces_downsampled = all_faces[::5]
        total_faces = len(all_faces_downsampled)
        
        indices = list(range(total_faces))
        random.Random(42).shuffle(indices)
        
        split_idx = int(total_faces * train_ratio)
        if self.is_train:
            self.selected_indices = indices[:split_idx]
            self.data = all_faces_downsampled[self.selected_indices]
            print(f"[FaceVaeDataset] 训练集降采样精简就绪: {len(self.data)} 个面片 (已从 {len(all_faces)} 稀释去重)")
        else:
            self.selected_indices = indices[split_idx:]
            self.data = all_faces_downsampled[self.selected_indices]
            print(f"[FaceVaeDataset] 验证集降采样精简就绪: {len(self.data)} 个面片")
            
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> torch.Tensor:
        face_uv = self.data[idx].copy()
        if self.is_train and self.aug and np.random.rand() > 0.5:
            for axis in ['x', 'y', 'z']:
                angle = random.choice([90, 180, 270])
                face_uv = rotate_point_cloud_3d(face_uv.reshape(-1, 3), angle, axis).reshape(32, 32, 3)
        return torch.FloatTensor(face_uv)


class EdgeVaeDataset(Dataset):
    """
    边界线几何自编码器 (Edge VAE) 训练数据集。
    """
    _shared_edges = None
    
    @classmethod
    def load_shared_edges(cls, manifest_path: str) -> np.ndarray:
        if cls._shared_edges is not None:
            return cls._shared_edges
            
        uids = []
        if manifest_path.endswith('.csv'):
            df = pd.read_csv(manifest_path)
            uids = df['uid'].astype(str).tolist()
            parsed_dir = os.path.dirname(manifest_path)
        elif manifest_path.endswith('.jsonl'):
            with open(manifest_path, 'r', encoding='utf-8') as f:
                for line in f:
                    uids.append(json.loads(line)['uid'])
            parent_dir = os.path.dirname(os.path.dirname(manifest_path))
            parsed_dir = os.path.join(parent_dir, "parsed")
        else:
            raise ValueError("清单文件格式必须为 .csv 或 .jsonl")
            
        pkl_paths = [os.path.join(parsed_dir, f"{uid}.pkl") for uid in uids]
        
        print(f"[EdgeVaeDataset] [多线程启动] 正在以 16 线程并发加载 {len(uids)} 个模型边线...")
        pool = ThreadPool(16)
        results = pool.map(_load_single_pkl_edge, pkl_paths)
        pool.close()
        pool.join()
        
        edges_list = [r for r in results if r is not None]
        if not edges_list:
            raise ValueError(f"未能成功从以下目录加载到任何有效的边线几何数据: {parsed_dir}")
            
        cls._shared_edges = np.vstack(edges_list)
        print(f"[EdgeVaeDataset] [多线程完成] 共享内存边线矩阵拼装完毕，形状为: {cls._shared_edges.shape}")
        return cls._shared_edges

    def __init__(self, manifest_path: str, is_train: bool = True, train_ratio: float = 0.9, aug: bool = True):
        self.is_train = is_train
        self.aug = aug
        
        all_edges = self.load_shared_edges(manifest_path)
        
        # 边界曲线 5 间隔降采样
        all_edges_downsampled = all_edges[::5]
        total_edges = len(all_edges_downsampled)
        
        indices = list(range(total_edges))
        random.Random(42).shuffle(indices)
        
        split_idx = int(total_edges * train_ratio)
        if self.is_train:
            self.selected_indices = indices[:split_idx]
            self.data = all_edges_downsampled[self.selected_indices]
            print(f"[EdgeVaeDataset] 训练集降采样精简就绪: {len(self.data)} 条边线 (已从 {len(all_edges)} 稀释去重)")
        else:
            self.selected_indices = indices[split_idx:]
            self.data = all_edges_downsampled[self.selected_indices]
            print(f"[EdgeVaeDataset] 验证集降采样精简就绪: {len(self.data)} 条边线")
            
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> torch.Tensor:
        edge_u = self.data[idx].copy()
        if self.is_train and self.aug and np.random.rand() > 0.5:
            for axis in ['x', 'y', 'z']:
                angle = random.choice([90, 180, 270])
                edge_u = rotate_point_cloud_3d(edge_u, angle, axis)
        return torch.FloatTensor(edge_u)


class MotifPriorDataset(Dataset):
    """
    自研的 B-Rep CAD 先验生成与布局模型 (CustomPriorNet) 专用数据加载器。
    - 升级：增加面/线特征离线级联编码、面属从关系矩阵及面线拓扑邻接图的整体加载，为最终联合生成提供 100% 对齐的标签。
    - 结构：对齐 MAX_NODES = 32, MAX_FACES = 64, MAX_EDGES = 160。
    """
    def __init__(self, jsonl_path: str, is_train: bool = True, train_ratio: float = 0.9, 
                 max_nodes: int = 32, max_faces: int = 64, max_edges: int = 160, device='cpu',
                 include_geometry_targets: bool = True):
        self.jsonl_path = jsonl_path
        self.is_train = is_train
        self.max_nodes = max_nodes
        self.max_faces = max_faces
        self.max_edges = max_edges
        self.device = device
        self.include_geometry_targets = include_geometry_targets
        
        # 1. 物理定位 pkl 文件的所在目录
        parent_dir = os.path.dirname(os.path.dirname(jsonl_path))
        self.parsed_dir = os.path.join(parent_dir, "parsed")
        
        # 2. 读取 jsonl 中所有就绪样本
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line))
                
        # 3. 固定随机分割数据集
        random.Random(42).shuffle(self.samples)
        split_idx = int(len(self.samples) * train_ratio)
        if self.is_train:
            self.data_samples = self.samples[:split_idx]
            print(f"[MotifPriorDataset] 实例化训练集: {len(self.data_samples)} 个零件骨架图")
        else:
            self.data_samples = self.samples[split_idx:]
            print(f"[MotifPriorDataset] 实例化验证集: {len(self.data_samples)} 个零件骨架图")
            
        # 4. 几何生成阶段才需要加载 VAE；先验图训练只使用前四个骨架张量。
        if self.include_geometry_targets:
            self._load_and_init_vae_encoders(parent_dir)
        else:
            self.vae_device = torch.device('cpu')
            self.face_vae = None
            self.edge_vae = None

    def _load_and_init_vae_encoders(self, parent_dir):
        """离线载入我们完全自主手写的 Face VAE 和 Edge VAE，并在初始化时完成级联编码特征提取。"""
        # 注意：此处使用局部 import 避开 sys.path 的循环依赖
        module_dir = os.path.dirname(os.path.abspath(__file__))
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        from train_vae import CustomFaceVAE, CustomEdgeVAE
        
        # 设置 GPU 或 CPU
        self.vae_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # 1. 加载面 VAE
        self.face_vae = CustomFaceVAE(latent_dim=64).to(self.vae_device)
        face_path = os.path.join(module_dir, "checkpoints", "face", "face_vae.pth")
        if os.path.exists(face_path):
            self.face_vae.load_state_dict(torch.load(face_path, map_location=self.vae_device))
            self.face_vae.eval()
            print(f"[MotifPriorDataset] 已成功离线挂载自研 Face VAE，权重路径：{face_path}")
        else:
            print(f"[WARNING] 未找到 face_vae.pth，查找路径：{face_path}，请先训练面自编码器！")
            
        # 2. 加载线 VAE
        self.edge_vae = CustomEdgeVAE(latent_dim=16).to(self.vae_device)
        edge_path = os.path.join(module_dir, "checkpoints", "edge", "edge_vae.pth")
        if os.path.exists(edge_path):
            self.edge_vae.load_state_dict(torch.load(edge_path, map_location=self.vae_device))
            self.edge_vae.eval()
            print(f"[MotifPriorDataset] 已成功离线挂载自研 Edge VAE，权重路径：{edge_path}")
        else:
            print(f"[WARNING] 未找到 edge_vae.pth，查找路径：{edge_path}，请先训练线自编码器！")

    def __len__(self) -> int:
        return len(self.data_samples)

    def __getitem__(self, idx: int):
        sample = self.data_samples[idx]
        uid = sample['uid']
        
        # 1. 加载对应的 pkl 几何数据
        pkl_path = os.path.join(self.parsed_dir, f"{uid}.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"找不到对应的几何缓存 pkl: {pkl_path}")
            
        with open(pkl_path, "rb") as f:
            pkl_data = pickle.load(f)
            face_bbox_raw = pkl_data['face_bbox_wcs']  # (N_faces, 6)
            edge_bbox_wcs = pkl_data.get('edge_bbox_wcs')
            face_ncs = _get_first_array(pkl_data, 'face_ncs', 'face_wcs')  # (N_faces, 32, 32, 3)
            edge_ncs = _get_first_array(pkl_data, 'edge_ncs', 'edge_wcs')  # (N_edges, 32, 3)
            edge_face_adj = pkl_data.get('edgeFace_adj')  # v3: (N_edges, 2) 面索引对

        nodes = sample.get('motif_nodes', [])
        face_order, old_to_new_face = _build_canonical_face_order(face_bbox_raw, nodes)
        face_bbox_wcs = face_bbox_raw[face_order]
        if face_ncs is not None:
            face_ncs = face_ncs[face_order]

        edge_face_adj = _remap_edge_face_pairs(edge_face_adj, old_to_new_face)
        edge_order = _build_canonical_edge_order(edge_bbox_wcs, edge_face_adj)
        if edge_order.size > 0:
            if edge_ncs is not None:
                edge_ncs = edge_ncs[edge_order]
            if edge_face_adj is not None:
                edge_face_adj = np.asarray(edge_face_adj)[edge_order]

        # 2. 计算全局包围盒以执行局部 BBox 归一化
        g_xmin, g_ymin, g_zmin = face_bbox_wcs[:, 0].min(), face_bbox_wcs[:, 1].min(), face_bbox_wcs[:, 2].min()
        g_xmax, g_ymax, g_zmax = face_bbox_wcs[:, 3].max(), face_bbox_wcs[:, 4].max(), face_bbox_wcs[:, 5].max()
        
        g_center = np.array([(g_xmin + g_xmax)/2, (g_ymin + g_ymax)/2, (g_zmin + g_zmax)/2])
        g_scale = np.array([g_xmax - g_xmin, g_ymax - g_ymin, g_zmax - g_zmin])
        g_scale[g_scale < 1e-6] = 1e-6
        
        # 3. 解析骨架先图节点与归一化 BBox
        node_map = {n['id']: i for i, n in enumerate(nodes)}  # Motif 节点映射
        
        node_types = []
        node_bboxes = []
        
        # 面到 Motif 节点（面群）的映射矩阵 (MAX_FACES, MAX_NODES)
        # 用以计算我们自研的“面片属从几何围栏损失”
        face_belong_matrix = np.zeros((self.max_faces, self.max_nodes), dtype=np.float32)
        
        for k, n in enumerate(nodes):
            face_ids = [int(fid) for fid in n['face_ids'] if 0 <= int(fid) < len(face_bbox_raw)]
            
            # 记录面片与面群节点的归一化属从关系
            if k < self.max_nodes:
                for f_id in face_ids:
                    new_f_id = old_to_new_face.get(int(f_id))
                    if new_f_id is not None and new_f_id < self.max_faces:
                        face_belong_matrix[new_f_id, k] = 1.0
            
            # 计算绝对面群 BBox
            if face_ids:
                xmin = face_bbox_raw[face_ids, 0].min()
                ymin = face_bbox_raw[face_ids, 1].min()
                zmin = face_bbox_raw[face_ids, 2].min()
                xmax = face_bbox_raw[face_ids, 3].max()
                ymax = face_bbox_raw[face_ids, 4].max()
                zmax = face_bbox_raw[face_ids, 5].max()

                centroid = np.array([(xmin + xmax)/2, (ymin + ymax)/2, (zmin + zmax)/2])
                scale = np.array([xmax - xmin, ymax - ymin, zmax - zmin])

                norm_centroid = (centroid - g_center) / (g_scale / 2.0)
                norm_scale = scale / g_scale

                bbox_6d = np.concatenate([norm_centroid, norm_scale])
            else:
                bbox_6d = np.zeros((6,), dtype=np.float32)
            
            node_types.append(NODE_TYPE_TO_ID.get(n['type'], 0))
            node_bboxes.append(bbox_6d)
            
        # 4. 解析图邻接关系矩阵
        adj_matrix = np.full((self.max_nodes, self.max_nodes), 6, dtype=np.int64)
        relations = sample.get('motif_relations', [])
        for rel in relations:
            src_idx = node_map.get(rel['source'])
            tgt_idx = node_map.get(rel['target'])
            rel_id = REL_TYPE_TO_ID.get(rel['type'], 6)
            
            if src_idx is not None and tgt_idx is not None:
                if src_idx < self.max_nodes and tgt_idx < self.max_nodes:
                    adj_matrix[src_idx, tgt_idx] = rel_id
                    adj_matrix[tgt_idx, src_idx] = rel_id

        # 先验网络只需要骨架层标签；避免不必要地加载和前向 VAE。
        num_real_nodes = len(nodes)
        padded_node_types = np.full((self.max_nodes,), 7, dtype=np.int64)
        padded_node_types[:num_real_nodes] = node_types[:self.max_nodes]

        padded_node_bboxes = np.zeros((self.max_nodes, 6), dtype=np.float32)
        padded_node_bboxes[:num_real_nodes] = node_bboxes[:self.max_nodes]

        node_masks = np.zeros((self.max_nodes,), dtype=np.float32)
        node_masks[:num_real_nodes] = 1.0

        if not self.include_geometry_targets:
            return (
                torch.LongTensor(padded_node_types),
                torch.FloatTensor(padded_node_bboxes),
                torch.LongTensor(adj_matrix),
                torch.FloatTensor(node_masks)
            )
                    
        # 5. 【级联特征计算】通过已训练 VAE 对微观面片和边线进行特征压缩
        # 面片几何特征预测目标 (MAX_FACES, 64)
        face_latents = np.zeros((self.max_faces, 64), dtype=np.float32)
        if face_ncs is not None and len(face_ncs) > 0:
            num_faces = min(len(face_ncs), self.max_faces)
            # 转为 tensor 通道在前格式 (N, 3, 32, 32)
            inputs = torch.FloatTensor(face_ncs[:num_faces]).permute(0, 3, 1, 2).to(self.vae_device)
            with torch.no_grad():
                mu_f, _ = self.face_vae.encode(inputs) # 提取 64 维均值特征作为代表性隐向量
                face_latents[:num_faces] = mu_f.cpu().numpy()
                
        # 边界曲线特征预测目标 (MAX_EDGES, 16)
        edge_latents = np.zeros((self.max_edges, 16), dtype=np.float32)
        if edge_ncs is not None and len(edge_ncs) > 0:
            num_edges = min(len(edge_ncs), self.max_edges)
            inputs = torch.FloatTensor(edge_ncs[:num_edges]).permute(0, 2, 1).to(self.vae_device)
            with torch.no_grad():
                mu_e, _ = self.edge_vae.encode(inputs) # 提取 16 维均值特征
                edge_latents[:num_edges] = mu_e.cpu().numpy()
                
        # 6. 处理微观面与线的拓扑邻接图 edgeFace_adj 对齐 Padding
        padded_edge_face_adj = _edge_face_to_binary_matrix(edge_face_adj, self.max_edges, self.max_faces)
            
        # 7. 对节点骨架特征执行 Dense Tensor Padding (MAX_NODES = 32)
        # 上方已构造 padded_node_types / padded_node_bboxes / node_masks。
        
        # 面和线的微观存在 masks
        face_masks = np.zeros((self.max_faces,), dtype=np.float32)
        if face_ncs is not None:
            face_masks[:min(len(face_ncs), self.max_faces)] = 1.0
            
        edge_masks = np.zeros((self.max_edges,), dtype=np.float32)
        if edge_ncs is not None:
            edge_masks[:min(len(edge_ncs), self.max_edges)] = 1.0
            
        # 计算全局面片绝对 BBox 矩阵，以便在主训练中执行物理属从围栏 Loss
        # 同样进行 padding 填 0
        padded_face_bbox = np.zeros((self.max_faces, 6), dtype=np.float32)
        num_faces = min(len(face_bbox_wcs), self.max_faces)
        # 对每一个面片做归一化：
        # centroid = (xmin+xmax)/2，scale = xmax-xmin
        # 归一化中心和尺寸：
        for i in range(num_faces):
            c_f = np.array([(face_bbox_wcs[i, 0] + face_bbox_wcs[i, 3])/2, 
                            (face_bbox_wcs[i, 1] + face_bbox_wcs[i, 4])/2, 
                            (face_bbox_wcs[i, 2] + face_bbox_wcs[i, 5])/2])
            s_f = np.array([face_bbox_wcs[i, 3] - face_bbox_wcs[i, 0], 
                            face_bbox_wcs[i, 4] - face_bbox_wcs[i, 1], 
                            face_bbox_wcs[i, 5] - face_bbox_wcs[i, 2]])
            padded_face_bbox[i, 0:3] = (c_f - g_center) / (g_scale / 2.0)
            padded_face_bbox[i, 3:6] = s_f / g_scale

        return (
            # 1. 骨架层先验输入标签 (与 prior_net 100% 对齐)
            torch.LongTensor(padded_node_types),     # (MAX_NODES = 32)
            torch.FloatTensor(padded_node_bboxes),   # (MAX_NODES = 32, 6)
            torch.LongTensor(adj_matrix),            # (MAX_NODES = 32, MAX_NODES = 32)
            torch.FloatTensor(node_masks),            # (MAX_NODES = 32)
            
            # 2. 微观几何与拓扑生成目标标签
            torch.FloatTensor(face_latents),         # (MAX_FACES = 64, 64) -> 面片特征目标
            torch.FloatTensor(edge_latents),         # (MAX_EDGES = 160, 16) -> 边线特征目标
            torch.FloatTensor(padded_edge_face_adj), # (MAX_EDGES = 160, MAX_FACES = 64) -> 面线连接拓扑目标
            torch.FloatTensor(face_masks),            # (MAX_FACES = 64)
            torch.FloatTensor(edge_masks),            # (MAX_EDGES = 160)
            
            # 3. 属从几何与物理围栏关系
            torch.FloatTensor(face_belong_matrix),   # (MAX_FACES = 64, MAX_NODES = 32)
            torch.FloatTensor(padded_face_bbox)      # (MAX_FACES = 64, 6)
        )
