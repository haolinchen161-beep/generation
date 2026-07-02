# -*- coding: utf-8 -*-
"""
程序名称：dataset.py
程序功能：
    本程序定义了用于训练 StructPriorBRepCVAE 深度生成模型的数据集读取器 (BRepDataset)。
    程序会读取扁平化数据集中的 JSON 构型参数与 PKL 几何拓扑大张量，
    将其对齐、归一化并执行统一填充 (Padding)，输出 PyTorch 批次张量。
    本数据集处理的是用于多模态监督学习的高维张量流，支持训练集、验证集与测试集划分。

主要模块功能：
    1. BRepDataset: PyTorch Dataset 实现，负责读取 uid.json 与 uid.pkl，对齐构型与 B-Rep。
    2. collate_fn: 自定义批次拼接函数，将变长的面、边数通过 Padding 统一对齐到最大维度。
       - 最大面数 max_num_faces = 30
       - 最大边数 max_num_edges = 68
       - 最大顶点数 max_num_vertices = 40
       - 最大构型节点数 max_num_config_nodes = 9
       - 最大构型关系数 max_num_config_relations = 18

使用方法：
    由 run_innovation2.py 实例化并加载为 DataLoader。
"""

import os
import csv
import json
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

class BRepDataset(Dataset):
    """
    航空复材构件图及 B-Rep 几何拓扑联合数据集。
    支持读取 JSON 参数与 PKL 拓扑，对齐几何面布局与节点对齐映射。
    """
    def __init__(self, dataset_dir, parsed_dir, split_name="train", max_samples=None):
        self.dataset_dir = dataset_dir
        self.parsed_dir = parsed_dir
        self.split_name = split_name

        # 1. 载入词表与基本模式定义 (基于第一创新点沉淀的 Schema)
        schema_path = os.path.join(parsed_dir, "tensor_schema.json").replace('\\', '/')
        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"Schema file not found at {schema_path}")
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)

        self.part_type_vocab = schema["part_type_vocab"]
        self.node_type_vocab = schema["node_type_vocab"]
        self.relation_type_vocab = schema["relation_type_vocab"]
        self.parameter_keys = schema["parameter_keys"]

        # 2. 读取数据划分
        splits_path = os.path.join(parsed_dir, "data_splits.csv").replace('\\', '/')
        if not os.path.exists(splits_path):
            raise FileNotFoundError(f"Splits file not found at {splits_path}")
        
        self.uids = []
        with open(splits_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split_name:
                    self.uids.append(row["uid"])

        if max_samples is not None:
            self.uids = self.uids[:max_samples]

        # 3. 预加载面组对齐映射表，方便将 face_group 映射为面节点分配标签
        face_group_path = os.path.join(parsed_dir, "face_group_index.jsonl").replace('\\', '/')
        self.face_groups_map = {}
        if os.path.exists(face_group_path):
            with open(face_group_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        self.face_groups_map[item["uid"]] = item["face_groups"]

        # 4. 执行内存预加载，避免 Epoch 反复读取磁盘导致的 IO 阻塞
        self.samples = []
        for uid in self.uids:
            json_path = os.path.join(self.dataset_dir, f"{uid}.json").replace('\\', '/')
            pkl_path = os.path.join(self.parsed_dir, f"{uid}.pkl").replace('\\', '/')

            if not os.path.exists(json_path) or not os.path.exists(pkl_path):
                continue

            with open(json_path, 'r', encoding='utf-8') as f:
                jd = json.load(f)
            with open(pkl_path, 'rb') as f:
                pd = pickle.load(f)

            # A. 提取构型先验与类别
            part_type = jd["part_type"]
            part_type_id = self.part_type_vocab.get(part_type, 0)
            
            param_vals = jd.get("parameters", {})
            param_vec = [float(param_vals.get(k, 0.0)) for k in self.parameter_keys]
            param_vec_scaled = []
            for k, val in zip(self.parameter_keys, param_vec):
                if k in ["length", "width"]:
                    param_vec_scaled.append(val / 200.0)
                elif k in ["height", "flange_width", "rib_width", "rib_height"]:
                    param_vec_scaled.append(val / 50.0)
                elif k in ["thickness", "fillet_radius"]:
                    param_vec_scaled.append(val / 3.0)
                else:
                    param_vec_scaled.append(val)

            cg = jd.get("configuration_graph", {})
            nodes = cg.get("nodes", [])
            relations = cg.get("relations", [])

            # 解析构型节点与类型 (最大 Padding 到 9 个)
            node_ids_map = {}
            config_node_type_ids = []
            config_node_valid = []
            for i, n in enumerate(nodes[:9]):
                node_ids_map[n["id"]] = i
                config_node_type_ids.append(self.node_type_vocab.get(n["type"], 0))
                config_node_valid.append(1.0)
            while len(config_node_type_ids) < 9:
                config_node_type_ids.append(0)
                config_node_valid.append(0.0)

            # 解析关系 (最大 Padding 到 18 个)
            config_relation_src = []
            config_relation_dst = []
            config_relation_type_ids = []
            config_relation_valid = []

            for r in relations[:18]:
                src_name = r["source"]
                dst_name = r["target"]
                src_idx = node_ids_map.get(src_name, 0)
                dst_idx = node_ids_map.get(dst_name, 0)
                r_type = self.relation_type_vocab.get(r["type"], 0)

                config_relation_src.append(src_idx)
                config_relation_dst.append(dst_idx)
                config_relation_type_ids.append(r_type)
                config_relation_valid.append(1.0)
            while len(config_relation_type_ids) < 18:
                config_relation_src.append(0)
                config_relation_dst.append(0)
                config_relation_type_ids.append(0)
                config_relation_valid.append(0.0)

            # B. 提取低层 B-Rep 几何面/边/顶点
            face_bboxes = pd.get("face_bbox_wcs", np.zeros((0, 6)))
            num_faces = min(face_bboxes.shape[0], 30)

            face_bbox_norm = np.zeros((30, 6), dtype=np.float32)
            face_valid_mask = np.zeros(30, dtype=np.float32)
            face_role_label = np.zeros(30, dtype=np.int64)
            face_node_assignment_label = np.ones(30, dtype=np.int64) * 9

            # 解析面组角色与节点分配映射
            fg_list = self.face_groups_map.get(uid, [])
            face_to_node_map = {}
            face_to_role_map = {}
            for fg in fg_list:
                node_name = fg["config_node"]
                n_type = fg["node_type"]
                n_idx = node_ids_map.get(node_name, 9)
                role_idx = self.node_type_vocab.get(n_type, 6)
                for fid in fg["face_ids"]:
                    face_to_node_map[fid] = min(n_idx, 9)
                    face_to_role_map[fid] = role_idx

            # 填充面几何包络
            for f_id in range(num_faces):
                bbox = face_bboxes[f_id]
                cx = (bbox[0] + bbox[1]) / 2.0
                cy = (bbox[2] + bbox[3]) / 2.0
                cz = (bbox[4] + bbox[5]) / 2.0
                dx = bbox[1] - bbox[0]
                dy = bbox[3] - bbox[2]
                dz = bbox[5] - bbox[4]
                face_bbox_norm[f_id] = [cx, cy, cz, dx, dy, dz]
                face_valid_mask[f_id] = 1.0
                face_role_label[f_id] = face_to_role_map.get(f_id, 6)
                face_node_assignment_label[f_id] = face_to_node_map.get(f_id, 9)

            # 填充边/顶点/邻接拓扑
            edge_bboxes = pd.get("edge_bbox_wcs", np.zeros((0, 6)))
            num_edges = min(edge_bboxes.shape[0], 68)
            edge_valid_mask = np.zeros(68, dtype=np.float32)
            for e_id in range(num_edges):
                edge_valid_mask[e_id] = 1.0

            vert_wcs = pd.get("vert_wcs", np.zeros((0, 3)))
            num_verts = min(vert_wcs.shape[0], 40)
            vertex_valid_mask = np.zeros(40, dtype=np.float32)
            vert_wcs_norm = np.zeros((40, 3), dtype=np.float32)
            for v_id in range(num_verts):
                vertex_valid_mask[v_id] = 1.0
                vert_wcs_norm[v_id] = vert_wcs[v_id]

            # 边面/边点邻接拓扑填充
            edgeFace_adj_raw = pd.get("edgeFace_adj", np.zeros((0, 2)))
            edgeVert_adj_raw = pd.get("edgeVert_adj", np.zeros((0, 2)))
            
            edgeFace_adj = np.zeros((68, 2), dtype=np.int64)
            edgeVert_adj = np.zeros((68, 2), dtype=np.int64)

            for e_id in range(num_edges):
                if e_id < edgeFace_adj_raw.shape[0]:
                    edgeFace_adj[e_id] = [min(int(x), 29) for x in edgeFace_adj_raw[e_id]]
                if e_id < edgeVert_adj_raw.shape[0]:
                    edgeVert_adj[e_id] = [min(int(x), 39) for x in edgeVert_adj_raw[e_id]]

            # C. 采样点云监督提取
            face_wcs = pd.get("face_wcs", np.zeros((num_faces, 32, 32, 3)))
            edge_wcs = pd.get("edge_wcs", np.zeros((num_edges, 32, 3)))

            face_wcs_norm = np.zeros((30, 32, 32, 3), dtype=np.float32)
            edge_wcs_norm = np.zeros((68, 32, 3), dtype=np.float32)

            for f_id in range(num_faces):
                face_wcs_norm[f_id] = face_wcs[f_id]
            for e_id in range(num_edges):
                edge_wcs_norm[e_id] = edge_wcs[e_id]

            self.samples.append({
                "uid": uid,
                "part_type_id": torch.tensor(part_type_id, dtype=torch.long),
                "parameter_vector": torch.tensor(param_vec_scaled, dtype=torch.float32),
                "config_node_type_ids": torch.tensor(config_node_type_ids, dtype=torch.long),
                "config_node_valid": torch.tensor(config_node_valid, dtype=torch.float32),
                "config_relation_src": torch.tensor(config_relation_src, dtype=torch.long),
                "config_relation_dst": torch.tensor(config_relation_dst, dtype=torch.long),
                "config_relation_type_ids": torch.tensor(config_relation_type_ids, dtype=torch.long),
                "config_relation_valid": torch.tensor(config_relation_valid, dtype=torch.float32),
                "face_bbox_norm": torch.tensor(face_bbox_norm, dtype=torch.float32),
                "face_valid_mask": torch.tensor(face_valid_mask, dtype=torch.float32),
                "face_role_label": torch.tensor(face_role_label, dtype=torch.long),
                "face_node_assignment_label": torch.tensor(face_node_assignment_label, dtype=torch.long),
                "edge_valid_mask": torch.tensor(edge_valid_mask, dtype=torch.float32),
                "vertex_valid_mask": torch.tensor(vertex_valid_mask, dtype=torch.float32),
                "edgeFace_adj": torch.tensor(edgeFace_adj, dtype=torch.long),
                "edgeVert_adj": torch.tensor(edgeVert_adj, dtype=torch.long),
                "vert_wcs_norm": torch.tensor(vert_wcs_norm, dtype=torch.float32),
                "edge_wcs_norm": torch.tensor(edge_wcs_norm, dtype=torch.float32),
                "face_wcs_norm": torch.tensor(face_wcs_norm, dtype=torch.float32)
            })

        print(f" -> 加载子集 [{split_name}] 完毕，预加载样本数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    """
    自定义批次整理函数，合并字典中所有张量。
    """
    keys = batch[0].keys()
    collated = {}
    for k in keys:
        if k == "uid":
            collated[k] = [x[k] for x in batch]
        else:
            collated[k] = torch.stack([x[k] for x in batch], dim=0)
    return collated
