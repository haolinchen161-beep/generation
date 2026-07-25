"""Data preprocessing and feature extraction for PriorFaceBbox (v6.1 Final).

Implements Core Directives:
- Strict Input Schema Assertions (motif_instance: [B,F,3], motif_confidence: [B,F,3], pair_relations: [B,F,F,>=2]).
- O(1) Hash Map Node Lookup using dict.
- Returns diagnostic count tensors (node_count, sheet_node_count, hole_node_count, repeat_node_count, has_structural_prior).
"""

from __future__ import annotations

import warnings
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple

MAX_MOTIF_NODES = 96
PAD_ROLE = -2
BACKGROUND_ROLE = -1
SHEET_ROLE = 0
HOLE_ROLE = 1
REPEAT_ROLE = 2


def extract_global_prior_features(prior_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Extract sample-level global prior feature vector (14D)."""
    device = prior_dict["face_mask"].device
    batch_size = prior_dict["face_mask"].shape[0]
    face_mask = prior_dict["face_mask"].bool()  # [b, max_faces]

    # 1. Surface Type Histogram (6 categories)
    stype = prior_dict["surface_type"].long().clamp(0, 5)  # [b, max_faces]
    stype_oh = F.one_hot(stype, num_classes=6).float() * face_mask.unsqueeze(-1).float()
    stype_dist = stype_oh.sum(dim=1) / (face_mask.sum(dim=1, keepdim=True).clamp_min(1.0))

    # 2. Mean Surface Confidence
    sconf = prior_dict["surface_confidence"].float() * face_mask.float()
    mean_sconf = sconf.sum(dim=1, keepdim=True) / (face_mask.sum(dim=1, keepdim=True).clamp_min(1.0))

    # 3. Motif Membership Counts
    mmem = prior_dict["motif_membership"].float() * face_mask.unsqueeze(-1).float()  # [b, max_faces, 3]
    motif_counts = mmem.sum(dim=1) / 10.0

    # 4. Pair Relations
    pair_rel = prior_dict["pair_relations"].float()  # [b, max_faces, max_faces, R]
    if pair_rel.shape[-1] >= 1:
        hosted_count = pair_rel[:, :, :, 0].sum(dim=(1, 2)).reshape(batch_size, 1) / 10.0
    else:
        hosted_count = torch.zeros((batch_size, 1), device=device)

    if pair_rel.shape[-1] >= 2:
        thin_wall_count = pair_rel[:, :, :, 1].sum(dim=(1, 2)).reshape(batch_size, 1) / 10.0
    else:
        thin_wall_count = torch.zeros((batch_size, 1), device=device)

    # 5. Valid Face Ratio & Mean Degree
    face_ratio = face_mask.float().sum(dim=1, keepdim=True) / 30.0
    fdegree = prior_dict["face_edge_cont"].float() if "face_edge_cont" in prior_dict else torch.zeros_like(sconf)
    while fdegree.ndim > 2:
        fdegree = fdegree.squeeze(-1) if fdegree.shape[-1] == 1 else fdegree.mean(dim=-1)
    mean_degree = (fdegree * face_mask.float()).sum(dim=1, keepdim=True) / (
        face_mask.sum(dim=1, keepdim=True).clamp_min(1.0) * 10.0
    )

    global_prior = torch.cat(
        [stype_dist, mean_sconf, motif_counts, hosted_count, thin_wall_count, face_ratio, mean_degree],
        dim=-1,
    )
    return global_prior


def extract_motif_node_graph(
    prior_dict: Dict[str, torch.Tensor],
    max_nodes: int = MAX_MOTIF_NODES,
) -> Dict[str, torch.Tensor]:
    """Construct Motif Graph Nodes with O(1) Dict Lookup & Strict Schema Assertions."""
    device = prior_dict["face_mask"].device
    batch_size, max_faces = prior_dict["face_mask"].shape

    face_mask = prior_dict["face_mask"].bool()  # [b, max_faces]
    stype = prior_dict["surface_type"].long().clamp(0, 5)  # [b, max_faces]
    stype_oh = F.one_hot(stype, num_classes=6).float() * face_mask.unsqueeze(-1).float()  # [b, max_faces, 6]

    # Strict Schema Assertions (Directive 1)
    minst = prior_dict["motif_instance"]
    if minst.ndim != 3 or minst.shape[-1] != 3:
        raise ValueError(f"motif_instance must have shape [B, F, 3], got {tuple(minst.shape)}")

    if "motif_confidence" in prior_dict:
        mconf = prior_dict["motif_confidence"].float()
        if mconf.ndim != 3 or mconf.shape[-1] != 3:
            raise ValueError(f"motif_confidence must have shape [B, F, 3], got {tuple(mconf.shape)}")
    else:
        mconf = prior_dict["surface_confidence"].float().unsqueeze(-1).repeat(1, 1, 3)

    pair_rel = prior_dict["pair_relations"].float()  # [b, max_faces, max_faces, R]
    if pair_rel.ndim != 4 or pair_rel.shape[-1] < 2:
        raise ValueError(f"pair_relations must have shape [B, F, F, >=2], got {tuple(pair_rel.shape)}")

    node_features_list = []
    node_mask_list = []
    node_roles_list = []
    assignment_list = []
    hosted_adj_list = []
    thin_wall_adj_list = []

    node_count_list = []
    sheet_node_count_list = []
    hole_node_count_list = []
    repeat_node_count_list = []
    has_structural_prior_list = []

    for b in range(batch_size):
        # Node 0 is reserved for background node (role = BACKGROUND_ROLE = -1)
        node_dicts = [{"key": "background", "role": BACKGROUND_ROLE, "faces": []}]
        node_lookup = {"background": 0}  # O(1) Dict Lookup (Directive 2)

        valid_faces = torch.where(face_mask[b])[0].tolist()

        for f_idx in valid_faces:
            for ch in range(3):
                inst_val = int(minst[b, f_idx, ch].item())
                if inst_val > 0:
                    group_key = (ch, inst_val)
                    if group_key not in node_lookup:
                        if len(node_dicts) >= max_nodes:
                            raise RuntimeError(f"motif node overflow: required more than {max_nodes} nodes for sample {b}!")
                        node_idx = len(node_dicts)
                        node_lookup[group_key] = node_idx
                        node_dicts.append({"key": group_key, "role": ch, "faces": [f_idx]})
                    else:
                        node_idx = node_lookup[group_key]
                        node_dicts[node_idx]["faces"].append(f_idx)

        num_nodes = len(node_dicts)

        nf = torch.zeros((max_nodes, 11), device=device)
        nm = torch.zeros((max_nodes,), dtype=torch.bool, device=device)
        nr = torch.full((max_nodes,), PAD_ROLE, dtype=torch.long, device=device)
        asgn = torch.zeros((max_faces, max_nodes), device=device)
        ha = torch.zeros((max_nodes, max_nodes), device=device)
        ta = torch.zeros((max_nodes, max_nodes), device=device)

        # Background node 0 is ALWAYS valid and has BACKGROUND_ROLE = -1
        nm[0] = True
        nr[0] = BACKGROUND_ROLE
        nf[0, :] = 0.0

        sheet_node_indices = []
        sheet_cnt = 0
        hole_cnt = 0
        repeat_cnt = 0

        for n_idx in range(1, num_nodes):
            nd = node_dicts[n_idx]
            f_list = nd["faces"]
            nm[n_idx] = True
            nr[n_idx] = nd["role"]

            if nd["role"] == SHEET_ROLE:
                sheet_node_indices.append(n_idx)
                sheet_cnt += 1
            elif nd["role"] == HOLE_ROLE:
                hole_cnt += 1
            elif nd["role"] == REPEAT_ROLE:
                repeat_cnt += 1

            for f in f_list:
                asgn[f, n_idx] = 1.0

            # Node features (11D)
            role_oh = F.one_hot(torch.tensor(nd["role"], device=device), num_classes=3).float()
            st_mean = stype_oh[b, f_list].mean(dim=0)
            mc_val = mconf[b, f_list, nd["role"]].mean(dim=0, keepdim=True)
            sz_ratio = torch.tensor([len(f_list) / 30.0], device=device)
            nf[n_idx] = torch.cat([role_oh, st_mean, mc_val, sz_ratio], dim=-1)

        # Background Host Target definition
        for f in valid_faces:
            has_sheet = any(asgn[f, sheet_id] > 0 for sheet_id in sheet_node_indices)
            if not has_sheet:
                asgn[f, 0] = 1.0

        # Pair relations with TYPE CONSTRAINTS
        rel_h = pair_rel[b, :, :, 0]
        rel_t = pair_rel[b, :, :, 1]

        for n1 in range(1, num_nodes):
            r1 = node_dicts[n1]["role"]
            f1_list = node_dicts[n1]["faces"]
            for n2 in range(1, num_nodes):
                if n1 == n2:
                    continue
                r2 = node_dicts[n2]["role"]
                f2_list = node_dicts[n2]["faces"]

                if (r1 in (HOLE_ROLE, REPEAT_ROLE)) and (r2 == SHEET_ROLE):
                    h_sum = rel_h[f1_list][:, f2_list].sum().item()
                    if h_sum > 0:
                        ha[n1, n2] = float(h_sum)

                if (r1 == SHEET_ROLE) and (r2 == SHEET_ROLE):
                    t_sum = rel_t[f1_list][:, f2_list].sum().item()
                    if t_sum > 0:
                        ta[n1, n2] = float(t_sum)

        # Row normalization
        ha_denom = ha.sum(dim=-1, keepdim=True).clamp_min(1.0)
        ta_denom = ta.sum(dim=-1, keepdim=True).clamp_min(1.0)
        ha = ha / ha_denom
        ta = ta / ta_denom

        node_features_list.append(nf)
        node_mask_list.append(nm)
        node_roles_list.append(nr)
        assignment_list.append(asgn)
        hosted_adj_list.append(ha)
        thin_wall_adj_list.append(ta)

        node_count_list.append(num_nodes)
        sheet_node_count_list.append(sheet_cnt)
        hole_node_count_list.append(hole_cnt)
        repeat_node_count_list.append(repeat_cnt)
        has_structural_prior_list.append(num_nodes > 1)

    return {
        "node_features": torch.stack(node_features_list, dim=0),
        "node_mask": torch.stack(node_mask_list, dim=0),
        "node_roles": torch.stack(node_roles_list, dim=0),
        "assignment_target": torch.stack(assignment_list, dim=0),
        "hosted_adj": torch.stack(hosted_adj_list, dim=0),
        "thin_wall_adj": torch.stack(thin_wall_adj_list, dim=0),
        "node_count": torch.tensor(node_count_list, device=device, dtype=torch.long),
        "sheet_node_count": torch.tensor(sheet_node_count_list, device=device, dtype=torch.long),
        "hole_node_count": torch.tensor(hole_node_count_list, device=device, dtype=torch.long),
        "repeat_node_count": torch.tensor(repeat_node_count_list, device=device, dtype=torch.long),
        "has_structural_prior": torch.tensor(has_structural_prior_list, device=device, dtype=torch.bool),
    }
