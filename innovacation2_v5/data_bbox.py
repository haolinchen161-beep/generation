"""Data preprocessing and feature extraction for PriorFaceBbox.

Refactored extract_motif_node_graph according to Core Directives 1, 2, 3, 5:
- Preserves 3D motif_instance [B, N_faces, 3] (sheet, hole, repeat channels) without argmax!
- Constructs multi-hot face_to_node assignment matrix [B, N_faces, 32] (a face can belong to multiple nodes).
- Includes Node 0 as permanent Background/Null Node (node_mask[:, 0] = True) to prevent NaN in MultiheadAttention.
- Uses motif_confidence for node features.
- Row-normalizes relation adjacencies (hosted_adj, thin_wall_adj).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple


def extract_global_prior_features(prior_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Extract sample-level global prior feature vector (14D) with CORRECTED relation channel indices."""
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

    # 4. Pair Relations: Channel 0 for hosted_by, Channel 1 for thin_wall_pair
    pair_rel = prior_dict["pair_relations"].float()  # [b, max_faces, max_faces, 2 or 3]
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
    max_nodes: int = 32,
) -> Dict[str, torch.Tensor]:
    """Core Directives 1, 2, 3, 5: Construct Motif Graph Nodes with Background Node & Multi-Hot Assignment.

    Node 0: Permanent Background Node (node_mask[:, 0] = True).
    Nodes 1..max_nodes-1: Real Motif Instance Nodes (channel, instance_id).

    Returns:
      node_features:     [B, max_nodes, 11]
      node_mask:         [B, max_nodes] (bool, node 0 always True)
      assignment_target: [B, N_faces, max_nodes] (multi-hot, float)
      hosted_adj:        [B, max_nodes, max_nodes] (row-normalized)
      thin_wall_adj:     [B, max_nodes, max_nodes] (row-normalized)
    """
    device = prior_dict["face_mask"].device
    batch_size, max_faces = prior_dict["face_mask"].shape

    face_mask = prior_dict["face_mask"].bool()  # [b, max_faces]
    stype = prior_dict["surface_type"].long().clamp(0, 5)  # [b, max_faces]
    stype_oh = F.one_hot(stype, num_classes=6).float() * face_mask.unsqueeze(-1).float()  # [b, max_faces, 6]

    # Use motif_confidence if available, fallback to surface_confidence
    if "motif_confidence" in prior_dict:
        mconf = prior_dict["motif_confidence"].float()
        if mconf.ndim == 2:
            mconf = mconf.unsqueeze(-1)
    else:
        mconf = prior_dict["surface_confidence"].float().unsqueeze(-1)  # [b, max_faces, 1]

    mmem = prior_dict["motif_membership"].float() * face_mask.unsqueeze(-1).float()  # [b, max_faces, 3]
    minst = prior_dict["motif_instance"] if "motif_instance" in prior_dict else torch.zeros_like(mmem)  # [b, max_faces, 3] or [b, max_faces]

    if minst.ndim == 2:
        minst = minst.unsqueeze(-1)  # [b, max_faces, 1]

    pair_rel = prior_dict["pair_relations"].float()  # [b, max_faces, max_faces, R]

    node_features_list = []
    node_mask_list = []
    assignment_list = []
    hosted_adj_list = []
    thin_wall_adj_list = []

    for b in range(batch_size):
        # Node 0 is reserved for background node
        node_dicts = [{"key": "background", "role": -1, "faces": []}]

        # Collect instance nodes across 3 channels (0: sheet, 1: hole, 2: repeat)
        valid_faces = torch.where(face_mask[b])[0].tolist()

        for f_idx in valid_faces:
            for ch in range(minst.shape[-1]):
                inst_val = int(minst[b, f_idx, ch].item()) if ch < minst.shape[-1] else 0
                if inst_val > 0:
                    group_key = (ch, inst_val)
                    found = False
                    for n_idx, nd in enumerate(node_dicts):
                        if nd.get("key") == group_key:
                            nd["faces"].append(f_idx)
                            found = True
                            break
                    if not found and len(node_dicts) < max_nodes:
                        node_dicts.append({"key": group_key, "role": ch, "faces": [f_idx]})

        num_nodes = len(node_dicts)

        nf = torch.zeros((max_nodes, 11), device=device)
        nm = torch.zeros((max_nodes,), dtype=torch.bool, device=device)
        asgn = torch.zeros((max_faces, max_nodes), device=device)
        ha = torch.zeros((max_nodes, max_nodes), device=device)
        ta = torch.zeros((max_nodes, max_nodes), device=device)

        # Background node 0 is ALWAYS valid
        nm[0] = True
        nf[0, :] = 0.0  # background features

        # For faces that belong to no motif node, assign them to background node 0
        assigned_faces = set()
        for n_idx in range(1, num_nodes):
            nd = node_dicts[n_idx]
            f_list = nd["faces"]
            nm[n_idx] = True
            for f in f_list:
                asgn[f, n_idx] = 1.0
                assigned_faces.add(f)

            # Node features (11D)
            role_oh = F.one_hot(torch.tensor(nd["role"], device=device), num_classes=3).float()
            st_mean = stype_oh[b, f_list].mean(dim=0)
            mc_val = mconf[b, f_list].mean(dim=0)
            if mc_val.numel() > 1:
                mc_val = mc_val[nd["role"]:nd["role"]+1]
            sz_ratio = torch.tensor([len(f_list) / 30.0], device=device)
            nf[n_idx] = torch.cat([role_oh, st_mean, mc_val.reshape(1), sz_ratio], dim=-1)

        for f in valid_faces:
            if f not in assigned_faces:
                asgn[f, 0] = 1.0  # Assign unassigned faces to Background Node 0

        # Relation Adjacencies
        if pair_rel.shape[-1] >= 1:
            rel_h = pair_rel[b, :, :, 0]
        else:
            rel_h = torch.zeros((max_faces, max_faces), device=device)

        if pair_rel.shape[-1] >= 2:
            rel_t = pair_rel[b, :, :, 1]
        else:
            rel_t = torch.zeros((max_faces, max_faces), device=device)

        # Aggregate face-level relations to node-level
        for n1 in range(1, num_nodes):
            f1_list = node_dicts[n1]["faces"]
            for n2 in range(1, num_nodes):
                if n1 == n2:
                    continue
                f2_list = node_dicts[n2]["faces"]
                h_sum = rel_h[f1_list][:, f2_list].sum().item()
                t_sum = rel_t[f1_list][:, f2_list].sum().item()
                if h_sum > 0:
                    ha[n1, n2] = float(h_sum)
                if t_sum > 0:
                    ta[n1, n2] = float(t_sum)

        # Row normalization for relation adjacencies
        ha_denom = ha.sum(dim=-1, keepdim=True).clamp_min(1.0)
        ta_denom = ta.sum(dim=-1, keepdim=True).clamp_min(1.0)
        ha = ha / ha_denom
        ta = ta / ta_denom

        node_features_list.append(nf)
        node_mask_list.append(nm)
        assignment_list.append(asgn)
        hosted_adj_list.append(ha)
        thin_wall_adj_list.append(ta)

    return {
        "node_features": torch.stack(node_features_list, dim=0),  # [B, max_nodes, 11]
        "node_mask": torch.stack(node_mask_list, dim=0),  # [B, max_nodes]
        "assignment_target": torch.stack(assignment_list, dim=0),  # [B, max_faces, max_nodes]
        "hosted_adj": torch.stack(hosted_adj_list, dim=0),  # [B, max_nodes, max_nodes]
        "thin_wall_adj": torch.stack(thin_wall_adj_list, dim=0),  # [B, max_nodes, max_nodes]
    }
