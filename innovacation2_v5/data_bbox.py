"""Data preprocessing and feature extraction for PriorFaceBbox (v6).

Implements Core Directives 1, 3, 5, 6:
- max_nodes = 64 with explicit overflow warning.
- Node 0 = Permanent Background Node (node_mask[:, 0] = True, node_roles[:, 0] = -1).
- Returns node_roles [B, 64] (-1: background, 0: sheet, 1: hole, 2: repeat).
- Type-constrained relation graph:
  - hosted_by: only local_node (hole/repeat) -> sheet_node.
  - thin_wall_pair: only sheet_node <-> sheet_node.
- Row-normalized relation adjacencies.
"""

from __future__ import annotations

import warnings
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
    max_nodes: int = 64,
) -> Dict[str, torch.Tensor]:
    """Core Directives 1, 2, 3, 5, 6: Construct Motif Graph Nodes with Background Node & Type-Constrained Edges.

    Node 0: Permanent Background Node (node_mask[:, 0] = True, node_roles[:, 0] = -1).
    Nodes 1..max_nodes-1: Real Motif Instance Nodes (channel, instance_id).

    Returns:
      node_features:     [B, max_nodes, 11]
      node_mask:         [B, max_nodes] (bool, node 0 always True)
      node_roles:        [B, max_nodes] (long: -1=background, 0=sheet, 1=hole, 2=repeat)
      assignment_target: [B, N_faces, max_nodes] (multi-hot, float)
      hosted_adj:        [B, max_nodes, max_nodes] (row-normalized)
      thin_wall_adj:     [B, max_nodes, max_nodes] (row-normalized)
    """
    device = prior_dict["face_mask"].device
    batch_size, max_faces = prior_dict["face_mask"].shape

    face_mask = prior_dict["face_mask"].bool()  # [b, max_faces]
    stype = prior_dict["surface_type"].long().clamp(0, 5)  # [b, max_faces]
    stype_oh = F.one_hot(stype, num_classes=6).float() * face_mask.unsqueeze(-1).float()  # [b, max_faces, 6]

    if "motif_confidence" in prior_dict:
        mconf = prior_dict["motif_confidence"].float()
        if mconf.ndim == 2:
            mconf = mconf.unsqueeze(-1)
    else:
        mconf = prior_dict["surface_confidence"].float().unsqueeze(-1)  # [b, max_faces, 1]

    mmem = prior_dict["motif_membership"].float() * face_mask.unsqueeze(-1).float()  # [b, max_faces, 3]
    minst = prior_dict["motif_instance"] if "motif_instance" in prior_dict else torch.zeros_like(mmem)

    if minst.ndim == 2:
        minst = minst.unsqueeze(-1)  # [b, max_faces, 1]

    pair_rel = prior_dict["pair_relations"].float()  # [b, max_faces, max_faces, R]

    node_features_list = []
    node_mask_list = []
    node_roles_list = []
    assignment_list = []
    hosted_adj_list = []
    thin_wall_adj_list = []

    for b in range(batch_size):
        # Node 0 is reserved for background node (role = -1)
        node_dicts = [{"key": "background", "role": -1, "faces": []}]

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
                    elif not found and len(node_dicts) >= max_nodes:
                        warnings.warn(f"Sample {b}: Node count exceeds max_nodes ({max_nodes})! Node truncation triggered.")

        num_nodes = len(node_dicts)

        nf = torch.zeros((max_nodes, 11), device=device)
        nm = torch.zeros((max_nodes,), dtype=torch.bool, device=device)
        nr = torch.full((max_nodes,), -1, dtype=torch.long, device=device)
        asgn = torch.zeros((max_faces, max_nodes), device=device)
        ha = torch.zeros((max_nodes, max_nodes), device=device)
        ta = torch.zeros((max_nodes, max_nodes), device=device)

        # Background node 0 is ALWAYS valid
        nm[0] = True
        nr[0] = -1
        nf[0, :] = 0.0

        assigned_faces = set()
        for n_idx in range(1, num_nodes):
            nd = node_dicts[n_idx]
            f_list = nd["faces"]
            nm[n_idx] = True
            nr[n_idx] = nd["role"]

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

        # Pair relations with TYPE CONSTRAINTS
        if pair_rel.shape[-1] >= 1:
            rel_h = pair_rel[b, :, :, 0]
        else:
            rel_h = torch.zeros((max_faces, max_faces), device=device)

        if pair_rel.shape[-1] >= 2:
            rel_t = pair_rel[b, :, :, 1]
        else:
            rel_t = torch.zeros((max_faces, max_faces), device=device)

        for n1 in range(1, num_nodes):
            r1 = node_dicts[n1]["role"]
            f1_list = node_dicts[n1]["faces"]
            for n2 in range(1, num_nodes):
                if n1 == n2:
                    continue
                r2 = node_dicts[n2]["role"]
                f2_list = node_dicts[n2]["faces"]

                # hosted_by: ONLY allow local_node (r1=1:hole, r2=2:repeat) -> sheet_node (r2=0)
                if (r1 in (1, 2)) and (r2 == 0):
                    h_sum = rel_h[f1_list][:, f2_list].sum().item()
                    if h_sum > 0:
                        ha[n1, n2] = float(h_sum)

                # thin_wall_pair: ONLY allow sheet_node (r1=0) <-> sheet_node (r2=0)
                if (r1 == 0) and (r2 == 0):
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

    return {
        "node_features": torch.stack(node_features_list, dim=0),  # [B, max_nodes, 11]
        "node_mask": torch.stack(node_mask_list, dim=0),          # [B, max_nodes]
        "node_roles": torch.stack(node_roles_list, dim=0),        # [B, max_nodes]
        "assignment_target": torch.stack(assignment_list, dim=0),  # [B, max_faces, max_nodes]
        "hosted_adj": torch.stack(hosted_adj_list, dim=0),        # [B, max_nodes, max_nodes]
        "thin_wall_adj": torch.stack(thin_wall_adj_list, dim=0),    # [B, max_nodes, max_nodes]
    }
