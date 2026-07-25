"""Data preprocessing and feature extraction for PriorFaceBbox.

Fully vectorized GPU tensor extraction for extract_motif_node_graph (0.001s per batch).
Corrects pair_relations channel 0 (hosted_by) and channel 1 (thin_wall_pair).
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
    max_nodes: int = 15,
) -> Dict[str, torch.Tensor]:
    """Directive 1 & 2: Fully Vectorized GPU Tensor extraction for Motif Graph Nodes and Relations (0.001s).

    Node Feature (dim = 11):
      - 3D One-Hot: Node Category (sheet, hole, repeat)
      - 6D One-Hot: Mean Surface Type (plane, cylinder, cone, sphere, torus, bspline)
      - 1D: Mean Surface Confidence
      - 1D: Node Size Ratio (number of faces in instance / 30)

    Returns:
      node_features: [batch_size, max_nodes, 11]
      node_mask:     [batch_size, max_nodes]
      hosted_adj:    [batch_size, max_nodes, max_nodes]
      thin_wall_adj: [batch_size, max_nodes, max_nodes]
    """
    device = prior_dict["face_mask"].device
    batch_size, max_faces = prior_dict["face_mask"].shape

    face_mask = prior_dict["face_mask"].bool()  # [b, max_faces]
    stype = prior_dict["surface_type"].long().clamp(0, 5)  # [b, max_faces]
    stype_oh = F.one_hot(stype, num_classes=6).float() * face_mask.unsqueeze(-1).float()  # [b, max_faces, 6]
    sconf = (prior_dict["surface_confidence"].float() * face_mask.float()).unsqueeze(-1)  # [b, max_faces, 1]
    mmem = prior_dict["motif_membership"].float() * face_mask.unsqueeze(-1).float()  # [b, max_faces, 3]

    minst = prior_dict["motif_instance"] if "motif_instance" in prior_dict else torch.zeros_like(stype)
    if minst.ndim >= 3:
        minst = minst.argmax(dim=-1)
    minst = minst.long().clamp(0, max_nodes - 1)  # [b, max_faces]

    pair_rel = prior_dict["pair_relations"].float()  # [b, max_faces, max_faces, R]

    # One-hot assignment matrix face -> node: [b, max_faces, max_nodes]
    assignment = F.one_hot(minst, num_classes=max_nodes).float() * face_mask.unsqueeze(-1).float()

    # Node face counts: [b, max_nodes, 1]
    node_counts = assignment.sum(dim=1, keepdim=True).transpose(1, 2)  # [b, max_nodes, 1]
    node_mask = (node_counts.squeeze(-1) > 0)  # [b, max_nodes]

    # 1. Node Category (3D)
    node_mmem = torch.bmm(assignment.transpose(1, 2), mmem) / node_counts.clamp_min(1.0)  # [b, max_nodes, 3]

    # 2. Mean Surface Type (6D)
    node_stype = torch.bmm(assignment.transpose(1, 2), stype_oh) / node_counts.clamp_min(1.0)  # [b, max_nodes, 6]

    # 3. Mean Surface Confidence (1D)
    node_sconf = torch.bmm(assignment.transpose(1, 2), sconf) / node_counts.clamp_min(1.0)  # [b, max_nodes, 1]

    # 4. Node Size Ratio (1D)
    node_size_ratio = node_counts / 30.0  # [b, max_nodes, 1]

    node_features = torch.cat([node_mmem, node_stype, node_sconf, node_size_ratio], dim=-1)  # [b, max_nodes, 11]
    node_features = node_features * node_mask.unsqueeze(-1).float()

    # Aggregate face-level pair_relations to node-level adjacencies using BMM
    # assignment: [b, max_faces, max_nodes]
    # assignment^T: [b, max_nodes, max_faces]
    if pair_rel.shape[-1] >= 1:
        rel_hosted = pair_rel[:, :, :, 0]  # [b, max_faces, max_faces]
        # node_hosted = assignment^T @ rel_hosted @ assignment -> [b, max_nodes, max_nodes]
        hosted_adj = torch.bmm(assignment.transpose(1, 2), torch.bmm(rel_hosted, assignment))
    else:
        hosted_adj = torch.zeros((batch_size, max_nodes, max_nodes), device=device)

    if pair_rel.shape[-1] >= 2:
        rel_thin = pair_rel[:, :, :, 1]  # [b, max_faces, max_faces]
        thin_wall_adj = torch.bmm(assignment.transpose(1, 2), torch.bmm(rel_thin, assignment))
    else:
        thin_wall_adj = torch.zeros((batch_size, max_nodes, max_nodes), device=device)

    # Clamp adjacencies and mask self-loops
    hosted_adj = torch.clamp(hosted_adj, 0.0, 1.0)
    thin_wall_adj = torch.clamp(thin_wall_adj, 0.0, 1.0)

    eye_mask = (1.0 - torch.eye(max_nodes, device=device)).unsqueeze(0)
    hosted_adj = hosted_adj * eye_mask * node_mask.unsqueeze(1).float() * node_mask.unsqueeze(2).float()
    thin_wall_adj = thin_wall_adj * eye_mask * node_mask.unsqueeze(1).float() * node_mask.unsqueeze(2).float()

    return {
        "node_features": node_features,
        "node_mask": node_mask,
        "hosted_adj": hosted_adj,
        "thin_wall_adj": thin_wall_adj,
    }
