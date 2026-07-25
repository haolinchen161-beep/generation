"""PriorFaceBboxModel: Cross-Attention Face Role & Instance Allocator Architecture (v6.1).

Implements Core Directives 1, 2, 4, 5, 7:
- MAX_MOTIF_NODES = 96.
- MotifNodeEncoder: Projects K & V and masks with node_mask to eliminate linear projection bias on padding nodes.
- Pairwise Dot-Product assignment_head with invalid node masked_fill(-1e4).
- Node Role Filters:
  - host_node_mask = valid_node_mask & ((node_roles == -1) | (node_roles == 0))
  - local_node_mask = valid_node_mask & ((node_roles == 1) | (node_roles == 2))
- Host Branch: Masked Softmax -> host_context.
- Local Branch: Sigmoid normalized by local_mass (local_probs.sum(-1, keepdim=True).clamp_min(1.0)) -> local_context.
- Fusion: mlp_fuse(concat([host_context, local_context])) -> zero_proj -> per_face_prior.
"""

from __future__ import annotations

import sys
import math
from pathlib import Path
from typing import Optional, Dict, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MAX_MOTIF_NODES = 96


class MotifNodeEncoder(nn.Module):
    """Directive 2 & 4: Encodes Motif Graph Nodes with LayerNorm & Row-Normalized Relation Message Passing."""

    def __init__(self, node_in_dim: int = 11, embed_dim: int = 512):
        super().__init__()
        self.node_proj = nn.Linear(node_in_dim, embed_dim)
        self.hosted_proj = nn.Linear(embed_dim, embed_dim)
        self.thin_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        hosted_adj: torch.Tensor,
        thin_wall_adj: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # node_features: [b, N_nodes, 11]
        # hosted_adj:    [b, N_nodes, N_nodes] (row-normalized)
        # thin_wall_adj: [b, N_nodes, N_nodes] (row-normalized)
        # node_mask:     [b, N_nodes]
        h = self.node_proj(node_features)  # [b, N_nodes, 512]

        m_hosted = torch.bmm(hosted_adj, self.hosted_proj(h))
        m_thin = torch.bmm(thin_wall_adj, self.thin_proj(h))

        h_rel = self.norm(h + m_hosted + m_thin)

        K = self.k_proj(h_rel)
        V = self.v_proj(h_rel)

        # Directive 1: Multiply K & V by node_mask AFTER linear projection to eliminate bias leakage on padding nodes!
        mask = node_mask.unsqueeze(-1).to(K.dtype)
        K = K * mask
        V = V * mask
        return K, V


class TopoFaceEncoder(nn.Module):
    """Directive 4: 2-layer GCN Encoder for generated fef_adj topology with raw-degree neighbor calculation."""

    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.input_proj = nn.Linear(4, 128)
        self.layer1 = nn.Linear(128, 256)
        self.layer2 = nn.Linear(256, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, fef_adj: torch.Tensor, face_mask: torch.Tensor) -> torch.Tensor:
        # fef_adj:   [b, N_faces, N_faces]
        # face_mask: [b, N_faces]
        A_weight = fef_adj.float()
        A_binary = (fef_adj > 0).float()

        # Directive 4: Compute unscaled raw features FIRST!
        degree_raw = A_binary.sum(dim=-1, keepdim=True)
        weighted_degree_raw = A_weight.sum(dim=-1, keepdim=True)
        neighbor_degree_mean_raw = torch.bmm(A_binary, degree_raw) / (degree_raw.clamp_min(1.0))
        max_multiplicity_raw = A_weight.max(dim=-1, keepdim=True).values

        # Normalize features individually
        degree = degree_raw / 10.0
        weighted_degree = weighted_degree_raw / 20.0
        neighbor_degree_mean = neighbor_degree_mean_raw / 10.0
        max_multiplicity = max_multiplicity_raw / 5.0

        x0 = torch.cat([degree, weighted_degree, neighbor_degree_mean, max_multiplicity], dim=-1)
        x0 = x0 * face_mask.unsqueeze(-1).float()

        # Row-normalized adjacency
        A_norm = A_weight / A_weight.sum(dim=-1, keepdim=True).clamp_min(1.0)

        # Layer 1 Message Passing
        h1 = F.silu(self.input_proj(x0))
        m1 = torch.bmm(A_norm, h1)
        h1 = F.silu(self.layer1(h1 + m1))

        # Layer 2 Message Passing
        m2 = torch.bmm(A_norm, h1)
        h2 = F.silu(self.layer2(h1 + m2))

        h2 = h2 * face_mask.unsqueeze(-1).float()
        Q = self.q_proj(h2)  # [b, N_faces, 512]
        return Q


class CrossAttentionAllocator(nn.Module):
    """Directive 1, 2, 4: Pairwise Dot-Product assignment_head & Dual-Branch (Host/Local) Context Fusion."""

    def __init__(self, embed_dim: int = 512, max_nodes: int = MAX_MOTIF_NODES, num_heads: int = 4):
        super().__init__()
        self.max_nodes = max_nodes
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Directive 1: Pairwise Dot-Product Assignment Head
        self.assignment_q = nn.Linear(embed_dim, 128)
        self.assignment_k = nn.Linear(embed_dim, 128)

        # Role Classification Head [B, N_faces, 3] (sheet, hole, repeat)
        self.role_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )

        # Directive 2: Dual Branch Fusion MLP (Concat host_context [512] + local_context [512] -> 512)
        self.mlp_fuse = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Zero-initialization for residual prior projection
        self.zero_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.zero_proj.weight)
        nn.init.zeros_(self.zero_proj.bias)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        face_mask: torch.Tensor,
        node_mask: torch.Tensor,
        node_roles: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Q: [b, N_faces, 512]
        # K, V: [b, N_nodes, 512]
        # face_mask: [b, N_faces]
        # node_mask: [b, N_nodes] (node 0 background is always True)
        # node_roles: [b, N_nodes] (-2: pad, -1: background, 0: sheet, 1: hole, 2: repeat)

        key_padding_mask = ~node_mask  # [b, N_nodes]

        attn_out, _ = self.attn(
            query=Q,
            key=K,
            value=V,
            key_padding_mask=key_padding_mask,
        )  # [b, N_faces, 512]

        # 1. Role Logits [B, N_faces, 3]
        role_logits = self.role_head(attn_out)

        # 2. Pairwise Dot-Product Assignment Logits [B, N_faces, N_nodes]
        fq = self.assignment_q(Q)  # [b, N_faces, 128]
        nk = self.assignment_k(K)  # [b, N_nodes, 128]
        assignment_logits = torch.bmm(fq, nk.transpose(1, 2)) / math.sqrt(128.0)  # [b, N_faces, N_nodes]

        # Directive 1: Mask out invalid nodes in assignment_logits
        assignment_logits = assignment_logits.masked_fill(~node_mask.unsqueeze(1), -1e4)

        # 3. Directive 1 & 2: Node Role Filters
        valid_node_mask = node_mask.bool()  # [b, N_nodes]

        # Host Mask: Background (-1) and Sheet (0) nodes ONLY! (Excludes padding -2 and hole/repeat)
        host_node_mask = valid_node_mask & ((node_roles == -1) | (node_roles == 0))
        host_mask_expanded = host_node_mask.unsqueeze(1).expand(-1, Q.shape[1], -1)

        # Local Mask: Hole (1) and Repeat (2) nodes ONLY!
        local_node_mask = valid_node_mask & ((node_roles == 1) | (node_roles == 2))
        local_mask_expanded = local_node_mask.unsqueeze(1).expand(-1, Q.shape[1], -1)

        # Branch 1: Host Branch (Masked Softmax over Host nodes)
        host_logits = assignment_logits.masked_fill(~host_mask_expanded, -1e4)
        host_probs = F.softmax(host_logits, dim=-1) * host_mask_expanded.float()
        host_context = torch.bmm(host_probs, V)  # [b, N_faces, 512]

        # Branch 2: Local Branch (Sigmoid Multi-Label normalized by local_mass)
        local_probs = torch.sigmoid(assignment_logits) * local_mask_expanded.float()

        # Directive 4: Normalize by local_mass to prevent amplitude explosion!
        local_mass = local_probs.sum(dim=-1, keepdim=True).clamp_min(1.0)
        local_context = torch.bmm(local_probs, V) / local_mass  # [b, N_faces, 512]

        # Directive 2: Dual Branch Fusion without unified normalization
        fused_context = self.mlp_fuse(torch.cat([host_context, local_context], dim=-1))

        # Zero-projected residual prior
        per_face_prior = self.zero_proj(fused_context)  # [b, N_faces, 512]
        per_face_prior = per_face_prior * face_mask.unsqueeze(-1).float()

        return per_face_prior, role_logits, assignment_logits


class PriorAllocator(nn.Module):
    """Unified PriorAllocator Module encapsulating NodeEncoder, TopoEncoder, and Allocator."""

    def __init__(self, embed_dim: int = 512, max_nodes: int = MAX_MOTIF_NODES):
        super().__init__()
        self.node_encoder = MotifNodeEncoder(node_in_dim=11, embed_dim=embed_dim)
        self.topo_encoder = TopoFaceEncoder(embed_dim=embed_dim)
        self.allocator = CrossAttentionAllocator(embed_dim=embed_dim, max_nodes=max_nodes, num_heads=4)

    def forward(
        self,
        node_features: torch.Tensor,
        hosted_adj: torch.Tensor,
        thin_wall_adj: torch.Tensor,
        node_mask: torch.Tensor,
        node_roles: torch.Tensor,
        fef_adj: torch.Tensor,
        face_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        K, V = self.node_encoder(node_features, hosted_adj, thin_wall_adj, node_mask)
        Q = self.topo_encoder(fef_adj, face_mask)
        per_face_prior, role_logits, assignment_logits = self.allocator(Q, K, V, face_mask, node_mask, node_roles)
        return per_face_prior, role_logits, assignment_logits


class GuidedMLPInX(nn.Module):
    """Wrapper Module for base.mlp_in_X that adds per_face_prior [b, N_faces, 512] to face tokens."""

    def __init__(self, orig_mlp_in_x: nn.Module):
        super().__init__()
        self.orig = orig_mlp_in_x
        self.per_face_prior: Optional[torch.Tensor] = None

    def set_per_face_prior(self, per_face_prior: Optional[torch.Tensor]) -> None:
        self.per_face_prior = per_face_prior

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.orig(x)  # [batch_size, num_faces, 512]
        if self.per_face_prior is not None:
            prior = self.per_face_prior  # [b, N_faces, 512]
            if prior.shape[0] < out.shape[0]:
                repeat_factor = (out.shape[0] + prior.shape[0] - 1) // prior.shape[0]
                prior = prior.repeat(repeat_factor, *([1] * (prior.ndim - 1)))
            prior = prior[: out.shape[0]]

            if prior.shape[1] < out.shape[1]:
                pad_len = out.shape[1] - prior.shape[1]
                prior = F.pad(prior, (0, 0, 0, pad_len))
            elif prior.shape[1] > out.shape[1]:
                prior = prior[:, : out.shape[1]]

            out = out + prior
        return out


class PriorFaceBboxModel(nn.Module):
    """Wrapper around official DTG FaceBboxTransformer to inject per_face_prior embeddings."""

    def __init__(self, base_model: nn.Module, embed_dim: int = 512, max_nodes: int = MAX_MOTIF_NODES):
        super().__init__()
        self.base = base_model
        self.prior_allocator = PriorAllocator(embed_dim=embed_dim, max_nodes=max_nodes)

        embed_attr = "mlp_in_X" if hasattr(self.base, "mlp_in_X") else ("node_embed" if hasattr(self.base, "node_embed") else None)
        if embed_attr is None:
            raise AttributeError("Base model has neither mlp_in_X nor node_embed attribute!")

        orig_fn = getattr(self.base, embed_attr)
        self.guided_embed_module = GuidedMLPInX(orig_fn)
        setattr(self.base, embed_attr, self.guided_embed_module)

    def set_prior(self, per_face_prior: Optional[torch.Tensor]) -> None:
        self.guided_embed_module.set_per_face_prior(per_face_prior)

    def forward(
        self,
        face_bbox: torch.Tensor,
        e: torch.Tensor,
        face_mask: torch.Tensor,
        class_label: Optional[torch.Tensor] = None,
        point_data: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.base(face_bbox, e, face_mask, class_label, point_data, timesteps)


def build_prior_face_bbox_model(official_model: nn.Module) -> PriorFaceBboxModel:
    """Build PriorFaceBboxModel wrapping official pretrained FaceBbox model."""
    embed_dim = 512
    if hasattr(official_model, "mlp_in_X"):
        if isinstance(official_model.mlp_in_X, nn.Sequential) and hasattr(official_model.mlp_in_X[-1], "out_features"):
            embed_dim = official_model.mlp_in_X[-1].out_features
        elif hasattr(official_model.mlp_in_X, "out_features"):
            embed_dim = official_model.mlp_in_X.out_features
    elif hasattr(official_model, "embed_dim"):
        embed_dim = official_model.embed_dim
    elif hasattr(official_model, "d_model"):
        embed_dim = official_model.d_model

    model = PriorFaceBboxModel(official_model, embed_dim=embed_dim, max_nodes=MAX_MOTIF_NODES)
    return model
