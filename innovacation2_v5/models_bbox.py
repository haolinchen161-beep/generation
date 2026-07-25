"""PriorFaceBboxModel: Cross-Attention Face Role Allocator Architecture.

Implements Directives 1 to 7:
- MotifNodeEncoder: Node-level features + 1-layer relation message passing (hosted_by, thin_wall_pair).
- TopoFaceEncoder: 2-layer Graph Message Passing over fef_adj topology.
- CrossAttentionAllocator: Multi-Head Cross-Attention with key_padding_mask and zero_proj initialization.
- RoleHead: Auxiliary role supervision head predicting 3-class motif_membership role_logits [B, N_faces, 3].
- Unified PriorAllocator module for clean optimization and checkpoint saving.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Dict, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class MotifNodeEncoder(nn.Module):
    """Directive 1 & 2: Encodes Motif Graph Nodes with 1-layer Relation Message Passing."""

    def __init__(self, node_in_dim: int = 11, embed_dim: int = 512):
        super().__init__()
        self.node_proj = nn.Linear(node_in_dim, embed_dim)
        self.hosted_emb = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.thin_wall_emb = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

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
        # hosted_adj:    [b, N_nodes, N_nodes]
        # thin_wall_adj: [b, N_nodes, N_nodes]
        # node_mask:     [b, N_nodes]
        h = self.node_proj(node_features)  # [b, N_nodes, 512]

        # 1-layer Relation Graph Message Passing
        # Message from hosted_by edges
        m_hosted = torch.bmm(hosted_adj, h) * self.hosted_emb  # [b, N_nodes, 512]
        # Message from thin_wall_pair edges
        m_thin = torch.bmm(thin_wall_adj, h) * self.thin_wall_emb  # [b, N_nodes, 512]

        h_rel = h + m_hosted + m_thin
        h_rel = h_rel * node_mask.unsqueeze(-1).float()

        K = self.k_proj(h_rel)
        V = self.v_proj(h_rel)
        return K, V


class TopoFaceEncoder(nn.Module):
    """Directive 4: 2-layer Graph Message Passing Encoder for generated fef_adj topology."""

    def __init__(self, embed_dim: int = 512):
        super().__init__()
        # Initial feature dimension: Degree (1), Weighted Degree (1), Neighbor Degree Mean (1), Self-loop (1)
        self.input_proj = nn.Linear(4, 128)
        self.layer1 = nn.Linear(128, 256)
        self.layer2 = nn.Linear(256, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, fef_adj: torch.Tensor, face_mask: torch.Tensor) -> torch.Tensor:
        # fef_adj:   [b, N_faces, N_faces]
        # face_mask: [b, N_faces]
        b, n_faces, _ = fef_adj.shape
        fef_float = fef_adj.float()

        degree = fef_float.sum(dim=-1, keepdim=True) / 10.0  # [b, N_faces, 1]
        weighted_degree = fef_float.sum(dim=-1, keepdim=True) / 20.0  # [b, N_faces, 1]
        neighbor_deg = torch.bmm(fef_float, degree) / (degree.clamp_min(1.0) * 10.0)  # [b, N_faces, 1]
        self_loop = torch.diagonal(fef_float, dim1=1, dim2=2).unsqueeze(-1)  # [b, N_faces, 1]

        x0 = torch.cat([degree, weighted_degree, neighbor_deg, self_loop], dim=-1)
        x0 = x0 * face_mask.unsqueeze(-1).float()

        # Layer 1 Message Passing
        h1 = F.silu(self.input_proj(x0))
        m1 = torch.bmm(fef_float, h1) / (degree.clamp_min(1.0))
        h1 = F.silu(self.layer1(h1 + m1))

        # Layer 2 Message Passing
        m2 = torch.bmm(fef_float, h1) / (degree.clamp_min(1.0))
        h2 = F.silu(self.layer2(h1 + m2))

        h2 = h2 * face_mask.unsqueeze(-1).float()
        Q = self.q_proj(h2)  # [b, N_faces, 512]
        return Q


class CrossAttentionAllocator(nn.Module):
    """Directive 3 & 5: Multi-Head Cross-Attention Allocator with Auxiliary Role Classification Head."""

    def __init__(self, embed_dim: int = 512, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Zero-initialization guarantee (Directive 7)
        self.zero_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.zero_proj.weight)
        nn.init.zeros_(self.zero_proj.bias)

        # Directive 3: Auxiliary Role Classification Head
        self.role_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 3),  # Predicts 3-class motif_membership (sheet, hole, repeat)
        )

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        face_mask: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Q: [b, N_faces, 512]
        # K, V: [b, N_nodes, 512]
        # face_mask: [b, N_faces]
        # node_mask: [b, N_nodes]

        # Directive 5: Padding mask for invalid key nodes (True indicates padding/masked out)
        key_padding_mask = ~node_mask  # [b, N_nodes]

        attn_out, _ = self.attn(
            query=Q,
            key=K,
            value=V,
            key_padding_mask=key_padding_mask,
        )  # [b, N_faces, 512]

        # Directive 3: Predict auxiliary role_logits
        role_logits = self.role_head(attn_out)  # [b, N_faces, 3]

        # Zero-projected residual prior embedding
        per_face_prior = self.zero_proj(attn_out)  # [b, N_faces, 512]

        # Directive 5: Mask out invalid face tokens
        per_face_prior = per_face_prior * face_mask.unsqueeze(-1).float()

        return per_face_prior, role_logits


class PriorAllocator(nn.Module):
    """Directive 7: Unified PriorAllocator Module encapsulating all prior components."""

    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.node_encoder = MotifNodeEncoder(node_in_dim=11, embed_dim=embed_dim)
        self.topo_encoder = TopoFaceEncoder(embed_dim=embed_dim)
        self.allocator = CrossAttentionAllocator(embed_dim=embed_dim, num_heads=4)

    def forward(
        self,
        node_features: torch.Tensor,
        hosted_adj: torch.Tensor,
        thin_wall_adj: torch.Tensor,
        node_mask: torch.Tensor,
        fef_adj: torch.Tensor,
        face_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        K, V = self.node_encoder(node_features, hosted_adj, thin_wall_adj, node_mask)
        Q = self.topo_encoder(fef_adj, face_mask)
        per_face_prior, role_logits = self.allocator(Q, K, V, face_mask, node_mask)
        return per_face_prior, role_logits


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
            # Handle CFG batch duplication (e.g. 2x b when class_label/CFG is active)
            if prior.shape[0] < out.shape[0]:
                repeat_factor = (out.shape[0] + prior.shape[0] - 1) // prior.shape[0]
                prior = prior.repeat(repeat_factor, *([1] * (prior.ndim - 1)))
            prior = prior[: out.shape[0]]

            # Match face dimension if padded
            if prior.shape[1] < out.shape[1]:
                pad_len = out.shape[1] - prior.shape[1]
                prior = F.pad(prior, (0, 0, 0, pad_len))
            elif prior.shape[1] > out.shape[1]:
                prior = prior[:, : out.shape[1]]

            out = out + prior
        return out


class PriorFaceBboxModel(nn.Module):
    """Wrapper around official DTG FaceBboxTransformer to inject per_face_prior embeddings."""

    def __init__(self, base_model: nn.Module, embed_dim: int = 512):
        super().__init__()
        self.base = base_model
        # Directive 7: Unified prior_allocator module
        self.prior_allocator = PriorAllocator(embed_dim=embed_dim)

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
    if hasattr(official_model, "mlp_in_X") and hasattr(official_model.mlp_in_X[-1], "out_features"):
        embed_dim = official_model.mlp_in_X[-1].out_features
    elif hasattr(official_model, "embed_dim"):
        embed_dim = official_model.embed_dim
    elif hasattr(official_model, "d_model"):
        embed_dim = official_model.d_model

    model = PriorFaceBboxModel(official_model, embed_dim=embed_dim)
    return model
