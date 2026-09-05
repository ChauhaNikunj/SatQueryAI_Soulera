"""
SatQuery AI — Visual Spatial Reasoning (VSR) Adapter Model
==========================================================
Spatial relation reasoning adapter on top of RemoteCLIP embeddings.
Evaluates spatial statements (e.g., 'object A is north of object B', 'water is adjacent to urban area').
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VSRSpatialAdapter(nn.Module):
    """
    Multimodal Spatial Reasoning Head.
    Fuses visual embedding (512-d) and text query embedding (512-d)
    to classify spatial relation validity (Binary: True / False, or multi-class relation).
    """

    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Cross-modal projection
        self.vis_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.txt_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Bilinear & Concat Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # [v, t, v*t]
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, img_embed: torch.Tensor, txt_embed: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for spatial relation verification.
        
        Parameters
        ----------
        img_embed : torch.Tensor (B, 512)
        txt_embed : torch.Tensor (B, 512)
        
        Returns
        -------
        logits : torch.Tensor (B, 2) [False, True]
        """
        v = self.vis_proj(img_embed)      # (B, H)
        t = self.txt_proj(txt_embed)      # (B, H)
        elem = v * t                      # Hadamard product interaction (B, H)
        fused = torch.cat([v, t, elem], dim=-1)  # (B, 3*H)
        logits = self.fusion(fused)       # (B, num_classes)
        return logits
