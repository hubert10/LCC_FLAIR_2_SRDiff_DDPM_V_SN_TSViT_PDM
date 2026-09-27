import torch
import torch.nn as nn
import math


class FFCA(nn.Module):
    """
    Feature Fusion Cross-Attention (FFCA) for single-scale aerial and SITS
    feature maps.

    Inputs:
        aer_feats:
            Aerial feature map of shape (B, C_aer, H_aer, W_aer)

        sits_feats:
            SITS feature map of shape (B, C_sits, H_sits, W_sits)

    Output:
        Fused aerial feature map of shape (B, C_aer, H_aer, W_aer)
    """

    def __init__(
        self,
        aer_channels,
        sits_channels,
        num_heads=8,
        pos_enc=True,
    ):
        super().__init__()

        self.pos_enc = pos_enc

        # Project SITS channels to aerial channels
        self.sits_proj = nn.Conv2d(
            sits_channels,
            aer_channels,
            kernel_size=1,
        )

        # Cross-attention
        # Query  = aerial features
        # Key    = SITS features
        # Value  = SITS features
        self.attention = nn.MultiheadAttention(
            embed_dim=aer_channels,
            num_heads=num_heads,
            batch_first=True,
        )

        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(aer_channels, aer_channels * 4),
            nn.GELU(),
            nn.Linear(aer_channels * 4, aer_channels),
        )

        # Normalization
        self.norm = nn.LayerNorm(aer_channels)

        # Learnable fusion gate
        self.gate = nn.Parameter(torch.zeros(1))

    def _make_pos_enc(self, H, W, C, device):
        """Create 2D sinusoidal positional encoding."""

        y, x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        )

        pos_y = y.flatten().unsqueeze(1)
        pos_x = x.flatten().unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                C,
                2,
                device=device,
                dtype=torch.float32,
            )
            * -(math.log(10000.0) / C)
        )

        pe = torch.zeros(
            H * W,
            C,
            device=device,
        )

        pe[:, 0::2] = torch.sin(pos_x * div_term)
        pe[:, 1::2] = torch.cos(pos_y * div_term)

        return pe

    def forward(self, aer_feats, sits_feats):

        # ---------------------------------------------------------
        # 1. Input feature shapes
        # ---------------------------------------------------------
        B, C_aer, H_aer, W_aer = aer_feats.shape

        # ---------------------------------------------------------
        # 2. Project SITS features to aerial feature dimension
        # ---------------------------------------------------------
        sits_proj = self.sits_proj(sits_feats)

        # ---------------------------------------------------------
        # 3. Flatten spatial dimensions
        # ---------------------------------------------------------
        # Aerial:
        # (B, C_aer, H_aer, W_aer)
        #       ->
        # (B, H_aer*W_aer, C_aer)
        aer_flat = aer_feats.flatten(2).permute(0, 2, 1)

        # SITS:
        # (B, C_aer, H_sits, W_sits)
        #       ->
        # (B, H_sits*W_sits, C_aer)
        sits_flat = sits_proj.flatten(2).permute(0, 2, 1)

        # ---------------------------------------------------------
        # 4. Add positional encodings
        # ---------------------------------------------------------
        if self.pos_enc:

            pos_aer = self._make_pos_enc(
                H_aer,
                W_aer,
                C_aer,
                aer_feats.device,
            )

            pos_sits = self._make_pos_enc(
                sits_proj.shape[2],
                sits_proj.shape[3],
                C_aer,
                sits_proj.device,
            )

            aer_flat = aer_flat + pos_aer.unsqueeze(0)
            sits_flat = sits_flat + pos_sits.unsqueeze(0)

        # ---------------------------------------------------------
        # 5. Cross-attention
        # ---------------------------------------------------------
        # Query  = aerial
        # Key    = SITS
        # Value  = SITS
        attn_out, _ = self.attention(
            aer_flat,
            sits_flat,
            sits_flat,
        )

        # ---------------------------------------------------------
        # 6. Gated residual connection + LayerNorm
        # ---------------------------------------------------------
        alpha = torch.sigmoid(self.gate)

        fs_mhsa = self.norm(aer_flat + alpha * attn_out)

        # ---------------------------------------------------------
        # 7. MLP + residual + LayerNorm
        # ---------------------------------------------------------
        fused = self.norm(fs_mhsa + self.mlp(fs_mhsa))

        # ---------------------------------------------------------
        # 8. Reshape back to feature map
        # ---------------------------------------------------------
        fused = fused.permute(0, 2, 1).contiguous().view(B, C_aer, H_aer, W_aer)

        return fused
