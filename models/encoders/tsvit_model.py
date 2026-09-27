import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from utils.utils import Attention, FeedForward, PreNorm
from utils.utils_models import init_weights


class Conv(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        stride=1,
        bias=False,
    ):
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2

        super().__init__(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        )


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        stride=1,
        norm_layer=nn.BatchNorm2d,
        bias=False,
    ):
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2

        super().__init__(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            norm_layer(out_channels),
            nn.ReLU6(),
        )


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=dropout,
                            ),
                        ),
                        PreNorm(
                            dim,
                            FeedForward(
                                dim,
                                mlp_dim,
                                dropout=dropout,
                            ),
                        ),
                    ]
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for attn, ff in self.layers:
            x = x + attn(x)
            x = x + ff(x)

        return self.norm(x)


class TSViT_global_attention_spatial_encoder(nn.Module):
    """
    Temporal-Spatial Vision Transformer.

    The temporal encoder processes the temporal sequence independently
    for each spatial patch using class-specific temporal tokens.

    The resulting class tokens are then processed jointly by the
    spatial transformer.
    """

    def __init__(
        self,
        image_size=64,
        patch_size=2,
        num_patches_1d=1,
        num_channels=4,
        num_classes=19,
        num_frames=1,
        dim=128,
        depth=4,
        heads=4,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        pool="cls",
        scale_dim=4,
    ):
        super().__init__()

        assert pool in {"cls", "mean"}, "pool must be either 'cls' or 'mean'"
        assert (
            image_size % patch_size == 0
        ), "image_size must be divisible by patch_size"

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches_1d = image_size // patch_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.pool = pool
        self.scale_dim = scale_dim

        num_patches = self.num_patches_1d**2
        patch_dim = num_channels * patch_size**2

        # ---------------------------------------------------------
        # Temporal embedding
        # ---------------------------------------------------------

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)",
                p1=patch_size,
                p2=patch_size,
            ),
            nn.Linear(
                patch_dim,
                dim,
            ),
        )

        # Temporal positions are represented using a 365-dimensional
        # one-hot encoding.
        self.to_temporal_embedding_input = nn.Linear(
            365,
            dim,
        )

        # One temporal class token per segmentation class.
        self.temporal_token = nn.Parameter(
            torch.randn(
                1,
                num_classes,
                dim,
            )
        )

        self.temporal_transformer = Transformer(
            dim=dim,
            depth=depth + 2,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=dim * scale_dim,
            dropout=dropout,
        )

        # ---------------------------------------------------------
        # Spatial embedding
        # ---------------------------------------------------------

        self.space_pos_embedding = nn.Parameter(
            torch.randn(
                1,
                num_patches,
                dim,
            )
        )

        self.space_transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=dim * scale_dim,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(emb_dropout)

        # ---------------------------------------------------------
        # Segmentation Head
        # ---------------------------------------------------------

        self.segmentation_head = nn.Sequential(
            ConvBNReLU(
                dim,
                dim,
            ),
            nn.Dropout2d(
                p=dropout,
                inplace=True,
            ),
            Conv(
                dim,
                num_classes,
                kernel_size=1,
            ),
        )

    def forward(self, x, batch_positions):
        """
        Args:
            x:
                SITS input tensor of shape:
                (B, T, C, H, W)

            batch_positions:
                Normalized temporal positions of shape:
                (B, T).

                Values are expected in [0, 1).

        Returns:
            out:
                Segmentation output of shape:
                (B, num_classes, H, W).

            feats:
                Decoder feature map.
        """

        B, T, C, H, W = x.shape

        # ---------------------------------------------------------
        # Temporal positional encoding
        # ---------------------------------------------------------
        # print("x:", x.shape)
        xt = (batch_positions * 365.0001).to(torch.int64)
        # print("xt:", xt.shape)
        xt = F.one_hot(
            xt,
            num_classes=365,
        ).float()
        # print("xt:", xt.shape)

        temporal_pos_embedding = self.to_temporal_embedding_input(
            xt.reshape(-1, 365)
        ).reshape(
            B,
            T,
            self.dim,
        )

        # ---------------------------------------------------------
        # Patch embedding
        # ---------------------------------------------------------

        x = self.to_patch_embedding(x)

        # (B * H_p * W_p, T, D)
        x = x.reshape(
            B,
            -1,
            T,
            self.dim,
        )

        # Add temporal positional embedding.
        x = x + temporal_pos_embedding.unsqueeze(1)

        # (B * H_p * W_p, T, D)
        x = x.reshape(
            -1,
            T,
            self.dim,
        )

        # ---------------------------------------------------------
        # Temporal transformer
        # ---------------------------------------------------------

        cls_temporal_tokens = repeat(
            self.temporal_token,
            "() N D -> B N D",
            B=B * self.num_patches_1d**2,
        )

        x = torch.cat(
            (
                cls_temporal_tokens,
                x,
            ),
            dim=1,
        )

        x = self.temporal_transformer(x)

        # Keep only the class-specific temporal tokens.
        x = x[:, : self.num_classes]

        # ---------------------------------------------------------
        # Spatial transformer
        # ---------------------------------------------------------

        # (B, H_p*W_p, K, D)
        x = x.reshape(
            B,
            self.num_patches_1d**2,
            self.num_classes,
            self.dim,
        )

        # (B, K, H_p*W_p, D)
        x = x.permute(
            0,
            2,
            1,
            3,
        )

        # Process all classes in parallel.
        x = x.reshape(
            B * self.num_classes,
            self.num_patches_1d**2,
            self.dim,
        )

        # Add spatial positional encoding.
        x = x + self.space_pos_embedding

        x = self.dropout(x)

        # Combine class and spatial dimensions.
        x = x.reshape(
            B,
            self.num_classes * self.num_patches_1d**2,
            self.dim,
        )

        x = self.space_transformer(x)

        # ---------------------------------------------------------
        # Segmentation decoder
        # ---------------------------------------------------------

        GS = H // self.patch_size
        feats = rearrange(x, "b (h w) c -> b c h w", h=GS)

        feats = F.interpolate(
            feats,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        out = self.segmentation_head(feats)

        return out, feats
