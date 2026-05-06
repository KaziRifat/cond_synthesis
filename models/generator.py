"""
UNet Generator for Conditional Image Synthesis
Supports two conditioning modes (can be used together):
  1. Text conditioning via CLIP embeddings (cross-attention injection)
  2. Segmentation map conditioning (channel concatenation + SPADE normalization)

Architecture highlights:
  - Encoder-decoder UNet with skip connections
  - SPADE (Spatially-Adaptive Denormalization) for segmentation conditioning
  - Cross-attention blocks for CLIP text conditioning
  - Spectral normalization on all conv layers for training stability
  - Multi-scale feature injection at every decoder level
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Spectral Norm helper
# ---------------------------------------------------------------------------

def spectral_conv(in_ch, out_ch, kernel=3, stride=1, padding=1, bias=True):
    return nn.utils.spectral_norm(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=bias)
    )


def spectral_conv_transpose(in_ch, out_ch, kernel=4, stride=2, padding=1, bias=True):
    return nn.utils.spectral_norm(
        nn.ConvTranspose2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=bias)
    )


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------

class LeakyReLU(nn.Module):
    def __init__(self, slope=0.2):
        super().__init__()
        self.act = nn.LeakyReLU(slope, inplace=True)

    def forward(self, x):
        return self.act(x)


# ---------------------------------------------------------------------------
# SPADE: Spatially-Adaptive Denormalization
# Park et al. 2019 (GauGAN / SPADE)
# ---------------------------------------------------------------------------

class SPADE(nn.Module):
    """
    Replaces batch/instance norm with spatially-adaptive normalization
    conditioned on a segmentation map.

    For each spatial location (i,j), learns per-channel scale (gamma) and
    bias (beta) from the segmentation map. This lets the generator preserve
    semantic structure from the layout even in deep layers.
    """

    def __init__(self, channels: int, num_seg_classes: int, hidden: int = 128):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)

        # Shared encoder for segmentation map
        self.shared = nn.Sequential(
            nn.Conv2d(num_seg_classes, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # Separate heads for gamma (scale) and beta (bias)
        self.gamma = nn.Conv2d(hidden, channels, kernel_size=3, padding=1)
        self.beta  = nn.Conv2d(hidden, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        """
        x:   feature map  (B, C, H, W)
        seg: segmentation (B, num_classes, H_seg, W_seg) — one-hot or soft
        """
        # Resize seg to match feature map resolution
        if seg.shape[2:] != x.shape[2:]:
            seg = F.interpolate(seg, size=x.shape[2:], mode='nearest')

        # Normalize feature map
        normed = self.norm(x)

        # Compute spatially-varying scale/bias from segmentation
        shared = self.shared(seg)
        gamma  = self.gamma(shared)
        beta   = self.beta(shared)

        return normed * (1 + gamma) + beta


# ---------------------------------------------------------------------------
# SPADE Residual Block
# ---------------------------------------------------------------------------

class SPADEResBlock(nn.Module):
    """Residual block with SPADE normalization conditioned on segmentation."""

    def __init__(self, in_ch: int, out_ch: int, num_seg_classes: int):
        super().__init__()
        mid_ch = min(in_ch, out_ch)

        self.spade1 = SPADE(in_ch, num_seg_classes)
        self.conv1  = spectral_conv(in_ch, mid_ch)

        self.spade2 = SPADE(mid_ch, num_seg_classes)
        self.conv2  = spectral_conv(mid_ch, out_ch)

        self.act = nn.LeakyReLU(0.2, inplace=True)

        self.skip_spade = SPADE(in_ch, num_seg_classes) if in_ch != out_ch else None
        self.skip_conv  = spectral_conv(in_ch, out_ch, kernel=1, padding=0) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        h = self.act(self.spade1(x, seg))
        h = self.conv1(h)
        h = self.act(self.spade2(h, seg))
        h = self.conv2(h)

        if self.skip_conv is not None:
            x = self.skip_conv(self.act(self.skip_spade(x, seg)))

        return h + x


# ---------------------------------------------------------------------------
# Cross-Attention for CLIP Text Conditioning
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    """
    Injects CLIP text embeddings into spatial feature maps via cross-attention.

    Query  = flattened spatial features  (B, HW, C)
    Key/Value = CLIP token sequence      (B, L, clip_dim)

    This lets the generator attend to relevant parts of the text description
    at each spatial location independently.
    """

    def __init__(self, channels: int, clip_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = max(channels // num_heads, 1)
        self.scale     = self.head_dim ** -0.5

        self.norm_x    = nn.LayerNorm(channels)
        self.norm_clip = nn.LayerNorm(clip_dim)

        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_k = nn.Linear(clip_dim,  channels, bias=False)
        self.to_v = nn.Linear(clip_dim,  channels, bias=False)
        self.out  = nn.Linear(channels,  channels, bias=False)

    def forward(
        self,
        x: torch.Tensor,          # (B, C, H, W)
        clip_emb: torch.Tensor,   # (B, L, clip_dim)  L=token count
    ) -> torch.Tensor:
        B, C, H, W = x.shape

        # Flatten spatial dims
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)   # (B, HW, C)
        x_norm = self.norm_x(x_flat)
        c_norm = self.norm_clip(clip_emb)

        Q = self.to_q(x_norm)                             # (B, HW, C)
        K = self.to_k(c_norm)                             # (B, L, C)
        V = self.to_v(c_norm)                             # (B, L, C)

        # Multi-head reshape
        def split_heads(t, seq_len):
            return t.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        Q = split_heads(Q, H * W)   # (B, heads, HW, head_dim)
        K = split_heads(K, clip_emb.shape[1])
        V = split_heads(V, clip_emb.shape[1])

        attn = torch.einsum('bhqd,bhkd->bhqk', Q, K) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.einsum('bhqk,bhkd->bhqd', attn, V)   # (B, heads, HW, head_dim)
        out = out.transpose(1, 2).reshape(B, H * W, C)
        out = self.out(out)

        # Residual + reshape back to spatial
        out = (x_flat + out).permute(0, 2, 1).view(B, C, H, W)
        return out


# ---------------------------------------------------------------------------
# Encoder Block
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_norm: bool = True):
        super().__init__()
        layers = [spectral_conv(in_ch, out_ch, stride=2)]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        layers.append(LeakyReLU())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ---------------------------------------------------------------------------
# Decoder Block (with SPADE + optional cross-attention)
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        num_seg_classes: int,
        clip_dim: Optional[int] = None,
        use_attn: bool = False,
    ):
        super().__init__()
        self.upsample = spectral_conv_transpose(in_ch, out_ch)
        self.spade_res = SPADEResBlock(out_ch + skip_ch, out_ch, num_seg_classes)

        self.cross_attn = None
        if use_attn and clip_dim is not None:
            self.cross_attn = CrossAttention(out_ch, clip_dim)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        seg: torch.Tensor,
        clip_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.upsample(x)

        # Handle potential size mismatch from strided conv
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.spade_res(x, seg)

        if self.cross_attn is not None and clip_emb is not None:
            x = self.cross_attn(x, clip_emb)

        return x


# ---------------------------------------------------------------------------
# Full UNet Generator
# ---------------------------------------------------------------------------

class UNetGenerator(nn.Module):
    """
    Conditional UNet Generator.

    Conditioning:
      - Segmentation map: fed through SPADE at every decoder level
      - CLIP text embedding: injected via cross-attention at specified decoder levels

    Input:  noise z OR segmentation map (depending on mode)
    Output: generated RGB image in [-1, 1]
    """

    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,           # output channels
        seg_channels: int = 3,          # channels fed to encoder (noise+seg concat)
        num_seg_classes: int = 19,      # number of segmentation classes (one-hot)
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8, 8),
        clip_dim: int = 512,            # CLIP embedding dimension
        attn_levels: Tuple[int, ...] = (2, 3),  # decoder levels to apply cross-attn
        z_dim: int = 256,               # noise latent dimension
        use_noise: bool = True,
    ):
        super().__init__()
        self.use_noise = use_noise
        self.z_dim = z_dim
        self.num_levels = len(channel_mults)

        channels = [base_channels * m for m in channel_mults]

        # If using noise, project z + seg to initial feature map
        enc_in = num_seg_classes  # feed one-hot seg to encoder
        if use_noise:
            self.noise_proj = nn.Sequential(
                nn.Linear(z_dim, image_size * image_size),
                nn.Unflatten(1, (1, image_size, image_size)),
            )
            enc_in = num_seg_classes + 1  # seg + noise channel

        # ---- Encoder ----
        self.input_conv = spectral_conv(enc_in, channels[0], kernel=7, padding=3)

        self.encoders = nn.ModuleList()
        for i in range(self.num_levels - 1):
            self.encoders.append(EncoderBlock(channels[i], channels[i + 1]))

        # ---- Bottleneck ----
        self.bottleneck = SPADEResBlock(channels[-1], channels[-1], num_seg_classes)
        if len(attn_levels) > 0:
            self.bottleneck_attn = CrossAttention(channels[-1], clip_dim)
        else:
            self.bottleneck_attn = None

        # ---- Decoder ----
        rev = list(reversed(channels))
        self.decoders = nn.ModuleList()
        for i in range(self.num_levels - 1):
            use_attn = i in attn_levels
            self.decoders.append(DecoderBlock(
                in_ch=rev[i],
                skip_ch=rev[i + 1],
                out_ch=rev[i + 1],
                num_seg_classes=num_seg_classes,
                clip_dim=clip_dim,
                use_attn=use_attn,
            ))

        # ---- Output ----
        self.output_conv = nn.Sequential(
            nn.InstanceNorm2d(channels[0], affine=True),
            nn.ReLU(inplace=True),
            spectral_conv(channels[0], in_channels, kernel=7, padding=3),
            nn.Tanh(),
        )

    def forward(
        self,
        seg: torch.Tensor,              # (B, num_seg_classes, H, W) one-hot
        clip_emb: Optional[torch.Tensor] = None,  # (B, L, clip_dim) or (B, clip_dim)
        z: Optional[torch.Tensor] = None,          # (B, z_dim) noise
    ) -> torch.Tensor:
        B, _, H, W = seg.shape

        # Prepare clip_emb: ensure (B, L, clip_dim)
        if clip_emb is not None and clip_emb.dim() == 2:
            clip_emb = clip_emb.unsqueeze(1)  # (B, 1, clip_dim) if single vector

        # Build encoder input
        if self.use_noise:
            if z is None:
                z = torch.randn(B, self.z_dim, device=seg.device)
            noise_map = self.noise_proj(z)  # (B, 1, H, W)
            enc_in = torch.cat([seg, noise_map], dim=1)
        else:
            enc_in = seg

        # Encode
        x = self.input_conv(enc_in)
        skips = [x]
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)

        # Bottleneck
        x = self.bottleneck(x, seg)
        if self.bottleneck_attn is not None and clip_emb is not None:
            x = self.bottleneck_attn(x, clip_emb)

        # Decode
        skips = list(reversed(skips))
        for i, dec in enumerate(self.decoders):
            x = dec(x, skips[i + 1], seg, clip_emb)

        return self.output_conv(x)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    device = 'cpu'
    B, H, W = 2, 256, 256
    num_classes = 19

    gen = UNetGenerator(
        image_size=H,
        in_channels=3,
        num_seg_classes=num_classes,
        base_channels=32,
        channel_mults=(1, 2, 4, 4),
        clip_dim=512,
        attn_levels=(1, 2),
        z_dim=128,
    ).to(device)

    seg = torch.zeros(B, num_classes, H, W)
    seg[:, 3, :, :] = 1  # class 3 everywhere

    clip = torch.randn(B, 77, 512)   # 77 CLIP tokens
    z    = torch.randn(B, 128)

    out = gen(seg, clip, z)
    params = sum(p.numel() for p in gen.parameters()) / 1e6
    print(f"Generator output: {out.shape}  params: {params:.2f}M")
