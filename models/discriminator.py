"""
Multi-Scale PatchGAN Discriminator

Architecture:
  - Three discriminators operating at different scales (original, 0.5x, 0.25x)
  - Each discriminator is a PatchGAN: classifies overlapping NxN patches as real/fake
  - Conditioning: real/fake decision is conditioned on segmentation map
  - Feature matching loss: intermediate activations from all discriminators
  - Spectral normalization throughout

Why multi-scale?
  The coarse discriminator sees global structure (layout, composition).
  The fine discriminator sees local texture and detail.
  This prevents mode collapse and encourages both global coherence and fine detail.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Spectral norm convenience
# ---------------------------------------------------------------------------

def sn_conv(in_ch, out_ch, kernel=4, stride=2, padding=1, bias=True):
    return nn.utils.spectral_norm(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=bias)
    )


# ---------------------------------------------------------------------------
# Single PatchGAN Discriminator
# ---------------------------------------------------------------------------

class PatchDiscriminator(nn.Module):
    """
    Standard PatchGAN discriminator.

    Input: concat(image, seg_map) → patch-level real/fake predictions
    Returns list of intermediate feature maps (for feature matching loss).

    Architecture:
      Conv → [InstanceNorm + LeakyReLU] × N → Conv (1-channel output)
    """

    def __init__(
        self,
        in_channels: int,         # image channels + seg channels
        base_channels: int = 64,
        num_layers: int = 4,
        use_sigmoid: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        # First layer: no norm
        self.layers.append(nn.Sequential(
            sn_conv(in_channels, base_channels, kernel=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ))

        ch = base_channels
        for i in range(1, num_layers):
            ch_next = min(ch * 2, 512)
            stride = 1 if i == num_layers - 1 else 2
            self.layers.append(nn.Sequential(
                sn_conv(ch, ch_next, kernel=4, stride=stride, padding=1),
                nn.InstanceNorm2d(ch_next, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            ch = ch_next

        # Output: 1-channel patch prediction
        final = [sn_conv(ch, 1, kernel=4, stride=1, padding=1)]
        if use_sigmoid:
            final.append(nn.Sigmoid())
        self.layers.append(nn.Sequential(*final))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
            pred:     patch logit map   (B, 1, H', W')
            features: list of intermediate feature maps (for feature matching)
        """
        features = []
        h = x
        for layer in self.layers[:-1]:
            h = layer(h)
            features.append(h)
        pred = self.layers[-1](h)
        return pred, features


# ---------------------------------------------------------------------------
# Multi-Scale Discriminator
# ---------------------------------------------------------------------------

class MultiScaleDiscriminator(nn.Module):
    """
    Runs 3 PatchGAN discriminators on the same image at 3 different scales.

    Scale 0: original resolution
    Scale 1: 0.5x downsampled
    Scale 2: 0.25x downsampled

    Conditions all discriminators on the segmentation map (resized to match).
    """

    def __init__(
        self,
        image_channels: int = 3,
        seg_channels: int = 19,
        base_channels: int = 64,
        num_layers: int = 4,
        num_scales: int = 3,
    ):
        super().__init__()
        self.num_scales = num_scales
        # Input = image + seg for each discriminator
        in_ch = image_channels + seg_channels

        self.discriminators = nn.ModuleList([
            PatchDiscriminator(in_ch, base_channels, num_layers)
            for _ in range(num_scales)
        ])
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(
        self,
        image: torch.Tensor,   # (B, 3, H, W)
        seg: torch.Tensor,     # (B, num_classes, H, W)
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Returns:
            preds:    list of patch prediction maps, one per scale
            features: list of feature lists, one per scale
        """
        all_preds = []
        all_feats = []

        img_scaled = image
        seg_scaled = seg

        for disc in self.discriminators:
            # Resize seg to match image scale
            if seg_scaled.shape[2:] != img_scaled.shape[2:]:
                seg_scaled = F.interpolate(
                    seg, size=img_scaled.shape[2:], mode='nearest'
                )
            inp = torch.cat([img_scaled, seg_scaled], dim=1)
            pred, feats = disc(inp)
            all_preds.append(pred)
            all_feats.append(feats)

            # Downsample for next scale
            img_scaled = self.downsample(img_scaled)

        return all_preds, all_feats


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class GANLoss(nn.Module):
    """
    Unified GAN loss supporting:
      - hinge loss (default, most stable)
      - lsgan (least-squares)
      - vanilla (BCE)
    """

    def __init__(self, mode: str = 'hinge'):
        super().__init__()
        self.mode = mode

    def discriminator_loss(
        self, real_preds: List[torch.Tensor], fake_preds: List[torch.Tensor]
    ) -> torch.Tensor:
        """Discriminator: maximize margin between real and fake."""
        loss = 0.0
        for real, fake in zip(real_preds, fake_preds):
            if self.mode == 'hinge':
                loss += F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean()
            elif self.mode == 'lsgan':
                loss += F.mse_loss(real, torch.ones_like(real)) + \
                        F.mse_loss(fake, torch.zeros_like(fake))
            else:  # vanilla
                loss += F.binary_cross_entropy_with_logits(real, torch.ones_like(real)) + \
                        F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake))
        return loss / len(real_preds)

    def generator_loss(self, fake_preds: List[torch.Tensor]) -> torch.Tensor:
        """Generator: fool the discriminator."""
        loss = 0.0
        for fake in fake_preds:
            if self.mode == 'hinge':
                loss += -fake.mean()
            elif self.mode == 'lsgan':
                loss += F.mse_loss(fake, torch.ones_like(fake))
            else:
                loss += F.binary_cross_entropy_with_logits(fake, torch.ones_like(fake))
        return loss / len(fake_preds)


class FeatureMatchingLoss(nn.Module):
    """
    Pix2PixHD feature matching loss.
    Minimizes L1 distance between real and fake intermediate discriminator features.
    Stabilizes training and encourages realistic textures.
    """

    def __init__(self, weight: float = 10.0):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        real_feats: List[List[torch.Tensor]],
        fake_feats: List[List[torch.Tensor]],
    ) -> torch.Tensor:
        loss = 0.0
        n = 0
        for real_scale_feats, fake_scale_feats in zip(real_feats, fake_feats):
            for real_f, fake_f in zip(real_scale_feats, fake_scale_feats):
                loss += F.l1_loss(fake_f, real_f.detach())
                n += 1
        return self.weight * loss / max(n, 1)


class VGGPerceptualLoss(nn.Module):
    """
    VGG-19 perceptual loss on relu1_2, relu2_2, relu3_3, relu4_3 activations.
    Encourages generated images to have realistic high-level features.
    Falls back gracefully if torchvision isn't available.
    """

    def __init__(self, weight: float = 10.0):
        super().__init__()
        self.weight = weight
        self.vgg = None
        self.slice_ids = [3, 8, 15, 22]  # relu1_2, relu2_2, relu3_3, relu4_3

        try:
            import torchvision.models as tvm
            vgg = tvm.vgg19(weights=tvm.VGG19_Weights.IMAGENET1K_V1)
            features = vgg.features
            self.slices = nn.ModuleList([
                nn.Sequential(*list(features.children())[:sid])
                for sid in self.slice_ids
            ])
            for p in self.parameters():
                p.requires_grad_(False)
            self.vgg = True
            print("VGG perceptual loss: loaded ✓")
        except Exception as e:
            print(f"VGG perceptual loss unavailable ({e}), skipping.")

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        if self.vgg is None:
            return torch.tensor(0.0, device=fake.device)

        # Normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=fake.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=fake.device).view(1, 3, 1, 1)

        fake_n = (fake * 0.5 + 0.5 - mean) / std
        real_n = (real * 0.5 + 0.5 - mean) / std

        loss = 0.0
        f_in, r_in = fake_n, real_n
        prev_sid = 0
        for sid, slc in zip(self.slice_ids, self.slices):
            f_feat = slc(fake_n)
            r_feat = slc(real_n)
            loss += F.l1_loss(f_feat, r_feat.detach())

        return self.weight * loss / len(self.slice_ids)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    B, H, W = 2, 256, 256
    num_classes = 19

    disc = MultiScaleDiscriminator(image_channels=3, seg_channels=num_classes)
    image = torch.randn(B, 3, H, W)
    seg   = torch.zeros(B, num_classes, H, W)
    seg[:, 5] = 1

    preds, feats = disc(image, seg)
    print(f"Discriminator scales: {len(preds)}")
    for i, p in enumerate(preds):
        print(f"  Scale {i}: {p.shape}")

    gan_loss = GANLoss('hinge')
    fake_preds = [torch.randn_like(p) for p in preds]
    d_loss = gan_loss.discriminator_loss(preds, fake_preds)
    g_loss = gan_loss.generator_loss(fake_preds)
    print(f"D loss: {d_loss.item():.4f}  G loss: {g_loss.item():.4f}")
