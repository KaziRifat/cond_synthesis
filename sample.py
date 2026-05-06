"""
Inference Script — Conditional Image Synthesis

Usage:
  # Generate from segmentation map + text prompt:
  python sample.py --checkpoint outputs/cityscapes/ckpts/best.pt \
                   --config configs/cityscapes.yaml \
                   --seg_path input/seg.png \
                   --text "a rainy city street at night"

  # Generate from seg map only (no text):
  python sample.py --checkpoint outputs/cityscapes/ckpts/best.pt \
                   --config configs/cityscapes.yaml \
                   --seg_path input/seg.png

  # Style variation: same seg + text, different noise seeds:
  python sample.py --checkpoint outputs/cityscapes/ckpts/best.pt \
                   --config configs/cityscapes.yaml \
                   --seg_path input/seg.png \
                   --text "a sunny city street" \
                   --n_variations 8

  # Batch inference from a folder of seg maps:
  python sample.py --checkpoint outputs/cityscapes/ckpts/best.pt \
                   --config configs/cityscapes.yaml \
                   --seg_dir input/segs/ \
                   --text "a city at dusk"

  # Text interpolation: morph between two prompts:
  python sample.py --checkpoint outputs/cityscapes/ckpts/best.pt \
                   --config configs/cityscapes.yaml \
                   --seg_path input/seg.png \
                   --text_interp "a sunny day" "a rainy night" \
                   --n_steps 8
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from models.generator    import UNetGenerator
from models.clip_encoder import CLIPTextEncoder, CLIPProjector, NullTextEmbedding
from data.datasets       import get_seg_transform, seg_to_onehot, denormalize_image, colorize_seg
from utils.training      import EMA, load_checkpoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def dict_to_ns(d):
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, dict_to_ns(v) if isinstance(v, dict) else v)
    return ns


def get_device():
    if torch.cuda.is_available():         return torch.device('cuda')
    if torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')


def load_seg(seg_path: str, image_size: int, num_classes: int, device) -> torch.Tensor:
    """Load a segmentation PNG as (1, num_classes, H, W) one-hot tensor."""
    seg_img = Image.open(seg_path)
    tfm = get_seg_transform(image_size, augment=False)
    seg = tfm(seg_img)                             # (1, H, W)
    seg_onehot = seg_to_onehot(seg, num_classes)   # (num_classes, H, W)
    return seg_onehot.unsqueeze(0).to(device)       # (1, num_classes, H, W)


def build_generator_from_cfg(cfg, num_seg_classes):
    m = cfg.model
    return UNetGenerator(
        image_size=m.image_size,
        in_channels=m.in_channels,
        num_seg_classes=num_seg_classes,
        base_channels=m.base_channels,
        channel_mults=tuple(m.channel_mults),
        clip_dim=m.clip_dim,
        attn_levels=tuple(m.attn_levels),
        z_dim=m.z_dim,
        use_noise=m.use_noise,
    )


# ---------------------------------------------------------------------------
# Generation functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    generator, clip_encoder, clip_proj, null_emb,
    seg, text, z, device, cfg,
):
    B = seg.shape[0]
    if text:
        if isinstance(text, str):
            text = [text] * B
        token_emb, _ = clip_encoder.encode_text(text)
        token_emb = clip_proj(token_emb.to(device))
    else:
        token_emb, _ = null_emb(B, device)
        token_emb = clip_proj(token_emb)

    return generator(seg, token_emb, z)


@torch.no_grad()
def text_interpolation(
    generator, clip_encoder, clip_proj,
    seg, text_a, text_b, n_steps, z, device, cfg,
):
    """Linearly interpolate between two text embeddings in CLIP space."""
    token_a, sent_a = clip_encoder.encode_text([text_a])
    token_b, sent_b = clip_encoder.encode_text([text_b])

    token_a = clip_proj(token_a.to(device))
    token_b = clip_proj(token_b.to(device))

    outputs = []
    for i in range(n_steps):
        alpha = i / max(n_steps - 1, 1)
        token_interp = (1 - alpha) * token_a + alpha * token_b
        fake = generator(seg, token_interp, z)
        outputs.append(fake)

    return torch.cat(outputs, dim=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config',     required=True)
    parser.add_argument('--output_dir', default='outputs/generated')
    parser.add_argument('--seg_path',   default=None, help='Single seg map PNG')
    parser.add_argument('--seg_dir',    default=None, help='Folder of seg maps')
    parser.add_argument('--text',       default=None, help='Text conditioning prompt')
    parser.add_argument('--n_variations', type=int, default=1, help='Noise variations per sample')
    parser.add_argument('--text_interp', nargs=2, default=None, metavar=('TEXT_A', 'TEXT_B'))
    parser.add_argument('--n_steps',    type=int, default=8, help='Interpolation steps')
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--use_ema',    action='store_true', default=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    cfg = dict_to_ns(load_config(args.config))
    num_seg_classes = cfg.data.num_classes
    os.makedirs(args.output_dir, exist_ok=True)

    # Build models
    generator = build_generator_from_cfg(cfg, num_seg_classes).to(device)
    ema = EMA(generator)
    from models.discriminator import MultiScaleDiscriminator
    discriminator = MultiScaleDiscriminator(
        image_channels=cfg.model.in_channels,
        seg_channels=num_seg_classes,
        base_channels=cfg.model.disc_base_channels,
        num_layers=cfg.model.disc_num_layers,
        num_scales=cfg.model.disc_num_scales,
    ).to(device)

    load_checkpoint(args.checkpoint, generator, discriminator, ema=ema, device=str(device))
    if args.use_ema:
        ema.apply_shadow(generator)
    generator.eval()

    # CLIP
    clip_encoder = CLIPTextEncoder(model_name=cfg.clip.model_name, device=str(device))
    clip_proj    = CLIPProjector(clip_encoder.embed_dim, cfg.model.clip_dim).to(device)
    null_emb     = NullTextEmbedding(cfg.model.clip_dim, clip_encoder.max_tokens).to(device)

    # Collect seg paths
    seg_paths = []
    if args.seg_path:
        seg_paths = [args.seg_path]
    elif args.seg_dir:
        exts = {'.png', '.jpg', '.jpeg'}
        seg_paths = sorted([
            str(p) for p in Path(args.seg_dir).iterdir()
            if p.suffix.lower() in exts
        ])
    else:
        print("Provide --seg_path or --seg_dir")
        return

    for seg_path in seg_paths:
        stem = Path(seg_path).stem
        seg  = load_seg(seg_path, cfg.model.image_size, num_seg_classes, device)

        # ---- Text interpolation ----
        if args.text_interp:
            z = torch.randn(1, cfg.model.z_dim, device=device)
            interp_imgs = text_interpolation(
                generator, clip_encoder, clip_proj,
                seg, args.text_interp[0], args.text_interp[1],
                args.n_steps, z, device, cfg,
            )
            grid = make_grid(denormalize_image(interp_imgs), nrow=args.n_steps)
            path = os.path.join(args.output_dir, f'{stem}_interp.png')
            save_image(grid, path)
            print(f"Interpolation saved: {path}")
            continue

        # ---- Style variations ----
        outputs = []
        for v in range(args.n_variations):
            z = torch.randn(1, cfg.model.z_dim, device=device)
            fake = generate(generator, clip_encoder, clip_proj, null_emb,
                            seg, args.text, z, device, cfg)
            outputs.append(denormalize_image(fake))

        # Also save seg colormap for reference
        seg_colored = colorize_seg(seg.cpu(), num_seg_classes)  # (1, 3, H, W)

        all_imgs = [seg_colored] + outputs
        grid = make_grid(torch.cat(all_imgs, dim=0), nrow=len(all_imgs))
        suffix = f'_{args.text[:30].replace(" ","_")}' if args.text else ''
        path   = os.path.join(args.output_dir, f'{stem}{suffix}.png')
        save_image(grid, path)
        print(f"Saved: {path}")

    if args.use_ema:
        ema.restore(generator)


if __name__ == '__main__':
    main()
