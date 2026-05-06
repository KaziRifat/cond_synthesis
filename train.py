"""
Main Training Script — Conditional Image Synthesis GAN

Features:
  - Alternating G / D updates with configurable update ratio
  - Feature matching loss (pix2pixHD style)
  - VGG perceptual loss
  - EMA on generator
  - Multi-scale PatchGAN discriminator
  - CLIP text + segmentation conditioning
  - WandB / TensorBoard logging

Usage:
  python train.py --config configs/cityscapes.yaml
  python train.py --config configs/ade20k.yaml
  python train.py --config configs/debug.yaml
  python train.py --config configs/cityscapes.yaml --resume outputs/cityscapes/ckpts/best.pt
  python train.py --config configs/cityscapes.yaml training.batch_size=2
"""

import os
import sys
import time
import argparse
import yaml
import random
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

from models.generator     import UNetGenerator
from models.discriminator import MultiScaleDiscriminator, GANLoss, FeatureMatchingLoss, VGGPerceptualLoss
from models.clip_encoder  import CLIPTextEncoder, CLIPProjector, NullTextEmbedding
from data.datasets        import build_dataloaders
from utils.training       import (
    EMA, Logger, WarmupLinearScheduler,
    save_checkpoint, load_checkpoint,
    save_sample_grid, save_images_for_fid,
    count_parameters, grad_norm,
)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg, overrides):
    for override in overrides:
        if '=' not in override:
            continue
        key_path, value = override.split('=', 1)
        keys = key_path.split('.')
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        try:    value = int(value)
        except: 
            try:    value = float(value)
            except:
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
        d[keys[-1]] = value
    return cfg


def dict_to_ns(d):
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, dict_to_ns(v) if isinstance(v, dict) else v)
    return ns


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():      return torch.device('cuda')
    if torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')


# ---------------------------------------------------------------------------
# Build models
# ---------------------------------------------------------------------------

def build_generator(cfg, num_seg_classes):
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


def build_discriminator(cfg, num_seg_classes):
    m = cfg.model
    return MultiScaleDiscriminator(
        image_channels=m.in_channels,
        seg_channels=num_seg_classes,
        base_channels=m.disc_base_channels,
        num_layers=m.disc_num_layers,
        num_scales=m.disc_num_scales,
    )


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_fixed_samples(generator, clip_encoder, clip_proj, null_emb, fixed_batch, device, cfg):
    generator.eval()
    seg = fixed_batch['seg_onehot'].to(device)
    B   = seg.shape[0]

    captions = fixed_batch['caption']
    if any(c for c in captions):
        token_emb, _ = clip_encoder.encode_text(captions)
        token_emb = clip_proj(token_emb.to(device))
    else:
        token_emb, _ = null_emb(B, device)
        token_emb = clip_proj(token_emb)

    z = torch.randn(B, cfg.model.z_dim, device=device)
    fake = generator(seg, token_emb, z)
    generator.train()
    return fake


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg, resume_path=None):
    set_seed(cfg.training.seed)
    device = get_device()
    print(f"Training on: {device}")

    # Data
    train_loader, val_loader, num_seg_classes = build_dataloaders(
        dataset_name=cfg.data.dataset,
        data_root=cfg.data.root,
        image_size=cfg.model.image_size,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        augment=cfg.data.augment,
        num_classes=cfg.data.num_classes,
    )
    print(f"Dataset: {cfg.data.dataset} | Seg classes: {num_seg_classes} | "
          f"Train: {len(train_loader)} batches")

    # Models
    generator     = build_generator(cfg, num_seg_classes).to(device)
    discriminator = build_discriminator(cfg, num_seg_classes).to(device)

    print(f"Generator:     {count_parameters(generator)/1e6:.2f}M params")
    print(f"Discriminator: {count_parameters(discriminator)/1e6:.2f}M params")

    # CLIP
    clip_encoder = CLIPTextEncoder(
        model_name=cfg.clip.model_name,
        device=str(device),
        freeze=True,
    )
    clip_proj = CLIPProjector(
        clip_dim=clip_encoder.embed_dim,
        target_dim=cfg.model.clip_dim,
    ).to(device)
    null_emb = NullTextEmbedding(
        clip_dim=cfg.model.clip_dim,
        num_tokens=clip_encoder.max_tokens,
    ).to(device)

    # Losses
    gan_loss  = GANLoss(mode=cfg.training.gan_loss_mode)
    feat_loss = FeatureMatchingLoss(weight=cfg.training.lambda_feat)
    vgg_loss  = VGGPerceptualLoss(weight=cfg.training.lambda_vgg)

    # Optimizers
    g_params = (
        list(generator.parameters()) +
        list(clip_proj.parameters()) +
        list(null_emb.parameters())
    )
    opt_g = torch.optim.Adam(g_params, lr=cfg.training.lr_g, betas=(0.0, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=cfg.training.lr_d, betas=(0.0, 0.999))

    total_steps   = cfg.training.epochs * len(train_loader)
    warmup_steps  = cfg.training.warmup_epochs * len(train_loader)
    g_sched = torch.optim.lr_scheduler.LambdaLR(opt_g, WarmupLinearScheduler(warmup_steps, total_steps))
    d_sched = torch.optim.lr_scheduler.LambdaLR(opt_d, WarmupLinearScheduler(warmup_steps, total_steps))

    # EMA
    ema = EMA(generator, decay=cfg.training.ema_decay)

    # Logger
    logger = Logger(
        config={},
        project='cond-synthesis',
        use_wandb=cfg.logging.use_wandb,
        use_tb=cfg.logging.use_tensorboard,
    )

    # Directories
    ckpt_dir   = os.path.join(cfg.training.output_dir, 'ckpts')
    sample_dir = os.path.join(cfg.training.output_dir, 'samples')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # Resume
    start_epoch  = 0
    global_step  = 0
    if resume_path:
        meta = load_checkpoint(resume_path, generator, discriminator, opt_g, opt_d, ema, str(device))
        start_epoch = meta.get('epoch', 0) + 1
        global_step = meta.get('step', 0)

    # Fixed validation batch for consistent sample grids
    fixed_batch = next(iter(val_loader))

    best_g_loss = float('inf')
    generator.train()
    discriminator.train()

    for epoch in range(start_epoch, cfg.training.epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            real_img   = batch['image'].to(device)
            seg_onehot = batch['seg_onehot'].to(device)
            captions   = batch['caption']
            B          = real_img.shape[0]

            # --- CLIP encoding ---
            if any(c for c in captions):
                with torch.no_grad():
                    token_emb, _ = clip_encoder.encode_text(captions)
                token_emb = clip_proj(token_emb.to(device))
            else:
                token_emb, _ = null_emb(B, device)
                token_emb = clip_proj(token_emb)

            # Random noise for generator
            z = torch.randn(B, cfg.model.z_dim, device=device)

            # ============================================================
            # 1. Update Discriminator
            # ============================================================
            for _ in range(cfg.training.n_disc_steps):
                with torch.no_grad():
                    fake_img = generator(seg_onehot, token_emb, z)

                real_preds, real_feats = discriminator(real_img.detach(), seg_onehot)
                fake_preds, fake_feats = discriminator(fake_img.detach(), seg_onehot)

                loss_d = gan_loss.discriminator_loss(real_preds, fake_preds)

                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                if cfg.training.grad_clip > 0:
                    nn.utils.clip_grad_norm_(discriminator.parameters(), cfg.training.grad_clip)
                opt_d.step()

            # ============================================================
            # 2. Update Generator
            # ============================================================
            fake_img = generator(seg_onehot, token_emb, z)
            fake_preds_g, fake_feats_g = discriminator(fake_img, seg_onehot)
            real_preds_r, real_feats_r = discriminator(real_img.detach(), seg_onehot)

            loss_g_adv  = gan_loss.generator_loss(fake_preds_g)
            loss_g_feat = feat_loss(real_feats_r, fake_feats_g)
            loss_g_vgg  = vgg_loss(fake_img, real_img)
            loss_g      = loss_g_adv + loss_g_feat + loss_g_vgg

            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            if cfg.training.grad_clip > 0:
                nn.utils.clip_grad_norm_(generator.parameters(), cfg.training.grad_clip)
            opt_g.step()

            g_sched.step()
            d_sched.step()
            ema.update(generator)

            epoch_g_loss += loss_g.item()
            epoch_d_loss += loss_d.item()
            global_step  += 1

            # Logging
            if global_step % cfg.logging.log_every == 0:
                logger.log({
                    'train/loss_G':     loss_g.item(),
                    'train/loss_G_adv': loss_g_adv.item(),
                    'train/loss_G_feat':loss_g_feat.item(),
                    'train/loss_G_vgg': loss_g_vgg.item(),
                    'train/loss_D':     loss_d.item(),
                    'train/lr_G':       opt_g.param_groups[0]['lr'],
                }, step=global_step)

            # Sample grid
            if global_step % cfg.logging.sample_every == 0:
                ema.apply_shadow(generator)
                fake_val = generate_fixed_samples(
                    generator, clip_encoder, clip_proj, null_emb, fixed_batch, device, cfg
                )
                ema.restore(generator)
                grid_path = os.path.join(sample_dir, f'step_{global_step:07d}.png')
                save_sample_grid(
                    fixed_batch['image'], fake_val.cpu(),
                    fixed_batch['seg_onehot'], grid_path,
                    nrow=min(4, B), num_seg_classes=num_seg_classes,
                )
                print(f"  → Samples saved: {grid_path}")

        # End of epoch
        avg_g = epoch_g_loss / len(train_loader)
        avg_d = epoch_d_loss / len(train_loader)
        elapsed = time.time() - t0
        print(f"\nEpoch [{epoch+1}/{cfg.training.epochs}] "
              f"G={avg_g:.4f}  D={avg_d:.4f}  time={elapsed:.1f}s")

        if (epoch + 1) % cfg.logging.save_every_epochs == 0:
            save_checkpoint(ckpt_dir, epoch, global_step,
                            generator, discriminator, opt_g, opt_d, ema)

        if avg_g < best_g_loss:
            best_g_loss = avg_g
            save_checkpoint(ckpt_dir, epoch, global_step,
                            generator, discriminator, opt_g, opt_d, ema, best=True)

    logger.finish()
    print(f"\nTraining complete. Best G loss: {best_g_loss:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', default=None)
    parser.add_argument('overrides', nargs='*')
    args = parser.parse_args()

    cfg_dict = load_config(args.config)
    cfg_dict = apply_overrides(cfg_dict, args.overrides)
    cfg = dict_to_ns(cfg_dict)
    os.makedirs(cfg.training.output_dir, exist_ok=True)
    train(cfg, resume_path=args.resume)
