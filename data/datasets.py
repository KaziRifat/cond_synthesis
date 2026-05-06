"""
Dataset utilities for Conditional Image Synthesis.

Supports:
  - Cityscapes (semantic segmentation → street scene)
  - ADE20K     (semantic segmentation → indoor/outdoor scenes)
  - COCO-Stuff (segmentation → diverse scenes)
  - Custom paired datasets (seg_map + real_image + optional caption)

All datasets return:
  image:    (3, H, W) in [-1, 1]
  seg_onehot: (num_classes, H, W) one-hot segmentation map
  seg_label:  (1, H, W) integer class labels (for visualization)
  caption:  str or None
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def get_image_transform(image_size: int, augment: bool = True) -> transforms.Compose:
    """Image to [-1, 1] tensor."""
    t = []
    if augment:
        t.append(transforms.RandomHorizontalFlip())
    t += [
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    return transforms.Compose(t)


def get_seg_transform(image_size: int, augment: bool = True) -> transforms.Compose:
    """Segmentation label map to (1, H, W) int tensor."""
    t = []
    if augment:
        t.append(transforms.RandomHorizontalFlip())
    t += [
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST),
        transforms.PILToTensor(),
    ]
    return transforms.Compose(t)


def seg_to_onehot(seg: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convert (1, H, W) integer label map to (num_classes, H, W) one-hot.
    Clamps invalid class indices to 0.
    """
    seg = seg.squeeze(0).long().clamp(0, num_classes - 1)  # (H, W)
    onehot = torch.zeros(num_classes, seg.shape[0], seg.shape[1])
    onehot.scatter_(0, seg.unsqueeze(0), 1)
    return onehot


def denormalize_image(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1) / 2


# ---------------------------------------------------------------------------
# Dataset: Cityscapes
# ---------------------------------------------------------------------------

CITYSCAPES_CLASSES = [
    'unlabeled', 'ego vehicle', 'rectification border', 'out of roi', 'static',
    'dynamic', 'ground', 'road', 'sidewalk', 'parking', 'rail track', 'building',
    'wall', 'fence', 'guard rail', 'bridge', 'tunnel', 'pole', 'polegroup',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person',
    'rider', 'car', 'truck', 'bus', 'caravan', 'trailer', 'train', 'motorcycle',
    'bicycle',
]
NUM_CITYSCAPES_CLASSES = 35


class CityscapesDataset(Dataset):
    """
    Cityscapes semantic segmentation dataset.

    Expected structure:
      root/leftImg8bit/{train,val}/city_name/xxx_leftImg8bit.png
      root/gtFine/{train,val}/city_name/xxx_gtFine_labelIds.png

    Download: https://www.cityscapes-dataset.com/
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        image_size: int = 256,
        augment: bool = True,
        num_classes: int = NUM_CITYSCAPES_CLASSES,
    ):
        self.root = root
        self.split = split
        self.num_classes = num_classes
        self.img_transform = get_image_transform(image_size, augment)
        self.seg_transform = get_seg_transform(image_size, augment)

        self.pairs = self._collect_pairs()
        if len(self.pairs) == 0:
            raise FileNotFoundError(
                f"No Cityscapes pairs found in {root}. "
                f"Please download from https://www.cityscapes-dataset.com/"
            )

    def _collect_pairs(self) -> List[Tuple[str, str]]:
        img_root = os.path.join(self.root, 'leftImg8bit', self.split)
        seg_root = os.path.join(self.root, 'gtFine', self.split)
        pairs = []
        if not os.path.exists(img_root):
            return pairs
        for city in sorted(os.listdir(img_root)):
            city_img = os.path.join(img_root, city)
            city_seg = os.path.join(seg_root, city)
            for fname in sorted(os.listdir(city_img)):
                if not fname.endswith('_leftImg8bit.png'):
                    continue
                seg_fname = fname.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
                seg_path = os.path.join(city_seg, seg_fname)
                if os.path.exists(seg_path):
                    pairs.append((os.path.join(city_img, fname), seg_path))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, seg_path = self.pairs[idx]

        # Apply same random flip to both image and seg
        flip = random.random() < 0.5

        img = Image.open(img_path).convert('RGB')
        seg = Image.open(seg_path)

        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            seg = seg.transpose(Image.FLIP_LEFT_RIGHT)

        img = self.img_transform(img)
        seg = self.seg_transform(seg)
        seg_onehot = seg_to_onehot(seg, self.num_classes)

        return {
            'image': img,
            'seg_onehot': seg_onehot,
            'seg_label': seg,
            'caption': '',
        }


# ---------------------------------------------------------------------------
# Dataset: ADE20K
# ---------------------------------------------------------------------------

NUM_ADE20K_CLASSES = 150


class ADE20KDataset(Dataset):
    """
    ADE20K scene parsing dataset.

    Expected structure:
      root/images/training/*.jpg
      root/annotations/training/*.png
      root/images/validation/*.jpg
      root/annotations/validation/*.png

    Download: http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
    """

    SPLIT_MAP = {'train': 'training', 'val': 'validation'}

    def __init__(
        self,
        root: str,
        split: str = 'train',
        image_size: int = 256,
        augment: bool = True,
        num_classes: int = NUM_ADE20K_CLASSES,
    ):
        self.root = root
        self.num_classes = num_classes
        folder = self.SPLIT_MAP.get(split, split)
        self.img_dir = os.path.join(root, 'images', folder)
        self.seg_dir = os.path.join(root, 'annotations', folder)
        self.img_transform = get_image_transform(image_size, augment)
        self.seg_transform = get_seg_transform(image_size, augment and split == 'train')
        self.augment = augment and split == 'train'

        self.images = sorted([
            f for f in os.listdir(self.img_dir) if f.endswith('.jpg')
        ]) if os.path.exists(self.img_dir) else []

        if len(self.images) == 0:
            raise FileNotFoundError(
                f"No ADE20K images found in {self.img_dir}. "
                f"Download from http://data.csail.mit.edu/places/ADEchallenge/"
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        img_path = os.path.join(self.img_dir, name)
        seg_path = os.path.join(self.seg_dir, name.replace('.jpg', '.png'))

        flip = self.augment and random.random() < 0.5

        img = Image.open(img_path).convert('RGB')
        seg = Image.open(seg_path) if os.path.exists(seg_path) else Image.new('L', img.size)

        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            seg = seg.transpose(Image.FLIP_LEFT_RIGHT)

        img = self.img_transform(img)
        seg = self.seg_transform(seg)
        # ADE20K: class 0 = unlabeled, classes 1–150 are semantic
        seg = seg.clamp(0, self.num_classes - 1)
        seg_onehot = seg_to_onehot(seg, self.num_classes)

        return {
            'image': img,
            'seg_onehot': seg_onehot,
            'seg_label': seg,
            'caption': '',
        }


# ---------------------------------------------------------------------------
# Dataset: Generic Paired (custom)
# ---------------------------------------------------------------------------

class PairedDataset(Dataset):
    """
    Generic paired dataset for custom data.

    Expected structure:
      root/images/001.jpg 002.jpg ...
      root/segs/001.png   002.png ...
      root/captions.json  (optional) {"001": "a cat on a sofa", ...}

    The seg PNG should store integer class labels as pixel values.
    """

    def __init__(
        self,
        root: str,
        image_size: int = 256,
        num_classes: int = 20,
        augment: bool = True,
        captions_file: Optional[str] = None,
    ):
        self.root = root
        self.num_classes = num_classes
        self.img_transform = get_image_transform(image_size, augment)
        self.seg_transform = get_seg_transform(image_size, augment)
        self.augment = augment

        img_dir = os.path.join(root, 'images')
        seg_dir = os.path.join(root, 'segs')
        exts = {'.jpg', '.jpeg', '.png', '.bmp'}

        self.samples = []
        if os.path.exists(img_dir):
            for fname in sorted(os.listdir(img_dir)):
                stem, ext = os.path.splitext(fname)
                if ext.lower() not in exts:
                    continue
                seg_path = os.path.join(seg_dir, stem + '.png')
                self.samples.append((os.path.join(img_dir, fname), seg_path, stem))

        # Load captions
        self.captions: Dict[str, str] = {}
        cap_path = captions_file or os.path.join(root, 'captions.json')
        if os.path.exists(cap_path):
            with open(cap_path) as f:
                self.captions = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, seg_path, stem = self.samples[idx]
        flip = self.augment and random.random() < 0.5

        img = Image.open(img_path).convert('RGB')
        seg = Image.open(seg_path).convert('L') if os.path.exists(seg_path) else Image.new('L', img.size)

        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            seg = seg.transpose(Image.FLIP_LEFT_RIGHT)

        img = self.img_transform(img)
        seg = self.seg_transform(seg)
        seg_onehot = seg_to_onehot(seg, self.num_classes)
        caption = self.captions.get(stem, '')

        return {
            'image': img,
            'seg_onehot': seg_onehot,
            'seg_label': seg,
            'caption': caption,
        }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
    'cityscapes': CityscapesDataset,
    'ade20k':     ADE20KDataset,
    'custom':     PairedDataset,
}

NUM_CLASSES_MAP = {
    'cityscapes': NUM_CITYSCAPES_CLASSES,
    'ade20k':     NUM_ADE20K_CLASSES,
}


def build_dataloaders(
    dataset_name: str,
    data_root: str,
    image_size: int = 256,
    batch_size: int = 4,
    num_workers: int = 4,
    augment: bool = True,
    num_classes: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, int]:
    """Returns (train_loader, val_loader, num_classes)."""
    if num_classes is None:
        num_classes = NUM_CLASSES_MAP.get(dataset_name, 20)

    DatasetClass = DATASET_REGISTRY.get(dataset_name, PairedDataset)

    try:
        train_ds = DatasetClass(data_root, split='train', image_size=image_size,
                                augment=augment, num_classes=num_classes)
        val_ds   = DatasetClass(data_root, split='val',   image_size=image_size,
                                augment=False, num_classes=num_classes)
    except TypeError:
        # PairedDataset doesn't have split parameter
        full_ds = DatasetClass(data_root, image_size=image_size,
                               num_classes=num_classes, augment=augment)
        n_val   = max(1, int(len(full_ds) * 0.1))
        n_train = len(full_ds) - n_val
        train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=num_workers > 0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True, drop_last=False,
                              persistent_workers=num_workers > 0)

    return train_loader, val_loader, num_classes
