"""
PyTorch Dataset for 9-style sketch training.
Manifest entry: {"photo": str, "sketch": str, "style": int, "color": bool}
"""
import json
import logging
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class SketchDataset(Dataset):
    def __init__(self, manifest_path: str, transform: Optional[Callable] = None, size: int = 256):
        self.size = size
        self.transform = transform

        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f'Manifest not found: {manifest_path}')

        with open(manifest_path, 'r') as f:
            pairs = json.load(f)

        valid = []
        for p in pairs:
            if Path(p['photo']).exists() and Path(p['sketch']).exists():
                valid.append(p)
        dropped = len(pairs) - len(valid)
        if dropped:
            log.warning(f'Dropped {dropped} pairs with missing files')
        self.pairs = valid
        log.info(f'SketchDataset: {len(self.pairs)} pairs from {manifest_path}')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        is_color = p.get('color', False)
        style_id = int(p.get('style', 0))

        # Load photo (RGB)
        bgr = cv2.imread(p['photo'])
        if bgr is None:
            raise IOError(f"Cannot read photo: {p['photo']}")
        photo_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        photo_rgb = cv2.resize(photo_rgb, (self.size, self.size), interpolation=cv2.INTER_AREA)

        # Load sketch
        if is_color:
            s_bgr = cv2.imread(p['sketch'])
            if s_bgr is None:
                raise IOError(f"Cannot read sketch: {p['sketch']}")
            sketch_rgb = cv2.cvtColor(s_bgr, cv2.COLOR_BGR2RGB)
            sketch_rgb = cv2.resize(sketch_rgb, (self.size, self.size), interpolation=cv2.INTER_AREA)
        else:
            s_gray = cv2.imread(p['sketch'], cv2.IMREAD_GRAYSCALE)
            if s_gray is None:
                raise IOError(f"Cannot read sketch: {p['sketch']}")
            s_gray = cv2.resize(s_gray, (self.size, self.size), interpolation=cv2.INTER_AREA)
            sketch_rgb = cv2.cvtColor(s_gray, cv2.COLOR_GRAY2RGB)

        # Apply augmentation
        if self.transform is not None:
            aug = self.transform(image=photo_rgb, sketch=sketch_rgb)
            photo_rgb = aug['image']
            sketch_rgb = aug['sketch']

        # photo -> (3,H,W) float32 normalized [-1,1]
        photo_t = torch.from_numpy(photo_rgb.transpose(2, 0, 1)).float() / 255.0
        photo_t = (photo_t - 0.5) / 0.5

        # sketch -> (3,H,W) float32 [0,1]
        sketch_t = torch.from_numpy(sketch_rgb.transpose(2, 0, 1)).float() / 255.0

        return photo_t, sketch_t, torch.tensor(style_id, dtype=torch.long)
