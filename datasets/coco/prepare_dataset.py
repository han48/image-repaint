"""
Prepare 15,000 images from COCO 2017 for training.
Usage: python datasets/coco/prepare_dataset.py
"""
import os
import json
import random
import logging
from pathlib import Path
from functools import partial

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw" / "train2017"
PROCESSED_DIR = BASE_DIR / "processed"
SPLITS_JSON = BASE_DIR / "splits.json"

TOTAL = 15000
TRAIN_N = 12000
VAL_N = 1500
TEST_N = 1500
SIZE = 256
MIN_LAPLACIAN = 100
MIN_MEAN = 30
MAX_MEAN = 225


def is_valid_image(path: Path) -> bool:
    try:
        img = cv2.imread(str(path))
        if img is None:
            return False
        if img.ndim < 2:
            return False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        mean_val = gray.mean()
        if lap_var < MIN_LAPLACIAN:
            return False
        if mean_val < MIN_MEAN or mean_val > MAX_MEAN:
            return False
        return True
    except Exception:
        return False


def process_image(args):
    src_path, dst_path = args
    try:
        img = cv2.imread(str(src_path))
        if img is None:
            return False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(dst_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return True
    except Exception as e:
        log.warning(f"Failed to process {src_path}: {e}")
        return False


def prepare_dataset():
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw COCO images not found at {RAW_DIR}.\n"
            "Run: python datasets/coco/download_coco.py first."
        )

    all_images = sorted(RAW_DIR.glob("*.jpg"))
    log.info(f"Found {len(all_images)} raw images.")

    if len(all_images) < TOTAL:
        raise ValueError(
            f"Need {TOTAL} images but only {len(all_images)} found."
        )

    random.seed(42)
    random.shuffle(all_images)

    log.info("Filtering images for quality...")
    valid_images = []
    candidates = all_images[:min(len(all_images), TOTAL * 3)]
    for p in candidates:
        if is_valid_image(p):
            valid_images.append(p)
        if len(valid_images) >= TOTAL:
            break

    if len(valid_images) < TOTAL:
        log.warning(f"Only {len(valid_images)} valid images found (need {TOTAL}). Using all available.")
    else:
        valid_images = valid_images[:TOTAL]

    log.info(f"Selected {len(valid_images)} valid images.")

    train_imgs = valid_images[:TRAIN_N]
    val_imgs = valid_images[TRAIN_N:TRAIN_N + VAL_N]
    test_imgs = valid_images[TRAIN_N + VAL_N:TRAIN_N + VAL_N + TEST_N]

    splits = {'train': [], 'val': [], 'test': []}
    tasks = []

    for split_name, split_imgs in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
        split_dir = PROCESSED_DIR / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for src in split_imgs:
            dst = split_dir / src.name
            tasks.append((src, dst))
            splits[split_name].append(str(dst))

    log.info(f"Processing {len(tasks)} images sequentially...")
    results = []
    iterable = tqdm(tasks, desc='Processing') if tqdm else tasks
    for task in iterable:
        results.append(process_image(task))

    success = sum(results)
    log.info(f"Successfully processed {success}/{len(tasks)} images.")

    with open(SPLITS_JSON, 'w') as f:
        json.dump(splits, f, indent=2)
    log.info(f"Splits saved to {SPLITS_JSON}")
    log.info(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")


if __name__ == "__main__":
    prepare_dataset()
