"""
Generate 9 sketch styles from photos (5 B&W + 4 color).
Usage: python datasets/dataset_builder/sketch_generator.py
"""
import os
import sys
import json
import logging
import random
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# B&W STYLES (return uint8 H×W grayscale)
# ─────────────────────────────────────────────────────────────────

def pencil_sketch(img_gray: np.ndarray) -> np.ndarray:
    """Style 0: Pencil Sketch — multi-scale DoG + cross-hatch grain."""
    g = img_gray.astype(np.float32)
    b1 = cv2.GaussianBlur(g, (0, 0), 1.0)
    b2 = cv2.GaussianBlur(g, (0, 0), 3.0)
    dog = b1 - 0.95 * b2
    dog = np.clip(dog + 128, 0, 255)
    kernel_45 = np.zeros((9, 9), np.float32)
    for i in range(9):
        kernel_45[i, i] = 1.0 / 9
    hatch = cv2.filter2D(g, -1, kernel_45)
    mask = dog / 255.0
    result = mask * dog + (1 - mask) * hatch
    grain = np.random.normal(0, 3, img_gray.shape).astype(np.float32)
    return np.clip(result + grain, 0, 255).astype(np.uint8)


def charcoal_sketch(img_gray: np.ndarray) -> np.ndarray:
    """Style 1: Charcoal — wide DoG + smudge + heavy grain."""
    g = img_gray.astype(np.float32)
    b1 = cv2.GaussianBlur(g, (0, 0), 2.0)
    b2 = cv2.GaussianBlur(g, (0, 0), 8.0)
    dog = b1 - 0.9 * b2
    dog = np.clip(dog * 1.5 + 128, 0, 255)
    smudge = cv2.GaussianBlur(dog, (7, 1), 0)
    smudge = cv2.GaussianBlur(smudge, (1, 5), 0)
    result = cv2.addWeighted(dog, 0.6, smudge, 0.4, 0)
    grain = np.random.normal(0, 8, img_gray.shape).astype(np.float32)
    return np.clip(result + grain, 0, 255).astype(np.uint8)


def ink_drawing(img_gray: np.ndarray) -> np.ndarray:
    """Style 2: Ink Drawing — sharp DoG + organic bleed."""
    g = img_gray.astype(np.float32)
    b1 = cv2.GaussianBlur(g, (0, 0), 0.5)
    b2 = cv2.GaussianBlur(g, (0, 0), 2.0)
    dog = b1 - 0.98 * b2
    dog = np.clip(dog * 2.0 + 128, 0, 255)
    dark_mask = (dog < 128).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bled = cv2.dilate(dark_mask, kernel, iterations=1)
    result = np.where(bled > 0, dog * 0.7, dog)
    noise = np.random.uniform(-2, 2, img_gray.shape).astype(np.float32)
    return np.clip(result + noise, 0, 255).astype(np.uint8)


def technical_sketch(img_gray: np.ndarray) -> np.ndarray:
    """Style 3: Technical Sketch — fine DoG + directional hatching."""
    g = img_gray.astype(np.float32)
    b1 = cv2.GaussianBlur(g, (0, 0), 0.8)
    b2 = cv2.GaussianBlur(g, (0, 0), 1.8)
    dog = b1 - 0.97 * b2
    dog = np.clip(dog * 3.0 + 128, 0, 255)
    gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    angle = np.arctan2(gy, gx)
    hatch_h = cv2.GaussianBlur(dog, (11, 1), 0)
    hatch_v = cv2.GaussianBlur(dog, (1, 11), 0)
    weight = (np.cos(angle) ** 2).astype(np.float32)
    result = weight * hatch_h + (1 - weight) * hatch_v
    return np.clip(result, 0, 255).astype(np.uint8)


def line_art(img_gray: np.ndarray) -> np.ndarray:
    """Style 4: Line Art — soft confident DoG lines with jitter."""
    g = img_gray.astype(np.float32)
    b1 = cv2.GaussianBlur(g, (0, 0), 1.0)
    b2 = cv2.GaussianBlur(g, (0, 0), 4.0)
    dog = b1 - 0.92 * b2
    dog = np.clip(dog * 4.0 + 128, 0, 255)
    dark = np.clip(255 - dog, 0, 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    thick = cv2.dilate(dark, kernel, iterations=1)
    result = 255 - thick.astype(np.float32)
    jitter = np.random.normal(0, 1.5, img_gray.shape).astype(np.float32)
    return np.clip(result + jitter, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────
# COLOR STYLES (return uint8 H×W×3 RGB)
# ─────────────────────────────────────────────────────────────────

def oil_painting(img_rgb: np.ndarray) -> np.ndarray:
    """Style 5: Oil Painting — stylization + saturation boost + impasto."""
    try:
        stylized = cv2.stylization(img_rgb, sigma_s=60, sigma_r=0.45)
    except (cv2.error, AttributeError):
        # Fallback if contrib not available
        stylized = img_rgb.copy()
        for _ in range(3):
            stylized = cv2.bilateralFilter(stylized, 9, 60, 60)
    hsv = cv2.cvtColor(stylized, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)
    rich = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    stroke_blur = cv2.GaussianBlur(rich, (7, 3), 0)
    result = cv2.addWeighted(rich, 0.65, stroke_blur, 0.35, 0)
    gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    edges = cv2.Laplacian(gray, cv2.CV_64F)
    edges_norm = np.clip(edges * 0.15, -20, 20).astype(np.int16)
    for c in range(3):
        result[:, :, c] = np.clip(
            result[:, :, c].astype(np.int16) + edges_norm, 0, 255
        ).astype(np.uint8)
    return result


def colored_pencil(img_rgb: np.ndarray) -> np.ndarray:
    """Style 6: Colored Pencil — bilateral + hatch + edge darkening."""
    smooth = cv2.bilateralFilter(img_rgb, d=9, sigmaColor=75, sigmaSpace=75)
    hsv = cv2.cvtColor(smooth, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.75, 0, 255)
    muted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    gray = cv2.cvtColor(muted, cv2.COLOR_RGB2GRAY).astype(np.float32)
    k1 = np.zeros((7, 7), np.float32)
    np.fill_diagonal(k1, 1.0 / 7)
    k2 = np.zeros((7, 7), np.float32)
    for i in range(7):
        k2[i, 6 - i] = 1.0 / 7
    hatch = cv2.filter2D(gray, -1, k1) * 0.5 + cv2.filter2D(gray, -1, k2) * 0.5
    hatch_norm = hatch / 255.0
    for c in range(3):
        ch = muted[:, :, c].astype(np.float32)
        muted[:, :, c] = np.clip(ch * (0.85 + 0.15 * hatch_norm), 0, 255).astype(np.uint8)
    edges = cv2.Canny(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY), 30, 80)
    edges_dilated = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    edge_mask = (edges_dilated / 255.0 * 0.4).astype(np.float32)
    for c in range(3):
        muted[:, :, c] = np.clip(
            muted[:, :, c].astype(np.float32) * (1 - edge_mask), 0, 255
        ).astype(np.uint8)
    grain = np.random.normal(0, 4, img_rgb.shape).astype(np.float32)
    return np.clip(muted.astype(np.float32) + grain, 0, 255).astype(np.uint8)


def crayon(img_rgb: np.ndarray) -> np.ndarray:
    """Style 7: Crayon — k-means quantization + vivid saturation + wax texture."""
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, 12, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    quantized = centers[labels.flatten()].reshape(img_rgb.shape).astype(np.uint8)
    hsv = cv2.cvtColor(quantized, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.6, 0, 255)
    vivid = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    wax = np.random.uniform(-12, 12, img_rgb.shape).astype(np.float32)
    result = np.clip(vivid.astype(np.float32) + wax, 0, 255).astype(np.uint8)
    result = cv2.GaussianBlur(result, (3, 3), 0)
    paper = np.random.normal(0, 6, img_rgb.shape).astype(np.float32)
    return np.clip(result.astype(np.float32) + paper, 0, 255).astype(np.uint8)


def watercolor(img_rgb: np.ndarray) -> np.ndarray:
    """Style 8: Watercolor — edge-preserving smooth + pigment bleed + granulation."""
    try:
        smooth = cv2.edgePreservingFilter(img_rgb, flags=1, sigma_s=50, sigma_r=0.4)
    except (cv2.error, AttributeError):
        # Fallback if contrib not available
        smooth = img_rgb.copy()
        for _ in range(5):
            smooth = cv2.bilateralFilter(smooth, 9, 50, 50)
    hsv = cv2.cvtColor(smooth, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    bright_mask = (sat > 80).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bleed_mask = cv2.dilate(bright_mask, kernel, iterations=2)
    bleed_float = bleed_mask.astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(smooth, (11, 11), 0)
    result = (
        smooth.astype(np.float32) * (1 - 0.3 * bleed_float[:, :, None])
        + blurred.astype(np.float32) * 0.3 * bleed_float[:, :, None]
    )
    result = result.astype(np.uint8)
    hsv2 = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv2[:, :, 1] = np.clip(hsv2[:, :, 1] * 0.85, 0, 255)
    result = cv2.cvtColor(hsv2.astype(np.uint8), cv2.COLOR_HSV2RGB)
    grain = np.random.normal(0, 5, img_rgb.shape).astype(np.float32)
    result = np.clip(result.astype(np.float32) + grain, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 100)
    edges_soft = cv2.GaussianBlur(edges, (3, 3), 0)
    edge_mask = (edges_soft / 255.0 * 0.35).astype(np.float32)
    for c in range(3):
        result[:, :, c] = np.clip(
            result[:, :, c].astype(np.float32) * (1 - edge_mask), 0, 255
        ).astype(np.uint8)
    return result


# ─────────────────────────────────────────────────────────────────
# STYLE REGISTRY
# ─────────────────────────────────────────────────────────────────

# (style_index, file_suffix, is_color)
STYLE_REGISTRY = [
    (0, 'style0_pencil',       False),
    (1, 'style1_charcoal',     False),
    (2, 'style2_ink',          False),
    (3, 'style3_technical',    False),
    (4, 'style4_lineart',      False),
    (5, 'style5_oil',          True),
    (6, 'style6_pencilcolor',  True),
    (7, 'style7_crayon',       True),
    (8, 'style8_watercolor',   True),
]

BW_FUNCS = [pencil_sketch, charcoal_sketch, ink_drawing, technical_sketch, line_art]
COLOR_FUNCS = [oil_painting, colored_pencil, crayon, watercolor]


def generate_all_sketches(photo_path: Path, output_dir: Path) -> List[Dict]:
    """
    Generate all 9 sketch styles for one photo.
    B&W styles saved as grayscale PNG.
    Color styles saved as RGB PNG.
    Skips generation if output PNG already exists (resume support).
    Returns list of manifest entry dicts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    photo_path = Path(photo_path)

    # Build expected output paths first to check if all already exist
    expected = []
    for style_idx, suffix, is_color in STYLE_REGISTRY:
        out_path = output_dir / f'{photo_path.stem}_{suffix}.png'
        expected.append((style_idx, suffix, is_color, out_path))

    all_exist = all(op.exists() for _, _, _, op in expected)

    # Only load the image if at least one style needs generating
    img_bgr = None
    img_rgb = None
    img_gray = None
    if not all_exist:
        img_bgr = cv2.imread(str(photo_path))
        if img_bgr is None:
            raise ValueError(f'Cannot read image: {photo_path}')
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    entries = []
    for style_idx, suffix, is_color, out_path in expected:
        try:
            if not out_path.exists():
                if is_color:
                    func_idx = style_idx - 5
                    result = COLOR_FUNCS[func_idx](img_rgb)
                    # save RGB as BGR for cv2
                    cv2.imwrite(str(out_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
                else:
                    func_idx = style_idx
                    result = BW_FUNCS[func_idx](img_gray)
                    cv2.imwrite(str(out_path), result)
            entries.append({
                'photo': str(photo_path),
                'sketch': str(out_path),
                'style': style_idx,
                'color': is_color,
            })
        except Exception as e:
            log.warning(f'Style {style_idx} failed for {photo_path.name}: {e}')
    return entries


def build_sketch_dataset(
    photos_dir: Path,
    sketches_dir: Path,
    manifest_path: Path,
) -> int:
    photos_dir = Path(photos_dir)
    sketches_dir = Path(sketches_dir)
    sketches_dir.mkdir(parents=True, exist_ok=True)

    photo_files = sorted(
        p for p in photos_dir.glob('*')
        if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}
    )
    log.info(f'Found {len(photo_files)} photos in {photos_dir}')

    all_entries = []
    iterable = tqdm(photo_files, desc=f'Generating {sketches_dir.name}') if tqdm else photo_files
    for photo_path in iterable:
        entries = generate_all_sketches(photo_path, sketches_dir)
        all_entries.extend(entries)

    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w') as f:
        json.dump(all_entries, f)
    log.info(f'Manifest: {manifest_path} ({len(all_entries)} pairs)')
    return len(all_entries)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )
    base = Path('datasets')
    splits = {
        'train': (base / 'coco/processed/train', base / 'sketches/train'),
        'val':   (base / 'coco/processed/val',   base / 'sketches/val'),
        'test':  (base / 'coco/processed/test',  base / 'sketches/test'),
    }
    for split, (photos_dir, sketches_dir) in splits.items():
        manifest_path = base / 'sketches' / f'{split}_manifest.json'
        total = build_sketch_dataset(photos_dir, sketches_dir, manifest_path)
        print(f'{split}: {total} pairs generated.')
    print('Done.')
