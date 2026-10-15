"""
Image quality checker and dataset filter.
"""
import logging
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


def is_valid_image(
    path,
    min_laplacian: float = 100.0,
    min_mean: float = 30.0,
    max_mean: float = 225.0
) -> bool:
    """
    Check if an image is valid (not corrupt, not blurry, not over/under-exposed).
    Returns True if valid.
    """
    try:
        img = cv2.imread(str(path))
        if img is None:
            return False
        if img.ndim < 2 or img.size == 0:
            return False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Blur check via Laplacian variance
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < min_laplacian:
            return False
        # Exposure check
        mean_val = float(gray.mean())
        if mean_val < min_mean or mean_val > max_mean:
            return False
        return True
    except Exception as e:
        log.debug(f"Validation error for {path}: {e}")
        return False


def _check_single(args):
    path, min_laplacian, min_mean, max_mean = args
    return str(path), is_valid_image(path, min_laplacian, min_mean, max_mean)


def filter_dataset(
    input_dir,
    output_dir,
    min_laplacian: float = 100.0,
    min_mean: float = 30.0,
    max_mean: float = 225.0,
    extensions=None
) -> dict:
    """
    Filter images from input_dir to output_dir.
    Returns stats dict with counts: total, valid, invalid, copied.
    """
    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.webp'}

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_paths = [p for p in input_dir.rglob('*') if p.suffix.lower() in extensions]
    log.info(f"Found {len(all_paths)} images in {input_dir}")

    args_list = [(p, min_laplacian, min_mean, max_mean) for p in all_paths]
    n_workers = min(cpu_count(), 8)

    valid_paths = []
    invalid_count = 0

    with Pool(n_workers) as pool:
        for path_str, valid in pool.imap(_check_single, args_list, chunksize=32):
            if valid:
                valid_paths.append(Path(path_str))
            else:
                invalid_count += 1

    log.info(f"Valid: {len(valid_paths)}, Invalid/filtered: {invalid_count}")

    copied = 0
    for src in valid_paths:
        dst = output_dir / src.name
        if not dst.exists():
            import shutil
            shutil.copy2(src, dst)
            copied += 1

    stats = {
        'total': len(all_paths),
        'valid': len(valid_paths),
        'invalid': invalid_count,
        'copied': copied
    }
    log.info(f"Filter stats: {stats}")
    return stats


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Filter dataset images by quality")
    parser.add_argument('input_dir', help='Input directory')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--min-laplacian', type=float, default=100.0)
    parser.add_argument('--min-mean', type=float, default=30.0)
    parser.add_argument('--max-mean', type=float, default=225.0)
    args = parser.parse_args()
    stats = filter_dataset(args.input_dir, args.output_dir,
                           args.min_laplacian, args.min_mean, args.max_mean)
    print(stats)
