"""
Evaluation script for trained sketch generator.
Usage:
    python training/evaluate.py --checkpoint models/best_model.pth \\
                                --manifest datasets/sketches/test_manifest.json
"""
import json
import logging
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from pytorch_msssim import SSIM
    HAS_MSSSIM = True
except ImportError:
    HAS_MSSSIM = False

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='models/best_model.pth')
    parser.add_argument('--manifest', default='datasets/sketches/test_manifest.json')
    parser.add_argument('--output-dir', default='models/eval_samples')
    parser.add_argument('--n-samples', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")

    # Load model
    from training.model import MobileNetV3UNet
    from training.dataset import SketchDataset
    from datasets.dataset_builder.augment import get_val_transforms

    model = MobileNetV3UNet(pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    log.info(f"Model loaded from {args.checkpoint}")

    ds = SketchDataset(args.manifest, transform=get_val_transforms())
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    ssim_metric = None
    if HAS_MSSSIM:
        from pytorch_msssim import SSIM as SSIMMetric
        ssim_metric = SSIMMetric(data_range=1.0, size_average=True, channel=1).to(device)

    all_mae = []
    all_ssim = []
    all_psnr = []
    sample_count = 0

    iterable = tqdm(loader, desc='Evaluating') if tqdm else loader
    with torch.no_grad():
        for photos, sketches in iterable:
            photos = photos.to(device)
            sketches = sketches.to(device)
            preds = model(photos)

            mae = torch.mean(torch.abs(preds - sketches)).item()
            all_mae.append(mae)

            if ssim_metric is not None:
                ssim_val = ssim_metric(preds, sketches).item()
                all_ssim.append(ssim_val)

            # PSNR per batch
            preds_np = (preds.squeeze(1).cpu().numpy() * 255).astype(np.uint8)
            sketches_np = (sketches.squeeze(1).cpu().numpy() * 255).astype(np.uint8)
            for p, g in zip(preds_np, sketches_np):
                psnr = compute_psnr(p, g)
                all_psnr.append(psnr)

            # Save side-by-side comparison images
            if sample_count < args.n_samples:
                # photos: (B,3,H,W) in [-1,1]
                photos_vis = ((photos.cpu().numpy().transpose(0, 2, 3, 1) * 0.5 + 0.5) * 255).astype(np.uint8)
                for i in range(min(len(preds_np), args.n_samples - sample_count)):
                    photo_vis = photos_vis[i]
                    sketch_gt = np.stack([sketches_np[i]] * 3, axis=-1)
                    sketch_pred = np.stack([preds_np[i]] * 3, axis=-1)
                    comparison = np.hstack([photo_vis, sketch_gt, sketch_pred])
                    comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
                    out_path = output_dir / f"sample_{sample_count:04d}.jpg"
                    cv2.imwrite(str(out_path), comparison_bgr)
                    sample_count += 1

    mean_mae = np.mean(all_mae)
    mean_ssim = np.mean(all_ssim) if all_ssim else 0.0
    mean_psnr = np.mean(all_psnr)

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"MAE:  {mean_mae:.4f}")
    print(f"SSIM: {mean_ssim:.4f}")
    print(f"PSNR: {mean_psnr:.2f} dB")
    print("="*50)
    print(f"Saved {sample_count} comparison images to {output_dir}")

    metrics = {
        'mae': float(mean_mae),
        'ssim': float(mean_ssim),
        'psnr': float(mean_psnr),
        'n_samples_evaluated': len(all_psnr),
        'n_comparison_images': sample_count,
    }
    metrics_path = Path('models/eval_metrics.json')
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Metrics saved to {metrics_path}")


if __name__ == '__main__':
    evaluate()
