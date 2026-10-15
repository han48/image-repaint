"""
Training script for StyleConditionedUNet (9 styles).
Usage:
    python training/train.py \\
        --train-manifest datasets/sketches/train_manifest_5k.json \\
        --val-manifest datasets/sketches/val_manifest.json \\
        --epochs 20 --batch-size 16
"""
import csv
import time
import random
import logging
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from pytorch_msssim import ssim as ssim_fn
    HAS_MSSSIM = True
except ImportError:
    HAS_MSSSIM = False
    print('WARNING: pytorch-msssim not found. Using L1 loss only.')

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from training.model import StyleConditionedUNet, count_parameters
from training.dataset import SketchDataset
from datasets.dataset_builder.augment import get_train_transforms, get_val_transforms

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)


def combined_loss(pred, target, l1_w=0.7, ssim_w=0.3):
    l1 = nn.functional.l1_loss(pred, target)
    if HAS_MSSSIM:
        ssim_val = ssim_fn(pred, target, data_range=1.0, size_average=True)
        return l1_w * l1 + ssim_w * (1.0 - ssim_val)
    return l1


def train_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    n = 0
    it = tqdm(loader, desc='Train', leave=False) if tqdm else loader
    for photos, sketches, style_ids in it:
        photos, sketches, style_ids = (
            photos.to(device), sketches.to(device), style_ids.to(device)
        )
        optimizer.zero_grad()
        pred = model(photos, style_ids)
        loss = combined_loss(pred, sketches)
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
        if tqdm and hasattr(it, 'set_postfix'):
            it.set_postfix(loss=f'{loss.item():.4f}')
    return total / max(n, 1)


def val_epoch(model, loader, device):
    model.eval()
    total_loss = total_mae = total_ssim = 0.0
    n = 0
    with torch.no_grad():
        it = tqdm(loader, desc='Val', leave=False) if tqdm else loader
        for photos, sketches, style_ids in it:
            photos, sketches, style_ids = (
                photos.to(device), sketches.to(device), style_ids.to(device)
            )
            pred = model(photos, style_ids)
            total_loss += combined_loss(pred, sketches).item()
            total_mae += torch.mean(torch.abs(pred - sketches)).item()
            if HAS_MSSSIM:
                total_ssim += ssim_fn(pred, sketches, data_range=1.0, size_average=True).item()
            n += 1
    n = max(n, 1)
    return total_loss / n, total_mae / n, total_ssim / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-manifest', default='datasets/sketches/train_manifest.json')
    parser.add_argument('--val-manifest', default='datasets/sketches/val_manifest.json')
    parser.add_argument('--output', default='models')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--no-pretrained', action='store_true')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--val-batch-size', type=int, default=None,
                        help='Batch size for validation (default: same as --batch-size)')
    parser.add_argument('--val-subset', type=int, default=None,
                        help='Randomly subsample this many val samples per epoch (default: full val set)')
    parser.add_argument('--train-subset', type=int, default=None,
                        help='Randomly subsample this many train samples per epoch (default: full train set)')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cpu')
    log.info(f'Device: {device}')

    train_ds = SketchDataset(args.train_manifest, transform=get_train_transforms())
    val_ds = SketchDataset(args.val_manifest, transform=get_val_transforms())
    log.info(f'Train: {len(train_ds)}, Val: {len(val_ds)}')

    val_batch_size = args.val_batch_size if args.val_batch_size else args.batch_size

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=val_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False
    )

    model = StyleConditionedUNet(num_styles=9, pretrained=not args.no_pretrained).to(device)
    log.info(f'Parameters: {count_parameters(model):,}')

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 0
    best_val = float('inf')
    patience_ctr = 0

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val = ckpt.get('best_val_loss', float('inf'))
        patience_ctr = ckpt.get('patience_counter', 0)
        log.info(f'Resumed from epoch {start_epoch}')

    log_path = output_dir / 'training_log.csv'
    csv_file = open(log_path, 'a', newline='')
    csv_writer = csv.writer(csv_file)
    if start_epoch == 0:
        csv_writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_mae', 'val_ssim', 'lr'])

    log.info('Training...')
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Build (possibly subsampled) train loader for this epoch
        if args.train_subset and args.train_subset < len(train_ds):
            train_indices = random.sample(range(len(train_ds)), args.train_subset)
            train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
            epoch_train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, sampler=train_sampler,
                num_workers=args.workers, pin_memory=False, drop_last=True
            )
        else:
            epoch_train_loader = train_loader
        tl = train_epoch(model, epoch_train_loader, optimizer, device)

        # Build (possibly subsampled) val loader for this epoch
        if args.val_subset and args.val_subset < len(val_ds):
            subset_indices = random.sample(range(len(val_ds)), args.val_subset)
            subset_sampler = torch.utils.data.SubsetRandomSampler(subset_indices)
            epoch_val_loader = DataLoader(
                val_ds, batch_size=val_batch_size, sampler=subset_sampler,
                num_workers=args.workers, pin_memory=False
            )
        else:
            epoch_val_loader = val_loader
        vl, vm, vs = val_epoch(model, epoch_val_loader, device)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0
        log.info(f'Epoch {epoch+1:3d}/{args.epochs} | train={tl:.4f} | val={vl:.4f} | mae={vm:.4f} | ssim={vs:.4f} | lr={lr:.2e} | {elapsed:.1f}s')
        csv_writer.writerow([epoch+1, tl, vl, vm, vs, lr])
        csv_file.flush()

        if vl < best_val:
            best_val = vl
            patience_ctr = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val,
                'val_mae': vm,
                'val_ssim': vs,
                'patience_counter': patience_ctr,
                'num_styles': 9,
            }, output_dir / 'best_model.pth')
            log.info(f'  -> Best model saved (val={vl:.4f})')
        else:
            patience_ctr += 1
            log.info(f'  -> No improvement ({patience_ctr}/{args.patience})')

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val,
                'patience_counter': patience_ctr,
                'num_styles': 9,
            }, output_dir / f'checkpoint_epoch{epoch+1}.pth')

        if patience_ctr >= args.patience:
            log.info('Early stopping.')
            break

    csv_file.close()
    log.info(f'Done. Best val_loss={best_val:.4f}')


if __name__ == '__main__':
    main()
