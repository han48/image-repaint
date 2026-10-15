"""
Export StyleConditionedUNet to ONNX.
Usage: python export/export_onnx.py
"""
import sys
import logging
import argparse
from pathlib import Path

import torch
import onnx
import onnxruntime as ort
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def export_onnx(checkpoint_path='models/best_model.pth', output_path='models/sketch.onnx', opset=17):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from training.model import StyleConditionedUNet, count_parameters

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    ckpt = torch.load(ckpt_path, map_location='cpu')
    num_styles = ckpt.get('num_styles', 9)
    model = StyleConditionedUNet(num_styles=num_styles, pretrained=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    log.info(f'Loaded model: {count_parameters(model):,} params, num_styles={num_styles}')

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_photo = torch.randn(1, 3, 256, 256)
    dummy_style = torch.zeros(1, dtype=torch.long)

    log.info(f'Exporting to {output_path} (opset {opset})...')
    torch.onnx.export(
        model,
        (dummy_photo, dummy_style),
        str(output_path),
        opset_version=opset,
        input_names=['input', 'style_id'],
        output_names=['output'],
        dynamic_axes={
            'input':    {0: 'batch'},
            'style_id': {0: 'batch'},
            'output':   {0: 'batch'},
        },
        do_constant_folding=True,
        export_params=True,
        external_data=False,  # embed weights in single .onnx file
    )

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    log.info('ONNX check passed.')

    sess = ort.InferenceSession(str(output_path), providers=['CPUExecutionProvider'])
    out = sess.run(['output'], {'input': dummy_photo.numpy(), 'style_id': dummy_style.numpy()})
    log.info(f'ORT test passed. Output shape: {out[0].shape}')

    mb = output_path.stat().st_size / (1024 * 1024)
    print(f'\nONNX Export:')
    print(f'  Inputs: input(1,3,256,256) + style_id(1,) int64')
    print(f'  Output: (1,3,256,256)')
    print(f'  Opset: {opset}')
    print(f'  Size: {mb:.1f} MB -> {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='models/best_model.pth')
    parser.add_argument('--output', default='models/sketch.onnx')
    parser.add_argument('--opset', type=int, default=17)
    args = parser.parse_args()
    export_onnx(args.checkpoint, args.output, args.opset)
