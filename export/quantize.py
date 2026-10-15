"""
INT8 quantization of ONNX model.
Usage: python export/quantize.py
"""
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def quantize_model(
    input_path='models/sketch.onnx',
    output_path='models/sketch_int8.onnx',
    web_output_path='web/public/models/sketch_int8.onnx',
):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from onnxruntime.quantization.shape_inference import quant_pre_process
    import onnxruntime as ort
    import numpy as np
    import tempfile, os

    input_path = Path(input_path)
    output_path = Path(output_path)
    web_out = Path(web_output_path)

    if not input_path.exists():
        raise FileNotFoundError(f'Input not found: {input_path}. Run export_onnx.py first.')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    web_out.parent.mkdir(parents=True, exist_ok=True)

    before_mb = input_path.stat().st_size / (1024 * 1024)
    log.info(f'Input: {input_path} ({before_mb:.1f} MB)')

    # Pre-process: strip existing value_info that confuses shape inference,
    # then attempt quant_pre_process; fall back to stripped model if that fails too.
    import onnx
    stripped_path = Path(str(input_path).replace('.onnx', '_stripped.onnx'))
    model_stripped = onnx.load(str(input_path))
    del model_stripped.graph.value_info[:]  # remove intermediate shape annotations
    onnx.save(model_stripped, str(stripped_path))
    log.info(f'Stripped value_info, saved to {stripped_path}')

    preprocessed_path = Path(str(input_path).replace('.onnx', '_preprocessed.onnx'))
    log.info(f'Pre-processing model for quantization...')
    try:
        quant_pre_process(str(stripped_path), str(preprocessed_path), skip_optimization=False)
        quant_input = preprocessed_path
        log.info(f'Pre-processing done: {preprocessed_path}')
    except Exception as e:
        log.warning(f'quant_pre_process failed ({e}), trying with skip_optimization=True')
        try:
            quant_pre_process(str(stripped_path), str(preprocessed_path), skip_optimization=True)
            quant_input = preprocessed_path
        except Exception as e2:
            log.warning(f'quant_pre_process still failed ({e2}), quantizing stripped model directly')
            quant_input = stripped_path

    quantize_dynamic(
        str(quant_input),
        str(output_path),
        weight_type=QuantType.QUInt8,
        extra_options={'SkipInferenceSessionValidation': True},
    )

    # Clean up temp files
    for tmp in [preprocessed_path, stripped_path]:
        if tmp.exists() and tmp != input_path:
            tmp.unlink(missing_ok=True)

    after_mb = output_path.stat().st_size / (1024 * 1024)
    shutil.copy2(output_path, web_out)

    # Verify quantized model runs
    sess = ort.InferenceSession(str(output_path), providers=['CPUExecutionProvider'])
    dummy_photo = np.random.randn(1, 3, 256, 256).astype(np.float32)
    dummy_style = np.zeros((1,), dtype=np.int64)
    out = sess.run(['output'], {'input': dummy_photo, 'style_id': dummy_style})
    log.info(f'Quantized ORT test passed. Output shape: {out[0].shape}')

    print(f'\nQuantization Summary:')
    print(f'  Input:   {before_mb:.1f} MB')
    print(f'  Output:  {after_mb:.1f} MB ({(1-after_mb/before_mb)*100:.0f}% reduction)')
    print(f'  Web:     {web_out}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='models/sketch.onnx')
    parser.add_argument('--output', default='models/sketch_int8.onnx')
    parser.add_argument('--web-output', default='web/public/models/sketch_int8.onnx')
    args = parser.parse_args()
    quantize_model(args.input, args.output, args.web_output)
