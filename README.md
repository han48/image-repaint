# Sketch AI - Chuyển Ảnh Thành Tranh Vẽ Tay

Hệ thống AI chuyển ảnh chụp thành tranh vẽ tay, chạy hoàn toàn trên trình duyệt.

## Tổng quan

Dự án này xây dựng một mô hình AI có khả năng:
- Chuyển ảnh chụp thành tranh vẽ tay (5 phong cách khác nhau)
- Huấn luyện bằng PyTorch trên dữ liệu sinh tổng hợp từ COCO 2017
- Export sang ONNX (opset 17) với INT8 quantization
- Chạy hoàn toàn trên trình duyệt bằng ONNX Runtime Web
- Không cần backend, không cần GPU server khi deploy
- Deploy được lên GitHub Pages hoặc Cloudflare Pages

## Kiến trúc

```
┌─────────────────────────────────────────────────┐
│              TRAINING PIPELINE                  │
│                                                 │
│  COCO 2017 (118k ảnh)                           │
│       │                                         │
│       ▼                                         │
│  Chọn 15,000 ảnh → Lọc chất lượng              │
│       │                                         │
│       ▼                                         │
│  Sinh 5 sketch styles × 15,000 = 75,000 cặp    │
│       │                                         │
│       ▼                                         │
│  MobileNetV3 Encoder + U-Net Decoder            │
│  Input: (B, 3, 256, 256)                        │
│  Output: (B, 1, 256, 256)                       │
│       │                                         │
│       ▼                                         │
│  Export ONNX (opset 17) → INT8 Quantize         │
│  15-20 MB → 5-8 MB                              │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│              BROWSER INFERENCE                  │
│                                                 │
│  Upload ảnh (JPG/PNG/WEBP)                      │
│       │                                         │
│       ▼                                         │
│  Preprocess: resize 256×256, normalize [-1,1]   │
│       │                                         │
│       ▼                                         │
│  ONNX Runtime Web (WebAssembly)                 │
│       │                                         │
│       ▼                                         │
│  Postprocess: [0,1] → grayscale ImageData       │
│       │                                         │
│       ▼                                         │
│  Hiển thị sketch + Download PNG                 │
└─────────────────────────────────────────────────┘
```

## Cấu trúc dự án

```
image-repaint/
├── datasets/
│   ├── coco/
│   │   ├── download_coco.py       # Tải COCO 2017
│   │   └── prepare_dataset.py     # Chuẩn bị 15,000 ảnh
│   └── dataset_builder/
│       ├── clean.py               # Lọc ảnh chất lượng
│       ├── sketch_generator.py    # Sinh 5 phong cách sketch
│       └── augment.py             # Augmentation pipeline
├── training/
│   ├── model.py                   # MobileNetV3 + U-Net
│   ├── dataset.py                 # PyTorch Dataset
│   ├── train.py                   # Training script
│   └── evaluate.py                # Evaluation script
├── export/
│   ├── export_onnx.py             # Export sang ONNX
│   └── quantize.py                # INT8 Quantization
├── models/                        # Lưu model sau training
├── web/                           # Frontend
│   ├── public/models/             # ONNX model cho browser
│   ├── src/
│   │   ├── app.js                 # Logic chính
│   │   ├── preprocess.js          # Tiền xử lý ảnh
│   │   ├── inference.js           # ONNX Runtime Web
│   │   ├── postprocess.js         # Hậu xử lý kết quả
│   │   └── download.js            # Tải ảnh về
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
└── README.md
```

## Cài đặt môi trường

### Yêu cầu hệ thống
- Python 3.11+
- CUDA (khuyến nghị, không bắt buộc)
- Node.js 18+

### Python dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python albumentations pytorch-msssim onnx onnxruntime tqdm numpy Pillow
# Nếu cần cv2.ximgproc cho Style 4:
pip install opencv-contrib-python
```

## Quy trình huấn luyện

### Bước 1: Tải COCO 2017

```bash
python datasets/coco/download_coco.py
```

> ⚠️ File zip ~18 GB. Cần kết nối internet tốt và ổ cứng đủ dung lượng (~40 GB tổng).

### Bước 2: Chuẩn bị dataset

```bash
python datasets/coco/prepare_dataset.py
```

Script sẽ:
- Chọn ngẫu nhiên 15,000 ảnh từ COCO
- Lọc ảnh mờ, quá tối, quá sáng, lỗi
- Resize về 256×256 RGB
- Chia train/val/test: 12,000 / 1,500 / 1,500
- Lưu manifest vào `datasets/coco/splits.json`

### Bước 3: Sinh sketch dataset

```bash
# Train set
python datasets/dataset_builder/sketch_generator.py \
    --photos datasets/coco/processed/train \
    --output datasets/sketches/train

# Val set
python datasets/dataset_builder/sketch_generator.py \
    --photos datasets/coco/processed/val \
    --output datasets/sketches/val

# Test set
python datasets/dataset_builder/sketch_generator.py \
    --photos datasets/coco/processed/test \
    --output datasets/sketches/test
```

Kết quả: 75,000 cặp ảnh (15,000 × 5 phong cách).

### Bước 4: Huấn luyện mô hình

```bash
python training/train.py \
    --manifest datasets/sketches/train_manifest.json \
    --val-manifest datasets/sketches/val_manifest.json \
    --output models/ \
    --epochs 100 \
    --batch-size 32
```

Tiếp tục từ checkpoint:

```bash
python training/train.py \
    --manifest datasets/sketches/train_manifest.json \
    --val-manifest datasets/sketches/val_manifest.json \
    --resume models/checkpoint_epoch50.pth
```

### Bước 5: Đánh giá mô hình

```bash
python training/evaluate.py \
    --checkpoint models/best_model.pth \
    --manifest datasets/sketches/test_manifest.json
```

Kết quả lưu tại:
- `models/eval_metrics.json` — MAE, SSIM, PSNR
- `models/eval_samples/` — Ảnh so sánh (photo | sketch_gt | sketch_pred)

### Bước 6: Export ONNX

```bash
python export/export_onnx.py --checkpoint models/best_model.pth
```

Output: `models/sketch.onnx` (~15-20 MB)

### Bước 7: Quantize INT8

```bash
python export/quantize.py
```

Output:
- `models/sketch_int8.onnx` (~5-8 MB)
- `web/public/models/sketch_int8.onnx` (tự động copy)

## Chạy Web App

### Development

```bash
cd web
npm install
npm run dev
```

Mở trình duyệt tại `http://localhost:5173`

### Build production

```bash
cd web
npm run build
```

Output tĩnh tại `web/dist/`.

## Deploy

### GitHub Pages

1. Push code lên GitHub repository
2. Trong Settings → Pages → Source: chọn `gh-pages` branch hoặc `/docs` folder
3. Copy nội dung `web/dist/` vào thư mục deploy
4. Hoặc dùng GitHub Actions để tự động build và deploy:

```yaml
# .github/workflows/deploy.yml
name: Deploy to GitHub Pages
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 18
      - run: cd web && npm install && npm run build
      - uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{secrets.GITHUB_TOKEN}}
          publish_dir: ./web/dist
```

### Cloudflare Pages

1. Kết nối GitHub repository với Cloudflare Pages
2. Cấu hình build:
   - **Build command**: `cd web && npm install && npm run build`
   - **Build output directory**: `web/dist`
3. Deploy!

> ⚠️ **Lưu ý**: File `sketch_int8.onnx` (~5-8 MB) phải có trong `web/public/models/` trước khi build.

## Kiến trúc mô hình

```
Input (B, 3, 256, 256)
       │
       ▼
┌─────────────────┐
│  MobileNetV3    │  Encoder (pretrained ImageNet)
│  Encoder        │
│                 │
│  enc0: 16ch     │  128×128
│  enc1: 16ch     │  128×128
│  enc2: 24ch     │  64×64
│  enc3: 48ch     │  32×32
│  enc4: 96ch     │  16×16
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Bridge: 128ch  │  16×16
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  U-Net Decoder  │  Skip connections từ encoder
│                 │
│  dec4: 64ch     │  32×32  (+ enc3 skip)
│  dec3: 32ch     │  64×64  (+ enc2 skip)
│  dec2: 24ch     │  128×128 (+ enc1 skip)
│  dec1: 16ch     │  256×256 (+ enc0 skip)
│  dec0: 16ch     │  256×256
└────────┬────────┘
         │
         ▼
Output Conv (1ch) + Sigmoid
         │
         ▼
Output (B, 1, 256, 256) ∈ [0, 1]
```

**Tham số**: ~2-4M (< 5M mục tiêu)

## Loss Function

```
Loss = 0.7 × L1Loss + 0.3 × SSIMLoss
```

- **L1Loss**: đảm bảo độ chính xác pixel-level
- **SSIMLoss (1 - SSIM)**: đảm bảo chất lượng cấu trúc và độ sáng

## Các phong cách Sketch

| Style | Tên | Mô tả |
|-------|-----|-------|
| 1 | Pencil Sketch | Giả lập bút chì qua Color Dodge |
| 2 | Line Art | Nét vẽ sắc nét qua Canny Edge Detection |
| 3 | Ink Drawing | Nét mực qua Adaptive Threshold |
| 4 | Technical Sketch | Phác thảo kỹ thuật qua Thin Edge Extraction |
| 5 | Charcoal Sketch | Phong cách than chì với texture noise |

## Performance Targets

| Thiết bị | Inference Time |
|----------|---------------|
| Desktop  | < 150 ms |
| Mobile (tầm trung) | < 1 giây |
| Model load | < 3 giây |

## Mẹo huấn luyện

1. **GPU**: Nếu có CUDA, training ~80-100 epochs mất khoảng 4-6 giờ với batch size 32.
2. **CPU only**: Giảm batch size xuống 8-16, tăng learning rate scheduler patience.
3. **Mixed precision**: Script tự động bật AMP nếu CUDA available.
4. **Early stopping**: Patience=10 epoch, tránh overfitting.
5. **Kiểm tra sketch quality**: Chạy `python training/evaluate.py` để xem ảnh so sánh trước khi export.
6. **cv2.ximgproc**: Style 4 (Technical Sketch) cần `opencv-contrib-python`. Nếu không cài, script sẽ dùng morphological fallback.

## Troubleshooting

**Lỗi WASM khi chạy web app:**
```
Cross-Origin-Opener-Policy / Cross-Origin-Embedder-Policy không được set
```
Solution: Vite dev server đã được cấu hình headers COOP/COEP. Nếu deploy lên CDN, thêm headers tương tự.

**Model không load được:**
- Kiểm tra file `web/public/models/sketch_int8.onnx` tồn tại
- Chạy `python export/quantize.py` để copy model vào web/public/models/

**Training loss không giảm:**
- Kiểm tra dataset manifest JSON có đúng paths không
- Thử giảm learning rate: `--lr 5e-5`
- Kiểm tra augmentation không làm hỏng dữ liệu

## License

MIT License
