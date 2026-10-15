/**
 * Preprocess an image element for model inference.
 * Returns Float32Array of shape [1, 3, 256, 256] in CHW format.
 * Normalized: mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5] -> range [-1, 1]
 */
export function preprocessImage(imageElement) {
    const SIZE = 256;
    const canvas = document.createElement('canvas');
    canvas.width = SIZE;
    canvas.height = SIZE;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imageElement, 0, 0, SIZE, SIZE);
    const imageData = ctx.getImageData(0, 0, SIZE, SIZE);
    const data = imageData.data; // RGBA, length = SIZE*SIZE*4

    const float32 = new Float32Array(1 * 3 * SIZE * SIZE);
    const nPixels = SIZE * SIZE;

    for (let i = 0; i < nPixels; i++) {
        const r = data[i * 4]     / 255.0;
        const g = data[i * 4 + 1] / 255.0;
        const b = data[i * 4 + 2] / 255.0;
        // Channel-first layout: [R plane, G plane, B plane]
        float32[0 * nPixels + i] = (r - 0.5) / 0.5;  // R
        float32[1 * nPixels + i] = (g - 0.5) / 0.5;  // G
        float32[2 * nPixels + i] = (b - 0.5) / 0.5;  // B
    }
    return float32;
}
