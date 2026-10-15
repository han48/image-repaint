/**
 * Convert model output Float32Array [1, 3, H, W] to ImageData.
 * Channel-first layout: R plane first, then G, then B.
 * Values in [0,1].
 */
export function postprocessOutput(outputData, width = 256, height = 256) {
    const imageData = new ImageData(width, height);
    const pixelCount = width * height;
    for (let i = 0; i < pixelCount; i++) {
        imageData.data[i * 4]     = Math.round(Math.min(1, Math.max(0, outputData[i])) * 255);                     // R
        imageData.data[i * 4 + 1] = Math.round(Math.min(1, Math.max(0, outputData[pixelCount + i])) * 255);        // G
        imageData.data[i * 4 + 2] = Math.round(Math.min(1, Math.max(0, outputData[2 * pixelCount + i])) * 255);    // B
        imageData.data[i * 4 + 3] = 255;
    }
    return imageData;
}
