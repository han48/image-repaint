import * as ort from 'onnxruntime-web';

let session = null;

export async function loadModel(modelUrl) {
    ort.env.wasm.wasmPaths = './';
    session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
    });
    return session;
}

/**
 * Run inference.
 * @param {Float32Array} inputData - shape [1, 3, 256, 256]
 * @param {number} styleId - integer 0-8
 * @returns {{output: Float32Array, inferenceTime: number}}
 */
export async function runInference(inputData, styleId = 0) {
    if (!session) throw new Error('Model not loaded.');
    const photoTensor = new ort.Tensor('float32', inputData, [1, 3, 256, 256]);
    const styleTensor = new ort.Tensor('int64', BigInt64Array.from([BigInt(styleId)]), [1]);
    const feeds = { input: photoTensor, style_id: styleTensor };
    const start = performance.now();
    const results = await session.run(feeds);
    const elapsed = performance.now() - start;
    return { output: results.output.data, inferenceTime: elapsed };
}

export function getSession() { return session; }
