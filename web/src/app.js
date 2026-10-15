import { preprocessImage } from './preprocess.js';
import { loadModel, runInference } from './inference.js';
import { postprocessOutput } from './postprocess.js';
import { downloadCanvas } from './download.js';

const MODEL_URL = './models/sketch_int8.onnx';

// DOM references
const uploadArea     = document.getElementById('upload-area');
const fileInput      = document.getElementById('file-input');
const originalCanvas = document.getElementById('original-canvas');
const sketchCanvas   = document.getElementById('sketch-canvas');
const generateBtn    = document.getElementById('generate-btn');
const downloadBtn    = document.getElementById('download-btn');
const spinner        = document.getElementById('spinner');
const spinnerText    = document.getElementById('spinner-text');
const errorMsg       = document.getElementById('error-msg');
const styleSelect    = document.getElementById('style-select');
const statTime       = document.getElementById('stat-time');
const statModel      = document.getElementById('stat-model');
const statStatus     = document.getElementById('stat-status');
const statStyle      = document.getElementById('stat-style');

let loadedImage = null;
let modelLoaded = false;

// ── Helpers ──────────────────────────────────────────────────────────────────

function showError(msg) {
    errorMsg.textContent = msg;
    errorMsg.style.display = 'block';
}

function clearError() {
    errorMsg.style.display = 'none';
}

function showSpinner(text = 'Running inference...') {
    spinnerText.textContent = text;
    spinner.classList.add('active');
}

function hideSpinner() {
    spinner.classList.remove('active');
}

function setStatus(text) {
    statStatus.textContent = text;
}

async function getModelSizeMB(url) {
    try {
        const res = await fetch(url, { method: 'HEAD' });
        const cl = res.headers.get('content-length');
        if (cl) return (parseInt(cl, 10) / (1024 * 1024)).toFixed(1);
        // Fallback: full fetch just for size
        const fullRes = await fetch(url);
        const buf = await fullRes.arrayBuffer();
        return (buf.byteLength / (1024 * 1024)).toFixed(1);
    } catch {
        return '?';
    }
}

// ── Image display ─────────────────────────────────────────────────────────────

function displayOriginal(img) {
    const ctx = originalCanvas.getContext('2d');
    ctx.clearRect(0, 0, 256, 256);
    ctx.drawImage(img, 0, 0, 256, 256);
}

function clearSketch() {
    const ctx = sketchCanvas.getContext('2d');
    ctx.clearRect(0, 0, 256, 256);
}

// ── File handling ─────────────────────────────────────────────────────────────

function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) {
        showError('Please upload a valid image file (JPG, PNG, or WEBP).');
        return;
    }
    clearError();
    clearSketch();
    downloadBtn.disabled = true;

    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            loadedImage = img;
            displayOriginal(img);
            generateBtn.disabled = !modelLoaded;
            setStatus(modelLoaded ? 'Ready' : 'Waiting for model...');
        };
        img.onerror = () => showError('Failed to decode image.');
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

// ── Upload events ─────────────────────────────────────────────────────────────

uploadArea.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    if (e.target.files[0]) handleFile(e.target.files[0]);
});

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
});

// ── Generate ──────────────────────────────────────────────────────────────────

generateBtn.addEventListener('click', async () => {
    if (!loadedImage) {
        showError('Please upload an image first.');
        return;
    }
    clearError();
    generateBtn.disabled = true;
    downloadBtn.disabled = true;
    showSpinner('Running inference...');

    try {
        const inputData = preprocessImage(loadedImage);
        const styleId = parseInt(styleSelect.value, 10);
        const { output, inferenceTime } = await runInference(inputData, styleId);
        const imageData = postprocessOutput(output);

        const ctx = sketchCanvas.getContext('2d');
        ctx.putImageData(imageData, 0, 0);

        statTime.textContent = `${inferenceTime.toFixed(0)} ms`;
        const styleNames = ['Pencil Sketch','Charcoal','Ink Drawing','Technical Sketch','Line Art','Oil Painting','Colored Pencil','Crayon','Watercolor'];
        statStyle.textContent = styleNames[styleId] || String(styleId);
        statStatus.textContent = `Style: ${styleNames[styleId] || styleId} | Done ✓`;
        downloadBtn.disabled = false;
    } catch (err) {
        console.error(err);
        showError(`Inference failed: ${err.message}`);
        setStatus('Error');
    } finally {
        generateBtn.disabled = false;
        hideSpinner();
    }
});

// ── Download ──────────────────────────────────────────────────────────────────

downloadBtn.addEventListener('click', () => {
    downloadCanvas(sketchCanvas, 'sketch.png');
});

// ── Model loading ─────────────────────────────────────────────────────────────

async function initModel() {
    setStatus('Loading model...');
    showSpinner('Loading AI model...');
    try {
        await loadModel(MODEL_URL);
        modelLoaded = true;

        const sizeMB = await getModelSizeMB(MODEL_URL);
        statModel.textContent = `${sizeMB} MB`;
        setStatus('Model ready');

        if (loadedImage) {
            generateBtn.disabled = false;
        }
    } catch (err) {
        console.error('Model load failed:', err);
        statModel.textContent = 'Load failed';
        setStatus('Model failed to load');
        showError(
            'Failed to load AI model. Make sure sketch_int8.onnx is in web/public/models/. ' +
            err.message
        );
    } finally {
        hideSpinner();
    }
}

initModel();
