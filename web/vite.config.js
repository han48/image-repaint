import { defineConfig } from 'vite';
import { viteStaticCopy } from 'vite-plugin-static-copy';

export default defineConfig({
  // Exclude onnxruntime-web from dependency pre-bundling
  // so its WASM files are served correctly
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  plugins: [
    viteStaticCopy({
      targets: [
        {
          // Copy WASM & mjs files from onnxruntime-web dist to output root
          src: 'node_modules/onnxruntime-web/dist/*.{wasm,mjs,js}',
          dest: '.',
        },
      ],
    }),
  ],
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          ort: ['onnxruntime-web'],
        },
      },
    },
  },
});
