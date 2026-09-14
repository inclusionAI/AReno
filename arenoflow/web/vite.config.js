import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  build: { outDir: '../static', emptyOutDir: true },
  server: {
    proxy: { '/api': 'http://127.0.0.1:8787', '/brand': 'http://127.0.0.1:8787' },
  },
});
