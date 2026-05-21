import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/queue':   { target: 'http://localhost:8000', changeOrigin: true },
      '/patient': { target: 'http://localhost:8000', changeOrigin: true },
      '/review':  { target: 'http://localhost:8000', changeOrigin: true },
      '/upload':  { target: 'http://localhost:8000', changeOrigin: true },
      '/uploads': { target: 'http://localhost:8000', changeOrigin: true },
      '/gradcam': { target: 'http://localhost:8000', changeOrigin: true },
      '/ws':      { target: 'ws://localhost:8000',  ws: true, changeOrigin: true },
    },
  },
})
