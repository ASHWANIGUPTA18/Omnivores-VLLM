import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The backend is reached through Vite's dev proxy so the browser only ever
// talks to one origin. That keeps CORS out of the development loop entirely,
// and means VITE_API_URL only has to be set for a deployed build.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://localhost:8000', changeOrigin: true },
      '/health': { target: 'http://localhost:8000', changeOrigin: true },
      '/config': { target: 'http://localhost:8000', changeOrigin: true },
      '/ws': { target: 'ws://localhost:8000', ws: true },
    },
  },
  build: { outDir: 'dist', sourcemap: true },
})
