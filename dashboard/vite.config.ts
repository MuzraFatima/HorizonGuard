import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The Python API serves the built dashboard; in development, Vite proxies /api to it.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8765" },
  },
  build: {
    target: "es2020",
    cssMinify: true,
    minify: "esbuild",
    sourcemap: false,
    rollupOptions: {
      output: {
        // keep the framework in its own long-cached chunk; pages split per route
        manualChunks: { react: ["react", "react-dom", "react-router-dom"] },
      },
    },
  },
});
