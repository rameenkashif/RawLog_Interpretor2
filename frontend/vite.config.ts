import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    proxy: {
      // Forward API calls to the FastAPI backend during local dev so the
      // frontend can just call relative paths like "/wells".
      "/wells": "http://127.0.0.1:8000",
      "/dashboard": "http://127.0.0.1:8000",
      "/chat": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
      "/seismic": "http://127.0.0.1:8000",
      "/tie": "http://127.0.0.1:8000",
      "/api": "http://127.0.0.1:8000",
    },
  },
});
