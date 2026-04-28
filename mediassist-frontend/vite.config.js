import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/**
 * Dev:
 *  - Frontend calls /api/...
 *  - Vite proxies to FastAPI at http://127.0.0.1:8000
 *
 * Prod:
 *  - Set VITE_API_BASE to your deployed backend base (example: https://your-domain.com/api)
 *  - No proxy is used in production builds.
 */
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        secure: false,
        // Optional: if you ever run backend under a different prefix, map it here.
        // rewrite: (path) => path.replace(/^\/api/, "/api"),
      },
    },
  },
});
