import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The frontend calls the backend via relative /api/* paths. In dev, Vite proxies
// those to the backend on :8000, so the browser sees same-origin requests and no
// CORS config is needed on the backend. In production this is handled by whatever
// serves the built assets (reverse proxy / ingress) routing /api to the backend.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
