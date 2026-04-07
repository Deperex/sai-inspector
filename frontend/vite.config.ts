import { defineConfig } from "vite";
import { resolve } from "node:path";
import sirv from "sirv";

export default defineConfig({
  publicDir: "public",

  optimizeDeps: {
    exclude: ["@mujoco/mujoco"],
  },

  plugins: [
    {
      name: "serve-exports",
      configureServer(server) {
        const exportsDir = resolve(__dirname, "../exports");

        server.middlewares.use((req, res, next) => {
          if (req.url?.startsWith("/exports")) {
            res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
            res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
            res.setHeader("Cross-Origin-Resource-Policy", "cross-origin");
          }
          next();
        });

        server.middlewares.use("/exports", sirv(exportsDir, { dev: true }));
      },
    },
  ],

  server: {
    host: "0.0.0.0",
    port: 5173,
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
    fs: {
      allow: [resolve(__dirname, "..")],
    },
  },
});