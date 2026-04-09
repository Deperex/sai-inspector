import { defineConfig } from "vite";
import { resolve } from "node:path";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { spawn, type ChildProcess } from "node:child_process";
import sirv from "sirv";

type ExportEntry = {
  env_name: string;
  task_id: number;
  benchmark_name: string;
  instruction: string;
  manifest_url: string;
};

function collectExportEntries(exportsDir: string): ExportEntry[] {
  const entries: ExportEntry[] = [];

  function walk(dir: string) {
    for (const name of readdirSync(dir)) {
      const fullPath = resolve(dir, name);
      const stat = statSync(fullPath);
      if (stat.isDirectory()) {
        walk(fullPath);
        continue;
      }
      if (name !== "manifest.json") continue;
      const manifest = JSON.parse(readFileSync(fullPath, "utf-8")) as {
        env_name: string;
        task_id: number;
        benchmark_name: string;
        instruction: string;
      };
      const relPath = fullPath.slice(exportsDir.length).replaceAll("\\", "/");
      entries.push({
        env_name: manifest.env_name,
        task_id: manifest.task_id,
        benchmark_name: manifest.benchmark_name,
        instruction: manifest.instruction,
        manifest_url: `/exports${relPath}`,
      });
    }
  }

  walk(exportsDir);
  entries.sort((a, b) => {
    if (a.env_name !== b.env_name) return a.env_name.localeCompare(b.env_name);
    return a.task_id - b.task_id;
  });
  return entries;
}

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
        const backendScript = resolve(__dirname, "../scripts/run_backend.sh");
        let backendProcess: ChildProcess | null = null;

        const stopBackend = () => {
          if (!backendProcess || backendProcess.killed) return;
          backendProcess.kill("SIGTERM");
          backendProcess = null;
        };

        if (!backendProcess) {
          backendProcess = spawn("bash", [backendScript], {
            cwd: resolve(__dirname, ".."),
            stdio: "inherit",
          });
          backendProcess.on("exit", (code, signal) => {
            console.log(`[backend] exited with code=${code} signal=${signal}`);
            backendProcess = null;
          });
          process.once("exit", stopBackend);
          process.once("SIGINT", stopBackend);
          process.once("SIGTERM", stopBackend);
        }

        server.middlewares.use((req, res, next) => {
          if (req.url?.startsWith("/exports")) {
            res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
            res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
            res.setHeader("Cross-Origin-Resource-Policy", "cross-origin");
          }
          next();
        });

        server.middlewares.use("/__exports_index", (_req, res) => {
          const entries = collectExportEntries(exportsDir);
          res.setHeader("content-type", "application/json");
          res.end(JSON.stringify({ exports: entries }));
        });

        server.middlewares.use("/exports", sirv(exportsDir, { dev: true }));

        server.httpServer?.once("close", stopBackend);
      },
    },
  ],

  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/healthz": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
    fs: {
      allow: [resolve(__dirname, "..")],
    },
  },
});
