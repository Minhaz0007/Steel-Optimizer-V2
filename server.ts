import express from "express";
import { fileURLToPath } from "url";
import path from "path";
import { trainModels } from "./src/lib/ml-engine.ts";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

async function startServer() {
  const app = express();
  // Cloud Run injects PORT=8080; fall back to 3000 for local dev
  const PORT = Number(process.env.PORT) || 3000;

  // ── /api/train — Server-Sent Events training endpoint ──────────────────
  // Accepts: POST { data: any[], config: TrainingConfig }
  // Streams:  data: {"type":"progress","label":"...","pct":0-100}
  //           data: {"type":"result","models":[...]}
  //           data: {"type":"error","message":"..."}
  app.post(
    "/api/train",
    express.json({ limit: "50mb" }),
    async (req, res) => {
      const { data, config } = req.body ?? {};

      if (!Array.isArray(data) || !config) {
        res.status(400).json({ error: "Missing data or config in request body." });
        return;
      }

      // Set up SSE — keep the connection open and stream events
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");
      res.setHeader("X-Accel-Buffering", "no"); // disable nginx/proxy buffering
      res.flushHeaders();

      let clientGone = false;
      req.on("close", () => {
        clientGone = true;
      });

      const send = (payload: object) => {
        if (clientGone) return;
        try {
          res.write(`data: ${JSON.stringify(payload)}\n\n`);
        } catch {
          clientGone = true;
        }
      };

      try {
        const models = await trainModels(
          data,
          config,
          (label: string, pct: number) => {
            send({ type: "progress", label, pct });
          }
        );

        // modelInstance holds a live JS object that cannot be JSON-serialised.
        // Predictions reconstruct it on demand from modelJSON, so strip it here.
        const serialised = models.map((m) => ({ ...m, modelInstance: null }));
        send({ type: "result", models: serialised });
      } catch (err: any) {
        send({ type: "error", message: err?.message ?? "Training failed." });
      }

      res.end();
    }
  );

  // ── Health check ────────────────────────────────────────────────────────
  app.get("/api/health", (_req, res) => {
    res.json({ status: "ok" });
  });

  // ── Frontend ────────────────────────────────────────────────────────────
  if (process.env.NODE_ENV === "production") {
    // Serve Vite-built static assets
    app.use(express.static(path.join(__dirname, "dist")));
    // SPA fallback — return index.html for all non-API routes
    app.get("*", (_req, res) => {
      res.sendFile(path.join(__dirname, "dist", "index.html"));
    });
  } else {
    // Dynamic import keeps vite out of the production bundle/container
    const { createServer: createViteServer } = await import("vite");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
