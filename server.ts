import express from "express";
import { fileURLToPath } from "url";
import path from "path";
import { initDB, getPool } from "./lib/db.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

async function startServer() {
  const app = express();
  // Cloud Run injects PORT=8080; fall back to 3000 for local dev
  const PORT = Number(process.env.PORT) || 3000;

  app.use(express.json({ limit: "50mb" }));

  // Health check
  app.get("/api/health", (_req, res) => {
    res.json({ status: "ok" });
  });

  // ── Datasets ────────────────────────────────────────────────────────────
  app.get("/api/datasets", async (_req, res) => {
    const db = getPool();
    if (!db) return res.json([]);
    try {
      const result = await db.query(
        "SELECT id, name, upload_date, row_count, column_count, data, mappings, health_score FROM datasets ORDER BY created_at DESC"
      );
      res.json(
        result.rows.map((r) => ({
          id: r.id,
          name: r.name,
          uploadDate: r.upload_date,
          rowCount: r.row_count,
          columnCount: r.column_count,
          data: r.data,
          mappings: r.mappings,
          healthScore: r.health_score,
        }))
      );
    } catch (e) {
      res.status(500).json({ error: String(e) });
    }
  });

  app.post("/api/datasets", async (req, res) => {
    const db = getPool();
    if (!db) return res.json({ ok: true });
    const { id, name, uploadDate, rowCount, columnCount, data, mappings, healthScore } = req.body;
    try {
      await db.query(
        `INSERT INTO datasets (id, name, upload_date, row_count, column_count, data, mappings, health_score)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
         ON CONFLICT (id) DO UPDATE SET
           name = EXCLUDED.name,
           data = EXCLUDED.data,
           mappings = EXCLUDED.mappings,
           health_score = EXCLUDED.health_score`,
        [id, name, uploadDate, rowCount, columnCount, JSON.stringify(data), JSON.stringify(mappings), healthScore]
      );
      res.json({ ok: true });
    } catch (e) {
      res.status(500).json({ error: String(e) });
    }
  });

  app.put("/api/datasets/:id/mappings", async (req, res) => {
    const db = getPool();
    if (!db) return res.json({ ok: true });
    try {
      await db.query("UPDATE datasets SET mappings = $1 WHERE id = $2", [
        JSON.stringify(req.body.mappings),
        req.params.id,
      ]);
      res.json({ ok: true });
    } catch (e) {
      res.status(500).json({ error: String(e) });
    }
  });

  app.delete("/api/datasets/:id", async (req, res) => {
    const db = getPool();
    if (!db) return res.json({ ok: true });
    try {
      await db.query("DELETE FROM datasets WHERE id = $1", [req.params.id]);
      res.json({ ok: true });
    } catch (e) {
      res.status(500).json({ error: String(e) });
    }
  });

  // ── Trained Models ───────────────────────────────────────────────────────
  app.get("/api/models", async (_req, res) => {
    const db = getPool();
    if (!db) return res.json([]);
    try {
      const result = await db.query(
        "SELECT id, type, metrics, model_json, feature_importance, config FROM trained_models ORDER BY created_at DESC"
      );
      res.json(
        result.rows.map((r) => ({
          id: r.id,
          type: r.type,
          metrics: r.metrics,
          modelJSON: r.model_json,
          featureImportance: r.feature_importance,
          config: r.config,
          modelInstance: null,
        }))
      );
    } catch (e) {
      res.status(500).json({ error: String(e) });
    }
  });

  app.post("/api/models", async (req, res) => {
    const db = getPool();
    if (!db) return res.json({ ok: true });
    const { id, type, metrics, modelJSON, featureImportance, config } = req.body;
    try {
      await db.query(
        `INSERT INTO trained_models (id, type, metrics, model_json, feature_importance, config)
         VALUES ($1, $2, $3, $4, $5, $6)
         ON CONFLICT (id) DO NOTHING`,
        [
          id,
          type,
          JSON.stringify(metrics),
          JSON.stringify(modelJSON),
          JSON.stringify(featureImportance),
          JSON.stringify(config),
        ]
      );
      res.json({ ok: true });
    } catch (e) {
      res.status(500).json({ error: String(e) });
    }
  });

  app.delete("/api/models/:id", async (req, res) => {
    const db = getPool();
    if (!db) return res.json({ ok: true });
    try {
      await db.query("DELETE FROM trained_models WHERE id = $1", [req.params.id]);
      res.json({ ok: true });
    } catch (e) {
      res.status(500).json({ error: String(e) });
    }
  });

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

  await initDB();
  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
