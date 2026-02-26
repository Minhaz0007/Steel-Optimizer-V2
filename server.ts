import express from "express";
import { fileURLToPath } from "url";
import path from "path";
import { spawn } from "child_process";
import { createInterface } from "readline";
import { writeFileSync, unlinkSync, existsSync } from "fs";
import { tmpdir } from "os";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ---------------------------------------------------------------------------
// Helper: write raw CSV text to a temporary file and return the path.
// ---------------------------------------------------------------------------
function writeCsvToTmp(csvText: string): string {
  const tmpPath = path.join(
    tmpdir(),
    `steel_${Date.now()}_${Math.random().toString(36).slice(2)}.csv`
  );
  writeFileSync(tmpPath, csvText, "utf8");
  return tmpPath;
}

function cleanupTmp(p: string | null): void {
  if (p && existsSync(p)) { try { unlinkSync(p); } catch { /* ignore */ } }
}

async function startServer() {
  const app = express();
  const PORT = Number(process.env.PORT) || 3000;
  const ARTIFACT_DIR = process.env.ARTIFACT_DIR || "ml/artifacts";

  // Global body size limit — must be set before any route/middleware that reads the body.
  // A 27k-row dataset can easily exceed the default 100kb Express limit.
  app.use(express.json({ limit: "200mb" }));
  app.use(express.urlencoded({ limit: "200mb", extended: true }));

  // ── /api/train ─────────────────────────────────────────────────────────────
  // POST <csv text body>  Content-Type: text/csv
  // Sending raw CSV (not JSON) keeps the payload ~10-20x smaller, avoiding
  // proxy body-size limits (nginx default: 1 MB) that would cause 413 errors.
  //
  // Streams SSE events from Python stdout:
  //   { type:"progress", label:string, pct:number }
  //   { type:"result", regressors:[...], classifiers:[...], ... }
  //   { type:"error", message:string }
  app.post("/api/train", express.text({ type: "text/csv", limit: "200mb" }), async (req, res) => {
    const csvText = typeof req.body === "string" ? req.body.trim() : "";
    if (!csvText) {
      res.status(400).json({ error: "Missing or empty CSV body." });
      return;
    }

    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    res.setHeader("X-Accel-Buffering", "no");
    res.flushHeaders();

    let csvPath: string | null = null;
    let clientGone = false;
    req.on("close", () => { clientGone = true; });

    const send = (payload: object) => {
      if (clientGone) return;
      try { res.write(`data: ${JSON.stringify(payload)}\n\n`); } catch { clientGone = true; }
    };

    try {
      send({ type: "progress", label: "Writing dataset to temp file…", pct: 2 });
      csvPath = writeCsvToTmp(csvText);

      const py = spawn(
        "python3",
        ["-m", "ml.train_server", csvPath, ARTIFACT_DIR],
        { cwd: __dirname, env: { ...process.env, PYTHONUNBUFFERED: "1" } }
      );

      const rl = createInterface({ input: py.stdout });
      rl.on("line", (line) => {
        const t = line.trim();
        if (!t) return;
        try { send(JSON.parse(t)); }
        catch { process.stderr.write(`[python] ${t}\n`); }
      });

      py.stderr.on("data", (chunk: Buffer) => process.stderr.write(chunk));

      py.on("close", (code) => {
        cleanupTmp(csvPath);
        if (!clientGone && code !== 0)
          send({ type: "error", message: `Python exited with code ${code}.` });
        res.end();
      });

      py.on("error", (err) => {
        cleanupTmp(csvPath);
        send({ type: "error", message: `Cannot start Python: ${err.message}` });
        res.end();
      });
    } catch (err: any) {
      cleanupTmp(csvPath);
      send({ type: "error", message: err?.message ?? "Server error." });
      res.end();
    }
  });

  // ── /api/recommend ─────────────────────────────────────────────────────────
  // POST { context: Record<string, number> }
  // Returns JSON RecommendationResult
  app.post("/api/recommend", async (req, res) => {
    const { context } = req.body ?? {};
    if (!context || typeof context !== "object") {
      res.status(400).json({ error: "Missing context object." });
      return;
    }
    try {
      const py = spawn(
        "python3",
        ["-m", "ml.predict_cli", JSON.stringify(context), ARTIFACT_DIR],
        { cwd: __dirname, env: { ...process.env, PYTHONUNBUFFERED: "1" } }
      );
      let stdout = "";
      py.stdout.on("data", (d: Buffer) => { stdout += d.toString(); });
      py.stderr.on("data", (d: Buffer) => process.stderr.write(d));
      py.on("close", () => {
        try {
          const result = JSON.parse(stdout.trim());
          result?.error ? res.status(500).json({ error: result.error }) : res.json(result);
        } catch {
          res.status(500).json({ error: "Failed to parse Python output.", raw: stdout.slice(0, 500) });
        }
      });
      py.on("error", (err) => res.status(500).json({ error: `Cannot start Python: ${err.message}` }));
    } catch (err: any) {
      res.status(500).json({ error: err?.message ?? "Server error." });
    }
  });

  // ── /api/models/status ─────────────────────────────────────────────────────
  app.get("/api/models/status", (_req, res) => {
    const af = (name: string) => path.join(__dirname, ARTIFACT_DIR, name);
    const regressors = ["yield_pct", "steel_output_tons", "energy_cost_usd", "production_cost_usd", "scrap_rate_pct"]
      .map((t) => ({ target: t, trained: existsSync(af(`${t}_lgbm.pkl`)) }));
    const classifiers = ["quality_grade_pass", "rework_required"]
      .map((t) => ({ target: t, trained: existsSync(af(`${t}_catboost.pkl`)) }));
    const anomaly = existsSync(af("anomaly_iforest.pkl"));
    const forecaster = existsSync(af("forecast_feature_cols.pkl"));
    res.json({
      regressors, classifiers, anomaly, forecaster,
      allTrained: regressors.every((r) => r.trained) && classifiers.every((c) => c.trained) && anomaly && forecaster,
    });
  });

  // ── /api/health ────────────────────────────────────────────────────────────
  app.get("/api/health", (_req, res) => res.json({ status: "ok" }));

  // ── Frontend ───────────────────────────────────────────────────────────────
  if (process.env.NODE_ENV === "production") {
    app.use(express.static(path.join(__dirname, "dist")));
    app.get("*", (_req, res) => res.sendFile(path.join(__dirname, "dist", "index.html")));
  } else {
    const { createServer: createViteServer } = await import("vite");
    const vite = await createViteServer({ server: { middlewareMode: true }, appType: "spa" });
    app.use(vite.middlewares);
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server → http://localhost:${PORT}  |  artifacts: ${path.resolve(ARTIFACT_DIR)}`);
  });
}

startServer();
