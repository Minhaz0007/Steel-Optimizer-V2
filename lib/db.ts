import pg from 'pg';

const { Pool } = pg;

let pool: pg.Pool | null = null;

export function getPool(): pg.Pool | null {
  if (!process.env.DATABASE_URL) return null;
  if (!pool) {
    pool = new Pool({
      connectionString: process.env.DATABASE_URL,
      ssl: { rejectUnauthorized: false },
      max: 5,
    });
  }
  return pool;
}

export async function initDB(): Promise<void> {
  const db = getPool();
  if (!db) {
    console.log('No DATABASE_URL set — running without database (localStorage only)');
    return;
  }

  await db.query(`
    CREATE TABLE IF NOT EXISTS datasets (
      id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      upload_date TEXT NOT NULL,
      row_count INTEGER NOT NULL,
      column_count INTEGER NOT NULL,
      data JSONB NOT NULL,
      mappings JSONB NOT NULL,
      health_score REAL NOT NULL,
      created_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS trained_models (
      id TEXT PRIMARY KEY,
      type TEXT NOT NULL,
      metrics JSONB NOT NULL,
      model_json JSONB,
      feature_importance JSONB,
      config JSONB NOT NULL,
      created_at TIMESTAMPTZ DEFAULT NOW()
    );
  `);

  console.log('Database schema ready');
}
