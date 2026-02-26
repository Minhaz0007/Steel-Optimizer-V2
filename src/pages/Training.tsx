import { useState, useRef, useEffect } from 'react';
import Papa from 'papaparse';
import { useStore } from '@/store/useStore';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { toast } from 'sonner';
import type { TrainingSession, RegressionModelResult, ClassifierResult, ShapEntry } from '@/lib/ml-engine';
import { motion } from 'framer-motion';
import {
  Loader2, Trophy, ArrowRight, TrendingUp, AlertCircle,
  ChevronDown, ChevronUp, CheckCircle2, Cpu, Zap, ShieldCheck,
} from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts';
import { useNavigate } from 'react-router-dom';
import { v4 as uuidv4 } from 'uuid';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const REGRESSION_TARGETS = [
  { key: 'yield_pct',           label: 'Yield %',            icon: '🏭', direction: 'maximize' },
  { key: 'steel_output_tons',   label: 'Steel Output (t)',   icon: '⚙️', direction: 'maximize' },
  { key: 'energy_cost_usd',     label: 'Energy Cost ($)',    icon: '⚡', direction: 'minimize' },
  { key: 'production_cost_usd', label: 'Production Cost ($)', icon: '💰', direction: 'minimize' },
  { key: 'scrap_rate_pct',      label: 'Scrap Rate %',       icon: '♻️', direction: 'minimize' },
];

const CLASSIFIER_TARGETS = [
  { key: 'quality_grade_pass', label: 'Quality Grade Pass', icon: '⭐', direction: 'maximize pass rate' },
  { key: 'rework_required',    label: 'Rework Required',    icon: '🔄', direction: 'minimize' },
];

const SHAP_COLORS: Record<string, string> = {
  Controllable: '#3b82f6',
  Context:      '#8b5cf6',
  Engineered:   '#10b981',
};

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function MetricPill({ label, value, good }: { label: string; value: string; good?: boolean }) {
  return (
    <div className="text-center p-2 bg-background rounded-lg border">
      <div className="text-[10px] text-muted-foreground mb-0.5">{label}</div>
      <div className={`text-base font-bold ${good === true ? 'text-green-600 dark:text-green-400' : good === false ? 'text-amber-600' : ''}`}>
        {value}
      </div>
    </div>
  );
}

function r2Color(r2: number) {
  if (r2 >= 0.9) return 'text-green-600 dark:text-green-400';
  if (r2 >= 0.7) return 'text-amber-600 dark:text-amber-400';
  return 'text-red-500';
}

function aucColor(auc: number) {
  if (auc >= 0.85) return 'text-green-600 dark:text-green-400';
  if (auc >= 0.7) return 'text-amber-600 dark:text-amber-400';
  return 'text-red-500';
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function Training() {
  const currentDataset = useStore((state) => state.currentDataset);
  const setTrainingSession = useStore((state) => state.setTrainingSession);
  const navigate = useNavigate();

  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressLabel, setProgressLabel] = useState('');
  const [session, setSession] = useState<TrainingSession | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'regressors' | 'classifiers' | 'shap'>('regressors');
  const [shapTarget, setShapTarget] = useState<string>('');
  const [stepsOpen, setStepsOpen] = useState(true);

  const abortRef = useRef<AbortController | null>(null);
  useEffect(() => () => { abortRef.current?.abort(); }, []);

  const handleTrain = async () => {
    if (!currentDataset) return;

    setIsTraining(true);
    setProgress(2);
    setProgressLabel('Preparing data…');
    setError(null);
    setSession(null);

    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    try {
      // Convert to CSV (not JSON) so the request body is ~10-20x smaller.
      // JSON repeats every column name on every row; CSV only writes it once
      // in the header — this keeps the payload well under proxy size limits
      // (nginx defaults to 1 MB) that cause 413 errors.
      const csvBody = Papa.unparse(currentDataset.data);

      const res = await fetch('/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'text/csv' },
        body: csvBody,
        signal: ctrl.signal,
      });

      if (!res.ok || !res.body) {
        throw new Error(`Server error ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // SSE lines: "data: {...}\n\n"
        const parts = buffer.split('\n\n');
        buffer = parts.pop() ?? '';

        for (const part of parts) {
          const line = part.replace(/^data: /, '').trim();
          if (!line) continue;
          try {
            const msg = JSON.parse(line);
            if (msg.type === 'progress') {
              setProgressLabel(msg.label ?? '');
              setProgress(msg.pct ?? 0);
            } else if (msg.type === 'result') {
              const newSession: TrainingSession = {
                id: uuidv4(),
                datasetId: currentDataset.id,
                trainedAt: new Date().toISOString(),
                rows: msg.rows ?? 0,
                features: msg.features ?? 0,
                regressors: msg.regressors ?? [],
                classifiers: msg.classifiers ?? [],
                anomaly: msg.anomaly ?? { threshold: 0 },
                forecaster: msg.forecaster ?? { targets: [] },
                shap: msg.shap ?? {},
              };
              setSession(newSession);
              setTrainingSession(newSession);
              setProgress(100);
              setProgressLabel('Done!');
              setIsTraining(false);
              if (newSession.shap && Object.keys(newSession.shap).length > 0) {
                setShapTarget(Object.keys(newSession.shap)[0]);
              }
              toast.success(`Training complete — ${newSession.regressors.length + newSession.classifiers.length} models trained`);
            } else if (msg.type === 'error') {
              throw new Error(msg.message ?? 'Training failed.');
            }
          } catch (parseErr: any) {
            if (parseErr.message !== 'Training failed.' && !parseErr.message?.includes('Unexpected token')) {
              throw parseErr;
            } else if (parseErr.message?.startsWith('Training failed') || parseErr.message?.includes('Python') || parseErr.message?.includes('Error')) {
              throw parseErr;
            }
          }
        }
      }
    } catch (err: any) {
      if (err.name === 'AbortError') return;
      const msg = err?.message ?? 'Training failed.';
      setError(msg);
      setIsTraining(false);
      setProgress(0);
      toast.error('Training failed');
    }
  };

  if (!currentDataset) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center space-y-4">
          <h2 className="text-2xl font-bold">No Dataset Selected</h2>
          <p className="text-muted-foreground">Please upload a dataset first.</p>
        </div>
      </div>
    );
  }

  // ── SHAP chart data ────────────────────────────────────────────────────────
  const shapEntries: ShapEntry[] = (shapTarget && session?.shap?.[shapTarget]) ? session.shap[shapTarget] : [];
  const shapChartData = shapEntries.slice(0, 12).map((e) => ({
    name: e.feature.length > 18 ? e.feature.slice(0, 16) + '…' : e.feature,
    fullName: e.feature,
    value: parseFloat(e.mean_abs_shap.toFixed(4)),
    category: e.category,
    color: SHAP_COLORS[e.category] ?? '#94a3b8',
  }));

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">Model Training</h1>
        <p className="text-muted-foreground">
          Trains the full Python ML pipeline: LightGBM regressors, CatBoost classifiers,
          Isolation Forest anomaly detector, temporal forecaster, and SHAP analysis — all in one click.
        </p>
      </div>

      {/* ── What will be trained ─────────────────────────────────────────────── */}
      <Card>
        <button
          type="button"
          className="w-full text-left"
          onClick={() => setStepsOpen((o) => !o)}
        >
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <div>
              <CardTitle className="text-base">What gets trained</CardTitle>
              <CardDescription>7 models across 4 stages — trained automatically on your dataset</CardDescription>
            </div>
            {stepsOpen
              ? <ChevronUp className="h-5 w-5 text-muted-foreground" />
              : <ChevronDown className="h-5 w-5 text-muted-foreground" />}
          </CardHeader>
        </button>

        {stepsOpen && (
          <CardContent className="pt-0">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="space-y-2">
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide flex items-center gap-1">
                  <Cpu className="h-3.5 w-3.5" /> Stage 1 — LightGBM Regressors
                </p>
                {REGRESSION_TARGETS.map((t) => (
                  <div key={t.key} className="flex items-center gap-2 text-sm">
                    <span>{t.icon}</span>
                    <span className="font-medium">{t.label}</span>
                    <span className="text-xs text-muted-foreground ml-auto">{t.direction}</span>
                  </div>
                ))}
              </div>
              <div className="space-y-2">
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide flex items-center gap-1">
                  <Cpu className="h-3.5 w-3.5" /> Stage 1 — CatBoost Classifiers
                </p>
                {CLASSIFIER_TARGETS.map((t) => (
                  <div key={t.key} className="flex items-center gap-2 text-sm">
                    <span>{t.icon}</span>
                    <span className="font-medium">{t.label}</span>
                    <span className="text-xs text-muted-foreground ml-auto">{t.direction}</span>
                  </div>
                ))}
                <div className="mt-3 space-y-1">
                  <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide flex items-center gap-1">
                    <ShieldCheck className="h-3.5 w-3.5" /> Stage 2–4
                  </p>
                  <div className="text-sm text-muted-foreground space-y-0.5">
                    <div>• Bayesian Optimizer (Optuna)</div>
                    <div>• Isolation Forest anomaly detector</div>
                    <div>• Temporal forecaster (LightGBM + lags)</div>
                    <div>• SHAP feature importance analysis</div>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        )}
      </Card>

      {/* ── Train button + dataset info ──────────────────────────────────────── */}
      <Card>
        <CardContent className="pt-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div>
            <p className="font-medium">{currentDataset.name}</p>
            <p className="text-sm text-muted-foreground">
              {currentDataset.rowCount.toLocaleString()} rows · {currentDataset.columnCount} columns
              {session && (
                <span className="ml-2 text-green-600 dark:text-green-400 font-medium">
                  ✓ Last trained {new Date(session.trainedAt).toLocaleTimeString()}
                </span>
              )}
            </p>
          </div>
          <Button
            className="min-w-[180px]"
            size="lg"
            onClick={handleTrain}
            disabled={isTraining}
          >
            {isTraining ? (
              <><Loader2 className="mr-2 h-4 w-4 animate-spin" />Training…</>
            ) : session ? (
              <><Zap className="mr-2 h-4 w-4" />Re-train All Models</>
            ) : (
              <><Cpu className="mr-2 h-4 w-4" />Train All Models</>
            )}
          </Button>
        </CardContent>

        {/* Progress */}
        {isTraining && (
          <CardContent className="pt-0 pb-6 space-y-3">
            <Progress value={progress} className="h-2" />
            <p className="text-sm text-muted-foreground text-center">{progressLabel}</p>
          </CardContent>
        )}
      </Card>

      {/* ── Error ───────────────────────────────────────────────────────────── */}
      {error && (
        <Card className="border-destructive/40 bg-destructive/5">
          <CardContent className="p-4 flex items-start gap-3">
            <AlertCircle className="h-5 w-5 text-destructive flex-shrink-0 mt-0.5" />
            <div>
              <p className="font-medium text-destructive">Training Failed</p>
              <pre className="text-xs text-muted-foreground mt-1 whitespace-pre-wrap">{error}</pre>
            </div>
          </CardContent>
        </Card>
      )}

      {/* ── Results ─────────────────────────────────────────────────────────── */}
      {!isTraining && session && (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-5">

          {/* Summary banner */}
          <Card className="bg-primary/5 border-primary/20">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-primary">
                <Trophy className="h-5 w-5" />
                Training Complete
              </CardTitle>
              <CardDescription>
                {session.regressors.length} regressors + {session.classifiers.length} classifiers ·{' '}
                {session.rows.toLocaleString()} rows · {session.features} features ·{' '}
                Trained {new Date(session.trainedAt).toLocaleString()}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-3">
                {session.anomaly && (
                  <div className="flex items-center gap-1.5 text-sm bg-background rounded-lg border px-3 py-1.5">
                    <ShieldCheck className="h-4 w-4 text-green-600" />
                    <span>Anomaly detector ready</span>
                  </div>
                )}
                {session.forecaster.targets.length > 0 && (
                  <div className="flex items-center gap-1.5 text-sm bg-background rounded-lg border px-3 py-1.5">
                    <TrendingUp className="h-4 w-4 text-blue-600" />
                    <span>Forecaster: {session.forecaster.targets.join(', ')}</span>
                  </div>
                )}
                {Object.keys(session.shap).length > 0 && (
                  <div className="flex items-center gap-1.5 text-sm bg-background rounded-lg border px-3 py-1.5">
                    <CheckCircle2 className="h-4 w-4 text-purple-600" />
                    <span>SHAP analysis complete</span>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Tabs */}
          <div className="flex gap-2">
            {(['regressors', 'classifiers', 'shap'] as const).map((tab) => (
              <Button
                key={tab}
                size="sm"
                variant={activeTab === tab ? 'default' : 'outline'}
                onClick={() => setActiveTab(tab)}
              >
                {tab === 'regressors' ? 'LightGBM Regressors' : tab === 'classifiers' ? 'CatBoost Classifiers' : 'SHAP Importance'}
              </Button>
            ))}
          </div>

          {/* ── Regressors tab ─────────────────────────────────────────────── */}
          {activeTab === 'regressors' && (
            <Card>
              <CardHeader>
                <CardTitle>LightGBM Regressors — Test & CV Metrics</CardTitle>
                <CardDescription>5-fold cross-validation with early stopping. Higher R² = better fit.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="overflow-x-auto rounded-md border">
                  <table className="w-full text-sm">
                    <thead className="bg-muted/50">
                      <tr>
                        {['Target', 'Test R²', 'Test RMSE', 'CV R² (mean)', 'CV RMSE (mean ± std)', 'Best Iter'].map((h) => (
                          <th key={h} className="h-9 px-3 text-left font-medium text-muted-foreground whitespace-nowrap">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {session.regressors.map((r: RegressionModelResult) => (
                        <tr key={r.target} className="border-b last:border-0 hover:bg-muted/30">
                          <td className="px-3 py-2 font-medium whitespace-nowrap">
                            {REGRESSION_TARGETS.find((t) => t.key === r.target)?.icon} {r.target}
                          </td>
                          <td className={`px-3 py-2 font-semibold tabular-nums ${r2Color(r.test_r2)}`}>
                            {(r.test_r2 * 100).toFixed(1)}%
                          </td>
                          <td className="px-3 py-2 tabular-nums">{r.test_rmse.toFixed(4)}</td>
                          <td className="px-3 py-2 tabular-nums">{(r.cv_r2_mean * 100).toFixed(1)}%</td>
                          <td className="px-3 py-2 tabular-nums whitespace-nowrap">
                            {r.cv_rmse_mean.toFixed(4)}
                            <span className="text-muted-foreground"> ±{r.cv_rmse_std.toFixed(4)}</span>
                          </td>
                          <td className="px-3 py-2 tabular-nums text-muted-foreground">{r.best_iteration}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* R² bar chart */}
                <div className="h-[200px] mt-4">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={session.regressors.map((r) => ({
                        name: r.target.replace('_', ' ').replace('_', ' '),
                        'R²': parseFloat((r.test_r2 * 100).toFixed(1)),
                      }))}
                      margin={{ top: 4, right: 16, left: 0, bottom: 4 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                      <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                      <YAxis domain={[0, 100]} unit="%" tick={{ fontSize: 11 }} />
                      <Tooltip formatter={(v: any) => [`${v}%`, 'R²']} />
                      <Bar dataKey="R²" radius={[4, 4, 0, 0]}>
                        {session.regressors.map((_, i) => (
                          <Cell key={i} fill={['#3b82f6', '#8b5cf6', '#f59e0b', '#10b981', '#ef4444'][i % 5]} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          )}

          {/* ── Classifiers tab ────────────────────────────────────────────── */}
          {activeTab === 'classifiers' && (
            <Card>
              <CardHeader>
                <CardTitle>CatBoost Classifiers — Classification Metrics</CardTitle>
                <CardDescription>5-fold stratified CV · Class weighting applied when imbalanced.</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid gap-4">
                  {session.classifiers.map((c: ClassifierResult) => (
                    <div key={c.target} className="p-4 rounded-lg border space-y-3">
                      <p className="font-semibold text-sm">
                        {CLASSIFIER_TARGETS.find((t) => t.key === c.target)?.icon}{' '}
                        {c.target}
                        <span className="ml-2 text-xs text-muted-foreground font-normal">CatBoost</span>
                      </p>
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                        <MetricPill label="Test F1" value={c.test_f1.toFixed(4)} good={c.test_f1 >= 0.8} />
                        <MetricPill label="Test AUC" value={c.test_auc.toFixed(4)} good={c.test_auc >= 0.85} />
                        <MetricPill label="Precision" value={c.test_precision.toFixed(4)} />
                        <MetricPill label="Recall" value={c.test_recall.toFixed(4)} />
                      </div>
                      <div className="grid grid-cols-2 gap-3 text-xs text-muted-foreground">
                        <div className="bg-muted/40 rounded p-2">
                          CV F1: <span className="font-semibold text-foreground">{c.cv_f1_mean.toFixed(4)}</span>
                          <span> ±{c.cv_f1_std.toFixed(4)}</span>
                        </div>
                        <div className="bg-muted/40 rounded p-2">
                          CV AUC: <span className={`font-semibold ${aucColor(c.cv_auc_mean)}`}>{c.cv_auc_mean.toFixed(4)}</span>
                          <span> ±{c.cv_auc_std.toFixed(4)}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* ── SHAP tab ───────────────────────────────────────────────────── */}
          {activeTab === 'shap' && (
            <Card>
              <CardHeader>
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <CardTitle className="flex items-center gap-2">
                      <TrendingUp className="h-5 w-5" />
                      SHAP Feature Importance
                    </CardTitle>
                    <CardDescription>
                      Mean |SHAP| value per feature. Controllable = process levers; Context = uncontrollable conditions.
                    </CardDescription>
                  </div>
                  {/* Target selector */}
                  <div className="flex flex-wrap gap-1.5">
                    {Object.keys(session.shap).map((t) => (
                      <button
                        key={t}
                        onClick={() => setShapTarget(t)}
                        className={`text-xs px-2 py-1 rounded border transition-colors ${
                          shapTarget === t
                            ? 'bg-primary text-primary-foreground border-primary'
                            : 'border-border hover:border-primary/50'
                        }`}
                      >
                        {t}
                      </button>
                    ))}
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                {shapChartData.length > 0 ? (
                  <>
                    <div className="h-[340px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={shapChartData}
                          layout="vertical"
                          margin={{ top: 4, right: 50, left: 8, bottom: 4 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" opacity={0.3} horizontal={false} />
                          <XAxis type="number" tick={{ fontSize: 11 }} />
                          <YAxis dataKey="name" type="category" width={140} tick={{ fontSize: 11 }} />
                          <Tooltip
                            contentStyle={{ backgroundColor: 'hsl(var(--card))', borderColor: 'hsl(var(--border))' }}
                            formatter={(val: any, _: any, props: any) => [
                              val.toFixed(4),
                              `${props.payload.category} feature`,
                            ]}
                            labelFormatter={(label: string) =>
                              shapChartData.find((d) => d.name === label)?.fullName ?? label
                            }
                          />
                          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                            {shapChartData.map((d, i) => (
                              <Cell key={i} fill={d.color} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                    {/* Legend */}
                    <div className="flex gap-4 mt-3 text-xs text-muted-foreground">
                      {Object.entries(SHAP_COLORS).map(([cat, color]) => (
                        <div key={cat} className="flex items-center gap-1.5">
                          <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: color }} />
                          {cat}
                        </div>
                      ))}
                    </div>
                  </>
                ) : (
                  <p className="text-sm text-muted-foreground text-center py-8">
                    {shapTarget ? 'No SHAP data for this target.' : 'Select a target above.'}
                  </p>
                )}
              </CardContent>
            </Card>
          )}

          <div className="flex justify-end">
            <Button size="lg" onClick={() => navigate('/predictions')}>
              Go to Optimization <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </div>
        </motion.div>
      )}
    </div>
  );
}
