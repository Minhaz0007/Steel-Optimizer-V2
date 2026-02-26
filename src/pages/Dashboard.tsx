import { useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { useStore } from '@/store/useStore';
import { motion } from 'framer-motion';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import {
  Database,
  Cpu,
  Target,
  Upload,
  CheckCircle2,
  ArrowRight,
  FileSpreadsheet,
  Sparkles,
  Award,
  TrendingUp,
  Clock,
  ShieldCheck,
  BarChart3,
  Layers,
  AlertTriangle,
  Activity,
  Settings,
} from 'lucide-react';
import { cn } from '@/lib/utils';

// ─── helpers ──────────────────────────────────────────────────────────────────

function r2ColorClass(r2: number) {
  if (r2 >= 0.9) return 'text-green-600 dark:text-green-400';
  if (r2 >= 0.7) return 'text-amber-600 dark:text-amber-400';
  return 'text-red-500';
}

function healthBadgeClass(score: number) {
  if (score >= 80)
    return 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300';
  if (score >= 60)
    return 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300';
  return 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300';
}

function formatDate(iso: string) {
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  } catch {
    return iso;
  }
}

function formatTime(iso: string) {
  try {
    return new Date(iso).toLocaleTimeString(undefined, {
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return '';
  }
}

const CHART_COLORS = [
  '#3b82f6',
  '#8b5cf6',
  '#f59e0b',
  '#10b981',
  '#ef4444',
  '#6366f1',
];

// ─── component ────────────────────────────────────────────────────────────────

export default function Dashboard() {
  const datasets = useStore((s) => s.datasets);
  const trainingSession = useStore((s) => s.trainingSession);
  const optimizationRecords = useStore((s) => s.optimizationRecords);
  const navigate = useNavigate();

  // ── derived values ──────────────────────────────────────────────────────────

  const bestRegressor = useMemo(() => {
    if (!trainingSession?.regressors?.length) return null;
    return [...trainingSession.regressors].sort((a, b) => b.test_r2 - a.test_r2)[0];
  }, [trainingSession]);

  const avgHealthScore = useMemo(() => {
    if (!datasets.length) return null;
    return Math.round(datasets.reduce((s, d) => s + d.healthScore, 0) / datasets.length);
  }, [datasets]);

  const totalRows = useMemo(
    () => datasets.reduce((s, d) => s + d.rowCount, 0),
    [datasets]
  );

  // Regressor leaderboard
  const regressorLeaderboard = useMemo(
    () =>
      trainingSession?.regressors
        ? [...trainingSession.regressors].sort((a, b) => b.test_r2 - a.test_r2)
        : [],
    [trainingSession]
  );

  // Classifier leaderboard
  const classifierLeaderboard = useMemo(
    () =>
      trainingSession?.classifiers
        ? [...trainingSession.classifiers].sort((a, b) => b.test_auc - a.test_auc)
        : [],
    [trainingSession]
  );

  // Bar chart for regressors
  const chartData = useMemo(
    () =>
      regressorLeaderboard.map((r) => ({
        name:
          r.target.length > 18 ? r.target.substring(0, 16) + '…' : r.target,
        fullName: r.target,
        r2Pct: parseFloat((r.test_r2 * 100).toFixed(1)),
        cvR2Pct: parseFloat((r.cv_r2_mean * 100).toFixed(1)),
      })),
    [regressorLeaderboard]
  );

  // Workflow steps
  const steps = useMemo(
    () => [
      {
        label: 'Upload Data',
        desc: 'Import CSV or Excel files',
        done: datasets.length > 0,
        path: '/upload',
        Icon: Upload,
      },
      {
        label: 'Explore Data',
        desc: 'Review distributions & correlations',
        done: datasets.some((d) => d.mappings.length > 0),
        path: '/explorer',
        Icon: Database,
      },
      {
        label: 'Train Pipeline',
        desc: 'Run LightGBM, CatBoost & Optuna',
        done: trainingSession !== null,
        path: '/training',
        Icon: Cpu,
      },
      {
        label: 'Optimize',
        desc: 'Run Bayesian setpoint optimization',
        done: optimizationRecords.length > 0,
        path: '/predictions',
        Icon: Target,
      },
    ],
    [datasets, trainingSession, optimizationRecords]
  );

  const completedSteps = steps.filter((s) => s.done).length;
  const nextStep = steps.find((s) => !s.done);

  // ── KPI cards ────────────────────────────────────────────────────────────────

  const kpiCards = [
    {
      label: 'Datasets',
      value: datasets.length,
      sub: datasets.length
        ? `${totalRows.toLocaleString()} total rows`
        : 'No data yet',
      Icon: FileSpreadsheet,
      iconBg: 'bg-blue-100 dark:bg-blue-900/30',
      iconColor: 'text-blue-600 dark:text-blue-400',
      onClick: () => navigate('/upload'),
    },
    {
      label: 'ML Pipeline',
      value: trainingSession ? 'Trained' : 'Not trained',
      sub: trainingSession
        ? `${trainingSession.rows.toLocaleString()} rows · ${trainingSession.features} features`
        : 'Run training first',
      Icon: Cpu,
      iconBg: 'bg-purple-100 dark:bg-purple-900/30',
      iconColor: 'text-purple-600 dark:text-purple-400',
      onClick: () => navigate('/training'),
    },
    {
      label: 'Best R²',
      value: bestRegressor ? `${(bestRegressor.test_r2 * 100).toFixed(1)}%` : '—',
      sub: bestRegressor
        ? `LightGBM → ${bestRegressor.target}`
        : 'Train models first',
      Icon: Award,
      iconBg: 'bg-amber-100 dark:bg-amber-900/30',
      iconColor: 'text-amber-600 dark:text-amber-400',
      onClick: () => navigate('/predictions'),
    },
    {
      label: 'Optimizations',
      value: optimizationRecords.length,
      sub:
        optimizationRecords.length > 0
          ? `Last: ${formatTime(optimizationRecords[0].timestamp)}`
          : 'No runs yet',
      Icon: Sparkles,
      iconBg: 'bg-green-100 dark:bg-green-900/30',
      iconColor: 'text-green-600 dark:text-green-400',
      onClick: () => navigate('/predictions'),
    },
  ];

  // ── render ──────────────────────────────────────────────────────────────────

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Page heading */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
          <p className="text-muted-foreground mt-1">
            Steel production ML optimization workspace
          </p>
        </div>
        <Button onClick={() => navigate('/upload')} className="hidden sm:flex">
          <Upload className="mr-2 h-4 w-4" />
          Upload Data
        </Button>
      </div>

      {/* ── KPI cards ── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {kpiCards.map((card, i) => (
          <motion.div
            key={card.label}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.07 }}
          >
            <Card
              className="cursor-pointer hover:shadow-md transition-shadow"
              onClick={card.onClick}
            >
              <CardContent className="p-5">
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                      {card.label}
                    </p>
                    <p className="text-2xl font-bold mt-1 truncate">{card.value}</p>
                    <p className="text-xs text-muted-foreground mt-1 truncate">
                      {card.sub}
                    </p>
                  </div>
                  <div
                    className={cn(
                      'h-10 w-10 rounded-xl flex items-center justify-center flex-shrink-0',
                      card.iconBg
                    )}
                  >
                    <card.Icon className={cn('h-5 w-5', card.iconColor)} />
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>

      {/* ── Workflow progress ── */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base">Workflow Progress</CardTitle>
              <span className="text-sm text-muted-foreground">
                {completedSteps} / {steps.length} completed
              </span>
            </div>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2 sm:gap-0">
              {steps.map((step, i) => (
                <div key={step.path} className="flex items-center flex-1 min-w-0">
                  <button
                    onClick={() => navigate(step.path)}
                    className={cn(
                      'flex-1 flex items-center gap-3 px-3 py-2.5 rounded-lg text-left transition-colors hover:opacity-80',
                      step.done
                        ? 'bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800'
                        : nextStep?.path === step.path
                        ? 'bg-primary/5 border border-primary/30'
                        : 'bg-muted border border-transparent'
                    )}
                  >
                    {step.done ? (
                      <CheckCircle2 className="h-5 w-5 text-green-600 dark:text-green-400 flex-shrink-0" />
                    ) : (
                      <step.Icon
                        className={cn(
                          'h-5 w-5 flex-shrink-0',
                          nextStep?.path === step.path
                            ? 'text-primary'
                            : 'text-muted-foreground'
                        )}
                      />
                    )}
                    <div className="min-w-0">
                      <p
                        className={cn(
                          'text-sm font-medium truncate',
                          step.done
                            ? 'text-green-700 dark:text-green-400'
                            : nextStep?.path === step.path
                            ? 'text-primary'
                            : 'text-muted-foreground'
                        )}
                      >
                        {step.label}
                      </p>
                      <p className="text-xs text-muted-foreground truncate hidden sm:block">
                        {step.desc}
                      </p>
                    </div>
                  </button>
                  {i < steps.length - 1 && (
                    <ArrowRight className="h-4 w-4 text-muted-foreground flex-shrink-0 mx-1 hidden sm:block" />
                  )}
                </div>
              ))}
            </div>

            {nextStep && (
              <div className="mt-4 flex items-center gap-3">
                <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary rounded-full transition-all"
                    style={{ width: `${(completedSteps / steps.length) * 100}%` }}
                  />
                </div>
                <Button size="sm" onClick={() => navigate(nextStep.path)}>
                  Next: {nextStep.label}
                  <ArrowRight className="ml-2 h-3.5 w-3.5" />
                </Button>
              </div>
            )}
            {!nextStep && (
              <p className="mt-4 text-sm text-green-600 dark:text-green-400 font-medium">
                ✓ All workflow steps complete — your workspace is fully set up.
              </p>
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* ── Main content ── */}
      {trainingSession || datasets.length > 0 ? (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left column */}
          <div className="lg:col-span-8 space-y-6">
            {/* LightGBM Regressor leaderboard */}
            {regressorLeaderboard.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.38 }}
              >
                <Card>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div>
                        <CardTitle>LightGBM Regressors</CardTitle>
                        <CardDescription>
                          5 continuous targets — ranked by Test R²
                        </CardDescription>
                      </div>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => navigate('/training')}
                      >
                        <Cpu className="mr-2 h-4 w-4" />
                        Retrain
                      </Button>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b text-muted-foreground">
                            <th className="text-left pb-2 font-medium w-8">#</th>
                            <th className="text-left pb-2 font-medium">Target</th>
                            <th className="text-right pb-2 font-medium">Test R²</th>
                            <th className="text-right pb-2 font-medium">Test RMSE</th>
                            <th className="text-right pb-2 font-medium hidden md:table-cell">
                              CV R²
                            </th>
                            <th className="text-right pb-2 font-medium hidden md:table-cell">
                              Best Iter
                            </th>
                          </tr>
                        </thead>
                        <tbody>
                          {regressorLeaderboard.map((m, i) => (
                            <tr
                              key={m.target}
                              className={cn(
                                'border-b last:border-0 hover:bg-muted/40 transition-colors',
                                i === 0 && 'bg-amber-50/50 dark:bg-amber-900/10'
                              )}
                            >
                              <td className="py-2.5 pr-2">
                                {i === 0 ? (
                                  <Award className="h-4 w-4 text-amber-500" />
                                ) : (
                                  <span className="text-muted-foreground">{i + 1}</span>
                                )}
                              </td>
                              <td className="py-2.5 font-mono text-xs text-muted-foreground max-w-[140px] truncate">
                                {m.target}
                              </td>
                              <td
                                className={cn(
                                  'py-2.5 text-right font-semibold tabular-nums',
                                  r2ColorClass(m.test_r2)
                                )}
                              >
                                {(m.test_r2 * 100).toFixed(1)}%
                              </td>
                              <td className="py-2.5 text-right tabular-nums text-muted-foreground">
                                {m.test_rmse.toFixed(3)}
                              </td>
                              <td className="py-2.5 text-right hidden md:table-cell tabular-nums text-muted-foreground">
                                {(m.cv_r2_mean * 100).toFixed(1)}%
                                <span className="text-xs opacity-60">
                                  {' '}±{(m.cv_rmse_std).toFixed(2)}
                                </span>
                              </td>
                              <td className="py-2.5 text-right hidden md:table-cell tabular-nums text-muted-foreground">
                                {m.best_iteration}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <p className="text-xs text-muted-foreground mt-3 flex items-center gap-1">
                      <ShieldCheck className="h-3.5 w-3.5" />
                      CV R² = 5-fold cross-validation mean (generalisation estimate)
                    </p>
                  </CardContent>
                </Card>
              </motion.div>
            )}

            {/* CatBoost Classifier leaderboard */}
            {classifierLeaderboard.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.42 }}
              >
                <Card>
                  <CardHeader>
                    <CardTitle>CatBoost Classifiers</CardTitle>
                    <CardDescription>
                      2 binary targets — ranked by AUC-ROC
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b text-muted-foreground">
                            <th className="text-left pb-2 font-medium">Target</th>
                            <th className="text-right pb-2 font-medium">Test F1</th>
                            <th className="text-right pb-2 font-medium">AUC-ROC</th>
                            <th className="text-right pb-2 font-medium hidden md:table-cell">
                              Precision
                            </th>
                            <th className="text-right pb-2 font-medium hidden md:table-cell">
                              Recall
                            </th>
                          </tr>
                        </thead>
                        <tbody>
                          {classifierLeaderboard.map((c) => (
                            <tr
                              key={c.target}
                              className="border-b last:border-0 hover:bg-muted/40 transition-colors"
                            >
                              <td className="py-2.5 font-mono text-xs text-muted-foreground max-w-[140px] truncate">
                                {c.target}
                              </td>
                              <td className="py-2.5 text-right font-semibold tabular-nums">
                                {c.test_f1.toFixed(3)}
                              </td>
                              <td
                                className={cn(
                                  'py-2.5 text-right font-semibold tabular-nums',
                                  r2ColorClass(c.test_auc)
                                )}
                              >
                                {c.test_auc.toFixed(3)}
                              </td>
                              <td className="py-2.5 text-right hidden md:table-cell tabular-nums text-muted-foreground">
                                {c.test_precision.toFixed(3)}
                              </td>
                              <td className="py-2.5 text-right hidden md:table-cell tabular-nums text-muted-foreground">
                                {c.test_recall.toFixed(3)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            )}

            {/* Regressor R² chart */}
            {chartData.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.46 }}
              >
                <Card>
                  <CardHeader>
                    <CardTitle>Regressor Performance</CardTitle>
                    <CardDescription>
                      Test R² vs CV R² — higher is better (max 100%)
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={220}>
                      <BarChart
                        data={chartData}
                        margin={{ top: 4, right: 8, left: -8, bottom: 40 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" className="opacity-40" />
                        <XAxis
                          dataKey="name"
                          tick={{ fontSize: 10 }}
                          tickLine={false}
                          interval={0}
                          angle={-20}
                          textAnchor="end"
                          height={52}
                        />
                        <YAxis
                          domain={[0, 100]}
                          tick={{ fontSize: 11 }}
                          tickLine={false}
                          unit="%"
                        />
                        <Tooltip
                          formatter={(v: number, name: string) => [
                            `${v.toFixed(1)}%`,
                            name === 'r2Pct' ? 'Test R²' : 'CV R²',
                          ]}
                          labelFormatter={(label, payload) =>
                            payload?.[0]?.payload?.fullName ?? label
                          }
                        />
                        <Bar dataKey="r2Pct" name="r2Pct" radius={[4, 4, 0, 0]}>
                          {chartData.map((_, i) => (
                            <Cell
                              key={i}
                              fill={CHART_COLORS[i % CHART_COLORS.length]}
                              fillOpacity={0.85}
                            />
                          ))}
                        </Bar>
                        <Bar
                          dataKey="cvR2Pct"
                          name="cvR2Pct"
                          radius={[4, 4, 0, 0]}
                          fill="#10b981"
                          fillOpacity={0.5}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </motion.div>
            )}

            {/* Recent optimizations */}
            {optimizationRecords.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.52 }}
              >
                <Card>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div>
                        <CardTitle>Recent Optimizations</CardTitle>
                        <CardDescription>
                          Last {Math.min(optimizationRecords.length, 8)} Optuna runs
                        </CardDescription>
                      </div>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => navigate('/predictions')}
                      >
                        <Target className="mr-2 h-4 w-4" />
                        Optimize
                      </Button>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      {optimizationRecords.slice(0, 8).map((rec) => (
                        <div
                          key={rec.id}
                          className="flex items-center justify-between py-2 border-b last:border-0"
                        >
                          <div className="flex items-center gap-3 min-w-0">
                            <div
                              className={cn(
                                'h-7 w-7 rounded-full flex items-center justify-center flex-shrink-0',
                                rec.isAnomaly
                                  ? 'bg-red-100 dark:bg-red-900/30'
                                  : 'bg-green-100 dark:bg-green-900/30'
                              )}
                            >
                              {rec.isAnomaly ? (
                                <AlertTriangle className="h-3.5 w-3.5 text-red-500" />
                              ) : (
                                <CheckCircle2 className="h-3.5 w-3.5 text-green-600 dark:text-green-400" />
                              )}
                            </div>
                            <div className="min-w-0">
                              <p className="text-sm font-medium truncate">
                                {rec.anomalyLabel}
                              </p>
                              <p className="text-xs text-muted-foreground">
                                Score: {(rec.compositeScore * 100).toFixed(1)} · Quality:{' '}
                                {(rec.qualityPassProbability * 100).toFixed(0)}%
                              </p>
                            </div>
                          </div>
                          <div className="text-right flex-shrink-0 ml-4">
                            <p className="text-xs text-muted-foreground flex items-center justify-end gap-1">
                              <Clock className="h-3 w-3" />
                              {formatDate(rec.timestamp)}
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            )}
          </div>

          {/* Right column */}
          <div className="lg:col-span-4 space-y-6">
            {/* Datasets */}
            {datasets.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
              >
                <Card>
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-base">Datasets</CardTitle>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => navigate('/upload')}
                        className="text-xs h-7 px-2"
                      >
                        + Add
                      </Button>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {datasets.map((d) => (
                      <div
                        key={d.id}
                        className="p-3 rounded-lg border bg-muted/30 hover:bg-muted/60 transition-colors cursor-pointer"
                        onClick={() => navigate('/explorer')}
                      >
                        <div className="flex items-start justify-between gap-2 mb-2">
                          <div className="min-w-0">
                            <p className="text-sm font-medium truncate">{d.name}</p>
                            <p className="text-xs text-muted-foreground">
                              {formatDate(d.uploadDate)}
                            </p>
                          </div>
                          <span
                            className={cn(
                              'text-xs font-semibold px-2 py-0.5 rounded flex-shrink-0',
                              healthBadgeClass(d.healthScore)
                            )}
                          >
                            {d.healthScore}%
                          </span>
                        </div>
                        <div className="flex gap-3 text-xs text-muted-foreground">
                          <span className="flex items-center gap-1">
                            <Layers className="h-3 w-3" />
                            {d.rowCount.toLocaleString()} rows
                          </span>
                          <span className="flex items-center gap-1">
                            <BarChart3 className="h-3 w-3" />
                            {d.columnCount} cols
                          </span>
                          {d.mappings.length > 0 && (
                            <span className="flex items-center gap-1 text-green-600 dark:text-green-400">
                              <CheckCircle2 className="h-3 w-3" />
                              Mapped
                            </span>
                          )}
                        </div>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              </motion.div>
            )}

            {/* Training session spotlight */}
            {trainingSession && (
              <motion.div
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.47 }}
              >
                <Card className="border-purple-200 dark:border-purple-800">
                  <CardHeader className="pb-2">
                    <div className="flex items-center gap-2">
                      <Cpu className="h-4 w-4 text-purple-500" />
                      <CardTitle className="text-base">Active Pipeline</CardTitle>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="text-xs text-muted-foreground">
                      Trained {formatDate(trainingSession.trainedAt)}
                    </div>

                    <Separator />

                    <div className="grid grid-cols-2 gap-3 text-center">
                      {[
                        { label: 'Training Rows', value: trainingSession.rows.toLocaleString() },
                        { label: 'Features', value: trainingSession.features },
                        { label: 'Regressors', value: trainingSession.regressors?.length ?? 0 },
                        { label: 'Classifiers', value: trainingSession.classifiers?.length ?? 0 },
                      ].map((m) => (
                        <div key={m.label} className="bg-muted/40 rounded-lg p-2">
                          <p className="text-xs text-muted-foreground">{m.label}</p>
                          <p className="text-sm font-bold tabular-nums mt-0.5">{m.value}</p>
                        </div>
                      ))}
                    </div>

                    <div className="flex flex-wrap gap-1.5 text-xs">
                      {trainingSession.shap?.computed && (
                        <span className="px-2 py-0.5 rounded bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300">
                          SHAP ✓
                        </span>
                      )}
                      {trainingSession.anomaly?.trained && (
                        <span className="px-2 py-0.5 rounded bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300">
                          Isolation Forest ✓
                        </span>
                      )}
                      {trainingSession.forecaster?.trained && (
                        <span className="px-2 py-0.5 rounded bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300">
                          Forecaster ✓
                        </span>
                      )}
                    </div>

                    <Button
                      className="w-full"
                      size="sm"
                      onClick={() => navigate('/predictions')}
                    >
                      <Sparkles className="mr-2 h-4 w-4" />
                      Run Optimization
                    </Button>
                  </CardContent>
                </Card>
              </motion.div>
            )}

            {/* Quick actions */}
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.54 }}
            >
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base">Quick Actions</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  {[
                    { label: 'Upload New Dataset', Icon: Upload, path: '/upload' },
                    { label: 'Explore Data', Icon: Database, path: '/explorer' },
                    { label: 'Train ML Pipeline', Icon: Cpu, path: '/training' },
                    { label: 'Run Optimization', Icon: Target, path: '/predictions' },
                    { label: 'View Reports', Icon: Activity, path: '/reports' },
                    { label: 'Settings', Icon: Settings, path: '/settings' },
                  ].map(({ label, Icon, path }) => (
                    <Button
                      key={path}
                      variant="ghost"
                      className="w-full justify-start text-sm h-9"
                      onClick={() => navigate(path)}
                    >
                      <Icon className="mr-3 h-4 w-4 text-muted-foreground" />
                      {label}
                    </Button>
                  ))}
                </CardContent>
              </Card>
            </motion.div>
          </div>
        </div>
      ) : (
        /* ── Empty / getting-started state ── */
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.35 }}
        >
          <Card>
            <CardHeader>
              <CardTitle>Getting Started</CardTitle>
              <CardDescription>
                Follow these four steps to set up your steel production ML optimization pipeline.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                {steps.map((step, i) => (
                  <button
                    key={step.path}
                    onClick={() => navigate(step.path)}
                    className="group p-4 rounded-xl border bg-muted/30 hover:bg-muted/60 hover:border-primary/40 text-left transition-all"
                  >
                    <div className="flex items-center gap-2 mb-3">
                      <div className="h-8 w-8 rounded-lg bg-primary/10 flex items-center justify-center text-primary group-hover:bg-primary group-hover:text-primary-foreground transition-colors">
                        <step.Icon className="h-4 w-4" />
                      </div>
                      <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                        Step {i + 1}
                      </span>
                    </div>
                    <p className="font-semibold text-sm">{step.label}</p>
                    <p className="text-xs text-muted-foreground mt-1">{step.desc}</p>
                    <div className="mt-3 flex items-center text-xs text-primary font-medium opacity-0 group-hover:opacity-100 transition-opacity">
                      Get started
                      <ArrowRight className="ml-1 h-3 w-3" />
                    </div>
                  </button>
                ))}
              </div>

              <Separator className="my-6" />

              <div className="flex items-start gap-3 text-sm text-muted-foreground">
                <TrendingUp className="h-4 w-4 mt-0.5 flex-shrink-0 text-primary" />
                <p>
                  Upload a CSV with historical steel plant shift data, then run the Python ML
                  pipeline — LightGBM regressors, CatBoost classifiers, SHAP analysis, Isolation
                  Forest anomaly detection, and Optuna Bayesian optimization will automatically
                  train and find optimal furnace setpoints for your KPIs.
                </p>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}
    </div>
  );
}
