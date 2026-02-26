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
  Legend,
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
  Activity,
  Award,
  TrendingUp,
  Clock,
  ShieldCheck,
  AlertTriangle,
  BarChart3,
  Layers,
} from 'lucide-react';
import { cn } from '@/lib/utils';

// ─── helpers ──────────────────────────────────────────────────────────────────

function modelTypeLabel(type: string) {
  if (type === 'linear') return 'Linear Regression';
  if (type === 'rf') return 'Random Forest';
  return 'Gradient Boosting';
}

function modelTypeBadgeClass(type: string) {
  if (type === 'linear')
    return 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300';
  if (type === 'rf')
    return 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300';
  return 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300';
}

function r2ColorClass(r2: number) {
  if (r2 >= 0.9) return 'text-green-600 dark:text-green-400';
  if (r2 >= 0.7) return 'text-amber-600 dark:text-amber-400';
  return 'text-red-500';
}

function healthColorClass(score: number) {
  if (score >= 80) return 'text-green-600 dark:text-green-400';
  if (score >= 60) return 'text-amber-600 dark:text-amber-400';
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
  const trainedModels = useStore((s) => s.trainedModels);
  const predictions = useStore((s) => s.predictions);
  const navigate = useNavigate();

  // ── derived values ──────────────────────────────────────────────────────────

  const bestModel = useMemo(
    () =>
      trainedModels.length
        ? [...trainedModels].sort((a, b) => b.metrics.r2 - a.metrics.r2)[0]
        : null,
    [trainedModels]
  );

  const avgHealthScore = useMemo(() => {
    if (!datasets.length) return null;
    return Math.round(
      datasets.reduce((s, d) => s + d.healthScore, 0) / datasets.length
    );
  }, [datasets]);

  const totalRows = useMemo(
    () => datasets.reduce((s, d) => s + d.rowCount, 0),
    [datasets]
  );

  // Leaderboard: all models sorted by R²
  const leaderboard = useMemo(
    () => [...trainedModels].sort((a, b) => b.metrics.r2 - a.metrics.r2),
    [trainedModels]
  );

  // Bar chart: top 8 models
  const chartData = useMemo(
    () =>
      leaderboard.slice(0, 8).map((m, i) => ({
        name:
          modelTypeLabel(m.type).replace(' Regression', ' Reg.') +
          (i + 1 > 1 ? ` #${i + 1}` : ''),
        r2Pct: parseFloat((m.metrics.r2 * 100).toFixed(1)),
        accuracy: parseFloat(m.metrics.accuracy.toFixed(1)),
        target: m.config.targetVariable,
        type: m.type,
      })),
    [leaderboard]
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
        label: 'Train Models',
        desc: 'Run Linear, RF & Gradient Boosting',
        done: trainedModels.length > 0,
        path: '/training',
        Icon: Cpu,
      },
      {
        label: 'Run Predictions',
        desc: 'Predict & optimise parameters',
        done: predictions.length > 0,
        path: '/predictions',
        Icon: Target,
      },
    ],
    [datasets, trainedModels, predictions]
  );

  const completedSteps = steps.filter((s) => s.done).length;
  const nextStep = steps.find((s) => !s.done);

  // ── KPI card helper ─────────────────────────────────────────────────────────

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
      label: 'Models Trained',
      value: trainedModels.length,
      sub: bestModel
        ? `Best R² ${(bestModel.metrics.r2 * 100).toFixed(1)}%`
        : 'None yet',
      Icon: Cpu,
      iconBg: 'bg-purple-100 dark:bg-purple-900/30',
      iconColor: 'text-purple-600 dark:text-purple-400',
      onClick: () => navigate('/training'),
    },
    {
      label: 'Best Accuracy',
      value: bestModel ? `${bestModel.metrics.accuracy.toFixed(1)}%` : '—',
      sub: bestModel
        ? `${modelTypeLabel(bestModel.type)} → ${bestModel.config.targetVariable}`
        : 'Train a model first',
      Icon: Award,
      iconBg: 'bg-amber-100 dark:bg-amber-900/30',
      iconColor: 'text-amber-600 dark:text-amber-400',
      onClick: () => navigate('/predictions'),
    },
    {
      label: 'Predictions Made',
      value: predictions.length,
      sub:
        predictions.length > 0
          ? `Last: ${formatTime(predictions[0].timestamp)}`
          : 'No predictions yet',
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
            Steel production optimisation workspace overview
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
                    <p className="text-2xl font-bold mt-1 truncate">
                      {card.value}
                    </p>
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

      {/* ── Main content: leaderboard + sidebar ── */}
      {trainedModels.length > 0 || datasets.length > 0 ? (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left column */}
          <div className="lg:col-span-8 space-y-6">
            {/* Model leaderboard */}
            {trainedModels.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.38 }}
              >
                <Card>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div>
                        <CardTitle>Model Leaderboard</CardTitle>
                        <CardDescription>
                          All trained models ranked by R² — higher is better (max 1.0)
                        </CardDescription>
                      </div>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => navigate('/training')}
                      >
                        <Cpu className="mr-2 h-4 w-4" />
                        Train More
                      </Button>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b text-muted-foreground">
                            <th className="text-left pb-2 font-medium w-8">#</th>
                            <th className="text-left pb-2 font-medium">Model Type</th>
                            <th className="text-left pb-2 font-medium">Target</th>
                            <th className="text-right pb-2 font-medium">R²</th>
                            <th className="text-right pb-2 font-medium">Accuracy</th>
                            <th className="text-right pb-2 font-medium">RMSE</th>
                            <th className="text-right pb-2 font-medium hidden md:table-cell">
                              CV R²
                            </th>
                          </tr>
                        </thead>
                        <tbody>
                          {leaderboard.map((m, i) => (
                            <tr
                              key={m.id}
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
                              <td className="py-2.5">
                                <span
                                  className={cn(
                                    'px-2 py-0.5 rounded text-xs font-medium',
                                    modelTypeBadgeClass(m.type)
                                  )}
                                >
                                  {modelTypeLabel(m.type)}
                                </span>
                              </td>
                              <td className="py-2.5 max-w-[120px] truncate text-muted-foreground">
                                {m.config.targetVariable}
                              </td>
                              <td
                                className={cn(
                                  'py-2.5 text-right font-semibold tabular-nums',
                                  r2ColorClass(m.metrics.r2)
                                )}
                              >
                                {(m.metrics.r2 * 100).toFixed(1)}%
                              </td>
                              <td className="py-2.5 text-right tabular-nums text-muted-foreground">
                                {m.metrics.accuracy.toFixed(1)}%
                              </td>
                              <td className="py-2.5 text-right tabular-nums text-muted-foreground">
                                {m.metrics.rmse.toFixed(3)}
                              </td>
                              <td className="py-2.5 text-right hidden md:table-cell tabular-nums text-muted-foreground">
                                {m.cvMetrics
                                  ? `${(m.cvMetrics.mean.r2 * 100).toFixed(1)}%`
                                  : '—'}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>

                    {/* Legend for CV column */}
                    {leaderboard.some((m) => m.cvMetrics) && (
                      <p className="text-xs text-muted-foreground mt-3 flex items-center gap-1">
                        <ShieldCheck className="h-3.5 w-3.5" />
                        CV R² = 5-fold cross-validation score (generalisation estimate)
                      </p>
                    )}
                  </CardContent>
                </Card>
              </motion.div>
            )}

            {/* Model comparison chart */}
            {chartData.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.45 }}
              >
                <Card>
                  <CardHeader>
                    <CardTitle>Model Performance Comparison</CardTitle>
                    <CardDescription>
                      R² score (% explained variance) and overall accuracy per model
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={240}>
                      <BarChart
                        data={chartData}
                        margin={{ top: 4, right: 8, left: -8, bottom: 4 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" className="opacity-40" />
                        <XAxis
                          dataKey="name"
                          tick={{ fontSize: 11 }}
                          tickLine={false}
                          interval={0}
                          angle={-25}
                          textAnchor="end"
                          height={48}
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
                            name === 'r2Pct' ? 'R² Score' : 'Accuracy',
                          ]}
                          labelFormatter={(label, payload) =>
                            payload?.[0]?.payload?.target
                              ? `${label} → ${payload[0].payload.target}`
                              : label
                          }
                        />
                        <Legend
                          formatter={(v) =>
                            v === 'r2Pct' ? 'R² Score (%)' : 'Accuracy (%)'
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
                          dataKey="accuracy"
                          name="accuracy"
                          radius={[4, 4, 0, 0]}
                          fill="#10b981"
                          fillOpacity={0.55}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </motion.div>
            )}

            {/* Recent predictions */}
            {predictions.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.52 }}
              >
                <Card>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div>
                        <CardTitle>Recent Predictions</CardTitle>
                        <CardDescription>
                          Last {Math.min(predictions.length, 10)} prediction runs
                        </CardDescription>
                      </div>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => navigate('/predictions')}
                      >
                        <Target className="mr-2 h-4 w-4" />
                        Predict
                      </Button>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      {predictions.slice(0, 10).map((p) => (
                        <div
                          key={p.id}
                          className="flex items-center justify-between py-2 border-b last:border-0"
                        >
                          <div className="flex items-center gap-3 min-w-0">
                            <div className="h-7 w-7 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
                              <Sparkles className="h-3.5 w-3.5 text-primary" />
                            </div>
                            <div className="min-w-0">
                              <p className="text-sm font-medium truncate">
                                {p.targetVariable}
                              </p>
                              <p className="text-xs text-muted-foreground">
                                {modelTypeLabel(p.modelType)}
                              </p>
                            </div>
                          </div>
                          <div className="text-right flex-shrink-0 ml-4">
                            <p className="text-sm font-semibold tabular-nums">
                              {p.result.toFixed(3)}
                            </p>
                            <p className="text-xs text-muted-foreground flex items-center justify-end gap-1">
                              <Clock className="h-3 w-3" />
                              {formatDate(p.timestamp)}
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

            {/* Best model spotlight */}
            {bestModel && (
              <motion.div
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.47 }}
              >
                <Card className="border-amber-200 dark:border-amber-800">
                  <CardHeader className="pb-2">
                    <div className="flex items-center gap-2">
                      <Award className="h-4 w-4 text-amber-500" />
                      <CardTitle className="text-base">Best Model</CardTitle>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div>
                      <span
                        className={cn(
                          'text-xs font-medium px-2 py-0.5 rounded',
                          modelTypeBadgeClass(bestModel.type)
                        )}
                      >
                        {modelTypeLabel(bestModel.type)}
                      </span>
                      <p className="text-xs text-muted-foreground mt-1">
                        Target: <span className="font-medium text-foreground">{bestModel.config.targetVariable}</span>
                      </p>
                    </div>

                    <Separator />

                    <div className="grid grid-cols-2 gap-3 text-center">
                      {[
                        { label: 'R² Score', value: `${(bestModel.metrics.r2 * 100).toFixed(1)}%`, color: r2ColorClass(bestModel.metrics.r2) },
                        { label: 'Accuracy', value: `${bestModel.metrics.accuracy.toFixed(1)}%`, color: '' },
                        { label: 'RMSE', value: bestModel.metrics.rmse.toFixed(4), color: '' },
                        { label: 'MAE', value: bestModel.metrics.mae.toFixed(4), color: '' },
                      ].map((m) => (
                        <div key={m.label} className="bg-muted/40 rounded-lg p-2">
                          <p className="text-xs text-muted-foreground">{m.label}</p>
                          <p className={cn('text-sm font-bold tabular-nums mt-0.5', m.color)}>
                            {m.value}
                          </p>
                        </div>
                      ))}
                    </div>

                    {bestModel.cvMetrics && (
                      <div className="text-xs text-muted-foreground bg-muted/40 rounded-lg p-2">
                        <p className="flex items-center gap-1 font-medium text-foreground mb-1">
                          <ShieldCheck className="h-3.5 w-3.5 text-green-600" />
                          5-Fold CV
                        </p>
                        <p>
                          R²: {(bestModel.cvMetrics.mean.r2 * 100).toFixed(1)}%
                          {' ± '}
                          {(bestModel.cvMetrics.std.r2 * 100).toFixed(1)}%
                        </p>
                        {bestModel.metrics.r2 - bestModel.cvMetrics.mean.r2 > 0.1 ? (
                          <p className="text-amber-600 dark:text-amber-400 flex items-center gap-1 mt-1">
                            <AlertTriangle className="h-3 w-3" />
                            Possible overfitting detected
                          </p>
                        ) : (
                          <p className="text-green-600 dark:text-green-400 flex items-center gap-1 mt-1">
                            <CheckCircle2 className="h-3 w-3" />
                            Good generalisation
                          </p>
                        )}
                      </div>
                    )}

                    <Button
                      className="w-full"
                      size="sm"
                      onClick={() => navigate('/predictions')}
                    >
                      <Sparkles className="mr-2 h-4 w-4" />
                      Use This Model
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
                    { label: 'Train a Model', Icon: Cpu, path: '/training' },
                    { label: 'Make a Prediction', Icon: Target, path: '/predictions' },
                    { label: 'View Reports', Icon: Activity, path: '/reports' },
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
                Follow these four steps to set up your steel production optimisation
                pipeline.
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
                  Start by uploading a CSV or Excel file containing historical steel
                  plant data (heats, temperatures, alloy additions, energy usage, etc.).
                  The system will auto-categorise your columns and guide you through
                  training Linear Regression, Random Forest, and Gradient Boosting models
                  to predict and optimise your target KPIs.
                </p>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}
    </div>
  );
}
