import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useStore, PredictionRecord } from '@/store/useStore';
import { motion } from 'framer-motion';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
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
  LineChart,
  Line,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
} from 'recharts';
import {
  Download,
  Award,
  Database,
  Cpu,
  Target,
  Sparkles,
  ShieldCheck,
  AlertTriangle,
  CheckCircle2,
  TrendingUp,
  BarChart3,
  Layers,
  Clock,
  FileText,
  ArrowRight,
  Upload,
  Star,
  Activity,
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

function healthBadgeClass(score: number) {
  if (score >= 80)
    return 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300';
  if (score >= 60)
    return 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300';
  return 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300';
}

function healthLabel(score: number) {
  if (score >= 80) return 'Excellent';
  if (score >= 60) return 'Fair';
  return 'Poor';
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

function formatDateTime(iso: string) {
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return iso;
  }
}

const CHART_COLORS = [
  '#3b82f6',
  '#8b5cf6',
  '#f59e0b',
  '#10b981',
  '#ef4444',
  '#6366f1',
  '#14b8a6',
  '#f97316',
];

// ─── CSV export ───────────────────────────────────────────────────────────────

function exportPredictionsToCSV(predictions: PredictionRecord[]) {
  const headers = ['Date & Time', 'Target Variable', 'Model Type', 'Predicted Value', 'Model ID'];
  const rows = predictions.map((p) => [
    formatDateTime(p.timestamp),
    p.targetVariable,
    p.modelType,
    p.result.toFixed(4),
    p.modelId,
  ]);
  const csv = [headers, ...rows].map((r) => r.map((c) => `"${c}"`).join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `steel-optimizer-predictions-${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

// ─── component ────────────────────────────────────────────────────────────────

export default function Reports() {
  const datasets = useStore((s) => s.datasets);
  const trainedModels = useStore((s) => s.trainedModels);
  const predictions = useStore((s) => s.predictions);
  const navigate = useNavigate();

  const [historyFilter, setHistoryFilter] = useState('');

  // ── derived values ──────────────────────────────────────────────────────────

  const bestModel = useMemo(
    () =>
      trainedModels.length
        ? [...trainedModels].sort((a, b) => b.metrics.r2 - a.metrics.r2)[0]
        : null,
    [trainedModels]
  );

  const leaderboard = useMemo(
    () => [...trainedModels].sort((a, b) => b.metrics.r2 - a.metrics.r2),
    [trainedModels]
  );

  const modelComparisonChart = useMemo(
    () =>
      leaderboard.slice(0, 8).map((m, i) => ({
        name:
          modelTypeLabel(m.type).replace(' Regression', ' Reg.') +
          (i > 0 ? ` #${i + 1}` : ''),
        r2: parseFloat((m.metrics.r2 * 100).toFixed(1)),
        accuracy: parseFloat(m.metrics.accuracy.toFixed(1)),
        type: m.type,
        target: m.config.targetVariable,
      })),
    [leaderboard]
  );

  const featureImportanceData = useMemo(() => {
    if (!bestModel?.featureImportance) return [];
    return [...bestModel.featureImportance]
      .sort((a, b) => b.importance - a.importance)
      .slice(0, 10)
      .map((f) => ({
        feature: f.feature.length > 20 ? f.feature.slice(0, 18) + '…' : f.feature,
        importance: parseFloat((f.importance * 100).toFixed(2)),
      }));
  }, [bestModel]);

  // Prediction trend: group by day
  const predictionTrend = useMemo(() => {
    const byDay: Record<string, number> = {};
    predictions.forEach((p) => {
      const day = p.timestamp.slice(0, 10);
      byDay[day] = (byDay[day] || 0) + 1;
    });
    return Object.entries(byDay)
      .sort(([a], [b]) => a.localeCompare(b))
      .slice(-14)
      .map(([date, count]) => ({
        date: new Date(date).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }),
        count,
      }));
  }, [predictions]);

  // Radar data for best model
  const radarData = useMemo(() => {
    if (!bestModel) return [];
    const { r2, accuracy, rmse, mae, mape } = bestModel.metrics;
    return [
      { metric: 'R² Score', value: parseFloat((r2 * 100).toFixed(1)) },
      { metric: 'Accuracy', value: parseFloat(accuracy.toFixed(1)) },
      { metric: 'RMSE (inv)', value: parseFloat(Math.max(0, 100 - rmse * 10).toFixed(1)) },
      { metric: 'MAE (inv)', value: parseFloat(Math.max(0, 100 - mae * 10).toFixed(1)) },
      { metric: 'MAPE (inv)', value: parseFloat(Math.max(0, 100 - mape).toFixed(1)) },
    ];
  }, [bestModel]);

  // Filtered history
  const filteredHistory = useMemo(() => {
    if (!historyFilter) return predictions;
    const f = historyFilter.toLowerCase();
    return predictions.filter(
      (p) =>
        p.targetVariable.toLowerCase().includes(f) ||
        p.modelType.toLowerCase().includes(f)
    );
  }, [predictions, historyFilter]);

  const totalRows = useMemo(
    () => datasets.reduce((s, d) => s + d.rowCount, 0),
    [datasets]
  );

  const avgHealthScore = useMemo(() => {
    if (!datasets.length) return null;
    return Math.round(datasets.reduce((s, d) => s + d.healthScore, 0) / datasets.length);
  }, [datasets]);

  // ── empty states ────────────────────────────────────────────────────────────

  const hasNoData = datasets.length === 0 && trainedModels.length === 0;

  if (hasNoData) {
    return (
      <div className="space-y-6 max-w-7xl mx-auto">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Reports</h1>
          <p className="text-muted-foreground mt-1">
            Your optimization performance reports and insights
          </p>
        </div>
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}>
          <Card className="border-dashed">
            <CardContent className="py-16 text-center">
              <FileText className="h-12 w-12 mx-auto text-muted-foreground mb-4 opacity-40" />
              <h3 className="text-lg font-semibold mb-2">No data yet</h3>
              <p className="text-muted-foreground text-sm max-w-sm mx-auto mb-6">
                Reports will appear here once you have uploaded data and trained at least
                one model. Follow the workflow steps to get started.
              </p>
              <div className="flex flex-col sm:flex-row gap-3 justify-center">
                <Button onClick={() => navigate('/upload')}>
                  <Upload className="mr-2 h-4 w-4" />
                  Upload Data
                </Button>
                <Button variant="outline" onClick={() => navigate('/dashboard')}>
                  View Workflow
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    );
  }

  // ── render ──────────────────────────────────────────────────────────────────

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Page heading */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Reports</h1>
          <p className="text-muted-foreground mt-1">
            Your steel optimization performance reports and insights
          </p>
        </div>
        {predictions.length > 0 && (
          <Button variant="outline" onClick={() => exportPredictionsToCSV(predictions)}>
            <Download className="mr-2 h-4 w-4" />
            Export Predictions CSV
          </Button>
        )}
      </div>

      {/* Summary KPI strip */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        {[
          {
            label: 'Datasets Uploaded',
            value: datasets.length,
            sub: `${totalRows.toLocaleString()} total rows`,
            Icon: Database,
            iconBg: 'bg-blue-100 dark:bg-blue-900/30',
            iconColor: 'text-blue-600 dark:text-blue-400',
          },
          {
            label: 'Models Trained',
            value: trainedModels.length,
            sub: bestModel ? `Best: ${modelTypeLabel(bestModel.type)}` : 'None yet',
            Icon: Cpu,
            iconBg: 'bg-purple-100 dark:bg-purple-900/30',
            iconColor: 'text-purple-600 dark:text-purple-400',
          },
          {
            label: 'Best Accuracy',
            value: bestModel ? `${bestModel.metrics.accuracy.toFixed(1)}%` : '—',
            sub: bestModel ? bestModel.config.targetVariable : 'Train a model',
            Icon: Award,
            iconBg: 'bg-amber-100 dark:bg-amber-900/30',
            iconColor: 'text-amber-600 dark:text-amber-400',
          },
          {
            label: 'Predictions Run',
            value: predictions.length,
            sub:
              predictions.length > 0
                ? `Last: ${formatDate(predictions[0].timestamp)}`
                : 'No predictions yet',
            Icon: Sparkles,
            iconBg: 'bg-green-100 dark:bg-green-900/30',
            iconColor: 'text-green-600 dark:text-green-400',
          },
        ].map((card, i) => (
          <motion.div
            key={card.label}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.06 }}
          >
            <Card>
              <CardContent className="p-5">
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                      {card.label}
                    </p>
                    <p className="text-2xl font-bold mt-1">{card.value}</p>
                    <p className="text-xs text-muted-foreground mt-1 truncate">{card.sub}</p>
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

      {/* Tabs */}
      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4 lg:w-auto lg:inline-flex">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="models">Models</TabsTrigger>
          <TabsTrigger value="datasets">Datasets</TabsTrigger>
          <TabsTrigger value="history">History</TabsTrigger>
        </TabsList>

        {/* ── Overview Tab ── */}
        <TabsContent value="overview" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left: best model details + feature importance */}
            <div className="lg:col-span-5 space-y-6">
              {bestModel ? (
                <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
                  <Card className="border-amber-200 dark:border-amber-800">
                    <CardHeader className="pb-2">
                      <div className="flex items-center gap-2">
                        <Award className="h-4 w-4 text-amber-500" />
                        <CardTitle className="text-base">Best Performing Model</CardTitle>
                      </div>
                      <CardDescription>
                        Your top model based on R² score (highest explained variance)
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="flex items-center gap-3 flex-wrap">
                        <span
                          className={cn(
                            'text-xs font-medium px-2 py-1 rounded',
                            modelTypeBadgeClass(bestModel.type)
                          )}
                        >
                          {modelTypeLabel(bestModel.type)}
                        </span>
                        <span className="text-sm text-muted-foreground">
                          Predicts:{' '}
                          <span className="font-semibold text-foreground">
                            {bestModel.config.targetVariable}
                          </span>
                        </span>
                      </div>

                      <div className="grid grid-cols-2 gap-3">
                        {[
                          {
                            label: 'R² Score',
                            value: `${(bestModel.metrics.r2 * 100).toFixed(1)}%`,
                            desc: 'Variance explained',
                            color: r2ColorClass(bestModel.metrics.r2),
                          },
                          {
                            label: 'Accuracy',
                            value: `${bestModel.metrics.accuracy.toFixed(1)}%`,
                            desc: 'Overall accuracy',
                            color: '',
                          },
                          {
                            label: 'RMSE',
                            value: bestModel.metrics.rmse.toFixed(4),
                            desc: 'Prediction error',
                            color: '',
                          },
                          {
                            label: 'MAE',
                            value: bestModel.metrics.mae.toFixed(4),
                            desc: 'Mean abs. error',
                            color: '',
                          },
                        ].map((m) => (
                          <div key={m.label} className="bg-muted/40 rounded-lg p-3">
                            <p className="text-xs text-muted-foreground">{m.label}</p>
                            <p className={cn('text-lg font-bold tabular-nums mt-0.5', m.color)}>
                              {m.value}
                            </p>
                            <p className="text-xs text-muted-foreground">{m.desc}</p>
                          </div>
                        ))}
                      </div>

                      {bestModel.cvMetrics && (
                        <div className="rounded-lg border p-3 space-y-1.5">
                          <p className="text-xs font-semibold flex items-center gap-1">
                            <ShieldCheck className="h-3.5 w-3.5 text-green-600" />
                            5-Fold Cross-Validation — Generalisation Check
                          </p>
                          <p className="text-sm tabular-nums">
                            CV R²:{' '}
                            <span className="font-semibold">
                              {(bestModel.cvMetrics.mean.r2 * 100).toFixed(1)}%
                            </span>{' '}
                            <span className="text-muted-foreground text-xs">
                              ± {(bestModel.cvMetrics.std.r2 * 100).toFixed(1)}%
                            </span>
                          </p>
                          <p className="text-sm tabular-nums">
                            CV Accuracy:{' '}
                            <span className="font-semibold">
                              {bestModel.cvMetrics.mean.accuracy.toFixed(1)}%
                            </span>
                          </p>
                          {bestModel.metrics.r2 - bestModel.cvMetrics.mean.r2 > 0.1 ? (
                            <p className="text-xs text-amber-600 dark:text-amber-400 flex items-center gap-1">
                              <AlertTriangle className="h-3 w-3" />
                              Model may be overfitting — more data could help
                            </p>
                          ) : (
                            <p className="text-xs text-green-600 dark:text-green-400 flex items-center gap-1">
                              <CheckCircle2 className="h-3 w-3" />
                              Good generalisation — model performs reliably on new data
                            </p>
                          )}
                        </div>
                      )}

                      <Button
                        className="w-full"
                        size="sm"
                        onClick={() => navigate('/predictions')}
                      >
                        <Target className="mr-2 h-4 w-4" />
                        Use This Model for Predictions
                      </Button>
                    </CardContent>
                  </Card>
                </motion.div>
              ) : (
                <Card className="border-dashed">
                  <CardContent className="py-10 text-center">
                    <Cpu className="h-8 w-8 mx-auto text-muted-foreground opacity-40 mb-3" />
                    <p className="text-sm font-medium mb-1">No models trained yet</p>
                    <p className="text-xs text-muted-foreground mb-4">
                      Train a model to see your performance report
                    </p>
                    <Button size="sm" onClick={() => navigate('/training')}>
                      Go to Training
                    </Button>
                  </CardContent>
                </Card>
              )}

              {/* Radar chart for best model */}
              {radarData.length > 0 && (
                <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-base">Model Quality Radar</CardTitle>
                      <CardDescription>
                        Balanced view of your best model across 5 dimensions
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={220}>
                        <RadarChart data={radarData}>
                          <PolarGrid />
                          <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11 }} />
                          <Radar
                            dataKey="value"
                            stroke="#3b82f6"
                            fill="#3b82f6"
                            fillOpacity={0.25}
                          />
                        </RadarChart>
                      </ResponsiveContainer>
                      <p className="text-xs text-muted-foreground text-center mt-1">
                        Higher values (farther from centre) are better in all dimensions
                      </p>
                    </CardContent>
                  </Card>
                </motion.div>
              )}
            </div>

            {/* Right: Feature importance + prediction trend */}
            <div className="lg:col-span-7 space-y-6">
              {featureImportanceData.length > 0 && (
                <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.12 }}>
                  <Card>
                    <CardHeader>
                      <CardTitle>Top Process Variables (Feature Importance)</CardTitle>
                      <CardDescription>
                        Which inputs have the biggest impact on{' '}
                        <span className="font-semibold text-foreground">
                          {bestModel?.config.targetVariable}
                        </span>
                        . Focus your attention on these for the greatest results.
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={260}>
                        <BarChart
                          data={featureImportanceData}
                          layout="vertical"
                          margin={{ top: 4, right: 16, left: 8, bottom: 4 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" horizontal={false} className="opacity-40" />
                          <XAxis type="number" tick={{ fontSize: 11 }} tickLine={false} unit="%" />
                          <YAxis
                            type="category"
                            dataKey="feature"
                            tick={{ fontSize: 11 }}
                            tickLine={false}
                            width={110}
                          />
                          <Tooltip formatter={(v: number) => [`${v.toFixed(2)}%`, 'Importance']} />
                          <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                            {featureImportanceData.map((_, i) => (
                              <Cell
                                key={i}
                                fill={CHART_COLORS[i % CHART_COLORS.length]}
                                fillOpacity={0.85}
                              />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                      <div className="mt-3 p-3 bg-muted/40 rounded-lg">
                        <p className="text-xs text-muted-foreground">
                          <span className="font-medium text-foreground">How to read this:</span>{' '}
                          A higher bar means that variable has a stronger influence on the predicted
                          output. Adjust these process parameters first for maximum impact.
                        </p>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              )}

              {predictionTrend.length > 0 && (
                <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.18 }}>
                  <Card>
                    <CardHeader>
                      <CardTitle>Prediction Activity</CardTitle>
                      <CardDescription>
                        Number of predictions run per day over the last 14 days
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={180}>
                        <LineChart
                          data={predictionTrend}
                          margin={{ top: 4, right: 8, left: -16, bottom: 4 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" className="opacity-40" />
                          <XAxis dataKey="date" tick={{ fontSize: 11 }} tickLine={false} />
                          <YAxis tick={{ fontSize: 11 }} tickLine={false} allowDecimals={false} />
                          <Tooltip formatter={(v: number) => [v, 'Predictions']} />
                          <Line
                            type="monotone"
                            dataKey="count"
                            stroke="#3b82f6"
                            strokeWidth={2}
                            dot={{ r: 3 }}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>
                </motion.div>
              )}

              {predictionTrend.length === 0 && featureImportanceData.length === 0 && (
                <Card className="border-dashed">
                  <CardContent className="py-12 text-center">
                    <Activity className="h-8 w-8 mx-auto text-muted-foreground opacity-40 mb-3" />
                    <p className="text-sm text-muted-foreground">
                      Charts will appear here after you train models and run predictions.
                    </p>
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        </TabsContent>

        {/* ── Models Tab ── */}
        <TabsContent value="models" className="space-y-6">
          {trainedModels.length === 0 ? (
            <Card className="border-dashed">
              <CardContent className="py-16 text-center">
                <Cpu className="h-10 w-10 mx-auto text-muted-foreground opacity-40 mb-4" />
                <h3 className="font-semibold mb-2">No trained models</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Train your first model to see a detailed comparison here.
                </p>
                <Button onClick={() => navigate('/training')}>
                  <Cpu className="mr-2 h-4 w-4" />
                  Go to Training
                </Button>
              </CardContent>
            </Card>
          ) : (
            <>
              {/* Model comparison chart */}
              {modelComparisonChart.length > 0 && (
                <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
                  <Card>
                    <CardHeader>
                      <CardTitle>Model Performance Comparison</CardTitle>
                      <CardDescription>
                        R² score (explained variance) vs accuracy for all trained models.
                        Aim for high values in both.
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={260}>
                        <BarChart
                          data={modelComparisonChart}
                          margin={{ top: 4, right: 8, left: -8, bottom: 8 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" className="opacity-40" />
                          <XAxis
                            dataKey="name"
                            tick={{ fontSize: 11 }}
                            tickLine={false}
                            angle={-20}
                            textAnchor="end"
                            height={52}
                            interval={0}
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
                              name === 'r2' ? 'R² Score' : 'Accuracy',
                            ]}
                            labelFormatter={(label, payload) =>
                              payload?.[0]?.payload?.target
                                ? `${label} → Target: ${payload[0].payload.target}`
                                : label
                            }
                          />
                          <Legend
                            formatter={(v) => (v === 'r2' ? 'R² Score (%)' : 'Accuracy (%)')}
                          />
                          <Bar dataKey="r2" name="r2" radius={[4, 4, 0, 0]}>
                            {modelComparisonChart.map((_, i) => (
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
                            fillOpacity={0.5}
                          />
                        </BarChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>
                </motion.div>
              )}

              {/* Full model table */}
              <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
                <Card>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div>
                        <CardTitle>All Models — Detailed Metrics</CardTitle>
                        <CardDescription>
                          Ranked by R² score. CV = cross-validation (generalisation estimate).
                        </CardDescription>
                      </div>
                      <Button variant="outline" size="sm" onClick={() => navigate('/training')}>
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
                            <th className="text-left pb-3 font-medium w-8">#</th>
                            <th className="text-left pb-3 font-medium">Model</th>
                            <th className="text-left pb-3 font-medium">Target Variable</th>
                            <th className="text-right pb-3 font-medium">R²</th>
                            <th className="text-right pb-3 font-medium">Accuracy</th>
                            <th className="text-right pb-3 font-medium">RMSE</th>
                            <th className="text-right pb-3 font-medium">MAE</th>
                            <th className="text-right pb-3 font-medium hidden lg:table-cell">MAPE</th>
                            <th className="text-right pb-3 font-medium hidden xl:table-cell">CV R²</th>
                            <th className="text-right pb-3 font-medium hidden xl:table-cell">CV Acc.</th>
                          </tr>
                        </thead>
                        <tbody>
                          {leaderboard.map((m, i) => (
                            <tr
                              key={m.id}
                              className={cn(
                                'border-b last:border-0 hover:bg-muted/40 transition-colors',
                                i === 0 && 'bg-amber-50/40 dark:bg-amber-900/10'
                              )}
                            >
                              <td className="py-3 pr-2">
                                {i === 0 ? (
                                  <Star className="h-4 w-4 text-amber-500 fill-amber-500" />
                                ) : (
                                  <span className="text-muted-foreground text-xs">{i + 1}</span>
                                )}
                              </td>
                              <td className="py-3">
                                <span
                                  className={cn(
                                    'px-2 py-0.5 rounded text-xs font-medium',
                                    modelTypeBadgeClass(m.type)
                                  )}
                                >
                                  {modelTypeLabel(m.type)}
                                </span>
                              </td>
                              <td className="py-3 text-muted-foreground text-xs max-w-[120px] truncate">
                                {m.config.targetVariable}
                              </td>
                              <td className={cn('py-3 text-right font-semibold tabular-nums', r2ColorClass(m.metrics.r2))}>
                                {(m.metrics.r2 * 100).toFixed(1)}%
                              </td>
                              <td className="py-3 text-right tabular-nums text-muted-foreground">
                                {m.metrics.accuracy.toFixed(1)}%
                              </td>
                              <td className="py-3 text-right tabular-nums text-muted-foreground">
                                {m.metrics.rmse.toFixed(4)}
                              </td>
                              <td className="py-3 text-right tabular-nums text-muted-foreground">
                                {m.metrics.mae.toFixed(4)}
                              </td>
                              <td className="py-3 text-right tabular-nums text-muted-foreground hidden lg:table-cell">
                                {m.metrics.mape.toFixed(2)}%
                              </td>
                              <td className="py-3 text-right tabular-nums text-muted-foreground hidden xl:table-cell">
                                {m.cvMetrics ? `${(m.cvMetrics.mean.r2 * 100).toFixed(1)}%` : '—'}
                              </td>
                              <td className="py-3 text-right tabular-nums text-muted-foreground hidden xl:table-cell">
                                {m.cvMetrics ? `${m.cvMetrics.mean.accuracy.toFixed(1)}%` : '—'}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <div className="mt-4 p-3 bg-muted/40 rounded-lg text-xs text-muted-foreground space-y-1">
                      <p><span className="font-medium text-foreground">R²:</span> Proportion of variance explained (1.0 = perfect, &gt;0.9 = excellent, &gt;0.7 = good)</p>
                      <p><span className="font-medium text-foreground">RMSE/MAE:</span> Average prediction error in the same units as your target variable</p>
                      <p><span className="font-medium text-foreground">CV R²:</span> Cross-validation score — how well the model generalises to unseen data</p>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>

              {/* Feature importance for best model */}
              {featureImportanceData.length > 0 && (
                <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}>
                  <Card>
                    <CardHeader>
                      <CardTitle>Key Process Variables — Best Model</CardTitle>
                      <CardDescription>
                        Top 10 inputs driving predictions for{' '}
                        <span className="font-semibold text-foreground">
                          {bestModel?.config.targetVariable}
                        </span>
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3">
                        {featureImportanceData.map((f, i) => (
                          <div key={f.feature} className="flex items-center gap-3">
                            <span className="text-xs text-muted-foreground w-5 tabular-nums text-right flex-shrink-0">
                              {i + 1}
                            </span>
                            <span className="text-sm min-w-0 flex-1 truncate">{f.feature}</span>
                            <div className="w-40 h-2 bg-muted rounded-full overflow-hidden flex-shrink-0">
                              <div
                                className="h-full rounded-full"
                                style={{
                                  width: `${Math.min(100, (f.importance / featureImportanceData[0].importance) * 100)}%`,
                                  backgroundColor: CHART_COLORS[i % CHART_COLORS.length],
                                }}
                              />
                            </div>
                            <span className="text-xs tabular-nums text-muted-foreground w-14 text-right flex-shrink-0">
                              {f.importance.toFixed(2)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              )}
            </>
          )}
        </TabsContent>

        {/* ── Datasets Tab ── */}
        <TabsContent value="datasets" className="space-y-6">
          {datasets.length === 0 ? (
            <Card className="border-dashed">
              <CardContent className="py-16 text-center">
                <Database className="h-10 w-10 mx-auto text-muted-foreground opacity-40 mb-4" />
                <h3 className="font-semibold mb-2">No datasets uploaded</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Upload your first dataset to see a data quality report here.
                </p>
                <Button onClick={() => navigate('/upload')}>
                  <Upload className="mr-2 h-4 w-4" />
                  Upload Data
                </Button>
              </CardContent>
            </Card>
          ) : (
            <>
              {/* Average health summary */}
              {avgHealthScore !== null && (
                <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
                  <Card
                    className={cn(
                      'border',
                      avgHealthScore >= 80
                        ? 'border-green-200 dark:border-green-800'
                        : avgHealthScore >= 60
                        ? 'border-amber-200 dark:border-amber-800'
                        : 'border-red-200 dark:border-red-800'
                    )}
                  >
                    <CardContent className="p-5">
                      <div className="flex items-center gap-4">
                        <div
                          className={cn(
                            'h-12 w-12 rounded-full flex items-center justify-center text-xl font-bold',
                            avgHealthScore >= 80
                              ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                              : avgHealthScore >= 60
                              ? 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400'
                              : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                          )}
                        >
                          {avgHealthScore}
                        </div>
                        <div>
                          <p className="font-semibold text-lg">
                            Average Data Health: {healthLabel(avgHealthScore)}
                          </p>
                          <p className="text-sm text-muted-foreground">
                            Across {datasets.length} dataset{datasets.length > 1 ? 's' : ''} ·{' '}
                            {totalRows.toLocaleString()} total records
                          </p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              )}

              {/* Dataset cards */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {datasets.map((d, i) => {
                  const colsByCategory = d.mappings.reduce(
                    (acc: Record<string, number>, m) => {
                      acc[m.category] = (acc[m.category] || 0) + 1;
                      return acc;
                    },
                    {} as Record<string, number>
                  );
                  return (
                    <motion.div
                      key={d.id}
                      initial={{ opacity: 0, y: 12 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: i * 0.07 }}
                    >
                      <Card>
                        <CardHeader className="pb-3">
                          <div className="flex items-start justify-between gap-3">
                            <div className="min-w-0">
                              <CardTitle className="text-base truncate">{d.name}</CardTitle>
                              <CardDescription>Uploaded {formatDate(d.uploadDate)}</CardDescription>
                            </div>
                            <span
                              className={cn(
                                'text-sm font-bold px-3 py-1 rounded-full flex-shrink-0',
                                healthBadgeClass(d.healthScore)
                              )}
                            >
                              {d.healthScore}% {healthLabel(d.healthScore)}
                            </span>
                          </div>
                        </CardHeader>
                        <CardContent className="space-y-4">
                          {/* Stats row */}
                          <div className="grid grid-cols-3 gap-3 text-center">
                            {[
                              { label: 'Rows', value: d.rowCount.toLocaleString(), Icon: Layers },
                              { label: 'Columns', value: d.columnCount, Icon: BarChart3 },
                              {
                                label: 'Health',
                                value: `${d.healthScore}%`,
                                Icon: TrendingUp,
                              },
                            ].map((s) => (
                              <div key={s.label} className="bg-muted/40 rounded-lg p-2">
                                <p className="text-xs text-muted-foreground">{s.label}</p>
                                <p className="font-bold text-sm mt-0.5">{s.value}</p>
                              </div>
                            ))}
                          </div>

                          {/* Column category breakdown */}
                          {d.mappings.length > 0 && (
                            <div>
                              <p className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wide">
                                Column Categories
                              </p>
                              <div className="flex flex-wrap gap-2">
                                {(
                                  [
                                    ['identifier', 'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400'],
                                    ['controllable', 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400'],
                                    ['uncontrollable', 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400'],
                                    ['output', 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'],
                                  ] as const
                                ).map(([cat, cls]) =>
                                  colsByCategory[cat] ? (
                                    <span
                                      key={cat}
                                      className={cn('text-xs px-2 py-1 rounded font-medium', cls)}
                                    >
                                      {colsByCategory[cat]} {cat}
                                    </span>
                                  ) : null
                                )}
                              </div>
                              <p className="text-xs text-muted-foreground mt-2">
                                <span className="font-medium text-foreground">Controllable</span> columns
                                are inputs you can adjust. <span className="font-medium text-foreground">Output</span>{' '}
                                columns are what you want to predict.
                              </p>
                            </div>
                          )}

                          <div className="flex gap-2">
                            <Button
                              variant="outline"
                              size="sm"
                              className="flex-1"
                              onClick={() => navigate('/explorer')}
                            >
                              <Database className="mr-2 h-3.5 w-3.5" />
                              Explore
                            </Button>
                            <Button
                              variant="outline"
                              size="sm"
                              className="flex-1"
                              onClick={() => navigate('/training')}
                            >
                              <Cpu className="mr-2 h-3.5 w-3.5" />
                              Train Model
                            </Button>
                          </div>
                        </CardContent>
                      </Card>
                    </motion.div>
                  );
                })}
              </div>
            </>
          )}
        </TabsContent>

        {/* ── History Tab ── */}
        <TabsContent value="history" className="space-y-4">
          <div className="flex items-center justify-between gap-3 flex-wrap">
            <div>
              <h3 className="font-semibold">Prediction History</h3>
              <p className="text-sm text-muted-foreground">
                {predictions.length} prediction{predictions.length !== 1 ? 's' : ''} recorded
                {predictions.length === 200 ? ' (showing latest 200)' : ''}
              </p>
            </div>
            {predictions.length > 0 && (
              <Button variant="outline" size="sm" onClick={() => exportPredictionsToCSV(predictions)}>
                <Download className="mr-2 h-4 w-4" />
                Export CSV
              </Button>
            )}
          </div>

          {predictions.length === 0 ? (
            <Card className="border-dashed">
              <CardContent className="py-16 text-center">
                <Sparkles className="h-10 w-10 mx-auto text-muted-foreground opacity-40 mb-4" />
                <h3 className="font-semibold mb-2">No predictions yet</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Run your first prediction to start building your history log.
                </p>
                <Button onClick={() => navigate('/predictions')}>
                  <Target className="mr-2 h-4 w-4" />
                  Make a Prediction
                </Button>
              </CardContent>
            </Card>
          ) : (
            <>
              {/* Search filter */}
              <div className="relative">
                <input
                  type="text"
                  placeholder="Filter by target variable or model type…"
                  value={historyFilter}
                  onChange={(e) => setHistoryFilter(e.target.value)}
                  className="w-full max-w-sm rounded-md border bg-background px-3 py-2 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                />
              </div>

              <Card>
                <CardContent className="p-0">
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b bg-muted/40">
                          <th className="text-left p-3 font-medium text-muted-foreground">Date & Time</th>
                          <th className="text-left p-3 font-medium text-muted-foreground">Target Variable</th>
                          <th className="text-left p-3 font-medium text-muted-foreground">Model</th>
                          <th className="text-right p-3 font-medium text-muted-foreground">Predicted Value</th>
                        </tr>
                      </thead>
                      <tbody>
                        {filteredHistory.map((p) => (
                          <tr
                            key={p.id}
                            className="border-b last:border-0 hover:bg-muted/30 transition-colors"
                          >
                            <td className="p-3 text-muted-foreground whitespace-nowrap">
                              <div className="flex items-center gap-1.5">
                                <Clock className="h-3.5 w-3.5 flex-shrink-0" />
                                {formatDateTime(p.timestamp)}
                              </div>
                            </td>
                            <td className="p-3 font-medium">{p.targetVariable}</td>
                            <td className="p-3">
                              <span
                                className={cn(
                                  'px-2 py-0.5 rounded text-xs font-medium',
                                  modelTypeBadgeClass(
                                    p.modelType === 'Linear Regression'
                                      ? 'linear'
                                      : p.modelType === 'Random Forest'
                                      ? 'rf'
                                      : 'xgb'
                                  )
                                )}
                              >
                                {p.modelType}
                              </span>
                            </td>
                            <td className="p-3 text-right font-semibold tabular-nums">
                              {p.result.toFixed(4)}
                            </td>
                          </tr>
                        ))}
                        {filteredHistory.length === 0 && (
                          <tr>
                            <td colSpan={4} className="p-8 text-center text-muted-foreground text-sm">
                              No predictions match your filter.
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>
            </>
          )}
        </TabsContent>
      </Tabs>

      {/* Next step prompt */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <Card className="bg-muted/30">
          <CardContent className="p-5 flex items-center justify-between gap-4 flex-wrap">
            <div className="flex items-center gap-3">
              <div className="h-9 w-9 rounded-lg bg-primary/10 flex items-center justify-center">
                <TrendingUp className="h-4 w-4 text-primary" />
              </div>
              <div>
                <p className="font-medium text-sm">Keep optimising</p>
                <p className="text-xs text-muted-foreground">
                  Run more predictions or train additional models to improve accuracy.
                </p>
              </div>
            </div>
            <div className="flex gap-2">
              <Button variant="outline" size="sm" onClick={() => navigate('/training')}>
                <Cpu className="mr-2 h-4 w-4" />
                Train More
              </Button>
              <Button size="sm" onClick={() => navigate('/predictions')}>
                <Sparkles className="mr-2 h-4 w-4" />
                Predict
              </Button>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}
