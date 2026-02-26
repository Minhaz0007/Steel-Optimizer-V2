import { useState, useRef, useEffect } from 'react';
import { useStore } from '@/store/useStore';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Progress } from '@/components/ui/progress';
import { toast } from 'sonner';
import type { TrainedModel, TrainingConfig } from '@/lib/ml-engine';
import { motion } from 'framer-motion';
import { Loader2, Trophy, ArrowRight, TrendingUp, AlertCircle, ChevronDown, ChevronUp } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell
} from 'recharts';
import { useNavigate } from 'react-router-dom';

const MODEL_COLORS: Record<string, string> = {
  'Linear Regression': 'hsl(215, 52%, 45%)',
  'Random Forest':     'hsl(24, 94%, 53%)',
  'Gradient Boosting': 'hsl(183, 95%, 38%)',
};

// ---------------------------------------------------------------------------
// Goal presets — keywords are matched against output column names
// ---------------------------------------------------------------------------
const GOAL_PRESETS = [
  // ── Row 1 — core production goals ──────────────────────────
  {
    id: 'yield',
    icon: '🏭',
    label: 'Maximize Yield',
    description: 'Find the settings that produce the most steel per heat.',
    keywords: ['yield', 'output', 'production', 'throughput', 'ton', 'weight'],
  },
  {
    id: 'energy',
    icon: '⚡',
    label: 'Minimize Energy',
    description: 'Predict power / energy consumption to lower operating cost.',
    keywords: ['energy', 'power', 'kwh', 'consumption', 'electricity', 'fuel', 'specific_energy'],
  },
  {
    id: 'temperature',
    icon: '🌡️',
    label: 'Tap Temperature',
    description: 'Predict furnace or tap temperature for stable process control.',
    keywords: ['temp', 'temperature', 'heat', 'furnace', 'tap', 'tapping', 'superheat'],
  },
  {
    id: 'quality',
    icon: '⭐',
    label: 'Improve Quality',
    description: 'Predict defect rate, hardness, or grade to raise steel quality.',
    keywords: ['quality', 'defect', 'hardness', 'grade', 'tensile', 'strength', 'purity'],
  },
  // ── Row 2 — advanced steel process goals ───────────────────
  {
    id: 'heat_time',
    icon: '⏱️',
    label: 'Reduce Heat Time',
    description: 'Minimise tap-to-tap cycle time to increase furnace throughput.',
    keywords: ['time', 'duration', 'cycle', 'tap-to-tap', 'tap_to_tap', 'ttt', 'minutes', 'taptime'],
  },
  {
    id: 'alloy_cost',
    icon: '💰',
    label: 'Minimize Alloy Cost',
    description: 'Predict alloy and flux consumption to cut raw-material spend.',
    keywords: ['cost', 'alloy', 'lime', 'flux', 'dolomite', 'consumption', 'charge', 'scrap'],
  },
  {
    id: 'carbon',
    icon: '🔩',
    label: 'Carbon Content',
    description: 'Predict endpoint carbon to hit chemistry and grade targets.',
    keywords: ['carbon', 'c%', 'c_pct', 'chemistry', 'composition', 'endpoint', 'analysis', 'carb'],
  },
  {
    id: 'oxygen',
    icon: '💨',
    label: 'Oxygen Blowing',
    description: 'Predict O₂ blow volume or lance height for oxidation control.',
    keywords: ['oxygen', 'o2', 'blow', 'lance', 'nm3', 'oxidation', 'o2_blow', 'blowing'],
  },
] as const;

const STEPS = [
  {
    number: 1,
    title: 'Pick a goal (or configure manually)',
    detail: 'Click one of the goal presets below. It will automatically select the most relevant output column as the target variable. You can also skip presets and choose manually.',
  },
  {
    number: 2,
    title: 'Review the target variable',
    detail: 'The "Target Variable" is what you want the AI to predict (e.g. Yield, Temperature). Confirm the right column is selected in the Configuration panel on the left.',
  },
  {
    number: 3,
    title: 'Choose input features (optional)',
    detail: 'Input features are the process parameters the AI learns from. By default all controllable and uncontrollable numeric columns are used — this is the recommended setting. Only un-check columns you are sure are irrelevant.',
  },
  {
    number: 4,
    title: 'Click "Start Training (3 Models)"',
    detail: 'The app will train three different AI models simultaneously: Linear Regression (fast baseline), Random Forest, and Gradient Boosting. Training usually takes a few seconds.',
  },
  {
    number: 5,
    title: 'Read the results',
    detail: 'After training, the best model is highlighted. Look at the Accuracy % — above 85 % is generally good. Click "Feature Importance" to see which process parameters matter most.',
  },
  {
    number: 6,
    title: 'Go to Predictions',
    detail: 'Once satisfied with the accuracy, click "Go to Predictions" to use the trained models to optimise your plant settings for a target outcome.',
  },
];

function MetricPill({ label, value, unit = '' }: { label: string; value: number | string; unit?: string }) {
  return (
    <div className="text-center p-3 bg-background rounded-lg border">
      <div className="text-xs text-muted-foreground mb-1">{label}</div>
      <div className="text-xl font-bold">{typeof value === 'number' ? value.toFixed(3) : value}{unit}</div>
    </div>
  );
}

function CVMetricPill({ label, mean, std, unit = '' }: { label: string; mean: number; std: number; unit?: string }) {
  return (
    <div className="text-center p-3 bg-background rounded-lg border">
      <div className="text-xs text-muted-foreground mb-1">{label}</div>
      <div className="text-lg font-bold">{mean.toFixed(3)}{unit}</div>
      <div className="text-xs text-muted-foreground">± {std.toFixed(3)}{unit}</div>
    </div>
  );
}

export default function Training() {
  const currentDataset = useStore((state) => state.currentDataset);
  const addTrainedModel = useStore((state) => state.addTrainedModel);
  const navigate = useNavigate();

  const [targetVar, setTargetVar] = useState<string>('');
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressLabel, setProgressLabel] = useState('');
  const [results, setResults] = useState<TrainedModel[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'comparison' | 'importance'>('comparison');
  const [activePreset, setActivePreset] = useState<string | null>(null);
  const [stepsOpen, setStepsOpen] = useState(true);

  // Keep a ref to the active worker so we can terminate it on unmount or cancel
  const workerRef = useRef<Worker | null>(null);
  useEffect(() => () => { workerRef.current?.terminate(); }, []);

  const applyPreset = (presetId: string) => {
    if (!currentDataset) return;
    const preset = GOAL_PRESETS.find(p => p.id === presetId);
    if (!preset) return;

    const outputCols = currentDataset.mappings.filter(m => m.category === 'output');
    // Try to find a column whose name contains any of the preset keywords
    const match = outputCols.find(col =>
      preset.keywords.some(kw => col.columnName.toLowerCase().includes(kw))
    );
    const chosen = match ?? outputCols[0];

    if (!chosen) {
      toast.error('No output columns found. Go to Upload → set at least one column to "Output".');
      return;
    }

    setTargetVar(chosen.columnName);
    setSelectedFeatures([]); // use all features
    setActivePreset(presetId);
    toast.success(`Preset applied — target: "${chosen.columnName}". All features selected. Click Start Training when ready.`);
  };

  const handleTrain = () => {
    if (!currentDataset || !targetVar) return;

    setIsTraining(true);
    setProgress(5);
    setProgressLabel('Preparing data...');
    setError(null);
    setResults([]);

    const features = selectedFeatures.length > 0
      ? selectedFeatures
      : currentDataset.mappings
          .filter(m => (m.category === 'controllable' || m.category === 'uncontrollable') && m.dataType === 'number' && m.columnName !== targetVar)
          .map(m => m.columnName);

    if (features.length === 0) {
      toast.error('No numeric feature columns found. Please mark at least one column as Controllable or Uncontrollable.');
      setIsTraining(false);
      return;
    }

    const config: TrainingConfig = {
      targetVariable: targetVar,
      features,
      testSplit: 0.2,
      models: ['linear', 'rf', 'xgboost'],
    };

    // ── Spin up a Web Worker so training runs on a background thread ────────
    // This keeps the UI fully responsive — no freezing — regardless of
    // dataset size. Vite bundles the worker file automatically.
    workerRef.current?.terminate(); // cancel any previous run
    const worker = new Worker(
      new URL('../lib/training.worker.ts', import.meta.url),
      { type: 'module' }
    );
    workerRef.current = worker;

    worker.onmessage = (e: MessageEvent) => {
      const payload = e.data;
      if (payload.type === 'progress') {
        setProgressLabel(payload.label as string);
        setProgress(payload.pct as number);
      } else if (payload.type === 'result') {
        const trainedModels: TrainedModel[] = payload.models;
        trainedModels.forEach(m => addTrainedModel(m));
        setResults(trainedModels);
        setProgress(100);
        setProgressLabel('Done!');
        setIsTraining(false);
        toast.success(`Training complete! ${trainedModels.length} models ready.`);
        worker.terminate();
      } else if (payload.type === 'error') {
        console.error(payload.message);
        setError(payload.message as string);
        setIsTraining(false);
        setProgress(0);
        toast.error('Training failed');
        worker.terminate();
      }
    };

    worker.onerror = (e) => {
      console.error(e);
      setError(e.message ?? 'Training failed. Check that selected features and target are numeric columns with sufficient data.');
      setIsTraining(false);
      setProgress(0);
      toast.error('Training failed');
      worker.terminate();
    };

    worker.postMessage({ data: currentDataset.data, config });
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

  const outputVars = currentDataset.mappings.filter(m => m.category === 'output');
  const featureVars = currentDataset.mappings.filter(
    m => (m.category === 'controllable' || m.category === 'uncontrollable') && m.columnName !== targetVar && m.dataType === 'number'
  );

  const sortedResults = [...results].sort((a, b) => b.metrics.r2 - a.metrics.r2);
  const bestModel = sortedResults[0];

  // Feature importance data for best model
  const importanceData = bestModel?.featureImportance?.slice(0, 12).map(fi => ({
    name: fi.feature.length > 16 ? fi.feature.substring(0, 14) + '…' : fi.feature,
    fullName: fi.feature,
    importance: parseFloat((fi.importance * 100).toFixed(1)),
  })) ?? [];

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">Model Training</h1>
        <p className="text-muted-foreground">
          Train Linear Regression, Random Forest, and Gradient Boosting on your steel plant data. All three models are saved for predictions.
        </p>
      </div>

      {/* ── Step-by-step guide ─────────────────────────────────────────────── */}
      <Card>
        <button
          type="button"
          className="w-full text-left"
          onClick={() => setStepsOpen(o => !o)}
        >
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <div>
              <CardTitle className="text-base">How to train your first model — step by step</CardTitle>
              <CardDescription className="mt-0.5">New here? Follow these 6 steps.</CardDescription>
            </div>
            {stepsOpen
              ? <ChevronUp className="h-5 w-5 text-muted-foreground shrink-0" />
              : <ChevronDown className="h-5 w-5 text-muted-foreground shrink-0" />}
          </CardHeader>
        </button>

        {stepsOpen && (
          <CardContent className="pt-0">
            <ol className="space-y-3">
              {STEPS.map(step => (
                <li key={step.number} className="flex gap-3">
                  <span className="flex-shrink-0 w-7 h-7 rounded-full bg-primary text-primary-foreground text-sm font-bold flex items-center justify-center">
                    {step.number}
                  </span>
                  <div>
                    <p className="font-medium text-sm">{step.title}</p>
                    <p className="text-xs text-muted-foreground mt-0.5">{step.detail}</p>
                  </div>
                </li>
              ))}
            </ol>
          </CardContent>
        )}
      </Card>

      {/* ── Goal presets ───────────────────────────────────────────────────── */}
      <div className="space-y-3">
        <div>
          <h2 className="text-base font-semibold">Recommended presets — pick your goal</h2>
          <p className="text-xs text-muted-foreground mt-0.5">
            Click a preset to auto-fill the configuration below. You can still adjust anything afterwards.
          </p>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {GOAL_PRESETS.map(preset => {
            const isActive = activePreset === preset.id;
            return (
              <button
                key={preset.id}
                type="button"
                onClick={() => applyPreset(preset.id)}
                className={`
                  rounded-xl border p-4 text-left transition-all hover:shadow-md focus:outline-none focus-visible:ring-2 focus-visible:ring-primary
                  ${isActive
                    ? 'border-primary bg-primary/8 shadow-sm ring-1 ring-primary'
                    : 'border-border bg-card hover:border-primary/40'}
                `}
              >
                <span className="text-2xl">{preset.icon}</span>
                <p className={`mt-2 text-sm font-semibold ${isActive ? 'text-primary' : ''}`}>{preset.label}</p>
                <p className="mt-1 text-xs text-muted-foreground leading-snug">{preset.description}</p>
              </button>
            );
          })}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Config panel */}
        <Card className="lg:col-span-1 h-fit">
          <CardHeader>
            <CardTitle>Configuration</CardTitle>
            <CardDescription>Select target variable and input features</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-2">
              <Label>Target Variable (to predict)</Label>
              <Select onValueChange={(v) => { setTargetVar(v); setSelectedFeatures([]); setActivePreset(null); }} value={targetVar}>
                <SelectTrigger>
                  <SelectValue placeholder="Select output variable..." />
                </SelectTrigger>
                <SelectContent>
                  {outputVars.map(v => (
                    <SelectItem key={v.columnName} value={v.columnName}>
                      {v.columnName}
                      <span className="ml-1 text-xs text-muted-foreground">({v.category})</span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label>
                Input Features
                <span className="ml-1 text-xs text-muted-foreground font-normal">(controllable + uncontrollable numeric)</span>
              </Label>
              <div className="border rounded-md p-3 max-h-52 overflow-y-auto space-y-2">
                {featureVars.length === 0 && (
                  <p className="text-xs text-muted-foreground text-center py-4">
                    No numeric feature columns found. Adjust column categories in the Upload page.
                  </p>
                )}
                {featureVars.map(v => (
                  <div key={v.columnName} className="flex items-center space-x-2">
                    <Checkbox
                      id={v.columnName}
                      checked={selectedFeatures.includes(v.columnName)}
                      onCheckedChange={(checked) => {
                        if (checked) setSelectedFeatures(prev => [...prev, v.columnName]);
                        else setSelectedFeatures(prev => prev.filter(f => f !== v.columnName));
                      }}
                    />
                    <label htmlFor={v.columnName} className="text-sm leading-none cursor-pointer flex items-center gap-1.5">
                      {v.columnName}
                      <span className={`text-[10px] px-1 py-0.5 rounded font-medium ${
                        v.category === 'controllable'
                          ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300'
                          : 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400'
                      }`}>
                        {v.category === 'controllable' ? 'ctrl' : 'unctrl'}
                      </span>
                    </label>
                  </div>
                ))}
              </div>
              <p className="text-xs text-muted-foreground">
                Leave all unchecked to auto-select all controllable + uncontrollable numeric columns.
              </p>
            </div>

            <Button className="w-full" onClick={handleTrain} disabled={!targetVar || isTraining}>
              {isTraining ? (
                <><Loader2 className="mr-2 h-4 w-4 animate-spin" />Training...</>
              ) : (
                'Start Training (3 Models)'
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Results panel */}
        <div className="lg:col-span-2 space-y-6">
          {/* Error */}
          {error && (
            <Card className="border-destructive/40 bg-destructive/5">
              <CardContent className="p-4 flex items-start gap-3">
                <AlertCircle className="h-5 w-5 text-destructive flex-shrink-0 mt-0.5" />
                <div>
                  <p className="font-medium text-destructive">Training Failed</p>
                  <p className="text-sm text-muted-foreground mt-1">{error}</p>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Progress */}
          {isTraining && (
            <Card>
              <CardContent className="p-10 text-center space-y-4">
                <h3 className="text-xl font-semibold">Training in Progress</h3>
                <Progress value={progress} className="h-2 w-full max-w-md mx-auto" />
                <p className="text-sm text-muted-foreground">{progressLabel}</p>
              </CardContent>
            </Card>
          )}

          {/* Results */}
          {!isTraining && results.length > 0 && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-5">

              {/* Best model highlight */}
              <Card className="bg-primary/5 border-primary/20">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-primary">
                    <Trophy className="h-5 w-5" />
                    Best Model: {bestModel.type}
                  </CardTitle>
                  <CardDescription>Target: <strong>{bestModel.config.targetVariable}</strong> · {bestModel.config.features.length} features</CardDescription>
                </CardHeader>
                <CardContent>
                  {/* Train / Test metrics */}
                  <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">Train / Test split (80 / 20)</p>
                  <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
                    <MetricPill label="R² Score" value={bestModel.metrics.r2} />
                    <MetricPill label="RMSE" value={bestModel.metrics.rmse} />
                    <MetricPill label="MAE" value={bestModel.metrics.mae} />
                    <MetricPill label="MAPE" value={bestModel.metrics.mape} unit="%" />
                    <MetricPill label="Accuracy" value={bestModel.metrics.accuracy.toFixed(1)} unit="%" />
                  </div>

                  {/* Cross-Validation metrics */}
                  {bestModel.cvMetrics && (
                    <div className="mt-4 p-3 rounded-lg bg-muted/40 border">
                      <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
                        5-Fold Cross-Validation — reliability check
                      </p>
                      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
                        <CVMetricPill label="CV R²"       mean={bestModel.cvMetrics.mean.r2}       std={bestModel.cvMetrics.std.r2} />
                        <CVMetricPill label="CV RMSE"     mean={bestModel.cvMetrics.mean.rmse}     std={bestModel.cvMetrics.std.rmse} />
                        <CVMetricPill label="CV MAE"      mean={bestModel.cvMetrics.mean.mae}      std={bestModel.cvMetrics.std.mae} />
                        <CVMetricPill label="CV MAPE"     mean={bestModel.cvMetrics.mean.mape}     std={bestModel.cvMetrics.std.mape} unit="%" />
                        <CVMetricPill label="CV Accuracy" mean={bestModel.cvMetrics.mean.accuracy} std={bestModel.cvMetrics.std.accuracy} unit="%" />
                      </div>
                      <p className="text-xs text-muted-foreground mt-2">
                        Mean ± std across {bestModel.cvMetrics.folds} folds.
                        Low std = stable model. CV R² close to Train R² = no overfitting.
                        {bestModel.metrics.r2 - bestModel.cvMetrics.mean.r2 > 0.1 && (
                          <span className="text-amber-600 dark:text-amber-400 font-medium"> ⚠ Gap &gt;0.10 detected — model may be overfitting.</span>
                        )}
                      </p>
                    </div>
                  )}

                  <p className="text-xs text-muted-foreground mt-3">
                    * Accuracy = 100 − MAPE. R² closest to 1.0 is ideal. All 3 models are saved for predictions.
                  </p>
                </CardContent>
              </Card>

              {/* Tab toggle */}
              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant={activeTab === 'comparison' ? 'default' : 'outline'}
                  onClick={() => setActiveTab('comparison')}
                >
                  Model Comparison
                </Button>
                <Button
                  size="sm"
                  variant={activeTab === 'importance' ? 'default' : 'outline'}
                  onClick={() => setActiveTab('importance')}
                >
                  Feature Importance
                </Button>
              </div>

              {/* Model comparison chart */}
              {activeTab === 'comparison' && (
                <Card>
                  <CardHeader>
                    <CardTitle>All Models — R² Score & Accuracy</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="h-[260px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={sortedResults.map(r => ({
                            name: r.type,
                            'R²': parseFloat(r.metrics.r2.toFixed(3)),
                            'Accuracy %': parseFloat((r.metrics.accuracy / 100).toFixed(3)),
                          }))}
                          margin={{ top: 4, right: 16, left: 0, bottom: 4 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                          <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                          <YAxis domain={[0, 1]} tick={{ fontSize: 12 }} />
                          <Tooltip
                            contentStyle={{ backgroundColor: 'hsl(var(--card))', borderColor: 'hsl(var(--border))' }}
                            formatter={(val: any) => [(Number(val) * (val < 1.01 ? 100 : 1)).toFixed(1) + '%', '']}
                          />
                          <Bar dataKey="R²" radius={[4, 4, 0, 0]}>
                            {sortedResults.map(r => (
                              <Cell key={r.id} fill={MODEL_COLORS[r.type] ?? 'hsl(var(--secondary))'} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                    {/* Metrics table */}
                    <div className="mt-4 rounded-md border overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead className="bg-muted/50">
                          <tr>
                            {['Model', 'R²', 'CV R² (mean±std)', 'RMSE', 'MAE', 'MAPE', 'Accuracy'].map(h => (
                              <th key={h} className="h-9 px-3 text-left font-medium text-muted-foreground whitespace-nowrap">{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {sortedResults.map((r, i) => (
                            <tr key={r.id} className={i === 0 ? 'bg-primary/5 font-medium' : 'hover:bg-muted/30'}>
                              <td className="px-3 py-2 whitespace-nowrap">{i === 0 ? '🏆 ' : ''}{r.type}</td>
                              <td className="px-3 py-2">{r.metrics.r2.toFixed(4)}</td>
                              <td className="px-3 py-2 whitespace-nowrap">
                                {r.cvMetrics
                                  ? <><span className="font-medium">{r.cvMetrics.mean.r2.toFixed(3)}</span><span className="text-muted-foreground"> ±{r.cvMetrics.std.r2.toFixed(3)}</span></>
                                  : '—'}
                              </td>
                              <td className="px-3 py-2">{r.metrics.rmse.toFixed(3)}</td>
                              <td className="px-3 py-2">{r.metrics.mae.toFixed(3)}</td>
                              <td className="px-3 py-2">{r.metrics.mape.toFixed(2)}%</td>
                              <td className="px-3 py-2 font-semibold text-primary">{r.metrics.accuracy.toFixed(1)}%</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Feature importance chart */}
              {activeTab === 'importance' && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <TrendingUp className="h-5 w-5" />
                      Feature Importance — {bestModel.type}
                    </CardTitle>
                    <CardDescription>
                      Permutation importance: how much model accuracy drops when each feature is shuffled. Higher = more important to model predictions.
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    {importanceData.length > 0 ? (
                      <div className="h-[320px]">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            data={importanceData}
                            layout="vertical"
                            margin={{ top: 4, right: 40, left: 8, bottom: 4 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" opacity={0.3} horizontal={false} />
                            <XAxis type="number" unit="%" tick={{ fontSize: 11 }} />
                            <YAxis dataKey="name" type="category" width={120} tick={{ fontSize: 11 }} />
                            <Tooltip
                              contentStyle={{ backgroundColor: 'hsl(var(--card))', borderColor: 'hsl(var(--border))' }}
                              formatter={(val: any) => [val + '%', 'Importance']}
                              labelFormatter={(label: string) => {
                                const item = importanceData.find(d => d.name === label);
                                return item?.fullName ?? label;
                              }}
                            />
                            <Bar dataKey="importance" fill="hsl(var(--secondary))" radius={[0, 4, 4, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    ) : (
                      <p className="text-sm text-muted-foreground text-center py-8">Feature importance not available for this model.</p>
                    )}
                  </CardContent>
                </Card>
              )}

              <div className="flex justify-end">
                <Button size="lg" onClick={() => navigate('/predictions')}>
                  Go to Predictions <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </div>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
}
