import { useState } from 'react';
import { useStore } from '@/store/useStore';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { toast } from 'sonner';
import { motion } from 'framer-motion';
import { Sparkles, TrendingUp, AlertCircle, AlertTriangle, CheckCircle2, Settings2 } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { v4 as uuidv4 } from 'uuid';
import type { OptimizationRecord } from '@/lib/ml-engine';

// Context variable definitions (uncontrollable inputs passed to the optimizer)
const CONTEXT_FIELDS: Array<{
  key: string;
  label: string;
  type: 'number' | 'select';
  options?: string[];
  placeholder: string;
}> = [
  { key: 'shift_type', label: 'Shift Type', type: 'select', options: ['A', 'B', 'C'], placeholder: 'Select shift' },
  { key: 'ambient_temperature_c', label: 'Ambient Temperature (°C)', type: 'number', placeholder: '25.0' },
  { key: 'raw_material_quality_index', label: 'Raw Material Quality Index', type: 'number', placeholder: '0.85' },
  { key: 'operator_experience_years', label: 'Operator Experience (years)', type: 'number', placeholder: '5' },
  { key: 'scheduled_maintenance_due', label: 'Maintenance Due', type: 'select', options: ['0', '1'], placeholder: 'Select' },
  { key: 'grade_change_flag', label: 'Grade Change Flag', type: 'select', options: ['0', '1'], placeholder: 'Select' },
];

const OUTCOME_META: Record<string, { label: string; unit: string; emoji: string }> = {
  yield_pct:           { label: 'Yield',            unit: '%', emoji: '⚙️' },
  steel_output_tons:   { label: 'Steel Output',      unit: 't', emoji: '🏭' },
  energy_cost_usd:     { label: 'Energy Cost',       unit: '$', emoji: '⚡' },
  production_cost_usd: { label: 'Production Cost',   unit: '$', emoji: '💰' },
  scrap_rate_pct:      { label: 'Scrap Rate',        unit: '%', emoji: '♻️' },
};

export default function Predictions() {
  const trainingSession = useStore((s) => s.trainingSession);
  const addOptimizationRecord = useStore((s) => s.addOptimizationRecord);
  const navigate = useNavigate();

  const [context, setContext] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<OptimizationRecord | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleRunOptimization = async () => {
    setError(null);
    setResult(null);

    const missing = CONTEXT_FIELDS.filter(
      (f) => context[f.key] === undefined || context[f.key] === ''
    );
    if (missing.length > 0) {
      setError(`Please fill in: ${missing.map((f) => f.label).join(', ')}`);
      return;
    }

    const contextDict: Record<string, number | string> = {};
    for (const f of CONTEXT_FIELDS) {
      const raw = context[f.key];
      contextDict[f.key] = f.type === 'number' ? parseFloat(raw) : raw;
    }

    setLoading(true);
    try {
      const res = await fetch('/api/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ context: contextDict }),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Server error ${res.status}`);
      }

      const data = await res.json();

      const record: OptimizationRecord = {
        id: uuidv4(),
        timestamp: new Date().toISOString(),
        context: contextDict,
        isAnomaly: data.is_anomaly ?? false,
        anomalyScore: data.anomaly_score ?? 0,
        anomalyLabel: data.anomaly_label ?? 'Unknown',
        recommendedSetpoints: data.recommended_setpoints ?? {},
        predictedOutcomes: data.predicted_outcomes ?? {},
        qualityPassProbability: data.quality_pass_probability ?? 0,
        reworkProbability: data.rework_probability ?? 0,
        compositeScore: data.composite_score ?? 0,
        warnings: data.warnings ?? [],
      };

      setResult(record);
      addOptimizationRecord(record);
      toast.success('Optimization complete');
    } catch (e: any) {
      const msg = e?.message ?? 'Optimization failed';
      setError(msg);
      toast.error('Optimization failed');
    } finally {
      setLoading(false);
    }
  };

  if (!trainingSession) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center space-y-4">
          <h2 className="text-2xl font-bold">No Trained Models</h2>
          <p className="text-muted-foreground">
            Run the Python ML pipeline on the Training page first.
          </p>
          <Button onClick={() => navigate('/training')}>Go to Training</Button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">Optimization</h1>
        <p className="text-muted-foreground">
          Enter current plant context — Optuna Bayesian optimization finds the best controllable setpoints
          using your trained LightGBM and CatBoost models.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* ── Input panel ── */}
        <Card className="lg:col-span-1 h-fit">
          <CardHeader>
            <CardTitle>Context Variables</CardTitle>
            <CardDescription>Uncontrollable conditions for this shift</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {CONTEXT_FIELDS.map((f) => (
              <div key={f.key} className="space-y-1.5">
                <Label htmlFor={f.key} className="text-xs font-medium">
                  {f.label}
                </Label>
                {f.type === 'select' ? (
                  <Select
                    value={context[f.key] ?? ''}
                    onValueChange={(v) =>
                      setContext((prev) => ({ ...prev, [f.key]: v }))
                    }
                  >
                    <SelectTrigger id={f.key}>
                      <SelectValue placeholder={f.placeholder} />
                    </SelectTrigger>
                    <SelectContent>
                      {f.options!.map((opt) => (
                        <SelectItem key={opt} value={opt}>
                          {opt === '0' ? 'No' : opt === '1' ? 'Yes' : opt}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                ) : (
                  <Input
                    id={f.key}
                    type="number"
                    step="any"
                    placeholder={f.placeholder}
                    value={context[f.key] ?? ''}
                    onChange={(e) =>
                      setContext((prev) => ({ ...prev, [f.key]: e.target.value }))
                    }
                  />
                )}
              </div>
            ))}

            {error && (
              <div className="flex items-start gap-2 text-xs text-destructive bg-destructive/10 rounded p-2">
                <AlertCircle className="h-3.5 w-3.5 flex-shrink-0 mt-0.5" />
                {error}
              </div>
            )}

            <Button
              className="w-full mt-2"
              onClick={handleRunOptimization}
              disabled={loading}
            >
              {loading ? (
                <>
                  <span className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent inline-block" />
                  Optimizing…
                </>
              ) : (
                <>
                  <Sparkles className="mr-2 h-4 w-4" />
                  Run Optimization
                </>
              )}
            </Button>

            <p className="text-xs text-muted-foreground text-center">
              300 Optuna TPE trials · quality constraint P≥0.85
            </p>
          </CardContent>
        </Card>

        {/* ── Results panel ── */}
        <div className="lg:col-span-2 space-y-5">
          {loading && (
            <div className="h-64 flex flex-col items-center justify-center gap-4">
              <div className="h-10 w-10 rounded-full border-4 border-primary border-t-transparent animate-spin" />
              <p className="text-sm text-muted-foreground">
                Running Bayesian optimization (300 Optuna trials)…
              </p>
            </div>
          )}

          {!loading && !result && (
            <div className="h-64 flex flex-col items-center justify-center text-muted-foreground border-2 border-dashed rounded-lg gap-3">
              <TrendingUp className="h-8 w-8 opacity-30" />
              <p className="text-sm">Fill in context variables and click Run Optimization</p>
            </div>
          )}

          {!loading && result && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-5"
            >
              {/* Anomaly / health banner */}
              <Card
                className={
                  result.isAnomaly
                    ? 'border-red-300 dark:border-red-800 bg-red-50/40 dark:bg-red-900/10'
                    : 'border-green-300 dark:border-green-800 bg-green-50/40 dark:bg-green-900/10'
                }
              >
                <CardContent className="p-4 flex items-center gap-3">
                  {result.isAnomaly ? (
                    <AlertTriangle className="h-5 w-5 text-red-500 flex-shrink-0" />
                  ) : (
                    <CheckCircle2 className="h-5 w-5 text-green-600 dark:text-green-400 flex-shrink-0" />
                  )}
                  <div>
                    <p
                      className={`font-semibold text-sm ${
                        result.isAnomaly
                          ? 'text-red-600 dark:text-red-400'
                          : 'text-green-700 dark:text-green-400'
                      }`}
                    >
                      {result.anomalyLabel}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Isolation Forest score: {result.anomalyScore.toFixed(4)}
                    </p>
                  </div>
                  <div className="ml-auto text-right">
                    <p className="text-xs text-muted-foreground">Composite Score</p>
                    <p className="text-xl font-bold tabular-nums">
                      {(result.compositeScore * 100).toFixed(1)}
                    </p>
                  </div>
                </CardContent>
              </Card>

              {/* Warnings */}
              {result.warnings.length > 0 && (
                <Card className="border-amber-200 dark:border-amber-800">
                  <CardContent className="p-4 space-y-1.5">
                    {result.warnings.map((w, i) => (
                      <div key={i} className="flex items-start gap-2 text-sm">
                        <AlertTriangle className="h-4 w-4 text-amber-500 flex-shrink-0 mt-0.5" />
                        <span className="text-amber-700 dark:text-amber-400">{w}</span>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              )}

              {/* Predicted outcomes */}
              <div>
                <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide mb-3">
                  Predicted Outcomes
                </h3>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                  {Object.entries(OUTCOME_META).map(([key, meta]) => {
                    const val = result.predictedOutcomes[key];
                    if (val === undefined) return null;
                    return (
                      <Card key={key} className="bg-muted/30">
                        <CardContent className="p-4">
                          <p className="text-xs text-muted-foreground">
                            {meta.emoji} {meta.label}
                          </p>
                          <p className="text-2xl font-bold tabular-nums mt-1">
                            {val.toFixed(2)}
                            <span className="text-sm font-normal text-muted-foreground ml-1">
                              {meta.unit}
                            </span>
                          </p>
                        </CardContent>
                      </Card>
                    );
                  })}

                  {/* Quality pass probability */}
                  <Card className="bg-muted/30">
                    <CardContent className="p-4">
                      <p className="text-xs text-muted-foreground">✅ Quality Pass</p>
                      <p className="text-2xl font-bold tabular-nums mt-1 text-green-600 dark:text-green-400">
                        {(result.qualityPassProbability * 100).toFixed(1)}
                        <span className="text-sm font-normal text-muted-foreground ml-1">%</span>
                      </p>
                    </CardContent>
                  </Card>

                  {/* Rework probability */}
                  <Card className="bg-muted/30">
                    <CardContent className="p-4">
                      <p className="text-xs text-muted-foreground">🔁 Rework Risk</p>
                      <p className="text-2xl font-bold tabular-nums mt-1 text-amber-600 dark:text-amber-400">
                        {(result.reworkProbability * 100).toFixed(1)}
                        <span className="text-sm font-normal text-muted-foreground ml-1">%</span>
                      </p>
                    </CardContent>
                  </Card>
                </div>
              </div>

              {/* Recommended setpoints */}
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center gap-2">
                    <Settings2 className="h-4 w-4 text-primary" />
                    Recommended Setpoints
                  </CardTitle>
                  <CardDescription>
                    Optuna-optimized controllable parameters for this shift
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b text-muted-foreground">
                          <th className="text-left pb-2 font-medium">Parameter</th>
                          <th className="text-right pb-2 font-medium">Recommended Value</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(result.recommendedSetpoints).map(([k, v]) => (
                          <tr key={k} className="border-b last:border-0">
                            <td className="py-2.5 font-mono text-xs text-muted-foreground">
                              {k}
                            </td>
                            <td className="py-2.5 text-right font-semibold tabular-nums">
                              {v.toFixed(3)}
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
        </div>
      </div>
    </div>
  );
}
