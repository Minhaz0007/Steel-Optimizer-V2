import { useState, useMemo } from 'react';
import { useStore } from '@/store/useStore';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { predictWithModel } from '@/lib/ml-engine';
import { toast } from 'sonner';
import { motion } from 'framer-motion';
import { Sparkles, TrendingUp, AlertCircle, Info } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell
} from 'recharts';
import { useNavigate } from 'react-router-dom';

// Score color: green for high, orange for mid, red for low
function accuracyColor(acc: number) {
  if (acc >= 85) return 'text-green-600 dark:text-green-400';
  if (acc >= 65) return 'text-amber-600 dark:text-amber-400';
  return 'text-red-500';
}

export default function Predictions() {
  const trainedModels = useStore((state) => state.trainedModels);
  const [selectedModelId, setSelectedModelId] = useState<string>('');
  const [inputs, setInputs] = useState<Record<string, string>>({});
  const [prediction, setPrediction] = useState<number | null>(null);
  const [predError, setPredError] = useState<string | null>(null);
  const navigate = useNavigate();

  const selectedModel = useMemo(
    () => trainedModels.find(m => m.id === selectedModelId),
    [selectedModelId, trainedModels]
  );

  const handlePredict = () => {
    if (!selectedModel) return;
    setPredError(null);

    // Validate all inputs
    const missing = selectedModel.config.features.filter(f => inputs[f] === '' || inputs[f] === undefined);
    if (missing.length > 0) {
      setPredError(`Please enter values for: ${missing.join(', ')}`);
      return;
    }

    const inputVector = selectedModel.config.features.map(f => parseFloat(inputs[f] ?? '0'));
    if (inputVector.some(v => isNaN(v))) {
      setPredError('All input values must be valid numbers.');
      return;
    }

    const result = predictWithModel(selectedModel, inputVector);
    if (result === null) {
      setPredError('Model could not generate a prediction. The model may need to be retrained.');
      toast.error('Prediction failed');
      return;
    }

    setPrediction(result);
    toast.success('Prediction generated');
  };

  // 95% CI using model RMSE: prediction ± 1.96 * RMSE
  const ciLow  = prediction !== null && selectedModel ? prediction - 1.96 * selectedModel.metrics.rmse : null;
  const ciHigh = prediction !== null && selectedModel ? prediction + 1.96 * selectedModel.metrics.rmse : null;

  // Top-5 most important features for optimization tips
  const topFeatures = useMemo(() => {
    if (!selectedModel?.featureImportance) return [];
    return selectedModel.featureImportance.slice(0, 5);
  }, [selectedModel]);

  // Feature importance chart data
  const importanceChartData = useMemo(() => {
    if (!selectedModel?.featureImportance) return [];
    return selectedModel.featureImportance.slice(0, 8).map(fi => ({
      name: fi.feature.length > 14 ? fi.feature.substring(0, 12) + '…' : fi.feature,
      fullName: fi.feature,
      importance: parseFloat((fi.importance * 100).toFixed(1)),
    }));
  }, [selectedModel]);

  if (trainedModels.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center space-y-4">
          <h2 className="text-2xl font-bold">No Trained Models</h2>
          <p className="text-muted-foreground">Please train a model first on the Training page.</p>
          <Button onClick={() => navigate('/training')}>Go to Training</Button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">Predictions</h1>
        <p className="text-muted-foreground">
          Enter controllable process parameters to predict the target output and get optimization recommendations.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Input panel */}
        <Card className="lg:col-span-1 h-fit">
          <CardHeader>
            <CardTitle>Input Parameters</CardTitle>
            <CardDescription>Select model and enter process values</CardDescription>
          </CardHeader>
          <CardContent className="space-y-5">
            {/* Model selector */}
            <div className="space-y-2">
              <Label>Select Model</Label>
              <Select
                onValueChange={(v) => {
                  setSelectedModelId(v);
                  setInputs({});
                  setPrediction(null);
                  setPredError(null);
                }}
                value={selectedModelId}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Choose a model..." />
                </SelectTrigger>
                <SelectContent>
                  {trainedModels.map(m => (
                    <SelectItem key={m.id} value={m.id}>
                      <span className="font-medium">{m.config.targetVariable}</span>
                      <span className="ml-1 text-muted-foreground text-xs">({m.type}) R²: {m.metrics.r2.toFixed(3)} · Acc: {m.metrics.accuracy.toFixed(1)}%</span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Model accuracy badge */}
            {selectedModel && (
              <div className="flex items-center gap-2 px-3 py-2 bg-muted/50 rounded-lg text-sm">
                <Info className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                <span className="text-muted-foreground">Model accuracy: </span>
                <span className={`font-bold ${accuracyColor(selectedModel.metrics.accuracy)}`}>
                  {selectedModel.metrics.accuracy.toFixed(1)}%
                </span>
                <span className="text-muted-foreground">(±{selectedModel.metrics.rmse.toFixed(2)} RMSE)</span>
              </div>
            )}

            {/* Feature inputs */}
            {selectedModel && (
              <div className="space-y-3 border-t pt-4">
                <h4 className="text-sm font-medium">Controllable Inputs</h4>
                {selectedModel.config.features.map(f => {
                  const rank = selectedModel.featureImportance?.findIndex(fi => fi.feature === f) ?? -1;
                  const isTop3 = rank >= 0 && rank < 3;
                  return (
                    <div key={f} className="space-y-1">
                      <Label htmlFor={f} className="text-xs flex items-center gap-1">
                        {f}
                        {isTop3 && (
                          <span className="text-[10px] bg-secondary/20 text-secondary-foreground px-1 rounded">
                            top feature
                          </span>
                        )}
                      </Label>
                      <Input
                        id={f}
                        type="number"
                        placeholder="Enter value"
                        value={inputs[f] ?? ''}
                        onChange={(e) => setInputs(prev => ({ ...prev, [f]: e.target.value }))}
                        className={isTop3 ? 'border-secondary/50 focus:border-secondary' : ''}
                      />
                    </div>
                  );
                })}

                {predError && (
                  <div className="flex items-start gap-2 text-xs text-destructive bg-destructive/10 rounded p-2">
                    <AlertCircle className="h-3.5 w-3.5 flex-shrink-0 mt-0.5" />
                    {predError}
                  </div>
                )}

                <Button className="w-full mt-2" onClick={handlePredict}>
                  <Sparkles className="mr-2 h-4 w-4" />
                  Generate Prediction
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Results panel */}
        <div className="lg:col-span-2 space-y-5">
          {/* Prediction result */}
          {prediction !== null && selectedModel ? (
            <motion.div initial={{ opacity: 0, scale: 0.97 }} animate={{ opacity: 1, scale: 1 }}>
              <Card className="bg-primary/5 border-primary/20">
                <CardHeader>
                  <CardTitle className="text-primary">Prediction Result</CardTitle>
                  <CardDescription>
                    Target: <strong>{selectedModel.config.targetVariable}</strong> · Model: {selectedModel.type}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {/* Main value */}
                  <div className="text-center py-6">
                    <div className="text-7xl font-bold text-primary">
                      {prediction.toFixed(2)}
                    </div>
                    <p className="text-muted-foreground mt-2">Predicted {selectedModel.config.targetVariable}</p>
                  </div>

                  {/* Metrics row */}
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-2">
                    <div className="bg-background rounded-lg border p-4">
                      <h4 className="text-sm font-medium mb-2">95% Confidence Interval</h4>
                      <div className="text-base font-semibold">
                        {ciLow!.toFixed(2)} — {ciHigh!.toFixed(2)}
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">Prediction ± 1.96 × RMSE ({selectedModel.metrics.rmse.toFixed(3)})</p>
                    </div>
                    <div className="bg-background rounded-lg border p-4">
                      <h4 className="text-sm font-medium mb-2">Model Accuracy</h4>
                      <div className={`text-2xl font-bold ${accuracyColor(selectedModel.metrics.accuracy)}`}>
                        {selectedModel.metrics.accuracy.toFixed(1)}%
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">100 − MAPE on held-out test set</p>
                    </div>
                    <div className="bg-background rounded-lg border p-4">
                      <h4 className="text-sm font-medium mb-2">Model R² Score</h4>
                      <div className="text-2xl font-bold">
                        {selectedModel.metrics.r2.toFixed(4)}
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">Variance explained by model</p>
                    </div>
                  </div>

                  {/* Optimization tips */}
                  {topFeatures.length > 0 && (
                    <div className="mt-5 bg-background rounded-lg border p-4">
                      <h4 className="font-medium mb-3 flex items-center gap-2">
                        <TrendingUp className="h-4 w-4 text-secondary" />
                        Optimization Recommendations
                      </h4>
                      <p className="text-xs text-muted-foreground mb-3">
                        These controllable parameters have the highest impact on <strong>{selectedModel.config.targetVariable}</strong>. Fine-tuning them will produce the greatest improvement in future batches.
                      </p>
                      <div className="space-y-2">
                        {topFeatures.map((fi, i) => (
                          <div key={fi.feature} className="flex items-center gap-3 text-sm">
                            <span className="w-5 h-5 rounded-full bg-secondary/20 text-secondary-foreground text-xs flex items-center justify-center font-bold flex-shrink-0">
                              {i + 1}
                            </span>
                            <span className="font-medium flex-1">{fi.feature}</span>
                            <div className="w-24 bg-muted rounded-full h-2">
                              <div
                                className="h-2 rounded-full bg-secondary"
                                style={{ width: `${Math.min(100, fi.importance * 100)}%` }}
                              />
                            </div>
                            <span className="text-muted-foreground text-xs w-10 text-right">
                              {(fi.importance * 100).toFixed(1)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            </motion.div>
          ) : (
            !predError && (
              <div className="h-48 flex items-center justify-center text-muted-foreground border-2 border-dashed rounded-lg">
                {selectedModel
                  ? 'Enter all parameter values and click Generate Prediction.'
                  : 'Select a model to begin.'}
              </div>
            )
          )}

          {/* Feature importance chart (always visible when model is selected) */}
          {selectedModel && importanceChartData.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Feature Importance — {selectedModel.type}</CardTitle>
                <CardDescription>Which inputs drive {selectedModel.config.targetVariable} the most</CardDescription>
              </CardHeader>
              <CardContent className="h-[240px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={importanceChartData}
                    layout="vertical"
                    margin={{ top: 4, right: 36, left: 8, bottom: 4 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" opacity={0.3} horizontal={false} />
                    <XAxis type="number" unit="%" tick={{ fontSize: 10 }} />
                    <YAxis dataKey="name" type="category" width={110} tick={{ fontSize: 10 }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: 'hsl(var(--card))', borderColor: 'hsl(var(--border))' }}
                      formatter={(val: any) => [val + '%', 'Importance']}
                      labelFormatter={(label) => importanceChartData.find(d => d.name === label)?.fullName ?? label}
                    />
                    <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                      {importanceChartData.map((_, i) => (
                        <Cell key={i} fill={i < 3 ? 'hsl(var(--secondary))' : 'hsl(var(--primary))'} opacity={1 - i * 0.06} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
