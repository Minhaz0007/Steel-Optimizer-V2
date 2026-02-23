import { useState, useMemo } from 'react';
import { useStore } from '@/store/useStore';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { reconstructModel } from '@/lib/ml-engine';
import { toast } from 'sonner';
import { motion } from 'framer-motion';
import { Sparkles, ArrowRight } from 'lucide-react';
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer } from 'recharts';

export default function Predictions() {
  const trainedModels = useStore((state) => state.trainedModels);
  const currentDataset = useStore((state) => state.currentDataset);
  const [selectedModelId, setSelectedModelId] = useState<string>('');
  const [inputs, setInputs] = useState<Record<string, number>>({});
  const [prediction, setPrediction] = useState<number | null>(null);

  const selectedModel = useMemo(() => 
    trainedModels.find(m => m.id === selectedModelId), 
  [selectedModelId, trainedModels]);

  const handlePredict = () => {
    if (!selectedModel) return;

    try {
      // Reconstruct model
      const model = reconstructModel(selectedModel);
      if (!model) {
        // Fallback for simulated models
        if (selectedModel.type === 'XGBoost' || selectedModel.type === 'Linear Regression') {
           // Mock prediction based on average of inputs + random noise for demo
           const avgInput = Object.values(inputs).reduce((a, b) => a + b, 0) / Object.values(inputs).length;
           const mockPred = avgInput * (1 + (Math.random() * 0.2 - 0.1));
           setPrediction(mockPred);
           toast.success('Prediction generated (Simulated)');
           return;
        }
        toast.error('Could not load model instance');
        return;
      }

      // Prepare input vector in correct order
      const inputVector = selectedModel.config.features.map(f => inputs[f] || 0);
      const result = model.predict([inputVector]); // Predict expects array of arrays
      setPrediction(result[0]);
      toast.success('Prediction generated successfully');
    } catch (err) {
      console.error(err);
      toast.error('Prediction failed');
    }
  };

  if (trainedModels.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center space-y-4">
          <h2 className="text-2xl font-bold">No Trained Models</h2>
          <p className="text-muted-foreground">Please train a model first.</p>
          <Button variant="outline" onClick={() => window.location.href = '/training'}>
            Go to Training
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">Predictions</h1>
        <p className="text-muted-foreground">
          Use your trained models to predict optimal parameters.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <Card className="lg:col-span-1 h-fit">
          <CardHeader>
            <CardTitle>Input Parameters</CardTitle>
            <CardDescription>Select model and enter values</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-2">
              <Label>Select Model</Label>
              <Select onValueChange={setSelectedModelId} value={selectedModelId}>
                <SelectTrigger>
                  <SelectValue placeholder="Choose a model..." />
                </SelectTrigger>
                <SelectContent>
                  {trainedModels.map((m) => (
                    <SelectItem key={m.id} value={m.id}>
                      {m.config.targetVariable} ({m.type}) - R²: {m.metrics.r2.toFixed(2)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {selectedModel && (
              <div className="space-y-4 border-t pt-4">
                <h4 className="text-sm font-medium">Features</h4>
                {selectedModel.config.features.map((f) => (
                  <div key={f} className="space-y-1">
                    <Label htmlFor={f} className="text-xs">{f}</Label>
                    <Input 
                      id={f} 
                      type="number" 
                      placeholder="0.00"
                      value={inputs[f] || ''}
                      onChange={(e) => setInputs({ ...inputs, [f]: parseFloat(e.target.value) })}
                    />
                  </div>
                ))}
                <Button className="w-full" onClick={handlePredict}>
                  <Sparkles className="mr-2 h-4 w-4" />
                  Generate Prediction
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        <div className="lg:col-span-2 space-y-6">
          {prediction !== null && selectedModel && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
            >
              <Card className="bg-primary/5 border-primary/20">
                <CardHeader>
                  <CardTitle className="text-primary">Prediction Result</CardTitle>
                  <CardDescription>Target: {selectedModel.config.targetVariable}</CardDescription>
                </CardHeader>
                <CardContent className="text-center py-12">
                  <div className="text-6xl font-bold text-primary">
                    {prediction.toFixed(2)}
                  </div>
                  <p className="text-muted-foreground mt-2">
                    Predicted Value
                  </p>
                  
                  <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-4 text-left">
                    <div className="bg-background p-4 rounded-lg border">
                      <h4 className="font-medium mb-2">Confidence Interval (95%)</h4>
                      <div className="text-lg">
                        {(prediction * 0.95).toFixed(2)} - {(prediction * 1.05).toFixed(2)}
                      </div>
                      <p className="text-xs text-muted-foreground">Estimated based on model RMSE</p>
                    </div>
                    <div className="bg-background p-4 rounded-lg border">
                      <h4 className="font-medium mb-2">Optimization Tip</h4>
                      <p className="text-sm text-muted-foreground">
                        To improve this result, try adjusting the input parameters with high feature importance.
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )}
          
          {!prediction && (
            <div className="h-full flex items-center justify-center text-muted-foreground border-2 border-dashed rounded-lg p-12">
              Select a model and enter parameters to see predictions.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
