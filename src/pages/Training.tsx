import { useState } from 'react';
import { useStore } from '@/store/useStore';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Progress } from '@/components/ui/progress';
import { toast } from 'sonner';
import { trainModels, TrainedModel } from '@/lib/ml-engine';
import { motion } from 'framer-motion';
import { Loader2, Trophy, ArrowRight } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function Training() {
  const currentDataset = useStore((state) => state.currentDataset);
  const [targetVar, setTargetVar] = useState<string>('');
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState<TrainedModel[]>([]);

  const addTrainedModel = useStore((state) => state.addTrainedModel);

  const handleTrain = async () => {
    if (!currentDataset || !targetVar) return;

    setIsTraining(true);
    setProgress(10);

    try {
      // Simulate steps
      setTimeout(() => setProgress(30), 500); // Preprocessing
      
      const config = {
        targetVariable: targetVar,
        features: selectedFeatures.length > 0 ? selectedFeatures : currentDataset.mappings.filter(m => m.category === 'controllable').map(m => m.columnName),
        testSplit: 0.2,
        models: ['linear', 'rf', 'xgboost']
      };

      // Run training (async to allow UI update)
      setTimeout(async () => {
        setProgress(60); // Training
        const trainedModels = await trainModels(currentDataset.data, config);
        setResults(trainedModels);
        
        // Save best model to store
        const bestModel = trainedModels.sort((a, b) => b.metrics.r2 - a.metrics.r2)[0];
        addTrainedModel(bestModel);

        setProgress(100);
        setIsTraining(false);
        toast.success('Training complete!');
      }, 1000);

    } catch (err) {
      console.error(err);
      toast.error('Training failed');
      setIsTraining(false);
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

  const controllableVars = currentDataset.mappings.filter(m => m.category === 'controllable' || m.category === 'output');
  const featureVars = currentDataset.mappings.filter(m => m.category !== 'identifier' && m.columnName !== targetVar);

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">Model Training</h1>
        <p className="text-muted-foreground">
          Configure and train machine learning models to predict your target variables.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Configuration Panel */}
        <Card className="lg:col-span-1 h-fit">
          <CardHeader>
            <CardTitle>Configuration</CardTitle>
            <CardDescription>Select target and features</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-2">
              <Label>Target Variable (Output)</Label>
              <Select onValueChange={setTargetVar} value={targetVar}>
                <SelectTrigger>
                  <SelectValue placeholder="Select target..." />
                </SelectTrigger>
                <SelectContent>
                  {controllableVars.map((v) => (
                    <SelectItem key={v.columnName} value={v.columnName}>
                      {v.columnName}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label>Input Features</Label>
              <div className="border rounded-md p-4 h-48 overflow-y-auto space-y-2">
                {featureVars.map((v) => (
                  <div key={v.columnName} className="flex items-center space-x-2">
                    <Checkbox 
                      id={v.columnName} 
                      checked={selectedFeatures.includes(v.columnName)}
                      onCheckedChange={(checked) => {
                        if (checked) setSelectedFeatures([...selectedFeatures, v.columnName]);
                        else setSelectedFeatures(selectedFeatures.filter(f => f !== v.columnName));
                      }}
                    />
                    <label
                      htmlFor={v.columnName}
                      className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                    >
                      {v.columnName}
                    </label>
                  </div>
                ))}
              </div>
            </div>

            <Button 
              className="w-full" 
              onClick={handleTrain} 
              disabled={!targetVar || isTraining}
            >
              {isTraining ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Training...
                </>
              ) : (
                'Start Training'
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Results Panel */}
        <div className="lg:col-span-2 space-y-6">
          {isTraining && (
            <Card>
              <CardContent className="p-12 text-center space-y-4">
                <h3 className="text-xl font-semibold">Training in Progress</h3>
                <Progress value={progress} className="h-2 w-full max-w-md mx-auto" />
                <p className="text-muted-foreground">Optimizing hyperparameters...</p>
              </CardContent>
            </Card>
          )}

          {!isTraining && results.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-6"
            >
              <Card className="bg-primary/5 border-primary/20">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-primary">
                    <Trophy className="h-6 w-6" />
                    Best Model: {results.sort((a, b) => b.metrics.r2 - a.metrics.r2)[0].type}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-3 gap-4 text-center">
                    <div>
                      <div className="text-sm text-muted-foreground">R² Score</div>
                      <div className="text-2xl font-bold">{results.sort((a, b) => b.metrics.r2 - a.metrics.r2)[0].metrics.r2.toFixed(3)}</div>
                    </div>
                    <div>
                      <div className="text-sm text-muted-foreground">RMSE</div>
                      <div className="text-2xl font-bold">{results.sort((a, b) => b.metrics.r2 - a.metrics.r2)[0].metrics.rmse.toFixed(2)}</div>
                    </div>
                    <div>
                      <div className="text-sm text-muted-foreground">MAE</div>
                      <div className="text-2xl font-bold">{results.sort((a, b) => b.metrics.r2 - a.metrics.r2)[0].metrics.mae.toFixed(2)}</div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Model Comparison</CardTitle>
                </CardHeader>
                <CardContent className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={results.map(r => ({ name: r.type, r2: r.metrics.r2 }))}>
                      <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                      <XAxis dataKey="name" />
                      <YAxis domain={[0, 1]} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: 'hsl(var(--card))', borderColor: 'hsl(var(--border))' }}
                      />
                      <Bar dataKey="r2" fill="hsl(var(--secondary))" radius={[4, 4, 0, 0]} name="R² Score" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <div className="flex justify-end">
                <Button variant="outline" size="lg">
                  View Detailed Report <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </div>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
}
