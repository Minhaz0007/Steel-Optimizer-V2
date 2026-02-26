import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { ArrowRight, Check, Database, Cpu, Target } from 'lucide-react';

const steps = [
  {
    id: 1,
    title: "Welcome to Steel Optimizer",
    description: "Your AI-powered assistant for optimizing steel production yield and efficiency.",
    icon: Target,
    content: (
      <div className="space-y-4 text-center">
        <p>We help you analyze historical data to find the perfect furnace parameters.</p>
        <div className="flex justify-center gap-4 text-sm text-muted-foreground">
          <div className="flex flex-col items-center">
            <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center mb-2">1</div>
            <span>Upload Data</span>
          </div>
          <div className="flex flex-col items-center">
            <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center mb-2">2</div>
            <span>Train Models</span>
          </div>
          <div className="flex flex-col items-center">
            <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center mb-2">3</div>
            <span>Predict</span>
          </div>
        </div>
      </div>
    )
  },
  {
    id: 2,
    title: "Prepare Your Data",
    description: "Upload your historical logs in CSV or Excel format.",
    icon: Database,
    content: (
      <div className="space-y-4 text-center">
        <p>Ensure your file contains:</p>
        <ul className="text-left list-disc list-inside space-y-1 text-sm bg-muted p-4 rounded-md">
          <li>Date/Time column</li>
          <li>Furnace ID / Heat Number</li>
          <li>Input parameters (e.g., Oxygen, Power)</li>
          <li>Output metrics (e.g., Yield, Temp)</li>
        </ul>
        <p className="text-xs text-muted-foreground">We'll help you classify columns automatically.</p>
      </div>
    )
  },
  {
    id: 3,
    title: "How ML Works",
    description: "A 4-stage Python pipeline trains specialized models for steel production.",
    icon: Cpu,
    content: (
      <div className="space-y-4 text-center">
        <p>Our pipeline runs:</p>
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div className="bg-muted p-2 rounded">LightGBM Regressors</div>
          <div className="bg-muted p-2 rounded">CatBoost Classifiers</div>
          <div className="bg-muted p-2 rounded">Isolation Forest</div>
          <div className="bg-muted p-2 rounded">Optuna Optimizer</div>
        </div>
        <p>Optuna Bayesian optimization finds the best furnace setpoints.</p>
      </div>
    )
  },
  {
    id: 4,
    title: "Ready to Start?",
    description: "Let's optimize your production process.",
    icon: Check,
    content: (
      <div className="text-center py-8">
        <div className="inline-flex h-20 w-20 items-center justify-center rounded-full bg-green-100 text-green-600 mb-4">
          <Check className="h-10 w-10" />
        </div>
        <p className="text-lg font-medium">Click below to upload your first dataset.</p>
      </div>
    )
  }
];

export default function Onboarding() {
  const [currentStep, setCurrentStep] = useState(0);
  const navigate = useNavigate();

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      navigate('/upload');
    }
  };

  const handleSkip = () => {
    navigate('/upload');
  };

  const StepIcon = steps[currentStep].icon;

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4">
      <Card className="w-full max-w-lg shadow-xl border-primary/10">
        <CardHeader className="text-center pb-2">
          <div className="mx-auto h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center text-primary mb-4">
            <StepIcon className="h-6 w-6" />
          </div>
          <CardTitle className="text-2xl">{steps[currentStep].title}</CardTitle>
          <p className="text-muted-foreground">{steps[currentStep].description}</p>
        </CardHeader>
        
        <CardContent className="min-h-[200px] flex items-center justify-center">
          <AnimatePresence mode="wait">
            <motion.div
              key={currentStep}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
              className="w-full"
            >
              {steps[currentStep].content}
            </motion.div>
          </AnimatePresence>
        </CardContent>

        <CardFooter className="flex flex-col gap-4 pt-6 border-t">
          <div className="w-full flex justify-between items-center">
            <Button variant="ghost" onClick={handleSkip} className="text-muted-foreground">
              Skip
            </Button>
            <div className="flex gap-1">
              {steps.map((_, i) => (
                <div 
                  key={i} 
                  className={`h-2 w-2 rounded-full transition-colors ${i === currentStep ? 'bg-primary' : 'bg-muted'}`}
                />
              ))}
            </div>
            <Button onClick={handleNext}>
              {currentStep === steps.length - 1 ? 'Get Started' : 'Next'}
              {currentStep !== steps.length - 1 && <ArrowRight className="ml-2 h-4 w-4" />}
            </Button>
          </div>
          <Progress value={((currentStep + 1) / steps.length) * 100} className="h-1" />
        </CardFooter>
      </Card>
    </div>
  );
}
