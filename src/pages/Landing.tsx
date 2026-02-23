import { Link } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { motion } from 'framer-motion';
import { ArrowRight, Upload, Cpu, Target } from 'lucide-react';

export default function Landing() {
  return (
    <div className="min-h-screen bg-background flex flex-col items-center justify-center p-4 relative overflow-hidden">
      {/* Background decoration */}
      <div className="absolute inset-0 z-0 opacity-10 pointer-events-none">
        <div className="absolute top-0 left-0 w-96 h-96 bg-primary rounded-full blur-3xl transform -translate-x-1/2 -translate-y-1/2" />
        <div className="absolute bottom-0 right-0 w-96 h-96 bg-secondary rounded-full blur-3xl transform translate-x-1/2 translate-y-1/2" />
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="z-10 text-center max-w-4xl space-y-8"
      >
        <h1 className="text-5xl md:text-7xl font-bold tracking-tight text-foreground">
          AI-Powered <span className="text-primary">Steel Production</span> Optimization
        </h1>
        <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
          Ingest historical data, train advanced ML models, and predict optimal furnace parameters to maximize yield and efficiency.
        </p>

        <div className="flex flex-col sm:flex-row gap-4 justify-center pt-8">
          <Link to="/onboarding">
            <Button size="lg" className="text-lg px-8 py-6 rounded-full shadow-lg hover:shadow-xl transition-all">
              Get Started <ArrowRight className="ml-2 h-5 w-5" />
            </Button>
          </Link>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-16 text-left">
          <FeatureCard
            icon={Upload}
            title="Upload Data"
            description="Drag & drop your historical CSV/Excel logs. We handle cleaning and parsing."
            delay={0.2}
          />
          <FeatureCard
            icon={Cpu}
            title="Train Models"
            description="Auto-train 7+ ML algorithms including XGBoost & Random Forest."
            delay={0.4}
          />
          <FeatureCard
            icon={Target}
            title="Predict & Optimize"
            description="Get actionable parameter recommendations to improve output quality."
            delay={0.6}
          />
        </div>
      </motion.div>
    </div>
  );
}

function FeatureCard({ icon: Icon, title, description, delay }: { icon: any, title: string, description: string, delay: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.5 }}
    >
      <Card className="h-full border-primary/10 bg-card/50 backdrop-blur hover:border-primary/30 transition-colors">
        <CardContent className="p-6 space-y-4">
          <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center text-primary">
            <Icon className="h-6 w-6" />
          </div>
          <h3 className="text-xl font-semibold">{title}</h3>
          <p className="text-muted-foreground">{description}</p>
        </CardContent>
      </Card>
    </motion.div>
  );
}
