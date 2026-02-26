import React, { useState } from 'react';
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
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import { toast } from 'sonner';
import {
  Settings as SettingsIcon,
  User,
  Database,
  Cpu,
  Trash2,
  CheckCircle2,
  Upload,
  BarChart3,
  Target,
  FileText,
  ArrowRight,
  AlertTriangle,
  Layers,
  BookOpen,
  Lightbulb,
  ShieldCheck,
  Info,
  TrendingUp,
  Sparkles,
  Clock,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { deleteDatasetRows } from '@/lib/db';

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

// ─── Guide step card ──────────────────────────────────────────────────────────

interface GuideStepProps {
  step: number;
  title: string;
  description: string;
  details: string[];
  tips?: string[];
  icon: React.ElementType;
  path: string;
  actionLabel: string;
  done: boolean;
}

function GuideStep({
  step,
  title,
  description,
  details,
  tips,
  icon: Icon,
  path,
  actionLabel,
  done,
}: GuideStepProps) {
  const navigate = useNavigate();
  return (
    <Card
      className={cn(
        'border',
        done
          ? 'border-green-200 dark:border-green-800'
          : 'border-border'
      )}
    >
      <CardContent className="p-5">
        <div className="flex items-start gap-4">
          {/* Step number + icon */}
          <div className="flex flex-col items-center gap-1.5 flex-shrink-0">
            <div
              className={cn(
                'h-10 w-10 rounded-xl flex items-center justify-center',
                done
                  ? 'bg-green-100 dark:bg-green-900/30'
                  : 'bg-primary/10'
              )}
            >
              {done ? (
                <CheckCircle2 className="h-5 w-5 text-green-600 dark:text-green-400" />
              ) : (
                <Icon className="h-5 w-5 text-primary" />
              )}
            </div>
            <span
              className={cn(
                'text-xs font-bold',
                done ? 'text-green-600 dark:text-green-400' : 'text-muted-foreground'
              )}
            >
              Step {step}
            </span>
          </div>

          {/* Content */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap mb-1">
              <h3 className="font-semibold">{title}</h3>
              {done && (
                <span className="text-xs px-2 py-0.5 rounded-full bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400 font-medium">
                  Completed
                </span>
              )}
            </div>
            <p className="text-sm text-muted-foreground mb-3">{description}</p>

            <div className="space-y-1.5 mb-3">
              {details.map((d, i) => (
                <div key={i} className="flex items-start gap-2 text-sm">
                  <div className="h-1.5 w-1.5 rounded-full bg-primary/60 mt-2 flex-shrink-0" />
                  <span>{d}</span>
                </div>
              ))}
            </div>

            {tips && tips.length > 0 && (
              <div className="bg-amber-50 dark:bg-amber-900/10 border border-amber-200 dark:border-amber-800 rounded-lg p-3 mb-3">
                <div className="flex items-center gap-1.5 mb-1.5">
                  <Lightbulb className="h-3.5 w-3.5 text-amber-600" />
                  <span className="text-xs font-semibold text-amber-700 dark:text-amber-400">
                    Tips
                  </span>
                </div>
                <div className="space-y-1">
                  {tips.map((t, i) => (
                    <p key={i} className="text-xs text-amber-700 dark:text-amber-400">
                      • {t}
                    </p>
                  ))}
                </div>
              </div>
            )}

            <Button
              size="sm"
              variant={done ? 'outline' : 'default'}
              onClick={() => navigate(path)}
            >
              {done ? 'Review' : actionLabel}
              <ArrowRight className="ml-2 h-3.5 w-3.5" />
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ─── component ────────────────────────────────────────────────────────────────

export default function Settings() {
  const navigate = useNavigate();
  const datasets = useStore((s) => s.datasets);
  const trainedModels = useStore((s) => s.trainedModels);
  const predictions = useStore((s) => s.predictions);
  const userName = useStore((s) => s.userName);
  const setUserName = useStore((s) => s.setUserName);
  const removeDataset = useStore((s) => s.removeDataset);
  const removeTrainedModel = useStore((s) => s.removeTrainedModel);
  const clearPredictions = useStore((s) => s.clearPredictions);

  const [nameInput, setNameInput] = useState(userName);
  const [confirmClearAll, setConfirmClearAll] = useState(false);
  const [confirmClearPredictions, setConfirmClearPredictions] = useState(false);

  // ── workflow completion state (for guide) ────────────────────────────────────

  const stepsDone = {
    upload: datasets.length > 0,
    explore: datasets.some((d) => d.mappings.length > 0),
    train: trainedModels.length > 0,
    predict: predictions.length > 0,
    report: predictions.length > 0 && trainedModels.length > 0,
  };

  // ── handlers ─────────────────────────────────────────────────────────────────

  function handleSaveName() {
    const trimmed = nameInput.trim();
    if (!trimmed) {
      toast.error('Please enter a name.');
      return;
    }
    setUserName(trimmed);
    toast.success('Name updated successfully.');
  }

  async function handleDeleteDataset(id: string, name: string) {
    if (!window.confirm(`Delete dataset "${name}"? This cannot be undone.`)) return;
    removeDataset(id);
    try {
      await deleteDatasetRows(id);
    } catch {
      // ignore IndexedDB errors
    }
    toast.success(`Dataset "${name}" deleted.`);
  }

  function handleDeleteModel(id: string, type: string, target: string) {
    if (!window.confirm(`Delete ${type} model for "${target}"? This cannot be undone.`)) return;
    removeTrainedModel(id);
    toast.success('Model deleted.');
  }

  function handleClearPredictions() {
    clearPredictions();
    setConfirmClearPredictions(false);
    toast.success('Prediction history cleared.');
  }

  function handleClearAll() {
    // Clear all datasets from IndexedDB
    datasets.forEach((d) => {
      deleteDatasetRows(d.id).catch(() => {});
      removeDataset(d.id);
    });
    // Clear all models
    [...trainedModels].forEach((m) => removeTrainedModel(m.id));
    // Clear predictions
    clearPredictions();
    setConfirmClearAll(false);
    toast.success('All data cleared. The workspace has been reset.');
  }

  // ── render ───────────────────────────────────────────────────────────────────

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      {/* Page heading */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Settings</h1>
        <p className="text-muted-foreground mt-1">
          Manage your preferences, data, and learn how to use the system
        </p>
      </div>

      <Tabs defaultValue="general" className="space-y-4">
        <TabsList className="grid w-full grid-cols-3 lg:w-auto lg:inline-flex">
          <TabsTrigger value="general">
            <User className="h-4 w-4 mr-1.5" />
            General
          </TabsTrigger>
          <TabsTrigger value="data">
            <Database className="h-4 w-4 mr-1.5" />
            Data & Models
          </TabsTrigger>
          <TabsTrigger value="guide">
            <BookOpen className="h-4 w-4 mr-1.5" />
            How to Use
          </TabsTrigger>
        </TabsList>

        {/* ── General Tab ── */}
        <TabsContent value="general" className="space-y-6">
          {/* Profile */}
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <User className="h-4 w-4" />
                  Your Profile
                </CardTitle>
                <CardDescription>
                  Personalise how your name appears in the app
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center gap-4">
                  <div className="h-14 w-14 rounded-full bg-primary/10 flex items-center justify-center text-primary text-xl font-bold flex-shrink-0">
                    {nameInput.charAt(0).toUpperCase() || 'U'}
                  </div>
                  <div className="flex-1 space-y-3">
                    <div className="space-y-1.5">
                      <Label htmlFor="user-name">Your Name</Label>
                      <div className="flex gap-2">
                        <Input
                          id="user-name"
                          value={nameInput}
                          onChange={(e) => setNameInput(e.target.value)}
                          placeholder="Enter your name"
                          className="max-w-xs"
                          onKeyDown={(e) => e.key === 'Enter' && handleSaveName()}
                        />
                        <Button onClick={handleSaveName}>Save</Button>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* Theme */}
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.07 }}>
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <SettingsIcon className="h-4 w-4" />
                  Appearance
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium">Dark / Light Mode</p>
                    <p className="text-sm text-muted-foreground">
                      Toggle the theme using the sun/moon icon in the top-right header bar.
                    </p>
                  </div>
                  <div className="h-9 w-9 rounded-lg bg-muted flex items-center justify-center text-muted-foreground">
                    ☀
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* App info */}
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.14 }}>
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <Info className="h-4 w-4" />
                  About Steel Optimizer
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                  {[
                    { label: 'Version', value: '2.0' },
                    { label: 'ML Models', value: '3 algorithms' },
                    { label: 'Max Data', value: '5 GB per file' },
                    { label: 'Storage', value: 'Local (browser)' },
                    { label: 'Cross-Validation', value: '5-fold CV' },
                    { label: 'Background Training', value: 'Web Worker' },
                  ].map((item) => (
                    <div key={item.label} className="bg-muted/40 rounded-lg p-3">
                      <p className="text-xs text-muted-foreground">{item.label}</p>
                      <p className="text-sm font-semibold mt-0.5">{item.value}</p>
                    </div>
                  ))}
                </div>
                <p className="text-xs text-muted-foreground">
                  All data is stored locally in your browser — nothing is sent to external
                  servers. Your steel plant data stays private.
                </p>
              </CardContent>
            </Card>
          </motion.div>

          {/* Workspace summary */}
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Workspace Summary</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-3 gap-3 text-center">
                  {[
                    { label: 'Datasets', value: datasets.length, Icon: Database },
                    { label: 'Models', value: trainedModels.length, Icon: Cpu },
                    { label: 'Predictions', value: predictions.length, Icon: Sparkles },
                  ].map((s) => (
                    <div key={s.label} className="bg-muted/40 rounded-lg p-3">
                      <s.Icon className="h-4 w-4 mx-auto text-muted-foreground mb-1" />
                      <p className="text-xl font-bold">{s.value}</p>
                      <p className="text-xs text-muted-foreground">{s.label}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </TabsContent>

        {/* ── Data & Models Tab ── */}
        <TabsContent value="data" className="space-y-6">
          {/* Datasets */}
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="text-base flex items-center gap-2">
                      <Database className="h-4 w-4" />
                      Datasets ({datasets.length})
                    </CardTitle>
                    <CardDescription>
                      Manage your uploaded data files
                    </CardDescription>
                  </div>
                  <Button size="sm" variant="outline" onClick={() => navigate('/upload')}>
                    <Upload className="mr-2 h-4 w-4" />
                    Upload New
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                {datasets.length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground">
                    <Database className="h-8 w-8 mx-auto opacity-30 mb-2" />
                    <p className="text-sm">No datasets yet.</p>
                    <Button
                      size="sm"
                      variant="outline"
                      className="mt-3"
                      onClick={() => navigate('/upload')}
                    >
                      Upload Your First Dataset
                    </Button>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {datasets.map((d) => (
                      <div
                        key={d.id}
                        className="flex items-center gap-3 p-3 rounded-lg border bg-muted/20 hover:bg-muted/40 transition-colors"
                      >
                        <div className="flex-1 min-w-0">
                          <p className="font-medium text-sm truncate">{d.name}</p>
                          <div className="flex gap-3 mt-0.5 text-xs text-muted-foreground flex-wrap">
                            <span className="flex items-center gap-1">
                              <Layers className="h-3 w-3" />
                              {d.rowCount.toLocaleString()} rows
                            </span>
                            <span className="flex items-center gap-1">
                              <BarChart3 className="h-3 w-3" />
                              {d.columnCount} cols
                            </span>
                            <span className="flex items-center gap-1">
                              <Clock className="h-3 w-3" />
                              {formatDate(d.uploadDate)}
                            </span>
                          </div>
                        </div>
                        <span
                          className={cn(
                            'text-xs font-semibold px-2 py-1 rounded flex-shrink-0',
                            healthBadgeClass(d.healthScore)
                          )}
                        >
                          {d.healthScore}%
                        </span>
                        <Button
                          size="icon"
                          variant="ghost"
                          className="h-8 w-8 text-destructive hover:text-destructive hover:bg-destructive/10 flex-shrink-0"
                          onClick={() => handleDeleteDataset(d.id, d.name)}
                          title="Delete dataset"
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>

          {/* Models */}
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.07 }}>
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="text-base flex items-center gap-2">
                      <Cpu className="h-4 w-4" />
                      Trained Models ({trainedModels.length})
                    </CardTitle>
                    <CardDescription>
                      Manage your trained AI models
                    </CardDescription>
                  </div>
                  <Button size="sm" variant="outline" onClick={() => navigate('/training')}>
                    <Cpu className="mr-2 h-4 w-4" />
                    Train New
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                {trainedModels.length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground">
                    <Cpu className="h-8 w-8 mx-auto opacity-30 mb-2" />
                    <p className="text-sm">No trained models yet.</p>
                    <Button
                      size="sm"
                      variant="outline"
                      className="mt-3"
                      onClick={() => navigate('/training')}
                    >
                      Train Your First Model
                    </Button>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {[...trainedModels]
                      .sort((a, b) => b.metrics.r2 - a.metrics.r2)
                      .map((m, i) => (
                        <div
                          key={m.id}
                          className="flex items-center gap-3 p-3 rounded-lg border bg-muted/20 hover:bg-muted/40 transition-colors"
                        >
                          {i === 0 && (
                            <div className="h-2 w-2 rounded-full bg-amber-500 flex-shrink-0" title="Best model" />
                          )}
                          {i > 0 && <div className="h-2 w-2 flex-shrink-0" />}
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 flex-wrap">
                              <span
                                className={cn(
                                  'text-xs font-medium px-2 py-0.5 rounded',
                                  modelTypeBadgeClass(m.type)
                                )}
                              >
                                {modelTypeLabel(m.type)}
                              </span>
                              <span className="text-xs text-muted-foreground truncate">
                                → {m.config.targetVariable}
                              </span>
                            </div>
                            <div className="flex gap-3 mt-1 text-xs text-muted-foreground">
                              <span>R²: <span className="font-semibold text-foreground">{(m.metrics.r2 * 100).toFixed(1)}%</span></span>
                              <span>Accuracy: <span className="font-semibold text-foreground">{m.metrics.accuracy.toFixed(1)}%</span></span>
                            </div>
                          </div>
                          <Button
                            size="icon"
                            variant="ghost"
                            className="h-8 w-8 text-destructive hover:text-destructive hover:bg-destructive/10 flex-shrink-0"
                            onClick={() =>
                              handleDeleteModel(m.id, modelTypeLabel(m.type), m.config.targetVariable)
                            }
                            title="Delete model"
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>

          {/* Predictions */}
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.14 }}>
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="text-base flex items-center gap-2">
                      <Target className="h-4 w-4" />
                      Prediction History ({predictions.length} records)
                    </CardTitle>
                    <CardDescription>
                      Clear your prediction log if you want a fresh start
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                {predictions.length === 0 ? (
                  <p className="text-sm text-muted-foreground py-4 text-center">
                    No prediction history to clear.
                  </p>
                ) : confirmClearPredictions ? (
                  <div className="flex items-center gap-3 p-3 border border-destructive/30 rounded-lg bg-destructive/5">
                    <AlertTriangle className="h-5 w-5 text-destructive flex-shrink-0" />
                    <p className="text-sm flex-1">
                      Clear all {predictions.length} prediction records? This cannot be undone.
                    </p>
                    <Button
                      size="sm"
                      variant="destructive"
                      onClick={handleClearPredictions}
                    >
                      Yes, clear
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => setConfirmClearPredictions(false)}
                    >
                      Cancel
                    </Button>
                  </div>
                ) : (
                  <Button
                    variant="outline"
                    className="text-destructive border-destructive/30 hover:bg-destructive/10"
                    onClick={() => setConfirmClearPredictions(true)}
                  >
                    <Trash2 className="mr-2 h-4 w-4" />
                    Clear Prediction History
                  </Button>
                )}
              </CardContent>
            </Card>
          </motion.div>

          {/* Danger zone */}
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
            <Card className="border-destructive/40">
              <CardHeader>
                <CardTitle className="text-base text-destructive flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4" />
                  Danger Zone
                </CardTitle>
                <CardDescription>
                  Permanently delete all data, models, and predictions from this workspace
                </CardDescription>
              </CardHeader>
              <CardContent>
                {confirmClearAll ? (
                  <div className="p-4 border border-destructive rounded-lg bg-destructive/5 space-y-3">
                    <p className="text-sm font-semibold text-destructive">
                      Are you absolutely sure?
                    </p>
                    <p className="text-sm text-muted-foreground">
                      This will permanently delete:
                    </p>
                    <ul className="text-sm text-muted-foreground space-y-1">
                      <li>• {datasets.length} dataset{datasets.length !== 1 ? 's' : ''}</li>
                      <li>• {trainedModels.length} trained model{trainedModels.length !== 1 ? 's' : ''}</li>
                      <li>• {predictions.length} prediction record{predictions.length !== 1 ? 's' : ''}</li>
                    </ul>
                    <div className="flex gap-2">
                      <Button
                        variant="destructive"
                        onClick={handleClearAll}
                      >
                        Yes, delete everything
                      </Button>
                      <Button
                        variant="outline"
                        onClick={() => setConfirmClearAll(false)}
                      >
                        Cancel
                      </Button>
                    </div>
                  </div>
                ) : (
                  <Button
                    variant="destructive"
                    onClick={() => setConfirmClearAll(true)}
                  >
                    <Trash2 className="mr-2 h-4 w-4" />
                    Reset Entire Workspace
                  </Button>
                )}
              </CardContent>
            </Card>
          </motion.div>
        </TabsContent>

        {/* ── Guide Tab ── */}
        <TabsContent value="guide" className="space-y-6">
          {/* Intro */}
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
            <Card className="bg-primary/5 border-primary/20">
              <CardContent className="p-5">
                <div className="flex items-start gap-4">
                  <div className="h-10 w-10 rounded-xl bg-primary/10 flex items-center justify-center flex-shrink-0">
                    <BookOpen className="h-5 w-5 text-primary" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-lg mb-1">How to Use Steel Optimizer</h3>
                    <p className="text-sm text-muted-foreground">
                      This guide walks you through the system step by step — no technical
                      background needed. Follow the steps in order for the best results.
                    </p>
                    <div className="mt-3 flex items-center gap-2">
                      <div className="flex-1 h-2 bg-primary/20 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-primary rounded-full transition-all"
                          style={{
                            width: `${(Object.values(stepsDone).filter(Boolean).length / Object.keys(stepsDone).length) * 100}%`,
                          }}
                        />
                      </div>
                      <span className="text-xs text-muted-foreground whitespace-nowrap">
                        {Object.values(stepsDone).filter(Boolean).length} /{' '}
                        {Object.keys(stepsDone).length} steps done
                      </span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* Step-by-step guide */}
          <div className="space-y-4">
            <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.07 }}>
              <GuideStep
                step={1}
                title="Upload Your Steel Plant Data"
                description="Start by importing your historical production data. This is the raw material the AI learns from."
                details={[
                  'Prepare a spreadsheet (Excel or CSV) with your production records — one row per heat or batch.',
                  'Include columns for process inputs (oxygen flow, power, alloy additions) and outputs (yield, temperature, energy).',
                  'The system automatically detects and categorises your columns — no manual setup needed.',
                  'Files up to 5 GB are supported. More data = better predictions.',
                ]}
                tips={[
                  'At least 50–100 rows of data is recommended for reliable model training.',
                  'Make sure numeric columns have consistent units (e.g., all temperatures in °C).',
                  'Missing values are handled automatically, but less is better.',
                ]}
                icon={Upload}
                path="/upload"
                actionLabel="Upload Data Now"
                done={stepsDone.upload}
              />
            </motion.div>

            <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
              <GuideStep
                step={2}
                title="Explore and Understand Your Data"
                description="Before training, review your data quality and see which variables are related to each other."
                details={[
                  'The Data Health Score (0–100) tells you how clean your data is. Higher is better.',
                  'The distribution chart shows the spread of your target variable — look for unusual peaks or gaps.',
                  'The correlation matrix highlights which inputs are most strongly linked to your outputs.',
                  'Columns are colour-coded: Controllable (blue) = inputs you can adjust; Output (green) = what you want to predict.',
                ]}
                tips={[
                  'A health score above 80 means your data is in great shape.',
                  'High correlation between an input and output column (±0.7 or more) is a good sign — that input is likely to be predictive.',
                  'You can update column categories on the Upload page if the system mis-categorised any.',
                ]}
                icon={Database}
                path="/explorer"
                actionLabel="Explore Data"
                done={stepsDone.explore}
              />
            </motion.div>

            <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.13 }}>
              <GuideStep
                step={3}
                title="Train Your AI Models"
                description="The system trains three different AI models on your data and picks the best one automatically."
                details={[
                  'Choose a Goal Preset (e.g., "Maximise Yield", "Minimise Energy") — or select your target variable manually.',
                  'The system trains Linear Regression, Random Forest, and Gradient Boosting models simultaneously.',
                  'Training runs in the background — the page stays responsive. It usually takes 20–60 seconds.',
                  'Results show R² score, accuracy, and a feature importance chart telling you which inputs matter most.',
                ]}
                tips={[
                  'R² above 90% means excellent fit. Above 70% is good. Below 50% may mean more data is needed.',
                  'If "Possible overfitting" appears, try collecting more diverse production data.',
                  'The Gradient Boosting model is usually the most accurate but slowest to train.',
                  'You can train multiple models for different target variables (yield, energy, temperature, etc.).',
                ]}
                icon={Cpu}
                path="/training"
                actionLabel="Train a Model"
                done={stepsDone.train}
              />
            </motion.div>

            <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.16 }}>
              <GuideStep
                step={4}
                title="Make Predictions and Optimise"
                description="Enter your current process conditions to predict the output and get recommendations for improvement."
                details={[
                  'Select a trained model and the target you want to predict.',
                  'Enter your current process parameters (e.g., oxygen flow, charge weight, power).',
                  'The AI instantly predicts the expected output value with a 95% confidence interval.',
                  'Optimisation recommendations show which parameters to adjust for maximum impact.',
                ]}
                tips={[
                  'The confidence interval shows the range of likely outcomes — a narrower range means higher certainty.',
                  'Focus on the top 3–5 recommended parameters — they have the biggest influence on results.',
                  'Run multiple predictions to explore different scenarios before deciding on process changes.',
                  'All predictions are saved to your history in the Reports section.',
                ]}
                icon={Target}
                path="/predictions"
                actionLabel="Make a Prediction"
                done={stepsDone.predict}
              />
            </motion.div>

            <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.19 }}>
              <GuideStep
                step={5}
                title="Review Reports and Export Results"
                description="See a full summary of your models, datasets, and prediction history. Export data for management reports."
                details={[
                  'The Overview tab shows your best model performance and key process variables at a glance.',
                  'The Models tab compares all trained models side-by-side with detailed metrics.',
                  'The Datasets tab shows the health and structure of your uploaded data.',
                  'The History tab lists every prediction run — filter by variable or model, then export to CSV.',
                ]}
                tips={[
                  'Export prediction history to CSV for monthly or weekly production reports.',
                  'The Feature Importance chart (Overview) is a great starting point for process improvement discussions.',
                  'Retrain your models periodically as new production data becomes available for better accuracy.',
                ]}
                icon={FileText}
                path="/reports"
                actionLabel="View Reports"
                done={stepsDone.report}
              />
            </motion.div>
          </div>

          {/* Key concepts glossary */}
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.25 }}>
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <BookOpen className="h-4 w-4" />
                  Key Terms Explained
                </CardTitle>
                <CardDescription>
                  Plain-language explanations of the metrics you'll see
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {[
                    {
                      term: 'R² Score (R-squared)',
                      def: 'How much of your output variation the model explains. 100% = perfect, 90%+ = excellent, 70%+ = good, below 50% = needs more data.',
                    },
                    {
                      term: 'Accuracy',
                      def: 'How close predictions are to actual values, on average. 100% means the model predicts exactly right every time. 90%+ is very good for steel production.',
                    },
                    {
                      term: 'RMSE (Root Mean Square Error)',
                      def: 'The average prediction error in the same units as your target. If predicting yield in %, an RMSE of 0.5 means predictions are typically ±0.5% off.',
                    },
                    {
                      term: 'Cross-Validation (CV)',
                      def: 'A reliability test: the model is tested on data it has never seen. CV R² close to training R² means the model generalises well to new heats.',
                    },
                    {
                      term: 'Feature Importance',
                      def: 'Which input variables have the biggest impact on the prediction. A longer bar = bigger influence. Use this to prioritise process parameters.',
                    },
                    {
                      term: 'Confidence Interval',
                      def: "The range where the true value is likely to fall, 95% of the time. E.g., 'Predicted: 87.2% ± 1.5%' means the real value is almost certainly between 85.7% and 88.7%.",
                    },
                    {
                      term: 'Controllable vs Uncontrollable Columns',
                      def: 'Controllable columns are inputs your operators can adjust (e.g., oxygen flow, alloy weight). Uncontrollable are conditions you cannot change (e.g., ambient temperature, material quality).',
                    },
                    {
                      term: 'Data Health Score',
                      def: 'A 0–100 score measuring how clean your data is. Points are deducted for missing values and statistical outliers. 80+ is excellent.',
                    },
                  ].map((item) => (
                    <div key={item.term} className="border-b last:border-0 pb-3 last:pb-0">
                      <p className="text-sm font-semibold">{item.term}</p>
                      <p className="text-sm text-muted-foreground mt-0.5">{item.def}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* Tips for non-technical users */}
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
            <Card className="bg-blue-50/50 dark:bg-blue-950/20 border-blue-200 dark:border-blue-900">
              <CardHeader className="pb-3">
                <CardTitle className="text-base flex items-center gap-2 text-blue-700 dark:text-blue-400">
                  <ShieldCheck className="h-4 w-4" />
                  Best Practices for Steel Plant Use
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {[
                  'Collect data consistently — same sensors, same column names every time — for best model accuracy.',
                  'More historical heats = better predictions. Aim for at least 3–6 months of production data.',
                  "Retrain your models every quarter or when you change equipment, alloy grades, or production targets.",
                  'Use predictions as guidance, not absolute truth — always apply operator judgement alongside AI recommendations.',
                  'If accuracy drops after retraining, check for new sensors or changed column names in your data files.',
                ].map((tip, i) => (
                  <div key={i} className="flex items-start gap-2 text-sm">
                    <TrendingUp className="h-4 w-4 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
                    <span className="text-blue-800 dark:text-blue-300">{tip}</span>
                  </div>
                ))}
              </CardContent>
            </Card>
          </motion.div>

          {/* Quick nav */}
          <Separator />
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.35 }}>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {[
                { label: 'Upload Data', Icon: Upload, path: '/upload' },
                { label: 'Explore Data', Icon: Database, path: '/explorer' },
                { label: 'Train Models', Icon: Cpu, path: '/training' },
                { label: 'Predictions', Icon: Target, path: '/predictions' },
              ].map(({ label, Icon, path }) => (
                <Button
                  key={path}
                  variant="outline"
                  className="h-auto py-3 flex-col gap-1.5"
                  onClick={() => navigate(path)}
                >
                  <Icon className="h-5 w-5 text-muted-foreground" />
                  <span className="text-xs">{label}</span>
                </Button>
              ))}
            </div>
          </motion.div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
