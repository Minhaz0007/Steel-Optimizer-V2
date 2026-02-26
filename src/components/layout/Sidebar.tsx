import { Link, useLocation } from 'react-router-dom';
import { cn } from '@/lib/utils';
import {
  LayoutDashboard,
  Upload,
  Database,
  Cpu,
  Target,
  FileText,
  Settings,
  ChevronLeft,
  ChevronRight,
  CheckCircle2,
  ArrowRight,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import { useStore } from '@/store/useStore';

// ─── Step definitions ──────────────────────────────────────────────────────────

const WORKFLOW_STEPS = [
  {
    step: 1,
    name: 'Upload Data',
    icon: Upload,
    path: '/upload',
    hint: 'Import your CSV or Excel file',
  },
  {
    step: 2,
    name: 'Explore Data',
    icon: Database,
    path: '/explorer',
    hint: 'Review quality & correlations',
  },
  {
    step: 3,
    name: 'Train Models',
    icon: Cpu,
    path: '/training',
    hint: 'Run LightGBM, CatBoost & Optuna',
  },
  {
    step: 4,
    name: 'Predictions',
    icon: Target,
    path: '/predictions',
    hint: 'Predict & optimise parameters',
  },
];

const UTILITY_LINKS = [
  { name: 'Reports', icon: FileText, path: '/reports' },
  { name: 'Settings', icon: Settings, path: '/settings' },
];

// ─── component ────────────────────────────────────────────────────────────────

export function Sidebar() {
  const location = useLocation();
  const [collapsed, setCollapsed] = useState(false);

  const datasets = useStore((s) => s.datasets);
  const trainingSession = useStore((s) => s.trainingSession);
  const optimizationRecords = useStore((s) => s.optimizationRecords);
  const userName = useStore((s) => s.userName);

  // Determine completion state for each step
  const stepsDone = useMemo(
    () => [
      datasets.length > 0,
      datasets.some((d) => d.mappings.length > 0),
      trainingSession !== null,
      optimizationRecords.length > 0,
    ],
    [datasets, trainingSession, optimizationRecords]
  );

  const completedCount = stepsDone.filter(Boolean).length;
  const nextStepIndex = stepsDone.findIndex((done) => !done);
  const allDone = nextStepIndex === -1;

  return (
    <motion.div
      initial={{ width: 240 }}
      animate={{ width: collapsed ? 72 : 240 }}
      transition={{ type: 'spring', stiffness: 300, damping: 30 }}
      className="h-screen bg-card border-r flex flex-col relative z-20 overflow-hidden"
    >
      {/* ── Header ── */}
      <div className="p-3 flex items-center justify-between h-16 border-b flex-shrink-0">
        {!collapsed && (
          <motion.span
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="font-bold text-base text-primary truncate leading-tight"
          >
            Steel Optimizer
          </motion.span>
        )}
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setCollapsed(!collapsed)}
          className="ml-auto h-8 w-8 flex-shrink-0"
        >
          {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
        </Button>
      </div>

      <nav className="flex-1 flex flex-col gap-1 p-2 overflow-y-auto overflow-x-hidden">
        {/* Dashboard link */}
        <Link to="/dashboard">
          <Button
            variant={location.pathname === '/dashboard' ? 'secondary' : 'ghost'}
            className={cn(
              'w-full mb-1',
              collapsed ? 'justify-center px-0' : 'justify-start px-3'
            )}
            size="sm"
          >
            <LayoutDashboard
              className={cn(
                'h-4 w-4 flex-shrink-0',
                location.pathname === '/dashboard'
                  ? 'text-secondary-foreground'
                  : 'text-muted-foreground',
                !collapsed && 'mr-2.5'
              )}
            />
            {!collapsed && <span className="truncate">Dashboard</span>}
          </Button>
        </Link>

        {/* Divider with "MY WORKFLOW" label */}
        {!collapsed ? (
          <div className="flex items-center gap-2 mt-2 mb-1 px-1">
            <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider whitespace-nowrap">
              My Workflow
            </span>
            <div className="flex-1 h-px bg-border" />
            <span className="text-xs text-muted-foreground whitespace-nowrap tabular-nums">
              {completedCount}/4
            </span>
          </div>
        ) : (
          <div className="h-px bg-border my-2 mx-1" />
        )}

        {/* Workflow steps */}
        {WORKFLOW_STEPS.map(({ step, name, icon: Icon, path, hint }, index) => {
          const isActive = location.pathname === path;
          const isDone = stepsDone[index];
          const isNext = !allDone && nextStepIndex === index;

          return (
            <Link to={path} key={path}>
              <div
                className={cn(
                  'w-full rounded-md transition-all group',
                  isActive
                    ? 'bg-secondary'
                    : isNext && !collapsed
                    ? 'bg-primary/5 hover:bg-primary/10'
                    : 'hover:bg-accent'
                )}
              >
                {collapsed ? (
                  /* Collapsed: icon only with step dot */
                  <div className="relative flex items-center justify-center h-9 w-full">
                    {isDone ? (
                      <CheckCircle2 className="h-4 w-4 text-green-500" />
                    ) : (
                      <Icon
                        className={cn(
                          'h-4 w-4',
                          isNext ? 'text-primary' : 'text-muted-foreground'
                        )}
                      />
                    )}
                    {isNext && (
                      <span className="absolute top-1 right-1 h-2 w-2 rounded-full bg-primary" />
                    )}
                  </div>
                ) : (
                  /* Expanded: full row */
                  <div className="flex items-center gap-2.5 px-2.5 py-2 min-w-0">
                    {/* Step number or checkmark */}
                    <div
                      className={cn(
                        'h-5 w-5 rounded flex items-center justify-center text-xs font-bold flex-shrink-0 transition-colors',
                        isDone
                          ? 'bg-green-100 dark:bg-green-900/30'
                          : isNext
                          ? 'bg-primary/15'
                          : 'bg-muted'
                      )}
                    >
                      {isDone ? (
                        <CheckCircle2 className="h-3.5 w-3.5 text-green-600 dark:text-green-400" />
                      ) : (
                        <span
                          className={cn(
                            isNext ? 'text-primary' : 'text-muted-foreground'
                          )}
                        >
                          {step}
                        </span>
                      )}
                    </div>

                    {/* Name + hint */}
                    <div className="flex-1 min-w-0">
                      <p
                        className={cn(
                          'text-sm font-medium leading-tight truncate',
                          isActive
                            ? 'text-secondary-foreground'
                            : isDone
                            ? 'text-foreground'
                            : isNext
                            ? 'text-primary'
                            : 'text-muted-foreground'
                        )}
                      >
                        {name}
                      </p>
                      {isNext && (
                        <p className="text-xs text-muted-foreground truncate leading-tight mt-0.5">
                          {hint}
                        </p>
                      )}
                    </div>

                    {/* "NEXT" badge */}
                    {isNext && !isActive && (
                      <span className="text-xs font-semibold text-primary bg-primary/10 px-1.5 py-0.5 rounded flex-shrink-0 flex items-center gap-0.5">
                        NEXT
                        <ArrowRight className="h-2.5 w-2.5" />
                      </span>
                    )}
                  </div>
                )}
              </div>
            </Link>
          );
        })}

        {/* All done indicator */}
        {allDone && !collapsed && (
          <div className="mx-1 mt-1 px-2.5 py-2 rounded-md bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800">
            <p className="text-xs font-medium text-green-700 dark:text-green-400 flex items-center gap-1.5">
              <CheckCircle2 className="h-3.5 w-3.5" />
              All steps complete!
            </p>
            <p className="text-xs text-green-600/70 dark:text-green-500/70 mt-0.5">
              Keep predicting &amp; training
            </p>
          </div>
        )}

        {/* Divider before utility links */}
        <div className="h-px bg-border my-2 mx-1" />

        {/* Utility links */}
        {!collapsed && (
          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider px-1 mb-1">
            Tools
          </p>
        )}
        {UTILITY_LINKS.map(({ name, icon: Icon, path }) => {
          const isActive = location.pathname === path;
          return (
            <Link to={path} key={path}>
              <Button
                variant={isActive ? 'secondary' : 'ghost'}
                size="sm"
                className={cn(
                  'w-full',
                  collapsed ? 'justify-center px-0' : 'justify-start px-3'
                )}
              >
                <Icon
                  className={cn(
                    'h-4 w-4 flex-shrink-0',
                    isActive ? 'text-secondary-foreground' : 'text-muted-foreground',
                    !collapsed && 'mr-2.5'
                  )}
                />
                {!collapsed && <span className="truncate">{name}</span>}
              </Button>
            </Link>
          );
        })}
      </nav>

      {/* ── Footer: progress + user ── */}
      <div className="border-t p-3 flex-shrink-0 space-y-3">
        {/* Progress bar */}
        {!collapsed && (
          <div className="space-y-1">
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>Workflow progress</span>
              <span className="tabular-nums">{Math.round((completedCount / 4) * 100)}%</span>
            </div>
            <div className="h-1.5 bg-muted rounded-full overflow-hidden">
              <div
                className={cn(
                  'h-full rounded-full transition-all duration-500',
                  completedCount === 4 ? 'bg-green-500' : 'bg-primary'
                )}
                style={{ width: `${(completedCount / 4) * 100}%` }}
              />
            </div>
          </div>
        )}

        {/* User row */}
        <div className={cn('flex items-center gap-2.5', collapsed && 'justify-center')}>
          <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center text-primary font-bold text-sm flex-shrink-0">
            {(userName || 'U').charAt(0).toUpperCase()}
          </div>
          {!collapsed && (
            <div className="flex flex-col min-w-0">
              <span className="text-sm font-medium truncate">{userName || 'Operator'}</span>
              <span className="text-xs text-muted-foreground">Steel Plant User</span>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}
