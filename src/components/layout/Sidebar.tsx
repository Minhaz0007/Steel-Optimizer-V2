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
  Menu,
  X,
  ChevronLeft,
  ChevronRight
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const navItems = [
  { name: 'Dashboard', icon: LayoutDashboard, path: '/dashboard' },
  { name: 'Upload Data', icon: Upload, path: '/upload' },
  { name: 'Data Explorer', icon: Database, path: '/explorer' },
  { name: 'Model Training', icon: Cpu, path: '/training' },
  { name: 'Predictions', icon: Target, path: '/predictions' },
  { name: 'Reports', icon: FileText, path: '/reports' },
  { name: 'Settings', icon: Settings, path: '/settings' },
];

export function Sidebar() {
  const location = useLocation();
  const [collapsed, setCollapsed] = useState(false);

  return (
    <motion.div
      initial={{ width: 240 }}
      animate={{ width: collapsed ? 80 : 240 }}
      className="h-screen bg-card border-r flex flex-col relative z-20"
    >
      <div className="p-4 flex items-center justify-between h-16 border-b">
        {!collapsed && (
          <motion.span
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="font-bold text-xl text-primary truncate"
          >
            Steel Optimizer
          </motion.span>
        )}
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setCollapsed(!collapsed)}
          className="ml-auto"
        >
          {collapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
        </Button>
      </div>

      <nav className="flex-1 p-2 space-y-1 overflow-y-auto">
        {navItems.map((item) => {
          const isActive = location.pathname === item.path;
          return (
            <Link to={item.path} key={item.path}>
              <Button
                variant={isActive ? 'secondary' : 'ghost'}
                className={cn(
                  'w-full justify-start mb-1',
                  collapsed ? 'px-2 justify-center' : 'px-4'
                )}
              >
                <item.icon
                  className={cn(
                    'h-5 w-5',
                    isActive ? 'text-secondary-foreground' : 'text-muted-foreground',
                    !collapsed && 'mr-3'
                  )}
                />
                {!collapsed && <span>{item.name}</span>}
              </Button>
            </Link>
          );
        })}
      </nav>

      <div className="p-4 border-t">
        <div className="flex items-center gap-3">
          <div className="h-8 w-8 rounded-full bg-accent flex items-center justify-center text-accent-foreground font-bold">
            U
          </div>
          {!collapsed && (
            <div className="flex flex-col">
              <span className="text-sm font-medium">User</span>
              <span className="text-xs text-muted-foreground">Admin</span>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}
