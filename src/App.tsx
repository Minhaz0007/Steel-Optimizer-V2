import { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/layout/Layout';
import Landing from './pages/Landing';
import Dashboard from './pages/Dashboard';
import UploadPage from './pages/Upload';
import Explorer from './pages/Explorer';
import Training from './pages/Training';
import Predictions from './pages/Predictions';
import Onboarding from './pages/Onboarding';
import Reports from './pages/Reports';
import Settings from './pages/Settings';
import { useStore } from './store/useStore';
import { loadDatasetRows } from './lib/db';

/**
 * On mount, restore raw dataset rows from IndexedDB into Zustand.
 * The persist middleware stores metadata but strips data[] to avoid
 * filling localStorage. This hook rehydrates the in-memory state so the
 * Explorer, Training, and Predictions pages work after a page refresh.
 */
function useIdbRehydration() {
  const datasets = useStore((s) => s.datasets);
  const hydrateDatasetData = useStore((s) => s.hydrateDatasetData);

  useEffect(() => {
    datasets.forEach(async (d) => {
      if (d.data.length === 0) {
        try {
          const rows = await loadDatasetRows(d.id);
          if (rows && rows.length > 0) {
            hydrateDatasetData(d.id, rows);
          }
        } catch {
          // IndexedDB unavailable in some environments — silently skip
        }
      }
    });
  // We intentionally run once on mount only (dataset IDs from localStorage).
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
}

function AppRoutes() {
  useIdbRehydration();

  return (
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route path="/onboarding" element={<Onboarding />} />
      <Route element={<Layout />}>
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/explorer" element={<Explorer />} />
        <Route path="/training" element={<Training />} />
        <Route path="/predictions" element={<Predictions />} />
        <Route path="/reports" element={<Reports />} />
        <Route path="/settings" element={<Settings />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

function App() {
  return (
    <Router>
      <AppRoutes />
    </Router>
  );
}

export default App;
