import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/layout/Layout';
import Landing from './pages/Landing';
import UploadPage from './pages/Upload';
import Explorer from './pages/Explorer';
import Training from './pages/Training';
import Predictions from './pages/Predictions';
import Onboarding from './pages/Onboarding';

// Placeholder pages
const Dashboard = () => <div className="p-4">Dashboard (Coming Soon)</div>;
const Reports = () => <div className="p-4">Reports (Coming Soon)</div>;
const Settings = () => <div className="p-4">Settings (Coming Soon)</div>;

function App() {
  return (
    <Router>
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
    </Router>
  );
}

export default App;
