import { Navigate, Route, Routes } from 'react-router-dom';
import Layout from './components/Layout';
import VoiceSetup from './pages/VoiceSetup';
import PersonaConfig from './pages/PersonaConfig';
import MeetingDashboard from './pages/MeetingDashboard';
import Settings from './pages/Settings';

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/voice" element={<VoiceSetup />} />
        <Route path="/persona" element={<PersonaConfig />} />
        <Route path="/meeting" element={<MeetingDashboard />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="/" element={<Navigate to="/meeting" replace />} />
        <Route path="*" element={<Navigate to="/meeting" replace />} />
      </Route>
    </Routes>
  );
}
