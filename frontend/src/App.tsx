import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import CampaignResults from './pages/CampaignResults'
import Login from './pages/Login'

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const auth = localStorage.getItem('causaliq-auth')
  if (!auth) {
    return <Navigate to="/login" replace />
  }
  return <>{children}</>
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
        <Route path="/campaigns/:id" element={<ProtectedRoute><CampaignResults /></ProtectedRoute>} />
      </Routes>
    </BrowserRouter>
  )
}
