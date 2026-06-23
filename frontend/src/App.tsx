import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import CampaignResults from './pages/CampaignResults'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/campaigns/:id" element={<CampaignResults />} />
      </Routes>
    </BrowserRouter>
  )
}
